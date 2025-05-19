import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torchmetrics.functional.retrieval as tm_functional # Using functional for NDCG

# Configuration & Device
CONFIG = {
    "test_size": 0.15,
    "val_size": 0.15,
    "embedding_dim_ids": 32,      # Embedding dim for user_id, item_id
    "embedding_dim_features": 16, # Embedding dim for other categorical features
    "final_mlp_embed_dim": 64,    # Output dimension of user/item tower MLPs (this is the dim used for dot product)
    "learning_rate": 5e-4,        # Adjusted learning rate
    "weight_decay": 1e-5,
    "epochs": 22,                 # Might need more with features
    "batch_size": 1024,
    "num_neg_samples": 4,
    "top_k": 10,
    "stratify_bins": 5,
    "random_state": 42,
    "loss_type": "BPR"            # "BCE" or "BPR"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature Column Names 
USER_ID_COL = 'user_idx'
ITEM_ID_COL = 'item_idx'
USER_FEATURE_COLS = {
    'segment': 'segment_idx', 'beh_segment': 'beh_segment_idx', 'active_ind': 'active_ind_idx'
}
ITEM_FEATURE_COLS = {
    'item_type': 'item_type_idx', 'item_descrip': 'item_descrip_idx'
}
ALL_USER_FEATURES_INDICES = [USER_ID_COL] + list(USER_FEATURE_COLS.values())
ALL_ITEM_FEATURES_INDICES = [ITEM_ID_COL] + list(ITEM_FEATURE_COLS.values())


# ----- Data Loading & Preprocessing -----
def load_and_preprocess_data():
    print(f"Using device: {device}, Config: {CONFIG}")
    try:
        df_raw = pd.read_csv("../../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv")
    except FileNotFoundError:
        print("Using dummy data as file not found.")
        n_users_dummy, n_items_dummy, n_interactions_dummy = 500, 100, 20000
        user_ids_raw_dummy = [f"user_{i}" for i in np.random.randint(1, n_users_dummy + 1, n_interactions_dummy)]
        item_ids_raw_dummy = [f"ITEM_{np.random.randint(1, n_items_dummy + 1)}" for _ in range(n_interactions_dummy)]
        interactions_dummy = np.random.choice(['CLICK', 'CHECKOUT', 'DISPLAY'], size=n_interactions_dummy, p=[0.3, 0.15, 0.55])
        dates_dummy = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 90, n_interactions_dummy), unit='D')
        
        df_raw = pd.DataFrame({
            USER_ID_COL_RAW: user_ids_raw_dummy, 'interaction': interactions_dummy, 
            'int_date': dates_dummy.strftime('%d-%b-%y'), ITEM_ID_COL_RAW: item_ids_raw_dummy
        })
        for feat in RAW_USER_FEATURE_COLS.keys(): df_raw[feat] = [f"{feat}_val{np.random.randint(1,4)}" for _ in range(len(df_raw))]
        for feat in RAW_ITEM_FEATURE_COLS.keys(): df_raw[feat] = [f"{feat}_val{np.random.randint(1,4)}" for _ in range(len(df_raw))]

    df_raw['int_date'] = pd.to_datetime(df_raw['int_date'], format='%d-%b-%y')
    df_raw['user_id_str'] = df_raw[USER_ID_COL_RAW].astype(str)
    df_raw['item_id_str'] = df_raw[ITEM_ID_COL_RAW].astype(str)

    feature_encoders = {}
    for col_str, new_col_idx in [('user_id_str', USER_ID_COL), ('item_id_str', ITEM_ID_COL)]:
        le = LabelEncoder()
        df_raw[new_col_idx] = le.fit_transform(df_raw[col_str])
        feature_encoders[new_col_idx] = le

    for raw_col, processed_idx_col_name in RAW_USER_FEATURE_COLS.items():
        df_raw[raw_col] = df_raw[raw_col].fillna("UNKNOWN").astype(str)
        le = LabelEncoder()
        df_raw[processed_idx_col_name] = le.fit_transform(df_raw[raw_col])
        feature_encoders[processed_idx_col_name] = le

    for raw_col, processed_idx_col_name in RAW_ITEM_FEATURE_COLS.items():
        df_raw[raw_col] = df_raw[raw_col].fillna("UNKNOWN").astype(str)
        le = LabelEncoder()
        df_raw[processed_idx_col_name] = le.fit_transform(df_raw[raw_col])
        feature_encoders[processed_idx_col_name] = le
    
    num_total_users = len(feature_encoders[USER_ID_COL].classes_)
    num_total_items = len(feature_encoders[ITEM_ID_COL].classes_)
    print(f"Total unique users: {num_total_users}, Total unique items: {num_total_items}")

    user_side_feature_idx_cols = list(RAW_USER_FEATURE_COLS.values())
    user_features_df = df_raw[[USER_ID_COL] + user_side_feature_idx_cols].drop_duplicates(subset=[USER_ID_COL]).set_index(USER_ID_COL).sort_index()
    
    item_side_feature_idx_cols = list(RAW_ITEM_FEATURE_COLS.values())
    item_features_df = df_raw[[ITEM_ID_COL] + item_side_feature_idx_cols].drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL).sort_index()

    df_positive_interactions = df_raw[df_raw['interaction'].isin(['CLICK', 'CHECKOUT'])].copy()
    
    # --- Prepare data for Hard Negative Sampling ---
    print("Preparing data for Hard Negative Sampling...")
    # Interactions with 'DISPLAY' (using encoded IDs)
    display_interactions_df = df_raw[df_raw['interaction'] == 'DISPLAY'][[USER_ID_COL, ITEM_ID_COL]]
    # Positive interactions (CLICK or CHECKOUT) (using encoded IDs)
    # We use df_positive_interactions which is already filtered
    
    # Create sets of positive items for each user for faster lookup
    user_positive_items_map = df_positive_interactions.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()

    user_to_hard_negatives_map = {}
    # Group displayed items by user
    for user_id_encoded, group in display_interactions_df.groupby(USER_ID_COL):
        displayed_items = set(group[ITEM_ID_COL])
        positive_items_for_this_user = user_positive_items_map.get(user_id_encoded, set())
        hard_negatives = list(displayed_items - positive_items_for_this_user)
        if hard_negatives: 
            user_to_hard_negatives_map[user_id_encoded] = hard_negatives
    print(f"Created hard negative map for {len(user_to_hard_negatives_map)} users.")

    # --- Prepare data for Popularity Baseline for Coldest Users ---
    print("Preparing data for Popularity Baseline...")
    all_user_ids_from_raw = set(df_raw[USER_ID_COL].unique())
    positive_user_ids = set(df_positive_interactions[USER_ID_COL].unique())
    cold_user_ids_set = all_user_ids_from_raw - positive_user_ids
    print(f"Identified {len(cold_user_ids_set)} cold users (no CLICK/CHECKOUT interactions).")

    item_popularity = df_positive_interactions[ITEM_ID_COL].value_counts()
    popular_items_ranked_list = item_popularity.index.tolist()
    print(f"Created ranked list of {len(popular_items_ranked_list)} popular items.")

    return (df_positive_interactions, feature_encoders, user_features_df, item_features_df,
            user_to_hard_negatives_map, cold_user_ids_set, popular_items_ranked_list)

class RecommenderDataset(Dataset):
    def __init__(self, positive_interactions_df, user_pos_items_map,
                 user_displayed_items_map, all_item_indices_list, 
                 user_features_df, item_features_df,
                 loss_type="BCE", num_neg_samples=4):
        self.positive_interactions_df = positive_interactions_df.drop_duplicates(subset=[USER_ID_COL, ITEM_ID_COL]) 
        self.user_pos_items_map = user_pos_items_map
        self.user_displayed_items_map = user_displayed_items_map
        self.all_item_indices_list = all_item_indices_list 
        self.user_features_df = user_features_df
        self.item_features_df = item_features_df
        self.loss_type = loss_type
        self.num_neg_samples = num_neg_samples
        self.training_samples = self._create_training_samples()

    def _sample_negative(self, user_idx, positive_item_idx):
        user_pos_set = self.user_pos_items_map.get(user_idx, set())
        user_disp_set = self.user_displayed_items_map.get(user_idx, set())
        possible_informed_negs = list(user_disp_set - user_pos_set - {positive_item_idx})
        if possible_informed_negs: return np.random.choice(possible_informed_negs)
        while True:
            neg_item_idx = np.random.choice(self.all_item_indices_list)
            if neg_item_idx not in user_pos_set and neg_item_idx != positive_item_idx: return neg_item_idx

    def _get_user_features_tensor(self, user_idx):
        try: features = self.user_features_df.loc[user_idx]
        except KeyError: raise KeyError(f"User index {user_idx} not found in user_features_df. Available: {self.user_features_df.index[:5]}")
        return torch.tensor([user_idx] + [features[col_idx] for col_idx in USER_FEATURE_COLS.values()], dtype=torch.long)

    def _get_item_features_tensor(self, item_idx):
        try: features = self.item_features_df.loc[item_idx]
        except KeyError: raise KeyError(f"Item index {item_idx} not found in item_features_df. Available: {self.item_features_df.index[:5]}")
        return torch.tensor([item_idx] + [features[col_idx] for col_idx in ITEM_FEATURE_COLS.values()], dtype=torch.long)

    def _create_training_samples(self):
        samples = []
        desc = f"Creating {self.loss_type} samples";
        for _, row in tqdm(self.positive_interactions_df.iterrows(), total=len(self.positive_interactions_df), desc=desc):
            user_idx, pos_item_idx = row[USER_ID_COL], row[ITEM_ID_COL]
            try:
                user_features_tensor = self._get_user_features_tensor(user_idx)
                pos_item_features_tensor = self._get_item_features_tensor(pos_item_idx)
            except KeyError as e: continue 
            if self.loss_type == "BCE":
                samples.append({'user_features': user_features_tensor, 'item_features': pos_item_features_tensor, 'label': torch.tensor(1.0)})
                for _ in range(self.num_neg_samples):
                    neg_item_idx = self._sample_negative(user_idx, pos_item_idx)
                    try: neg_item_features_tensor = self._get_item_features_tensor(neg_item_idx)
                    except KeyError as e: continue
                    samples.append({'user_features': user_features_tensor, 'item_features': neg_item_features_tensor, 'label': torch.tensor(0.0)})
            elif self.loss_type == "BPR":
                neg_item_idx = self._sample_negative(user_idx, pos_item_idx)
                try: neg_item_features_tensor = self._get_item_features_tensor(neg_item_idx)
                except KeyError as e: continue
                samples.append({'user_features': user_features_tensor, 'pos_item_features': pos_item_features_tensor, 'neg_item_features': neg_item_features_tensor})
        return samples
    def __len__(self): return len(self.training_samples)
    def __getitem__(self, idx): return self.training_samples[idx]

class TwoTowerModelWithFeatures(nn.Module):
    def __init__(self, feature_encoders, config):
        super().__init__(); self.config = config; self.feature_encoders = feature_encoders
        self.user_id_emb = nn.Embedding(len(feature_encoders[USER_ID_COL].classes_), config['embedding_dim_ids'])
        user_feature_embeddings = {}; total_user_feature_dim = config['embedding_dim_ids']
        for raw_col, idx_col in USER_FEATURE_COLS.items():
            num_embeddings = len(feature_encoders[idx_col].classes_); emb = nn.Embedding(num_embeddings, config['embedding_dim_features'])
            user_feature_embeddings[idx_col] = emb; total_user_feature_dim += config['embedding_dim_features']
        self.user_feature_embeddings = nn.ModuleDict(user_feature_embeddings)
        
        # User MLP
        self.user_mlp = nn.Sequential(nn.Linear(total_user_feature_dim, total_user_feature_dim // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(total_user_feature_dim // 2, config['final_mlp_embed_dim']))
        self.item_id_emb = nn.Embedding(len(feature_encoders[ITEM_ID_COL].classes_), config['embedding_dim_ids'])
        item_feature_embeddings = {}; total_item_feature_dim = config['embedding_dim_ids']
        for raw_col, idx_col in ITEM_FEATURE_COLS.items():
            num_embeddings = len(feature_encoders[idx_col].classes_); emb = nn.Embedding(num_embeddings, config['embedding_dim_features'])
            item_feature_embeddings[idx_col] = emb; total_item_feature_dim += config['embedding_dim_features']
        self.item_feature_embeddings = nn.ModuleDict(item_feature_embeddings)
        
        # Item MLP
        self.item_mlp = nn.Sequential(nn.Linear(total_item_feature_dim, total_item_feature_dim // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(total_item_feature_dim // 2, config['final_mlp_embed_dim']))
        self._init_weights()
    def _init_weights(self):
        for emb_layer in [self.user_id_emb, self.item_id_emb] + list(self.user_feature_embeddings.values()) + list(self.item_feature_embeddings.values()): nn.init.xavier_uniform_(emb_layer.weight)
        for mlp_layer in [self.user_mlp, self.item_mlp]:
            for layer in mlp_layer:
                if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias) if layer.bias is not None else None
    def get_user_representation(self, user_features_tensor_batch):
        u_id_emb = self.user_id_emb(user_features_tensor_batch[:, 0]); feature_embs = [u_id_emb]
        for i, idx_col in enumerate(USER_FEATURE_COLS.values()): feature_embs.append(self.user_feature_embeddings[idx_col](user_features_tensor_batch[:, i+1]))
        return self.user_mlp(torch.cat(feature_embs, dim=1))
    def get_item_representation(self, item_features_tensor_batch):
        i_id_emb = self.item_id_emb(item_features_tensor_batch[:, 0]); feature_embs = [i_id_emb]
        for i, idx_col in enumerate(ITEM_FEATURE_COLS.values()): feature_embs.append(self.item_feature_embeddings[idx_col](item_features_tensor_batch[:, i+1]))
        return self.item_mlp(torch.cat(feature_embs, dim=1))
    def forward(self, user_features_batch, item_features_batch):
        user_repr = self.get_user_representation(user_features_batch); item_repr = self.get_item_representation(item_features_batch)
        return (user_repr * item_repr).sum(dim=1)

class BPRLoss(nn.Module):
    def __init__(self): super().__init__(); self.epsilon = 1e-9
    def forward(self, pos_scores, neg_scores): return -torch.log(torch.sigmoid(pos_scores - neg_scores) + self.epsilon).mean()

def evaluate_model(eval_model, user_indices_to_eval, ground_truth_map,
                   user_features_df, item_features_df,
                   top_k_val, desc="Evaluating"):
    eval_model.eval()
    all_user_recalls_list = []
    all_user_precisions_list = []
    all_user_ndcgs_list = [] # List to store per-user NDCG

    if not user_indices_to_eval or not ground_truth_map:
        return {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}
    
    unique_eval_users_list = list(user_indices_to_eval)
    if not unique_eval_users_list:
        return {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}

    all_item_indices_sorted = sorted(item_features_df.index.tolist())
    all_items_features_list = []
    for item_idx_sorted in all_item_indices_sorted:
        features = item_features_df.loc[item_idx_sorted]
        all_items_features_list.append([item_idx_sorted] + [features[col_idx] for col_idx in ITEM_FEATURE_COLS.values()])
    all_items_features_t = torch.tensor(all_items_features_list, dtype=torch.long, device=device)
    
    sorted_idx_to_original_item_idx_map = {i: item_idx for i, item_idx in enumerate(all_item_indices_sorted)}

    with torch.no_grad():
        item_representations_all = eval_model.get_item_representation(all_items_features_t)

        eval_users_features_list = []
        valid_users_for_processing = []
        for user_idx_original in unique_eval_users_list:
            if user_idx_original in user_features_df.index:
                 features = user_features_df.loc[user_idx_original]
                 eval_users_features_list.append([user_idx_original] + [features[col_idx] for col_idx in USER_FEATURE_COLS.values()])
                 valid_users_for_processing.append(user_idx_original)

        if not eval_users_features_list:
            print(f"Warning: No valid users found for feature processing in {desc}.")
            return {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}

        eval_users_features_t = torch.tensor(eval_users_features_list, dtype=torch.long, device=device)
        user_representations_eval = eval_model.get_user_representation(eval_users_features_t)
        all_scores_eval_batched = torch.matmul(user_representations_eval, item_representations_all.T)

        for i, user_idx_original in enumerate(tqdm(valid_users_for_processing, desc=desc, leave=False)):
            user_scores = all_scores_eval_batched[i] # Shape: (num_total_items,)
            actual_pos_items = ground_truth_map.get(user_idx_original, set())
            
            # Recall and Precision
            if not actual_pos_items:
                all_user_recalls_list.append(0.0)
                all_user_precisions_list.append(0.0)
            else:
                _, topk_sorted_indices = torch.topk(user_scores, k=min(top_k_val, len(user_scores)))
                recommended_original_item_indices = {sorted_idx_to_original_item_idx_map[idx.item()] for idx in topk_sorted_indices}
                hits = len(recommended_original_item_indices.intersection(actual_pos_items))
                
                recall = hits / len(actual_pos_items)
                precision = hits / top_k_val
                all_user_recalls_list.append(recall)
                all_user_precisions_list.append(precision)

            # For Functional NDCG: target tensor needs to be 1D and aligned with user_scores
            target_tensor = torch.zeros_like(user_scores, dtype=torch.bool, device=device) # Shape: (num_total_items,)
            for pos_item_idx in actual_pos_items:
                try:
                    sorted_idx_for_pos_item = all_item_indices_sorted.index(pos_item_idx)
                    target_tensor[sorted_idx_for_pos_item] = True
                except ValueError: pass
            
            ndcg_val = tm_functional.retrieval_normalized_dcg(
                user_scores.float(),  
                target_tensor,
                top_k=top_k_val
            )
            all_user_ndcgs_list.append(ndcg_val.item()) # .item() to get Python number

    mean_recall = np.mean(all_user_recalls_list) if all_user_recalls_list else 0.0
    mean_precision = np.mean(all_user_precisions_list) if all_user_precisions_list else 0.0
    mean_ndcg = np.mean(all_user_ndcgs_list) if all_user_ndcgs_list else 0.0
    
    return {"recall": mean_recall, "precision": mean_precision, "ndcg": mean_ndcg}

# Popularity Baseline Function (using functional NDCG) 
def calculate_popularity_baseline(train_df_positive, user_indices_to_eval, ground_truth_map,
                                  all_item_indices_sorted, # Master sorted list of all item_idx
                                  top_k_val, desc="Popularity Baseline"):
    if train_df_positive.empty:
        print("Train_df for popularity baseline is empty. Skipping baseline.")
        return {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}
        
    item_popularity = train_df_positive[ITEM_ID_COL].value_counts()
    top_k_popular_items_set = set(item_popularity.nlargest(top_k_val).index.tolist())

    all_user_recalls_list = []
    all_user_precisions_list = []
    all_user_ndcgs_list = [] # List for per-user NDCG

    if not user_indices_to_eval:
         return {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}

    original_item_idx_to_sorted_idx_map = {item_idx: i for i, item_idx in enumerate(all_item_indices_sorted)}
    num_total_items_global = len(all_item_indices_sorted)

    base_scores_popular_all_items = torch.zeros(num_total_items_global, device=device, dtype=torch.float) # Ensure float for scores
    for rank_idx, popular_item_idx in enumerate(item_popularity.nlargest(num_total_items_global).index):
        if popular_item_idx in original_item_idx_to_sorted_idx_map:
            sorted_idx = original_item_idx_to_sorted_idx_map[popular_item_idx]
            base_scores_popular_all_items[sorted_idx] = float(num_total_items_global - rank_idx) 

    for user_idx in tqdm(user_indices_to_eval, desc=desc, leave=False):
        actual_pos_items = ground_truth_map.get(user_idx, set())
        
        # Recall and Precision
        if not actual_pos_items:
            all_user_recalls_list.append(0.0)
            all_user_precisions_list.append(0.0)
        else:
            hits = len(top_k_popular_items_set.intersection(actual_pos_items))
            recall = hits / len(actual_pos_items)
            precision = hits / top_k_val
            all_user_recalls_list.append(recall)
            all_user_precisions_list.append(precision)

        # For Functional NDCG
        target_tensor_popular = torch.zeros(num_total_items_global, dtype=torch.bool, device=device)
        for pos_item_idx in actual_pos_items:
            if pos_item_idx in original_item_idx_to_sorted_idx_map:
                sorted_idx = original_item_idx_to_sorted_idx_map[pos_item_idx]
                target_tensor_popular[sorted_idx] = True
        
        ndcg_val_pop = tm_functional.retrieval_normalized_dcg(
            base_scores_popular_all_items, # Shape (D,)
            target_tensor_popular,         # Shape (D,)
            top_k=top_k_val
        )
        all_user_ndcgs_list.append(ndcg_val_pop.item())

    mean_recall_pop = np.mean(all_user_recalls_list) if all_user_recalls_list else 0.0
    mean_precision_pop = np.mean(all_user_precisions_list) if all_user_precisions_list else 0.0
    mean_ndcg_pop = np.mean(all_user_ndcgs_list) if all_user_ndcgs_list else 0.0

    return {"recall": mean_recall_pop, "precision": mean_precision_pop, "ndcg": mean_ndcg_pop}

# ----- Training Loop with Early Stopping & Hard Negative Sampling -----
def train_model(model, train_loader, val_loader, optimizer, num_total_items, 
                global_item_features_tensor, # Renamed for clarity
                user_to_hard_negatives_map,  # New argument
                user_features_df_for_eval, item_features_df_for_eval,
                ground_truth_val_df):
    
    best_val_ndcg = -1.0
    epochs_no_improve = 0
    best_model_state_dict = None
    
    # Configuration for Hard Negative Sampling
    HARD_NEGATIVE_SAMPLING_PROBABILITY = CONFIG.get("hard_negative_sampling_probability", 0.7) # Make it configurable

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for user_ids_batch, pos_item_ids_batch, user_side_feats_batch, pos_item_side_feats_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            user_ids_batch, pos_item_ids_batch = user_ids_batch.to(device), pos_item_ids_batch.to(device)
            user_side_feats_batch, pos_item_side_feats_batch = user_side_feats_batch.to(device), pos_item_side_feats_batch.to(device)

            batch_size = user_ids_batch.size(0)
            neg_item_ids_list = []

            for i in range(batch_size):
                current_user_id = user_ids_batch[i].item()
                current_pos_item_id = pos_item_ids_batch[i].item()
                sampled_neg_item_id = -1

                if np.random.rand() < HARD_NEGATIVE_SAMPLING_PROBABILITY:
                    hard_negs_for_user = user_to_hard_negatives_map.get(current_user_id, [])
                    # Ensure hard negatives do not include the current positive item
                    # (should be true by definition but good to be safe if lists are pre-filtered)
                    # valid_hard_negs = [neg for neg in hard_negs_for_user if neg != current_pos_item_id]
                    valid_hard_negs = hard_negs_for_user # Assuming map creation already handled this
                    
                    if valid_hard_negs:
                        sampled_neg_item_id = np.random.choice(valid_hard_negs)
                
                if sampled_neg_item_id == -1: # Fallback to random sampling
                    while True:
                        random_neg_item_id = np.random.randint(0, num_total_items)
                        if random_neg_item_id != current_pos_item_id:
                            sampled_neg_item_id = random_neg_item_id
                            break
                neg_item_ids_list.append(sampled_neg_item_id)
            
            neg_item_ids = torch.tensor(neg_item_ids_list, dtype=torch.long, device=device)
            neg_item_side_feats = global_item_features_tensor[neg_item_ids] # Fetch side features

            user_vec = model.user_tower(user_ids_batch, user_side_feats_batch)
            pos_item_vec = model.item_tower(pos_item_ids_batch, pos_item_side_feats_batch)
            neg_item_vec = model.item_tower(neg_item_ids, neg_item_side_feats)

            pos_scores = torch.sum(user_vec * pos_item_vec, dim=1)
            neg_scores = torch.sum(user_vec * neg_item_vec, dim=1)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f}")

        if val_loader and len(val_loader.dataset)>0: # Check if val_loader is valid and has data
            val_metrics = evaluate_model(model, val_loader, num_total_items, 
                                         item_features_df_for_eval,
                                         user_features_df_for_eval, # user_features_df_for_eval is not used by current evaluate_model
                                         ground_truth_val_df, desc="Validating")
            
            val_ndcg = val_metrics[f"NDCG@{CONFIG['top_k']}"] 
            print(f"Epoch {epoch+1} Validation - Recall: {val_metrics[f'Recall@{CONFIG['top_k']}']:.4f}, "
                  f"Precision: {val_metrics[f'Precision@{CONFIG['top_k']}']:.4f}, NDCG: {val_ndcg:.4f}")

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                epochs_no_improve = 0
                best_model_state_dict = copy.deepcopy(model.state_dict())
                print(f"New best validation NDCG: {best_val_ndcg:.4f}. Saving model.")
            else:
                epochs_no_improve += 1
                print(f"Validation NDCG did not improve. Count: {epochs_no_improve}/{CONFIG['early_stopping_patience']}")

            if epochs_no_improve >= CONFIG['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        else: # No validation loader or empty validation dataset
             print("Skipping validation as validation loader/data is not available.")


    if best_model_state_dict:
        print("Loading best model weights for final evaluation.")
        model.load_state_dict(best_model_state_dict)
    return model


def get_recommendations_for_user_conceptual(user_id_to_query, model, 
                                            user_features_tensor, # Full tensor of user side features
                                            all_item_ids_tensor, # Tensor of all item IDs e.g. torch.arange(num_items)
                                            all_items_side_features_tensor, # Full tensor of item side features
                                            cold_user_ids_set, popular_items_ranked_list, 
                                            config, device):
    """
    Conceptual function to get recommendations for a single user,
    applying popularity baseline for cold users or model for known users.
    """
    if user_id_to_query in cold_user_ids_set:
        # Cold user: recommend most popular items
        # Ensure popular_items_ranked_list contains encoded item_idx values
        return popular_items_ranked_list[:config['top_k']]
    else:
        # Known user: use the trained model
        model.eval()
        with torch.no_grad():
            # Get the specific user's ID and side features
            user_id_tensor = torch.tensor([user_id_to_query], dtype=torch.long).to(device)
            
            # user_features_tensor is indexed by user_idx.
            # Ensure user_id_to_query is a valid index for user_features_tensor
            if user_id_to_query < user_features_tensor.shape[0]:
                 user_side_feats = user_features_tensor[user_id_to_query].unsqueeze(0).to(device) # Add batch dim
            else: # Fallback if user_id somehow out of bounds for precomputed feature tensor
                print(f"Warning: User ID {user_id_to_query} out of bounds for user_features_tensor. Using zero features.")
                # Assuming user_features_tensor has shape (num_users, num_user_features)
                # And the second dimension is user_side_feature_idx_cols length
                num_user_side_features = user_features_tensor.shape[1] if user_features_tensor.ndim == 2 else 0
                user_side_feats = torch.zeros((1, num_user_side_features), dtype=torch.long).to(device)

            user_embedding = model.user_tower(user_id_tensor, user_side_feats) # (1, embed_dim)
            
            # Get all item embeddings (this can be precomputed if all_item_vectors are available globally)
            all_item_embeddings = model.item_tower(all_item_ids_tensor, all_items_side_features_tensor) # (num_items, embed_dim)
            
            scores = torch.matmul(user_embedding, all_item_embeddings.T).squeeze(0) # (num_items)
            top_k_scores, top_k_indices = torch.topk(scores, k=config['top_k'])
            
            return top_k_indices.cpu().tolist()
        
        
# ----- Main Execution -----
def main():
    # Unpack new returned values
    (df_pos, encoders, user_features_df, item_features_df,
     user_to_hard_negatives_map, cold_user_ids_set, popular_items_ranked_list) = load_and_preprocess_data()

    # --- Sanity check prints for new data (optional) ---
    print(f"\n--- Post-processing Info ---")
    # Print some stats about hard negatives map
    num_users_with_hard_negs = len(user_to_hard_negatives_map)
    avg_hard_negs_per_user = np.mean([len(v) for v in user_to_hard_negatives_map.values()]) if num_users_with_hard_negs > 0 else 0
    print(f"Number of users with identified hard negatives: {num_users_with_hard_negs}")
    if num_users_with_hard_negs > 0:
      print(f"Average number of hard negatives per user (with hard negs): {avg_hard_negs_per_user:.2f}")
    # Print some popular items
    print(f"Top 5 most popular items (by CLICK/CHECKOUT): {popular_items_ranked_list[:5]}")
    # Print some cold user IDs (example)
    # print(f"Example cold user IDs: {list(cold_user_ids_set)[:5]}") # Be careful if IDs are sensitive

    user_side_feature_idx_cols = list(RAW_USER_FEATURE_COLS.values())
    item_side_feature_idx_cols = list(RAW_ITEM_FEATURE_COLS.values())

    global_user_features_tensor = torch.tensor(user_features_df[user_side_feature_idx_cols].values, dtype=torch.long).to(device)
    global_item_features_tensor = torch.tensor(item_features_df[item_side_feature_idx_cols].values, dtype=torch.long).to(device)

    all_user_ids = df_pos[USER_ID_COL].unique() # Users with positive interactions
    
    # Ensure that train_val_user_ids and test_user_ids are not empty if all_user_ids is small
    if len(all_user_ids) < 2 : # Need at least 2 users to attempt a split for train/test
        print("Warning: Very few users with positive interactions to perform a robust train/test split.")
        train_val_user_ids = all_user_ids
        test_user_ids = np.array([]) 
    else:
        train_val_user_ids, test_user_ids = train_test_split(all_user_ids, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])

    if len(train_val_user_ids) == 0 : # If test_size was 1.0 or no users left
         train_user_ids = np.array([])
         val_user_ids = np.array([])
    elif len(train_val_user_ids) < 2 and (CONFIG['val_size'] > 0) : # Not enough to split train_val further
        print("Warning: Not enough users in train_val pool to create a separate validation set.")
        train_user_ids = train_val_user_ids
        val_user_ids = np.array([])
    else:
        relative_val_size = CONFIG['val_size'] / (1 - CONFIG['test_size']) if (1 - CONFIG['test_size']) > 0 else 0
        if relative_val_size >= 1.0 or relative_val_size <= 0:
            print(f"Adjusting validation split due to relative_val_size: {relative_val_size:.2f}")
            if len(train_val_user_ids) > 1 :
                # Default to a small validation set if original val_size is problematic
                # Ensure test_size for this split is reasonable, e.g. 0.1 or based on remaining users
                val_split_proportion = 0.1 if (relative_val_size >=1.0 or relative_val_size <=0 and CONFIG['val_size'] > 0) else 0.0
                if val_split_proportion > 0 and len(train_val_user_ids) * val_split_proportion < 1: # avoid creating empty val set if not intended
                    train_user_ids = train_val_user_ids
                    val_user_ids = np.array([])
                elif val_split_proportion > 0 :
                     train_user_ids, val_user_ids = train_test_split(train_val_user_ids, test_size=val_split_proportion, random_state=CONFIG['random_state'])
                else: # No validation set
                    train_user_ids = train_val_user_ids
                    val_user_ids = np.array([])
            else:
                train_user_ids = train_val_user_ids
                val_user_ids = np.array([])
        else:
            train_user_ids, val_user_ids = train_test_split(train_val_user_ids, test_size=relative_val_size, random_state=CONFIG['random_state'])


    train_df = df_pos[df_pos[USER_ID_COL].isin(train_user_ids)] if len(train_user_ids) > 0 else pd.DataFrame(columns=df_pos.columns)
    val_df = df_pos[df_pos[USER_ID_COL].isin(val_user_ids)] if len(val_user_ids) > 0 else pd.DataFrame(columns=df_pos.columns)
    test_df = df_pos[df_pos[USER_ID_COL].isin(test_user_ids)] if len(test_user_ids) > 0 else pd.DataFrame(columns=df_pos.columns)


    print(f"Train size: {len(train_df)} interactions, {len(train_user_ids)} users")
    print(f"Validation size: {len(val_df)} interactions, {len(val_user_ids)} users")
    print(f"Test size: {len(test_df)} interactions, {len(test_user_ids)} users")

    # Create datasets only if their DFs are not empty
    train_dataset = InteractionDataset(train_df, global_user_features_tensor, global_item_features_tensor) if not train_df.empty else None
    val_dataset = InteractionDataset(val_df, global_user_features_tensor, global_item_features_tensor) if not val_df.empty else None
    test_dataset = InteractionDataset(test_df, global_user_features_tensor, global_item_features_tensor) if not test_df.empty else None

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=0) if test_dataset else None
    
    if not train_loader:
        print("No training data available. Exiting.")
        return

    num_total_users_from_encoders = len(encoders[USER_ID_COL].classes_) # Use this for embedding table size
    num_total_items_from_encoders = len(encoders[ITEM_ID_COL].classes_)

    user_side_feature_cardinalities = [len(encoders[idx_col].classes_) for idx_col in RAW_USER_FEATURE_COLS.values()]
    item_side_feature_cardinalities = [len(encoders[idx_col].classes_) for idx_col in RAW_ITEM_FEATURE_COLS.values()]

    model_common_config = {
        "embed_dim_id": CONFIG['embedding_dim_ids'],
        "embed_dim_feat": CONFIG['embedding_dim_features'],
        "transformer_nhead": CONFIG['transformer_nhead'],
        "transformer_nlayers": CONFIG['transformer_nlayers'],
        "transformer_dim_feedforward": CONFIG['transformer_dim_feedforward'],
        "out_dim": CONFIG['final_mlp_embed_dim']
    }
    model = TwoTowerModel(
        user_tower_config={**model_common_config, "id_dim": num_total_users_from_encoders, "feature_cardinalities": user_side_feature_cardinalities},
        item_tower_config={**model_common_config, "id_dim": num_total_items_from_encoders, "feature_cardinalities": item_side_feature_cardinalities}
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    # Pass new arguments to train_model
    trained_model = train_model(model, train_loader, val_loader, optimizer, 
                                num_total_items_from_encoders, # num_total_items for random sampling range
                                global_item_features_tensor,   # For fetching neg item side features
                                user_to_hard_negatives_map,    # For hard negative sampling
                                user_features_df, item_features_df, 
                                val_df if val_dataset else pd.DataFrame()) # Pass val_df for ground truth

    print("\n--- Evaluating on Test Set (Best Model from Early Stopping) ---")
    if test_loader and not test_df.empty : # Check if test_df also not empty
        final_test_metrics = evaluate_model(trained_model, test_loader, num_total_items_from_encoders, 
                                            item_features_df, user_features_df, 
                                            test_df, desc="Testing")
        print(f"Final Test Model - Recall@{CONFIG['top_k']}: {final_test_metrics[f'Recall@{CONFIG['top_k']}']:.4f}, "
              f"Precision@{CONFIG['top_k']}: {final_test_metrics[f'Precision@{CONFIG['top_k']}']:.4f}, "
              f"NDCG@{CONFIG['top_k']}: {final_test_metrics[f'NDCG@{CONFIG['top_k']}']:.4f}")
    else:
        print("Test set is empty or test loader not available, skipping final evaluation.")
    
    print("\n--- Conceptual: How to use Popularity Baseline for a Cold User ---")
    # This is a conceptual demonstration, not part of the training/evaluation flow above
    example_cold_user_id = list(cold_user_ids_set)[0] if cold_user_ids_set else -1
    if example_cold_user_id != -1:
        recommendations = get_recommendations_for_user_conceptual(
            user_id_to_query=example_cold_user_id, 
            model=trained_model, 
            user_features_tensor=global_user_features_tensor, # Map user_id to its features for model
            all_item_ids_tensor=torch.arange(num_total_items_from_encoders, device=device),
            all_items_side_features_tensor=global_item_features_tensor,
            cold_user_ids_set=cold_user_ids_set, 
            popular_items_ranked_list=popular_items_ranked_list, 
            config=CONFIG,
            device=device
        )
        print(f"Recommendations for a cold user (e.g., {example_cold_user_id}) using popularity: {recommendations}")
    else:
        print("No cold users identified to demonstrate popularity baseline.")

    print("\nDone.")


if __name__ == "__main__":
    main()