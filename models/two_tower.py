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


def load_and_preprocess_data():
    print(f"Using device: {device}, Config: {CONFIG}")
    try:
        df_raw = pd.read_csv("../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv")
        print("Successfully loaded real dataset.")
    except FileNotFoundError:
        print("Dataset file not found.")

    df_raw['int_date'] = pd.to_datetime(df_raw['int_date'], format='%d-%b-%y')
    df_raw['user_id_str'] = df_raw['idcol'].astype(str)
    df_raw['item_id_str'] = df_raw['item'].astype(str) 

    feature_encoders = {}
    for col, new_col_idx in [( 'user_id_str', USER_ID_COL), ('item_id_str', ITEM_ID_COL)]:
        le = LabelEncoder()
        df_raw[new_col_idx] = le.fit_transform(df_raw[col])
        feature_encoders[new_col_idx] = le

    for col, new_col_idx in ITEM_FEATURE_COLS.items():
        df_raw[col] = df_raw[col].fillna("UNKNOWN").astype(str)
        le = LabelEncoder(); df_raw[new_col_idx] = le.fit_transform(df_raw[col])
        feature_encoders[new_col_idx] = le

    for col, new_col_idx in USER_FEATURE_COLS.items():
        df_raw[col] = df_raw[col].fillna("UNKNOWN").astype(str)
        le = LabelEncoder(); df_raw[new_col_idx] = le.fit_transform(df_raw[col])
        feature_encoders[new_col_idx] = le

    num_total_users = len(feature_encoders[USER_ID_COL].classes_)
    num_total_items = len(feature_encoders[ITEM_ID_COL].classes_)
    print(f"Total unique users: {num_total_users}, Total unique items: {num_total_items}")

    user_features_df = df_raw[ALL_USER_FEATURES_INDICES].drop_duplicates(subset=[USER_ID_COL]).set_index(USER_ID_COL).sort_index()
    item_features_df = df_raw[ALL_ITEM_FEATURES_INDICES].drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL).sort_index()
    
    df_positive_interactions = df_raw[df_raw['interaction'].isin(['CLICK', 'CHECKOUT'])].copy()
    df_display_interactions = df_raw[df_raw['interaction'] == 'DISPLAY'].copy()
    
    return df_raw, df_positive_interactions, df_display_interactions, feature_encoders, user_features_df, item_features_df

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


def main():
    df_raw, df_positive_interactions, df_display_interactions, feature_encoders, \
    user_features_df, item_features_df = load_and_preprocess_data()

    user_activity = df_positive_interactions.groupby(USER_ID_COL).size().reset_index(name='interaction_count')
    if user_activity.empty: raise ValueError("No positive interactions for split.")
    
    q_bins = min(CONFIG['stratify_bins'], max(1, user_activity[USER_ID_COL].nunique() -1 if user_activity[USER_ID_COL].nunique() >1 else 1) )
    user_activity['activity_bin'] = pd.qcut(
        user_activity['interaction_count'], q=q_bins, labels=False, duplicates='drop'
    )

    train_val_users, test_users_df = train_test_split(
        user_activity, test_size=CONFIG['test_size'],
        stratify=user_activity['activity_bin'] if user_activity['activity_bin'].nunique() > 1 else None,
        random_state=CONFIG['random_state']
    )
    relative_val_size = CONFIG['val_size'] / (1 - CONFIG['test_size']) if (1 - CONFIG['test_size']) > 0 else 0
    train_users_df, val_users_df = train_test_split(
        train_val_users, test_size=relative_val_size,
        stratify=train_val_users['activity_bin'] if train_val_users['activity_bin'].nunique() > 1 else None,
        random_state=CONFIG['random_state']
    )
    train_user_indices = set(train_users_df[USER_ID_COL]); val_user_indices = set(val_users_df[USER_ID_COL]); test_user_indices = set(test_users_df[USER_ID_COL])
    train_df = df_positive_interactions[df_positive_interactions[USER_ID_COL].isin(train_user_indices)]
    val_df = df_positive_interactions[df_positive_interactions[USER_ID_COL].isin(val_user_indices)]
    test_df = df_positive_interactions[df_positive_interactions[USER_ID_COL].isin(test_user_indices)]
    print(f"Train interactions: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    user_pos_items_map_all = df_positive_interactions.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()
    user_displayed_items_map_all = df_display_interactions.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()
    master_all_item_indices_sorted = sorted(item_features_df.index.tolist())


    train_dataset = RecommenderDataset(
        train_df, user_pos_items_map_all, user_displayed_items_map_all,
        master_all_item_indices_sorted, 
        user_features_df, item_features_df,
        loss_type=CONFIG['loss_type'], num_neg_samples=CONFIG['num_neg_samples']
    )
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    model = TwoTowerModelWithFeatures(feature_encoders, CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    loss_fn = BPRLoss() if CONFIG['loss_type'] == "BPR" else nn.BCEWithLogitsLoss()

    val_ground_truth = val_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()
    test_ground_truth = test_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()

    print("\n--- Calculating Popularity Baseline (Validation Set) ---")
    if val_user_indices:
        pop_baseline_metrics_val = calculate_popularity_baseline(
            train_df, val_user_indices, val_ground_truth,
            master_all_item_indices_sorted, CONFIG['top_k']
        )
        print(f"Popularity Baseline (Validation) - Recall@{CONFIG['top_k']}: {pop_baseline_metrics_val['recall']:.4f}, "
              f"Precision@{CONFIG['top_k']}: {pop_baseline_metrics_val['precision']:.4f}, "
              f"NDCG@{CONFIG['top_k']}: {pop_baseline_metrics_val['ndcg']:.4f}")

    print("\n--- Starting Training ---")
    best_val_metrics = {"recall": -1.0, "precision": -1.0, "ndcg": -1.0} 
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        if CONFIG['loss_type'] == "BCE":
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
                user_f_batch = batch['user_features'].to(device); item_f_batch = batch['item_features'].to(device); labels_batch = batch['label'].to(device)
                optimizer.zero_grad(); logits = model(user_f_batch, item_f_batch); loss = loss_fn(logits, labels_batch.squeeze()); loss.backward(); optimizer.step(); epoch_loss += loss.item()
        elif CONFIG['loss_type'] == "BPR": 
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
                user_f_batch = batch['user_features'].to(device); pos_item_f_batch = batch['pos_item_features'].to(device); neg_item_f_batch = batch['neg_item_features'].to(device)
                optimizer.zero_grad(); pos_scores = model(user_f_batch, pos_item_f_batch); neg_scores = model(user_f_batch, neg_item_f_batch); loss = loss_fn(pos_scores, neg_scores); loss.backward(); optimizer.step(); epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f}")

        if val_user_indices:
            current_val_metrics = evaluate_model(model, val_user_indices, val_ground_truth,
                                                 user_features_df, item_features_df,
                                                 CONFIG['top_k'], desc="Validating")
            print(f"Epoch {epoch+1} Validation - Recall@{CONFIG['top_k']}: {current_val_metrics['recall']:.4f}, "
                  f"Precision@{CONFIG['top_k']}: {current_val_metrics['precision']:.4f}, "
                  f"NDCG@{CONFIG['top_k']}: {current_val_metrics['ndcg']:.4f}")
            
            if current_val_metrics['ndcg'] > best_val_metrics['ndcg']: 
                best_val_metrics = current_val_metrics
                # torch.save(model.state_dict(), "best_model_with_features.pth")
                print(f"New best validation NDCG: {best_val_metrics['ndcg']:.4f}. Model saved (simulated).")

    print("\n--- Calculating Popularity Baseline (Test Set) ---")
    if test_user_indices:
        pop_baseline_metrics_test = calculate_popularity_baseline(
            train_df, test_user_indices, test_ground_truth, 
            master_all_item_indices_sorted, CONFIG['top_k']
        )
        print(f"Popularity Baseline (Test) - Recall@{CONFIG['top_k']}: {pop_baseline_metrics_test['recall']:.4f}, "
              f"Precision@{CONFIG['top_k']}: {pop_baseline_metrics_test['precision']:.4f}, "
              f"NDCG@{CONFIG['top_k']}: {pop_baseline_metrics_test['ndcg']:.4f}")

    print("\n--- Evaluating on Test Set (Final Model) ---")
    if test_user_indices:
        # model.load_state_dict(torch.load("best_model_with_features.pth")) 
        final_test_metrics = evaluate_model(model, test_user_indices, test_ground_truth,
                                            user_features_df, item_features_df,
                                            CONFIG['top_k'], desc="Testing")
        print(f"Final Test Model - Recall@{CONFIG['top_k']}: {final_test_metrics['recall']:.4f}, "
              f"Precision@{CONFIG['top_k']}: {final_test_metrics['precision']:.4f}, "
              f"NDCG@{CONFIG['top_k']}: {final_test_metrics['ndcg']:.4f}")
    else:
        print("Test set is empty.")
    print("\nDone.")


if __name__ == "__main__":
    main()