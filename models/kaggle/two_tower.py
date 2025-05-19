import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torchmetrics.functional.retrieval as tm_functional
import copy

# Configuration & Device
CONFIG = {
    "test_size": 0.15,
    "val_size": 0.15,
    "embedding_dim_ids": 64,
    "embedding_dim_features": 32, # Embedding dim for Country, Description, Price Bins
    "final_mlp_embed_dim": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 30, 
    "batch_size": 2048,
    "num_neg_samples": 4,
    "top_k": 10,
    "random_state": 42,
    "loss_type": "BPR",
    "early_stopping_patience": 5,
    "price_bins": [0, 5, 10, 20, 50, 100, 200, np.inf], # Define price bins
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Column Name Definitions 
USER_ID_COL = 'user_idx'
ITEM_ID_COL = 'item_idx'

RAW_USER_ID_COL_NAME = 'CustomerID'
RAW_ITEM_ID_COL_NAME = 'StockCode'
RAW_PRICE_COL_NAME = 'UnitPrice' # For clarity when using it for binning

RAW_USER_CATEGORICAL_COLS = {'Country': 'country_idx'}
# Added Price Bin feature
RAW_ITEM_CATEGORICAL_COLS = {
    'Description': 'description_idx',
    'UnitPriceBin': 'price_bin_idx' # Conceptual name 'UnitPriceBin' maps to 'price_bin_idx'
}

MODEL_USER_FEATURE_COLS = RAW_USER_CATEGORICAL_COLS.copy()
MODEL_ITEM_FEATURE_COLS = RAW_ITEM_CATEGORICAL_COLS.copy()


def load_and_preprocess_data():
    print(f"Using device: {device}")
    try:
        df_raw = pd.read_csv("../../data/kaggle_dataset/raw/data.csv", encoding='ISO-8859-1')
    except FileNotFoundError:
        print("file not found.")
        
    print(f"Initial raw data rows: {len(df_raw)}")
    df_raw.dropna(subset=[RAW_USER_ID_COL_NAME, RAW_ITEM_ID_COL_NAME, RAW_PRICE_COL_NAME], inplace=True)
    df_raw = df_raw[(df_raw['Quantity'] > 0) & (df_raw[RAW_PRICE_COL_NAME] > 0)] # Combined filter
    print(f"Rows after cleaning (NA, Quantity>0, UnitPrice>0): {len(df_raw)}")

    if df_raw.empty: raise ValueError("DataFrame empty after cleaning. Cannot proceed.")

    df_raw[RAW_USER_ID_COL_NAME] = df_raw[RAW_USER_ID_COL_NAME].astype(float).astype(int).astype(str)
    df_raw[RAW_ITEM_ID_COL_NAME] = df_raw[RAW_ITEM_ID_COL_NAME].astype(str)

    feature_encoders = {}
    for raw_col, internal_col in [(RAW_USER_ID_COL_NAME, USER_ID_COL), (RAW_ITEM_ID_COL_NAME, ITEM_ID_COL)]:
        le = LabelEncoder()
        df_raw[internal_col] = le.fit_transform(df_raw[raw_col])
        feature_encoders[internal_col] = le
    
    for raw_feat_name, idx_col_name in RAW_USER_CATEGORICAL_COLS.items():
        df_raw[raw_feat_name] = df_raw[raw_feat_name].fillna("UNKNOWN").astype(str)
        le = LabelEncoder()
        df_raw[idx_col_name] = le.fit_transform(df_raw[raw_feat_name])
        feature_encoders[idx_col_name] = le

    # Bin UnitPrice (RAW_PRICE_COL_NAME is 'UnitPrice')
    price_bins = CONFIG["price_bins"]
    price_labels = [f"price_bin_{i}" for i in range(len(price_bins)-1)]
    # Create an intermediate column for the binned price values. This will be Categorical.
    intermediate_binned_price_col = 'binned_price_values_temp' 
    df_raw[intermediate_binned_price_col] = pd.cut(
        df_raw[RAW_PRICE_COL_NAME],
        bins=price_bins,
        labels=price_labels,
        right=False,
        include_lowest=True
    )
    
    
    for conceptual_feature_name, target_idx_col_name in RAW_ITEM_CATEGORICAL_COLS.items():
        # conceptual_feature_name is 'Description' or 'UnitPriceBin' (the key from the dict)
        # target_idx_col_name is 'description_idx' or 'price_bin_idx' (the value from the dict)
        
        source_series_for_encoding = None # Initialize
        
        if conceptual_feature_name == 'UnitPriceBin':
            # For 'UnitPriceBin', the source is our intermediate binned categorical column
            source_series_for_encoding = df_raw[intermediate_binned_price_col]
        elif conceptual_feature_name == 'Description':
            # For 'Description', the source is the original 'Description' column in df_raw
            source_series_for_encoding = df_raw['Description'] 
        if source_series_for_encoding is None:
            print(f"Warning: Source series for item feature '{conceptual_feature_name}' not defined. Skipping.")
            continue
            
        
        processed_series = source_series_for_encoding.astype(str)
        
        processed_series = processed_series.fillna("UNKNOWN").replace('nan', "UNKNOWN")
        
            
        le = LabelEncoder()
        # Store the label encoded result into the target _idx column (e.g., 'price_bin_idx' or 'description_idx')
        df_raw[target_idx_col_name] = le.fit_transform(processed_series)
        feature_encoders[target_idx_col_name] = le
    
    # Clean up the intermediate column used for binned prices as it's now encoded
    if intermediate_binned_price_col in df_raw.columns:
        df_raw.drop(columns=[intermediate_binned_price_col], inplace=True)

    user_feat_cols = list(MODEL_USER_FEATURE_COLS.values())
    user_features_df = df_raw[[USER_ID_COL] + user_feat_cols].drop_duplicates(subset=[USER_ID_COL]).set_index(USER_ID_COL).sort_index()
    
    item_feat_cols = list(MODEL_ITEM_FEATURE_COLS.values()) # Will include 'description_idx' and 'price_bin_idx'
    item_features_df = df_raw[[ITEM_ID_COL] + item_feat_cols].drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL).sort_index()

    df_positive_interactions = df_raw.copy()
    user_pos_items_map = df_positive_interactions.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()
    user_to_hard_negatives_map = {} # No display data for hard negatives

    all_users_enc = set(df_raw[USER_ID_COL].unique())
    pos_interaction_users_enc = set(df_positive_interactions[USER_ID_COL].unique())
    cold_user_ids_set_encoded = all_users_enc - pos_interaction_users_enc
    print(f"Identified {len(cold_user_ids_set_encoded)} cold users (encoded).")

    item_popularity = df_positive_interactions[ITEM_ID_COL].value_counts()
    popular_items_ranked_list_encoded = item_popularity.index.tolist()
    
    return (df_positive_interactions, feature_encoders, user_features_df, item_features_df,
            user_pos_items_map, user_to_hard_negatives_map, 
            cold_user_ids_set_encoded, popular_items_ranked_list_encoded)

class RecommenderDataset(Dataset):
    def __init__(self, positive_interactions_df, user_pos_items_map,
                 user_displayed_items_map, all_item_indices_list, 
                 user_features_df, item_features_df,
                 loss_type="BCE", num_neg_samples=4):
        self.positive_interactions_df = positive_interactions_df[[USER_ID_COL, ITEM_ID_COL]].drop_duplicates()
        self.user_pos_items_map = user_pos_items_map
        self.user_displayed_items_map = user_displayed_items_map
        self.all_item_indices_list = all_item_indices_list
        self.user_features_df = user_features_df
        self.item_features_df = item_features_df
        self.loss_type = loss_type
        self.num_neg_samples = num_neg_samples
        if not self.all_item_indices_list: raise ValueError("all_item_indices_list is empty.")
        self.training_samples = self._create_training_samples()

    def _sample_negative(self, user_idx, positive_item_idx):
        user_pos_set = self.user_pos_items_map.get(user_idx, set())
        # user_disp_set and possible_informed_negs will be empty
        user_disp_set = self.user_displayed_items_map.get(user_idx, set())
        possible_informed_negs = list(user_disp_set - user_pos_set - {positive_item_idx})
        if possible_informed_negs: return np.random.choice(possible_informed_negs)
        while True: # Random sampling
            neg_item_idx = np.random.choice(self.all_item_indices_list)
            if neg_item_idx not in user_pos_set and neg_item_idx != positive_item_idx:
                return neg_item_idx

    def _get_features_tensor(self, entity_idx, features_df, model_feature_cols_map, entity_type="User"):
        try:
            features_row = features_df.loc[entity_idx]
        except KeyError:
            # print(f"Warning: {entity_type} index {entity_idx} not found. Using zero features.") # Reduced verbosity
            num_other_features = len(model_feature_cols_map.values())
            return torch.tensor([entity_idx] + [0] * num_other_features, dtype=torch.long)
        feature_values = [features_row[col_idx_name] for col_idx_name in model_feature_cols_map.values()]
        return torch.tensor([entity_idx] + feature_values, dtype=torch.long)

    def _get_user_features_tensor(self, user_idx):
        return self._get_features_tensor(user_idx, self.user_features_df, MODEL_USER_FEATURE_COLS, "User")

    def _get_item_features_tensor(self, item_idx):
        return self._get_features_tensor(item_idx, self.item_features_df, MODEL_ITEM_FEATURE_COLS, "Item")

    def _create_training_samples(self):
        samples = []
        # desc = f"Creating {self.loss_type} samples" # tqdm will show its own description
        for _, row in tqdm(self.positive_interactions_df.iterrows(), total=len(self.positive_interactions_df), desc="Creating samples", leave=False):
            user_idx, pos_item_idx = row[USER_ID_COL], row[ITEM_ID_COL]
            user_feats = self._get_user_features_tensor(user_idx)
            pos_item_feats = self._get_item_features_tensor(pos_item_idx)
            
            if self.loss_type == "BCE":
                samples.append({'user_features': user_feats, 'item_features': pos_item_feats, 'label': torch.tensor(1.0)})
                for _ in range(self.num_neg_samples):
                    neg_item_idx = self._sample_negative(user_idx, pos_item_idx)
                    samples.append({'user_features': user_feats, 'item_features': self._get_item_features_tensor(neg_item_idx), 'label': torch.tensor(0.0)})
            elif self.loss_type == "BPR":
                neg_item_idx = self._sample_negative(user_idx, pos_item_idx)
                samples.append({'user_features': user_feats, 
                                'pos_item_features': pos_item_feats, 
                                'neg_item_features': self._get_item_features_tensor(neg_item_idx)})
        return samples
    def __len__(self): return len(self.training_samples)
    def __getitem__(self, idx): return self.training_samples[idx]

class TwoTowerModelWithFeatures(nn.Module):
    def __init__(self, feature_encoders, config): 
        super().__init__()
        self.config = config
        
        # User Tower
        self.user_id_emb = nn.Embedding(len(feature_encoders[USER_ID_COL].classes_), config['embedding_dim_ids'])
        user_feature_embeddings = {}
        total_user_feature_dim = config['embedding_dim_ids']
        for _, feature_idx_col_name in MODEL_USER_FEATURE_COLS.items(): # Key is conceptual, value is _idx name
            num_embeddings = len(feature_encoders[feature_idx_col_name].classes_)
            emb = nn.Embedding(num_embeddings, config['embedding_dim_features'])
            user_feature_embeddings[feature_idx_col_name] = emb
            total_user_feature_dim += config['embedding_dim_features']
        self.user_feature_embeddings = nn.ModuleDict(user_feature_embeddings)
        self.user_mlp = nn.Sequential(
            nn.Linear(total_user_feature_dim, total_user_feature_dim * 2), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(total_user_feature_dim * 2, config['final_mlp_embed_dim'])
        )

        # Item Tower (includes Description and new Price Bin)
        self.item_id_emb = nn.Embedding(len(feature_encoders[ITEM_ID_COL].classes_), config['embedding_dim_ids'])
        item_feature_embeddings = {}
        total_item_feature_dim = config['embedding_dim_ids']
        for _, feature_idx_col_name in MODEL_ITEM_FEATURE_COLS.items():
            num_embeddings = len(feature_encoders[feature_idx_col_name].classes_)
            emb = nn.Embedding(num_embeddings, config['embedding_dim_features'])
            item_feature_embeddings[feature_idx_col_name] = emb
            total_item_feature_dim += config['embedding_dim_features']
        self.item_feature_embeddings = nn.ModuleDict(item_feature_embeddings)
        self.item_mlp = nn.Sequential(
            nn.Linear(total_item_feature_dim, total_item_feature_dim * 2), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(total_item_feature_dim * 2, config['final_mlp_embed_dim'])
        )
        self._init_weights()

    def _init_weights(self):
        for module_list in [self.user_id_emb, self.item_id_emb] + \
                           list(self.user_feature_embeddings.values()) + \
                           list(self.item_feature_embeddings.values()):
            if isinstance(module_list, nn.Embedding): # Check it's an embedding layer
                 nn.init.xavier_uniform_(module_list.weight)
        for mlp in [self.user_mlp, self.item_mlp]:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None: nn.init.zeros_(layer.bias)

    def _get_representation(self, features_tensor_batch, id_emb_layer, feature_embeddings_dict, model_feature_cols_map, mlp):
        id_emb = id_emb_layer(features_tensor_batch[:, 0])
        embs = [id_emb]
        for i, idx_col_name in enumerate(model_feature_cols_map.values()):
            embs.append(feature_embeddings_dict[idx_col_name](features_tensor_batch[:, i+1]))
        return mlp(torch.cat(embs, dim=1))

    def get_user_representation(self, user_features_tensor_batch):
        return self._get_representation(user_features_tensor_batch, self.user_id_emb, self.user_feature_embeddings, MODEL_USER_FEATURE_COLS, self.user_mlp)

    def get_item_representation(self, item_features_tensor_batch):
        return self._get_representation(item_features_tensor_batch, self.item_id_emb, self.item_feature_embeddings, MODEL_ITEM_FEATURE_COLS, self.item_mlp)

    def forward(self, user_features_batch, item_features_batch):
        return (self.get_user_representation(user_features_batch) * self.get_item_representation(item_features_batch)).sum(dim=1)

class BPRLoss(nn.Module):
    def __init__(self, epsilon=1e-9): super().__init__(); self.epsilon = epsilon
    def forward(self, pos_scores, neg_scores): return -torch.log(torch.sigmoid(pos_scores - neg_scores) + self.epsilon).mean()

def train_model_adapted(model, train_loader, 
                        user_features_df_eval, item_features_df_eval, 
                        val_ground_truth_map, optimizer, criterion, config, device):
    best_val_metric, epochs_no_improve = -1.0, 0
    best_model_state_dict = None

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False):
            optimizer.zero_grad()
            user_features = batch_data['user_features'].to(device)
            loss = None # Initialize loss
            if config['loss_type'] == "BPR":
                pos_item_features, neg_item_features = batch_data['pos_item_features'].to(device), batch_data['neg_item_features'].to(device)
                user_repr, pos_item_repr, neg_item_repr = model.get_user_representation(user_features), model.get_item_representation(pos_item_features), model.get_item_representation(neg_item_features)
                pos_scores, neg_scores = (user_repr * pos_item_repr).sum(dim=1), (user_repr * neg_item_repr).sum(dim=1)
                loss = criterion(pos_scores, neg_scores)
            elif config['loss_type'] == "BCE":
                item_features, labels = batch_data['item_features'].to(device), batch_data['label'].to(device).float()
                logits = model(user_features, item_features)
                loss = criterion(logits, labels)
            if loss is not None: # Check if loss was computed
                loss.backward(); optimizer.step(); total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f}")

        if val_ground_truth_map and len(val_ground_truth_map) > 0:
            val_metrics = evaluate_model(model, list(val_ground_truth_map.keys()), val_ground_truth_map,
                                        user_features_df_eval, item_features_df_eval, config['top_k'], "Validating")
            current_val_metric = val_metrics["ndcg"]
            print(f"Epoch {epoch+1} Val - Recall@{config['top_k']}: {val_metrics['recall']:.4f}, "
                  f"Prec@{config['top_k']}: {val_metrics['precision']:.4f}, NDCG@{config['top_k']}: {current_val_metric:.4f}")
            if current_val_metric > best_val_metric:
                best_val_metric, epochs_no_improve, best_model_state_dict = current_val_metric, 0, copy.deepcopy(model.state_dict())
                print(f"New best val NDCG: {best_val_metric:.4f}.")
            else:
                epochs_no_improve += 1; print(f"Val NDCG no improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= config['early_stopping_patience']: print(f"Early stopping."); break
        else:
            if epoch == config['epochs'] - 1 and not best_model_state_dict: best_model_state_dict = copy.deepcopy(model.state_dict())

    if best_model_state_dict: model.load_state_dict(best_model_state_dict)
    return model

def evaluate_model(eval_model, user_indices_to_eval, ground_truth_map, 
                   user_features_df, item_features_df, top_k_val, desc="Evaluating"):
    eval_model.eval()
    metrics = {"recall": [], "precision": [], "ndcg": []}
    if not user_indices_to_eval or not ground_truth_map: return {k: 0.0 for k in metrics}
    
    valid_users = [u for u in user_indices_to_eval if u in user_features_df.index]
    if not valid_users: return {k: 0.0 for k in metrics}

    all_item_ids_sorted = sorted(item_features_df.index.tolist())
    if not all_item_ids_sorted: return {k: 0.0 for k in metrics}

    item_tensors = torch.tensor([[item_id] + [item_features_df.loc[item_id, col] for col in MODEL_ITEM_FEATURE_COLS.values()] 
                                 for item_id in all_item_ids_sorted], dtype=torch.long, device=device)
    map_sorted_idx_to_item_id = {i: item_id for i, item_id in enumerate(all_item_ids_sorted)}

    with torch.no_grad():
        all_item_repr = eval_model.get_item_representation(item_tensors)
        user_tensors = torch.tensor([[user_id] + [user_features_df.loc[user_id, col] for col in MODEL_USER_FEATURE_COLS.values()]
                                     for user_id in valid_users], dtype=torch.long, device=device)
        user_reprs = eval_model.get_user_representation(user_tensors)
        batch_scores = torch.matmul(user_reprs, all_item_repr.T)

        for i, user_id in enumerate(tqdm(valid_users, desc=desc, leave=False)):
            user_scores = batch_scores[i]
            actual_pos = ground_truth_map.get(user_id, set())
            if not actual_pos: [metrics[k].append(0.0) for k in metrics]; continue

            _, topk_indices = torch.topk(user_scores, k=min(top_k_val, len(user_scores)))
            recs = {map_sorted_idx_to_item_id[idx.item()] for idx in topk_indices}
            
            hits = len(recs.intersection(actual_pos))
            metrics["recall"].append(hits / len(actual_pos))
            metrics["precision"].append(hits / top_k_val)

            target_rel = torch.zeros_like(user_scores, dtype=torch.bool, device=device)
            for pos_id in actual_pos:
                try: target_rel[all_item_ids_sorted.index(pos_id)] = True
                except ValueError: pass
            metrics["ndcg"].append(tm_functional.retrieval_normalized_dcg(user_scores.float(), target_rel, top_k=top_k_val).item())
    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}

def calculate_popularity_baseline(train_df_positive, user_indices_to_eval, ground_truth_map,
                                  all_item_indices_sorted, top_k_val, desc="Popularity Baseline"):
    if train_df_positive.empty: return {"recall":0.0, "precision":0.0, "ndcg":0.0}
    item_pop = train_df_positive[ITEM_ID_COL].value_counts()
    top_k_popular_set = set(item_pop.nlargest(top_k_val).index.tolist())
    metrics = {"recall": [], "precision": [], "ndcg": []}

    if not user_indices_to_eval or not ground_truth_map or not all_item_indices_sorted: return {k:0.0 for k in metrics}
    map_item_id_to_sorted_idx = {item_id: i for i, item_id in enumerate(all_item_indices_sorted)}
    num_items = len(all_item_indices_sorted)
    pop_scores = torch.zeros(num_items, device=device, dtype=torch.float)
    for rank, item_id in enumerate(item_pop.index):
        if item_id in map_item_id_to_sorted_idx: pop_scores[map_item_id_to_sorted_idx[item_id]] = float(num_items - rank)

    for user_idx in tqdm(user_indices_to_eval, desc=desc, leave=False):
        actual_pos = ground_truth_map.get(user_idx, set())
        if not actual_pos: [metrics[k].append(0.0) for k in metrics]; continue
        hits = len(top_k_popular_set.intersection(actual_pos))
        metrics["recall"].append(hits / len(actual_pos)); metrics["precision"].append(hits / top_k_val)
        target_rel = torch.zeros(num_items, dtype=torch.bool, device=device)
        for pos_id in actual_pos:
            if pos_id in map_item_id_to_sorted_idx: target_rel[map_item_id_to_sorted_idx[pos_id]] = True
        metrics["ndcg"].append(tm_functional.retrieval_normalized_dcg(pop_scores, target_rel, top_k=top_k_val).item())
    return {k: np.mean(v) if v else 0.0 for k,v in metrics.items()}

def get_recommendations_for_user_conceptual(
    user_id_encoded, model, user_features_df, item_features_df,
    cold_user_ids_set_encoded, popular_items_ranked_list_encoded, config, device):
    if user_id_encoded in cold_user_ids_set_encoded: return popular_items_ranked_list_encoded[:config['top_k']]
    model.eval()
    with torch.no_grad():
        try:
            user_row = user_features_df.loc[user_id_encoded]
            user_tensor_parts = [user_id_encoded] + [user_row[col] for col in MODEL_USER_FEATURE_COLS.values()]
            user_feats_tensor = torch.tensor([user_tensor_parts], dtype=torch.long).to(device)
        except KeyError: return popular_items_ranked_list_encoded[:config['top_k']] # Fallback for user not in features_df

        user_repr = model.get_user_representation(user_feats_tensor)
        all_item_ids_sorted = sorted(item_features_df.index.tolist())
        if not all_item_ids_sorted: return []
        
        item_tensors = torch.tensor([[item_id] + [item_features_df.loc[item_id, col] for col in MODEL_ITEM_FEATURE_COLS.values()]
                                     for item_id in all_item_ids_sorted], dtype=torch.long, device=device)
        all_item_repr = model.get_item_representation(item_tensors)
        scores = torch.matmul(user_repr, all_item_repr.T).squeeze(0)
        _, top_k_indices = torch.topk(scores, k=config['top_k'])
        return [all_item_ids_sorted[idx.item()] for idx in top_k_indices]
        
def main():
    torch.manual_seed(CONFIG['random_state']); np.random.seed(CONFIG['random_state'])
    (df_pos, encoders, user_features_df, item_features_df, user_pos_map, _, cold_users, popular_items) = load_and_preprocess_data()

    if df_pos.empty or user_features_df.empty or item_features_df.empty:
        print("Critical dataframes empty after preprocessing. Exiting."); return
    all_items_global = item_features_df.index.tolist()
    if not all_items_global: print("No items available. Exiting."); return
        
    unique_users = df_pos[USER_ID_COL].unique()
    if len(unique_users) < 3: train_users, val_users, test_users = unique_users, np.array([]), np.array([])
    else:
        train_val_users, test_users = train_test_split(unique_users, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])
        if len(train_val_users) < 2 or CONFIG['val_size'] == 0.0: train_users, val_users = train_val_users, np.array([])
        else:
            eff_val_size = min(CONFIG['val_size'] / (1.0 - CONFIG['test_size']) if (1.0 - CONFIG['test_size']) > 0 else CONFIG['val_size'], 0.9)
            train_users, val_users = train_test_split(train_val_users, test_size=eff_val_size if eff_val_size > 0 else 0, random_state=CONFIG['random_state'])

    train_df, val_df, test_df = (df_pos[df_pos[USER_ID_COL].isin(ids)] for ids in [train_users, val_users, test_users])
    print(f"Train N: {len(train_df)} ({len(train_users)} users), Val N: {len(val_df)} ({len(val_users)} users), Test N: {len(test_df)} ({len(test_users)} users)")
    if train_df.empty: print("No training data. Exiting."); return

    train_dataset = RecommenderDataset(train_df, user_pos_map, {}, all_items_global, user_features_df, item_features_df, CONFIG['loss_type'], CONFIG['num_neg_samples'])
    if len(train_dataset) == 0: print("Train dataset empty. Exiting."); return
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    
    val_gt_map = val_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict() if not val_df.empty and len(val_users) > 0 else None
    model = TwoTowerModelWithFeatures(encoders, CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = BPRLoss().to(device) if CONFIG['loss_type'] == "BPR" else nn.BCEWithLogitsLoss().to(device)

    print("\n--- Model Training ---")
    trained_model = train_model_adapted(model, train_loader, user_features_df, item_features_df, val_gt_map, optimizer, criterion, CONFIG, device)
    
    test_gt_map = test_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict() if not test_df.empty and len(test_users) > 0 else None
    
    if test_gt_map:
        print("\n--- Popularity Baseline (Test) ---")
        pop_metrics = calculate_popularity_baseline(train_df, list(test_gt_map.keys()), test_gt_map, sorted(all_items_global), CONFIG['top_k'])
        print(f"Recall@{CONFIG['top_k']}: {pop_metrics['recall']:.4f}, Precision@{CONFIG['top_k']}: {pop_metrics['precision']:.4f}, NDCG@{CONFIG['top_k']}: {pop_metrics['ndcg']:.4f}")
        print("\n--- Trained Model (Test) ---")
        test_metrics = evaluate_model(trained_model, list(test_gt_map.keys()), test_gt_map, user_features_df, item_features_df, CONFIG['top_k'], "Testing")
        print(f"Recall@{CONFIG['top_k']}: {test_metrics['recall']:.4f}, Precision@{CONFIG['top_k']}: {test_metrics['precision']:.4f}, NDCG@{CONFIG['top_k']}: {test_metrics['ndcg']:.4f}")
    else: print("Test set empty, skipping final evaluations.")
    
    print("\n--- Example Recommendations ---")
    if len(unique_users) > 0:
        example_user_id = unique_users[0]
        recs_enc = get_recommendations_for_user_conceptual(example_user_id, trained_model, user_features_df, item_features_df, cold_users, popular_items, CONFIG, device)
        if encoders.get(ITEM_ID_COL) and recs_enc: # Check if decoder and recs exist
             readable_recs = encoders[ITEM_ID_COL].inverse_transform(recs_enc)
             orig_user_id = encoders[USER_ID_COL].inverse_transform([example_user_id])[0] if encoders.get(USER_ID_COL) else example_user_id
             print(f"Top {CONFIG['top_k']} recs for user {orig_user_id} (enc: {example_user_id}): {readable_recs[:CONFIG['top_k']]}")
    print("\nDone.")

if __name__ == "__main__":
    main()