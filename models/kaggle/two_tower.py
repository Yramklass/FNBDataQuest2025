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
    "val_size": 0.15, # Proportion of the total dataset
    "embedding_dim_ids": 64,
    "embedding_dim_features": 32, # Embedding dim for Country, Description, Price Bins
    "final_mlp_embed_dim": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 30,
    "batch_size": 2048,
    "num_neg_samples": 4, # Used if loss_type is BCE
    "top_k": 10,
    "random_state": 42,
    "loss_type": "BPR", # BPR or BCE
    "early_stopping_patience": 5,
    "price_bins": [0, 5, 10, 20, 50, 100, 200, np.inf], # Define price bins
    "use_popularity_negative_sampling": True, # New flag
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Column Name Definitions
USER_ID_COL = 'user_idx'
ITEM_ID_COL = 'item_idx'
DATE_COL = 'InvoiceDate' # For temporal split

RAW_USER_ID_COL_NAME = 'CustomerID'
RAW_ITEM_ID_COL_NAME = 'StockCode'
RAW_PRICE_COL_NAME = 'UnitPrice'

RAW_USER_CATEGORICAL_COLS = {'Country': 'country_idx'}
RAW_ITEM_CATEGORICAL_COLS = {
    'Description': 'description_idx',
    'UnitPriceBin': 'price_bin_idx'
}

MODEL_USER_FEATURE_COLS = RAW_USER_CATEGORICAL_COLS.copy()
MODEL_ITEM_FEATURE_COLS = RAW_ITEM_CATEGORICAL_COLS.copy()


def load_and_preprocess_data():
    print(f"Using device: {device}")
    try:
        df_raw = pd.read_csv("../../data/kaggle_dataset/raw/data.csv", encoding='ISO-8859-1')
    except FileNotFoundError:
        print("File not found. Please check the path: ../../data/kaggle_dataset/raw/data.csv")
        return None, None, None, None, None, None, None, None

    print(f"Initial raw data rows: {len(df_raw)}")
    df_raw.dropna(subset=[RAW_USER_ID_COL_NAME, RAW_ITEM_ID_COL_NAME, RAW_PRICE_COL_NAME, DATE_COL], inplace=True)
    df_raw = df_raw[(df_raw['Quantity'] > 0) & (df_raw[RAW_PRICE_COL_NAME] > 0)]
    print(f"Rows after cleaning (NA, Quantity>0, UnitPrice>0): {len(df_raw)}")

    if df_raw.empty: raise ValueError("DataFrame empty after cleaning. Cannot proceed.")

    df_raw[RAW_USER_ID_COL_NAME] = df_raw[RAW_USER_ID_COL_NAME].astype(float).astype(int).astype(str)
    df_raw[RAW_ITEM_ID_COL_NAME] = df_raw[RAW_ITEM_ID_COL_NAME].astype(str)
    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL]) # Parse date column

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

    price_bins = CONFIG["price_bins"]
    price_labels = [f"price_bin_{i}" for i in range(len(price_bins)-1)]
    intermediate_binned_price_col = 'binned_price_values_temp'
    df_raw[intermediate_binned_price_col] = pd.cut(
        df_raw[RAW_PRICE_COL_NAME], bins=price_bins, labels=price_labels, right=False, include_lowest=True
    )

    for conceptual_feature_name, target_idx_col_name in RAW_ITEM_CATEGORICAL_COLS.items():
        source_series_for_encoding = None
        if conceptual_feature_name == 'UnitPriceBin':
            source_series_for_encoding = df_raw[intermediate_binned_price_col]
        elif conceptual_feature_name == 'Description':
            source_series_for_encoding = df_raw['Description']
        if source_series_for_encoding is None:
            print(f"Warning: Source series for item feature '{conceptual_feature_name}' not defined. Skipping.")
            continue
        processed_series = source_series_for_encoding.astype(str).fillna("UNKNOWN").replace('nan', "UNKNOWN")
        le = LabelEncoder()
        df_raw[target_idx_col_name] = le.fit_transform(processed_series)
        feature_encoders[target_idx_col_name] = le

    if intermediate_binned_price_col in df_raw.columns:
        df_raw.drop(columns=[intermediate_binned_price_col], inplace=True)

    user_feat_cols = list(MODEL_USER_FEATURE_COLS.values())
    user_features_df = df_raw[[USER_ID_COL] + user_feat_cols].drop_duplicates(subset=[USER_ID_COL]).set_index(USER_ID_COL).sort_index()

    item_feat_cols = list(MODEL_ITEM_FEATURE_COLS.values())
    item_features_df = df_raw[[ITEM_ID_COL] + item_feat_cols].drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL).sort_index()

    # df_positive_interactions will be sorted by date in main() before splitting
    df_positive_interactions = df_raw.copy()

    # This map will be rebuilt based on train_df later for training dataset
    # For now, this is based on all interactions
    user_pos_items_map_all_data = df_positive_interactions.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()
    user_to_hard_negatives_map = {} # Not used currently

    all_users_enc = set(df_raw[USER_ID_COL].unique())
    pos_interaction_users_enc = set(df_positive_interactions[USER_ID_COL].unique())
    cold_user_ids_set_encoded = all_users_enc - pos_interaction_users_enc # Users with no interactions in the entire dataset
    print(f"Identified {len(cold_user_ids_set_encoded)} users with no interactions in the entire dataset (encoded).")

    # This will be recalculated based on train_df for cold user fallback recommendations
    # For now, this is based on all interactions
    item_popularity_all_data = df_positive_interactions[ITEM_ID_COL].value_counts()
    popular_items_ranked_list_encoded_all_data = item_popularity_all_data.index.tolist()

    return (df_positive_interactions, feature_encoders, user_features_df, item_features_df,
            user_pos_items_map_all_data, user_to_hard_negatives_map,
            cold_user_ids_set_encoded, popular_items_ranked_list_encoded_all_data)


class RecommenderDataset(Dataset):
    def __init__(self, positive_interactions_df, user_pos_items_map,
                 user_displayed_items_map, all_item_indices_list,
                 user_features_df, item_features_df,
                 loss_type="BCE", num_neg_samples=4,
                 item_popularities=None): # item_popularities is a pd.Series: index=item_idx, values=probability

        self.positive_interactions_df = positive_interactions_df[[USER_ID_COL, ITEM_ID_COL]].drop_duplicates()
        self.user_pos_items_map = user_pos_items_map # Should be based on training interactions only
        self.user_displayed_items_map = user_displayed_items_map # Currently empty
        self.all_item_indices_list = all_item_indices_list # Full list of item_idx
        self.user_features_df = user_features_df
        self.item_features_df = item_features_df
        self.loss_type = loss_type
        self.num_neg_samples = num_neg_samples

        if not self.all_item_indices_list: raise ValueError("all_item_indices_list is empty.")

        self.item_popularities = item_popularities
        if self.item_popularities is not None and not self.item_popularities.empty:
            self.sampling_item_indices = self.item_popularities.index.tolist()
            self.sampling_item_probs = self.item_popularities.values
            # Normalize probabilities if they don't sum to 1 (np.random.choice requires this)
            if not np.isclose(np.sum(self.sampling_item_probs), 1.0):
                 self.sampling_item_probs = self.sampling_item_probs / np.sum(self.sampling_item_probs)
        else: # Fallback to uniform random sampling over all items
            self.sampling_item_indices = self.all_item_indices_list
            self.sampling_item_probs = None # np.random.choice samples uniformly if p is None
        
        self.training_samples = self._create_training_samples()

    def _sample_negative(self, user_idx, positive_item_idx):
        user_pos_set = self.user_pos_items_map.get(user_idx, set())
        # user_disp_set logic remains unused as user_displayed_items_map is empty
        # user_disp_set = self.user_displayed_items_map.get(user_idx, set())
        # possible_informed_negs = list(user_disp_set - user_pos_set - {positive_item_idx})
        # if possible_informed_negs: return np.random.choice(possible_informed_negs)

        attempts = 0
        # Max attempts can be bounded, e.g. by the number of items available for sampling
        # or a fixed reasonably large number to avoid very long loops if user_pos_set is large.
        max_attempts = len(self.sampling_item_indices) if self.sampling_item_indices else len(self.all_item_indices_list)
        max_attempts = min(max_attempts, 2 * len(self.all_item_indices_list)) # Cap attempts

        if self.sampling_item_probs is not None: # Popularity sampling
            while attempts < max_attempts :
                neg_item_idx = np.random.choice(
                    self.sampling_item_indices,
                    p=self.sampling_item_probs
                )
                if neg_item_idx not in user_pos_set and neg_item_idx != positive_item_idx:
                    return neg_item_idx
                attempts += 1
        
        # Fallback to uniform random sampling if popularity sampling fails or not enabled
        attempts = 0 # Reset attempts for uniform fallback
        while attempts < max_attempts: # Use all_item_indices_list for uniform fallback
            neg_item_idx = np.random.choice(self.all_item_indices_list)
            if neg_item_idx not in user_pos_set and neg_item_idx != positive_item_idx:
                return neg_item_idx
            attempts +=1
        
        # Absolute fallback: iterate through all items (should ideally not be reached)
        # This can be slow if all_item_indices_list is very large
        print(f"Warning: Reached aggressive fallback in negative sampling for user {user_idx}. This might be slow.")
        permuated_items = np.random.permutation(self.all_item_indices_list)
        for neg_item_idx in permuated_items:
            if neg_item_idx not in user_pos_set and neg_item_idx != positive_item_idx:
                return neg_item_idx
        raise RuntimeError(f"Could not find a negative sample for user {user_idx} after extensive search.")


    def _get_features_tensor(self, entity_idx, features_df, model_feature_cols_map, entity_type="User"):
        try:
            features_row = features_df.loc[entity_idx]
        except KeyError:
            num_other_features = len(model_feature_cols_map.values())
            # Return 0 for ID and 0 for all feature indices (UNKNOWN if UNKNOWN was encoded as 0)
            # This assumes that label encoders will map UNKNOWN or a placeholder to index 0.
            # Or, ensure feature_encoders have an "UNKNOWN" category mapped to 0.
            # For safety, it's better if every user/item has an entry in features_df, even with default/UNKNOWN values.
            # For now, using 0s, which might map to the first category for each feature.
            return torch.tensor([entity_idx] + [0] * num_other_features, dtype=torch.long)

        feature_values = [features_row[col_idx_name] for col_idx_name in model_feature_cols_map.values()]
        return torch.tensor([entity_idx] + feature_values, dtype=torch.long)

    def _get_user_features_tensor(self, user_idx):
        return self._get_features_tensor(user_idx, self.user_features_df, MODEL_USER_FEATURE_COLS, "User")

    def _get_item_features_tensor(self, item_idx):
        return self._get_features_tensor(item_idx, self.item_features_df, MODEL_ITEM_FEATURE_COLS, "Item")

    def _create_training_samples(self):
        samples = []
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

        self.user_id_emb = nn.Embedding(len(feature_encoders[USER_ID_COL].classes_), config['embedding_dim_ids'])
        user_feature_embeddings = {}
        total_user_feature_dim = config['embedding_dim_ids']
        for _, feature_idx_col_name in MODEL_USER_FEATURE_COLS.items():
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
        module_list = [self.user_id_emb, self.item_id_emb] + \
                      list(self.user_feature_embeddings.values()) + \
                      list(self.item_feature_embeddings.values())
        for emb_layer in module_list: # Iterate directly over layers
            if isinstance(emb_layer, nn.Embedding): # Check it's an embedding layer
                 nn.init.xavier_uniform_(emb_layer.weight)

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

    def forward(self, user_features_batch, item_features_batch): # For BCE
        return (self.get_user_representation(user_features_batch) * self.get_item_representation(item_features_batch)).sum(dim=1)

class BPRLoss(nn.Module):
    def __init__(self, epsilon=1e-9): super().__init__(); self.epsilon = epsilon
    def forward(self, pos_scores, neg_scores): return -torch.log(torch.sigmoid(pos_scores - neg_scores) + self.epsilon).mean()

def train_model_adapted(model, train_loader,
                        user_features_df_eval, item_features_df_eval,
                        val_ground_truth_map, # Dict: user_idx -> set of item_idx for validation
                        optimizer, criterion, config, device):
    best_val_metric, epochs_no_improve = -1.0, 0
    best_model_state_dict = None

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False):
            optimizer.zero_grad()
            user_features = batch_data['user_features'].to(device)
            loss = None
            if config['loss_type'] == "BPR":
                pos_item_features = batch_data['pos_item_features'].to(device)
                neg_item_features = batch_data['neg_item_features'].to(device)
                user_repr = model.get_user_representation(user_features)
                pos_item_repr = model.get_item_representation(pos_item_features)
                neg_item_repr = model.get_item_representation(neg_item_features)
                pos_scores = (user_repr * pos_item_repr).sum(dim=1)
                neg_scores = (user_repr * neg_item_repr).sum(dim=1)
                loss = criterion(pos_scores, neg_scores)
            elif config['loss_type'] == "BCE":
                item_features = batch_data['item_features'].to(device)
                labels = batch_data['label'].to(device).float()
                logits = model(user_features, item_features) # Uses the forward method
                loss = criterion(logits, labels)

            if loss is not None:
                loss.backward(); optimizer.step(); total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f}")

        if val_ground_truth_map and len(val_ground_truth_map) > 0:
            # Ensure users for validation exist in user_features_df_eval
            val_users_to_eval = [u for u in val_ground_truth_map.keys() if u in user_features_df_eval.index]
            if val_users_to_eval:
                val_metrics = evaluate_model(model, val_users_to_eval, val_ground_truth_map,
                                             user_features_df_eval, item_features_df_eval, config['top_k'], "Validating")
                current_val_metric = val_metrics["ndcg"]
                print(f"Epoch {epoch+1} Val - Recall@{config['top_k']}: {val_metrics['recall']:.4f}, "
                      f"Prec@{config['top_k']}: {val_metrics['precision']:.4f}, NDCG@{config['top_k']}: {current_val_metric:.4f}")
                if current_val_metric > best_val_metric:
                    best_val_metric, epochs_no_improve = current_val_metric, 0
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    print(f"New best val NDCG: {best_val_metric:.4f}.")
                else:
                    epochs_no_improve += 1
                    print(f"Val NDCG no improvement for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}."); break
            else:
                print(f"Epoch {epoch+1}: No validation users found in user_features_df for evaluation.")
        else: # No validation set, save last model or handle as per requirement
            if epoch == config['epochs'] - 1 and not best_model_state_dict: # Save last model if no validation
                 best_model_state_dict = copy.deepcopy(model.state_dict())


    if best_model_state_dict: model.load_state_dict(best_model_state_dict)
    return model

def evaluate_model(eval_model, user_indices_to_eval, ground_truth_map,
                   user_features_df, item_features_df, top_k_val, desc="Evaluating"):
    eval_model.eval()
    metrics = {"recall": [], "precision": [], "ndcg": []}

    if not user_indices_to_eval or not ground_truth_map: return {k: 0.0 for k in metrics}

    # Filter user_indices_to_eval to only include those present in user_features_df
    valid_users_for_eval = [u for u in user_indices_to_eval if u in user_features_df.index]
    if not valid_users_for_eval:
        print(f"{desc}: No valid users to evaluate after filtering against user_features_df.")
        return {k: 0.0 for k in metrics}

    all_item_ids_sorted = sorted(item_features_df.index.tolist())
    if not all_item_ids_sorted: return {k: 0.0 for k in metrics}

    # Prepare item tensors once
    item_tensors_list = []
    for item_id in all_item_ids_sorted:
        try:
            item_row = item_features_df.loc[item_id]
            item_tensor_parts = [item_id] + [item_row[col] for col in MODEL_ITEM_FEATURE_COLS.values()]
            item_tensors_list.append(item_tensor_parts)
        except KeyError: # Should not happen if item_features_df is comprehensive
            print(f"Warning: Item ID {item_id} not found in item_features_df during evaluation tensor creation. Skipping this item.")
            # Need to adjust all_item_ids_sorted and map_sorted_idx_to_item_id if we skip items.
            # For simplicity, this example assumes all items in all_item_ids_sorted are in item_features_df.
            # A more robust way would be to filter all_item_ids_sorted first.
            continue # This item will be skipped; this could affect metric calculation if it's a positive item.
    
    if not item_tensors_list:
        print(f"{desc}: No valid items to create item tensors.")
        return {k: 0.0 for k in metrics}
        
    item_tensors = torch.tensor(item_tensors_list, dtype=torch.long, device=device)
    map_sorted_idx_to_item_id = {i: item_id for i, item_id in enumerate(all_item_ids_sorted) if item_id in item_features_df.index} # Ensure mapping is correct

    with torch.no_grad():
        all_item_repr = eval_model.get_item_representation(item_tensors)

        # Prepare user tensors
        user_tensors_list = []
        final_valid_users_for_batch = [] # Users for whom we successfully create tensors
        for user_id in valid_users_for_eval:
            try:
                user_row = user_features_df.loc[user_id]
                user_tensor_parts = [user_id] + [user_row[col] for col in MODEL_USER_FEATURE_COLS.values()]
                user_tensors_list.append(user_tensor_parts)
                final_valid_users_for_batch.append(user_id)
            except KeyError:
                print(f"Warning: User ID {user_id} not found in user_features_df during user tensor creation for batch. Skipping this user.")
        
        if not user_tensors_list:
            print(f"{desc}: No valid user tensors created for batch evaluation.")
            return {k: 0.0 for k in metrics}

        user_tensors_batch = torch.tensor(user_tensors_list, dtype=torch.long, device=device)
        user_reprs_batch = eval_model.get_user_representation(user_tensors_batch)
        batch_scores = torch.matmul(user_reprs_batch, all_item_repr.T)

        for i, user_id in enumerate(tqdm(final_valid_users_for_batch, desc=desc, leave=False)):
            user_scores = batch_scores[i] # Scores for this user against all items
            actual_pos = ground_truth_map.get(user_id, set())

            if not actual_pos:
                [metrics[k].append(0.0) for k in metrics]; continue

            # Get top K recommendations based on scores
            # Ensure k is not larger than the number of available scores
            current_top_k = min(top_k_val, len(user_scores))
            if current_top_k == 0: # No scores or items to recommend from
                 [metrics[k].append(0.0) for k in metrics]; continue

            _, topk_indices_sorted_list = torch.topk(user_scores, k=current_top_k)
            
            recs = set()
            for sorted_idx in topk_indices_sorted_list.cpu().numpy():
                if sorted_idx in map_sorted_idx_to_item_id:
                     recs.add(map_sorted_idx_to_item_id[sorted_idx])
            
            hits = len(recs.intersection(actual_pos))
            metrics["recall"].append(hits / len(actual_pos) if len(actual_pos) > 0 else 0.0)
            metrics["precision"].append(hits / top_k_val if top_k_val > 0 else 0.0) # Denominator is intended top_k

            # For NDCG: Create target relevance tensor based on actual positive items
            target_rel = torch.zeros_like(user_scores, dtype=torch.bool, device=device)
            found_pos_for_ndcg = 0
            for pos_item_id in actual_pos:
                try:
                    # Find index of pos_item_id in the original sorted list used for all_item_repr
                    # This needs all_item_ids_sorted to be the list of items whose representations are in all_item_repr
                    idx_in_sorted_list = all_item_ids_sorted.index(pos_item_id)
                    target_rel[idx_in_sorted_list] = True
                    found_pos_for_ndcg +=1
                except ValueError: # Positive item not in the list of scored items (e.g. filtered out)
                    pass 
            
            if found_pos_for_ndcg > 0: # Only calculate NDCG if some true positives are in the scored item set
                ndcg_val = tm_functional.retrieval_normalized_dcg(user_scores.float(), target_rel.float(), top_k=top_k_val).item()
                metrics["ndcg"].append(ndcg_val)
            else: # If no positive items are found among the scored items, NDCG is 0
                metrics["ndcg"].append(0.0)


    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


def calculate_popularity_baseline(train_df_positive, user_indices_to_eval, ground_truth_map,
                                  all_item_indices_sorted, # All unique item IDs, sorted, from item_features_df
                                  top_k_val, desc="Popularity Baseline"):
    metrics = {"recall": [], "precision": [], "ndcg": []}
    if train_df_positive.empty: return {k:0.0 for k in metrics}
    if not user_indices_to_eval or not ground_truth_map or not all_item_indices_sorted:
        return {k:0.0 for k in metrics}

    item_pop_counts = train_df_positive[ITEM_ID_COL].value_counts()
    # Get top K popular items (encoded indices)
    top_k_popular_set = set(item_pop_counts.nlargest(top_k_val).index.tolist())

    # For NDCG calculation: assign scores based on popularity rank
    map_item_id_to_sorted_idx = {item_id: i for i, item_id in enumerate(all_item_indices_sorted)}
    num_items_total = len(all_item_indices_sorted)
    pop_scores_tensor = torch.zeros(num_items_total, device=device, dtype=torch.float)

    for rank, item_id in enumerate(item_pop_counts.index):
        if item_id in map_item_id_to_sorted_idx:
            # Higher score for more popular (lower rank)
            pop_scores_tensor[map_item_id_to_sorted_idx[item_id]] = float(num_items_total - rank)


    for user_idx in tqdm(user_indices_to_eval, desc=desc, leave=False):
        actual_pos = ground_truth_map.get(user_idx, set())
        if not actual_pos:
            [metrics[k].append(0.0) for k in metrics]; continue

        hits = len(top_k_popular_set.intersection(actual_pos))
        metrics["recall"].append(hits / len(actual_pos) if len(actual_pos) > 0 else 0.0)
        metrics["precision"].append(hits / top_k_val if top_k_val > 0 else 0.0)

        # For NDCG
        target_rel = torch.zeros(num_items_total, dtype=torch.bool, device=device)
        found_pos_for_ndcg = 0
        for pos_id in actual_pos:
            if pos_id in map_item_id_to_sorted_idx:
                target_rel[map_item_id_to_sorted_idx[pos_id]] = True
                found_pos_for_ndcg +=1
        
        if found_pos_for_ndcg > 0:
            ndcg_val = tm_functional.retrieval_normalized_dcg(pop_scores_tensor, target_rel.float(), top_k=top_k_val).item()
            metrics["ndcg"].append(ndcg_val)
        else:
            metrics["ndcg"].append(0.0)
            
    return {k: np.mean(v) if v else 0.0 for k,v in metrics.items()}


def get_recommendations_for_user_conceptual(
    user_id_encoded, model, user_features_df, item_features_df,
    # cold_user_ids_set_encoded: users with NO interactions in the training data
    # popular_items_ranked_list_encoded: popular items from training data
    cold_user_ids_set_encoded, popular_items_ranked_list_encoded, config, device):

    model.eval() # Ensure model is in evaluation mode

    # Check if user is cold based on presence in user_features_df (could be a proxy or explicit check)
    # Or if they are in the explicitly passed cold_user_ids_set_encoded
    is_cold_user = user_id_encoded in cold_user_ids_set_encoded or user_id_encoded not in user_features_df.index

    if is_cold_user:
        # print(f"User {user_id_encoded} is cold or not in user_features_df. Returning popular items.")
        return popular_items_ranked_list_encoded[:config['top_k']]

    with torch.no_grad():
        try:
            user_row = user_features_df.loc[user_id_encoded]
            user_tensor_parts = [user_id_encoded] + [user_row[col] for col in MODEL_USER_FEATURE_COLS.values()]
            user_feats_tensor = torch.tensor([user_tensor_parts], dtype=torch.long).to(device)
        except KeyError: # Should be caught by is_cold_user check if user_id_encoded not in user_features_df.index
            # print(f"User {user_id_encoded} not found in user_features_df. Fallback to popular items.")
            return popular_items_ranked_list_encoded[:config['top_k']]

        user_repr = model.get_user_representation(user_feats_tensor)
        
        all_item_ids_sorted = sorted(item_features_df.index.tolist())
        if not all_item_ids_sorted: return [] # No items to recommend

        # Prepare item tensors for all items
        item_tensors_list = []
        valid_item_ids_for_scoring = []
        for item_id in all_item_ids_sorted:
            try:
                item_row = item_features_df.loc[item_id]
                item_tensor_parts = [item_id] + [item_row[col] for col in MODEL_ITEM_FEATURE_COLS.values()]
                item_tensors_list.append(item_tensor_parts)
                valid_item_ids_for_scoring.append(item_id)
            except KeyError:
                # print(f"Item {item_id} not found in item_features_df during recommendation. Skipping.")
                pass # Skip items not in feature df
        
        if not item_tensors_list: return [] # No valid items to score

        item_tensors = torch.tensor(item_tensors_list, dtype=torch.long, device=device)
        all_item_repr = model.get_item_representation(item_tensors)

        scores = torch.matmul(user_repr, all_item_repr.T).squeeze(0)
        
        # Ensure top_k is not greater than the number of scored items
        current_top_k = min(config['top_k'], len(scores))
        if current_top_k == 0: return []

        _, top_k_indices = torch.topk(scores, k=current_top_k)
        
        # Map indices back to original item_ids that were scored
        return [valid_item_ids_for_scoring[idx.item()] for idx in top_k_indices]


def main():
    torch.manual_seed(CONFIG['random_state']); np.random.seed(CONFIG['random_state'])

    # Load data (df_pos is all positive interactions, not yet sorted or split)
    data_load_result = load_and_preprocess_data()
    if data_load_result[0] is None: return # Error during data loading

    (df_pos_full, encoders, user_features_df, item_features_df,
     _, _, cold_users_global, _) = data_load_result
    # user_pos_map_all_data, user_to_hard_negatives_map_global,
    # cold_users_global, popular_items_ranked_list_encoded_all_data are from full dataset.
    # We will re-derive some of these from train_df for specific purposes.

    if df_pos_full.empty or user_features_df.empty or item_features_df.empty:
        print("Critical dataframes empty after preprocessing. Exiting."); return

    all_items_global_encoded = item_features_df.index.tolist() # All unique encoded item IDs
    if not all_items_global_encoded: print("No items available globally. Exiting."); return

    # --- Temporal Data Splitting ---
    df_pos_full_sorted = df_pos_full.sort_values(DATE_COL).reset_index(drop=True)
    n_total_interactions = len(df_pos_full_sorted)

    # Calculate split indices based on proportions of the dataset
    val_split_idx = int(n_total_interactions * (1 - CONFIG['test_size'] - CONFIG['val_size']))
    test_split_idx = int(n_total_interactions * (1 - CONFIG['test_size']))

    train_df = df_pos_full_sorted.iloc[:val_split_idx]
    val_df = df_pos_full_sorted.iloc[val_split_idx:test_split_idx]
    test_df = df_pos_full_sorted.iloc[test_split_idx:]
    
    print(f"--- Data Split (Temporal) ---")
    print(f"Train interactions: {len(train_df)}")
    if not train_df.empty: print(f"  Train date range: {train_df[DATE_COL].min()} to {train_df[DATE_COL].max()}")
    print(f"Val interactions: {len(val_df)}")
    if not val_df.empty: print(f"  Val date range: {val_df[DATE_COL].min()} to {val_df[DATE_COL].max()}")
    print(f"Test interactions: {len(test_df)}")
    if not test_df.empty: print(f"  Test date range: {test_df[DATE_COL].min()} to {test_df[DATE_COL].max()}")

    if train_df.empty: print("No training data after temporal split. Exiting."); return

    # user_pos_map for training dataset (interactions from training period only)
    user_pos_map_train_only = train_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()

    # Item popularities for negative sampling (from training data)
    item_popularity_for_sampling = None
    if CONFIG['use_popularity_negative_sampling']:
        item_counts_train = train_df[ITEM_ID_COL].value_counts()
        if not item_counts_train.empty:
            item_popularity_for_sampling = item_counts_train / item_counts_train.sum() # Normalize to probabilities
            # Ensure all items in all_items_global_encoded have a probability, even if 0
            # for items not in train_df (though sampler will pick from item_popularity_for_sampling.index)
            # This should be fine as RecommenderDataset handles items from item_popularity_for_sampling.index
        else:
            print("Warning: Training data has no item interactions to calculate popularity for sampling. Using uniform.")

    train_dataset = RecommenderDataset(
        train_df, user_pos_map_train_only, {}, # {} for user_displayed_items_map (unused)
        all_items_global_encoded,
        user_features_df, item_features_df,
        CONFIG['loss_type'], CONFIG['num_neg_samples'],
        item_popularities=item_popularity_for_sampling
    )
    if len(train_dataset) == 0: print("Train dataset empty after sample creation. Exiting."); return
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))

    # Ground truth maps for evaluation (users and their items *within* val/test periods)
    val_gt_map = val_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict() if not val_df.empty else None
    test_gt_map = test_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict() if not test_df.empty else None

    model = TwoTowerModelWithFeatures(encoders, CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = BPRLoss().to(device) if CONFIG['loss_type'] == "BPR" else nn.BCEWithLogitsLoss().to(device)

    print("\n--- Model Training ---")
    trained_model = train_model_adapted(model, train_loader,
                                        user_features_df, item_features_df, # These are global feature DFs
                                        val_gt_map, # GT for users in validation period
                                        optimizer, criterion, CONFIG, device)

    # Popular items for cold user fallback, derived from training data
    train_item_counts = train_df[ITEM_ID_COL].value_counts()
    popular_items_for_fallback_encoded = train_item_counts.index.tolist() if not train_item_counts.empty else []
    
    # Define cold users as those in the entire dataset who have no interactions in the training set
    # This is for the get_recommendations_for_user_conceptual's cold start strategy.
    train_user_ids_set = set(train_df[USER_ID_COL].unique())
    all_user_ids_global_set = set(user_features_df.index.tolist()) # All users for whom we have features
    cold_users_for_fallback = all_user_ids_global_set - train_user_ids_set
    print(f"Identified {len(cold_users_for_fallback)} users as cold for recommendation fallback (not in train_df).")


    if test_gt_map and len(test_gt_map.keys()) > 0:
        test_users_in_period = list(test_gt_map.keys()) # Users who interacted in the test period
        print("\n--- Popularity Baseline (Test Period) ---")
        # Popularity baseline uses popular items from train_df to recommend for test_users_in_period
        pop_metrics = calculate_popularity_baseline(train_df, test_users_in_period, test_gt_map,
                                                    all_items_global_encoded, CONFIG['top_k'])
        print(f"Recall@{CONFIG['top_k']}: {pop_metrics['recall']:.4f}, Precision@{CONFIG['top_k']}: {pop_metrics['precision']:.4f}, NDCG@{CONFIG['top_k']}: {pop_metrics['ndcg']:.4f}")

        print("\n--- Trained Model (Test Period) ---")
        test_metrics = evaluate_model(trained_model, test_users_in_period, test_gt_map,
                                      user_features_df, item_features_df, CONFIG['top_k'], "Testing")
        print(f"Recall@{CONFIG['top_k']}: {test_metrics['recall']:.4f}, Precision@{CONFIG['top_k']}: {test_metrics['precision']:.4f}, NDCG@{CONFIG['top_k']}: {test_metrics['ndcg']:.4f}")
    else:
        print("Test set empty or no users in test_gt_map, skipping final evaluations on test data.")

    print("\n--- Example Recommendations ---")
    if not user_features_df.empty: # Pick an example user from available features
        # Try to pick a user who is NOT cold for a more interesting recommendation
        non_cold_example_users = list(train_user_ids_set.intersection(user_features_df.index))
        if non_cold_example_users:
            example_user_id_encoded = non_cold_example_users[0]
            print(f"Example user (non-cold): {example_user_id_encoded}")
        else: # Fallback to any user if no non-cold users found (e.g. very small dataset)
            example_user_id_encoded = user_features_df.index[0]
            print(f"Example user (might be cold): {example_user_id_encoded}")


        recs_enc = get_recommendations_for_user_conceptual(
            example_user_id_encoded, trained_model, user_features_df, item_features_df,
            cold_users_for_fallback, # Set of users considered cold for this purpose
            popular_items_for_fallback_encoded, # Popular items from training
            CONFIG, device)

        if encoders.get(ITEM_ID_COL) and recs_enc:
            readable_recs = encoders[ITEM_ID_COL].inverse_transform(recs_enc)
            orig_user_id_str = "N/A"
            if encoders.get(USER_ID_COL):
                 try:
                    orig_user_id_str = encoders[USER_ID_COL].inverse_transform([example_user_id_encoded])[0]
                 except ValueError:
                    orig_user_id_str = f"(Unknown original ID for encoded: {example_user_id_encoded})"
            else:
                orig_user_id_str = str(example_user_id_encoded)

            print(f"Top {CONFIG['top_k']} recs for user {orig_user_id_str} (encoded: {example_user_id_encoded}): {readable_recs[:CONFIG['top_k']]}")
        elif not recs_enc:
            print(f"No recommendations generated for user {example_user_id_encoded}.")
        else:
            print("ITEM_ID_COL encoder not found, cannot show readable recommendations.")
    else:
        print("No users in user_features_df to generate example recommendations for.")

    print("\nDone.")

if __name__ == "__main__":
    main()