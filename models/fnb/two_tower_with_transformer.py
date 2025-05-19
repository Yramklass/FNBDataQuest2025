import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchmetrics.functional as TMF
import copy


# Configuration
CONFIG = {
    "test_size": 0.15,
    "val_size": 0.15,
    "embedding_dim_ids": 32,
    "embedding_dim_features": 16,
    "transformer_nhead": 4,
    "transformer_nlayers": 2,
    "transformer_dim_feedforward": 64,
    "final_mlp_embed_dim": 64,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "epochs": 50, # Max epochs; early stopping will determine actual 
    "batch_size": 1024,
    "top_k": 10,
    "random_state": 42,
    "early_stopping_patience": 5
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Column name constants
USER_ID_COL_RAW = 'idcol'
ITEM_ID_COL_RAW = 'item'
USER_ID_COL = "user_idx"
ITEM_ID_COL = "item_idx"

RAW_USER_FEATURE_COLS = {"segment": "segment_idx", "beh_segment": "beh_segment_idx", "active_ind": "active_ind_idx"}
RAW_ITEM_FEATURE_COLS = {"item_type": "item_type_idx", "item_descrip": "item_descrip_idx"}

# Data Loading & Preprocessing
def load_and_preprocess_data():
    print(f"Using device: {device}, Config: {CONFIG}")
    try:
        # For testing, you might want to ensure this path is correct or use the dummy data by default.
        df_raw = pd.read_csv("../../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv")
        print("Successfully loaded real data.")
    except FileNotFoundError:
        print("RFile not found.")
        

    df_raw['int_date'] = pd.to_datetime(df_raw['int_date'], format='%d-%b-%y')
    df_raw['user_id_str'] = df_raw[USER_ID_COL_RAW].astype(str)
    df_raw['item_id_str'] = df_raw[ITEM_ID_COL_RAW].astype(str)

    feature_encoders = {}
    for col_str, new_col_idx in [('user_id_str', USER_ID_COL), ('item_id_str', ITEM_ID_COL)]:
        le = LabelEncoder()
        df_raw[new_col_idx] = le.fit_transform(df_raw[col_str])
        feature_encoders[new_col_idx] = le

    for raw_col_map in [RAW_USER_FEATURE_COLS, RAW_ITEM_FEATURE_COLS]:
        for raw_col, processed_idx_col_name in raw_col_map.items():
            df_raw[raw_col] = df_raw[raw_col].fillna("UNKNOWN").astype(str)
            le = LabelEncoder()
            df_raw[processed_idx_col_name] = le.fit_transform(df_raw[raw_col])
            feature_encoders[processed_idx_col_name] = le

    num_total_users = len(feature_encoders[USER_ID_COL].classes_)
    num_total_items = len(feature_encoders[ITEM_ID_COL].classes_)
    print(f"Total unique users: {num_total_users}, Total unique items: {num_total_items}")

    user_side_feature_idx_cols = list(RAW_USER_FEATURE_COLS.values())
    if USER_ID_COL in df_raw and num_total_users > 0 :
        user_features_df = df_raw[[USER_ID_COL] + user_side_feature_idx_cols].drop_duplicates(subset=[USER_ID_COL]).set_index(USER_ID_COL).sort_index()
    else:
        user_features_df = pd.DataFrame(columns=[USER_ID_COL] + user_side_feature_idx_cols).set_index(USER_ID_COL)


    item_side_feature_idx_cols = list(RAW_ITEM_FEATURE_COLS.values())
    if ITEM_ID_COL in df_raw and num_total_items > 0:
        item_features_df = df_raw[[ITEM_ID_COL] + item_side_feature_idx_cols].drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL).sort_index()
    else:
        item_features_df = pd.DataFrame(columns=[ITEM_ID_COL] + item_side_feature_idx_cols).set_index(ITEM_ID_COL)

    df_positive_interactions = df_raw[df_raw['interaction'].isin(['CLICK', 'CHECKOUT'])].copy()
    return df_positive_interactions, feature_encoders, user_features_df, item_features_df, num_total_users, num_total_items

# Dataset Class
class InteractionDataset(Dataset):
    def __init__(self, df, user_features_tensor, item_features_tensor):
        self.users = df[USER_ID_COL].values
        self.items = df[ITEM_ID_COL].values
        self.user_features_tensor = user_features_tensor
        self.item_features_tensor = item_features_tensor
        self.num_user_features = user_features_tensor.shape[1] if user_features_tensor.ndim > 1 else 0
        self.num_item_features = item_features_tensor.shape[1] if item_features_tensor.ndim > 1 else 0


    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        item_id = self.items[idx]
        
        user_side_feats = self.user_features_tensor[user_id] if user_id < self.user_features_tensor.size(0) else torch.zeros(self.num_user_features, dtype=torch.long)
        item_side_feats = self.item_features_tensor[item_id] if item_id < self.item_features_tensor.size(0) else torch.zeros(self.num_item_features, dtype=torch.long)
        return user_id, item_id, user_side_feats, item_side_feats

# Model Definition
class Tower(nn.Module):
    def __init__(self, id_dim, feature_cardinalities, embed_dim_id, embed_dim_feat,
                 transformer_nhead, transformer_nlayers, transformer_dim_feedforward, out_dim):
        super().__init__()
        self.id_embedding = nn.Embedding(id_dim, embed_dim_id) if id_dim > 0 else None # Handle id_dim=0
        
        self.feature_embeddings = nn.ModuleList()
        if sum(feature_cardinalities) > 0 : # only create if there are features with cardinalities > 0
             self.feature_embeddings = nn.ModuleList([
                nn.Embedding(max(1,cardinality), embed_dim_feat) for cardinality in feature_cardinalities # max(1,cardinality) handles 0
            ])
        
        self.has_features = len(self.feature_embeddings) > 0

        fc_input_dim = 0
        if self.id_embedding:
            fc_input_dim += embed_dim_id
        
        if self.has_features:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim_feat, nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward, dropout=0.1, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
            fc_input_dim += embed_dim_feat # Add dimension of aggregated features
        else:
            self.transformer = None
        
        if fc_input_dim == 0: # Should not happen with valid inputs
            print("Warning: fc_input_dim is 0 in Tower. Model might not learn.")
            self.fc = nn.Linear(1, out_dim) 
            fc_input_dim = 1 # for sequential
        else:
             self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, fc_input_dim * 2 if fc_input_dim > 0 else embed_dim_id*2 ), nn.ReLU(), # handle fc_input_dim=0 case for multiply
                nn.Dropout(0.3), nn.Linear(fc_input_dim * 2 if fc_input_dim > 0 else embed_dim_id*2, out_dim)
            )


    def forward(self, main_id, side_features_batch):
    
        outputs = []
        if self.id_embedding and main_id is not None:
            # Ensure main_id is not empty and all values are within [0, id_dim-1]
            if main_id.numel() > 0 and (main_id.max() < self.id_embedding.num_embeddings):
                 outputs.append(self.id_embedding(main_id))
            elif main_id.numel() > 0 : # Log if IDs are out of bounds
                 outputs.append(self.id_embedding(main_id.clamp(max=self.id_embedding.num_embeddings -1)))


        if self.has_features and self.transformer and side_features_batch is not None and side_features_batch.ndim > 1 and side_features_batch.size(1) > 0:
            feat_embs_list = []
            for i, embed_layer in enumerate(self.feature_embeddings):
                # Clamp indices to be safe, although LabelEncoder should ensure they are 0 to C-1
                clamped_indices = side_features_batch[:, i].clamp(max=embed_layer.num_embeddings - 1)
                feat_embs_list.append(embed_layer(clamped_indices))

            if feat_embs_list:
                feat_stack = torch.stack(feat_embs_list, dim=1)
                transformed_feats = self.transformer(feat_stack)
                outputs.append(transformed_feats.mean(dim=1))
        
        if not outputs: # If no embeddings were generated 
            batch_s = main_id.size(0) if main_id is not None and main_id.numel() > 0 else 1
            # Get out_dim from self.fc's last layer
            out_dimension = self.fc[-1].out_features if isinstance(self.fc[-1], nn.Linear) else CONFIG['final_mlp_embed_dim'] # Fallback
            return torch.zeros(batch_s, out_dimension, device=device)


        combined = torch.cat(outputs, dim=1)
        return self.fc(combined)

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower_config, item_tower_config):
        super().__init__()
        self.user_tower = Tower(**user_tower_config)
        self.item_tower = Tower(**item_tower_config)

    def forward(self, user_ids, item_ids, user_side_feats, item_side_feats):
        user_vec = self.user_tower(user_ids, user_side_feats)
        item_vec = self.item_tower(item_ids, item_side_feats)
        return user_vec, item_vec

# BPR Loss
def bpr_loss(pos_score, neg_score):
    return -torch.mean(F.logsigmoid(pos_score - neg_score))

# Training Loop
def train_model(model, train_loader, val_loader, optimizer, num_total_items,
                global_item_features_tensor_for_neg_sampling, # Tensor
                user_features_df_for_eval, item_features_df_for_eval, # DFs for eval function
                ground_truth_val_df): # DF of interactions for eval function
    best_val_ndcg = -1.0
    epochs_no_improve = 0
    best_model_state_dict = None

    if num_total_items == 0:
        print("Error: num_total_items is 0 in train_model. Cannot sample negative items.")
        return model

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for user_ids, pos_item_ids, user_side_feats, pos_item_side_feats in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            user_ids, pos_item_ids = user_ids.to(device), pos_item_ids.to(device)
            user_side_feats, pos_item_side_feats = user_side_feats.to(device), pos_item_side_feats.to(device)

            neg_item_ids = torch.randint(0, num_total_items, (user_ids.size(0),), device=device)
            neg_item_side_feats = global_item_features_tensor_for_neg_sampling[neg_item_ids]

            user_vec = model.user_tower(user_ids, user_side_feats)
            pos_item_vec = model.item_tower(pos_item_ids, pos_item_side_feats)
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

        if val_loader and ground_truth_val_df is not None and not ground_truth_val_df.empty:
            val_metrics = evaluate_model(model, val_loader, num_total_items,
                                         item_features_df_for_eval, # Pass the DF here
                                         user_features_df_for_eval, # Pass the DF here
                                         ground_truth_val_df, desc="Validating") # Pass the DF of interactions
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
        else:
             is_val_df_empty = ground_truth_val_df.empty if ground_truth_val_df is not None else "N/A (df is None)"
             print(f"Epoch {epoch+1} completed. No validation performed (val_loader: {val_loader is not None}, val_df empty: {is_val_df_empty}).")


    if best_model_state_dict:
        print("Loading best model weights.")
        model.load_state_dict(best_model_state_dict)
    elif not (val_loader and ground_truth_val_df is not None and not ground_truth_val_df.empty): # No validation was done
        print("No validation performed or val_df empty, using model from last training epoch.")
    else: # Validation was done but no improvement or only one epoch
        print("Using model from last training epoch (either no improvement or training ended).")
    return model

# Evaluation for PyTorch Model
def evaluate_model(eval_model, eval_loader, num_total_items,
                   item_features_df_for_eval, # DF of item features
                   user_features_df_for_eval, # DF of user features
                   ground_truth_interactions_df, # DF of ground truth interactions for users in eval_loader
                   desc="Evaluating"):
    eval_model.eval()
    all_user_recalls_list, all_user_precisions_list = [], []
    all_preds_for_tm, all_targets_for_tm = [], []

    if num_total_items == 0:
        print(f"Warning ({desc}): num_total_items is 0. Cannot evaluate.")
        return {f"Recall@{CONFIG['top_k']}": 0.0, f"Precision@{CONFIG['top_k']}": 0.0, f"NDCG@{CONFIG['top_k']}": 0.0}

    all_item_ids_tensor = torch.arange(num_total_items, device=device)
    item_side_feature_idx_cols = list(RAW_ITEM_FEATURE_COLS.values())
    
    # Ensure item_features_df_for_eval is correctly indexed and covers all items
    _item_features_df_for_eval_reindexed = item_features_df_for_eval.reindex(np.arange(num_total_items))
    # Fill NaN that might result from reindexing (if an item_id had no features initially)
    for col in item_side_feature_idx_cols: # Fill NaNs for feature columns
        if col in _item_features_df_for_eval_reindexed:
             _item_features_df_for_eval_reindexed[col] = _item_features_df_for_eval_reindexed[col].fillna(0) # Assuming 0 is 'UNKNOWN' index

    all_items_side_features_tensor = torch.tensor(_item_features_df_for_eval_reindexed[item_side_feature_idx_cols].values, dtype=torch.long).to(device)


    with torch.no_grad():
        all_item_vectors = eval_model.item_tower(all_item_ids_tensor, all_items_side_features_tensor)

        for batch_user_ids, _, batch_user_side_feats, _ in tqdm(eval_loader, desc=desc):
            batch_user_ids = batch_user_ids.to(device)
            batch_user_side_feats = batch_user_side_feats.to(device)

            user_vec_batch = eval_model.user_tower(batch_user_ids, batch_user_side_feats)
            scores_batch = torch.matmul(user_vec_batch, all_item_vectors.T)
            target_batch_for_tm = torch.zeros_like(scores_batch, dtype=torch.bool, device='cpu') 

            current_k = min(CONFIG['top_k'], num_total_items)
            if current_k <=0: current_k = 1

            batch_recommended_items_indices = torch.topk(scores_batch.cpu(), k=current_k, dim=1).indices

            for i, user_id_scalar in enumerate(batch_user_ids):
                user_id_py = user_id_scalar.item()
                true_pos_items_for_user_arr = ground_truth_interactions_df[
                    ground_truth_interactions_df[USER_ID_COL] == user_id_py
                ][ITEM_ID_COL].values

                if len(true_pos_items_for_user_arr) > 0:
                    valid_true_pos_items = true_pos_items_for_user_arr[(true_pos_items_for_user_arr >= 0) & (true_pos_items_for_user_arr < num_total_items)]
                    if len(valid_true_pos_items) > 0:
                        target_batch_for_tm[i, valid_true_pos_items] = True

                recommended_for_user = set(batch_recommended_items_indices[i].tolist())
                actual_for_user = set(true_pos_items_for_user_arr.tolist())
                hits = len(recommended_for_user.intersection(actual_for_user))

                recall_user = hits / len(actual_for_user) if len(actual_for_user) > 0 else 0.0
                precision_user = hits / current_k if current_k > 0 else 0.0
                all_user_recalls_list.append(recall_user)
                all_user_precisions_list.append(precision_user)

            all_preds_for_tm.append(scores_batch) # scores_batch is already on device
            all_targets_for_tm.append(target_batch_for_tm.to(device)) # Move targets to device

    mean_recall = np.mean(all_user_recalls_list) if all_user_recalls_list else 0.0
    mean_precision = np.mean(all_user_precisions_list) if all_user_precisions_list else 0.0
    mean_ndcg = 0.0

    if all_preds_for_tm and all_targets_for_tm:
        stacked_preds = torch.cat(all_preds_for_tm, dim=0)
        stacked_targets = torch.cat(all_targets_for_tm, dim=0)
        effective_top_k_metric = min(CONFIG['top_k'], num_total_items)
        if effective_top_k_metric <= 0: effective_top_k_metric = 1

        try:
            ndcg_val = TMF.retrieval_normalized_dcg(
                preds=stacked_preds, target=stacked_targets.bool(), top_k=effective_top_k_metric
            )
            mean_ndcg = ndcg_val.item()
        except Exception as e:
            print(f"ERROR during TMF.retrieval_normalized_dcg in {desc}: {e}")
            if stacked_preds.isinf().any() or stacked_preds.isnan().any(): print(f"Warning ({desc}): Preds contain Inf or NaN for NDCG.")
            if not stacked_targets.any(dim=1).all(): print(f"Warning ({desc}): Some queries have no positive targets for NDCG calculation.")
            if stacked_targets.any(): # Only if there are some targets
                relevant_preds = stacked_preds[stacked_targets] # Preds corresponding to true items
                if relevant_preds.numel() > 0 and (relevant_preds <= 0).all() and stacked_preds.max() > 0:
                     print(f"Warning ({desc}): All predictions for relevant items are zero or negative, while some scores are positive overall.")

    metrics_dict = {
        f"Recall@{CONFIG['top_k']}": mean_recall,
        f"Precision@{CONFIG['top_k']}": mean_precision,
        f"NDCG@{CONFIG['top_k']}": mean_ndcg,
    }
    return metrics_dict

# Helper functions for baselines and segmentation
def get_user_interaction_counts(df_positive_interactions_overall):
    if df_positive_interactions_overall.empty or USER_ID_COL not in df_positive_interactions_overall.columns:
        return pd.Series(dtype=int)
    return df_positive_interactions_overall.groupby(USER_ID_COL).size()

def generate_random_recommendations(user_ids_in_segment, num_total_items, k):
    recommendations = {}
    if num_total_items == 0:
        for user_id in user_ids_in_segment:
            recommendations[user_id] = []
        return recommendations
        
    all_possible_item_indices = np.arange(num_total_items)
    actual_k = min(k, len(all_possible_item_indices)) # Cannot pick more items than available
    
    for user_id in user_ids_in_segment:
        if actual_k == 0:
            recommendations[user_id] = []
        else:
            recommendations[user_id] = np.random.choice(all_possible_item_indices, size=actual_k, replace=False).tolist()
    return recommendations

def generate_previously_bought_recommendations(user_ids_in_segment, df_positive_interactions_overall, k):
    recommendations = {}
    for user_id in user_ids_in_segment:
        user_history_df = df_positive_interactions_overall[df_positive_interactions_overall[USER_ID_COL] == user_id]
        if 'int_date' in user_history_df.columns: 
            user_history_df = user_history_df.sort_values(by='int_date', ascending=False)
        
        user_history_items = user_history_df[ITEM_ID_COL].unique() # Unique items, most recent first if sorted
        
        recs = user_history_items[:k].tolist() # Take top k (can be less if history is short)
        recommendations[user_id] = recs
    return recommendations

def evaluate_baseline_recommendations(recommendations_dict, ground_truth_segment_df, num_total_items, k, desc="Evaluating Baseline"):
    all_user_recalls_list, all_user_precisions_list = [], []
    all_preds_for_tm_baseline, all_targets_for_tm_baseline = [], []

    if num_total_items == 0:
        print(f"Warning ({desc}): num_total_items is 0. Cannot evaluate baseline.")
        return {f"Recall@{k}": 0.0, f"Precision@{k}": 0.0, f"NDCG@{k}": 0.0}

    unique_users_in_gt = ground_truth_segment_df[USER_ID_COL].unique()
    eval_user_ids = [uid for uid in recommendations_dict.keys() if uid in unique_users_in_gt]

    if not eval_user_ids:
        # print(f"No users from {desc} found in the ground truth for evaluation.") 
        return {f"Recall@{k}": 0.0, f"Precision@{k}": 0.0, f"NDCG@{k}": 0.0}

    for user_id_py in tqdm(eval_user_ids, desc=desc, leave=False):
        recommended_items = recommendations_dict.get(user_id_py, [])
        true_pos_items_for_user_arr = ground_truth_segment_df[
            ground_truth_segment_df[USER_ID_COL] == user_id_py
        ][ITEM_ID_COL].values

        recommended_set = set(recommended_items)
        actual_set = set(true_pos_items_for_user_arr.tolist())
        hits = len(recommended_set.intersection(actual_set))

        recall_user = hits / len(actual_set) if len(actual_set) > 0 else 0.0
        
        # Precision denominator: For random baseline, it's k. For "previously bought", it's len(recommended_items)
        # as it might recommend fewer than k items.
        num_recommended_for_precision = len(recommended_items) # For previously bought, this can be < k
        if desc.startswith("Random Baseline"): # For random, it should always recommend k (or num_total_items if fewer)
            num_recommended_for_precision = min(k, num_total_items if num_total_items > 0 else k)


        precision_user = hits / num_recommended_for_precision if num_recommended_for_precision > 0 else 0.0
        all_user_recalls_list.append(recall_user)
        all_user_precisions_list.append(precision_user)

        preds_user = torch.zeros(num_total_items, device='cpu', dtype=torch.float)
        if recommended_items:
            valid_recommended_items = [item for item in recommended_items if 0 <= item < num_total_items]
            if valid_recommended_items:
                for rank, item_idx in enumerate(valid_recommended_items): # Score based on rank
                     preds_user[item_idx] = float(len(valid_recommended_items) - rank) 


        target_user = torch.zeros(num_total_items, dtype=torch.bool, device='cpu')
        if len(true_pos_items_for_user_arr) > 0:
            valid_true_pos_items = true_pos_items_for_user_arr[ (true_pos_items_for_user_arr >=0) & (true_pos_items_for_user_arr < num_total_items)]
            if len(valid_true_pos_items) > 0:
                target_user[valid_true_pos_items] = True

        all_preds_for_tm_baseline.append(preds_user.unsqueeze(0))
        all_targets_for_tm_baseline.append(target_user.unsqueeze(0))

    mean_recall = np.mean(all_user_recalls_list) if all_user_recalls_list else 0.0
    mean_precision = np.mean(all_user_precisions_list) if all_user_precisions_list else 0.0
    mean_ndcg = 0.0

    if all_preds_for_tm_baseline and all_targets_for_tm_baseline:
        stacked_preds_baseline = torch.cat(all_preds_for_tm_baseline, dim=0)
        stacked_targets_baseline = torch.cat(all_targets_for_tm_baseline, dim=0)

     
        effective_top_k_metric = min(k, num_total_items)
        if effective_top_k_metric <= 0: effective_top_k_metric = 1

        try:
            ndcg_val = TMF.retrieval_normalized_dcg(
                preds=stacked_preds_baseline.to(device),
                target=stacked_targets_baseline.bool().to(device),
                top_k=effective_top_k_metric
            )
            mean_ndcg = ndcg_val.item()
        except Exception as e:
            print(f"ERROR during TMF.retrieval_normalized_dcg for baseline {desc}: {e}")
            if stacked_preds_baseline.isinf().any() or stacked_preds_baseline.isnan().any(): print(f"Warning (Baseline {desc}): Preds contain Inf or NaN for NDCG.")
            if not stacked_targets_baseline.any(dim=1).all(): print(f"Warning (Baseline {desc}): Some queries have no positive targets for NDCG.")
            if stacked_targets_baseline.any():
                relevant_preds_baseline = stacked_preds_baseline[stacked_targets_baseline]
                if relevant_preds_baseline.numel() > 0 and (relevant_preds_baseline <=0).all() and stacked_preds_baseline.max() > 0:
                    print(f"Warning (Baseline {desc}): All predictions for relevant items are zero or negative, while some scores are positive.")

    metrics_dict = {
        f"Recall@{k}": mean_recall,
        f"Precision@{k}": mean_precision, # Denominator varies based on baseline type.
        f"NDCG@{k}": mean_ndcg,
    }
    return metrics_dict

def calculate_popularity_baseline_metrics(train_df_positive, user_indices_to_eval,
                                   ground_truth_map_for_segment, # dict: user_id -> set(item_ids)
                                   num_total_items, top_k_val,
                                   desc="Popularity Baseline"):
    if train_df_positive.empty:
        print(f"Warning ({desc}): Training data for popularity is empty.")
        return {f"Recall@{top_k_val}": 0.0, f"Precision@{top_k_val}": 0.0, f"NDCG@{top_k_val}": 0.0}
    if num_total_items == 0:
        print(f"Warning ({desc}): num_total_items is 0. Cannot evaluate popularity.")
        return {f"Recall@{top_k_val}": 0.0, f"Precision@{top_k_val}": 0.0, f"NDCG@{top_k_val}": 0.0}

    item_pop = train_df_positive[ITEM_ID_COL].value_counts()
    actual_top_k_pop = min(top_k_val, len(item_pop))
    top_k_popular_item_ids = item_pop.nlargest(actual_top_k_pop).index.tolist()

    all_user_recalls_list, all_user_precisions_list = [], []
    all_preds_for_tm_pop, all_targets_for_tm_pop = [], []

    if user_indices_to_eval.size == 0 or not ground_truth_map_for_segment:
        # print(f"Warning ({desc}): No users or ground truth for evaluation.") # Can be verbose
        return {f"Recall@{top_k_val}": 0.0, f"Precision@{top_k_val}": 0.0, f"NDCG@{top_k_val}": 0.0}

    pop_scores_for_ndcg = torch.zeros(num_total_items, device='cpu', dtype=torch.float)
    for rank, item_id in enumerate(top_k_popular_item_ids):
        if 0 <= item_id < num_total_items: 
            pop_scores_for_ndcg[item_id] = float(actual_top_k_pop - rank) # Higher rank = higher score

    for user_idx in tqdm(user_indices_to_eval, desc=desc, leave=False):
        actual_pos_set = ground_truth_map_for_segment.get(user_idx, set())
        
        hits = len(set(top_k_popular_item_ids).intersection(actual_pos_set))
        recall = hits / len(actual_pos_set) if len(actual_pos_set) > 0 else 0.0
        # For popularity, precision is hits / number of items recommended (actual_top_k_pop)
        precision = hits / actual_top_k_pop if actual_top_k_pop > 0 else 0.0

        all_user_recalls_list.append(recall)
        all_user_precisions_list.append(precision)

        target_user_pop = torch.zeros(num_total_items, dtype=torch.bool, device='cpu')
        if actual_pos_set:
            valid_actual_pos_indices_list = [item_id for item_id in actual_pos_set if 0 <= item_id < num_total_items]
            if valid_actual_pos_indices_list:
                indices_tensor = torch.tensor(valid_actual_pos_indices_list, dtype=torch.long, device=target_user_pop.device)
                target_user_pop[indices_tensor] = True
        
        all_preds_for_tm_pop.append(pop_scores_for_ndcg.unsqueeze(0))
        all_targets_for_tm_pop.append(target_user_pop.unsqueeze(0))

    mean_recall = np.mean(all_user_recalls_list) if all_user_recalls_list else 0.0
    mean_precision = np.mean(all_user_precisions_list) if all_user_precisions_list else 0.0
    mean_ndcg = 0.0

    if all_preds_for_tm_pop and all_targets_for_tm_pop:
        stacked_preds_pop = torch.cat(all_preds_for_tm_pop, dim=0)
        stacked_targets_pop = torch.cat(all_targets_for_tm_pop, dim=0)
        
        # For NDCG, top_k is how many items we are considering from the fixed popular list
        effective_top_k_metric = min(actual_top_k_pop, num_total_items) # Should be actual_top_k_pop or CONFIG['top_k']?
                                                                      # TMF uses top_k to slice preds. We want to evaluate our list of actual_top_k_pop items.
                                                                      # So, if actual_top_k_pop < top_k_val, TMF will only look at first actual_top_k_pop.
        if effective_top_k_metric <=0 : effective_top_k_metric =1


        try:
            ndcg_val = TMF.retrieval_normalized_dcg(
                preds=stacked_preds_pop.to(device),
                target=stacked_targets_pop.bool().to(device),
                top_k=effective_top_k_metric # Use the number of items actually recommended by popularity for NDCG@k
            )
            mean_ndcg = ndcg_val.item()
        except Exception as e:
            print(f"ERROR during TMF.retrieval_normalized_dcg for {desc}: {e}")
            if stacked_preds_pop.isinf().any() or stacked_preds_pop.isnan().any(): print(f"Warning ({desc}): Preds contain Inf or NaN for NDCG.")
            if not stacked_targets_pop.any(dim=1).all(): print(f"Warning ({desc}): Some queries have no positive targets for NDCG.")
            if stacked_targets_pop.any():
                relevant_preds_pop = stacked_preds_pop[stacked_targets_pop]
                if relevant_preds_pop.numel() > 0 and (relevant_preds_pop <=0).all() and stacked_preds_pop.max() > 0:
                    print(f"Warning ({desc}): All predictions for relevant items are zero or negative, while some scores are positive.")

    # Reporting metrics @CONFIG['top_k'] but calculation details for precision/NDCG might use actual_top_k_pop
    return {
        f"Recall@{top_k_val}": mean_recall,
        f"Precision@{top_k_val}": mean_precision, # Note: Denominator is actual_top_k_pop
        f"NDCG@{top_k_val}": mean_ndcg # Note: NDCG was calculated up to effective_top_k_metric items
    }

# Main Execution
def main():
    df_pos, encoders, user_features_df, item_features_df, num_total_users, num_total_items = load_and_preprocess_data()

    if num_total_users == 0 :
        print("No users found after preprocessing. Exiting.")
        return
    if num_total_items == 0: # num_total_items can be 0 if no items in df_raw
        print("No items found after preprocessing. Exiting.")
        return


    user_side_feature_idx_cols = list(RAW_USER_FEATURE_COLS.values())
    item_side_feature_idx_cols = list(RAW_ITEM_FEATURE_COLS.values())

    # Reindex feature DFs
    # Fill with 0, assuming 0 is a safe 'unknown' index after LabelEncoding UNKNOWN values.
    user_features_df = user_features_df.reindex(np.arange(num_total_users)).fillna(0)
    item_features_df = item_features_df.reindex(np.arange(num_total_items)).fillna(0)


    global_user_features_tensor = torch.tensor(user_features_df[user_side_feature_idx_cols].values, dtype=torch.long).to(device)
    global_item_features_tensor = torch.tensor(item_features_df[item_side_feature_idx_cols].values, dtype=torch.long).to(device)

    user_interaction_counts = get_user_interaction_counts(df_pos)

    all_user_ids_positive = df_pos[USER_ID_COL].unique()
    if len(all_user_ids_positive) == 0:
        print("No positive interactions found. Cannot proceed.")
        return

    # Robust user splitting
    train_val_user_ids, test_user_ids = np.array([]), np.array([])
    train_user_ids, val_user_ids = np.array([]), np.array([])

    if len(all_user_ids_positive) > 0:
        test_size_config = CONFIG['test_size']
   
        if len(all_user_ids_positive) == 1: # Cannot split for test
            train_val_user_ids = all_user_ids_positive
        elif test_size_config == 0.0:
             train_val_user_ids = all_user_ids_positive
        else: 
            n_test = int(np.ceil(len(all_user_ids_positive) * test_size_config))
            if n_test >= len(all_user_ids_positive):
                n_test = len(all_user_ids_positive) - 1 # Keep at least one for train/val
            if n_test < 0 : n_test = 0
            
            if n_test > 0 and n_test < len(all_user_ids_positive):
                 train_val_user_ids, test_user_ids = train_test_split(all_user_ids_positive, test_size=n_test, random_state=CONFIG['random_state'])
            else: # n_test is 0 or would leave train_val empty
                train_val_user_ids = all_user_ids_positive


    if len(train_val_user_ids) > 0:
        val_size_config = CONFIG['val_size'] # This is proportion of the original total dataset
        # Calculate number of validation users based on original total and current train_val_user_ids
        n_total_for_val_calc = len(all_user_ids_positive)
        n_val = int(np.ceil(n_total_for_val_calc * val_size_config))

        if len(train_val_user_ids) == 1: # Cannot split for val
            train_user_ids = train_val_user_ids
        elif n_val == 0:
             train_user_ids = train_val_user_ids
        else: # len > 1 and n_val > 0
            if n_val >= len(train_val_user_ids):
                n_val = len(train_val_user_ids) - 1 # Keep at least one for train
            if n_val < 0: n_val = 0

            if n_val > 0 and n_val < len(train_val_user_ids):
                train_user_ids, val_user_ids = train_test_split(train_val_user_ids, test_size=n_val, random_state=CONFIG['random_state'])
            else: # n_val is 0 or would leave train empty
                train_user_ids = train_val_user_ids

    # Ensure all are numpy arrays
    train_user_ids = np.array(train_user_ids)
    val_user_ids = np.array(val_user_ids)
    test_user_ids = np.array(test_user_ids)


    train_df = df_pos[df_pos[USER_ID_COL].isin(train_user_ids)] if len(train_user_ids) > 0 else pd.DataFrame(columns=df_pos.columns)
    val_df = df_pos[df_pos[USER_ID_COL].isin(val_user_ids)] if len(val_user_ids) > 0 else pd.DataFrame(columns=df_pos.columns)
    test_df = df_pos[df_pos[USER_ID_COL].isin(test_user_ids)] if len(test_user_ids) > 0 else pd.DataFrame(columns=df_pos.columns)

    print(f"Total positive users: {len(all_user_ids_positive)}")
    print(f"Train size: {len(train_df)} interactions, {len(train_user_ids)} users")
    print(f"Validation size: {len(val_df)} interactions, {len(val_user_ids)} users")
    print(f"Test size: {len(test_df)} interactions, {len(test_user_ids)} users")

    trained_model = None

    if not train_df.empty:
        train_dataset = InteractionDataset(train_df, global_user_features_tensor, global_item_features_tensor)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

        val_dataset = InteractionDataset(val_df, global_user_features_tensor, global_item_features_tensor) if not val_df.empty else None
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=0) if val_dataset else None

        user_side_feature_cardinalities = [encoders[idx_col].classes_.size for idx_col in RAW_USER_FEATURE_COLS.values()]
        item_side_feature_cardinalities = [encoders[idx_col].classes_.size for idx_col in RAW_ITEM_FEATURE_COLS.values()]

        model_common_config = {
            "embed_dim_id": CONFIG['embedding_dim_ids'], "embed_dim_feat": CONFIG['embedding_dim_features'],
            "transformer_nhead": CONFIG['transformer_nhead'], "transformer_nlayers": CONFIG['transformer_nlayers'],
            "transformer_dim_feedforward": CONFIG['transformer_dim_feedforward'], "out_dim": CONFIG['final_mlp_embed_dim']
        }
        model = TwoTowerModel(
            user_tower_config={**model_common_config, "id_dim": num_total_users, "feature_cardinalities": user_side_feature_cardinalities},
            item_tower_config={**model_common_config, "id_dim": num_total_items, "feature_cardinalities": item_side_feature_cardinalities}
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        
        trained_model = train_model(model, train_loader, val_loader, optimizer, num_total_items,
                                    global_item_features_tensor, user_features_df, item_features_df, val_df)
    else:
        print("Training data is empty. Skipping model training.")


    print("\n--- Evaluating on Test Set ---")
    if not test_df.empty and len(test_user_ids) > 0 :
        # Define user segments based on overall interaction counts
        segments = {
            "All Users": test_user_ids, # These are already the test_user_ids
             "1-2 Interactions": user_interaction_counts[
                user_interaction_counts.isin([1, 2])
            ].index,
            "3-5 Interactions": user_interaction_counts[
                user_interaction_counts.isin([3, 4, 5])
            ].index,
            ">5 Interactions": user_interaction_counts[user_interaction_counts > 5].index
        }

        for segment_name, segment_defining_user_ids_from_counts in segments.items():
            if segment_name == "All Users":
                current_segment_test_user_ids = test_user_ids
            else:
                current_segment_test_user_ids = np.intersect1d(
                    test_user_ids,
                    segment_defining_user_ids_from_counts.to_numpy() if isinstance(segment_defining_user_ids_from_counts, (pd.Index, pd.Series)) else segment_defining_user_ids_from_counts
                )
            
            if len(current_segment_test_user_ids) == 0:
                print(f"\nSegment: {segment_name} - No test users in this segment. Skipping.")
                continue

            print(f"\n--- Segment: {segment_name} ({len(current_segment_test_user_ids)} users) ---")
            
            # Ground truth for this specific segment from the overall test_df
            segment_test_df = test_df[test_df[USER_ID_COL].isin(current_segment_test_user_ids)]
            if segment_test_df.empty: # Should not happen if current_segment_test_user_ids is not empty and from test_df
                print(f"Segment {segment_name} has no interactions in the test data (after filtering). Skipping.")
                continue

            segment_test_dataset = InteractionDataset(segment_test_df, global_user_features_tensor, global_item_features_tensor)
            segment_test_loader = DataLoader(segment_test_dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=0)

            # Create ground_truth_map for this segment
            ground_truth_map_segment = segment_test_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict()

            # 1. Evaluate Main Model
            if trained_model is not None:
                print(f"Evaluating Main Model for {segment_name}...")
                main_model_metrics = evaluate_model(trained_model, segment_test_loader, num_total_items,
                                                    item_features_df, user_features_df, # Pass full feature DFs
                                                    segment_test_df, # GT interactions for this segment
                                                    desc=f"Testing Main Model ({segment_name})")
                print(f"Main Model ({segment_name}) - Recall@{CONFIG['top_k']}: {main_model_metrics[f'Recall@{CONFIG['top_k']}']:.4f}, "
                      f"Precision@{CONFIG['top_k']}: {main_model_metrics[f'Precision@{CONFIG['top_k']}']:.4f}, "
                      f"NDCG@{CONFIG['top_k']}: {main_model_metrics[f'NDCG@{CONFIG['top_k']}']:.4f}")
            else:
                print(f"Main Model not trained. Skipping its evaluation for {segment_name}.")

            # 2. Evaluate Random Baseline
            print(f"Evaluating Random Baseline for {segment_name}...")
            random_recs = generate_random_recommendations(current_segment_test_user_ids, num_total_items, CONFIG['top_k'])
            random_metrics = evaluate_baseline_recommendations(random_recs, segment_test_df, num_total_items, CONFIG['top_k'], desc=f"Random Baseline ({segment_name})")
            print(f"Random Baseline ({segment_name}) - Recall@{CONFIG['top_k']}: {random_metrics[f'Recall@{CONFIG['top_k']}']:.4f}, "
                  f"Precision@{CONFIG['top_k']}: {random_metrics[f'Precision@{CONFIG['top_k']}']:.4f}, "
                  f"NDCG@{CONFIG['top_k']}: {random_metrics[f'NDCG@{CONFIG['top_k']}']:.4f}")
            
            # 3. Evaluate Previously Bought Baseline
            # Not necessary useful
            # print(f"Evaluating Previously Bought Baseline for {segment_name}...")
            # bought_recs = generate_previously_bought_recommendations(current_segment_test_user_ids, df_pos, CONFIG['top_k'])
            # bought_metrics = evaluate_baseline_recommendations(bought_recs, segment_test_df, num_total_items, CONFIG['top_k'], desc=f"Previously Bought Baseline ({segment_name})")
            # print(f"Previously Bought Baseline ({segment_name}) - Recall@{CONFIG['top_k']}: {bought_metrics[f'Recall@{CONFIG['top_k']}']:.4f}, "
            #       f"Precision@{CONFIG['top_k']}: {bought_metrics[f'Precision@{CONFIG['top_k']}']:.4f}, "
            #       f"NDCG@{CONFIG['top_k']}: {bought_metrics[f'NDCG@{CONFIG['top_k']}']:.4f}")

            # 4. Evaluate Popularity Baseline
            print(f"Evaluating Popularity Baseline for {segment_name}...")
            if not train_df.empty : # Popularity is derived from train_df
                pop_metrics = calculate_popularity_baseline_metrics(
                    train_df_positive=train_df,
                    user_indices_to_eval=current_segment_test_user_ids,
                    ground_truth_map_for_segment=ground_truth_map_segment,
                    num_total_items=num_total_items,
                    top_k_val=CONFIG['top_k'],
                    desc=f"Popularity Baseline ({segment_name})"
                )
                print(f"Popularity Baseline ({segment_name}) - Recall@{CONFIG['top_k']}: {pop_metrics[f'Recall@{CONFIG['top_k']}']:.4f}, "
                      f"Precision@{CONFIG['top_k']}: {pop_metrics[f'Precision@{CONFIG['top_k']}']:.4f}, "
                      f"NDCG@{CONFIG['top_k']}: {pop_metrics[f'NDCG@{CONFIG['top_k']}']:.4f}")
            else:
                print(f"Skipping Popularity Baseline for {segment_name} as training data was empty.")
    else:
        print("Test set is empty or no test users defined. Skipping final evaluation.")
    print("\nDone.")

if __name__ == "__main__":
    main()