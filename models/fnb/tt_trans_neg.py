import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
import random
from collections import Counter
import os

# Configuration
CONFIG = {
    "split_strategy": "temporal",  # "user_cold_start" or "temporal"
    "test_size": 0.15, # Used for user_cold_start; for temporal, it's effectively leave-one-out per user
    "val_size": 0.15,  # Used for user_cold_start; for temporal, it's effectively leave-second-to-last-out per user
    "embedding_dim_ids": 32,
    "embedding_dim_features": 16,
    "transformer_nhead": 8,
    "transformer_nlayers": 4,
    "transformer_dim_feedforward": 64,
    "final_mlp_embed_dim": 64,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "epochs": 50, # Max epochs
    "batch_size": 1024,
    "top_k": 10,
    "random_state": 42,
    "early_stopping_patience": 6
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Column name constants
USER_ID_COL_RAW = 'idcol'
ITEM_ID_COL_RAW = 'item'
USER_ID_COL = "user_idx"
ITEM_ID_COL = "item_idx"
INTERACTION_COL = 'interaction'
DATE_COL = 'int_date' # Assuming this is the name of your date column

RAW_USER_FEATURE_COLS = {"segment": "segment_idx", "beh_segment": "beh_segment_idx", "active_ind": "active_ind_idx"}
RAW_ITEM_FEATURE_COLS = {"item_type": "item_type_idx", "item_descrip": "item_descrip_idx"}

# NumPy Metric Functions 
def numpy_recall_at_k(recommended_list, actual_set, k):
    if not actual_set: return 0.0
    recommended_list_k = recommended_list[:k]
    hits = len(set(recommended_list_k).intersection(actual_set))
    return hits / len(actual_set)

def numpy_precision_at_k(recommended_list, actual_set, k):
    recommended_list_k = recommended_list[:k]
    if not recommended_list_k: return 0.0
    hits = len(set(recommended_list_k).intersection(actual_set))
    return hits / len(recommended_list_k)

def numpy_hit_rate_at_k(recommended_list, actual_set, k):
    recommended_list_k = recommended_list[:k]
    for item in recommended_list_k:
        if item in actual_set: return 1.0
    return 0.0

def numpy_mrr_at_k(recommended_list, actual_set, k):
    recommended_list_k = recommended_list[:k]
    for i, item in enumerate(recommended_list_k):
        if item in actual_set: return 1.0 / (i + 1.0)
    return 0.0

def numpy_dcg_at_k(recommended_list, actual_set, k):
    dcg = 0.0
    for i, item in enumerate(recommended_list[:k]):
        if item in actual_set: dcg += 1.0 / np.log2(i + 2)
    return dcg

def numpy_idcg_at_k(actual_set, k):
    if not actual_set: return 0.0
    num_relevant_items = len(actual_set)
    limit = min(num_relevant_items, k)
    idcg = 0.0
    for i in range(limit): idcg += 1.0 / np.log2(i + 2)
    return idcg

def numpy_ndcg_at_k(recommended_list, actual_set, k):
    dcg = numpy_dcg_at_k(recommended_list, actual_set, k)
    idcg = numpy_idcg_at_k(actual_set, k)
    if idcg == 0: return 0.0
    return dcg / idcg

def gini_coefficient(x):
    x = np.asarray(x, dtype=np.float64)
    x_min = np.amin(x)
    if x_min < 0: x = x - x_min
    if np.all(x == 0):
        if x.shape[0] == 0: return 0.0
    x = x + 1e-9
    x = np.sort(x)
    index = np.arange(1, x.shape[0] + 1)
    n = x.shape[0]
    if n == 0 or np.sum(x) == 0: return 0.0
    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

# Data Loading & Preprocessing
def load_and_preprocess_data():
    print(f"Using device: {device}, Config: {CONFIG}")
    try:
        df_raw = pd.read_csv("../../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv")
        print("Successfully loaded file.")
    except FileNotFoundError:
        print("File not Found.")
     

    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], format='%d-%b-%y')
    df_raw['user_id_str'] = df_raw[USER_ID_COL_RAW].astype(str)
    df_raw['item_id_str'] = df_raw[ITEM_ID_COL_RAW].astype(str)

    feature_encoders = {}
    for col_str, new_col_idx in [('user_id_str', USER_ID_COL), ('item_id_str', ITEM_ID_COL)]:
        le = LabelEncoder()
        df_raw[new_col_idx] = le.fit_transform(df_raw[col_str])
        feature_encoders[new_col_idx] = le

    for raw_col_map in [RAW_USER_FEATURE_COLS, RAW_ITEM_FEATURE_COLS]:
        for raw_col, processed_idx_col_name in raw_col_map.items():
            if raw_col not in df_raw.columns: # Handle missing raw feature columns in dummy data
                 df_raw[raw_col] = "UNKNOWN"
            df_raw[raw_col] = df_raw[raw_col].fillna("UNKNOWN").astype(str)
            le = LabelEncoder()
            df_raw[processed_idx_col_name] = le.fit_transform(df_raw[raw_col])
            feature_encoders[processed_idx_col_name] = le

    num_total_users = len(feature_encoders[USER_ID_COL].classes_)
    num_total_items = len(feature_encoders[ITEM_ID_COL].classes_)
    print(f"Total unique users: {num_total_users}, Total unique items: {num_total_items}")

    df_all_interactions = df_raw.copy()
    df_positive_interactions = df_raw[df_raw[INTERACTION_COL].isin(['CLICK', 'CHECKOUT'])].copy()
    df_display_interactions = df_raw[df_raw[INTERACTION_COL] == 'DISPLAY'].copy()
    
    df_negative_interactions = pd.DataFrame(columns=df_display_interactions.columns)
    if not df_display_interactions.empty and not df_positive_interactions.empty:
        positive_pairs_set = set(df_positive_interactions[USER_ID_COL].astype(str) + "_" + df_positive_interactions[ITEM_ID_COL].astype(str))
        df_display_interactions['temp_pair_id'] = df_display_interactions[USER_ID_COL].astype(str) + "_" + df_display_interactions[ITEM_ID_COL].astype(str)
        df_negative_interactions = df_display_interactions[~df_display_interactions['temp_pair_id'].isin(positive_pairs_set)].copy()
        if 'temp_pair_id' in df_negative_interactions.columns:
            df_negative_interactions.drop(columns=['temp_pair_id'], inplace=True)
    elif not df_display_interactions.empty:
        df_negative_interactions = df_display_interactions.copy()

    print(f"Total interactions: {len(df_all_interactions)}")
    print(f"Positive interactions (CLICK/CHECKOUT): {len(df_positive_interactions)}")
    print(f"Explicit negative interactions (DISPLAYs not leading to positive): {len(df_negative_interactions)}")

    user_side_feature_idx_cols = list(RAW_USER_FEATURE_COLS.values())
    if USER_ID_COL in df_raw and num_total_users > 0:
        user_features_df = df_raw[[USER_ID_COL] + user_side_feature_idx_cols].drop_duplicates(subset=[USER_ID_COL]).set_index(USER_ID_COL).sort_index()
    else:
        user_features_df = pd.DataFrame(columns=[USER_ID_COL] + user_side_feature_idx_cols).set_index(USER_ID_COL)

    item_side_feature_idx_cols = list(RAW_ITEM_FEATURE_COLS.values())
    if ITEM_ID_COL in df_raw and num_total_items > 0:
        item_features_df = df_raw[[ITEM_ID_COL] + item_side_feature_idx_cols].drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL).sort_index()
    else:
        item_features_df = pd.DataFrame(columns=[ITEM_ID_COL] + item_side_feature_idx_cols).set_index(ITEM_ID_COL)
    
    return df_all_interactions, df_positive_interactions, df_negative_interactions, \
           feature_encoders, user_features_df, item_features_df, \
           num_total_users, num_total_items

# Dataset Class 
class InteractionDataset(Dataset):
    def __init__(self, df, user_features_tensor, item_features_tensor):
        self.users = df[USER_ID_COL].values
        self.items = df[ITEM_ID_COL].values
        self.user_features_tensor = user_features_tensor
        self.item_features_tensor = item_features_tensor
        self.num_user_features = user_features_tensor.shape[1] if user_features_tensor.ndim > 1 and user_features_tensor.shape[1] > 0 else 0
        self.num_item_features = item_features_tensor.shape[1] if item_features_tensor.ndim > 1 and item_features_tensor.shape[1] > 0 else 0

    def __len__(self): return len(self.users)
    def __getitem__(self, idx):
        user_id = self.users[idx]; item_id = self.items[idx]
        user_side_feats = torch.zeros(self.num_user_features, dtype=torch.long)
        if 0 <= user_id < self.user_features_tensor.size(0):
            if self.num_user_features > 0: user_side_feats = self.user_features_tensor[user_id]
        item_side_feats = torch.zeros(self.num_item_features, dtype=torch.long)
        if self.num_item_features > 0 and 0 <= item_id < self.item_features_tensor.size(0):
            item_side_feats = self.item_features_tensor[item_id]
        return user_id, item_id, user_side_feats, item_side_feats

# Model Definition
class Tower(nn.Module):
    def __init__(self, id_dim, feature_cardinalities, embed_dim_id, embed_dim_feat, transformer_nhead, transformer_nlayers, transformer_dim_feedforward, out_dim):
        super().__init__()
        self.id_embedding = nn.Embedding(id_dim, embed_dim_id) if id_dim > 0 else None
        self.feature_embeddings = nn.ModuleList([nn.Embedding(max(1,c), embed_dim_feat) for c in feature_cardinalities if c > 0] if feature_cardinalities and sum(c > 0 for c in feature_cardinalities) > 0 else [])
        self.has_features = len(self.feature_embeddings) > 0
        self.num_actual_feature_embeddings = len(self.feature_embeddings)
        fc_input_dim = (embed_dim_id if self.id_embedding else 0) + (embed_dim_feat if self.has_features else 0)
        if self.has_features:
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim_feat, nhead=transformer_nhead, dim_feedforward=transformer_dim_feedforward, dropout=0.1, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
        else: self.transformer = None
        if fc_input_dim == 0:
            self.dummy_param_for_no_input = nn.Parameter(torch.empty(0)); self.fc = None
        else: self.fc = nn.Sequential(nn.Linear(fc_input_dim, fc_input_dim*2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(fc_input_dim*2, out_dim))

    def forward(self, main_id, side_features_batch):
        outputs = []
        if self.id_embedding and main_id is not None and main_id.numel() > 0: outputs.append(self.id_embedding(main_id.clamp(min=0, max=self.id_embedding.num_embeddings-1)))
        if self.has_features and self.transformer and side_features_batch is not None and side_features_batch.ndim == 2 and side_features_batch.size(1) == self.num_actual_feature_embeddings and self.num_actual_feature_embeddings > 0:
            feat_embs_list = [self.feature_embeddings[i](side_features_batch[:, i].clamp(min=0, max=emb_layer.num_embeddings-1)) for i, emb_layer in enumerate(self.feature_embeddings)]
            if feat_embs_list: outputs.append(self.transformer(torch.stack(feat_embs_list, dim=1)).mean(dim=1))
        if not outputs:
            batch_s = main_id.size(0) if main_id is not None and main_id.numel()>0 else (side_features_batch.size(0) if side_features_batch is not None and side_features_batch.numel()>0 else 1)
            out_dim_val = CONFIG['final_mlp_embed_dim'] if not (self.fc and isinstance(self.fc[-1], nn.Linear)) else self.fc[-1].out_features
            return torch.zeros(batch_s, out_dim_val, device=device if main_id is None or main_id.numel()==0 else main_id.device)
        combined = torch.cat(outputs, dim=1)
        return torch.zeros(combined.size(0), CONFIG['final_mlp_embed_dim'], device=combined.device) if self.fc is None else self.fc(combined)

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower_config, item_tower_config):
        super().__init__(); self.user_tower = Tower(**user_tower_config); self.item_tower = Tower(**item_tower_config)
    def forward(self, user_ids, item_ids, user_side_feats, item_side_feats):
        return self.user_tower(user_ids, user_side_feats), self.item_tower(item_ids, item_side_feats)

def bpr_loss(pos_score, neg_score): return -torch.mean(F.logsigmoid(pos_score - neg_score))

# Training Loop
def train_model(model, train_loader, val_loader, optimizer, num_total_items, global_item_features_tensor, user_features_df, item_features_df, ground_truth_val_df, user_to_explicit_negatives_map, feature_encoders, item_self_information_scores):
    best_val_ndcg = -1.0; epochs_no_improve = 0; best_model_state_dict = None
    if num_total_items == 0: print("Error: num_total_items is 0."); return model
    for epoch in range(CONFIG['epochs']):
        model.train(); total_loss = 0; num_batches_processed = 0
        for user_ids, pos_item_ids, user_side_feats, pos_item_side_feats in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            if user_ids.numel() == 0: continue
            user_ids, pos_item_ids, user_side_feats, pos_item_side_feats = user_ids.to(device), pos_item_ids.to(device), user_side_feats.to(device), pos_item_side_feats.to(device)
            batch_actual_size = user_ids.size(0); neg_item_ids_list = []
            if num_total_items <= 0: continue
            for i in range(batch_actual_size):
                uid, pid = user_ids[i].item(), pos_item_ids[i].item(); chosen_neg_id = -1
                user_explicit_negs = [idx for idx in user_to_explicit_negatives_map.get(uid, []) if idx != pid and 0 <= idx < num_total_items]
                if user_explicit_negs: chosen_neg_id = random.choice(user_explicit_negs)
                else:
                    if num_total_items > 1:
                        while True:
                            rand_neg_id = torch.randint(0, num_total_items, (1,)).item()
                            if rand_neg_id != pid: chosen_neg_id = rand_neg_id; break
                    else: chosen_neg_id = 0 # Only one item, must be positive, so negative is itself (or 0 if only 1 item exists)
                neg_item_ids_list.append(chosen_neg_id)
            neg_item_ids = torch.tensor(neg_item_ids_list, dtype=torch.long, device=device)
            neg_item_side_feats = global_item_features_tensor[neg_item_ids.clamp(min=0, max=max(0, num_total_items-1))] if global_item_features_tensor.ndim > 1 and global_item_features_tensor.shape[1] > 0 else torch.empty((batch_actual_size, 0), dtype=torch.long, device=device)
            user_vec, pos_item_vec, neg_item_vec = model(user_ids, pos_item_ids, user_side_feats, pos_item_side_feats)[0], model.item_tower(pos_item_ids, pos_item_side_feats), model.item_tower(neg_item_ids, neg_item_side_feats)
            for vec_idx, vec in enumerate([user_vec, pos_item_vec, neg_item_vec]):
                if vec is None: 
                    if vec_idx == 0: user_vec = torch.zeros(batch_actual_size, CONFIG['final_mlp_embed_dim'], device=device)
                    elif vec_idx == 1: pos_item_vec = torch.zeros(batch_actual_size, CONFIG['final_mlp_embed_dim'], device=device)
                    else: neg_item_vec = torch.zeros(batch_actual_size, CONFIG['final_mlp_embed_dim'], device=device)
            pos_scores, neg_scores = torch.sum(user_vec * pos_item_vec, dim=1), torch.sum(user_vec * neg_item_vec, dim=1)
            if pos_scores.numel() == 0 or neg_scores.numel() == 0: continue
            loss = bpr_loss(pos_scores, neg_scores)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
            num_batches_processed +=1
        avg_train_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
        print(f"Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f}")
        if val_loader and ground_truth_val_df is not None and not ground_truth_val_df.empty:
            val_metrics = evaluate_model(model, val_loader, num_total_items, item_features_df, user_features_df, ground_truth_val_df, feature_encoders, item_self_information_scores, desc="Validating")
            k_val = CONFIG.get('top_k',10); val_ndcg = val_metrics[f"NDCG@{k_val}"]
            print(f"Epoch {epoch+1} Validation - Recall: {val_metrics[f'Recall@{k_val}']:.4f}, Precision: {val_metrics[f'Precision@{k_val}']:.4f}, NDCG: {val_ndcg:.4f}, HitRate: {val_metrics[f'HitRate@{k_val}']:.4f}, MRR: {val_metrics[f'MRR@{k_val}']:.4f}")
            if val_ndcg > best_val_ndcg: best_val_ndcg = val_ndcg; epochs_no_improve = 0; best_model_state_dict = copy.deepcopy(model.state_dict()); print(f"New best: {best_val_ndcg:.4f}")
            else: epochs_no_improve += 1; print(f"No improvement. Count: {epochs_no_improve}/{CONFIG['early_stopping_patience']}")
            if epochs_no_improve >= CONFIG['early_stopping_patience']: print("Early stopping."); break
        else: print(f"Epoch {epoch+1} completed. No/empty validation.")
    if best_model_state_dict: model.load_state_dict(best_model_state_dict); print("Loaded best model.")
    else: print("Using model from last epoch.")
    return model

# Evaluation
@torch.no_grad()
def evaluate_model(eval_model, eval_loader, num_total_items,
                   item_features_df_for_eval, user_features_df_for_eval,
                   ground_truth_interactions_df, feature_encoders,
                   item_self_information_scores, desc="Evaluating"):
    eval_model.eval()
    k = CONFIG.get('top_k', 10)

    # Use singular, capitalized keys for accuracy metrics
    metrics_accumulator = {
        "Recall": [], "Precision": [], "NDCG": [], "HitRate": [], "MRR": [],
        "ILD_scores": [], "Novelty_scores": [] # Keep _scores for specific handling logic
    }
    all_recommended_items_global_set = set()
    item_recommendation_counts = Counter()

    default_metrics_dict = {
        f"Recall@{k}": 0.0, f"Precision@{k}": 0.0, f"NDCG@{k}": 0.0,
        f"HitRate@{k}": 0.0, f"MRR@{k}": 0.0,
        f"Coverage@{k}": 0.0, f"ILD@{k}": 0.0,
        f"Novelty@{k}": 0.0, f"Fairness_Gini@{k}": 0.0
    }

    if num_total_items == 0:
        print(f"Warning ({desc}): num_total_items is 0. Cannot evaluate.")
        return default_metrics_dict

    all_item_ids_t = torch.arange(num_total_items, device=device)
    item_feat_cols = list(RAW_ITEM_FEATURE_COLS.values())
    item_feat_df_reidx = item_features_df_for_eval.reindex(np.arange(num_total_items))
    for col in item_feat_cols:
        if col in item_feat_df_reidx.columns: #Check if column exists before trying to fill
            un_val = 0
            if col in feature_encoders and hasattr(feature_encoders[col],'classes_') and "UNKNOWN" in feature_encoders[col].classes_:
                try:
                    un_val = feature_encoders[col].transform(["UNKNOWN"])[0]
                except ValueError: # If "UNKNOWN" wasn't actually in classes_ seen during fit
                    if len(feature_encoders[col].classes_) > 0 : un_val = 0 # Default to 0 index if UNKNOWN fails
                    else: pass # if no classes, no fill needed for this column
            item_feat_df_reidx[col] = item_feat_df_reidx[col].fillna(un_val)
    item_feat_df_reidx = item_feat_df_reidx.fillna(0) # Catch any other NaNs

    valid_item_feat_cols = [c for c in item_feat_cols if c in item_feat_df_reidx.columns]
    all_items_side_feats_t = torch.tensor(item_feat_df_reidx[valid_item_feat_cols].values,dtype=torch.long).to(device) if valid_item_feat_cols and not item_feat_df_reidx.empty else torch.empty((num_total_items,0),dtype=torch.long,device=device)

    all_item_vectors = eval_model.item_tower(all_item_ids_t, all_items_side_feats_t)
    if all_item_vectors is None or all_item_vectors.numel()==0:
        print(f"Warning ({desc}): all_item_vectors are None or empty. Eval results will be 0.")
        return default_metrics_dict
    all_item_vectors_norm = F.normalize(all_item_vectors, p=2, dim=1) if all_item_vectors.numel() > 0 else None

    for batch_user_ids, _, batch_user_side_feats, _ in tqdm(eval_loader, desc=desc):
        if batch_user_ids.numel()==0: continue
        user_vec_batch = eval_model.user_tower(batch_user_ids.to(device), batch_user_side_feats.to(device))
        if user_vec_batch is None or (user_vec_batch.numel()==0 and batch_user_ids.size(0)>0):
            user_vec_batch = torch.zeros(batch_user_ids.size(0), CONFIG['final_mlp_embed_dim'], device=device)
        if batch_user_ids.size(0)>0 and user_vec_batch.size(0)==0: continue

        scores_batch = torch.matmul(user_vec_batch, all_item_vectors.T)
        current_k_topk = min(k, num_total_items if num_total_items>0 else 1)
        if current_k_topk<=0: current_k_topk=1

        batch_recs_indices = torch.empty((scores_batch.size(0),0),dtype=torch.long)
        if scores_batch.numel()>0 and scores_batch.size(1)>0: # Check if scores_batch has items to recommend
            actual_k_for_this_batch = min(current_k_topk, scores_batch.size(1))
            if actual_k_for_this_batch > 0 :
                 batch_recs_indices = torch.topk(scores_batch.cpu(), k=actual_k_for_this_batch, dim=1).indices

        for i, user_id_py in enumerate(batch_user_ids.tolist()):
            actual_set = set()
            if ground_truth_interactions_df is not None and not ground_truth_interactions_df.empty and USER_ID_COL in ground_truth_interactions_df.columns:
                user_gt_rows = ground_truth_interactions_df[ground_truth_interactions_df[USER_ID_COL]==user_id_py]
                if not user_gt_rows.empty and ITEM_ID_COL in user_gt_rows.columns:
                    actual_set = set(user_gt_rows[ITEM_ID_COL].values)

            recs_list = batch_recs_indices[i].tolist() if i < batch_recs_indices.shape[0] and batch_recs_indices.numel() > 0 else []

            metrics_accumulator["Recall"].append(numpy_recall_at_k(recs_list, actual_set, k))
            metrics_accumulator["Precision"].append(numpy_precision_at_k(recs_list, actual_set, k))
            metrics_accumulator["NDCG"].append(numpy_ndcg_at_k(recs_list, actual_set, k))
            metrics_accumulator["HitRate"].append(numpy_hit_rate_at_k(recs_list, actual_set, k))
            metrics_accumulator["MRR"].append(numpy_mrr_at_k(recs_list, actual_set, k))

            if recs_list:
                all_recommended_items_global_set.update(recs_list)
                item_recommendation_counts.update(recs_list)
                if len(recs_list) >= 2 and all_item_vectors_norm is not None:
                    valid_rec_indices = [r for r in recs_list if 0 <= r < all_item_vectors_norm.size(0)]
                    if len(valid_rec_indices) >= 2:
                        recs_item_vecs = all_item_vectors_norm[torch.tensor(valid_rec_indices, device=all_item_vectors_norm.device)]
                        if recs_item_vecs.shape[0] >= 2: # Check again after potential filtering
                            sim_matrix = torch.matmul(recs_item_vecs, recs_item_vecs.T)
                            iu = torch.triu_indices(*sim_matrix.shape, offset=1)
                            if iu.numel()>0: metrics_accumulator["ILD_scores"].append(torch.mean(1.0 - sim_matrix[iu[0],iu[1]]).item())
                            else: metrics_accumulator["ILD_scores"].append(0.0)
                        else: metrics_accumulator["ILD_scores"].append(0.0)
                    else: metrics_accumulator["ILD_scores"].append(0.0)
                elif recs_list: metrics_accumulator["ILD_scores"].append(0.0) # If <2 recs but some recs
                
                if item_self_information_scores is not None and item_self_information_scores.numel()>0:
                    valid_novel_recs = [r for r in recs_list if 0 <= r < len(item_self_information_scores)]
                    if valid_novel_recs:
                        metrics_accumulator["Novelty_scores"].append(torch.mean(item_self_information_scores[torch.tensor(valid_novel_recs,dtype=torch.long)]).item())

    final_metrics = {**default_metrics_dict}
    for metric_key_base, values in metrics_accumulator.items():
        if "_scores" not in metric_key_base: # Handles Recall, Precision, NDCG, HitRate, MRR
            final_metrics[f"{metric_key_base}@{k}"] = np.mean(values) if values else 0.0
        elif metric_key_base == "ILD_scores":
            final_metrics[f"ILD@{k}"] = np.mean(values) if values else 0.0
        elif metric_key_base == "Novelty_scores":
            final_metrics[f"Novelty@{k}"] = np.mean(values) if values else 0.0

    final_metrics[f"Coverage@{k}"] = len(all_recommended_items_global_set)/num_total_items if num_total_items>0 else 0.0
    if num_total_items > 0 and item_recommendation_counts:
        counts_arr = np.zeros(num_total_items)
        for item_id_count, cnt_val in item_recommendation_counts.items():
            if 0 <= item_id_count < num_total_items: counts_arr[item_id_count] = cnt_val
        final_metrics[f"Fairness_Gini@{k}"] = gini_coefficient(counts_arr)
    else: # Ensure Gini is 0 if no recs or items
         final_metrics[f"Fairness_Gini@{k}"] = 0.0
    return final_metrics

# Baselines
def get_user_interaction_counts(df_pos):
    return df_pos.groupby(USER_ID_COL).size() if not df_pos.empty and USER_ID_COL in df_pos else pd.Series(dtype='int64')

def generate_random_recommendations(uids, num_items, k_val):
    recs = {}; all_indices = np.arange(num_items); actual_k = min(k_val, num_items)
    for uid in uids: recs[uid] = np.random.choice(all_indices, size=actual_k, replace=False).tolist() if actual_k > 0 else []
    return recs

def generate_previously_bought_recommendations(uids, df_pos_all, k_val):
    recs = {}
    for uid in uids:
        hist = df_pos_all[df_pos_all[USER_ID_COL] == uid]
        if DATE_COL in hist.columns: hist = hist.sort_values(by=DATE_COL, ascending=False)
        recs[uid] = hist[ITEM_ID_COL].unique()[:k_val].tolist()
    return recs

def calculate_metrics_for_baseline_recs(
    recommendations_dict,      # Dict: user_id -> list of recommended item_ids
    ground_truth_segment_df,   # DataFrame of ground truth for the current segment
    num_total_items,
    k_val,                     # The @K value for metrics
    item_self_information_scores, # Tensor for Novelty
    all_item_vectors_normalized,  # Tensor for ILD (can be None)
    desc="Evaluating Baseline"
):
    # Use singular, capitalized keys
    acc = {
        "Recall": [], "Precision": [], "NDCG": [], "HitRate": [], "MRR": [],
        "ILD_scores": [], "Novelty_scores": []
    }
    all_recs_set = set()
    item_counts = Counter()

    # default_mets uses singular, capitalized keys
    default_mets = {
        f"Recall@{k_val}": 0.0, f"Precision@{k_val}": 0.0, f"NDCG@{k_val}": 0.0,
        f"HitRate@{k_val}": 0.0, f"MRR@{k_val}": 0.0,
        f"Coverage@{k_val}": 0.0, f"ILD@{k_val}": 0.0,
        f"Novelty@{k_val}": 0.0, f"Fairness_Gini@{k_val}": 0.0
    }

    if num_total_items == 0:
        print(f"Warning ({desc}): num_total_items is 0. Cannot evaluate baseline.")
        return {key_str:float(val_num) for key_str,val_num in default_mets.items()}


    eval_user_ids = []
    if ground_truth_segment_df is not None and not ground_truth_segment_df.empty and USER_ID_COL in ground_truth_segment_df.columns:
        # Users present in ground truth for this segment
        gt_users_in_segment = ground_truth_segment_df[USER_ID_COL].unique()
        # Evaluate only those users for whom we have recommendations AND ground truth
        eval_user_ids = [uid for uid in recommendations_dict.keys() if uid in gt_users_in_segment]
        if not eval_user_ids and len(gt_users_in_segment) > 0: # GT users exist, but no recs for them
            print(f"Warning ({desc}): No recommendations found for users in the ground truth segment. Metrics will be 0.")
            eval_user_ids = list(gt_users_in_segment) # Will result in 0 for accuracy metrics if recs are empty
    elif not recommendations_dict: # No recommendations at all
         print(f"Warning ({desc}): recommendations_dict is empty. Cannot evaluate baseline.")
         return {key_str:float(val_num) for key_str,val_num in default_mets.items()}
    else: # No ground truth, but recommendations exist. Accuracy metrics will be 0.
        print(f"Warning ({desc}): ground_truth_segment_df is None or empty. Accuracy metrics (Recall, Prec, etc.) will be 0.")
        eval_user_ids = list(recommendations_dict.keys())


    if not eval_user_ids:
        print(f"Warning ({desc}): No users to evaluate for this baseline segment. Returning default metrics.")
        return {key_str:float(val_num) for key_str,val_num in default_mets.items()}

    for uid in tqdm(eval_user_ids, desc=desc, leave=False):
        recs_list = recommendations_dict.get(uid, [])[:k_val] # Ensure recs are capped at k_val
        actual_set = set()
        if ground_truth_segment_df is not None and not ground_truth_segment_df.empty and USER_ID_COL in ground_truth_segment_df.columns:
            user_gt_rows = ground_truth_segment_df[ground_truth_segment_df[USER_ID_COL]==uid]
            if not user_gt_rows.empty and ITEM_ID_COL in user_gt_rows.columns:
                 actual_set = set(user_gt_rows[ITEM_ID_COL].values)

        acc["Recall"].append(numpy_recall_at_k(recs_list, actual_set, k_val))
        acc["Precision"].append(numpy_precision_at_k(recs_list, actual_set, k_val))
        acc["NDCG"].append(numpy_ndcg_at_k(recs_list, actual_set, k_val))
        acc["HitRate"].append(numpy_hit_rate_at_k(recs_list, actual_set, k_val))
        acc["MRR"].append(numpy_mrr_at_k(recs_list, actual_set, k_val))

        if recs_list:
            all_recs_set.update(recs_list)
            item_counts.update(recs_list)
            if len(recs_list) >= 2 and all_item_vectors_normalized is not None:
                valid_ids = [r for r in recs_list if 0 <= r < all_item_vectors_normalized.size(0)]
                if len(valid_ids) >= 2:
                    vecs = all_item_vectors_normalized[torch.tensor(valid_ids, device=all_item_vectors_normalized.device)]
                    if vecs.shape[0] >= 2: # Check again
                        sim = torch.matmul(vecs, vecs.T)
                        iu = torch.triu_indices(*sim.shape, offset=1)
                        if iu.numel()>0: acc["ILD_scores"].append(torch.mean(1.0-sim[iu[0],iu[1]]).item())
                        else: acc["ILD_scores"].append(0.0)
                    else: acc["ILD_scores"].append(0.0)
                else: acc["ILD_scores"].append(0.0)
            elif recs_list: acc["ILD_scores"].append(0.0)

            if item_self_information_scores is not None and item_self_information_scores.numel()>0:
                v_nov_recs = [r for r in recs_list if 0 <= r < len(item_self_information_scores)]
                if v_nov_recs: acc["Novelty_scores"].append(torch.mean(item_self_information_scores[torch.tensor(v_nov_recs,dtype=torch.long)]).item())

    final_mets = {**default_mets}
    for metric_key_base, values in acc.items():
        if "_scores" not in metric_key_base:
            final_mets[f"{metric_key_base}@{k_val}"] = np.mean(values) if values else 0.0
        elif metric_key_base == "ILD_scores":
            final_mets[f"ILD@{k_val}"] = np.mean(values) if values else 0.0
        elif metric_key_base == "Novelty_scores":
            final_mets[f"Novelty@{k_val}"] = np.mean(values) if values else 0.0

    final_mets[f"Coverage@{k_val}"] = len(all_recs_set)/num_total_items if num_total_items>0 else 0.0
    if num_total_items > 0 and item_counts:
        counts_arr = np.zeros(num_total_items)
        for item_id_count, cnt_val in item_counts.items(): # Corrected variable name here
            if 0 <= item_id_count < num_total_items: counts_arr[item_id_count] = cnt_val
        final_mets[f"Fairness_Gini@{k_val}"] = gini_coefficient(counts_arr)
    else: # Ensure Gini is 0 if no recs or items
        final_mets[f"Fairness_Gini@{k_val}"] = 0.0
    return {key_str:float(val_num) for key_str,val_num in final_mets.items()}

def calculate_popularity_baseline_metrics(train_df_pos, uids_to_eval, gt_map_segment, num_items, k_val, item_self_info, all_item_vecs_norm, desc):
    default_pop_mets = {f"{m}@{k_val}":0.0 for m in ["Recall","Precision","NDCG","HitRate","MRR","Coverage","ILD","Novelty","Fairness_Gini"]}
    if train_df_pos.empty or num_items == 0: return default_pop_mets
    item_pop = train_df_pos[ITEM_ID_COL].value_counts(); actual_top_k_pop = min(k_val, len(item_pop))
    top_k_pop_ids = item_pop.nlargest(actual_top_k_pop).index.tolist()
    if not hasattr(uids_to_eval, '__iter__') or isinstance(uids_to_eval, (str,bytes)) or not len(uids_to_eval)>0 or not gt_map_segment: return default_pop_mets
    pop_recs_dict = {uid: top_k_pop_ids for uid in uids_to_eval}
    gt_data = [{USER_ID_COL: uid, ITEM_ID_COL: iid} for uid, iids in gt_map_segment.items() if uid in uids_to_eval for iid in iids]
    gt_df_from_map = pd.DataFrame(gt_data) if gt_data else pd.DataFrame(columns=[USER_ID_COL, ITEM_ID_COL])
    return calculate_metrics_for_baseline_recs(pop_recs_dict, gt_df_from_map, num_items, k_val, item_self_info, all_item_vecs_norm, desc)

# Main Execution
def main():
    print(f"Starting evaluation with strategy: {CONFIG['split_strategy']}")
    df_all, df_pos, df_neg, encoders, user_features_df, item_features_df, \
    num_total_users, num_total_items = load_and_preprocess_data()

    if num_total_users == 0 or num_total_items == 0: print("No users/items. Exiting."); return

    user_feat_cols = list(RAW_USER_FEATURE_COLS.values()); item_feat_cols = list(RAW_ITEM_FEATURE_COLS.values())
    for col_map, df_ref, n_total, type_name in [(RAW_USER_FEATURE_COLS, user_features_df, num_total_users, "user"), (RAW_ITEM_FEATURE_COLS, item_features_df, num_total_items, "item")]:
        df_ref = df_ref.reindex(np.arange(n_total)) # Ensure all users/items have a row
        for raw_c, proc_c in col_map.items():
            if proc_c in df_ref.columns:
                 unknown_val = encoders[proc_c].transform(["UNKNOWN"])[0] if proc_c in encoders and hasattr(encoders[proc_c],'classes_') and "UNKNOWN" in encoders[proc_c].classes_ else 0
                 df_ref[proc_c] = df_ref[proc_c].fillna(unknown_val)
        df_ref = df_ref.fillna(0) # Fill any other NaNs, e.g. if a feature col was all NaN
        if type_name == "user": user_features_df = df_ref
        else: item_features_df = df_ref
            
    global_user_feats_t = torch.tensor(user_features_df[user_feat_cols].values, dtype=torch.long).to(device) if user_feat_cols and not user_features_df.empty and all(c in user_features_df for c in user_feat_cols) else torch.empty((num_total_users,0),dtype=torch.long).to(device)
    global_item_feats_t = torch.tensor(item_features_df[item_feat_cols].values, dtype=torch.long).to(device) if item_feat_cols and not item_features_df.empty and all(c in item_features_df for c in item_feat_cols) else torch.empty((num_total_items,0),dtype=torch.long).to(device)

    np.random.seed(CONFIG['random_state'])
    train_df, val_df_gt, test_df_gt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_user_ids, val_user_ids, test_user_ids = np.array([]), np.array([]), np.array([])

    if CONFIG['split_strategy'] == "user_cold_start":
        print("Using user cold-start split.")
        all_unique_user_ids = df_all[USER_ID_COL].unique(); np.random.shuffle(all_unique_user_ids)
        test_split_idx = int(len(all_unique_user_ids) * (1-CONFIG['test_size']))
        train_val_uids, test_user_ids_arr = all_unique_user_ids[:test_split_idx], all_unique_user_ids[test_split_idx:]
        val_split_idx = int(len(train_val_uids) * CONFIG['val_size']) # val_size is prop of train_val_uids
        val_user_ids_arr, train_user_ids_arr = train_val_uids[:val_split_idx], train_val_uids[val_split_idx:]
        
        train_df = df_pos[df_pos[USER_ID_COL].isin(train_user_ids_arr)] if len(train_user_ids_arr)>0 else pd.DataFrame(columns=df_pos.columns)
        val_df_gt = df_pos[df_pos[USER_ID_COL].isin(val_user_ids_arr)] if len(val_user_ids_arr)>0 else pd.DataFrame(columns=df_pos.columns)
        test_df_gt = df_pos[df_pos[USER_ID_COL].isin(test_user_ids_arr)] if len(test_user_ids_arr)>0 else pd.DataFrame(columns=df_pos.columns)
        train_user_ids, val_user_ids, test_user_ids = train_user_ids_arr, val_user_ids_arr, test_user_ids_arr

    elif CONFIG['split_strategy'] == "temporal":
        print("Using temporal split (leave-last-two-out per user).")
        df_pos_sorted = df_pos.sort_values([USER_ID_COL, DATE_COL])
        train_list, val_list, test_list = [], [], []
        for _, group in tqdm(df_pos_sorted.groupby(USER_ID_COL), desc="Splitting data temporally"):
            n = len(group)
            if n >= 3: # Must have at least 3 interactions for train, val, test
                train_list.append(group.iloc[:-2])
                val_list.append(group.iloc[-2:-1])
                test_list.append(group.iloc[-1:])
            elif n == 2: # Train and val only
                train_list.append(group.iloc[:-1])
                val_list.append(group.iloc[-1:])
            elif n == 1: # Train only
                train_list.append(group)
        if train_list: train_df = pd.concat(train_list)
        if val_list: val_df_gt = pd.concat(val_list)
        if test_list: test_df_gt = pd.concat(test_list)
        
        train_user_ids = train_df[USER_ID_COL].unique() if not train_df.empty else np.array([])
        val_user_ids = val_df_gt[USER_ID_COL].unique() if not val_df_gt.empty else np.array([])
        test_user_ids = test_df_gt[USER_ID_COL].unique() if not test_df_gt.empty else np.array([])
        # Ensure all users in val/test also exist in train for true temporal sequential eval
        # This is implicitly handled by how InteractionDataset and evaluate_model work with user features
        # as long as global_user_features_tensor covers all users.
    else: raise ValueError(f"Unknown split_strategy: {CONFIG['split_strategy']}")

    print(f"Data split: Train users: {len(train_user_ids)}, Val users: {len(val_user_ids)}, Test users: {len(test_user_ids)}")
    print(f"Interactions: Train: {len(train_df)}, Val GT: {len(val_df_gt)}, Test GT: {len(test_df_gt)}")

    item_pop_prob = torch.zeros(num_total_items,dtype=torch.float32,device='cpu')
    if not train_df.empty:
        counts = train_df[ITEM_ID_COL].value_counts()
        for item_idx, count_val in counts.items():
            if 0 <= item_idx < num_total_items: item_pop_prob[item_idx] = count_val / len(train_df)
    item_self_info_scores = -torch.log2(item_pop_prob + 1e-9)

    user_to_explicit_negs = {}
    if not df_neg.empty and len(train_user_ids) > 0:
        relevant_train_neg_df = df_neg[df_neg[USER_ID_COL].isin(train_user_ids)]
        if not relevant_train_neg_df.empty:
            user_to_explicit_negs = relevant_train_neg_df.groupby(USER_ID_COL)[ITEM_ID_COL].apply(list).to_dict()
        
    trained_model = None; train_loader = None; val_loader = None
    if not train_df.empty:
        train_dataset = InteractionDataset(train_df, global_user_feats_t, global_item_feats_t)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    else: print("Train data empty. Skipping training.")

    if not val_df_gt.empty: # Use actual val_df_gt if it exists from the split
        val_dataset = InteractionDataset(val_df_gt, global_user_feats_t, global_item_feats_t)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size']*2, shuffle=False, num_workers=0)
    elif len(val_user_ids) > 0 and CONFIG['split_strategy'] == 'user_cold_start': # Shell for user cold start val if no GT
        shell_val_df = pd.DataFrame({USER_ID_COL: val_user_ids, ITEM_ID_COL: -1})
        val_dataset = InteractionDataset(shell_val_df, global_user_feats_t, global_item_feats_t)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size']*2, shuffle=False, num_workers=0)
        print("Using shell validation loader as val_df_gt is empty but val_user_ids exist (user_cold_start).")
    else: print("No validation data or val_user_ids. val_loader not created.")


    if train_loader:
        user_cardinalities = [encoders[idx_col].classes_.size for idx_col in RAW_USER_FEATURE_COLS.values() if idx_col in encoders and hasattr(encoders[idx_col],'classes_')]
        item_cardinalities = [encoders[idx_col].classes_.size for idx_col in RAW_ITEM_FEATURE_COLS.values() if idx_col in encoders and hasattr(encoders[idx_col],'classes_')]
        common_cfg = {"embed_dim_id":CONFIG['embedding_dim_ids'], "embed_dim_feat":CONFIG['embedding_dim_features'], "transformer_nhead":CONFIG['transformer_nhead'], "transformer_nlayers":CONFIG['transformer_nlayers'], "transformer_dim_feedforward":CONFIG['transformer_dim_feedforward'], "out_dim":CONFIG['final_mlp_embed_dim']}
        model = TwoTowerModel(user_tower_config={**common_cfg, "id_dim":num_total_users, "feature_cardinalities":user_cardinalities}, item_tower_config={**common_cfg, "id_dim":num_total_items, "feature_cardinalities":item_cardinalities}).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        trained_model = train_model(model,train_loader,val_loader,optimizer,num_total_items,global_item_feats_t,user_features_df,item_features_df,val_df_gt,user_to_explicit_negs,encoders,item_self_info_scores)
       
        # MODEL SAVING SECTION 
        if trained_model:
            model_save_dir = CONFIG.get('model_save_path', 'saved_models')
            os.makedirs(model_save_dir, exist_ok=True)
            
            # PyTorch state_dict 
            state_dict_path = os.path.join(model_save_dir, "two_tower_recsys_state_dict.pth")
            torch.save(trained_model.state_dict(), state_dict_path)
            print(f"Model state_dict saved to: {state_dict_path}")

            # Exporting individual towers to ONNX for Netron visualization 
            print("\nExporting model towers to ONNX for Netron visualization...")
            trained_model.eval() 

            # Export User Tower
            try:
                user_tower_onnx_path = os.path.join(model_save_dir, "user_tower.onnx")
                # Create dummy inputs for the user tower
                dummy_user_ids = torch.zeros(1, dtype=torch.long).to(device)
                
                # Determine the number of user features expected by the tower
                num_user_features = 0
                if hasattr(trained_model, 'user_tower') and hasattr(trained_model.user_tower, 'user_feat_embeds'):
                    num_user_features = len(trained_model.user_tower.user_feat_embeds)
                elif 'user_cardinalities' in locals() and user_cardinalities: # from earlier in main
                    num_user_features = len(user_cardinalities)

                dummy_user_side_features = torch.zeros(1, num_user_features, dtype=torch.long).to(device) if num_user_features > 0 else torch.empty(1, 0, dtype=torch.long).to(device)

                if hasattr(trained_model, 'user_tower') and callable(trained_model.user_tower):
                    torch.onnx.export(
                        trained_model.user_tower,
                        (dummy_user_ids, dummy_user_side_features),
                        user_tower_onnx_path,
                        input_names=['user_ids', 'user_features'],
                        output_names=['user_embedding'],
                        opset_version=20, 
                        dynamic_axes={
                            'user_ids': {0: 'batch_size'},
                            'user_features': {0: 'batch_size'},
                            'user_embedding': {0: 'batch_size'}
                        }
                    )
                    print(f"User Tower exported to ONNX: {user_tower_onnx_path}")
                else:
                    print("Could not export User Tower: 'user_tower' attribute not found or not callable.")

            except Exception as e:
                print(f"Error exporting User Tower to ONNX: {e}")

            # Export Item Tower 
            try:
                item_tower_onnx_path = os.path.join(model_save_dir, "item_tower.onnx")
                # Create dummy inputs for the item tower
                dummy_item_ids = torch.zeros(1, dtype=torch.long).to(device)

                num_item_features = 0
                if hasattr(trained_model, 'item_tower') and hasattr(trained_model.item_tower, 'item_feat_embeds'):
                    num_item_features = len(trained_model.item_tower.item_feat_embeds)
                elif 'item_cardinalities' in locals() and item_cardinalities: # from earlier in main
                    num_item_features = len(item_cardinalities)
                
                dummy_item_side_features = torch.zeros(1, num_item_features, dtype=torch.long).to(device) if num_item_features > 0 else torch.empty(1, 0, dtype=torch.long).to(device)

                if hasattr(trained_model, 'item_tower') and callable(trained_model.item_tower):
                    torch.onnx.export(
                        trained_model.item_tower,
                        (dummy_item_ids, dummy_item_side_features),
                        item_tower_onnx_path,
                        input_names=['item_ids', 'item_features'],
                        output_names=['item_embedding'],
                        opset_version=20, # Consistent opset version
                        dynamic_axes={
                            'item_ids': {0: 'batch_size'},
                            'item_features': {0: 'batch_size'},
                            'item_embedding': {0: 'batch_size'}
                        }
                    )
                    print(f"Item Tower exported to ONNX: {item_tower_onnx_path}")
                else:
                    print("Could not export Item Tower: 'item_tower' attribute not found or not callable.")

            except Exception as e:
                print(f"Error exporting Item Tower to ONNX: {e}")
  

    all_item_vecs_norm_baseline = None
    if trained_model and num_total_items > 0:
        try:
            trained_model.eval()
            _all_item_ids_t = torch.arange(num_total_items, device=device)
            _item_feat_df_reidx = item_features_df.reindex(np.arange(num_total_items)) # Use global item_features_df
            for _c in item_feat_cols:
                if _c in _item_feat_df_reidx: _item_feat_df_reidx[_c] = _item_feat_df_reidx[_c].fillna(encoders[_c].transform(["UNKNOWN"])[0] if _c in encoders and hasattr(encoders[_c],'classes_') and "UNKNOWN" in encoders[_c].classes_ else 0)
            _valid_cols = [c for c in item_feat_cols if c in _item_feat_df_reidx.columns]
            _all_items_side_t = torch.tensor(_item_feat_df_reidx[_valid_cols].values,dtype=torch.long).to(device) if _valid_cols and not _item_feat_df_reidx.empty else torch.empty((num_total_items,0),dtype=torch.long,device=device)
            _item_vecs = trained_model.item_tower(_all_item_ids_t, _all_items_side_t)
            if _item_vecs is not None and _item_vecs.numel() > 0: all_item_vecs_norm_baseline = F.normalize(_item_vecs, p=2, dim=1).cpu()
        except Exception as e: print(f"Warn: Error getting item vecs for baseline ILD: {e}")
    if all_item_vecs_norm_baseline is None and num_total_items > 0: print("Info: ILD for baselines might be 0.0.")

    print("\n--- Evaluating on Test Set ---")
    if len(test_user_ids) > 0 and not test_df_gt.empty: 
        segments = {"All Test Users": test_user_ids} # test_user_ids are from the chosen split
        # Overall Cold Start: users with no positive interactions in the *entire original df_pos*
        all_pos_users_global = df_pos[USER_ID_COL].unique()
        overall_cold_start_global_ids = np.setdiff1d(df_all[USER_ID_COL].unique(), all_pos_users_global)
        segments["Overall Cold Start (0 Positive Interactions Globally)"] = np.intersect1d(test_user_ids, overall_cold_start_global_ids)
        
        # Training Cold Start: test users not in train_df (specific to current split)
        train_users_in_train_df = train_df[USER_ID_COL].unique() if not train_df.empty else np.array([])
        training_cold_start_test_ids = np.setdiff1d(test_user_ids, train_users_in_train_df)
        segments["Training Cold Start (0 Positive Train Interactions this Split)"] = training_cold_start_test_ids

        user_inter_counts_global = get_user_interaction_counts(df_pos) # Based on global positive interactions
        interaction_bands = {
                            "1-2 Pos Interactions Globally": user_inter_counts_global[user_inter_counts_global.isin([1, 2])].index,
                            "3-5 Pos Interactions Globally": user_inter_counts_global[user_inter_counts_global.isin([3, 4, 5])].index,
                            "6-25 Pos Interactions Globally": user_inter_counts_global[(user_inter_counts_global > 5) & (user_inter_counts_global <= 25)].index,
                            ">25 Pos Interactions Globally": user_inter_counts_global[user_inter_counts_global > 25].index
                        }

        for band_name, band_uids in interaction_bands.items(): segments[band_name] = np.intersect1d(test_user_ids, band_uids.to_numpy())

        for seg_name, seg_uids in segments.items():
            seg_uids_unique = np.unique(seg_uids)
            if len(seg_uids_unique) == 0: print(f"\n--- Segment: {seg_name} - No test users. Skipping."); continue
            print(f"\n--- Segment: {seg_name} ({len(seg_uids_unique)} users) ---")
            
            current_segment_test_df_gt = test_df_gt[test_df_gt[USER_ID_COL].isin(seg_uids_unique)]
            if current_segment_test_df_gt.empty and not (seg_name.startswith("Overall Cold Start") and CONFIG['split_strategy'] == 'user_cold_start'): # Overall cold start may have no GT
                 print(f"Segment {seg_name} has no ground truth in test_df_gt. Metrics might be 0 or skip.")

            seg_loader_df = pd.DataFrame({USER_ID_COL: seg_uids_unique, ITEM_ID_COL: -1}) # Shell df for users in segment
            seg_test_dataset = InteractionDataset(seg_loader_df, global_user_feats_t, global_item_feats_t)
            seg_test_loader = DataLoader(seg_test_dataset, batch_size=CONFIG['batch_size']*2, shuffle=False, num_workers=0)

            gt_map_seg = current_segment_test_df_gt.groupby(USER_ID_COL)[ITEM_ID_COL].apply(set).to_dict() if not current_segment_test_df_gt.empty else {}

            if trained_model and seg_test_loader:
                print(f"Evaluating Main Model for {seg_name}...")
                main_mets = evaluate_model(trained_model, seg_test_loader, num_total_items,item_features_df,user_features_df,current_segment_test_df_gt,encoders,item_self_info_scores,desc=f"Test Main ({seg_name})")
                for m_name, m_val in main_mets.items(): print(f"  {m_name}: {m_val:.4f}")
            
            # Baselines
            rand_recs = generate_random_recommendations(seg_uids_unique, num_total_items, CONFIG['top_k'])
            rand_mets = calculate_metrics_for_baseline_recs(rand_recs, current_segment_test_df_gt, num_total_items, CONFIG['top_k'], item_self_info_scores, all_item_vecs_norm_baseline, f"Random ({seg_name})")
            print(f"Random Baseline ({seg_name}) Metrics:"); [print(f"  {k}: {v:.4f}") for k,v in rand_mets.items()]
            bought_recs = generate_previously_bought_recommendations(seg_uids_unique, df_pos, CONFIG['top_k'])
            bought_mets = calculate_metrics_for_baseline_recs(bought_recs, current_segment_test_df_gt, num_total_items, CONFIG['top_k'], item_self_info_scores, all_item_vecs_norm_baseline, f"PrevBought ({seg_name})")
            print(f"Prev. Bought Baseline ({seg_name}) Metrics:"); [print(f"  {k}: {v:.4f}") for k,v in bought_mets.items()]

            if not train_df.empty: # Popularity uses train_df
                pop_mets = calculate_popularity_baseline_metrics(train_df, seg_uids_unique, gt_map_seg, num_total_items, CONFIG['top_k'], item_self_info_scores, all_item_vecs_norm_baseline, f"Popularity ({seg_name})")
                print(f"Popularity Baseline ({seg_name}) Metrics:"); [print(f"  {k}: {v:.4f}") for k,v in pop_mets.items()]
            else: print(f"Skipping Popularity for {seg_name} as train_df is empty.")
    else: print("Test set (test_user_ids or test_df_gt) is empty. Skipping final evaluation.")
    print("\nDone.")

if __name__ == "__main__":
    main()