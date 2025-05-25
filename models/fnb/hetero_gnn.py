import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.utils import degree 
from tqdm import tqdm
import torchmetrics.functional.retrieval as tm_functional

# Configuration
CONFIG = {
    "data_path": "../../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv",
    "epochs": 50,
    "batch_size_bpr": 2048,
    "hidden_channels": 64,  # Channels for SAGEConv hidden layers and output embeddings
    "num_gnn_layers": 2,    # Number of GNN layers in HeteroConv
    "learning_rate": 0.001,
    "weight_decay_gnn": 1e-5, # Weight decay for GNN layers
    "weight_decay_embed": 1e-4, # Weight decay for initial feature projection
    "top_k": 10,
    "test_split_ratio": 0.2,
    "random_seed": 42,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Data Loading and Feature Preprocessing
def load_and_preprocess_hetero_data(file_path, random_seed):
    print(f"Loading data from: {file_path}")
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        return None, None, None, None, None, None

    feature_cols_to_clean = ['item_type', 'item_descrip', 'segment', 'beh_segment', 'active_ind']
    for col in feature_cols_to_clean:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna("UNKNOWN").astype(str)
        else:
            print(f"Warning: Feature column '{col}' not found. It will be ignored or treated as UNKNOWN.")
            df_raw[col] = "UNKNOWN"

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df_raw['user_id_idx'] = user_encoder.fit_transform(df_raw['idcol'].astype(str))
    df_raw['item_id_idx'] = item_encoder.fit_transform(df_raw['item'].astype(str))

    num_users = df_raw['user_id_idx'].nunique()
    num_items = df_raw['item_id_idx'].nunique()
    print(f"Number of unique users: {num_users}, Number of unique items: {num_items}")

    df_user_features_grouped = df_raw.groupby('user_id_idx').agg({
        'segment': lambda x: list(set(x)),
        'beh_segment': lambda x: list(set(x)),
        'active_ind': lambda x: list(set(x))
    }).reset_index()
    
    # Create a DataFrame with all user_id_idx from 0 to num_users-1 to ensure correct alignment
    all_user_indices_df = pd.DataFrame({'user_id_idx': range(num_users)})
    df_user_features = pd.merge(all_user_indices_df, df_user_features_grouped, on='user_id_idx', how='left')
    # Fill NaN lists with lists containing 'UNKNOWN' for users who might not have specific features after left merge
    for col in ['segment', 'beh_segment', 'active_ind']:
         df_user_features[col] = df_user_features[col].apply(lambda x: x if isinstance(x, list) else ['UNKNOWN'])


    user_feature_tuples = []
    for _, row in df_user_features.iterrows():
        features = []
        features.extend([f"segment:{s}" for s in row['segment']])
        features.extend([f"beh_segment:{bs}" for bs in row['beh_segment']])
        features.extend([f"active_ind:{ai}" for ai in row['active_ind']])
        user_feature_tuples.append(features)
    
    mlb_user = MultiLabelBinarizer()
    user_features_encoded = mlb_user.fit_transform(user_feature_tuples)
    user_feat_dim = user_features_encoded.shape[1]
    print(f"User feature matrix shape: {user_features_encoded.shape}, Num unique user feature tags: {len(mlb_user.classes_)}")

    df_item_features_grouped = df_raw.groupby('item_id_idx').agg({
        'item_type': lambda x: list(set(x)),
        'item_descrip': lambda x: list(set(x))
    }).reset_index()

    all_item_indices_df = pd.DataFrame({'item_id_idx': range(num_items)})
    df_item_features = pd.merge(all_item_indices_df, df_item_features_grouped, on='item_id_idx', how='left')
    for col in ['item_type', 'item_descrip']:
         df_item_features[col] = df_item_features[col].apply(lambda x: x if isinstance(x, list) else ['UNKNOWN'])

    item_feature_tuples = []
    for _, row in df_item_features.iterrows():
        features = []
        features.extend([f"item_type:{it}" for it in row['item_type']])
        features.extend([f"item_descrip:{idsc}" for idsc in row['item_descrip']])
        item_feature_tuples.append(features)

    mlb_item = MultiLabelBinarizer()
    item_features_encoded = mlb_item.fit_transform(item_feature_tuples)
    item_feat_dim = item_features_encoded.shape[1]
    print(f"Item feature matrix shape: {item_features_encoded.shape}, Num unique item feature tags: {len(mlb_item.classes_)}")

    df_positive = df_raw[df_raw['interaction'].isin(['CLICK', 'CHECKOUT'])].copy()
    if df_positive.empty:
        print("No positive interactions (CLICK, CHECKOUT) found.")
        return None, None, None, None, None, None
    print(f"Found {len(df_positive)} positive interactions for edges.")

    data = HeteroData()
    data['user'].x = torch.tensor(user_features_encoded, dtype=torch.float)
    data['item'].x = torch.tensor(item_features_encoded, dtype=torch.float)

    interactions = df_positive[['user_id_idx', 'item_id_idx']].values
    train_interactions, test_interactions = sklearn_train_test_split(
        interactions, test_size=CONFIG['test_split_ratio'], random_state=random_seed
    )
    
    train_edge_src = torch.tensor(train_interactions[:, 0], dtype=torch.long)
    train_edge_dst = torch.tensor(train_interactions[:, 1], dtype=torch.long)
    
    data['user', 'interacts_with', 'item'].edge_index = torch.stack([train_edge_src, train_edge_dst], dim=0)
    data['item', 'interacted_by', 'user'].edge_index = torch.stack([train_edge_dst, train_edge_src], dim=0)

    print(f"Number of training interaction edges: {data['user', 'interacts_with', 'item'].edge_index.shape[1]}")

    train_user_item_set = set((u, i) for u, i in train_interactions)
    test_user_to_items = {}
    for u, i in test_interactions:
        if u not in test_user_to_items:
            test_user_to_items[u] = set()
        test_user_to_items[u].add(i)
    
    return data, num_users, num_items, train_user_item_set, test_user_to_items, \
           (mlb_user, mlb_item)


# Heterogeneous GNN Model 
class HeteroGNN(nn.Module):
    def __init__(self, hetero_metadata, user_feat_dim, item_feat_dim, hidden_channels, num_layers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.user_lin = Linear(user_feat_dim, hidden_channels)
        self.item_lin = Linear(item_feat_dim, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'interacts_with', 'item'): SAGEConv((-1, -1), hidden_channels), 
                ('item', 'interacted_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        
    def forward(self, x_dict, edge_index_dict):
        x_dict_user = self.user_lin(x_dict['user']).relu()
        x_dict_item = self.item_lin(x_dict['item']).relu()
        
        current_x_dict = {'user': x_dict_user, 'item': x_dict_item}

        for conv_layer in self.convs: 
            current_x_dict = conv_layer(current_x_dict, edge_index_dict)
            for node_type, x in current_x_dict.items():
                 current_x_dict[node_type] = x.relu()

        return current_x_dict['user'], current_x_dict['item']

    def bpr_loss(self, user_embeds_batch, all_item_embeds, pos_item_indices, neg_item_indices): 
        pos_item_embeds_batch = all_item_embeds[pos_item_indices]
        neg_item_embeds_batch = all_item_embeds[neg_item_indices]

        pos_scores = torch.sum(user_embeds_batch * pos_item_embeds_batch, dim=1)
        neg_scores = torch.sum(user_embeds_batch * neg_item_embeds_batch, dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss

    def gnn_regularization_loss(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'convs' in name and 'weight' in name: 
                reg_loss += param.norm(2).pow(2)
        return CONFIG['weight_decay_gnn'] * reg_loss / 2.0
    
    def embedding_regularization_loss(self): 
        reg_loss = 0
        if hasattr(self, 'user_lin'): # Check if attribute exists before accessing
            reg_loss += self.user_lin.weight.norm(2).pow(2)
        if hasattr(self, 'item_lin'): # Check if attribute exists
            reg_loss += self.item_lin.weight.norm(2).pow(2)
        return CONFIG['weight_decay_embed'] * reg_loss / 2.0


# BPR Data Sampler (Modified for Hetero)
def sample_bpr_hetero_batch(train_user_item_set, num_users, num_items, batch_size):
    train_interactions_list = list(train_user_item_set) 
    if not train_interactions_list:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    actual_batch_size = min(batch_size, len(train_interactions_list))
    indices = np.random.choice(len(train_interactions_list), size=actual_batch_size, 
                               replace=len(train_interactions_list) < actual_batch_size)
    
    users, pos_items, neg_items = [], [], []
    for idx in indices:
        u, pos_i = train_interactions_list[idx]
        users.append(u)
        pos_items.append(pos_i)
        
        neg_j = np.random.randint(0, num_items)
        while (u, neg_j) in train_user_item_set: 
            neg_j = np.random.randint(0, num_items)
        neg_items.append(neg_j)
        
    return torch.tensor(users, dtype=torch.long), \
           torch.tensor(pos_items, dtype=torch.long), \
           torch.tensor(neg_items, dtype=torch.long)


# Evaluation (Modified for Hetero)
def evaluate_hetero_model(model, hetero_data, test_user_to_items, train_user_item_set, num_users, num_items, k, device):
    model.eval()
    
    x_dict_dev = {key: val.to(device) for key, val in hetero_data.x_dict.items()}
    # Ensure all edge types present in metadata are also in edge_index_dict before moving to device
    edge_index_dict_dev = {}
    for edge_type in hetero_data.edge_types:
        if edge_type in hetero_data.edge_index_dict:
             edge_index_dict_dev[edge_type] = hetero_data.edge_index_dict[edge_type].to(device)
        # else:
            # print(f"Warning: Edge type {edge_type} from metadata not found in edge_index_dict during evaluation.")


    user_final_embeds_all, item_final_embeds_all = model(x_dict_dev, edge_index_dict_dev)
    user_final_embeds_all = user_final_embeds_all.cpu()
    item_final_embeds_all = item_final_embeds_all.cpu()

    all_precisions, all_recalls, all_ndcgs = [], [], []
    
    test_users_with_interactions = [u for u in test_user_to_items.keys() if u < num_users] 
    if not test_users_with_interactions:
        print("No valid users in the test set for evaluation.")
        return 0.0, 0.0, 0.0

    for user_idx in tqdm(test_users_with_interactions, desc="Evaluating", leave=False, ncols=80):
        if user_idx >= user_final_embeds_all.shape[0]: 
            print(f"Warning: user_idx {user_idx} out of bounds for user_final_embeds_all. Skipping.")
            continue
        user_embed = user_final_embeds_all[user_idx].unsqueeze(0) 
        
        scores = torch.matmul(user_embed, item_final_embeds_all.T).squeeze() 
        
        try:
            items_to_exclude = [item_j for u, item_j in train_user_item_set if u == user_idx]
            if items_to_exclude:
                valid_items_to_exclude = [item for item in items_to_exclude if item < scores.shape[0]]
                if valid_items_to_exclude:
                    scores[valid_items_to_exclude] = -np.inf
        except IndexError as e:
            print(f"IndexError while excluding items for user {user_idx}. Items to exclude: {items_to_exclude}. Scores shape: {scores.shape}. Error: {e}")
            pass

        if scores.ndim == 0: # Handle case where scores might become a 0-dim tensor (e.g. only 1 item)
            scores_for_topk = scores.unsqueeze(0) if scores.numel() > 0 else torch.tensor([-np.inf])
        else:
            scores_for_topk = scores
        
        # Ensure k is not greater than the number of available scores
        current_k = min(k, scores_for_topk.shape[0])
        if current_k == 0 : # No items to recommend from
            recommended_items = []
        else:
            _, top_k_indices = torch.topk(scores_for_topk, k=current_k) 
            recommended_items = top_k_indices.tolist()

        actual_items = test_user_to_items.get(user_idx, set())
        if not actual_items: 
            continue

        hits = len(set(recommended_items) & actual_items)
        precision = hits / k if k > 0 else 0.0 # Precision is calculated over original K
        recall = hits / len(actual_items) if len(actual_items) > 0 else 0.0
        all_precisions.append(precision)
        all_recalls.append(recall)

        target_tensor = torch.zeros(num_items, dtype=torch.bool) # Target tensor should be full size
        if actual_items: 
            valid_actual_items = [item for item in list(actual_items) if item < num_items] 
            if valid_actual_items:
                 target_tensor[valid_actual_items] = True
        
        # Ensure scores for NDCG are also full size, matching target_tensor
        full_scores_for_ndcg = torch.full((num_items,), -np.inf, dtype=scores.dtype)
        if scores.numel() > 0 and scores.ndim > 0 : # if scores is not empty and has dimensions
            valid_score_indices = torch.arange(scores.shape[0])
            full_scores_for_ndcg[valid_score_indices] = scores[valid_score_indices]
        elif scores.numel() == 1 and scores.ndim == 0: # if scores is a single scalar value
             full_scores_for_ndcg[0] = scores # place it at the first position, assuming it's for item 0

        ndcg_val = tm_functional.retrieval_normalized_dcg(
            full_scores_for_ndcg.unsqueeze(0), 
            target_tensor.unsqueeze(0), 
            top_k=k
        )
        all_ndcgs.append(ndcg_val.item())

    mean_precision = np.mean(all_precisions) if all_precisions else 0.0
    mean_recall = np.mean(all_recalls) if all_recalls else 0.0
    mean_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0
    
    return mean_precision, mean_recall, mean_ndcg


# Main Training and Evaluation Script 
if __name__ == "__main__":
    print(f"Using device: {CONFIG['device']}")
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    if CONFIG['device'].type == 'cuda':
        torch.cuda.manual_seed(CONFIG['random_seed'])

    hetero_data, num_users, num_items, \
    train_user_item_set, test_user_to_items, \
    feature_binarizers = load_and_preprocess_hetero_data(CONFIG['data_path'], CONFIG['random_seed'])

    if hetero_data is None: 
        exit()

    user_feat_dim = hetero_data['user'].x.shape[1]
    item_feat_dim = hetero_data['item'].x.shape[1]

    model = HeteroGNN(
        hetero_metadata=hetero_data.metadata(), 
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        hidden_channels=CONFIG['hidden_channels'],
        num_layers=CONFIG['num_gnn_layers']
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    print("Starting HeteroGNN training...")
    for epoch in range(CONFIG['epochs']):
        model.train() 
        total_epoch_loss = 0.0
        total_bpr_loss = 0.0
        total_reg_loss = 0.0
        
        if not train_user_item_set: 
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} - No training interactions. Skipping BPR sampling.")
            num_bpr_batches = 0
        else:
            num_bpr_batches = max(1, len(train_user_item_set) // CONFIG['batch_size_bpr'])

        epoch_iterator = tqdm(range(num_bpr_batches), desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", ncols=100)
        
        x_dict_dev = {key: val.to(CONFIG['device']) for key, val in hetero_data.x_dict.items()}
        edge_index_dict_dev = {key: val.to(CONFIG['device']) for key, val in hetero_data.edge_index_dict.items()}

        for batch_idx in epoch_iterator:
            optimizer.zero_grad()
            
            users_b, pos_items_b, neg_items_b = sample_bpr_hetero_batch(
                train_user_item_set, num_users, num_items, CONFIG['batch_size_bpr']
            )
            if users_b.numel() == 0: 
                continue

            users_b, pos_items_b, neg_items_b = users_b.to(CONFIG['device']), \
                                                pos_items_b.to(CONFIG['device']), \
                                                neg_items_b.to(CONFIG['device'])

            all_user_embeds, all_item_embeds = model(x_dict_dev, edge_index_dict_dev)
            
            user_embeds_batch = all_user_embeds[users_b]
            
            bpr_loss = model.bpr_loss(user_embeds_batch, all_item_embeds, pos_items_b, neg_items_b)
            gnn_reg_loss = model.gnn_regularization_loss() 
            emb_reg_loss = model.embedding_regularization_loss()
            reg_loss = gnn_reg_loss + emb_reg_loss
            
            loss = bpr_loss + reg_loss
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item() 
            
            epoch_iterator.set_postfix({
                "Total Loss": f"{loss.item():.4f}",
                "BPR Loss": f"{bpr_loss.item():.4f}",
                "Reg Loss": f"{reg_loss.item():.4f}"
            })

        avg_total_loss = total_epoch_loss / num_bpr_batches if num_bpr_batches > 0 else 0
        avg_bpr_loss = total_bpr_loss / num_bpr_batches if num_bpr_batches > 0 else 0
        avg_reg_loss = total_reg_loss / num_bpr_batches if num_bpr_batches > 0 else 0
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Avg Total Loss: {avg_total_loss:.4f}, Avg BPR Loss: {avg_bpr_loss:.4f}, Avg Reg Loss: {avg_reg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == CONFIG['epochs']: 
            print(f"\n--- Evaluating at Epoch {epoch+1} ---")
            precision, recall, ndcg = evaluate_hetero_model(
                model, hetero_data, test_user_to_items, train_user_item_set, 
                num_users, num_items, CONFIG['top_k'], CONFIG['device'] 
            )
            print(f"Precision@{CONFIG['top_k']}: {precision:.4f}")
            print(f"Recall@{CONFIG['top_k']}: {recall:.4f}")
            print(f"NDCG@{CONFIG['top_k']}: {ndcg:.4f}\n")

    print("Training finished.")

    print("\n--- Final Model Evaluation ---")
    precision, recall, ndcg = evaluate_hetero_model(
        model, hetero_data, test_user_to_items, train_user_item_set,
        num_users, num_items, CONFIG['top_k'], CONFIG['device'] 
    )
    print(f"Final Precision@{CONFIG['top_k']}: {precision:.4f}")
    print(f"Final Recall@{CONFIG['top_k']}: {recall:.4f}")
    print(f"Final NDCG@{CONFIG['top_k']}: {ndcg:.4f}")
