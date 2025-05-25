import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as sklearn_train_test_split # To avoid conflict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, to_undirected
from tqdm import tqdm
import torchmetrics.functional.retrieval as tm_functional # For NDCG

# Configuration 
CONFIG = {
    "data_path": "../../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv",
    "epochs": 50, # Number of epochs for training
    "batch_size_bpr": 2048, # Batch size for BPR loss calculation
    "embedding_dim": 64,    # Dimensionality of user/item embeddings
    "num_layers": 3,        # Number of LightGCN layers
    "learning_rate": 0.001,
    "weight_decay": 1e-4,   # L2 regularization for embeddings
    "top_k": 10,            # For evaluation metrics (Precision@K, Recall@K, NDCG@K)
    "test_split_ratio": 0.2, # Proportion of interactions for testing
    "random_seed": 42,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Data Loading and Preprocessing 
def load_and_preprocess_data(file_path, random_seed):
    """
    Loads data, preprocesses it for LightGCN, and creates train/test splits.
    """
    print(f"Loading data from: {file_path}")
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return None, None, None, None, None, None, None, None # Added one None for item_encoder

    # Use original string IDs for mapping
    df_raw['user_id_str'] = df_raw['idcol'].astype(str)
    df_raw['item_id_str'] = df_raw['item'].astype(str)

    # Filter for positive interactions
    df_positive = df_raw[df_raw['interaction'].isin(['CLICK', 'CHECKOUT'])].copy()
    if df_positive.empty:
        print("No positive interactions (CLICK, CHECKOUT) found. Exiting.")
        return None, None, None, None, None, None, None, None # Added one None

    print(f"Found {len(df_positive)} positive interactions.")

    # Encode user and item IDs to 0-indexed integers
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df_positive['user_id_idx'] = user_encoder.fit_transform(df_positive['user_id_str'])
    df_positive['item_id_idx'] = item_encoder.fit_transform(df_positive['item_id_str'])

    num_users = df_positive['user_id_idx'].nunique()
    num_items = df_positive['item_id_idx'].nunique()

    print(f"Number of unique users: {num_users}")
    print(f"Number of unique items: {num_items}")


    # Create a list of (user_idx, item_idx) pairs
    interactions = df_positive[['user_id_idx', 'item_id_idx']].values
    
    # Split interactions into train and test sets
    train_interactions_list, test_interactions_list = sklearn_train_test_split(
        interactions,
        test_size=CONFIG['test_split_ratio'],
        random_state=random_seed
        # stratify=df_positive['user_id_idx'] # Removed stratification
    )

    print(f"Number of training interactions: {len(train_interactions_list)}")
    print(f"Number of testing interactions: {len(test_interactions_list)}")
    

    u_nodes_all = torch.tensor(df_positive['user_id_idx'].values, dtype=torch.long)
    i_nodes_all = torch.tensor(df_positive['item_id_idx'].values, dtype=torch.long) + num_users # Offset item indices

    edge_index_u_i = torch.stack([u_nodes_all, i_nodes_all])
    edge_index_i_u = torch.stack([i_nodes_all, u_nodes_all]) # For undirected nature
    
    edge_index = torch.cat([edge_index_u_i, edge_index_i_u], dim=1)
    
    # Store train interactions as a set for efficient lookup during negative sampling
    train_user_item_set = set((u,i) for u,i in train_interactions_list)
    
    # For test set, create a dictionary: user -> set of test items
    test_user_to_items = {}
    for u, i in test_interactions_list:
        if u not in test_user_to_items:
            test_user_to_items[u] = set()
        test_user_to_items[u].add(i)

    return num_users, num_items, edge_index, \
           train_interactions_list, train_user_item_set, test_user_to_items, \
           user_encoder, item_encoder


# LightGCN Model 
class LightGCNConv(MessagePassing):
    """LightGCN Convolution Layer."""
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr='add') # 'add' aggregation

    def forward(self, x, edge_index):
        # x: node embeddings [N, D]
        # edge_index: graph connectivity [2, E]

        row, col = edge_index
        # Calculate degree for normalization (symmetric normalization)
        deg_col = degree(col, x.size(0), dtype=x.dtype) # In-degree for column nodes
        deg_row = degree(row, x.size(0), dtype=x.dtype) # Out-degree for row nodes (needed for symmetric norm)

        deg_col_inv_sqrt = deg_col.pow(-0.5)
        deg_row_inv_sqrt = deg_row.pow(-0.5)

        # Replace inf with 0 for isolated nodes
        deg_col_inv_sqrt[deg_col_inv_sqrt == float('inf')] = 0
        deg_row_inv_sqrt[deg_row_inv_sqrt == float('inf')] = 0
        
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] # Original
        norm = deg_row_inv_sqrt[row] * deg_col_inv_sqrt[col] # Symmetric normalization

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j: embeddings of neighboring nodes [E, D]
        # norm: normalization coefficients [E, 1]
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, weight_decay):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.weight_decay = weight_decay # For L2 regularization on embeddings

        # User and Item embeddings
        # Total nodes = num_users + num_items
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight) # Xavier initialization

        # LightGCN convolution layers
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

    def forward(self, edge_index):
        # Get initial embeddings
        x = self.embedding.weight
        all_embeddings = [x] # Store embeddings from all layers

        # Propagate through layers
        current_x = x
        for i in range(self.num_layers):
            current_x = self.convs[i](current_x, edge_index)
            all_embeddings.append(current_x)

        # Final embeddings are the mean of embeddings from all layers (including initial)
        final_embeddings_stack = torch.stack(all_embeddings, dim=0)
        final_embeddings = torch.mean(final_embeddings_stack, dim=0)

        user_embeds = final_embeddings[:self.num_users]
        item_embeds = final_embeddings[self.num_users:] # Corrected slicing
        
        return user_embeds, item_embeds

    def bpr_loss(self, user_embeds_batch, all_item_embeds, pos_item_indices, neg_item_indices):
        """
        Calculates BPR loss.
        user_embeds_batch: Embeddings for the users in the batch [batch_size, D]
        all_item_embeds: All item embeddings from the model [num_items, D]
        pos_item_indices: Positive item indices for these users [batch_size]
        neg_item_indices: Negative item indices for these users [batch_size]
        """
        pos_item_embeds_batch = all_item_embeds[pos_item_indices] # [batch_size, D]
        neg_item_embeds_batch = all_item_embeds[neg_item_indices] # [batch_size, D]

        pos_scores = torch.sum(user_embeds_batch * pos_item_embeds_batch, dim=1) # [batch_size]
        neg_scores = torch.sum(user_embeds_batch * neg_item_embeds_batch, dim=1) # [batch_size]

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss

    def regularization_loss(self):
        """ L2 regularization on initial embeddings (layer 0) """
        initial_embeds = self.embedding.weight
        return self.weight_decay * initial_embeds.norm(2).pow(2) / 2.0


# BPR Data Sampler 
def sample_bpr_batch(train_interactions_list, num_items, train_user_item_set, batch_size):
    """
    Samples a batch of (user, positive_item, negative_item) for BPR loss.
    train_interactions_list: List of (user_idx, item_idx) tuples for training.
    num_items: Total number of unique items.
    train_user_item_set: Set of (user,item) training interactions for fast negative sampling.
    batch_size: Number of samples in the batch.
    """
    if train_interactions_list.size == 0: # Handle empty array: 
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    actual_batch_size = min(batch_size, len(train_interactions_list))
    indices = np.random.choice(len(train_interactions_list), size=actual_batch_size, replace=False)
    batch = [train_interactions_list[i] for i in indices]
    
    users, pos_items, neg_items = [], [], []
    for u, pos_i in batch:
        users.append(u)
        pos_items.append(pos_i)
        
        neg_j = np.random.randint(0, num_items)
        while (u, neg_j) in train_user_item_set: 
            neg_j = np.random.randint(0, num_items)
        neg_items.append(neg_j)
        
    return torch.tensor(users, dtype=torch.long), \
           torch.tensor(pos_items, dtype=torch.long), \
           torch.tensor(neg_items, dtype=torch.long)


# Evaluation 
@torch.no_grad() 
def evaluate_model(model, edge_index, test_user_to_items, train_user_item_set, num_users, num_items, k, device):
    """
    Evaluates the model on the test set.
    test_user_to_items: Dict {user_idx: set_of_test_item_indices}
    train_user_item_set: Set of (user,item) training interactions to exclude from recs.
    """
    model.eval() 
    
    user_final_embeds_all, item_final_embeds_all = model(edge_index.to(device))
    user_final_embeds_all = user_final_embeds_all.cpu()
    item_final_embeds_all = item_final_embeds_all.cpu()

    all_precisions, all_recalls, all_ndcgs = [], [], []
    
    test_users_with_interactions = [u for u in test_user_to_items.keys() if u < num_users]
    if not test_users_with_interactions:
        print("No valid users in the test set for evaluation.")
        return 0.0, 0.0, 0.0

    for user_idx in tqdm(test_users_with_interactions, desc="Evaluating", leave=False, ncols=80):
        user_embed = user_final_embeds_all[user_idx].unsqueeze(0) 
        
        scores = torch.matmul(user_embed, item_final_embeds_all.T).squeeze() 
        
        # Exclude items the user interacted with in the training set
        try:
            # Get items interacted by user_idx in training set
            # train_user_item_set contains (user_idx, item_idx)
            items_to_exclude = [item_j for u, item_j in train_user_item_set if u == user_idx]
            if items_to_exclude:
                scores[items_to_exclude] = -np.inf
        except IndexError as e:
            print(f"IndexError while excluding items for user {user_idx}. Items to exclude: {items_to_exclude}. Scores shape: {scores.shape}. Error: {e}")
            pass


        _, top_k_indices = torch.topk(scores, k=min(k, num_items)) 
        recommended_items = top_k_indices.tolist()
        
        actual_items = test_user_to_items.get(user_idx, set())
        if not actual_items: 
            continue

        hits = len(set(recommended_items) & actual_items)
        precision = hits / k if k > 0 else 0.0
        recall = hits / len(actual_items) if len(actual_items) > 0 else 0.0
        all_precisions.append(precision)
        all_recalls.append(recall)

        target_tensor = torch.zeros(num_items, dtype=torch.bool)
        if actual_items: 
            valid_actual_items = [item for item in list(actual_items) if item < num_items] 
            if valid_actual_items:
                 target_tensor[valid_actual_items] = True
        
      
        ndcg_val = tm_functional.retrieval_normalized_dcg(
            scores.unsqueeze(0), 
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

    num_users, num_items, edge_index, \
    train_interactions_list, train_user_item_set, test_user_to_items, \
    user_encoder, item_encoder = load_and_preprocess_data(CONFIG['data_path'], CONFIG['random_seed'])

    if num_users is None: 
        exit()

    edge_index = edge_index.to(CONFIG['device'])

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=CONFIG['embedding_dim'],
        num_layers=CONFIG['num_layers'],
        weight_decay=CONFIG['weight_decay']
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    print("Starting training...")
    for epoch in range(CONFIG['epochs']):
        model.train() 
        total_bpr_loss = 0.0
        total_reg_loss = 0.0
        
        num_bpr_batches = len(train_interactions_list) // CONFIG['batch_size_bpr']
        if num_bpr_batches == 0 and len(train_interactions_list) > 0 : # Ensure at least one batch if data exists
             num_bpr_batches = 1
        elif len(train_interactions_list) == 0:
            print("Warning: No training interactions available for BPR sampling. Skipping epoch.")
            continue


        epoch_iterator = tqdm(range(num_bpr_batches), desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", ncols=100)
        for batch_idx in epoch_iterator:
            optimizer.zero_grad()
            
            users_b, pos_items_b, neg_items_b = sample_bpr_batch(
                train_interactions_list, num_items, train_user_item_set, CONFIG['batch_size_bpr']
            )
            if users_b.numel() == 0: # Skip if batch is empty
                continue

            users_b, pos_items_b, neg_items_b = users_b.to(CONFIG['device']), \
                                                pos_items_b.to(CONFIG['device']), \
                                                neg_items_b.to(CONFIG['device'])

            all_user_embeds, all_item_embeds = model(edge_index)
            user_embeds_batch = all_user_embeds[users_b]
            
            bpr_loss = model.bpr_loss(user_embeds_batch, all_item_embeds, pos_items_b, neg_items_b)
            reg_loss = model.regularization_loss() # Call without arguments
            
            loss = bpr_loss + reg_loss
            loss.backward()
            optimizer.step()
            
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()
            
            epoch_iterator.set_postfix({
                "BPR Loss": f"{bpr_loss.item():.4f}",
                "Reg Loss": f"{reg_loss.item():.4f}"
            })

        avg_bpr_loss = total_bpr_loss / num_bpr_batches if num_bpr_batches > 0 else 0
        avg_reg_loss = total_reg_loss / num_bpr_batches if num_bpr_batches > 0 else 0
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Avg BPR Loss: {avg_bpr_loss:.4f}, Avg Reg Loss: {avg_reg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == CONFIG['epochs']: 
            print(f"\n--- Evaluating at Epoch {epoch+1} ---")
            precision, recall, ndcg = evaluate_model(
                model, edge_index, test_user_to_items, train_user_item_set, 
                num_users, num_items, CONFIG['top_k'], CONFIG['device'] # Pass num_users
            )
            print(f"Precision@{CONFIG['top_k']}: {precision:.4f}")
            print(f"Recall@{CONFIG['top_k']}: {recall:.4f}")
            print(f"NDCG@{CONFIG['top_k']}: {ndcg:.4f}\n")

    print("Training finished.")

    print("\n--- Final Model Evaluation ---")
    precision, recall, ndcg = evaluate_model(
        model, edge_index, test_user_to_items, train_user_item_set,
        num_users, num_items, CONFIG['top_k'], CONFIG['device'] # Pass num_users
    )
    print(f"Final Precision@{CONFIG['top_k']}: {precision:.4f}")
    print(f"Final Recall@{CONFIG['top_k']}: {recall:.4f}")
    print(f"Final NDCG@{CONFIG['top_k']}: {ndcg:.4f}")

