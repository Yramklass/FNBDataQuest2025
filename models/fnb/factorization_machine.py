import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix # For interaction matrix if not using Dataset fully

from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset # Alias to avoid confusion
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

import torch # For NDCG calculation tensor operations
import torchmetrics.functional.retrieval as tm_functional # Using functional for NDCG
from tqdm import tqdm

# Configuration
CONFIG_LFM = {
    "no_components": 64,       # Dimensionality of the feature latent embeddings.
    "learning_rate": 0.01,
    "loss": 'warp',            # 'bpr', 'warp', 'logistic', 'warp-kos'
    "item_alpha": 1e-6,        # L2 penalty on item features.
    "user_alpha": 1e-6,        # L2 penalty on user features.
    "max_sampled": 10,         # Max number of positive samples used in WARP (-kos).
    "epochs": 20,              # Number of epochs to train for.
    "num_threads": 1,          # Number of parallel threads to use. 
    "random_state": 42,
    "top_k": 10
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # For NDCG calculation

# Load Data
def load_data_for_lightfm(file_path="../../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv"):
    print(f"Loading data from: {file_path}")
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")

    df_raw['user_id_str'] = df_raw['idcol'].astype(str)
    df_raw['item_id_str'] = df_raw['item'].astype(str)

    # Fill NaNs in feature columns that will be used by LightFM
    feature_cols_to_clean = ['item_type', 'item_descrip', 'segment', 'beh_segment', 'active_ind']
    for col in feature_cols_to_clean:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna("UNKNOWN").astype(str)
        else:
            print(f"Warning: Feature column '{col}' not found in DataFrame. It will be ignored.")
            df_raw[col] = "UNKNOWN" # Add if missing to prevent errors later, though it won't be useful

    # Filter for positive interactions
    df_positive = df_raw[df_raw['interaction'].isin(['CLICK', 'CHECKOUT'])].copy()
    return df_raw, df_positive

# Prepare Data using LightFM Dataset 
def prepare_lightfm_data(df_all_interactions, df_positive_interactions):
    dataset = LightFMDataset()

    # Fit the dataset with all unique users, items, and feature values
    # This builds the internal mappings in LightFM
    all_users = df_all_interactions['user_id_str'].unique()
    all_items = df_all_interactions['item_id_str'].unique()
    
    # Prepare iterables of feature names for fitting
    all_item_feature_names = []
    if 'item_type' in df_all_interactions.columns:
        all_item_feature_names.extend([f"item_type:{val}" for val in df_all_interactions['item_type'].unique()])
    if 'item_descrip' in df_all_interactions.columns:
         all_item_feature_names.extend([f"item_descrip:{val}" for val in df_all_interactions['item_descrip'].unique()])

    all_user_feature_names = []
    if 'segment' in df_all_interactions.columns:
        all_user_feature_names.extend([f"segment:{val}" for val in df_all_interactions['segment'].unique()])
    if 'beh_segment' in df_all_interactions.columns:
        all_user_feature_names.extend([f"beh_segment:{val}" for val in df_all_interactions['beh_segment'].unique()])
    if 'active_ind' in df_all_interactions.columns:
        all_user_feature_names.extend([f"active_ind:{val}" for val in df_all_interactions['active_ind'].unique()])

    print("DEBUG: All user features: {0}".format( len(all_user_feature_names)))
    
    print("DEBUG: All item features: {0}".format(len(all_item_feature_names)))
    

    dataset.fit(
        users=all_users,
        items=all_items,
        user_features=all_user_feature_names if all_user_feature_names else None,
        item_features=all_item_feature_names if all_item_feature_names else None
    )
    # Check internal mapping sizes right after fit
    user_id_to_internal_mapping, user_feature_to_internal_mapping, \
    item_id_to_internal_mapping, item_feature_to_internal_mapping = dataset.mapping()

    print(f"DEBUG: Internal LightFM user feature mapping size after fit: {len(user_feature_to_internal_mapping)}")
    print(f"DEBUG: Internal LightFM item feature mapping size after fit: {len(item_feature_to_internal_mapping)}")
    
    # Build interaction matrix (using positive interactions)
    interactions_iter = ((row['user_id_str'], row['item_id_str']) for _, row in df_positive_interactions.iterrows())
    (interactions_matrix, weights_matrix) = dataset.build_interactions(interactions_iter)
    print("Interactions matrix shape:", interactions_matrix.shape)

    # Build feature matrices
    def build_feature_iter(df_source, id_col_str, feature_cols_map):

        unique_df = df_source.drop_duplicates(subset=[id_col_str])
        for _, row in unique_df.iterrows():
            features = []
            for feat_name, raw_col_name in feature_cols_map.items():
                if raw_col_name in row and pd.notna(row[raw_col_name]):
                    features.append(f"{feat_name}:{row[raw_col_name]}")
            if features: # Only yield if there are actual features
                 yield (row[id_col_str], features)
    
    user_feature_cols_map = {'segment': 'segment', 'beh_segment': 'beh_segment', 'active_ind': 'active_ind'}
    item_feature_cols_map = {'item_type': 'item_type', 'item_descrip': 'item_descrip'}

    user_features_matrix = None
    if all_user_feature_names:
        user_features_matrix = dataset.build_user_features(
            build_feature_iter(df_all_interactions, 'user_id_str', user_feature_cols_map),
            normalize=False 
        )
        print("User features matrix shape:", user_features_matrix.shape)

    item_features_matrix = None
    if all_item_feature_names:
        item_features_matrix = dataset.build_item_features(
            build_feature_iter(df_all_interactions, 'item_id_str', item_feature_cols_map),
            normalize=False
        )
        print("Item features matrix shape:", item_features_matrix.shape)
        
    return dataset, interactions_matrix, user_features_matrix, item_features_matrix



# Model Training 
def train_lightfm_model(interactions, user_features, item_features, config):
    model = LightFM(
        no_components=config['no_components'],
        learning_rate=config['learning_rate'],
        loss=config['loss'],
        item_alpha=config['item_alpha'],
        user_alpha=config['user_alpha'],
        max_sampled=config['max_sampled'],
        random_state=np.random.RandomState(config['random_state']) # Ensure reproducibility
    )

    print("Training LightFM model...")
    model.fit(
        interactions,
        user_features=user_features,
        item_features=item_features,
        epochs=config['epochs'],
        num_threads=config['num_threads'],
        verbose=True
    )
    return model

# Evaluation (including NDCG) 
def evaluate_lightfm_model(model, test_interactions, train_interactions,
                           user_features, item_features, dataset_lfm, config):
    print("Evaluating LightFM model...")
    k = config['top_k']

    # LightFM's built-in metrics
    mean_precision = precision_at_k(model, test_interactions, train_interactions=train_interactions, k=k,
                                   user_features=user_features, item_features=item_features,
                                   num_threads=config['num_threads']).mean()
    mean_recall = recall_at_k(model, test_interactions, train_interactions=train_interactions, k=k,
                             user_features=user_features, item_features=item_features,
                             num_threads=config['num_threads']).mean()
    # mean_auc = auc_score(model, test_interactions, train_interactions=train_interactions,
    #                     user_features=user_features, item_features=item_features,
    #                     num_threads=config['num_threads']).mean()


    # Get all user and item internal IDs that are in the test set
    # test_interactions is a COO matrix: (row=user_id_internal, col=item_id_internal, data=rating)
    test_user_ids_internal = np.unique(test_interactions.row)
    num_all_items_internal = model.item_embeddings.shape[0] # Number of items LightFM knows
    all_items_internal_ids = np.arange(num_all_items_internal)

    all_user_ndcgs_list = []

    for user_id_internal in tqdm(test_user_ids_internal, desc="Calculating NDCG for LightFM"):
        # Predict scores for this user against all items
        # We need to create an array of [user_id_internal, user_id_internal, ...] for each item
        user_id_repeated = np.full_like(all_items_internal_ids, fill_value=user_id_internal, dtype=np.int32)
        
        scores_for_user = model.predict(
            user_id_repeated,
            all_items_internal_ids,
            user_features=user_features,
            item_features=item_features,
            num_threads=config['num_threads']
        )
        
        # Create target tensor for this user
        # Ground truth relevant items for this user from test_interactions
        actual_pos_item_ids_internal = test_interactions.tocsr()[user_id_internal].indices
        
        target_tensor = torch.zeros(num_all_items_internal, dtype=torch.bool, device=device)
        if len(actual_pos_item_ids_internal) > 0:
            target_tensor[actual_pos_item_ids_internal] = True # item IDs are already internal LightFM indices

        # Ensure scores are float and on the correct device for torchmetrics
        scores_for_user_tensor = torch.from_numpy(scores_for_user).float().to(device)

        ndcg_val = tm_functional.retrieval_normalized_dcg(
            scores_for_user_tensor,
            target_tensor,
            top_k=k
        )
        all_user_ndcgs_list.append(ndcg_val.item())

    mean_ndcg = np.mean(all_user_ndcgs_list) if all_user_ndcgs_list else 0.0
    
    return {"recall": mean_recall, "precision": mean_precision, "ndcg": mean_ndcg}


# --- Main Execution ---
if __name__ == "__main__":
    df_all_raw, df_pos = load_data_for_lightfm()
    
    if df_pos.empty:
        print("No positive interactions found. Exiting.")
    else:
        lfm_dataset, interactions, user_feat, item_feat = prepare_lightfm_data(df_all_raw, df_pos)
        
        # Split interactions into train and test
        # test_percentage: proportion of interactions to leave out for testing.
        # random_state ensures reproducibility
        train_interactions, test_interactions = random_train_test_split(
            interactions, test_percentage=0.2,
            random_state=np.random.RandomState(CONFIG_LFM['random_state'])
        )
        print("Train interactions shape:", train_interactions.shape)
        print("Test interactions shape:", test_interactions.shape)

        model = train_lightfm_model(train_interactions, user_feat, item_feat, CONFIG_LFM)

        metrics = evaluate_lightfm_model(model, test_interactions, train_interactions,
                                         user_feat, item_feat, lfm_dataset, CONFIG_LFM)
        
        print("\nLightFM Model Evaluation Metrics:")
        print(f"  Recall@{CONFIG_LFM['top_k']}:    {metrics['recall']:.4f}")
        print(f"  Precision@{CONFIG_LFM['top_k']}: {metrics['precision']:.4f}")
        print(f"  NDCG@{CONFIG_LFM['top_k']}:      {metrics['ndcg']:.4f}")