import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split as sklearn_train_test_split # To avoid naming conflict
import implicit
import lightgbm as lgb
from collections import Counter
import time

# --- Configuration ---
DATA_FILE_PATH = "../../data/kaggle_dataset/raw/data.csv" 
MIN_USER_INTERACTIONS = 5  # Min interactions for a user to be included in CF/Ranker training
MIN_ITEM_INTERACTIONS = 5  # Min interactions for an item to be included
TEST_SET_SIZE_DAYS = 30    # Use last N days for the test set for chronological split
N_LATENT_FACTORS_ALS = 50  # Number of latent factors for ALS
N_RANKER_NEGATIVE_SAMPLES = 4 # Number of negative samples per positive sample for ranker training

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    try:
        # Specify encoding, as default might fail
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Remove cancellations
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Drop rows with missing CustomerID as we need it for personalization
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str) # Ensure consistent type

    # Ensure Quantity and UnitPrice are positive for valid transactions
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Create TransactionValue
    df['TransactionValue'] = df['Quantity'] * df['UnitPrice']

    # Fill missing Descriptions with a placeholder
    df['Description'] = df['Description'].fillna('Unknown')
    
    # For consistency with analysis output, rename columns if needed (optional)
    # df.rename(columns={'StockCode': 'item', 'Description': 'item_descrip', 'CustomerID': 'idcol'}, inplace=True)
    # For this script, we'll stick to original names for clarity with the provided table.

    print(f"Data shape after initial cleaning: {df.shape}")
    if df.empty:
        print("Dataframe is empty after initial cleaning. Check filters.")
        return None
        
    # Create unique integer IDs for users and items
    df['UserIDX'] = df['CustomerID'].astype('category').cat.codes
    df['ItemIDX'] = df['StockCode'].astype('category').cat.codes
    
    # Store mappers
    user_id_map = dict(enumerate(df['CustomerID'].astype('category').cat.categories))
    item_id_map = dict(enumerate(df['StockCode'].astype('category').cat.categories))
    user_idx_to_id = {v: k for k, v in user_id_map.items()} # Not used in this script, but good to have
    item_idx_to_id = {v: k for k, v in item_id_map.items()} # Not used in this script

    print(f"Number of unique users: {df['UserIDX'].nunique()}")
    print(f"Number of unique items: {df['ItemIDX'].nunique()}")
    
    return df, user_id_map, item_id_map

# --- Chronological Train-Test Split ---
def chronological_split(df, days_for_test):
    print(f"\nSplitting data chronologically (last {days_for_test} days for test)...")
    df_sorted = df.sort_values('InvoiceDate')
    split_date = df_sorted['InvoiceDate'].max() - pd.Timedelta(days=days_for_test)
    
    train_df = df_sorted[df_sorted['InvoiceDate'] <= split_date]
    test_df = df_sorted[df_sorted['InvoiceDate'] > split_date]
    
    # Ensure test users/items are also in train, or handle cold start
    # For simplicity here, we'll proceed, but in practice, you'd handle this.
    # Test users/items not in train might not get CF/Ranker recommendations.
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

# --- 2. Candidate Generation ---

#   --- 2.a Content-Based Filtering ---
class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.item_tfidf_matrix = None
        self.item_descriptions = {} # Store ItemIDX -> Description
        self.item_idx_to_desc_idx = {} # Map ItemIDX to row index in tfidf_matrix

    def fit(self, df_items):
        print("\nFitting Content-Based Recommender...")
        # Ensure unique items for fitting TF-IDF
        unique_items_df = df_items[['ItemIDX', 'Description']].drop_duplicates(subset=['ItemIDX']).set_index('ItemIDX')
        self.item_descriptions = unique_items_df['Description'].to_dict()
        
        # Create a list of descriptions in the order of sorted unique ItemIDX
        sorted_item_indices = sorted(list(self.item_descriptions.keys()))
        self.item_idx_to_desc_idx = {item_idx: i for i, item_idx in enumerate(sorted_item_indices)}
        
        descriptions_list = [self.item_descriptions[item_idx] for item_idx in sorted_item_indices]
        
        if not descriptions_list:
            print("Warning: No descriptions found to fit TF-IDF vectorizer.")
            return

        self.item_tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions_list)
        print("TF-IDF matrix shape:", self.item_tfidf_matrix.shape)

    def get_recommendations(self, user_interacted_item_indices, all_item_indices, top_n=10):
        if self.item_tfidf_matrix is None or self.item_tfidf_matrix.shape[0] == 0 or not user_interacted_item_indices: # Added check for empty matrix
            return []

        # Get TF-IDF vectors for items the user interacted with
        user_item_desc_indices = [self.item_idx_to_desc_idx[idx] for idx in user_interacted_item_indices if idx in self.item_idx_to_desc_idx]
        
        if not user_item_desc_indices:
            return []
            
        # This line can produce an np.matrix object
        user_profile_vector_matrix_type = self.item_tfidf_matrix[user_item_desc_indices].mean(axis=0)
        # Convert to np.ndarray to avoid TypeError with cosine_similarity
        user_profile_vector = np.asarray(user_profile_vector_matrix_type)
        
        # Calculate cosine similarity between user profile and all items
        # Ensure all_item_indices are mapped to their TF-IDF matrix indices
        valid_all_item_indices = [idx for idx in all_item_indices if idx in self.item_idx_to_desc_idx]
        all_item_desc_indices = [self.item_idx_to_desc_idx[idx] for idx in valid_all_item_indices]
        
        if not all_item_desc_indices:
            return []

        # self.item_tfidf_matrix[all_item_desc_indices] is a slice of the original sparse matrix.
        # cosine_similarity can handle (dense_array, sparse_matrix) if the dense_array is np.ndarray.
        target_item_matrix_slice = self.item_tfidf_matrix[all_item_desc_indices]
        
        sim_scores = cosine_similarity(user_profile_vector, target_item_matrix_slice)
        
        # Sort and get top N
        # sim_scores is 1xN, so sim_scores[0]
        # Need to handle case where sim_scores might be empty if target_item_matrix_slice is empty
        if sim_scores.shape[1] == 0:
            return []

        sorted_indices = np.argsort(sim_scores[0])[::-1]
        
        recommendations = []
        for i in sorted_indices:
            original_item_idx = valid_all_item_indices[i] # Map back to original ItemIDX
            if original_item_idx not in user_interacted_item_indices: # Don't recommend already interacted items
                recommendations.append(original_item_idx)
                if len(recommendations) >= top_n:
                    break
        return recommendations

    def get_item_similarity_to_user_profile(self, user_interacted_item_indices, candidate_item_idx):
        if self.item_tfidf_matrix is None or self.item_tfidf_matrix.shape[0] == 0 or \
           candidate_item_idx not in self.item_idx_to_desc_idx:
            return 0.0 # Neutral score if item or model not ready

        # Create user profile vector
        user_profile_desc_indices = [self.item_idx_to_desc_idx[idx] for idx in user_interacted_item_indices if idx in self.item_idx_to_desc_idx]
        if not user_profile_desc_indices:
            return 0.0 # No profile, neutral score

        user_profile_vector_matrix_type = self.item_tfidf_matrix[user_profile_desc_indices].mean(axis=0)
        user_profile_vector = np.asarray(user_profile_vector_matrix_type)

        # Get candidate item vector
        candidate_item_desc_idx = self.item_idx_to_desc_idx[candidate_item_idx]
        candidate_item_vector = self.item_tfidf_matrix[candidate_item_desc_idx] # This is already a sparse row vector

        # Calculate cosine similarity
        # cosine_similarity expects 2D arrays. user_profile_vector is 1D after asarray from 1xN matrix.
        # candidate_item_vector is likely 1xN sparse matrix.
        if user_profile_vector.ndim == 1:
            user_profile_vector = user_profile_vector.reshape(1, -1)
        
        # candidate_item_vector from TF-IDF slice is already 2D (1, n_features)
        # No need to .toarray() if cosine_similarity handles (dense, sparse) well, which it does.
        
        try:
            similarity = cosine_similarity(user_profile_vector, candidate_item_vector)
            return similarity[0, 0] if similarity.size > 0 else 0.0
        except: # Catch any error during similarity calculation
            return 0.0
#   --- 2.b Collaborative Filtering (ALS) ---
class CollaborativeFilteringRecommender:
    def __init__(self, factors=N_LATENT_FACTORS_ALS, regularization=0.1, iterations=15):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        self.user_item_matrix = None
        self.user_map = None
        self.item_map = None # To map internal ALS IDs back to original ItemIDX

    def fit(self, train_df):
        print("\nFitting Collaborative Filtering Recommender (ALS)...")
        # Create interaction matrix (UserIDX, ItemIDX, interaction_strength)
        # For implicit feedback, interaction_strength can be 1 (occurrence) or Quantity
        interactions = train_df.groupby(['UserIDX', 'ItemIDX']).size().reset_index(name='Count')
        
        self.user_item_matrix = csr_matrix(
            (interactions['Count'], (interactions['UserIDX'], interactions['ItemIDX']))
        )
        
        # Store maps for later use if needed (though ALS handles this internally)
        self.user_map = {user_idx: i for i, user_idx in enumerate(interactions['UserIDX'].unique())}
        self.item_map = {item_idx: i for i, item_idx in enumerate(interactions['ItemIDX'].unique())}

        if self.user_item_matrix.shape[0] == 0 or self.user_item_matrix.shape[1] == 0:
            print("Warning: User-item matrix is empty. CF model not trained.")
            self.model = None # Invalidate model
            return

        self.model.fit(self.user_item_matrix)
        print("ALS model fitted.")

    def get_recommendations(self, user_idx, N=10, filter_already_liked_items=True, user_liked_items=None):
        if self.model is None or self.user_item_matrix is None or user_idx >= self.user_item_matrix.shape[0]:
            return []
        
        # ALS recommend method expects internal user id. UserIDX is already 0-indexed and matches.
        try:
            # N + len(user_liked_items) to have enough items after filtering
            num_recs_to_fetch = N
            if filter_already_liked_items and user_liked_items:
                num_recs_to_fetch += len(user_liked_items)

            ids, scores = self.model.recommend(user_idx, self.user_item_matrix[user_idx], N=num_recs_to_fetch, filter_already_liked_items=filter_already_liked_items)
            
            # ids are ItemIDX here because our matrix was (UserIDX, ItemIDX)
            if filter_already_liked_items and user_liked_items:
                final_recs = [item_id for item_id in ids if item_id not in user_liked_items][:N]
                return final_recs
            return ids[:N].tolist() # Return as list
        except IndexError: # User might not be in the matrix if it's very sparse or new
            print(f"Warning: UserIDX {user_idx} not found in CF model training data or issues with recommendation.")
            return []
        except Exception as e:
            print(f"Error during CF recommendation for user {user_idx}: {e}")
            return []


#   --- 2.c Popularity-Based Recommender ---
class PopularityRecommender:
    def __init__(self):
        self.popular_items = []

    def fit(self, train_df):
        print("\nFitting Popularity Recommender...")
        # Count purchase frequency for each item
        item_counts = train_df['ItemIDX'].value_counts()
        self.popular_items = item_counts.index.tolist()
        print(f"Found {len(self.popular_items)} popular items.")

    def get_recommendations(self, user_interacted_item_indices=None, top_n=10):
        if not self.popular_items:
            return []
        
        if user_interacted_item_indices:
            recs = [item for item in self.popular_items if item not in user_interacted_item_indices]
            return recs[:top_n]
        return self.popular_items[:top_n]

# --- 3. Feature Engineering and Ranking Model ---
def create_ranking_features(df, cf_model, content_model, popular_items_all_time, user_item_interaction_dict_train):
    print("\nCreating features for ranking model...")
    features = []
    labels = []

    all_items_in_train = df['ItemIDX'].unique()

    # For CF scores, get user and item factors
    user_factors = None
    item_factors = None
    if cf_model and hasattr(cf_model, 'model') and cf_model.model: # Check if model was trained
        user_factors = cf_model.model.user_factors
        item_factors = cf_model.model.item_factors

    # Item popularity (purchase count)
    item_popularity = df['ItemIDX'].value_counts().to_dict()
    
    # User activity (number of purchases)
    user_activity = df.groupby('UserIDX')['ItemIDX'].count().to_dict()

    # Item average price
    item_avg_price = df.groupby('ItemIDX')['UnitPrice'].mean().to_dict()

    processed_users = 0
    total_users = df['UserIDX'].nunique()

    for user_idx in df['UserIDX'].unique():
        processed_users += 1
        if processed_users % 100 == 0:
            print(f"Processing user {processed_users}/{total_users} for ranker features...")

        # Positive samples: items the user actually interacted with
        positive_items = user_item_interaction_dict_train.get(user_idx, set())
        
        user_interacted_item_features_for_content = []
        if content_model: # Get descriptions of items user interacted with
            user_interacted_item_features_for_content = [item_idx for item_idx in positive_items if item_idx in content_model.item_idx_to_desc_idx]


        for item_idx in positive_items:
            feature_vector = {}
            # CF feature: score (dot product of latent factors)
            if user_factors is not None and item_factors is not None and \
               user_idx < user_factors.shape[0] and item_idx < item_factors.shape[0]:
                feature_vector['cf_score'] = np.dot(user_factors[user_idx], item_factors[item_idx])
            else:
                feature_vector['cf_score'] = 0.0 # Default if not available

            current_user_interactions = user_item_interaction_dict_train.get(user_idx, set())
            if not current_user_interactions:
                 feature_vector['content_similarity'] = 0.0
            else:
                 feature_vector['content_similarity'] = content_model.get_item_similarity_to_user_profile(
                     list(current_user_interactions),
                     item_idx
                 )

            feature_vector['item_popularity'] = item_popularity.get(item_idx, 0)
            feature_vector['user_activity'] = user_activity.get(user_idx, 0)
            feature_vector['item_avg_price'] = item_avg_price.get(item_idx, 0)
            features.append(feature_vector)
            labels.append(1) # Positive label

        # Negative samples: items the user did not interact with
        # This needs careful sampling. Random sampling is a start.
        num_negative_to_sample = len(positive_items) * N_RANKER_NEGATIVE_SAMPLES
        
        # Candidate negative items: all items minus positive ones
        potential_negative_items = list(set(all_items_in_train) - positive_items)
        if not potential_negative_items:
            continue

        negative_item_indices = np.random.choice(
            potential_negative_items,
            size=min(num_negative_to_sample, len(potential_negative_items)),
            replace=False
        )

        for item_idx in negative_item_indices:
            feature_vector = {}
            if user_factors is not None and item_factors is not None and \
               user_idx < user_factors.shape[0] and item_idx < item_factors.shape[0]:
                feature_vector['cf_score'] = np.dot(user_factors[user_idx], item_factors[item_idx])
            else:
                feature_vector['cf_score'] = 0.0

            feature_vector['content_score_proxy'] = 0.0 # Proxy for negative samples

            feature_vector['item_popularity'] = item_popularity.get(item_idx, 0)
            feature_vector['user_activity'] = user_activity.get(user_idx, 0)
            feature_vector['item_avg_price'] = item_avg_price.get(item_idx, 0)
            features.append(feature_vector)
            labels.append(0) # Negative label
            
    feature_df = pd.DataFrame(features)
    print(f"Ranking features shape: {feature_df.shape}")
    return feature_df, np.array(labels)


class RankingModel:
    def __init__(self):
        self.model = lgb.LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.05, num_leaves=31) # Basic params

    def fit(self, X_train, y_train):
        print("\nTraining Ranking Model (LightGBM)...")
        if X_train.empty or len(y_train) == 0:
            print("Warning: No data to train ranking model.")
            self.model = None
            return

        # Ensure column names are strings and don't contain special JSON characters for LightGBM
        X_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]

        self.model.fit(X_train, y_train)
        print("Ranking model fitted.")

    def predict_scores(self, X_test):
        if self.model is None or X_test.empty:
            return np.array([])
        
        # Ensure column names match training
        X_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]
        
        try:
            return self.model.predict_proba(X_test)[:, 1] # Probability of class 1 (interaction)
        except Exception as e:
            print(f"Error predicting with ranker: {e}")
            # Fallback if columns mismatch significantly due to dynamic features not present at inference for certain items
            # This can happen if some features like item_avg_price aren't available for all candidate items
            # A more robust solution involves careful feature engineering and imputation.
            # For now, return zeros if prediction fails.
            print("Returning zero scores due to prediction error.")
            return np.zeros(X_test.shape[0])

# --- 5. Popularity Baseline Recommender ---
class PopularityBaselineRecommender:
    def __init__(self):
        self.popular_items_scores = {} # Store ItemIDX -> score (e.g., purchase count)
        self.sorted_popular_items = []

    def fit(self, train_df):
        print("\nFitting Popularity Baseline Recommender...")
        item_counts = train_df['ItemIDX'].value_counts()
        self.popular_items_scores = item_counts.to_dict()
        self.sorted_popular_items = item_counts.index.tolist()
        print(f"Popularity baseline fitted with {len(self.sorted_popular_items)} items.")

    def get_recommendations(self, user_idx, N=10, user_already_liked_items=None):
        # user_idx is not used for this simple baseline, but kept for consistent interface
        recommendations = []
        if user_already_liked_items is None:
            user_already_liked_items = set()

        for item_idx in self.sorted_popular_items:
            if item_idx not in user_already_liked_items:
                recommendations.append(item_idx)
            if len(recommendations) >= N:
                break
        return recommendations

# --- 6. Evaluation Metrics ---
def calculate_metrics_for_user(ground_truth_items, recommended_items, N_eval):
    # Ensure recommended_items is a list of actual item IDs, not scores yet
    actual_rec_items = recommended_items[:N_eval]
    
    hits = set(ground_truth_items) & set(actual_rec_items)
    num_hits = len(hits)

    # Precision@N
    precision_at_n = num_hits / N_eval if N_eval > 0 else 0.0
    
    # Recall@N
    recall_at_n = num_hits / len(ground_truth_items) if len(ground_truth_items) > 0 else 0.0
    
    # Hit Rate@N
    hit_rate_at_n = 1.0 if num_hits > 0 else 0.0
    
    # MRR@N (Mean Reciprocal Rank)
    mrr_at_n = 0.0
    for i, item_idx in enumerate(actual_rec_items):
        if item_idx in ground_truth_items:
            mrr_at_n = 1.0 / (i + 1)
            break
            
    # NDCG@N (Normalized Discounted Cumulative Gain)
    dcg_at_n = 0.0
    for i, item_idx in enumerate(actual_rec_items):
        if item_idx in ground_truth_items:
            dcg_at_n += 1.0 / np.log2(i + 2) # rank is i+1, log base 2 of (rank+1) or log2(i+2)
            
    idcg_at_n = 0.0
    num_relevant_to_consider = min(len(ground_truth_items), N_eval)
    for i in range(num_relevant_to_consider):
        idcg_at_n += 1.0 / np.log2(i + 2)
        
    ndcg_at_n = dcg_at_n / idcg_at_n if idcg_at_n > 0 else 0.0
    
    return precision_at_n, recall_at_n, hit_rate_at_n, mrr_at_n, ndcg_at_n

def evaluate_recommender(test_df, recommendations_dict, N_eval=10, model_name="Recommender"):
    print(f"\n--- Evaluating {model_name} @{N_eval} ---")
    
    # Prepare ground truth from test_df
    test_user_item_interaction_dict = test_df.groupby('UserIDX')['ItemIDX'].apply(set).to_dict()

    all_precision = []
    all_recall = []
    all_hit_rate = []
    all_mrr = []
    all_ndcg = []
    
    evaluated_users = 0
    for user_idx, ground_truth_items in test_user_item_interaction_dict.items():
        if not ground_truth_items: # Skip users with no interactions in test set
            continue
            
        # Get recommendations for this user (ensure they are item_idx, not tuples with scores)
        recommended_items = recommendations_dict.get(user_idx, [])
        
        if not recommended_items and not ground_truth_items: # No recs, no truth, skip
             continue
        if not recommended_items and ground_truth_items: # No recs but had truth, count as 0 for metrics if desired
            # For this setup, if no recs, metrics will be 0 for this user if they had ground truth.
            # Or, one might choose to only evaluate on users for whom recs were generated.
            # Here, if recs are empty, hits will be 0.
            pass


        p, r, hr, mrr, ndcg = calculate_metrics_for_user(ground_truth_items, recommended_items, N_eval)
        
        all_precision.append(p)
        all_recall.append(r)
        all_hit_rate.append(hr)
        all_mrr.append(mrr)
        all_ndcg.append(ndcg)
        evaluated_users +=1

    if evaluated_users == 0:
        print("No users were evaluated. Check test set or recommendation generation.")
        return

    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_hit_rate = np.mean(all_hit_rate)
    avg_mrr = np.mean(all_mrr)
    avg_ndcg = np.mean(all_ndcg)
    
    print(f"Evaluated on {evaluated_users} users from the test set.")
    print(f"Average Precision@{N_eval}: {avg_precision:.4f}")
    print(f"Average Recall@{N_eval}:    {avg_recall:.4f}")
    print(f"Average Hit Rate@{N_eval}:  {avg_hit_rate:.4f}")
    print(f"Average MRR@{N_eval}:       {avg_mrr:.4f}")
    print(f"Average NDCG@{N_eval}:      {avg_ndcg:.4f}")
    
    return {
        "precision": avg_precision, "recall": avg_recall, "hit_rate": avg_hit_rate,
        "mrr": avg_mrr, "ndcg": avg_ndcg
    }


# --- Main Training and Hybrid Recommendation Workflow (Modified) ---
def main():
    start_time = time.time()
    N_EVAL_TOP_K = 10 # Evaluate top-10 recommendations

    # 1. Load and Preprocess Data
    df_full, user_id_map_full, item_id_map_full = load_and_preprocess_data(DATA_FILE_PATH)
    if df_full is None:
        return

    item_idx_to_description = df_full.set_index('ItemIDX')['Description'].to_dict()
    all_item_indices_global = df_full['ItemIDX'].unique() 

    # Chronological Split
    train_df, test_df = chronological_split(df_full, TEST_SET_SIZE_DAYS)

    user_counts_train = train_df['UserIDX'].value_counts()
    item_counts_train = train_df['ItemIDX'].value_counts()
    
    # Filter train_df
    train_df_filtered = train_df[train_df['UserIDX'].isin(user_counts_train[user_counts_train >= MIN_USER_INTERACTIONS].index)]
    train_df_filtered = train_df_filtered[train_df_filtered['ItemIDX'].isin(item_counts_train[item_counts_train >= MIN_ITEM_INTERACTIONS].index)]
    
    print(f"Train shape after min interaction filtering: {train_df_filtered.shape}")
    if train_df_filtered.empty:
        print("Train DataFrame is empty after filtering. Adjust MIN_USER_INTERACTIONS or MIN_ITEM_INTERACTIONS.")
        return

    user_item_interaction_dict_train = train_df_filtered.groupby('UserIDX')['ItemIDX'].apply(set).to_dict()
    
    # For evaluation: get users present in the test set for whom we need to generate recommendations
    test_users_for_eval = test_df['UserIDX'].unique()
    # And their interactions in the training set (for filtering already liked items)
    user_item_interaction_dict_train_for_test_users = train_df[train_df['UserIDX'].isin(test_users_for_eval)]\
                                                        .groupby('UserIDX')['ItemIDX'].apply(set).to_dict()


    # --- TRAIN MODELS ---
    # Content-Based
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(train_df_filtered[['ItemIDX', 'Description']].drop_duplicates())

    # Collaborative Filtering
    cf_recommender = CollaborativeFilteringRecommender()
    cf_recommender.fit(train_df_filtered)

    # Popularity-Based (used as a candidate generator and also as a baseline)
    # This is the PopularityRecommender from your original script (candidate generator)
    popularity_candidate_gen = PopularityRecommender() 
    popularity_candidate_gen.fit(train_df_filtered)
    popular_items_all_time_train = popularity_candidate_gen.popular_items

    # Ranking Model Training
    X_ranker_train, y_ranker_train = create_ranking_features(
        train_df_filtered, 
        cf_recommender, # Pass the whole object which has the .model attribute
        content_recommender,
        popular_items_all_time_train,
        user_item_interaction_dict_train
    )
    
    ranker = RankingModel()
    if not X_ranker_train.empty:
        ranker.fit(X_ranker_train, y_ranker_train)
    else:
        print("Skipping ranker training due to empty feature set.")

    # Train Popularity Baseline Recommender (the new one for evaluation)
    popularity_baseline = PopularityBaselineRecommender()
    popularity_baseline.fit(train_df_filtered) # Train on filtered train data

    # --- GENERATE RECOMMENDATIONS FOR TEST USERS ---
    print("\n--- Generating Recommendations for Test Set Users ---")
    
    hybrid_recommendations_for_test = {}
    popularity_baseline_recommendations_for_test = {}

    # Pre-calculate some features for inference on test users to speed up ranker
    # These should be based on train_df_filtered characteristics
    item_popularity_train_dict = train_df_filtered['ItemIDX'].value_counts().to_dict()
    user_activity_train_dict = train_df_filtered.groupby('UserIDX')['ItemIDX'].count().to_dict()
    item_avg_price_train_dict = train_df_filtered.groupby('ItemIDX')['UnitPrice'].mean().to_dict()
    
    cf_user_factors_trained = None
    cf_item_factors_trained = None
    if cf_recommender.model and hasattr(cf_recommender.model, 'user_factors'):
        cf_user_factors_trained = cf_recommender.model.user_factors
        cf_item_factors_trained = cf_recommender.model.item_factors

    for user_idx in test_users_for_eval:
        if user_idx % 100 == 0: # Progress update
            print(f"Generating recommendations for test user {user_idx}/{len(test_users_for_eval)}")

        user_liked_items_train = user_item_interaction_dict_train_for_test_users.get(user_idx, set())

        # 1. Hybrid Model Recommendations
        if ranker.model is not None: # Only if ranker was trained
            # Candidate Generation
            candidates_cf_test = cf_recommender.get_recommendations(user_idx, N=50, user_liked_items=user_liked_items_train) # Fetch more for ranking
            
            # Content recs: Get items user liked in train (if any), get similar items
            interacted_for_content_test = [item for item in user_liked_items_train if item in content_recommender.item_idx_to_desc_idx]
            candidates_content_test = content_recommender.get_recommendations(interacted_for_content_test, all_item_indices_global, top_n=50)
            
            candidates_popular_test = popularity_candidate_gen.get_recommendations(user_liked_items_train, top_n=50)

            all_candidates_test = list(set(candidates_cf_test + candidates_content_test + candidates_popular_test))
            final_candidates_to_rank_test = [item for item in all_candidates_test if item not in user_liked_items_train]

            if final_candidates_to_rank_test:
                candidate_features_list_test = []
                user_activity_val_train = user_activity_train_dict.get(user_idx, 0) # Activity from train set

                user_factor_sample_test = None
                if cf_user_factors_trained is not None and user_idx < cf_user_factors_trained.shape[0]:
                     user_factor_sample_test = cf_user_factors_trained[user_idx]
                     
                user_liked_items_in_train_for_profile = user_item_interaction_dict_train_for_test_users.get(user_idx, set())

                for item_idx_cand in final_candidates_to_rank_test:
                    feature_vector = {}
                    if user_factor_sample_test is not None and cf_item_factors_trained is not None and \
                       item_idx_cand < cf_item_factors_trained.shape[0]:
                        feature_vector['cf_score'] = np.dot(user_factor_sample_test, cf_item_factors_trained[item_idx_cand])
                    else:
                        feature_vector['cf_score'] = 0.0
                    
                    # Real Content Similarity Score
                    if not user_liked_items_in_train_for_profile: # If user has no history for profile
                        feature_vector['content_similarity'] = 0.0
                    else:
                        feature_vector['content_similarity'] = content_recommender.get_item_similarity_to_user_profile(
                            list(user_liked_items_in_train_for_profile),
                            item_idx_cand
                        )
                    feature_vector['content_score_proxy'] = 0.0 
                    feature_vector['item_popularity'] = item_popularity_train_dict.get(item_idx_cand, 0)
                    feature_vector['user_activity'] = user_activity_val_train
                    feature_vector['item_avg_price'] = item_avg_price_train_dict.get(item_idx_cand, 0)
                    candidate_features_list_test.append(feature_vector)
                
                if candidate_features_list_test:
                    X_candidates_ranker_test = pd.DataFrame(candidate_features_list_test)
                    X_candidates_ranker_test.fillna(0, inplace=True)
                    
                    if not X_candidates_ranker_test.empty:
                        candidate_scores_test = ranker.predict_scores(X_candidates_ranker_test)
                        scored_candidates_test = sorted(zip(final_candidates_to_rank_test, candidate_scores_test), key=lambda x: x[1], reverse=True)
                        hybrid_recommendations_for_test[user_idx] = [item_idx for item_idx, score in scored_candidates_test[:N_EVAL_TOP_K]]
                    else:
                         hybrid_recommendations_for_test[user_idx] = candidates_cf_test[:N_EVAL_TOP_K] # Fallback
                else: # No features generated, fallback (e.g. to CF or popular)
                    hybrid_recommendations_for_test[user_idx] = candidates_cf_test[:N_EVAL_TOP_K] if candidates_cf_test else candidates_popular_test[:N_EVAL_TOP_K]
            else: # No candidates after filtering
                hybrid_recommendations_for_test[user_idx] = popularity_candidate_gen.get_recommendations(user_liked_items_train, top_n=N_EVAL_TOP_K) # Fallback to popular
        else: # Ranker not trained, use CF or Popularity as main hybrid output
            # Fallback: simple blend or just CF if available
            cf_recs = cf_recommender.get_recommendations(user_idx, N=N_EVAL_TOP_K, user_liked_items=user_liked_items_train)
            if cf_recs:
                hybrid_recommendations_for_test[user_idx] = cf_recs
            else: # Fallback further to global popular if CF fails for user
                hybrid_recommendations_for_test[user_idx] = popularity_candidate_gen.get_recommendations(user_liked_items_train, top_n=N_EVAL_TOP_K)


        # 2. Popularity Baseline Recommendations
        popularity_baseline_recommendations_for_test[user_idx] = popularity_baseline.get_recommendations(
            user_idx, N=N_EVAL_TOP_K, user_already_liked_items=user_liked_items_train
        )
        
    # --- EVALUATE MODELS ---
    evaluate_recommender(test_df, hybrid_recommendations_for_test, N_eval=N_EVAL_TOP_K, model_name="Hybrid Model")
    evaluate_recommender(test_df, popularity_baseline_recommendations_for_test, N_eval=N_EVAL_TOP_K, model_name="Popularity Baseline")

    # --- Example Recommendation for one user (as before) ---
    if user_item_interaction_dict_train: # Check if any users in train
        sample_user_idx_for_example = list(user_item_interaction_dict_train.keys())[0] 
        sample_user_original_id = user_id_map_full.get(sample_user_idx_for_example, "Unknown UserID")
        print(f"\n--- Example: Top {N_EVAL_TOP_K} Hybrid Recommendations for UserIDX {sample_user_idx_for_example} (Original ID: {sample_user_original_id}) ---")
        
        example_recs = hybrid_recommendations_for_test.get(sample_user_idx_for_example, []) # Get from test recs if user is in test
        if not example_recs and sample_user_idx_for_example in hybrid_recommendations_for_test: # If user was in test but got no recs
            example_recs = [] # Ensure it's an empty list
        elif not example_recs: # If user wasn't in test_users_for_eval, generate fresh (less ideal for demo)
            # This part may re-run logic if sample user wasn't in test_users_for_eval.
            # For simplicity, we'll just pull from hybrid_recommendations_for_test if possible.
             print(f"(User {sample_user_idx_for_example} might not have been in the evaluated test set, showing general potential recs if any generated)")


        if example_recs:
            for item_idx in example_recs:
                description = item_idx_to_description.get(item_idx, "Unknown Item")
                original_stock_code = item_id_map_full.get(item_idx, "Unknown StockCode")
                print(f"  - ItemIDX: {item_idx} (StockCode: {original_stock_code}), Description: {description}")
        else:
            print("No recommendations generated for this sample user in the test evaluation pass.")
            
    end_time = time.time()
    print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()