import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_recsys_data(file_path):
    print(f"Loading data from: {file_path}\n")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")

    df['int_date'] = pd.to_datetime(df['int_date'], format='%d-%b-%y')
    df['item'] = df['item'].fillna('UNKNOWN_ITEM') # Fill NaN items before analysis

    print("--- Basic Information ---")
    print(f"Shape of the dataset: {df.shape}")
    print(f"Number of unique users (idcol): {df['idcol'].nunique()}")
    print(f"Number of unique items: {df['item'].nunique()}")
    print(f"Date range: {df['int_date'].min()} to {df['int_date'].max()}")

    print("\n--- Interaction Analysis ---")
    interaction_counts = df['interaction'].value_counts()
    print("Interaction type counts:")
    print(interaction_counts)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=interaction_counts.index, y=interaction_counts.values)
    plt.title('Distribution of Interaction Types')
    plt.ylabel('Count')
    plt.xlabel('Interaction Type')
    plt.show()

    # Filter for positive interactions for sparsity calculation
    positive_interactions_df = df[df['interaction'].isin(['CLICK', 'CHECKOUT'])]
    num_positive_interactions = len(positive_interactions_df)
    num_users_positive = positive_interactions_df['idcol'].nunique()
    num_items_positive = positive_interactions_df['item'].nunique()

    print(f"\nNumber of positive interactions (CLICK/CHECKOUT): {num_positive_interactions}")
    print(f"Number of users with positive interactions: {num_users_positive}")
    print(f"Number of items with positive interactions: {num_items_positive}")
    
    if num_users_positive > 0 and num_items_positive > 0:
        sparsity = 1.0 - (num_positive_interactions / (num_users_positive * num_items_positive))
        print(f"Sparsity of the positive user-item interaction matrix: {sparsity:.4f}")
    else:
        print("Not enough data for sparsity calculation on positive interactions.")


    print("\n--- Interactions per User (Positive Interactions) ---")
    user_interaction_counts = positive_interactions_df.groupby('idcol').size()
    print(user_interaction_counts.describe())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(user_interaction_counts, bins=50, kde=False)
    plt.title('Distribution of Positive Interactions per User')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    plt.yscale('log') 
    plt.xlim(left=1) 
    plt.show()

    print("\n--- Interactions per Item (Positive Interactions) ---")
    item_interaction_counts = positive_interactions_df.groupby('item').size()
    print(item_interaction_counts.describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(item_interaction_counts, bins=50, kde=False)
    plt.title('Distribution of Positive Interactions per Item')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items')
    plt.yscale('log') # Use log scale
    plt.xlim(left=1)
    plt.show()

    print("\n--- Top Interacted Items (Positive Interactions) ---")
    top_items = item_interaction_counts.sort_values(ascending=False)
    print("Top 10 most interacted items:")
    print(top_items.head(10))

    if 'category' in df.columns:
        print("\n--- Interactions by Category (All Interactions) ---")
        category_counts = df['category'].value_counts()
        print(category_counts)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Interactions by Category')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    if 'segment1' in df.columns: 
        print("\n--- Interactions by segment1 (All Interactions) ---")
        segment1_counts = df['segment1'].value_counts()
        print(segment1_counts)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=segment1_counts.index, y=segment1_counts.values)
        plt.title('Interactions by Segment1')
        plt.xlabel('Segment1')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    print("\nData analysis complete.")


file_path = "../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv" 
analyze_recsys_data(file_path)