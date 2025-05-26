import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np


fnb_colors = {
    "dark_teal": "#116e71",        # Good for primary elements, bars
    "bright_teal": "#43d8c6",      # Good for accents, highlights, or a distinct category
    "very_dark_teal": "#0a3f3a",   # For text, titles, or dark elements
    "medium_teal": "#21746f",      # Another option for categorical data
    "light_teal_gray": "#80bfac",  # Lighter elements, or a lighter category
    "black": "#000000",
    "white": "#FFFFFF"
}


fnb_categorical_palette = [
    fnb_colors["dark_teal"],
    fnb_colors["bright_teal"],
    fnb_colors["medium_teal"],
    fnb_colors["light_teal_gray"],
    fnb_colors["very_dark_teal"] 
]

# FNB Dataset Specific Analysis Function
def preprocess_and_analyze_fnb_data(df, dataset_name="FNB"):
    print(f"\n--- Preprocessing and Analyzing {dataset_name} Dataset ---")

    df['idcol'] = df['idcol'].astype(str)
    df['item'] = df['item'].astype(str)
    
    try:
        df['int_date'] = pd.to_datetime(df['int_date'], format='%d-%b-%y')
    except ValueError as e:
        print(f"Warning: Could not parse all dates with format %d-%b-%y for FNB: {e}")
        df['int_date'] = pd.to_datetime(df['int_date'], errors='coerce')

    df_interactions_for_recs = df[
        df['interaction'].isin(['CLICK', 'CHECKOUT']) & (df['item'] != 'NONE')
    ].copy()

    print("\n--- Basic Information (FNB) ---")
    print(f"Shape of the dataset: {df.shape}")
    print(f"Number of unique users (idcol): {df['idcol'].nunique()}")
    print(f"Number of unique items (excluding 'NONE'): {df[df['item'] != 'NONE']['item'].nunique()}")
    print(f"Number of unique items (including 'NONE'): {df['item'].nunique()}")
    if 'int_date' in df.columns and df['int_date'].notna().any():
        print(f"Date range: {df['int_date'].min()} to {df['int_date'].max()}")
    print(f"Columns: {df.columns.tolist()}")

    print("\n--- Interaction Analysis (FNB) ---")
    interaction_counts = df['interaction'].value_counts()
    print("Interaction type counts:")
    print(interaction_counts)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=interaction_counts.index, y=interaction_counts.values, 
                palette=[fnb_colors["dark_teal"], fnb_colors["medium_teal"], fnb_colors["light_teal_gray"]]) 
    plt.title(f'Distribution of Interaction Types ({dataset_name})')
    plt.ylabel('Count')
    plt.xlabel('Interaction Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    num_positive_interactions = len(df_interactions_for_recs)
    num_users_positive = df_interactions_for_recs['idcol'].nunique()
    num_items_positive = df_interactions_for_recs['item'].nunique()

    print(f"\nNumber of CLICK/CHECKOUT interactions on actual items: {num_positive_interactions}")
    print(f"Number of users with CLICK/CHECKOUT interactions: {num_users_positive}")
    print(f"Number of items involved in CLICK/CHECKOUT interactions: {num_items_positive}")
    
    if num_users_positive > 0 and num_items_positive > 0:
        sparsity = 1.0 - (num_positive_interactions / (num_users_positive * num_items_positive))
        print(f"Sparsity (CLICK/CHECKOUT on items): {sparsity:.4f}")
    else:
        print("Not enough CLICK/CHECKOUT data on actual items for sparsity calculation.")

    if not df_interactions_for_recs.empty:
        print("\n--- Interactions per User (CLICK/CHECKOUT on items) (FNB) ---")
        user_interaction_counts = df_interactions_for_recs.groupby('idcol').size()
        print(user_interaction_counts.describe())
        plt.figure(figsize=(10, 6))
        sns.histplot(user_interaction_counts, bins=min(50, user_interaction_counts.nunique()), kde=False, 
                     color=fnb_colors["dark_teal"])
        plt.title(f'Distribution of Positive Interactions (CLICK/CHECKOUT) per User ({dataset_name})')
        plt.xlabel('Number of CLICK/CHECKOUT Interactions')
        plt.ylabel('Number of Users')
        plt.yscale('log')
        plt.xlim(left=1)
        plt.show()

        print("\n--- Interactions per Item (CLICK/CHECKOUT on items) (FNB) ---")
        item_interaction_counts = df_interactions_for_recs.groupby('item').size()
        print(item_interaction_counts.describe())
        plt.figure(figsize=(10, 6))
        sns.histplot(item_interaction_counts, bins=min(50, item_interaction_counts.nunique()), kde=False, 
                     color=fnb_colors["medium_teal"])
        plt.title(f'Distribution of Positive Interactions (CLICK/CHECKOUT) per Item ({dataset_name})')
        plt.xlabel('Number of CLICK/CHECKOUT Interactions')
        plt.ylabel('Number of Items')
        plt.yscale('log')
        plt.xlim(left=1)
        plt.show()

        print("\n--- Top Interacted Items (CLICK/CHECKOUT) (FNB) ---")
        top_items = item_interaction_counts.sort_values(ascending=False)
        print(f"Top 10 most interacted items ({dataset_name}):")
        print(top_items.head(10))
        if 'item_descrip' in df.columns:
            top_items_df = top_items.head(10).reset_index()
            top_items_df.columns = ['item', 'interaction_count'] 
            top_items_with_desc = pd.merge(
                top_items_df,
                df[['item', 'item_descrip']].drop_duplicates(),
                on='item'
            )
            print("\nTop 10 items with descriptions:")
            print(top_items_with_desc[['item', 'interaction_count', 'item_descrip']])

    if 'int_date' in df.columns and df['int_date'].notna().any():
        print("\n--- Temporal Interaction Patterns (All Interactions) (FNB) ---")
        plt.figure(figsize=(12,5))
        df.set_index('int_date')['idcol'].resample('M').count().plot(marker='o', 
                                                                   color=fnb_colors["dark_teal"], 
                                                                   linestyle='-')
        plt.title(f'Total Interactions Over Time (Monthly) ({dataset_name})')
        plt.ylabel('Number of Interactions')
        plt.xlabel('Date')
        plt.show()

        if 'tod' in df.columns:
            tod_order = ['Early', 'Morning', 'Afternoon', 'Evening']
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x='tod', order=tod_order, 
                          palette=[fnb_colors["dark_teal"], fnb_colors["medium_teal"], fnb_colors["light_teal_gray"], fnb_colors["bright_teal"]])
            plt.title(f'Interactions by Time of Day (tod) ({dataset_name})')
            plt.xlabel('Time of Day')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

    print("\n--- Page Analysis (All Interactions) (FNB) ---")
    if 'page' in df.columns:
        page_counts = df['page'].value_counts()
        print(page_counts)
        plt.figure(figsize=(8,5))
        sns.barplot(x=page_counts.index, y=page_counts.values, 
                    palette=[fnb_colors["dark_teal"], fnb_colors["medium_teal"]])
        plt.title(f'Interactions by Page ({dataset_name})')
        plt.show()

    print("\n--- Item Type Analysis (CLICK/CHECKOUT Interactions) (FNB) ---")
    if 'item_type' in df_interactions_for_recs.columns:
        item_type_counts = df_interactions_for_recs[df_interactions_for_recs['item_type'] != 'ALL']['item_type'].value_counts()
        print(item_type_counts)
        if not item_type_counts.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=item_type_counts.index, y=item_type_counts.values, 
                        palette=fnb_categorical_palette)
            plt.title(f'CLICK/CHECKOUT Interactions by Item Type ({dataset_name})')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    
    print("\n--- Customer Segment Analysis (All Interactions) (FNB) ---")
    if 'segment' in df.columns:
        segment_counts = df['segment'].value_counts().sort_index()
        print(segment_counts)
        plt.figure(figsize=(10,6))
        custom_segment_palette = [fnb_colors["dark_teal"], fnb_colors["medium_teal"], fnb_colors["light_teal_gray"], fnb_colors["bright_teal"]]
        sns.barplot(x=segment_counts.index, y=segment_counts.values, 
                    palette=custom_segment_palette[:len(segment_counts)])
        plt.title(f'Interactions by Customer Segment ({dataset_name})')
        plt.show()

    if 'beh_segment' in df.columns:
        beh_segment_counts = df['beh_segment'].value_counts()
        print(f"Number of unique behavioral segments: {df['beh_segment'].nunique()}")
        print("Top 10 Behavioral Segments by Interaction Count:")
        print(beh_segment_counts.head(10))
        if len(beh_segment_counts) > 0 and len(beh_segment_counts) <= 15:
            plt.figure(figsize=(12,7))
            sns.barplot(x=beh_segment_counts.head(15).index, y=beh_segment_counts.head(15).values, 
                        palette=fnb_categorical_palette) # Use full palette and let it cycle or truncate
            plt.title(f'Interactions by Behavioral Segment (Top 15) ({dataset_name})')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
        elif len(beh_segment_counts) > 15:
             print("Skipping direct plot for all behavioral segments due to high cardinality (showing top 10 in print).")

    print("\n--- Active Index Analysis (All Interactions) (FNB) ---")
    if 'active_ind' in df.columns:
        active_ind_counts = df['active_ind'].value_counts()
        print(active_ind_counts)
        plt.figure(figsize=(8,5))
        custom_active_palette = [fnb_colors["dark_teal"], fnb_colors["medium_teal"], fnb_colors["light_teal_gray"]]
        sns.barplot(x=active_ind_counts.index, y=active_ind_counts.values, 
                    palette=custom_active_palette[:len(active_ind_counts)])
        plt.title(f'Interactions by Active Index ({dataset_name})')
        plt.show()

        active_interaction_crosstab = pd.crosstab(df['active_ind'], df['interaction'])
        print("\nActive Index vs. Interaction Type:")
        print(active_interaction_crosstab)
        stacked_palette = [fnb_colors["dark_teal"], fnb_colors["medium_teal"], fnb_colors["bright_teal"]]
        active_interaction_crosstab.plot(kind='bar', stacked=True, figsize=(10,7), 
                                         color=stacked_palette)
        plt.title('Interaction Types by Customer Active Index')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Interaction Type')
        plt.show()
    
    if 'item_descrip' in df.columns:
        print("\n--- Item Description Analysis (FNB) ---")
        descriptions = df[df['item'] != 'NONE']['item_descrip'].dropna().astype(str)
        # Use .loc to avoid SettingWithCopyWarning when creating 'descrip_length'
        df_temp_desc = df[df['item'] != 'NONE'].copy()
        df_temp_desc['descrip_length'] = descriptions.apply(len)
        
        if 'descrip_length' in df_temp_desc.columns:
             print("Item description length stats (for actual items):")
             print(df_temp_desc['descrip_length'].describe())
        # Cleanup to avoid adding column to original df if not intended
        del df_temp_desc


    print(f"\n--- FNB Data Analysis Complete ---")
    return df


# Kaggle Dataset Specific Analysis Function 
def preprocess_and_analyze_kaggle_data(df, dataset_name="Kaggle_Retail"):
    print(f"\n--- Preprocessing and Analyzing {dataset_name} Dataset ---")

    df.rename(columns={'CustomerID': 'idcol', 'StockCode': 'item', 'Description': 'item_descrip'}, inplace=True)
    df['idcol'] = df['idcol'].astype(str)
    df['item'] = df['item'].astype(str)

    df_original_rows = len(df)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)] # Handle cancellations
    print(f"Removed {df_original_rows - len(df)} rows due to cancellation invoices.")

    try:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 
    except Exception as e:
        print(f"Warning: Could not parse all dates for Kaggle: {e}. Trying with specific format.")
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M', errors='coerce')
        except: # Final fallback
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    df.dropna(subset=['idcol'], inplace=True) # Drop rows with missing user IDs
    
    df_interactions_for_recs = df[df['Quantity'] > 0].copy() # Positive interactions = purchases
    df_interactions_for_recs['TransactionValue'] = df_interactions_for_recs['Quantity'] * df_interactions_for_recs['UnitPrice']

    print("\n--- Basic Information (Kaggle) ---")
    print(f"Shape of the dataset (after cleaning): {df_interactions_for_recs.shape}") # Use cleaned df
    print(f"Number of unique users (idcol): {df_interactions_for_recs['idcol'].nunique()}")
    print(f"Number of unique items (item): {df_interactions_for_recs['item'].nunique()}")
    if 'InvoiceDate' in df_interactions_for_recs.columns and df_interactions_for_recs['InvoiceDate'].notna().any():
        print(f"Date range: {df_interactions_for_recs['InvoiceDate'].min()} to {df_interactions_for_recs['InvoiceDate'].max()}")
    print(f"Columns: {df_interactions_for_recs.columns.tolist()}")

    num_positive_interactions = len(df_interactions_for_recs)
    num_users_positive = df_interactions_for_recs['idcol'].nunique()
    num_items_positive = df_interactions_for_recs['item'].nunique()

    print(f"\nNumber of purchase interactions (Quantity > 0): {num_positive_interactions}")
    print(f"Number of users with purchases: {num_users_positive}")
    print(f"Number of items purchased: {num_items_positive}")

    if num_users_positive > 0 and num_items_positive > 0:
        sparsity = 1.0 - (num_positive_interactions / (num_users_positive * num_items_positive))
        print(f"Sparsity (purchases): {sparsity:.4f}")
    else:
        print("Not enough purchase data for sparsity calculation.")

    if not df_interactions_for_recs.empty:
        print("\n--- Purchases per User (Kaggle) ---")
        user_purchase_counts = df_interactions_for_recs.groupby('idcol').size()
        print(user_purchase_counts.describe())
        plt.figure(figsize=(10, 6))
        sns.histplot(user_purchase_counts, bins=min(50, user_purchase_counts.nunique()), kde=False, 
                     color=fnb_colors["dark_teal"]) 
        plt.title(f'Distribution of Purchases per User ({dataset_name})')
        plt.xlabel('Number of Purchase Line Items')
        plt.ylabel('Number of Users')
        plt.yscale('log')
        plt.xlim(left=1)
        plt.show()

        print("\n--- Purchases per Item (Kaggle) ---")
        item_purchase_counts = df_interactions_for_recs.groupby('item').size()
        print(item_purchase_counts.describe())
        plt.figure(figsize=(10, 6))
        sns.histplot(item_purchase_counts, bins=min(50, item_purchase_counts.nunique()), kde=False, 
                     color=fnb_colors["medium_teal"])
        plt.title(f'Distribution of Purchases per Item ({dataset_name})')
        plt.xlabel('Number of Times Purchased')
        plt.ylabel('Number of Items')
        plt.yscale('log')
        plt.xlim(left=1)
        plt.show()

        print("\n--- Top Purchased Items (Kaggle) ---")
        top_items_kaggle = item_purchase_counts.sort_values(ascending=False)
        print(f"Top 10 most purchased items ({dataset_name}):")
        print(top_items_kaggle.head(10))
        if 'item_descrip' in df_interactions_for_recs.columns:
             top_items_df_k = top_items_kaggle.head(10).reset_index()
             top_items_df_k.columns = ['item', 'purchase_count']
             top_items_with_desc_kaggle = pd.merge(
                top_items_df_k,
                df_interactions_for_recs[['item', 'item_descrip']].drop_duplicates(subset=['item']),
                on='item'
            )
             print("\nTop 10 items with descriptions:")
             print(top_items_with_desc_kaggle[['item', 'purchase_count', 'item_descrip']])

    if 'InvoiceDate' in df_interactions_for_recs.columns and df_interactions_for_recs['InvoiceDate'].notna().any():
        print("\n--- Temporal Purchase Patterns (Kaggle) ---")
        df_interactions_for_recs.set_index('InvoiceDate')['InvoiceNo'].resample('M').nunique().plot(figsize=(12,5), marker='o', color=fnb_colors["dark_teal"])
        plt.title(f'Transactions Over Time (Monthly) ({dataset_name})')
        plt.ylabel('Number of Transactions')
        plt.xlabel('Date')
        plt.show()

        df_interactions_for_recs['HourOfDay'] = df_interactions_for_recs['InvoiceDate'].dt.hour
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_interactions_for_recs, x='HourOfDay', 
                      palette=sns.color_palette([fnb_colors["dark_teal"]], n_colors=24)) # Use one color, repeated
        plt.title(f'Purchases by Hour of Day ({dataset_name})')
        plt.xlabel('Hour of Day')
        plt.ylabel('Count of Purchased Items')
        plt.tight_layout()
        plt.show()

    if 'Country' in df_interactions_for_recs.columns:
        print("\n--- Purchase Analysis by Country (Kaggle) ---")
        country_counts = df_interactions_for_recs['Country'].value_counts()
        print("Top 10 Countries by Purchase Volume:")
        print(country_counts.head(10))
        
        top_n = 10
        plt.figure(figsize=(12, 7))
        sns.barplot(x=country_counts.head(top_n).index, y=country_counts.head(top_n).values, 
                    palette=fnb_categorical_palette) 
        plt.title(f'Top {top_n} Countries by Purchase Volume ({dataset_name})')
        plt.xlabel('Country')
        plt.ylabel('Number of Purchased Items')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    if 'TransactionValue' in df_interactions_for_recs.columns:
        print("\n--- Transaction Value Analysis (Kaggle) ---")
        print("Overall Transaction Value Stats:")
        print(df_interactions_for_recs['TransactionValue'].describe())

        user_total_value = df_interactions_for_recs.groupby('idcol')['TransactionValue'].sum()
        print("\nUser Total Transaction Value Stats:")
        print(user_total_value.describe())

        plt.figure(figsize=(10,6))
        # Plotting up to 95th percentile for better viz, using a theme color
        sns.histplot(user_total_value[user_total_value < user_total_value.quantile(0.95)], bins=50, color=fnb_colors["dark_teal"]) 
        plt.title('Distribution of Total Transaction Value per User (up to 95th percentile)')
        plt.xlabel('Total Value')
        plt.ylabel('Number of Users')
        plt.show()

    print(f"\n--- Kaggle Data Analysis Complete ---")
    return df_interactions_for_recs


# Main Execution and Comparison Outline 
if __name__ == '__main__':
    # Apply Global Theme Settings 
    sns.set_theme(
        style="whitegrid", 
        rc={
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.titlepad": 20,
            "axes.labelpad": 15,
            "figure.facecolor": fnb_colors["white"],
            "axes.facecolor": fnb_colors["white"],
            "axes.edgecolor": fnb_colors["very_dark_teal"], 
            "axes.grid": True,
            "axes.axisbelow": True,    
            "grid.color": "#EAEAEA",   
            "grid.linestyle": "--",
            "text.color": fnb_colors["very_dark_teal"], 
            "xtick.color": fnb_colors["very_dark_teal"],  
            "ytick.color": fnb_colors["very_dark_teal"],  
            "patch.edgecolor": fnb_colors["very_dark_teal"], 
            "lines.linewidth": 2,
            "font.family": "sans-serif",
            "font.sans-serif": ['Arial', 'DejaVu Sans', 'Verdana', 'Calibri'], 
            "axes.spines.top": False,    # Despine top border
            "axes.spines.right": False   # Despine right border
        }
    )

    # FNB Data 
    fnb_file_path = "../data/fnb_dataset/raw/dq_recsys_challenge_2025(in).csv"  
    try:
        df_fnb_raw = pd.read_csv(fnb_file_path)
        df_fnb_analyzed = preprocess_and_analyze_fnb_data(df_fnb_raw.copy())
    except FileNotFoundError:
        print(f"FNB data file not found at: {fnb_file_path}")
        df_fnb_analyzed = None
    except Exception as e:
        print(f"An error occurred during FNB data processing: {e}")
        df_fnb_analyzed = None

    # Kaggle Data 
    
    kaggle_file_path = "../data/kaggle_dataset/raw/data.csv" 
    try:
        df_kaggle_raw = pd.read_csv(kaggle_file_path, encoding='ISO-8859-1') 
        df_kaggle_analyzed = preprocess_and_analyze_kaggle_data(df_kaggle_raw.copy())
    except FileNotFoundError:
        print(f"Kaggle data file not found at: {kaggle_file_path}")
        df_kaggle_analyzed = None
    except Exception as e:
        print(f"An error occurred during Kaggle data processing: {e}")
        df_kaggle_analyzed = None
        
    print("\n\nCOMPARATIVE ANALYSIS SUMMARY")
    if df_fnb_analyzed is not None and df_kaggle_analyzed is not None:
        # Get positive interaction data from FNB for comparison
        fnb_pos_interactions_df = df_fnb_analyzed[
            df_fnb_analyzed['interaction'].isin(['CLICK', 'CHECKOUT']) & (df_fnb_analyzed['item'] != 'NONE')
        ]
        
        fnb_users = fnb_pos_interactions_df['idcol'].nunique()
        fnb_positive_items = fnb_pos_interactions_df['item'].nunique()
        fnb_positive_interactions_count = len(fnb_pos_interactions_df)
        
        kaggle_users = df_kaggle_analyzed['idcol'].nunique()
        kaggle_items = df_kaggle_analyzed['item'].nunique()
        kaggle_interactions_count = len(df_kaggle_analyzed)

        print("\n1. Core Statistics (based on interactions used for recommendations):")
        print(f"  FNB: Users={fnb_users}, Items (active in CLICK/CHECKOUT)={fnb_positive_items}, CLICK/CHECKOUT Interactions={fnb_positive_interactions_count}")
        if fnb_users > 0 and fnb_positive_items > 0:
             fnb_sparsity = 1.0 - (fnb_positive_interactions_count / (fnb_users * fnb_positive_items))
             print(f"  FNB Sparsity (CLICK/CHECKOUT): {fnb_sparsity:.4f}")

        print(f"  Kaggle: Users={kaggle_users}, Items={kaggle_items}, Purchase Interactions={kaggle_interactions_count}")
        if kaggle_users > 0 and kaggle_items > 0:
            kaggle_sparsity = 1.0 - (kaggle_interactions_count / (kaggle_users * kaggle_items))
            print(f"  Kaggle Sparsity (Purchases): {kaggle_sparsity:.4f}")

        print("\n2. Nature of Interactions:")
        print("  FNB: Multiple types (DISPLAY, CLICK, CHECKOUT) - richer signal of intent. Explicit 'NONE' item for display-only.")
        print("  Kaggle: Implicit (Purchase via Quantity > 0). Cancellations identifiable.")

        print("\n3. Item Information:")
        print("  FNB: `item_type` (TRANSACT, LEND etc.), `item_descrip`. Structured categories.")
        print("  Kaggle: `item_descrip` (product name).")

        print("\n4. User Information / Context:")
        print("  FNB: `segment`, `beh_segment` (50 detailed types), `active_ind`, `page`, `tod`. Rich user context.")
        print("  Kaggle: `Country`. Limited explicit user segmentation.")

        print("\n5. Temporal Granularity:")
        print("  FNB: `int_date` (daily), `tod` (4 slots).")
        print("  Kaggle: `InvoiceDate` (timestamp - date and time).")

        print("\n6. Business Objective Relevance:")
        print("  FNB: Offers are financial products/services. Goal: Maximize LTV, engagement across diverse offer types.")
        print("  Kaggle: Retail products. Goal: Typically sales volume, predict next purchase.")
        
        print("\n7. Key Differences for Recommender Systems:")
        print("  - FNB's richer interaction types allow for modeling user journey (e.g. view -> click -> checkout).")
        print("  - FNB's user segmentation can be directly fed into user tower of a two-tower model for better personalization.")
        print("  - Kaggle's monetary value (`UnitPrice`, `TransactionValue`) allows for value-based recommendations or evaluation, FNB lacks this directly.")
        print("  - Cold-start for FNB might leverage `item_type` and user segments more effectively.")
        print("  - FNB's 'DISPLAY' with 'NONE' item indicates general activity vs specific item interest - distinct signal.")

    else:
        print("One or both datasets were not loaded/analyzed successfully. Comparison skipped.")

    print("\nEnd of Analysis.")