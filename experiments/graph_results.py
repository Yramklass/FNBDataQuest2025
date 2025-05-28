import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

# --- Color Palette Definition ---
fnb_colors = {
    "dark_teal": "#116e71",
    "bright_teal": "#43d8c6",
    "very_dark_teal": "#0a3f3a",
    "medium_teal": "#21746f",
    "light_teal_gray": "#80bfac",
    "black": "#000000",
    "white": "#FFFFFF"
}

fnb_categorical_palette = [
    fnb_colors["dark_teal"],
    fnb_colors["bright_teal"],
    fnb_colors["light_teal_gray"],
    fnb_colors["medium_teal"],
    fnb_colors["very_dark_teal"]
]

# Apply a general style for plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial' # Using a commonly available font


# --- Graphing Functions ---

def plot_graph_1(df):
    """
    Graph 1: Compare performance of Main model, Popularity baseline and Random baseline
    on recall and ndcg for temporal split - for All Test Users.
    """
    title = "Graph 1: Main, Popularity & Random Baselines - Recall & NDCG\n(Temporal Split, All Test Users)"
    metrics_to_plot = ["Recall", "NDCG"]
    models_to_plot = ["Main Model", "Popularity Baseline", "Random Baseline"]
    
    # Filter data
    filtered_df = df[
        (df['Split'] == 'Temporal') &
        (df['Segment'] == 'All Test Users') &
        (df['Model'].isin(models_to_plot)) &
        (df['Metric'].isin(metrics_to_plot))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning if any modifications were to happen
    
    if filtered_df.empty:
        print(f"No data for Graph 1 with current filters. Check CSV content and column names.")
        print(f"Expected models: {models_to_plot}, metrics: {metrics_to_plot}, split: Temporal, segment: All Test Users")
        return

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x='Metric', 
        y='Score', 
        hue='Model', 
        data=filtered_df, 
        palette=fnb_categorical_palette[:len(models_to_plot)],
        hue_order=models_to_plot
    )
    
    plt.title(title, fontsize=15, color=fnb_colors["very_dark_teal"])
    plt.xlabel("Metric", fontsize=12, color=fnb_colors["very_dark_teal"])
    plt.ylabel("Score", fontsize=12, color=fnb_colors["very_dark_teal"])
    plt.ylim(0, max(filtered_df['Score'].max() * 1.15 if not filtered_df.empty and filtered_df['Score'].max() > 0 else 0.1, 0.1)) 
    
    # Add score labels on bars
    for p in ax.patches:
        # Only annotate if the bar height is greater than a small threshold
        if p.get_height() > 1e-4: 
            ax.annotate(format(p.get_height(), '.3f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=9, color=fnb_colors["black"])

    plt.legend(title='Model', loc='upper right', frameon=True, facecolor=fnb_colors["white"], edgecolor=fnb_colors["light_teal_gray"])
    plt.tight_layout()
    plt.show()

def plot_graph_2(df):
    """
    Graph 2: Compare NDCG, HR of main model and popularity baseline
    for 3-5 and >25 segments - on temporal set.
    """
    title = "Graph 2: Main Model vs Popularity - NDCG & HR by Segment\n(Temporal Split)"
    metrics_to_plot = ["NDCG", "HR"]
    models_to_plot = ["Main Model", "Popularity Baseline"]
    segments_to_plot = ["3-5 Pos Interactions Globally", ">25 Pos Interactions Globally"]
    
    # Filter data
    filtered_df = df[
        (df['Split'] == 'Temporal') &
        (df['Segment'].isin(segments_to_plot)) &
        (df['Model'].isin(models_to_plot)) &
        (df['Metric'].isin(metrics_to_plot))
    ].copy()

    if filtered_df.empty:
        print(f"No data for Graph 2 with current filters. Check CSV content and column names.")
        return

    g = sns.catplot(
        x='Metric', 
        y='Score', 
        hue='Model', 
        col='Segment', 
        data=filtered_df, 
        kind='bar', 
        palette=fnb_categorical_palette[:len(models_to_plot)],
        hue_order=models_to_plot,
        col_order=segments_to_plot,
        height=5, aspect=1.1,
        legend=False # Disable automatic legend from catplot as we add a custom one
    )
    
    # Adjust suptitle position and use subplots_adjust to make space
    g.fig.suptitle(title, y=0.97, fontsize=15, color=fnb_colors["very_dark_teal"]) # Adjusted y
    g.set_axis_labels("Metric", "Score")
    g.set_titles("{col_name}", size=12, color=fnb_colors["medium_teal"])
    
    # Add score labels on bars
    for ax_col_idx, segment_name in enumerate(g.col_names): 
        ax = g.axes[0, ax_col_idx] 
        max_score_in_ax = 0
        current_subplot_data = filtered_df[(filtered_df['Segment'] == segment_name)]
        if not current_subplot_data.empty:
            max_score_in_ax = current_subplot_data['Score'].max()

        for p in ax.patches:
            if p.get_height() > 1e-4:
                ax.annotate(format(p.get_height(), '.3f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize=9, color=fnb_colors["black"])
        ax.set_ylim(0, max(max_score_in_ax * 1.15 if max_score_in_ax > 0 else 0.1, 0.1))

    # Adjust legend
    handles, labels = g.axes[0,0].get_legend_handles_labels() 
    if handles: 
        g.fig.legend(handles=handles, labels=labels, title='Model', loc='upper right', 
                    bbox_to_anchor=(0.98, 0.90), # Adjusted anchor slightly if needed
                    frameon=True, 
                    facecolor=fnb_colors["white"], edgecolor=fnb_colors["light_teal_gray"])
    
    # Adjust layout to prevent title cutoff
    g.fig.subplots_adjust(top=0.88) # Make space for the suptitle and legend
    g.tight_layout() # Apply tight_layout to the FacetGrid
    plt.show()

def plot_graph_3(df):
    """
    Graph 3: Compare novelty, ILD, coverage, gini of popularity baseline and main model
    on temporal split - for all users.
    """
    title = "Graph 3: Main Model vs Popularity - Beyond-Accuracy Metrics\n(Temporal Split, All Test Users)"
    metrics_to_plot = ["Novelty", "ILD", "Coverage", "GINI"]
    models_to_plot = ["Main Model", "Popularity Baseline"]
    
    # Filter data
    filtered_df = df[
        (df['Split'] == 'Temporal') &
        (df['Segment'] == 'All Test Users') &
        (df['Model'].isin(models_to_plot)) &
        (df['Metric'].isin(metrics_to_plot))
    ].copy()

    if filtered_df.empty:
        print(f"No data for Graph 3 with current filters. Check CSV content and column names.")
        return

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        x='Metric', 
        y='Score', 
        hue='Model', 
        data=filtered_df, 
        palette=fnb_categorical_palette[:len(models_to_plot)],
        hue_order=models_to_plot,
        order=metrics_to_plot 
    )
    
    plt.title(title, fontsize=15, color=fnb_colors["very_dark_teal"])
    plt.xlabel("Metric", fontsize=12, color=fnb_colors["very_dark_teal"])
    plt.ylabel("Score", fontsize=12, color=fnb_colors["very_dark_teal"])
    
    max_score_overall = filtered_df['Score'].max() if not filtered_df.empty else 0
    if not filtered_df.empty and max_score_overall > 0 :
        plt.ylim(0, max_score_overall * 1.15)
    else:
        plt.ylim(0, 1)

    for p in ax.patches:
        if p.get_height() > 1e-4: 
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=9, color=fnb_colors["black"])

    plt.legend(title='Model', loc='upper right', frameon=True, facecolor=fnb_colors["white"], edgecolor=fnb_colors["light_teal_gray"])
    plt.tight_layout()
    plt.show()

def plot_graph_4(df):
    """
    Graph 4: Comparing Main Model vs. Popularity for "All Test Users" / "Training Cold Start"
    on metrics like NDCG@10, HitRate@10, Novelty@10, Fairness_Gini@10.
    Using User Cold Start split as per user's code selection.
    """
    title = "Graph 4: Main Model vs Popularity - @10 Metrics by Segment\n(User Cold Start Split)"
    metrics_to_plot = ["NDCG", "HR", "Novelty", "ILD"] # These are assumed to be @10 from context
    models_to_plot = ["Main Model", "Popularity Baseline"]
    segments_to_plot = ["All Test Users", "Training Cold Start (0 Positive Train Interactions this Split)"]
    split_to_use = 'User Cold-Start' 
    
    # Filter data
    filtered_df = df[
        (df['Split'] == split_to_use) & 
        (df['Segment'].isin(segments_to_plot)) &
        (df['Model'].isin(models_to_plot)) &
        (df['Metric'].isin(metrics_to_plot))
    ].copy()

    if filtered_df.empty:
        print(f"No data for Graph 4 with current filters. Check CSV content and column names.")
        print(f"Expected models: {models_to_plot}, metrics: {metrics_to_plot}, segments: {segments_to_plot}, split: {split_to_use}")
        return

    g = sns.catplot(
        x='Metric', 
        y='Score', 
        hue='Model', 
        col='Segment', 
        data=filtered_df, 
        kind='bar', 
        palette=fnb_categorical_palette[:len(models_to_plot)],
        hue_order=models_to_plot,
        col_order=segments_to_plot,
        order=metrics_to_plot, 
        height=6, aspect=1.0,
        legend=False 
    )
    
    g.fig.suptitle(title, y=0.97, fontsize=16, color=fnb_colors["very_dark_teal"]) # Adjusted y
    g.set_axis_labels("Metric (@10)", "Score") # Assuming these are @10 metrics
    g.set_titles("{col_name}", size=13, color=fnb_colors["medium_teal"])
    
    # Add score labels on bars
    for ax_col_idx, segment_name in enumerate(g.col_names):
        ax = g.axes[0, ax_col_idx]
        max_score_in_ax = 0
        current_subplot_data = filtered_df[(filtered_df['Segment'] == segment_name)]
        if not current_subplot_data.empty:
            max_score_in_ax = current_subplot_data['Score'].max()

        for p in ax.patches:
            if p.get_height() > 1e-4:
                ax.annotate(format(p.get_height(), '.3f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize=9, color=fnb_colors["black"])
        ax.set_ylim(0, max(max_score_in_ax * 1.15 if max_score_in_ax > 0 else 0.1, 0.1))
        ax.tick_params(axis='x', rotation=25, labelsize=9) 

    # Adjust legend
    handles, labels = g.axes[0,0].get_legend_handles_labels()
    if handles:
        g.fig.legend(handles=handles, labels=labels, title='Model', loc='upper right', 
                    bbox_to_anchor=(0.98, 0.90), # Adjusted anchor slightly if needed
                    frameon=True, 
                    facecolor=fnb_colors["white"], edgecolor=fnb_colors["light_teal_gray"])
            
    # Adjust layout to prevent title cutoff
    g.fig.subplots_adjust(top=0.85, bottom=0.15) # Make space for suptitle, legend, and rotated x-labels
    g.tight_layout() # Apply tight_layout to the FacetGrid
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    main_df = pd.DataFrame() # Initialize an empty DataFrame
    try:
        main_df = pd.read_csv('parsed_model_results.csv')
        print("Successfully loaded 'parsed_model_results.csv'")
    except FileNotFoundError:
        print("File 'parsed_model_results.csv' not found. Please ensure the file is in the same directory as the script.")
    except pd.errors.EmptyDataError:
        print("File 'parsed_model_results.csv' is empty. Cannot generate plots.")
    except Exception as e:
        print(f"An error occurred while loading 'parsed_model_results.csv': {e}")


    if main_df.empty:
        print("DataFrame is empty. Cannot generate plots. Exiting.")
    else:
        print("DataFrame loaded successfully.")
        # print("DataFrame Head:")
        # print(main_df.head())
        # print(f"\nUnique models: {main_df['Model'].unique()}")
        # print(f"Unique metrics: {main_df['Metric'].unique()}")
        # print(f"Unique segments for Temporal split: {main_df[main_df['Split'] == 'Temporal']['Segment'].unique()}")
        # print(f"Unique segments for User Cold-Start split: {main_df[main_df['Split'] == 'User Cold Start']['Segment'].unique()}")


        print("\nGenerating Graph 1...")
        plot_graph_1(main_df) 
        
        print("\nGenerating Graph 2...")
        plot_graph_2(main_df)
        
        print("\nGenerating Graph 3...")
        plot_graph_3(main_df)
        
        print("\nGenerating Graph 4...")
        plot_graph_4(main_df)
        
        print("\nAll graphs generated (or attempted).")
