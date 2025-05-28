import re
import pandas as pd

# --- Paste your Temporal Split Results Here ---
temporal_results_text = """
--- Evaluating on Test Set ---

--- Segment: All Test Users (24686 users) ---
Evaluating Main Model for All Test Users...
Test Main (All Test Users): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:27<00:00,  2.09s/it]
  Recall@10: 0.4033
  Precision@10: 0.0403
  NDCG@10: 0.2185
  HitRate@10: 0.4033
  MRR@10: 0.1634
  Coverage@10: 0.4231
  ILD@10: 0.5906
  Novelty@10: 5.2407
  Fairness_Gini@10: 0.8325
Random Baseline (All Test Users) Metrics:
  Recall@10: 0.0937
  Precision@10: 0.0094
  NDCG@10: 0.0422
  HitRate@10: 0.0937
  MRR@10: 0.0270
  Coverage@10: 1.0000
  ILD@10: 0.7694
  Novelty@10: 8.2042
  Fairness_Gini@10: 0.0130
Prev. Bought Baseline (All Test Users) Metrics:
  Recall@10: 1.0000
  Precision@10: 0.4664
  NDCG@10: 0.9585
  HitRate@10: 1.0000
  MRR@10: 0.9441
  Coverage@10: 0.9904
  ILD@10: 0.5878
  Novelty@10: 5.9525
  Fairness_Gini@10: 0.5461
Popularity Baseline (All Test Users) Metrics:
  Recall@10: 0.3517
  Precision@10: 0.0352
  NDCG@10: 0.1973
  HitRate@10: 0.3517
  MRR@10: 0.1512
  Coverage@10: 0.0962
  ILD@10: 0.4921
  Novelty@10: 4.8784
  Fairness_Gini@10: 0.9038

--- Segment: Overall Cold Start (0 Positive Interactions Globally) - No test users. Skipping.

--- Segment: Training Cold Start (0 Positive Train Interactions this Split) - No test users. Skipping.

--- Segment: 1-2 Pos Interactions Globally - No test users. Skipping.

--- Segment: 3-5 Pos Interactions Globally (12355 users) ---
Evaluating Main Model for 3-5 Pos Interactions Globally...
Test Main (3-5 Pos Interactions Globally): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:10<00:00,  1.51s/it]
  Recall@10: 0.3781
  Precision@10: 0.0378
  NDCG@10: 0.1877
  HitRate@10: 0.3781
  MRR@10: 0.1311
  Coverage@10: 0.4135
  ILD@10: 0.5905
  Novelty@10: 5.2420
  Fairness_Gini@10: 0.8317
Random Baseline (3-5 Pos Interactions Globally) Metrics:
  Recall@10: 0.0935
  Precision@10: 0.0093
  NDCG@10: 0.0423
  HitRate@10: 0.0935
  MRR@10: 0.0271
  Coverage@10: 1.0000
  ILD@10: 0.7670
  Novelty@10: 8.2248
  Fairness_Gini@10: 0.0156
Prev. Bought Baseline (3-5 Pos Interactions Globally) Metrics:
  Recall@10: 1.0000
  Precision@10: 0.6207
  NDCG@10: 0.9658
  HitRate@10: 1.0000
  MRR@10: 0.9538
  Coverage@10: 0.9904
  ILD@10: 0.4931
  Novelty@10: 6.0197
  Fairness_Gini@10: 0.5424
Popularity Baseline (3-5 Pos Interactions Globally) Metrics:
  Recall@10: 0.3255
  Precision@10: 0.0325
  NDCG@10: 0.1646
  HitRate@10: 0.3255
  MRR@10: 0.1169
  Coverage@10: 0.0962
  ILD@10: 0.4921
  Novelty@10: 4.8784
  Fairness_Gini@10: 0.9038

--- Segment: 6-25 Pos Interactions Globally (11558 users) ---
Evaluating Main Model for 6-25 Pos Interactions Globally...
Test Main (6-25 Pos Interactions Globally): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:10<00:00,  1.68s/it]
  Recall@10: 0.4278
  Precision@10: 0.0428
  NDCG@10: 0.2468
  HitRate@10: 0.4278
  MRR@10: 0.1928
  Coverage@10: 0.4135
  ILD@10: 0.5919
  Novelty@10: 5.2435
  Fairness_Gini@10: 0.8330
Random Baseline (6-25 Pos Interactions Globally) Metrics:
  Recall@10: 0.0958
  Precision@10: 0.0096
  NDCG@10: 0.0438
  HitRate@10: 0.0958
  MRR@10: 0.0284
  Coverage@10: 1.0000
  ILD@10: 0.7680
  Novelty@10: 8.2050
  Fairness_Gini@10: 0.0165
Prev. Bought Baseline (6-25 Pos Interactions Globally) Metrics:
  Recall@10: 1.0000
  Precision@10: 0.3216
  NDCG@10: 0.9504
  HitRate@10: 1.0000
  MRR@10: 0.9332
  Coverage@10: 0.9808
  ILD@10: 0.6803
  Novelty@10: 5.8859
  Fairness_Gini@10: 0.5500
Popularity Baseline (6-25 Pos Interactions Globally) Metrics:
  Recall@10: 0.3773
  Precision@10: 0.0377
  NDCG@10: 0.2276
  HitRate@10: 0.3773
  MRR@10: 0.1826
  Coverage@10: 0.0962
  ILD@10: 0.4921
  Novelty@10: 4.8784
  Fairness_Gini@10: 0.9038

--- Segment: >25 Pos Interactions Globally (773 users) ---
Evaluating Main Model for >25 Pos Interactions Globally...
Test Main (>25 Pos Interactions Globally): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.66it/s]
  Recall@10: 0.4386
  Precision@10: 0.0439
  NDCG@10: 0.2873
  HitRate@10: 0.4386
  MRR@10: 0.2417
  Coverage@10: 0.3462
  ILD@10: 0.5747
  Novelty@10: 5.1796
  Fairness_Gini@10: 0.8412
Random Baseline (>25 Pos Interactions Globally) Metrics:
  Recall@10: 0.1203
  Precision@10: 0.0120
  NDCG@10: 0.0576
  HitRate@10: 0.1203
  MRR@10: 0.0390
  Coverage@10: 1.0000
  ILD@10: 0.7699
  Novelty@10: 8.2053
  Fairness_Gini@10: 0.0574
Prev. Bought Baseline (>25 Pos Interactions Globally) Metrics:
  Recall@10: 0.9987
  Precision@10: 0.1636
  NDCG@10: 0.9633
  HitRate@10: 0.9987
  MRR@10: 0.9510
  Coverage@10: 0.9519
  ILD@10: 0.7172
  Novelty@10: 5.8741
  Fairness_Gini@10: 0.5657
Popularity Baseline (>25 Pos Interactions Globally) Metrics:
  Recall@10: 0.3881
  Precision@10: 0.0388
  NDCG@10: 0.2650
  HitRate@10: 0.3881
  MRR@10: 0.2278
  Coverage@10: 0.0962
  ILD@10: 0.4921
  Novelty@10: 4.8784
  Fairness_Gini@10: 0.9038
"""

# --- Paste your User Cold-Start Split Results Here ---
user_cold_start_results_text = """
--- Evaluating on Test Set ---

--- Segment: All Test Users (12657 users) ---
Evaluating Main Model for All Test Users...
Test Main (All Test Users): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:09<00:00,  1.31s/it]
  Recall@10: 0.1899
  Precision@10: 0.0408
  NDCG@10: 0.1167
  HitRate@10: 0.2796
  MRR@10: 0.1212
  Coverage@10: 0.3846
  ILD@10: 0.5877
  Novelty@10: 5.2173
  Fairness_Gini@10: 0.8420
Random Baseline (All Test Users) Metrics:
  Recall@10: 0.0994
  Precision@10: 0.0218
  NDCG@10: 0.0544
  HitRate@10: 0.1931
  MRR@10: 0.0575
  Coverage@10: 1.0000
  ILD@10: 0.7099
  Novelty@10: 8.1657
  Fairness_Gini@10: 0.0236
Prev. Bought Baseline (All Test Users) Metrics:
  Recall@10: 0.9988
  Precision@10: 1.0000
  NDCG@10: 1.0000
  HitRate@10: 1.0000
  MRR@10: 1.0000
  Coverage@10: 0.9808
  ILD@10: 0.3581
  Novelty@10: 6.0089
  Fairness_Gini@10: 0.5429
Popularity Baseline (All Test Users) Metrics:
  Recall@10: 0.3339
  Precision@10: 0.0710
  NDCG@10: 0.2084
  HitRate@10: 0.4992
  MRR@10: 0.2230
  Coverage@10: 0.0962
  ILD@10: 0.5413
  Novelty@10: 4.8961
  Fairness_Gini@10: 0.9038

--- Segment: Overall Cold Start (0 Positive Interactions Globally) (6299 users) ---
Evaluating Main Model for Overall Cold Start (0 Positive Interactions Globally)...
Test Main (Overall Cold Start (0 Positive Interactions Globally)): 100%|████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.05it/s]
  Recall@10: 0.0000
  Precision@10: 0.0000
  NDCG@10: 0.0000
  HitRate@10: 0.0000
  MRR@10: 0.0000
  Coverage@10: 0.3750
  ILD@10: 0.5938
  Novelty@10: 5.2154
  Fairness_Gini@10: 0.8416
Warning (Random (Overall Cold Start (0 Positive Interactions Globally))): ground_truth_segment_df is None or empty. Accuracy metrics (Recall, Prec, etc.) will be 0.
Random Baseline (Overall Cold Start (0 Positive Interactions Globally)) Metrics:
  Recall@10: 0.0000
  Precision@10: 0.0000
  NDCG@10: 0.0000
  HitRate@10: 0.0000
  MRR@10: 0.0000
  Coverage@10: 1.0000
  ILD@10: 0.7107
  Novelty@10: 8.1548
  Fairness_Gini@10: 0.0231
Warning (PrevBought (Overall Cold Start (0 Positive Interactions Globally))): ground_truth_segment_df is None or empty. Accuracy metrics (Recall, Prec, etc.) will be 0.
Prev. Bought Baseline (Overall Cold Start (0 Positive Interactions Globally)) Metrics:
  Recall@10: 0.0000
  Precision@10: 0.0000
  NDCG@10: 0.0000
  HitRate@10: 0.0000
  MRR@10: 0.0000
  Coverage@10: 0.0000
  ILD@10: 0.0000
  Novelty@10: 0.0000
  Fairness_Gini@10: 0.0000
Popularity Baseline (Overall Cold Start (0 Positive Interactions Globally)) Metrics:
  Recall@10: 0.0000
  Precision@10: 0.0000
  NDCG@10: 0.0000
  HitRate@10: 0.0000
  MRR@10: 0.0000
  Coverage@10: 0.0000
  ILD@10: 0.0000
  Novelty@10: 0.0000
  Fairness_Gini@10: 0.0000

--- Segment: Training Cold Start (0 Positive Train Interactions this Split) (12657 users) ---
Evaluating Main Model for Training Cold Start (0 Positive Train Interactions this Split)...
Test Main (Training Cold Start (0 Positive Train Interactions this Split)): 100%|██████████████████████████████████████████████████████████████████████████████████| 7/7 [00:09<00:00,  1.29s/it]
  Recall@10: 0.1899
  Precision@10: 0.0408
  NDCG@10: 0.1167
  HitRate@10: 0.2796
  MRR@10: 0.1212
  Coverage@10: 0.3846
  ILD@10: 0.5877
  Novelty@10: 5.2173
  Fairness_Gini@10: 0.8420
Random Baseline (Training Cold Start (0 Positive Train Interactions this Split)) Metrics:
  Recall@10: 0.1009
  Precision@10: 0.0216
  NDCG@10: 0.0559
  HitRate@10: 0.1914
  MRR@10: 0.0599
  Coverage@10: 1.0000
  ILD@10: 0.7126
  Novelty@10: 8.1529
  Fairness_Gini@10: 0.0243
Prev. Bought Baseline (Training Cold Start (0 Positive Train Interactions this Split)) Metrics:
  Recall@10: 0.9988
  Precision@10: 1.0000
  NDCG@10: 1.0000
  HitRate@10: 1.0000
  MRR@10: 1.0000
  Coverage@10: 0.9808
  ILD@10: 0.3581
  Novelty@10: 6.0089
  Fairness_Gini@10: 0.5429
Popularity Baseline (Training Cold Start (0 Positive Train Interactions this Split)) Metrics:
  Recall@10: 0.3339
  Precision@10: 0.0710
  NDCG@10: 0.2084
  HitRate@10: 0.4992
  MRR@10: 0.2230
  Coverage@10: 0.0962
  ILD@10: 0.5413
  Novelty@10: 4.8961
  Fairness_Gini@10: 0.9038

--- Segment: 1-2 Pos Interactions Globally (2663 users) ---
Evaluating Main Model for 1-2 Pos Interactions Globally...
Test Main (1-2 Pos Interactions Globally): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.05s/it]
  Recall@10: 0.3537
  Precision@10: 0.0374
  NDCG@10: 0.1717
  HitRate@10: 0.3661
  MRR@10: 0.1196
  Coverage@10: 0.3654
  ILD@10: 0.5856
  Novelty@10: 5.2121
  Fairness_Gini@10: 0.8434
Random Baseline (1-2 Pos Interactions Globally) Metrics:
  Recall@10: 0.0956
  Precision@10: 0.0101
  NDCG@10: 0.0421
  HitRate@10: 0.1003
  MRR@10: 0.0276
  Coverage@10: 1.0000
  ILD@10: 0.7107
  Novelty@10: 8.1704
  Fairness_Gini@10: 0.0308
Prev. Bought Baseline (1-2 Pos Interactions Globally) Metrics:
  Recall@10: 1.0000
  Precision@10: 1.0000
  NDCG@10: 1.0000
  HitRate@10: 1.0000
  MRR@10: 1.0000
  Coverage@10: 0.9231
  ILD@10: 0.0400
  Novelty@10: 6.1094
  Fairness_Gini@10: 0.5464
Popularity Baseline (1-2 Pos Interactions Globally) Metrics:
  Recall@10: 0.3085
  Precision@10: 0.0326
  NDCG@10: 0.1505
  HitRate@10: 0.3207
  MRR@10: 0.1052
  Coverage@10: 0.0962
  ILD@10: 0.5413
  Novelty@10: 4.8961
  Fairness_Gini@10: 0.9038

--- Segment: 3-5 Pos Interactions Globally (1849 users) ---
Evaluating Main Model for 3-5 Pos Interactions Globally...
Test Main (3-5 Pos Interactions Globally): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.43s/it]
  Recall@10: 0.3896
  Precision@10: 0.0711
  NDCG@10: 0.2276
  HitRate@10: 0.5663
  MRR@10: 0.2116
  Coverage@10: 0.3654
  ILD@10: 0.5812
  Novelty@10: 5.2169
  Fairness_Gini@10: 0.8437
Random Baseline (3-5 Pos Interactions Globally) Metrics:
  Recall@10: 0.0895
  Precision@10: 0.0167
  NDCG@10: 0.0501
  HitRate@10: 0.1606
  MRR@10: 0.0510
  Coverage@10: 1.0000
  ILD@10: 0.7048
  Novelty@10: 8.2070
  Fairness_Gini@10: 0.0422
Prev. Bought Baseline (3-5 Pos Interactions Globally) Metrics:
  Recall@10: 1.0000
  Precision@10: 1.0000
  NDCG@10: 1.0000
  HitRate@10: 1.0000
  MRR@10: 1.0000
  Coverage@10: 0.9327
  ILD@10: 0.4947
  Novelty@10: 5.9875
  Fairness_Gini@10: 0.5539
Popularity Baseline (3-5 Pos Interactions Globally) Metrics:
  Recall@10: 0.3432
  Precision@10: 0.0617
  NDCG@10: 0.1998
  HitRate@10: 0.4997
  MRR@10: 0.1866
  Coverage@10: 0.0962
  ILD@10: 0.5413
  Novelty@10: 4.8961
  Fairness_Gini@10: 0.9038

--- Segment: >5 Pos Interactions Globally (1846 users) ---
Evaluating Main Model for >5 Pos Interactions Globally...
Test Main (>5 Pos Interactions Globally): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.43s/it]
  Recall@10: 0.4018
  Precision@10: 0.1546
  NDCG@10: 0.3248
  HitRate@10: 0.8218
  MRR@10: 0.4464
  Coverage@10: 0.3558
  ILD@10: 0.5762
  Novelty@10: 5.2318
  Fairness_Gini@10: 0.8434
Random Baseline (>5 Pos Interactions Globally) Metrics:
  Recall@10: 0.0943
  Precision@10: 0.0395
  NDCG@10: 0.0674
  HitRate@10: 0.3321
  MRR@10: 0.1054
  Coverage@10: 1.0000
  ILD@10: 0.7081
  Novelty@10: 8.1705
  Fairness_Gini@10: 0.0376
Prev. Bought Baseline (>5 Pos Interactions Globally) Metrics:
  Recall@10: 0.9957
  Precision@10: 1.0000
  NDCG@10: 1.0000
  HitRate@10: 1.0000
  MRR@10: 1.0000
  Coverage@10: 0.9615
  ILD@10: 0.6801
  Novelty@10: 5.8854
  Fairness_Gini@10: 0.5474
Popularity Baseline (>5 Pos Interactions Globally) Metrics:
  Recall@10: 0.3612
  Precision@10: 0.1355
  NDCG@10: 0.3005
  HitRate@10: 0.7562
  MRR@10: 0.4292
  Coverage@10: 0.0962
  ILD@10: 0.5413
  Novelty@10: 4.8961
  Fairness_Gini@10: 0.9038
"""

def parse_results(text_block, split_type_name):
    """
    Parses a block of text results into a list of dictionaries.
    """
    parsed_data = []
    current_segment = None
    current_model = None

    # Define how raw metric names map to the desired shorter names
    metric_name_map = {
        "Recall@10": "Recall",
        "Precision@10": "Precision",
        "NDCG@10": "NDCG",
        "HitRate@10": "HR",
        "MRR@10": "MRR",
        "Coverage@10": "Coverage",
        "ILD@10": "ILD",
        "Novelty@10": "Novelty",
        "Fairness_Gini@10": "GINI"
    }

    lines = text_block.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Check for segment line
        segment_match = re.match(r"--- Segment: (.*?) \(([\d,]+ users)?\)", line)
        if segment_match:
            current_segment = segment_match.group(1).strip()
            current_model = None # Reset model when segment changes
            # print(f"Segment: {current_segment}") # for debugging
            continue

        segment_skip_match = re.match(r"--- Segment: (.*?) - No test users. Skipping.", line)
        if segment_skip_match:
            # For skipped segments, we could add an entry or just ignore.
            # To match the Excel image, we'll ignore them for now.
            # If you want to record them:
            # parsed_data.append({
            #     'Split': split_type_name,
            #     'Segment': segment_skip_match.group(1).strip(),
            #     'Model': 'N/A',
            #     'Metric': 'Status',
            #     'Score': 'Skipped'
            # })
            current_segment = None # Reset segment
            current_model = None
            # print(f"Skipped Segment: {segment_skip_match.group(1).strip()}") # for debugging
            continue

        # Check for model lines
        main_model_eval_match = re.match(r"Evaluating Main Model for (.*?)...", line)
        if main_model_eval_match:
            current_model = "Main Model"
            # print(f"Model: {current_model}") # for debugging
            continue
        
        main_model_test_match = re.match(r"Test Main \((.*?)\):", line) # Catches the line before metrics
        if main_model_test_match:
            current_model = "Main Model" # In case "Evaluating..." line was missed or structure varies
            # The segment name here should ideally match current_segment
            # print(f"Model (from Test Main): {current_model} for segment {main_model_test_match.group(1).strip()}")
            continue

        baseline_match = re.match(r"(Random|Prev\. Bought|Popularity) Baseline \((.*?)\) Metrics:", line)
        if baseline_match:
            model_name_prefix = baseline_match.group(1)
            if model_name_prefix == "Prev. Bought":
                 current_model = "Prev. Bought Baseline"
            else:
                 current_model = f"{model_name_prefix} Baseline"

            # The segment name here should ideally match current_segment
            # print(f"Model: {current_model} for segment {baseline_match.group(2).strip()}")
            continue
        
        # Check for metric lines
        metric_match = re.match(r"([A-Za-z_@0-9]+):\s*([-\d.]+)", line)
        if metric_match and current_segment and current_model:
            raw_metric_name = metric_match.group(1)
            score_str = metric_match.group(2)

            if raw_metric_name in metric_name_map:
                metric_name = metric_name_map[raw_metric_name]
                try:
                    score = float(score_str)
                    parsed_data.append({
                        'Split': split_type_name,
                        'Segment': current_segment,
                        'Model': current_model,
                        'Metric': metric_name,
                        'Score': score
                    })
                except ValueError:
                    print(f"Warning: Could not convert score '{score_str}' to float for metric '{raw_metric_name}'")
            # else:
                # print(f"Debug: Unmapped metric '{raw_metric_name}' or context not set (Segment: {current_segment}, Model: {current_model})")


    return parsed_data

# --- Main script execution ---
all_parsed_results = []

# Parse Temporal Results
print("Parsing Temporal Results...")
temporal_parsed = parse_results(temporal_results_text, "Temporal")
all_parsed_results.extend(temporal_parsed)

# Parse User Cold-Start Results
print("\nParsing User Cold-Start Results...")
cold_start_parsed = parse_results(user_cold_start_results_text, "User Cold-Start")
all_parsed_results.extend(cold_start_parsed)

# Create DataFrame
df_results = pd.DataFrame(all_parsed_results)

# Display the DataFrame (optional)
print("\n--- Parsed DataFrame ---")
print(df_results.to_string())

# Save to CSV and Excel
try:
    df_results.to_csv("parsed_model_results.csv", index=False)
    df_results.to_excel("parsed_model_results.xlsx", index=False)
    print("\nResults saved to parsed_model_results.csv and parsed_model_results.xlsx")
    print("\nCSV Output for easy copy-paste into Excel (if preferred over opening the file):")
    print("--------------------------------------------------------------------------------")
    print(df_results.to_csv(index=False))
    print("--------------------------------------------------------------------------------")

except Exception as e:
    print(f"\nError saving files: {e}")
    print("Make sure you have 'pandas' and 'openpyxl' (for .xlsx) installed.")
    print("You can install openpyxl with: pip install openpyxl")