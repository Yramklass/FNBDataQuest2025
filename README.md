# FNB Dataquest 2025

## Overview
This project focuses on developing and evaluating a recommender system to provide personalized product and service suggestions for FNB customers. The primary model is a Two-Tower architecture incorporating Transformer encoders for feature processing. The project also includes data analysis and comparison with a benchmark dataset (UCI Online Retail).

## Main Model
The main model, which represents the most developed and best-performing solution, is located in:
* `tt_trans_neg.py`
The fully-trained model weights are available in .onnx format and as a state dictionary.

This script implements the Two-Tower model. It features:
* Handling of diverse user and item categorical features.
* Use of explicit negative signals derived from 'DISPLAY' interactions.
* Configurable data splitting strategies:
    * **Temporal split:** For evaluating predictions of a user's next interaction.
    * **User cold-start split:** For evaluating generalization to new users.
    *(The split strategy can be changed via the `CONFIG['split_strategy']` variable within the script).*

## Running Scripts
All Python scripts in this project are designed to be run from the command line:

python <script_name>.py