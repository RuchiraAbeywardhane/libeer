"""
Quick Start Guide: MS-MDA with Emognition Dataset
===================================================

STEP 1: Install dependencies (if not done)
-------------------------------------------
python install_dependencies.py

STEP 2: Test your dataset loader
---------------------------------
python test_emognition_loader.py --dataset_path /path/to/emognition

STEP 3: Train MS-MDA model
---------------------------

BASIC USAGE:
python MsMda_train.py --dataset_path /path/to/emognition

ADVANCED USAGE (customize hyperparameters):
python MsMda_train.py \
    --dataset_path /path/to/emognition \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0005 \
    --seed 2024 \
    --output_dir ./results/msmda_custom

AVAILABLE PARAMETERS:
---------------------
--dataset_path    : Path to Emognition dataset (REQUIRED)
--epochs          : Number of training epochs (default: 50)
--batch_size      : Batch size for training (default: 64)
--lr              : Learning rate (default: 0.001)
--seed            : Random seed for reproducibility (default: 2024)
--output_dir      : Directory to save results (default: ./results/msmda_emognition)
--device          : Device to use: 'cuda' or 'cpu' (auto-detected)

WHAT THE SCRIPT DOES:
--------------------
1. Loads Emognition dataset (4 channels: TP9, AF7, AF8, TP10)
2. Extracts differential entropy features from 5 frequency bands
3. Applies LDS smoothing
4. Performs cross-subject training (leave-one-subject-out)
5. Each training subject becomes a "source domain" for MS-MDA
6. Trains the model to adapt across multiple source domains
7. Evaluates on held-out test subjects
8. Saves best models and final results

EXPECTED OUTPUT:
---------------
- Trained models saved in: {output_dir}/subject_X_fold_Y/
- Final results saved in: {output_dir}/final_results.txt
- Console output showing accuracy and F1-score for each fold

ABOUT MS-MDA:
------------
MS-MDA (Multi-Source Marginal Distribution Adaptation) is designed for 
cross-subject EEG emotion recognition. It:
- Uses multiple source domains (training subjects)
- Adapts to target domain (test subject) 
- Minimizes distribution discrepancy between domains
- Particularly effective for cross-subject scenarios

TIPS:
----
1. Start with default parameters first
2. If accuracy is low, try:
   - Increasing epochs (e.g., --epochs 100)
   - Adjusting learning rate (e.g., --lr 0.0005)
   - Using a smaller batch size (e.g., --batch_size 32)
3. Training time depends on number of subjects and epochs
4. GPU is recommended for faster training

For questions or issues, refer to the MS-MDA paper:
Chen H, Jin M, Li Z, et al. "MS-MDA: Multisource marginal distribution 
adaptation for cross-subject and cross-session EEG emotion recognition"
Frontiers in Neuroscience, 2021.
"""

# Example commands for different scenarios:

# 1. Quick test run (fewer epochs)
# python MsMda_train.py --dataset_path /path/to/emognition --epochs 10

# 2. Full training with GPU
# python MsMda_train.py --dataset_path /path/to/emognition --epochs 100 --device cuda

# 3. Custom configuration
# python MsMda_train.py --dataset_path /path/to/emognition --epochs 80 --batch_size 32 --lr 0.0003
