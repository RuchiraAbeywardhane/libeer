"""
MS-MDA Training Script for Emognition Dataset
==============================================
Multi-Source Marginal Distribution Adaptation for Cross-Subject EEG Emotion Recognition

This script trains the MS-MDA model on the Emognition dataset with MUSE headband data.

Usage:
    python MsMda_train.py --dataset_path /path/to/emognition --epochs 50 --batch_size 64

Author: Final Year Project Team
Date: February 12, 2026
Reference: Chen H, Jin M, Li Z, et al. MS-MDA: Multisource marginal distribution adaptation 
           for cross-subject and cross-session EEG emotion recognition. Frontiers in Neuroscience, 2021.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.setting import Setting
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, get_split_index, index_to_data
from models.MsMda import MSMDA
from Trainer.MsMdaTraining import train
from utils.args import get_args_parser
from utils.utils import setup_seed


def compute_class_weights(labels, num_classes=4):
    """
    Compute class weights to handle imbalanced data.
    Gives higher weight to minority classes.
    
    Args:
        labels: All training labels
        num_classes: Number of classes
    
    Returns:
        torch.Tensor: Class weights for loss function
    """
    # Flatten labels if needed
    if isinstance(labels[0], (list, np.ndarray)):
        flat_labels = []
        for label_list in labels:
            flat_labels.extend(label_list)
    else:
        flat_labels = labels
    
    # Count samples per class
    label_counts = Counter(flat_labels)
    
    print(f"\n   📊 Class Distribution:")
    for class_id in range(num_classes):
        count = label_counts.get(class_id, 0)
        percentage = (count / len(flat_labels)) * 100
        print(f"      Class {class_id}: {count:4d} samples ({percentage:5.2f}%)")
    
    # Calculate weights: inverse frequency
    total_samples = len(flat_labels)
    weights = []
    for class_id in range(num_classes):
        count = label_counts.get(class_id, 1)  # Avoid division by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    weights = torch.FloatTensor(weights)
    
    print(f"\n   ⚖️  Class Weights (higher = more important):")
    for class_id, weight in enumerate(weights):
        print(f"      Class {class_id}: {weight:.3f}")
    
    return weights


def main(args):
    """Main training function for MS-MDA on Emognition dataset."""
    
    print("\n" + "="*80)
    print("  MS-MDA TRAINING - EMOGNITION DATASET")
    print("="*80)
    print(f"📂 Dataset: {args.dataset}")
    print(f"📍 Path: {args.dataset_path}")
    print(f"🎯 Experiment: {args.experiment_mode}")
    print(f"⚙️  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print("="*80 + "\n")
    
    # Setup
    setup_seed(args.seed)
    device = torch.device(args.device)
    
    # Create setting for Emognition dataset
    setting = Setting(
        dataset='emognition',
        dataset_path=args.dataset_path,
        pass_band=[0.3, 50],  # Bandpass filter for MUSE data
        extract_bands=[[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]],  # 5 frequency bands
        time_window=1,  # 1 second windows
        overlap=0,
        sample_length=1,
        stride=1,
        seed=args.seed,
        feature_type='de_lds',  # Differential Entropy with LDS smoothing
        experiment_mode='subject-independent',  # Changed from 'cross-subject' - this is what merge_to_part expects
        split_type='train-val-test',
        test_size=0.2,
        val_size=0.2,
        label_used=['valence'],  # Can be changed based on your needs
        onehot=False,
        cross_trail='false'  # Important for MS-MDA
    )
    
    # Load and preprocess data
    print("📊 Loading Emognition dataset...")
    data, label, channels, feature_dim, num_classes = get_data(setting)
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Channels: {channels} (TP9, AF7, AF8, TP10)")
    print(f"   Feature bands: {feature_dim}")
    print(f"   Classes: {num_classes} (Emotion quadrants)")
    print(f"   Total feature dimension: {channels * feature_dim}")
    
    # CRITICAL FIX: Flatten the channel and feature dimensions for MS-MDA
    # MS-MDA expects input shape: (samples, channels * features)
    # Current shape: (session, subject, trial, sample, channel, feature)
    # Need to reshape to: (session, subject, trial, sample, channel*feature)
    
    print(f"\n🔧 Reshaping data for MS-MDA...")
    
    # Flatten channel and feature dimensions
    for ses_idx in range(len(data)):
        for sub_idx in range(len(data[ses_idx])):
            for trial_idx in range(len(data[ses_idx][sub_idx])):
                trial_data = np.array(data[ses_idx][sub_idx][trial_idx])
                # Current shape: (samples, channels, features)
                # Target shape: (samples, channels*features)
                original_shape = trial_data.shape
                if len(original_shape) == 3:
                    # Reshape from (samples, channels, features) to (samples, channels*features)
                    flattened = trial_data.reshape(original_shape[0], -1)
                    data[ses_idx][sub_idx][trial_idx] = flattened
    
    # Update feature_dim to reflect flattened features
    total_feature_dim = channels * feature_dim
    print(f"   ✅ Data reshaped: each sample now has {total_feature_dim} features")
    
    # Merge data according to experiment mode
    data, label = merge_to_part(data, label, setting)
    
    # Debug: Check what we got after merge
    print(f"\n🔍 Debug Info:")
    print(f"   Type of data: {type(data)}")
    print(f"   Length of data: {len(data)}")
    if len(data) > 0:
        print(f"   Type of data[0]: {type(data[0])}")
        print(f"   Length of data[0]: {len(data[0])}")
        if len(data[0]) > 0:
            print(f"   Type of data[0][0]: {type(data[0][0])}")
            if hasattr(data[0][0], '__len__'):
                print(f"   Length of data[0][0]: {len(data[0][0])}")
    print(f"   Type of label: {type(label)}")
    print(f"   Length of label: {len(label)}")
    
    # Storage for results
    best_metrics = []
    
    # Cross-subject training (leave-one-subject-out)
    print("\n" + "="*80)
    print("  STARTING CROSS-SUBJECT TRAINING (LEAVE-ONE-OUT)")
    print("="*80 + "\n")
    
    for subject_idx, (data_i, label_i) in enumerate(zip(data, label), 1):
        print(f"\n{'='*80}")
        print(f"  SUBJECT {subject_idx}/{len(data)} - LEAVE-ONE-OUT")
        print(f"{'='*80}")
        
        # Get train/val/test splits
        split_indices = get_split_index(data_i, label_i, setting)
        
        for fold_idx, (train_indexes, test_indexes, val_indexes) in enumerate(
            zip(split_indices['train'], split_indices['test'], split_indices['val']), 1
        ):
            setup_seed(args.seed)
            
            print(f"\n📋 Fold {fold_idx}")
            print(f"   Train subjects: {len(train_indexes)}")
            print(f"   Val subjects: {len(val_indexes)}")
            print(f"   Test subjects: {len(test_indexes)}")
            
            # Prepare data
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes)
            
            # Compute class weights from training data to handle imbalance
            print(f"\n   🔧 Computing class weights for imbalanced data...")
            class_weights = compute_class_weights(train_label, num_classes=num_classes)
            class_weights = class_weights.to(device)
            
            # For MS-MDA: each training subject is a "source domain"
            datasets_train = []
            samples_per_source = []
            
            for source_idx in train_indexes:
                source_data = data_i[source_idx]
                source_label = label_i[source_idx]
                
                dataset_source = TensorDataset(
                    torch.Tensor(source_data),
                    torch.Tensor(source_label)
                )
                datasets_train.append(dataset_source)
                samples_per_source.append(len(source_data))
            
            # Prepare validation and test datasets
            dataset_val = TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            
            # Calculate total samples from all sources
            total_source_samples = sum(samples_per_source)
            number_of_sources = len(datasets_train)
            
            print(f"   Number of source domains: {number_of_sources}")
            print(f"   Total source samples: {total_source_samples}")
            
            # Initialize MS-MDA model
            model = MSMDA(
                num_electrodes=channels,
                in_channels=feature_dim,
                num_classes=num_classes,
                number_of_source=number_of_sources,
                pretrained=False
            )
            
            # Setup optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-4
            )
            
            # Setup loss with class weights
            criterion = nn.NLLLoss(weight=class_weights)
            print(f"   ✅ Using weighted loss to handle class imbalance")
            
            # Output directory
            output_dir = f"{args.output_dir}/subject_{subject_idx}_fold_{fold_idx}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Train model
            print(f"\n🚀 Training MS-MDA model...")
            metric_value = train(
                model=model,
                datasets_train=datasets_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                samples_source=total_source_samples,
                device=device,
                output_dir=output_dir,
                metrics=['acc', 'macro-f1'],  # Changed 'f1' to 'macro-f1'
                metric_choose='acc',
                optimizer=optimizer,
                scheduler=None,
                batch_size=args.batch_size,
                epochs=args.epochs,
                criterion=criterion
            )
            
            best_metrics.append(metric_value)
            
            print(f"\n✅ Subject {subject_idx} Fold {fold_idx} completed!")
            print(f"   Test Accuracy: {metric_value['acc']:.2f}%")
            print(f"   Test F1-Score: {metric_value['macro-f1']:.2f}%")  # Changed 'f1' to 'macro-f1'
    
    # Calculate average metrics
    print("\n" + "="*80)
    print("  FINAL RESULTS - CROSS-SUBJECT EVALUATION")
    print("="*80)
    
    # Check if we have any results
    if len(best_metrics) == 0:
        print("\n❌ ERROR: No training results collected!")
        print("   This could mean:")
        print("   - Not enough subjects in the dataset")
        print("   - Data loading failed")
        print("   - Split configuration issue")
        print("\n💡 Try running the test script first:")
        print(f"   python test_emognition_loader.py -dataset_path {args.dataset_path}")
        return
    
    avg_metrics = {}
    for metric_name in best_metrics[0].keys():
        values = [m[metric_name] for m in best_metrics]
        avg_metrics[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        
        print(f"\n📊 {metric_name.upper()}:")
        print(f"   Mean: {avg_metrics[metric_name]['mean']:.2f}%")
        print(f"   Std:  {avg_metrics[metric_name]['std']:.2f}%")
    
    # Save final results
    results_file = f"{args.output_dir}/final_results.txt"
    with open(results_file, 'w') as f:
        f.write("MS-MDA Training Results - Emognition Dataset\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Experiment: {args.experiment_mode}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n\n")
        
        for metric_name, values in avg_metrics.items():
            f.write(f"{metric_name.upper()}:\n")
            f.write(f"  Mean: {values['mean']:.2f}%\n")
            f.write(f"  Std:  {values['std']:.2f}%\n\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n" + "="*80)
    print("  ✅ TRAINING COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Get arguments
    parser = get_args_parser()
    
    args = parser.parse_args()
    
    # Set default values for Emognition
    args.dataset = 'emognition'
    # Note: We use 'subject-independent' internally, but it's cross-subject training (MS-MDA)
    args.experiment_mode = 'subject-independent'
    
    # Set defaults if not provided
    if args.epochs is None or args.epochs == 40:  # default from parser
        args.epochs = 50
    if args.batch_size is None or args.batch_size == 128:  # default from parser
        args.batch_size = 64
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    main(args)
