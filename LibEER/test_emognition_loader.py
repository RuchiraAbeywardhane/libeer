"""
KAGGLE NOTEBOOK SETUP FOR EMOGNITION DATASET LOADER TEST
=========================================================

Run these commands in a Kaggle notebook cell:

STEP 1: Install dependencies (if needed)
-----------------------------------------
!pip install mne scipy tqdm pandas numpy xmltodict mat73

STEP 2: Navigate to LibEER directory
-------------------------------------
%cd /kaggle/working/LibEER/LibEER

STEP 3: Run the test script
----------------------------
!python test_emognition_loader.py --dataset_path /kaggle/input/emognition

ALTERNATIVE: If dataset is in a different location
---------------------------------------------------
!python test_emognition_loader.py --dataset_path /kaggle/input/your-dataset-name


FULL KAGGLE NOTEBOOK SETUP
===========================
Copy and paste this into a Kaggle notebook cell:
"""

# Cell 1: Install dependencies
print("Installing dependencies...")
!pip install -q mne scipy tqdm pandas numpy xmltodict mat73

# Cell 2: Navigate to LibEER directory
import os
os.chdir('/kaggle/working/LibEER/LibEER')
print(f"Current directory: {os.getcwd()}")

# Cell 3: Run the test
!python test_emognition_loader.py --dataset_path /kaggle/input/emognition

"""
QUICK SETUP (Single Cell)
===========================
If you want to run everything in one cell:
"""

# Install, navigate, and run test in one go
!pip install -q mne scipy tqdm pandas numpy xmltodict mat73 && \
cd /kaggle/working/LibEER/LibEER && \
python test_emognition_loader.py --dataset_path /kaggle/input/emognition

"""
NOTES:
------
1. Replace 'emognition' with your actual Kaggle dataset name
2. If your dataset is uploaded as 'emognition-eeg-dataset', use:
   --dataset_path /kaggle/input/emognition-eeg-dataset

3. To check available datasets, run:
   !ls /kaggle/input/

4. Make sure your LibEER code is uploaded to Kaggle or cloned from GitHub
"""

"""
Test script for Emognition dataset loader
==========================================
This script tests the Emognition data loader to ensure it's working correctly.

Run this script from the LibEER directory:
    python test_emognition_loader.py --dataset_path /path/to/your/emognition/data

Author: Final Year Project Team
Date: February 2026
"""

import os
import sys
import argparse
import numpy as np
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils.load_data import read_emognition, get_data
from config.setting import Setting


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_basic_loading(dataset_path):
    """Test basic data loading functionality."""
    print_section("TEST 1: Basic Data Loading")
    
    try:
        print(f"📂 Loading data from: {dataset_path}")
        data, baseline, labels, sample_rate, num_channels = read_emognition(dataset_path)
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Number of channels: {num_channels}")
        print(f"   Number of sessions: {len(data)}")
        print(f"   Number of subjects: {len(data[0])}")
        
        # Check data structure
        total_trials = 0
        for subject_idx, subject_trials in enumerate(data[0]):
            total_trials += len(subject_trials)
        
        print(f"   Total trials: {total_trials}")
        
        # Sample a trial to check shape
        if len(data[0]) > 0 and len(data[0][0]) > 0:
            sample_trial = data[0][0][0]
            print(f"\n📊 Sample trial shape: {sample_trial.shape}")
            print(f"   Expected format: (channels={num_channels}, time_samples)")
            
            if sample_trial.shape[0] == num_channels:
                print(f"   ✅ Channel dimension correct!")
            else:
                print(f"   ⚠️  WARNING: Expected {num_channels} channels, got {sample_trial.shape[0]}")
        
        return True, data, labels, sample_rate, num_channels
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None


def test_label_distribution(labels):
    """Test label distribution and validity."""
    print_section("TEST 2: Label Distribution")
    
    try:
        # Flatten all labels
        all_labels = []
        for session_labels in labels:
            for subject_labels in session_labels:
                all_labels.extend(subject_labels)
        
        label_counts = Counter(all_labels)
        
        print(f"\n📈 Label Distribution:")
        quadrant_names = {
            0: "Q1 (Positive, Active)",
            1: "Q2 (Negative, Active)",
            2: "Q3 (Negative, Calm)",
            3: "Q4 (Positive, Calm)"
        }
        
        for label_id in sorted(label_counts.keys()):
            count = label_counts[label_id]
            percentage = (count / len(all_labels)) * 100
            q_name = quadrant_names.get(label_id, f"Unknown ({label_id})")
            print(f"   {q_name}: {count:4d} trials ({percentage:5.2f}%)")
        
        # Check for invalid labels
        valid_labels = {0, 1, 2, 3}
        invalid_labels = set(all_labels) - valid_labels
        
        if invalid_labels:
            print(f"\n⚠️  WARNING: Found invalid labels: {invalid_labels}")
            return False
        else:
            print(f"\n✅ All labels are valid (0-3)")
            return True
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_quality(data, sample_rate):
    """Test data quality (check for NaNs, infinities, etc.)."""
    print_section("TEST 3: Data Quality")
    
    try:
        total_samples = 0
        nan_count = 0
        inf_count = 0
        zero_count = 0
        
        print("\n🔍 Checking for data issues...")
        
        for session_data in data:
            for subject_trials in session_data:
                for trial in subject_trials:
                    total_samples += trial.size
                    nan_count += np.isnan(trial).sum()
                    inf_count += np.isinf(trial).sum()
                    zero_count += (trial == 0).sum()
        
        print(f"\n📊 Quality Metrics:")
        print(f"   Total data points: {total_samples:,}")
        print(f"   NaN values: {nan_count:,} ({(nan_count/total_samples*100):.4f}%)")
        print(f"   Inf values: {inf_count:,} ({(inf_count/total_samples*100):.4f}%)")
        print(f"   Zero values: {zero_count:,} ({(zero_count/total_samples*100):.4f}%)")
        
        if nan_count > 0:
            print(f"\n⚠️  WARNING: Found {nan_count} NaN values!")
            return False
        
        if inf_count > 0:
            print(f"\n⚠️  WARNING: Found {inf_count} infinite values!")
            return False
        
        print(f"\n✅ Data quality looks good!")
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing_pipeline(dataset_path):
    """Test the full preprocessing pipeline with LibEER."""
    print_section("TEST 4: Full Preprocessing Pipeline")
    
    try:
        print("\n🔧 Creating LibEER Setting...")
        
        setting = Setting(
            dataset='emognition',
            dataset_path=dataset_path,
            pass_band=[0.3, 50],
            extract_bands=[[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]],
            time_window=1,
            overlap=0,
            sample_length=1,
            stride=1,
            seed=2024,
            feature_type='de_lds',
            experiment_mode="subject-dependent",
            split_type='train-val-test',
            test_size=0.2,
            val_size=0.2
        )
        
        print("✅ Setting created successfully!")
        
        print("\n🔄 Running full preprocessing pipeline...")
        print("   This may take a few minutes...")
        
        data, label, channels, feature_dim, num_classes = get_data(setting)
        
        print(f"\n✅ Preprocessing complete!")
        print(f"\n📊 Output dimensions:")
        print(f"   Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"   Label shape: {label.shape if hasattr(label, 'shape') else 'N/A'}")
        print(f"   Channels: {channels}")
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Number of classes: {num_classes}")
        
        # Check expected values
        if channels == 4 and num_classes == 4:
            print(f"\n✅ Expected channels (4) and classes (4) confirmed!")
        else:
            print(f"\n⚠️  WARNING: Unexpected values!")
            print(f"   Expected: channels=4, num_classes=4")
            print(f"   Got: channels={channels}, num_classes={num_classes}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_subject_information(data, labels):
    """Display subject-level information."""
    print_section("TEST 5: Subject-Level Information")
    
    try:
        print("\n👥 Subject Information:")
        print(f"{'Subject':<10} {'Trials':<10} {'Primary Label':<20} {'Label Distribution'}")
        print("-" * 80)
        
        for subject_idx, (subject_trials, subject_labels) in enumerate(zip(data[0], labels[0])):
            num_trials = len(subject_trials)
            label_counts = Counter(subject_labels)
            primary_label = max(label_counts, key=label_counts.get)
            
            quadrant_names = {
                0: "Q1 (Pos/Active)",
                1: "Q2 (Neg/Active)",
                2: "Q3 (Neg/Calm)",
                3: "Q4 (Pos/Calm)"
            }
            
            primary_name = quadrant_names.get(primary_label, f"Q{primary_label}")
            label_dist = ", ".join([f"{k}:{v}" for k, v in sorted(label_counts.items())])
            
            print(f"{subject_idx+1:<10} {num_trials:<10} {primary_name:<20} {label_dist}")
        
        print("\n✅ Subject information displayed successfully!")
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(dataset_path):
    """Run all tests."""
    print("\n" + "="*80)
    print("  EMOGNITION DATASET LOADER - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"\n📍 Dataset path: {dataset_path}")
    print(f"📅 Test date: February 12, 2026")
    
    results = {}
    
    # Test 1: Basic loading
    success, data, labels, sample_rate, num_channels = test_basic_loading(dataset_path)
    results['Basic Loading'] = success
    
    if not success:
        print("\n❌ Basic loading failed. Cannot proceed with other tests.")
        return results
    
    # Test 2: Label distribution
    results['Label Distribution'] = test_label_distribution(labels)
    
    # Test 3: Data quality
    results['Data Quality'] = test_data_quality(data, sample_rate)
    
    # Test 4: Preprocessing pipeline
    results['Preprocessing Pipeline'] = test_preprocessing_pipeline(dataset_path)
    
    # Test 5: Subject information
    results['Subject Information'] = test_subject_information(data, labels)
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<30} {status}")
    
    print(f"\n{'='*80}")
    if passed == total:
        print(f"  🎉 ALL TESTS PASSED! ({passed}/{total})")
        print(f"  Your Emognition dataset loader is working correctly! 🚀")
    else:
        print(f"  ⚠️  SOME TESTS FAILED: {passed}/{total} passed")
        print(f"  Please check the errors above and fix the issues.")
    print(f"{'='*80}\n")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test Emognition dataset loader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_emognition_loader.py --dataset_path /data/emognition
  python test_emognition_loader.py --dataset_path "E:/Datasets/Emognition"
        """
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the Emognition dataset directory'
    )
    
    args = parser.parse_args()
    
    # Check if path exists
    if not os.path.exists(args.dataset_path):
        print(f"\n❌ ERROR: Dataset path does not exist: {args.dataset_path}")
        print(f"Please provide a valid path to your Emognition dataset.")
        sys.exit(1)
    
    # Run tests
    results = run_all_tests(args.dataset_path)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
