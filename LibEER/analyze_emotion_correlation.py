"""
Emotion Correlation Analysis for Emognition Dataset
====================================================

This script analyzes correlations between the 10 emotions in the Emognition dataset
to identify which emotions are most similar/correlated based on EEG features.

The 10 emotions in Emognition are:
1. AMUSEMENT (Positive, Active)
2. ENTHUSIASM (Positive, Active)
3. AWE (Positive, Active)
4. ANGER (Negative, Active)
5. FEAR (Negative, Active)
6. DISGUST (Negative, Active)
7. SADNESS (Negative, Calm)
8. SURPRISE (Negative, Calm)
9. LIKING (Positive, Calm)
10. NEUTRAL

Usage:
    python analyze_emotion_correlation.py --dataset_path /path/to/emognition

This will:
- Load all emotion data
- Extract EEG features for each emotion
- Compute correlation matrices (Pearson, Spearman)
- Visualize correlations with heatmaps
- Identify most/least correlated emotion pairs
- Suggest emotion groupings

Author: Final Year Project Team
Date: February 12, 2026
"""

import os
import sys
import numpy as np
import json
import glob
from collections import defaultdict
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_emotion_correlations(dataset_path):
    """
    Analyze correlations between emotions in the Emognition dataset.
    
    Args:
        dataset_path: Path to Emognition dataset
    
    Returns:
        Dictionary containing correlation analysis results
    """
    
    print("\n" + "="*80)
    print("  EMOTION CORRELATION ANALYSIS - EMOGNITION DATASET")
    print("="*80)
    print(f"📂 Dataset path: {dataset_path}\n")
    
    # Emotion mapping (from your load_data.py)
    EMOTIONS = {
        "AMUSEMENT": 0,
        "ENTHUSIASM": 1,
        "AWE": 2,
        "ANGER": 3,
        "FEAR": 4,
        "DISGUST": 5,
        "SADNESS": 6,
        "SURPRISE": 7,
        "LIKING": 8,
        "NEUTRAL": 9
    }
    
    EMOTION_NAMES = list(EMOTIONS.keys())
    
    print("📊 Analyzing 10 emotions:")
    for i, emotion in enumerate(EMOTION_NAMES, 1):
        print(f"   {i:2d}. {emotion}")
    
    # Find all MUSE JSON files
    search_patterns = [
        os.path.join(dataset_path, "*_STIMULUS_MUSE.json"),
        os.path.join(dataset_path, "*", "*_STIMULUS_MUSE.json"),
        os.path.join(dataset_path, "*", "*", "*_STIMULUS_MUSE.json")
    ]
    
    all_files = set()
    for pattern in search_patterns:
        all_files.update(glob.glob(pattern))
    
    all_files = sorted(all_files)
    
    print(f"\n📁 Found {len(all_files)} data files\n")
    
    # Helper functions
    def _to_numeric(x):
        """Convert input to numeric numpy array."""
        if isinstance(x, list):
            if not x:
                return np.array([], dtype=np.float64)
            return np.asarray(x, dtype=np.float64)
        return np.asarray([x], dtype=np.float64)
    
    def _interpolate_nans(signal):
        """Interpolate NaN values in signal."""
        signal = signal.astype(np.float64, copy=True)
        valid_mask = np.isfinite(signal)
        
        if valid_mask.all():
            return signal
        if not valid_mask.any():
            return np.zeros_like(signal)
        
        indices = np.arange(len(signal))
        signal[~valid_mask] = np.interp(indices[~valid_mask], indices[valid_mask], signal[valid_mask])
        return signal
    
    # Collect data per emotion
    emotion_data = {emotion: [] for emotion in EMOTION_NAMES}
    emotion_counts = {emotion: 0 for emotion in EMOTION_NAMES}
    
    print("🔄 Loading and processing EEG data...")
    
    skipped = 0
    for file_path in all_files:
        filename = os.path.basename(file_path)
        parts = filename.replace(".json", "").split("_")
        
        if len(parts) < 2:
            skipped += 1
            continue
        
        emotion = parts[1].upper()
        
        if emotion not in EMOTIONS:
            skipped += 1
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Extract raw EEG channels
            channels_data = {
                'TP9': _interpolate_nans(_to_numeric(raw_data.get('RAW_TP9', []))),
                'AF7': _interpolate_nans(_to_numeric(raw_data.get('RAW_AF7', []))),
                'AF8': _interpolate_nans(_to_numeric(raw_data.get('RAW_AF8', []))),
                'TP10': _interpolate_nans(_to_numeric(raw_data.get('RAW_TP10', [])))
            }
            
            if all(len(ch) == 0 for ch in channels_data.values()):
                skipped += 1
                continue
            
            min_length = min(len(ch) for ch in channels_data.values())
            
            # Stack channels into matrix (channels, samples)
            signal = np.stack([
                channels_data['TP9'][:min_length],
                channels_data['AF7'][:min_length],
                channels_data['AF8'][:min_length],
                channels_data['TP10'][:min_length]
            ], axis=0)
            
            # Remove DC offset
            signal = signal - np.mean(signal, axis=1, keepdims=True)
            
            # Compute statistical features for correlation analysis
            features = []
            
            # Mean, std, variance per channel
            features.extend(np.mean(signal, axis=1))
            features.extend(np.std(signal, axis=1))
            features.extend(np.var(signal, axis=1))
            
            # Min, max per channel
            features.extend(np.min(signal, axis=1))
            features.extend(np.max(signal, axis=1))
            
            # Power in different frequency bands (simple approximation)
            for ch in range(signal.shape[0]):
                ch_signal = signal[ch]
                fft = np.fft.fft(ch_signal)
                power_spectrum = np.abs(fft) ** 2
                features.append(np.mean(power_spectrum))
            
            emotion_data[emotion].append(features)
            emotion_counts[emotion] += 1
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"✅ Processing complete! (Skipped: {skipped} files)\n")
    
    # Display emotion counts
    print("📊 Samples per emotion:")
    for emotion in EMOTION_NAMES:
        count = emotion_counts[emotion]
        print(f"   {emotion:15s}: {count:3d} samples")
    
    # Compute average features per emotion
    print("\n🔧 Computing average features per emotion...")
    
    emotion_features = {}
    for emotion in EMOTION_NAMES:
        if len(emotion_data[emotion]) > 0:
            emotion_features[emotion] = np.mean(emotion_data[emotion], axis=0)
        else:
            # Skip emotions with no data
            print(f"   ⚠️  Warning: {emotion} has no samples, skipping...")
            continue
    
    # Create feature matrix (only for emotions with data)
    available_emotions = [e for e in EMOTION_NAMES if e in emotion_features]
    feature_matrix = np.array([emotion_features[emotion] for emotion in available_emotions])
    
    print(f"   ✅ Created feature matrix for {len(available_emotions)} emotions")
    
    # Compute correlation matrix (Pearson)
    print("\n📈 Computing Pearson correlation matrix...")
    
    correlation_matrix = np.corrcoef(feature_matrix)
    
    # Display correlation matrix
    print("\n" + "="*80)
    print("  CORRELATION MATRIX (Pearson)")
    print("="*80)
    print("\n           ", end="")
    for i, emotion in enumerate(available_emotions):
        print(f"{emotion[:6]:>7s}", end="")
    print()
    print("-" * 80)
    
    for i, emotion in enumerate(available_emotions):
        print(f"{emotion:11s}", end="")
        for j in range(len(available_emotions)):
            corr = correlation_matrix[i, j]
            if i == j:
                print(f"  1.000", end="")
            else:
                print(f"  {corr:5.2f}", end="")
        print()
    
    # Find most correlated pairs
    print("\n" + "="*80)
    print("  TOP 10 MOST CORRELATED EMOTION PAIRS")
    print("="*80)
    
    correlations = []
    for i in range(len(available_emotions)):
        for j in range(i+1, len(available_emotions)):
            correlations.append({
                'emotion1': available_emotions[i],
                'emotion2': available_emotions[j],
                'correlation': correlation_matrix[i, j]
            })
    
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print("\nMost Similar (Highly Correlated):")
    print("-" * 80)
    for i, pair in enumerate(correlations[:10], 1):
        print(f"{i:2d}. {pair['emotion1']:15s} ↔ {pair['emotion2']:15s}  |  r = {pair['correlation']:6.3f}")
    
    print("\n" + "="*80)
    print("  TOP 10 LEAST CORRELATED EMOTION PAIRS")
    print("="*80)
    
    print("\nMost Different (Least Correlated):")
    print("-" * 80)
    for i, pair in enumerate(reversed(correlations[-10:]), 1):
        print(f"{i:2d}. {pair['emotion1']:15s} ↔ {pair['emotion2']:15s}  |  r = {pair['correlation']:6.3f}")
    
    # Identify natural groupings
    print("\n" + "="*80)
    print("  SUGGESTED EMOTION GROUPINGS")
    print("="*80)
    
    print("\n📌 Based on correlation analysis:\n")
    
    # Group highly correlated emotions (threshold: r > 0.7)
    threshold = 0.7
    groups = []
    used = set()
    
    for pair in correlations:
        if pair['correlation'] > threshold:
            e1, e2 = pair['emotion1'], pair['emotion2']
            
            # Check if either emotion is already in a group
            found_group = None
            for group in groups:
                if e1 in group or e2 in group:
                    found_group = group
                    break
            
            if found_group:
                found_group.add(e1)
                found_group.add(e2)
            else:
                groups.append({e1, e2})
            
            used.add(e1)
            used.add(e2)
    
    # Add ungrouped emotions as singletons
    for emotion in EMOTION_NAMES:
        if emotion not in used:
            groups.append({emotion})
    
    for i, group in enumerate(groups, 1):
        print(f"Group {i}: {', '.join(sorted(group))}")
    
    # Valence-Arousal mapping suggestion
    print("\n" + "="*80)
    print("  VALENCE-AROUSAL QUADRANT MAPPING")
    print("="*80)
    
    print("\nCurrent mapping (from load_data.py):")
    print("   Q0 (Positive, Active): AMUSEMENT, ENTHUSIASM, AWE")
    print("   Q1 (Negative, Active): ANGER, FEAR, DISGUST")
    print("   Q2 (Negative, Calm):   SADNESS, SURPRISE")
    print("   Q3 (Positive, Calm):   LIKING")
    print("   Note: NEUTRAL is not assigned to any quadrant")
    
    # Compute average correlations within quadrants
    quadrants = {
        'Q0': ['AMUSEMENT', 'ENTHUSIASM', 'AWE'],
        'Q1': ['ANGER', 'FEAR', 'DISGUST'],
        'Q2': ['SADNESS', 'SURPRISE'],
        'Q3': ['LIKING']  # Only LIKING, TENDERNESS doesn't exist
    }
    
    print("\n📊 Within-Quadrant Correlation (higher = better grouping):")
    
    for quad_name, emotions in quadrants.items():
        if len(emotions) > 1:
            # Check if all emotions in this quadrant are available
            available_quad_emotions = [e for e in emotions if e in available_emotions]
            if len(available_quad_emotions) > 1:
                indices = [available_emotions.index(e) for e in available_quad_emotions]
                within_corrs = []
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        within_corrs.append(correlation_matrix[indices[i], indices[j]])
                avg_corr = np.mean(within_corrs)
                print(f"   {quad_name}: {avg_corr:.3f}  ({', '.join(available_quad_emotions)})")
        else:
            print(f"   {quad_name}: N/A (only one emotion: {emotions[0]})")
    
    # Save results to file
    results_file = "emotion_correlation_analysis.txt"
    
    print(f"\n💾 Saving detailed results to: {results_file}")
    
    with open(results_file, 'w') as f:
        f.write("Emotion Correlation Analysis - Emognition Dataset\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Date: February 12, 2026\n\n")
        
        f.write("Emotion Counts:\n")
        for emotion in EMOTION_NAMES:
            f.write(f"  {emotion}: {emotion_counts[emotion]} samples\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Correlation Matrix:\n")
        f.write("="*80 + "\n\n")
        
        f.write("           ")
        for emotion in available_emotions:
            f.write(f"{emotion[:6]:>7s}")
        f.write("\n")
        
        for i, emotion in enumerate(available_emotions):
            f.write(f"{emotion:11s}")
            for j in range(len(available_emotions)):
                f.write(f"  {correlation_matrix[i, j]:5.2f}")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Top Correlated Pairs:\n")
        f.write("="*80 + "\n\n")
        
        for i, pair in enumerate(correlations[:20], 1):
            f.write(f"{i:2d}. {pair['emotion1']} ↔ {pair['emotion2']}: r = {pair['correlation']:.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Suggested Groupings:\n")
        f.write("="*80 + "\n\n")
        
        for i, group in enumerate(groups, 1):
            f.write(f"Group {i}: {', '.join(sorted(group))}\n")
    
    print("\n" + "="*80)
    print("  ✅ ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    
    return {
        'correlation_matrix': correlation_matrix,
        'emotion_names': available_emotions,
        'top_pairs': correlations[:20],
        'groups': groups
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze emotion correlations in Emognition dataset")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to Emognition dataset')
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_emotion_correlations(args.dataset_path)
    
    print("\n💡 RECOMMENDATIONS:")
    print("="*80)
    print("""
Based on the correlation analysis, you can:

1. **Merge highly correlated emotions** (r > 0.7)
   - Reduces 11 emotions → fewer classes
   - Improves classification accuracy
   - Example: If AMUSEMENT & ENTHUSIASM are r=0.85, combine them

2. **Keep current 4-quadrant mapping**
   - If within-quadrant correlations are high
   - Already reduces 11 → 4 classes
   - Balanced approach

3. **Binary classification**
   - Positive vs Negative (valence)
   - Active vs Calm (arousal)
   - Easiest to classify

4. **Custom grouping**
   - Based on top correlated pairs above
   - Data-driven grouping
   - May differ from valence-arousal theory

Run this analysis and examine the correlation matrix to make an informed decision!
    """)
