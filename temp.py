#!/usr/bin/env python3
"""
Brain Connectivity Classification - Minimal Implementation
===========================================================
Direct approach: Raw data → Zero diagonal → Logistic Regression → 3-fold CV
NO scaling, NO Fisher Z, NO complexity - just the essentials

This script tests whether diagonal=0 signal is preserved through preprocessing.

Expected results:
- WITH diagonal (232 features): ~98-99% validation accuracy
- WITHOUT diagonal (231 features): ~70-90% validation accuracy

Usage:
    python minimal_test.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("="*70)
print("BRAIN CONNECTIVITY CLASSIFICATION - MINIMAL TEST")
print("="*70)
print()

# ==============================================================================
# STEP 1: LOAD RAW DATA
# ==============================================================================
print("STEP 1: Loading raw data...")

data_path = "data/raw/PIOP2_restingstate.csv"  # Change this to your path
df = pd.read_csv(data_path)

print(f"✓ Loaded data shape: {df.shape}")
print(f"✓ Number of subjects: {len(df)}")

# Extract subject IDs and connection data
subject_ids = df.iloc[:, 0].values
connection_columns = [col for col in df.columns if '~' in str(col)]
print(f"✓ Number of connections: {len(connection_columns)}")
print(f"  Example connection: {connection_columns[0]}")
print()


# ==============================================================================
# STEP 2: EXTRACT REGION NAMES FROM CONNECTIONS
# ==============================================================================
print("STEP 2: Extracting regions...")

def extract_regions(connection_columns):
    """Extract unique region names from connection column names."""
    regions = []
    seen = set()
    
    for col in connection_columns:
        region_a, region_b = col.split('~')
        
        if region_a not in seen:
            regions.append(region_a)
            seen.add(region_a)
        
        if region_b not in seen:
            regions.append(region_b)
            seen.add(region_b)
    
    region_to_idx = {region: idx for idx, region in enumerate(regions)}
    
    return regions, region_to_idx

region_list, region_to_idx = extract_regions(connection_columns)
n_regions = len(region_list)

print(f"✓ Number of regions: {n_regions}")
print(f"  First 3 regions: {region_list[:3]}")
print()


# ==============================================================================
# STEP 3: RECONSTRUCT CONNECTIVITY MATRICES
# ==============================================================================
print("STEP 3: Reconstructing connectivity matrices...")

def reconstruct_connectivity_matrix(subject_row, connection_columns, region_to_idx, n_regions):
    """Reconstruct full symmetric connectivity matrix from flattened data."""
    
    # Initialize matrix
    matrix = np.zeros((n_regions, n_regions), dtype=float)
    
    # Fill in connections
    for col in connection_columns:
        region_a, region_b = col.split('~')
        idx_a = region_to_idx[region_a]
        idx_b = region_to_idx[region_b]
        
        value = subject_row[col]
        
        # Symmetric matrix
        matrix[idx_a, idx_b] = value
        matrix[idx_b, idx_a] = value
    
    # Diagonal is initially 1.0 (self-correlation)
    np.fill_diagonal(matrix, 1.0)
    
    return matrix

# Test on first subject
test_matrix = reconstruct_connectivity_matrix(
    df.iloc[0], 
    connection_columns, 
    region_to_idx, 
    n_regions
)

print(f"✓ Reconstructed matrix shape: {test_matrix.shape}")
print(f"  Min: {test_matrix.min():.4f}, Max: {test_matrix.max():.4f}, Mean: {test_matrix.mean():.4f}")
print(f"  Diagonal (first 5): {test_matrix.diagonal()[:5]}")
print(f"  Is symmetric: {np.allclose(test_matrix, test_matrix.T)}")
print()


# ==============================================================================
# STEP 4: SET DIAGONAL TO ZERO
# ==============================================================================
print("STEP 4: Setting diagonal to zero...")

def impute_diagonal_zero(matrix):
    """Set all diagonal elements to 0.0."""
    matrix_copy = matrix.copy()
    np.fill_diagonal(matrix_copy, 0.0)
    return matrix_copy

test_matrix_zero = impute_diagonal_zero(test_matrix)

print(f"✓ After diagonal imputation:")
print(f"  Diagonal (first 10): {test_matrix_zero.diagonal()[:10]}")
print(f"  All diagonal zeros: {np.all(test_matrix_zero.diagonal() == 0.0)}")
print()


# ==============================================================================
# STEP 5: CREATE CLASSIFICATION DATASET
# ==============================================================================
print("STEP 5: Creating classification dataset...")

def create_classification_dataset(df, connection_columns, region_to_idx, n_regions, include_diagonal=True):
    """
    Create classification dataset:
    - X: Connectivity features (one row per region per subject)
    - y: Region labels (which region each row represents)
    - subjects: Subject ID for each row (for GroupKFold)
    """
    
    n_subjects = len(df)
    n_samples = n_subjects * n_regions
    n_features = n_regions if include_diagonal else (n_regions - 1)
    
    X = np.zeros((n_samples, n_features), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    subjects = np.zeros(n_samples, dtype=int)
    
    sample_idx = 0
    
    print(f"  Processing {n_subjects} subjects...")
    
    for subj_idx in range(n_subjects):
        if (subj_idx + 1) % 50 == 0:
            print(f"    Processed {subj_idx + 1}/{n_subjects} subjects...")
        
        # Reconstruct matrix
        matrix = reconstruct_connectivity_matrix(
            df.iloc[subj_idx],
            connection_columns,
            region_to_idx,
            n_regions
        )
        
        # Impute diagonal to zero
        matrix = impute_diagonal_zero(matrix)
        
        # Extract one sample per region
        for region_idx in range(n_regions):
            row = matrix[region_idx, :]  # Connectivity of this region to all regions
            
            if include_diagonal:
                features = row  # Keep all 232 features (including diagonal=0)
            else:
                features = np.delete(row, region_idx)  # Remove diagonal
            
            X[sample_idx] = features
            y[sample_idx] = region_idx
            subjects[sample_idx] = subj_idx
            
            sample_idx += 1
    
    return X, y, subjects

# Create dataset WITH diagonal (should get ~98-99% accuracy)
print("\nCreating dataset WITH diagonal included...")
X_with_diag, y, subjects = create_classification_dataset(
    df, 
    connection_columns, 
    region_to_idx, 
    n_regions,
    include_diagonal=True
)

print(f"\n✓ Dataset WITH diagonal:")
print(f"  X shape: {X_with_diag.shape}")
print(f"  y shape: {y.shape}")
print(f"  Number of features: {X_with_diag.shape[1]}")
print(f"  Number of samples: {X_with_diag.shape[0]}")
print(f"  Samples per subject: {X_with_diag.shape[0] / len(df):.0f}")

# Check zeros
n_zeros = (X_with_diag == 0.0).sum()
total_values = X_with_diag.size
print(f"\n✓ Zero statistics:")
print(f"  Total zeros: {n_zeros}")
print(f"  Percentage: {100*n_zeros/total_values:.2f}%")
print(f"  Expected (1 per sample): {X_with_diag.shape[0]}")
print(f"  Match: {n_zeros == X_with_diag.shape[0]}")

# Check first few samples
print(f"\n✓ First 3 samples (first 10 features):")
for i in range(3):
    zeros_in_sample = (X_with_diag[i] == 0.0).sum()
    print(f"  Sample {i}: [{X_with_diag[i, 0]:.4f}, {X_with_diag[i, 1]:.4f}, {X_with_diag[i, 2]:.4f}, ...] (zeros: {zeros_in_sample})")
print()


# ==============================================================================
# STEP 6: 3-FOLD CROSS-VALIDATION (NO SCALING)
# ==============================================================================
print("STEP 6: Running 3-fold cross-validation (NO SCALING)...")

def cross_validate_simple(X, y, subjects, n_splits=3, C=1e18):
    """
    Simple 3-fold cross-validation with NO preprocessing.
    Just raw features → Logistic Regression.
    """
    
    # Get unique subjects
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    print(f"\n{'='*60}")
    print(f"3-FOLD CROSS-VALIDATION (NO SCALING)")
    print(f"{'='*60}")
    print(f"Total subjects: {n_subjects}")
    print(f"Total samples: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print(f"C parameter: {C:.2e} (essentially no regularization)")
    print()
    
    # GroupKFold splits by subjects
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    
    for fold, (train_subj_idx, val_subj_idx) in enumerate(
        gkf.split(unique_subjects, groups=unique_subjects), 1
    ):
        print(f"Fold {fold}/{n_splits}:")
        
        # Get subject IDs for this fold
        train_subjects = unique_subjects[train_subj_idx]
        val_subjects = unique_subjects[val_subj_idx]
        
        # Create masks for samples
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        
        print(f"  Train: {len(train_subjects)} subjects, {len(y_train)} samples")
        print(f"  Val:   {len(val_subjects)} subjects, {len(y_val)} samples")
        
        # Check zeros in training data
        train_zeros = (X_train == 0.0).sum()
        train_zero_pct = 100 * train_zeros / X_train.size
        print(f"  Train zeros: {train_zeros} ({train_zero_pct:.2f}%)")
        
        # Create model - NO SCALING, just raw features
        model = LogisticRegression(
            C=C,
            penalty='l2',
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=1000,
            random_state=42,
            verbose=0
        )
        
        # Train
        print(f"  Training...", end=" ", flush=True)
        model.fit(X_train, y_train)
        print("Done!")
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Evaluate
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        print(f"  Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Val accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
        print()
        
        fold_results.append({
            'fold': fold,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'n_train_subjects': len(train_subjects),
            'n_val_subjects': len(val_subjects)
        })
    
    # Summary
    train_accs = [r['train_acc'] for r in fold_results]
    val_accs = [r['val_acc'] for r in fold_results]
    
    print(f"{'='*60}")
    print(f"SUMMARY:")
    print(f"  Train accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"  Val accuracy:   {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"{'='*60}")
    
    return fold_results

# Run CV
results_with_diag = cross_validate_simple(X_with_diag, y, subjects, n_splits=3)


# ==============================================================================
# STEP 7: COMPARE WITH vs WITHOUT DIAGONAL
# ==============================================================================
print("\nSTEP 7: Creating dataset WITHOUT diagonal for comparison...")

X_no_diag, y_no_diag, subjects_no_diag = create_classification_dataset(
    df, 
    connection_columns, 
    region_to_idx, 
    n_regions,
    include_diagonal=False
)

print(f"\n✓ Dataset WITHOUT diagonal:")
print(f"  X shape: {X_no_diag.shape}")
print(f"  Number of features: {X_no_diag.shape[1]}")

# Check zeros (should be very few or none)
n_zeros_no_diag = (X_no_diag == 0.0).sum()
print(f"  Zeros: {n_zeros_no_diag} ({100*n_zeros_no_diag/X_no_diag.size:.2f}%)")

# Run CV without diagonal
results_no_diag = cross_validate_simple(X_no_diag, y_no_diag, subjects_no_diag, n_splits=3)


# ==============================================================================
# STEP 8: VISUALIZE RESULTS
# ==============================================================================
print("\nSTEP 8: Creating visualization...")

# Extract accuracies
with_diag_val = [r['val_acc'] for r in results_with_diag]
no_diag_val = [r['val_acc'] for r in results_no_diag]

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Fold-by-fold comparison
folds = [1, 2, 3]
axes[0].plot(folds, with_diag_val, 'o-', label='WITH diagonal (232 features)', 
             linewidth=2, markersize=10, color='#2ecc71')
axes[0].plot(folds, no_diag_val, 's-', label='WITHOUT diagonal (231 features)', 
             linewidth=2, markersize=10, color='#e74c3c')
axes[0].axhline(1/n_regions, color='red', linestyle='--', 
                label=f'Chance ({100/n_regions:.2f}%)', linewidth=1)
axes[0].set_xlabel('Fold', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Validation Accuracy by Fold', fontsize=14, fontweight='bold')
axes[0].set_xticks(folds)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0, 1.05])

# Plot 2: Mean comparison with error bars
means = [np.mean(with_diag_val), np.mean(no_diag_val)]
stds = [np.std(with_diag_val), np.std(no_diag_val)]
labels = ['WITH diagonal\n(232 features)', 'WITHOUT diagonal\n(231 features)']
colors = ['#2ecc71', '#e74c3c']

bars = axes[1].bar(labels, means, yerr=stds, capsize=10, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=2)
axes[1].axhline(1/n_regions, color='red', linestyle='--', label='Chance', linewidth=2)
axes[1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('Mean Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1.05])
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.2%}\n±{std:.2%}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('diagonal_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure: diagonal_comparison.png")
plt.close()


# ==============================================================================
# STEP 9: DIAGNOSTIC SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

print("\n1. DATA LOADING:")
print(f"   ✓ Loaded {len(df)} subjects")
print(f"   ✓ Extracted {n_regions} regions")
print(f"   ✓ Found {len(connection_columns)} connections")

print("\n2. PREPROCESSING:")
print(f"   ✓ Reconstructed {n_regions}×{n_regions} connectivity matrices")
print(f"   ✓ Set diagonal to 0.0 (deterministic)")
print(f"   ✓ NO Fisher Z transformation")
print(f"   ✓ NO StandardScaler")
print(f"   ✓ Raw correlation values fed directly to model")

print("\n3. DATASET WITH DIAGONAL:")
print(f"   ✓ Features: {X_with_diag.shape[1]}")
print(f"   ✓ Samples: {X_with_diag.shape[0]}")
print(f"   ✓ Zeros: {(X_with_diag == 0.0).sum()} ({100*(X_with_diag == 0.0).sum()/X_with_diag.size:.2f}%)")
print(f"   ✓ Expected zeros: {X_with_diag.shape[0]} (1 per sample)")

print("\n4. CROSS-VALIDATION RESULTS:")
with_diag_mean = np.mean(with_diag_val)
no_diag_mean = np.mean(no_diag_val)

print(f"   WITH diagonal:    {with_diag_mean:.2%} ± {np.std(with_diag_val):.2%}")
print(f"   WITHOUT diagonal: {no_diag_mean:.2%} ± {np.std(no_diag_val):.2%}")

print("\n5. INTERPRETATION:")
if with_diag_mean > 0.95:
    print("   ✓ EXCELLENT: Model correctly learned the diagonal=0 pattern!")
    print("   ✓ Zeros are preserved through preprocessing")
    print("   ✓ No scaling is destroying the signal")
elif with_diag_mean > 0.85:
    print("   ⚠️  GOOD but not great: Model partially learned the pattern")
    print("   → Possible issue: Some preprocessing affecting zeros")
elif with_diag_mean > 0.70:
    print("   ⚠️  WARNING: Model is NOT leveraging the diagonal signal well")
    print("   → Problem: Zeros might be getting normalized/transformed")
    print("   → Check if StandardScaler or other preprocessing is active")
else:
    print("   ✗ FAIL: Model performance similar to WITHOUT diagonal")
    print("   → Critical issue: Diagonal is NOT in features or being destroyed")

print("\n6. EXPECTED vs ACTUAL:")
print(f"   Expected WITH diagonal:    98-99%")
print(f"   Actual WITH diagonal:      {with_diag_mean:.2%}")
print(f"   Expected WITHOUT diagonal: 70-90%")
print(f"   Actual WITHOUT diagonal:   {no_diag_mean:.2%}")

difference = with_diag_mean - no_diag_mean
print(f"\n7. DIAGONAL CONTRIBUTION:")
print(f"   Accuracy gain from diagonal: {difference:.2%}")
if difference < 0.05:
    print("   → ⚠️  Very small gain - diagonal signal might be compromised!")
elif difference < 0.15:
    print("   → Moderate gain - some benefit but not as expected")
else:
    print("   → ✓ Good gain - diagonal signal is being used")

print("="*70)
print("\n✅ Test complete! Check 'diagonal_comparison.png' for visualization.")
print()