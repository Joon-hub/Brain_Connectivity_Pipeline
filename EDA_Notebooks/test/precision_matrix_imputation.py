"""
Precision Matrix Imputation and Brain Connectivity Classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def extract_regions(connection_columns):
    unique_regions = []
    seen = set()
    
    for col in connection_columns:
        region_a, region_b = col.split('~', 1)
        for region in [region_a, region_b]:
            if region not in seen:
                seen.add(region)
                unique_regions.append(region)
    
    region_to_idx = {region: idx for idx, region in enumerate(unique_regions)}
    n_regions = len(unique_regions)
    
    return unique_regions, region_to_idx, n_regions

def reconstruct_matrices_from_dataframe(df, connection_columns, region_to_idx, n_regions):
    n_subjects = df.shape[0]
    matrices = np.zeros((n_subjects, n_regions, n_regions))
    
    values = df[connection_columns].values
    
    for subj_idx in range(n_subjects):
        matrix = matrices[subj_idx]
        for col_idx, col in enumerate(connection_columns):
            region_a, region_b = col.split('~', 1)
            idx_a = region_to_idx[region_a]
            idx_b = region_to_idx[region_b]
            
            # Place value symmetrically
            value = values[subj_idx, col_idx]
            matrix[idx_a, idx_b] = value
            matrix[idx_b, idx_a] = value
        
        # Self-correlations
        np.fill_diagonal(matrix, 1.0)
    
    return matrices 

def impute_diagonal_precision(
        matrices: np.ndarray,
        regularization: str = 'tikhonov',
        alpha: float = 0.1,
        normalize: bool = True,
) -> np.ndarray:
    """
    Impute diagonal using precision matrix (inverse covariance).

    NEUROSCIENCE BASIS:
    - Diagonal of precision matrix ≈ strength of anatomical self-connections
    - Stronger in sensory/motor regions
    - Captures direct dependencies (partial correlations)

    Args:
        matrices: (n_subjects, n_regions, n_regions) correlation matrices with diag = 1.0
        region_list: Optional list of region names (unused here but kept for API consistency)
        regularization: 'tikhonov' (recommended), 'none'
        alpha: Regularization strength (Tikhonov); typical [0.01–0.5]
        normalize: Scale precision diagonal to reasonable range

    Returns:
        matrices_imputed: Same shape, with diagonal replaced by precision diagonal
    """
    results = matrices.copy()
    n_subjects, n_regions, _ = matrices.shape

    for s in range(n_subjects):
        corr = matrices[s].copy()

        # Ensure valid correlation matrix 
        np.fill_diagonal(corr, 1.0) 
        corr = (corr + corr.T) / 2.0  # enforce perfect symmetry

        # Apply regularization
        if regularization == 'tikhonov':  # ridge regularization (L2)
            regularized = corr + alpha * np.eye(n_regions)
        elif regularization == 'none':
            regularized = corr
        else: 
            raise ValueError(f"Invalid regularization: {regularization}")
        
        # Invert to get precision matrix
        precision = np.linalg.inv(regularized)
        precision_diagonal = np.diag(precision)

        # Normalise to avoid extreme values 
        if normalize:
            max_abs = np.max(np.abs(precision_diagonal)) 
            
            # Scale to [-1,1] based on max_abs value
            if max_abs > 1.0:
                precision_diagonal = precision_diagonal / max_abs
        
        # Replace diagonal
        np.fill_diagonal(results[s], precision_diagonal)

    return results
def fisher_z_transform(matrix, clip_value=0.999999):
    # Clip values to avoid infinities at r = ±1
    clipped_matrix = np.clip(matrix, -clip_value, clip_value)
    
    # Apply transformation to the CLIPPED matrix
    return np.arctanh(clipped_matrix)


def train_classifier_with_cv(
    matrices: np.ndarray,
    n_splits: int = 3,
    C: float = 0.0343304473310619,
    random_state: int = 42
) -> Tuple[List[float], float, float]:
    """
    Train logistic regression classifier with group K-fold cross-validation.
    
    Args:
        matrices: Array of connectivity matrices (n_subjects, n_regions, n_regions)
        n_splits: Number of cross-validation folds
        C: Regularization parameter for logistic regression
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (fold_scores, mean_cv_accuracy, std_cv_accuracy)
    """
    np.random.seed(random_state)
    n_subjects, n_regions, _ = matrices.shape
    
    # Prepare data: each row of a matrix is a sample, label is the row index
    X = matrices.reshape(-1, n_regions)
    y = np.tile(np.arange(n_regions), n_subjects)
    groups = np.repeat(np.arange(n_subjects), n_regions)
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"Fold {fold}/{n_splits}")
        
        # Get unique subject indices for this fold
        train_subjects = np.unique(groups[train_idx])
        val_subjects = np.unique(groups[val_idx])
        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects: {val_subjects}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit scaler on training data only, then transform both
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver='saga',
            multi_class='multinomial'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val_scaled)
        score = accuracy_score(y_val, y_pred)
        
        fold_scores.append(score)
        print(f"Fold {fold} Accuracy: {score:.4f}\n")
    
    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)
    
    return fold_scores, mean_acc, std_acc


def main():
    """Main pipeline for processing and modeling brain connectivity data."""
    
    # 1. Load data
    print("Loading data...")
    data_path = "/home/sjoon/projects/brain_connectivity_classifier/data/raw/PIOP2_restingstate.csv"
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # 2. Extract connection columns and regions
    print("\nExtracting regions...")
    connection_columns = [col for col in df.columns if '~' in str(col)]
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    print(f"Found {n_regions} regions")
    print(f"Connection columns: {len(connection_columns)}")
    print(f"Sample regions: {region_list[:3]}")
    
    # 3. Reconstruct connectivity matrices
    print("\nReconstructing matrices...")
    matrices = reconstruct_matrices_from_dataframe(
        df, connection_columns, region_to_idx, n_regions
    )
    print(f"Matrices shape: {matrices.shape}")
    
    # 4. Apply Fisher Z-transformation
    print("\nApplying Fisher Z-transformation...")
    matrices_fz = apply_fisher_z_transform(matrices)
    print(f"Transformed matrices shape: {matrices_fz.shape}")
    print(f"\n5x5 sample (subject 0):\n{matrices_fz[0][:5, :5]}")
    print(f"\n5x5 sample (subject 1):\n{matrices_fz[1][:5, :5]}")
    
    # 5. Train classifier with cross-validation
    print("\nTraining classifier with cross-validation...")
    fold_scores, mean_acc, std_acc = train_classifier_with_cv(matrices_fz)
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results:")
    print(f"Mean CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Individual fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()