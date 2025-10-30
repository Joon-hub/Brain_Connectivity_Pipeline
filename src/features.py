"""
Feature Engineering: Connectivity Matrix Preprocessing
======================================================
Reconstruct connectivity matrices and handle diagonal imputation.
"""

import numpy as np
from typing import Tuple, Dict, List
import warnings


def extract_regions(connection_columns: List[str]) -> Tuple[List[str], Dict[str, int], int]:
    """
    Extract unique brain regions from connection column names.
    
    Args:
        connection_columns: List of 'Region_A~Region_B' column names
        
    Returns:
        region_list: Ordered list of region names
        region_to_idx: Mapping from name to index
        n_regions: Total number of regions
    """
    unique_regions = []
    seen = set()
    
    for col in connection_columns:
        if '~' not in col:
            continue
        
        region_a, region_b = col.split('~')
        
        for region in [region_a, region_b]:
            if region not in seen:
                seen.add(region)
                unique_regions.append(region)
    
    region_to_idx = {region: idx for idx, region in enumerate(unique_regions)}
    
    print(f"✓ Extracted {len(unique_regions)} unique brain regions")
    return unique_regions, region_to_idx, len(unique_regions)


def reconstruct_connectivity_matrix(
    subject_values: np.ndarray,
    connection_columns: List[str],
    region_to_idx: Dict[str, int],
    n_regions: int
) -> np.ndarray:
    """
    Reconstruct symmetric connectivity matrix from flattened data.
    
    Args:
        subject_values: Connectivity values for one subject
        connection_columns: Column names
        region_to_idx: Region name to index mapping
        n_regions: Total regions
        
    Returns:
        Symmetric connectivity matrix (n_regions × n_regions)
    """
    matrix = np.zeros((n_regions, n_regions), dtype=float)
    
    # Fill off-diagonal elements
    for col, value in zip(connection_columns, subject_values):
        region_a, region_b = col.split('~')
        idx_a = region_to_idx[region_a]
        idx_b = region_to_idx[region_b]
        
        matrix[idx_a, idx_b] = value
        matrix[idx_b, idx_a] = value  # Symmetric
    
    # Set diagonal to 1.0 (perfect self-correlation)
    np.fill_diagonal(matrix, 1.0)
    
    return matrix


def impute_diagonal(
    matrix: np.ndarray,
    strategy: str = "mean",
    region_list: List[str] = None
) -> np.ndarray:
    """
    Impute diagonal values using specified strategy.
    
    Args:
        matrix: Connectivity matrix (n_regions × n_regions)
        strategy: Imputation method ('zero', 'one', 'mean', 'network_mean')
        region_list: Region names (needed for network_mean strategy)
        
    Returns:
        Matrix with imputed diagonal
    """
    m = matrix.copy()
    n = m.shape[0]
    
    if strategy == "zero":
        np.fill_diagonal(m, 0.0)
    
    elif strategy == "one":
        np.fill_diagonal(m, 1.0)
    
    elif strategy == "mean":
        # Row-wise mean of off-diagonal elements
        for i in range(n):
            row_sum = np.sum(m[i, :]) - m[i, i]
            row_mean = row_sum / (n - 1) if n > 1 else 0.0
            m[i, i] = row_mean
    
    elif strategy == "network_mean":
        if region_list is None:
            warnings.warn("network_mean requires region_list. Falling back to 'mean'.")
            return impute_diagonal(matrix, strategy="mean")
        
        # Parse networks and compute within-network means
        networks = parse_networks(region_list)
        
        for i in range(n):
            same_network = [j for j in range(n) if networks[j] == networks[i] and j != i]
            
            if same_network:
                m[i, i] = np.mean(m[i, same_network])
            else:
                # Fallback to row mean
                row_sum = np.sum(m[i, :]) - m[i, i]
                m[i, i] = row_sum / (n - 1) if n > 1 else 0.0
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use: zero, one, mean, network_mean")
    
    return m

def parse_networks(region_list: List[str]) -> List[str]:
    """
    Parse network membership from region names.
    ...
    """
    networks = []
    
    for region in region_list:
        name = region.lower()
        network = 'Unknown'
        
        # --- Cortical (Schaefer 7 or 17 networks) ---
        if region.startswith(('LH_', 'RH_')):
            if 'viscent' in name or 'visperi' in name or 'striate' in name:
                network = 'Visual'
            elif 'sommota' in name or 'sommotb' in name:
                network = 'Somatomotor'
            elif 'dorsattna' in name or 'dorsattnb' in name:
                network = 'DorsalAttention'
            elif 'salventattna' in name or 'salventattnb' in name:
                network = 'SalienceVentralAttention'
            elif 'limbica' in name or 'limbicb' in name:
                network = 'Limbic'
            elif 'conta' in name or 'contb' in name or 'contc' in name:
                network = 'Control'
            elif 'defaulta' in name or 'defaultb' in name or 'defaultc' in name:
                network = 'DefaultMode'
            elif 'temppar' in name:
                network = 'TemporalParietal'
            else:
                network = 'CorticalOther'
        
        # Subcortical Tian regions
        else:
            if 'hip' in name:
                network = 'Hippocampus'
            elif 'amy' in name:
                network = 'Amygdala'
            elif 'th' in name:
                network = 'Thalamus'
            elif 'nac' in name:
                network = 'Accumbens'
            elif 'put' in name:
                network = 'Putamen'
            elif 'pallid' in name or 'gp' in name:
                network = 'Pallidum'
            elif 'caud' in name:
                network = 'Caudate'
            else:
                network = 'SubcorticalOther'
        
        networks.append(network)
    
    return networks


def create_classification_dataset(
    df, connection_columns: List[str], diagonal_strategy: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create full classification dataset from connectivity DataFrame.
    
    Args:
        df: Connectivity DataFrame
        connection_columns: Connection column names
        diagonal_strategy: How to handle diagonal elements
        
    Returns:
        X: Features (n_samples × n_regions)
        y: Labels (region indices)
        subjects: Subject IDs per sample
        region_list: Ordered region names
    """
    # Extract regions
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    
    # Prepare output arrays
    X_list = []
    y_list = []
    subject_list = []
    
    # Process each subject
    for i in range(len(df)):
        subject_id = df.iloc[i, 0]
        subject_values = df.iloc[i, 1:].to_numpy(dtype=float)
        
        # Reconstruct matrix
        matrix = reconstruct_connectivity_matrix(
            subject_values, connection_columns, region_to_idx, n_regions
        )
        
        # Impute diagonal
        matrix = impute_diagonal(matrix, strategy=diagonal_strategy, region_list=region_list)
        
        # Create one sample per region
        for region_idx in range(n_regions):
            X_list.append(matrix[region_idx, :])  # Row = connectivity profile
            y_list.append(region_idx)             # Label = region identity
            subject_list.append(subject_id)
    
    X = np.array(X_list)
    y = np.array(y_list)
    subjects = np.array(subject_list)
    
    print(f"✓ Created dataset: {X.shape[0]} samples ({len(df)} subjects × {n_regions} regions)")
    return X, y, subjects, region_list


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from data import load_connectivity_data, extract_connection_columns
    
    # Test on sample
    df = load_connectivity_data("../data/sample/sample_piop2.csv")
    conn_cols = extract_connection_columns(df)
    
    X, y, subjects, regions = create_classification_dataset(
        df, conn_cols, diagonal_strategy="mean"
    )
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Unique regions: {len(regions)}")
    print(f"First region: {regions[0]}")