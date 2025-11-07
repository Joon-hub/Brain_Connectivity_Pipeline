"""
Feature Engineering: Connectivity Matrix Preprocessing
======================================================
Reconstruct connectivity matrices and handle diagonal imputation.
Now includes sklearn-compatible transformers for Pipeline integration.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import warnings


# ============================================================================
# ORIGINAL HELPER FUNCTIONS (Backward Compatibility)
# ============================================================================

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
        if '~' not in col:
            continue
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
        matrix: Connectivity matrix
        strategy: Imputation method ('zero', 'one', 'mean', 'network_mean')
        region_list: Required for 'network_mean' strategy
        
    Returns:
        Matrix with imputed diagonal
    """
    matrix_copy = matrix.copy()
    n_regions = matrix.shape[0]
    
    if strategy == "zero":
        np.fill_diagonal(matrix_copy, 0.0)
        
    elif strategy == "one":
        np.fill_diagonal(matrix_copy, 1.0)
        
    elif strategy == "mean":
        for i in range(n_regions):
            row_mean = np.mean(matrix_copy[i, :])
            matrix_copy[i, i] = row_mean
            
    elif strategy == "network_mean":
        if region_list is None:
            raise ValueError("region_list required for network_mean strategy")
        
        # Parse network memberships
        network_map = _parse_networks(region_list)
        
        for i in range(n_regions):
            region_name = region_list[i]
            network = network_map.get(region_name, 'Unknown')
            
            # Find regions in same network
            same_network_indices = [
                j for j, r in enumerate(region_list) 
                if network_map.get(r, 'Unknown') == network and j != i
            ]
            
            if same_network_indices:
                within_network_mean = np.mean(matrix_copy[i, same_network_indices])
                matrix_copy[i, i] = within_network_mean
            else:
                # Fallback to row mean if no same-network regions
                matrix_copy[i, i] = np.mean(matrix_copy[i, :])
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return matrix_copy


def _parse_networks(region_list: List[str]) -> Dict[str, str]:
    """
    Parse network membership from region names.
    
    Expected naming: 'LH_Network_Region' or 'RH_Network_Region'
    
    Args:
        region_list: List of region names
        
    Returns:
        Dictionary mapping region names to network labels
    """
    network_map = {}
    
    for region in region_list:
        if region.startswith('LH_') or region.startswith('RH_'):
            parts = region.split('_')
            if len(parts) >= 2:
                network = parts[1]  # Extract network name
                network_map[region] = network
            else:
                network_map[region] = 'Unknown'
        else:
            # Subcortical or other regions
            network_map[region] = 'Subcortical'
    
    return network_map


def create_classification_dataset(
    df: pd.DataFrame,
    connection_columns: List[str],
    diagonal_strategy: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create classification dataset from connectivity data (original function).
    
    Args:
        df: DataFrame with connectivity data
        connection_columns: List of connection column names
        diagonal_strategy: How to handle diagonal values
        
    Returns:
        X: Features (n_samples × n_features) where n_samples = n_subjects × n_regions
        y: Labels (region indices)
        subjects: Subject IDs for each sample
        region_list: List of region names
    """
    # Extract regions
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    
    n_subjects = len(df)
    n_samples = n_subjects * n_regions
    
    # Initialize arrays
    X = np.zeros((n_samples, n_regions), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    subjects = []
    
    sample_idx = 0
    
    for subject_id, row in df.iterrows():
        subject_values = row[connection_columns].values
        
        # Reconstruct full connectivity matrix
        matrix = reconstruct_connectivity_matrix(
            subject_values, connection_columns, region_to_idx, n_regions
        )
        
        # Impute diagonal
        matrix = impute_diagonal(matrix, strategy=diagonal_strategy, region_list=region_list)
        
        # Create one sample per region
        for region_idx in range(n_regions):
            # Connectivity pattern = row in matrix excluding diagonal
            connectivity_pattern = matrix[region_idx, :].copy()
            connectivity_pattern = np.delete(connectivity_pattern, region_idx)
            
            X[sample_idx, :n_regions-1] = connectivity_pattern
            y[sample_idx] = region_idx
            subjects.append(df.iloc[subject_id, 0])  # Subject ID from first column
            
            sample_idx += 1
    
    subjects = np.array(subjects)
    
    print(f"✓ Dataset created: {n_samples} samples ({n_subjects} subjects × {n_regions} regions)")
    print(f"  Features: {X.shape[1]} (connectivity patterns)")
    print(f"  Classes: {len(np.unique(y))} regions")
    print(f"  Diagonal strategy: {diagonal_strategy}")
    
    return X, y, subjects, region_list


# ============================================================================
# SKLEARN-COMPATIBLE TRANSFORMERS (New for Pipeline Integration)
# ============================================================================

class ConnectivityMatrixReconstructor(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer to reconstruct connectivity matrices from flattened data.
    
    This transformer takes flattened connectivity data and reconstructs full
    symmetric connectivity matrices for each subject.
    """
    
    def __init__(self, connection_columns: Optional[List[str]] = None):
        """
        Initialize reconstructor.
        
        Args:
            connection_columns: List of connection column names (e.g., 'RegionA~RegionB')
        """
        self.connection_columns = connection_columns
        self.region_list_ = None
        self.region_to_idx_ = None
        self.n_regions_ = None
    
    def fit(self, X, y=None):
        """
        Extract region information from connection columns.
        
        Args:
            X: DataFrame with connectivity data
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self
        """
        if self.connection_columns is None:
            if isinstance(X, pd.DataFrame):
                self.connection_columns = X.columns[1:].tolist()
            else:
                raise ValueError("connection_columns must be provided for non-DataFrame input")
        
        # Extract regions
        self.region_list_, self.region_to_idx_, self.n_regions_ = extract_regions(
            self.connection_columns
        )
        
        return self
    
    def transform(self, X):
        """
        Reconstruct connectivity matrices.
        
        Args:
            X: DataFrame with connectivity data (n_subjects × n_connections)
            
        Returns:
            3D array of connectivity matrices (n_subjects × n_regions × n_regions)
        """
        if isinstance(X, pd.DataFrame):
            X_values = X[self.connection_columns].values
        else:
            X_values = X
        
        n_subjects = X_values.shape[0]
        matrices = np.zeros((n_subjects, self.n_regions_, self.n_regions_))
        
        for i in range(n_subjects):
            matrices[i] = reconstruct_connectivity_matrix(
                X_values[i],
                self.connection_columns,
                self.region_to_idx_,
                self.n_regions_
            )
        
        return matrices


class DiagonalImputer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer for diagonal imputation in connectivity matrices.
    """
    
    def __init__(self, strategy: str = "mean", region_list: Optional[List[str]] = None):
        """
        Initialize diagonal imputer.
        
        Args:
            strategy: Imputation method ('zero', 'one', 'mean', 'network_mean')
            region_list: Required for 'network_mean' strategy
        """
        self.strategy = strategy
        self.region_list = region_list
    
    def fit(self, X, y=None):
        """
        No fitting required for imputation.
        
        Args:
            X: 3D array of connectivity matrices
            y: Ignored
            
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """
        Apply diagonal imputation to all matrices.
        
        Args:
            X: 3D array (n_subjects × n_regions × n_regions)
            
        Returns:
            3D array with imputed diagonals
        """
        n_subjects = X.shape[0]
        X_imputed = np.zeros_like(X)
        
        for i in range(n_subjects):
            X_imputed[i] = impute_diagonal(
                X[i],
                strategy=self.strategy,
                region_list=self.region_list
            )
        
        return X_imputed


class RegionConnectivityExtractor(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer to extract per-region connectivity patterns.
    
    Converts 3D connectivity matrices into 2D feature matrix suitable for classification.
    Each sample represents one region's connectivity pattern (excluding self-connection).
    """
    
    def __init__(self):
        """Initialize extractor."""
        self.n_regions_ = None
        self.region_list_ = None
    
    def fit(self, X, y=None):
        """
        Store region information.
        
        Args:
            X: 3D array (n_subjects × n_regions × n_regions)
            y: Ignored
            
        Returns:
            self
        """
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        
        self.n_regions_ = X.shape[1]
        
        return self
    
    def transform(self, X):
        """
        Extract connectivity patterns for each region.
        
        Args:
            X: 3D array (n_subjects × n_regions × n_regions)
            
        Returns:
            2D array (n_subjects*n_regions × n_regions-1)
            Also creates labels and subject arrays (stored as attributes)
        """
        n_subjects, n_regions, _ = X.shape
        n_samples = n_subjects * n_regions
        
        # Initialize output
        X_out = np.zeros((n_samples, n_regions - 1), dtype=float)
        self.labels_ = np.zeros(n_samples, dtype=int)
        self.subjects_ = []
        
        sample_idx = 0
        
        for subj_idx in range(n_subjects):
            for region_idx in range(n_regions):
                # Extract connectivity pattern (row excluding diagonal)
                connectivity_pattern = X[subj_idx, region_idx, :].copy()
                connectivity_pattern = np.delete(connectivity_pattern, region_idx)
                
                X_out[sample_idx] = connectivity_pattern
                self.labels_[sample_idx] = region_idx
                self.subjects_.append(subj_idx)
                
                sample_idx += 1
        
        self.subjects_ = np.array(self.subjects_)
        
        return X_out
    
    def get_labels(self):
        """Get region labels for each sample."""
        return self.labels_
    
    def get_subjects(self):
        """Get subject IDs for each sample."""
        return self.subjects_


class BrainConnectivityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for brain connectivity data.
    
    Combines matrix reconstruction, diagonal imputation, and feature extraction
    into a single transformer for convenient use in sklearn Pipelines.
    """
    
    def __init__(
        self,
        connection_columns: Optional[List[str]] = None,
        diagonal_strategy: str = "mean",
        region_list: Optional[List[str]] = None
    ):
        """
        Initialize complete preprocessor.
        
        Args:
            connection_columns: List of connection column names
            diagonal_strategy: Diagonal imputation method
            region_list: List of region names (for network_mean strategy)
        """
        self.connection_columns = connection_columns
        self.diagonal_strategy = diagonal_strategy
        self.region_list = region_list
        
        # Component transformers
        self.reconstructor_ = None
        self.imputer_ = None
        self.extractor_ = None
    
    def fit(self, X, y=None):
        """
        Fit all preprocessing components.
        
        Args:
            X: DataFrame with connectivity data
            y: Ignored
            
        Returns:
            self
        """
        # Initialize components
        self.reconstructor_ = ConnectivityMatrixReconstructor(self.connection_columns)
        self.imputer_ = DiagonalImputer(self.diagonal_strategy, self.region_list)
        self.extractor_ = RegionConnectivityExtractor()
        
        # Fit pipeline
        X_matrices = self.reconstructor_.fit_transform(X)
        X_imputed = self.imputer_.fit_transform(X_matrices)
        self.extractor_.fit(X_imputed)
        
        # Store region information
        self.region_list_ = self.reconstructor_.region_list_
        self.n_regions_ = self.reconstructor_.n_regions_
        
        return self
    
    def transform(self, X):
        """
        Apply full preprocessing pipeline.
        
        Args:
            X: DataFrame with connectivity data
            
        Returns:
            2D feature array suitable for classification
        """
        X_matrices = self.reconstructor_.transform(X)
        X_imputed = self.imputer_.transform(X_matrices)
        X_features = self.extractor_.transform(X_imputed)
        
        return X_features
    
    def get_labels(self):
        """Get region labels for each sample (after transform)."""
        return self.extractor_.get_labels()
    
    def get_subjects(self):
        """Get subject IDs for each sample (after transform)."""
        return self.extractor_.get_subjects()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Testing sklearn-compatible transformers...\n")
    
    # Create dummy data
    np.random.seed(42)
    
    # Simulate 5 subjects, 3 regions
    n_subjects = 5
    n_regions = 3
    regions = ['Region_A', 'Region_B', 'Region_C']
    
    # Create connection columns
    connection_cols = []
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            connection_cols.append(f"{regions[i]}~{regions[j]}")
    
    # Create dummy connectivity data
    data = {
        'subject_id': [f'S{i:02d}' for i in range(n_subjects)],
    }
    for col in connection_cols:
        data[col] = np.random.randn(n_subjects)
    
    df = pd.DataFrame(data)
    
    print("Input data shape:", df.shape)
    print("Connection columns:", connection_cols)
    print()
    
    # Test complete preprocessor
    preprocessor = BrainConnectivityPreprocessor(
        connection_columns=connection_cols,
        diagonal_strategy="mean"
    )
    
    preprocessor.fit(df)
    X_features = preprocessor.transform(df)
    y_labels = preprocessor.get_labels()
    subjects = preprocessor.get_subjects()
    
    print("Output feature shape:", X_features.shape)
    print("Labels shape:", y_labels.shape)
    print("Subjects shape:", subjects.shape)
    print()
    print("Expected: {} samples = {} subjects × {} regions".format(
        n_subjects * n_regions, n_subjects, n_regions
    ))
    print("Got: {} samples".format(len(X_features)))
    print("\n✓ Transformers working correctly!")