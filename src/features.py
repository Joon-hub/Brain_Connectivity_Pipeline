"""
Feature Engineering: Connectivity Matrix Preprocessing
======================================================
Reconstruct connectivity matrices and handle diagonal imputation.
includes sklearn-compatible transformers for Pipeline integration.
Includes regression model training for predictive diagonal imputation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
import warnings
import pickle
from pathlib import Path

# Internal functions
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

def parse_networks(region_list: List[str]) -> Dict[str, str]:
    """
    Parse network membership from region names.
    
    Args:
        region_list: List of region names
        
    Returns:
        Dictionary mapping region names to network labels
    """
    network_map = {}
    
    for region in region_list:
        name = region.lower()
        network = 'Unknown'
        
        # --- Cortical (Schaefer 17 networks) ---
        if region.startswith(('LH_', 'RH_')):
            if 'viscent' in name or 'vis_cent' in name:
                network = 'VisCent'
            elif 'visperi' in name or 'vis_peri' in name:
                network = 'VisPeri'
            elif 'sommota' in name or 'senmota' in name:
                network = 'SomMotA'
            elif 'sommotb' in name or 'senmotb' in name:
                network = 'SomMotB'
            elif 'dorsattna' in name or 'dorsattn_a' in name:
                network = 'DorsAttnA'
            elif 'dorsattnb' in name or 'dorsattn_b' in name:
                network = 'DorsAttnB'
            elif 'salventattna' in name or 'salventattn_a' in name:
                network = 'SalVentAttnA'
            elif 'salventattnb' in name or 'salventattn_b' in name:
                network = 'SalVentAttnB'
            elif 'limbica' in name or 'limbic_a' in name:
                network = 'LimbicA'
            elif 'limbicb' in name or 'limbic_b' in name:
                network = 'LimbicB'
            elif 'conta' in name or 'cont_a' in name:
                network = 'ContA'
            elif 'contb' in name or 'cont_b' in name:
                network = 'ContB'
            elif 'contc' in name or 'cont_c' in name:
                network = 'ContC'
            elif 'defaulta' in name or 'default_a' in name:
                network = 'DefaultA'
            elif 'defaultb' in name or 'default_b' in name:
                network = 'DefaultB'
            elif 'defaultc' in name or 'default_c' in name:
                network = 'DefaultC'
            elif 'temppar' in name:
                network = 'TempPar'
            else:
                network = 'CorticalOther'
        
        # Subcortical Tian II regions
        else:
            if 'ahip' in name:
                network = 'Hippocampus_ant'
            elif 'phip' in name:
                network = 'Hippocampus_post'
            elif 'lamy' in name:
                network = 'Amygdala_lat'
            elif 'mamy' in name:
                network = 'Amygdala_med'
            elif 'tha-dp' in name or 'tha_dp' in name:
                network = 'Thalamus_DP'
            elif 'tha-vp' in name or 'tha_vp' in name:
                network = 'Thalamus_VP'
            elif 'tha-va' in name or 'tha_va' in name:
                network = 'Thalamus_VA'
            elif 'tha-da' in name or 'tha_da' in name:
                network = 'Thalamus_DA'
            elif 'nac-shell' in name or 'nac_shell' in name:
                network = 'Accumbens_shell'
            elif 'nac-core' in name or 'nac_core' in name:
                network = 'Accumbens_core'
            elif 'pgp' in name:
                network = 'Pallidum_post'
            elif 'agp' in name:
                network = 'Pallidum_ant'
            elif 'aput' in name:
                network = 'Putamen_ant'
            elif 'pput' in name:
                network = 'Putamen_post'
            elif 'acau' in name:
                network = 'Caudate_ant'
            elif 'pcau' in name:
                network = 'Caudate_post'
            else:
                network = 'SubcorticalOther'
        
        network_map[region] = network
    
    return network_map

def impute_diagonal(
    matrix: np.ndarray,
    strategy: str = "region_mean",
    region_list: Optional[List[str]] = None,
    region_models: Optional[Dict[str, Any]] = None,
    k_neighbors: int = 5
) -> np.ndarray:
    """
    Impute diagonal values using specified strategy.
    
    Args:
        matrix: The connectivity matrix (n_regions × n_regions) for one subject.
        strategy: Method ('zero', 'region_mean', 'random', 'network_mean', 'regression_predictive', 'knn_imputation').
        region_list: Required list of region names for 'network_mean' and 'regression_predictive'.
        region_models: Dictionary mapping region names to a fitted scikit-learn
                       model (e.g., Ridge) required for 'regression_predictive'.
        k_neighbors: Number of neighbors for 'knn_imputation' strategy.
        
    Returns:
        Matrix with imputed diagonal
    """
    matrix_copy = matrix.copy()
    n_regions = matrix.shape[0]
    
    # 1. Simple strategies (Zero)
    if strategy == "zero":
        np.fill_diagonal(matrix_copy, 0.0)
    
    # 2. Simple strategies (Random values between -1 and 1)
    elif strategy == "random":
        np.fill_diagonal(matrix_copy, np.random.uniform(-1, 1, n_regions))

    # 3. Statistical strategies (region_mean per region/subject)
    elif strategy == "region_mean":
        for i in range(n_regions):
            row_mean = np.mean(matrix_copy[i, :])
            matrix_copy[i, i] = row_mean
  
    # 4. Statistical strategies (network_mean per region/subject)       
    elif strategy == "network_mean":
        if region_list is None:
            raise ValueError("region_list required for network_mean strategy")
        
        # Parse network memberships
        network_map = parse_networks(region_list)
        
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

    # 5. Regression strategies (Region-specific predictions)
    elif strategy == "regression_predictive":
        if region_list is None or region_models is None:
            raise ValueError(
                "Both 'region_list' and pre-trained 'region_models' are required "
                "for the 'regression_predictive' strategy."
            )
        
        for i in range(n_regions):
            region_name = region_list[i]
            model = region_models.get(region_name)
            
            if model is None:
                warnings.warn(
                    f"No trained model found for region: {region_name}. "
                    f"Falling back to 'region_mean' strategy for this region.", 
                    UserWarning
                )
                matrix_copy[i, i] = np.mean(matrix_copy[i, :])
                continue

            # a. Extract Features (X): The connectivity pattern for region i, 
            #    EXCLUDING the diagonal element (which we are predicting).
            connectivity_pattern = matrix_copy[i, :].copy()
            features = np.delete(connectivity_pattern, i)
            
            # b. Reshape for the model: sklearn models usually expect X to be 2D 
            #    (n_samples x n_features). Here n_samples=1.
            features = features.reshape(1, -1) 
            
            # c. Predict the missing diagonal value (M_i,i)
            predicted_M_ii = model.predict(features)[0]
            
            # d. Impute the diagonal
            matrix_copy[i, i] = predicted_M_ii

    # 6. KNN-based diagonal imputation
    elif strategy == "knn_imputation":
        # Remove diagonals temporarily (set to NaN)
        matrix_temp = matrix_copy.copy()
        np.fill_diagonal(matrix_temp, np.nan)
        
        # Each row is a region's connectivity pattern (features)
        # We will use NearestNeighbors to find similar regions
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_regions), metric='euclidean')
        nbrs.fit(np.nan_to_num(matrix_temp))
        
        for i in range(n_regions):
            # Find k nearest neighbors excluding self
            distances, indices = nbrs.kneighbors(matrix_temp[i, :].reshape(1, -1))
            neighbor_indices = [idx for idx in indices[0] if idx != i][:k_neighbors]

            # Compute mean of available diagonals among neighbors
            neighbor_diagonals = [
                matrix_copy[j, j] for j in neighbor_indices
                if not np.isnan(matrix_copy[j, j])
            ]

            if neighbor_diagonals:
                matrix_copy[i, i] = np.mean(neighbor_diagonals)
            else:
                # Fallback to row mean if no valid neighbor diagonals
                matrix_copy[i, i] = np.nanmean(matrix_temp[i, :])
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return matrix_copy


# ============================================================================
# REGRESSION MODEL TRAINING FOR PREDICTIVE DIAGONAL IMPUTATION
# ============================================================================

def train_region_models(
    df: pd.DataFrame,
    connection_columns: List[str],
    alpha: float = 1.0,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Train one Ridge regression model per region to predict its diagonal value.
    
    This function trains region-specific models that can later be used with the
    'regression_predictive' diagonal imputation strategy.
    
    Args:
        df: DataFrame with connectivity data
        connection_columns: List of connection column names
        alpha: Ridge regression regularization parameter
        verbose: Print progress messages
        
    Returns:
        Tuple of (region_models, region_list):
            - region_models: Dictionary mapping region names to trained models
            - region_list: List of region names
    """
    if verbose:
        print("Training region-specific diagonal prediction models...")
    
    # Extract regions
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    
    # Store models
    region_models = {}
    
    # For each region, collect training data
    for region_idx in range(n_regions):
        region_name = region_list[region_idx]
        
        X_train = []  # Features: connectivity patterns (excluding diagonal)
        y_train = []  # Target: actual diagonal value
        
        # Iterate through all subjects
        for idx, row in df.iterrows():
            subject_values = row[connection_columns].values
            
            # Reconstruct full matrix with diagonal = 1.0 (original values)
            matrix = reconstruct_connectivity_matrix(
                subject_values, connection_columns, region_to_idx, n_regions
            )
            
            # Extract features: connectivity pattern for this region (excluding diagonal)
            connectivity_pattern = matrix[region_idx, :].copy()
            features = np.delete(connectivity_pattern, region_idx)
            
            # Target: the actual diagonal value (1.0 in correlation matrices)
            target = matrix[region_idx, region_idx]
            
            X_train.append(features)
            y_train.append(target)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train Ridge regression model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        region_models[region_name] = model
        
        if verbose and (region_idx + 1) % 10 == 0:
            print(f"  Trained {region_idx + 1}/{n_regions} models...")
    
    if verbose:
        print(f"✓ Trained {n_regions} region-specific models")
    
    return region_models, region_list


def save_region_models(
    region_models: Dict[str, Any],
    region_list: List[str],
    output_path: str,
    verbose: bool = True
) -> None:
    """
    Save trained region models to disk.
    
    Args:
        region_models: Dictionary mapping region names to trained models
        region_list: List of region names
        output_path: Path where to save the models
        verbose: Print confirmation message
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'region_models': region_models,
        'region_list': region_list
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    if verbose:
        print(f"✓ Models saved to: {output_path}")


def load_region_models(filepath: str, verbose: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    """
    Load trained region models from disk.
    
    Args:
        filepath: Path to the saved models file
        verbose: Print confirmation message
        
    Returns:
        Tuple of (region_models, region_list):
            - region_models: Dictionary mapping region names to trained models
            - region_list: List of region names
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if verbose:
        print(f"✓ Models loaded from: {filepath}")
    
    return data['region_models'], data['region_list']


# ============================================================================
# ORIGINAL DATASET CREATION FUNCTION
# ============================================================================

def create_classification_dataset(
    df: pd.DataFrame,
    connection_columns: List[str],
    diagonal_strategy: str = "region_mean",
    region_models: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create classification dataset from connectivity data (original function).
    
    Args:
        df: DataFrame with connectivity data
        connection_columns: List of connection column names
        diagonal_strategy: How to handle diagonal values
        region_models: Optional pre-trained models for regression_predictive strategy
        
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
    X = np.zeros((n_samples, n_regions - 1), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    subjects = []
    
    sample_idx = 0
    
    for idx, row in df.iterrows():
        subject_values = row[connection_columns].values
        
        # Reconstruct full connectivity matrix
        matrix = reconstruct_connectivity_matrix(
            subject_values, connection_columns, region_to_idx, n_regions
        )
        
        # Impute diagonal
        matrix = impute_diagonal(
            matrix, 
            diagonal_strategy,  # Positional argument
            region_list,
            region_models
        )
        
        # Create one sample per region
        for region_idx in range(n_regions):
            # Connectivity pattern = row in matrix excluding diagonal
            connectivity_pattern = matrix[region_idx, :].copy()
            connectivity_pattern = np.delete(connectivity_pattern, region_idx)
            
            X[sample_idx] = connectivity_pattern
            y[sample_idx] = region_idx
            subjects.append(row.iloc[0])  # Subject ID from first column
            
            sample_idx += 1
    
    subjects = np.array(subjects)
    
    print(f"✓ Dataset created: {n_samples} samples ({n_subjects} subjects × {n_regions} regions)")
    print(f"  Features: {X.shape[1]} (connectivity patterns)")
    print(f"  Classes: {len(np.unique(y))} regions")
    print(f"  Diagonal strategy: {diagonal_strategy}")
    
    return X, y, subjects, region_list


# ============================================================================
# SKLEARN-COMPATIBLE TRANSFORMERS 
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
    
    def __init__(
        self, 
        strategy: str = "region_mean",
        region_list: Optional[List[str]] = None,
        region_models: Optional[Dict[str, Any]] = None,
        k_neighbors: int = 5
    ):
        """
        Initialize diagonal imputer.
        
        Args:
            strategy: Imputation method ('zero', 'region_mean', 'random', 'network_mean', 
                     'regression_predictive', 'knn_imputation')
            region_list: Required for 'network_mean' and 'regression_predictive' strategies
            region_models: Required for 'regression_predictive' strategy
            k_neighbors: Number of neighbors for 'knn_imputation' strategy
        """
        self.strategy = strategy
        self.region_list = region_list
        self.region_models = region_models
        self.k_neighbors = k_neighbors
    
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
                self.strategy,
                self.region_list,
                self.region_models,
                self.k_neighbors
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
        self.labels_ = None
        self.subjects_ = None
    
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
        if self.labels_ is None:
            raise ValueError("Transform must be called before getting labels")
        return self.labels_
    
    def get_subjects(self):
        """Get subject IDs for each sample."""
        if self.subjects_ is None:
            raise ValueError("Transform must be called before getting subjects")
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
        diagonal_strategy: str = "region_mean",
        region_list: Optional[List[str]] = None,
        region_models: Optional[Dict[str, Any]] = None,
        k_neighbors: int = 5
    ):
        """
        Initialize complete preprocessor.
        
        Args:
            connection_columns: List of connection column names
            diagonal_strategy: Diagonal imputation method
            region_list: List of region names (for network_mean strategy)
            region_models: Pre-trained models for regression_predictive strategy
            k_neighbors: Number of neighbors for knn_imputation strategy
        """
        self.connection_columns = connection_columns
        self.diagonal_strategy = diagonal_strategy
        self.region_list = region_list
        self.region_models = region_models
        self.k_neighbors = k_neighbors
        
        # Component transformers
        self.reconstructor_ = None
        self.imputer_ = None
        self.extractor_ = None
        self.region_list_ = None
        self.n_regions_ = None
    
    def fit(self, X, y=None):
        """
        Fit all preprocessing components.
        
        Args:
            X: DataFrame with connectivity data
            y: Ignored
            
        Returns:
            self
        """
        # Initialize reconstructor
        self.reconstructor_ = ConnectivityMatrixReconstructor(self.connection_columns)
        
        # Fit reconstructor to get region info
        X_matrices = self.reconstructor_.fit_transform(X)
        
        # Store region information
        self.region_list_ = self.reconstructor_.region_list_
        self.n_regions_ = self.reconstructor_.n_regions_

        # Initialize imputer with the derived region_list
        self.imputer_ = DiagonalImputer(
            self.diagonal_strategy, 
            self.region_list_,
            self.region_models,
            self.k_neighbors
        )
        
        # Continue with pipeline
        X_imputed = self.imputer_.fit_transform(X_matrices)
        self.extractor_ = RegionConnectivityExtractor()
        self.extractor_.fit(X_imputed)
        
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
        diagonal_strategy="region_mean"
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
    
    # Test regression model training
    print("\n" + "="*60)
    print("Testing regression model training...")
    print("="*60)
    
    region_models, region_list = train_region_models(
        df, connection_cols, alpha=1.0, verbose=True
    )
    
    print(f"\nTrained models for regions: {region_list}")
    print(f"Total models: {len(region_models)}")
    
    # Test with regression_predictive strategy
    print("\nTesting regression_predictive strategy...")
    preprocessor_reg = BrainConnectivityPreprocessor(
        connection_columns=connection_cols,
        diagonal_strategy="regression_predictive",
        region_models=region_models
    )
    
    preprocessor_reg.fit(df)
    X_features_reg = preprocessor_reg.transform(df)
    
    print(f"✓ Regression_predictive strategy works!")
    print(f"  Output shape: {X_features_reg.shape}")