"""
Feature Enginerring: Connectivity Matrix Processing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================
# Helper function to extract unique brain regions from connection column names
# ==============================

def extract_regions(connection_columns: List[str]) -> Tuple[List[str], Dict[str, int], int]:
    """
    Extract unique brain regions from connection column names.

    Parameters:
    connection_columns

    Returns:
    regions : Ordered list of unique brain regions.
    region_to_idx : Mapping from region name to its index.
    n_regions : Total number of unique regions.
    """
    unique_regions = []
    seen = set() # To maintain order of first appearance

    for col in connection_columns:
        if '~' not in col:
            raise ValueError(f"Invalid column name format: {col}. Expected 'RegionA~RegionB'.")
        
        region_a, region_b = col.split('~')
        for region in [region_a, region_b]:         # Iterate over both regions in the connection   
            if region not in seen:
                seen.add(region)                    # Keep track of seen regions
                unique_regions.append(region)       # Maintain order

    # Create mapping from region to index
    region_to_idx = {region: idx for idx, region in enumerate(unique_regions)}
    n_regions = len(unique_regions)

    return unique_regions, region_to_idx, n_regions

def reconstruct_matrices_from_dataframe(
        df: pd.DataFrame,
        connection_columns: List[str],
        region_to_idx: Dict[str, int],
        n_regions: int
) -> np.ndarray:
    """"
    Reconstruct connectivity matrices from flattened DataFrame.

    Parameters:
    df : DataFrame containing flattened connectivity data.
    connection_columns : List of connection column names in the format 'RegionA~RegionB'.
    region_to_idx : Mapping from region name to its index.
    n_regions : Total number of unique regions.

    Returns:
    connectivity_matrix : 3D array (n_subjects x n_regions x n_regions).
    """
    n_subjects = df.shape[0]
    matrices = np.zeros((n_subjects, n_regions, n_regions))

    # Get values as numpy array for efficiency
    values = df[connection_columns].values

    # Populate the connectivity matrices
    for subj_idx in range(n_subjects):
        matrix = matrices[subj_idx]
        for col_idx, col in enumerate(connection_columns):
            region_a, region_b = col.split('~', 1)
            idx_a = region_to_idx[region_a]
            idx_b = region_to_idx[region_b]

            value = values[subj_idx, col_idx]
            matrix[idx_a, idx_b] = value
            matrix[idx_b, idx_a] = value  # Ensure symmetry

        # Set diagonal to 1.0 (self-correlations)
        np.fill_diagonal(matrix, 1.0)
    return matrices

def parse_networks(region_list: List[str]) -> Dict[str, str]:
    """ Parse network memberships from region names. """
    network_map = {}

    for region in region_list:
        name = region.lower()
        network = 'unknown'

        # Cortical (Schaefer 17 networks)
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
        
        # Subcortical (Tian II)
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

# ==============================     
# Diagonal Imputation Methods 
# ==============================
def impute_diagonal_zero(matrix: np.ndarray) -> np.ndarray:
    """ Impute diagonal with zeros. """
    result = matrix.copy()
    n_subjects = result.shape[0]

    for s in range(n_subjects):
        np.fill_diagonal(result[s], 0.0)
    return result

def impute_diagonal_random(matrices: np.ndarray) -> np.ndarray:
    """ Impute diagonal with random values (Stochastic - TRUE randomness per call)"""
    result = matrices.copy()
    n_subjects = result.shape[0]
    n_regions = result.shape[1]

    for s in range(n_subjects):
        # Generate fresh random values on EVERY call (no fixed seed)
        random_values = np.random.uniform(-1, 1, n_regions)
        np.fill_diagonal(result[s], random_values)
    return result

def impute_diagonal_region_mean(matrices: np.ndarray) -> np.ndarray:
    """ 
    Impute diagonal with region-wise mean connectivity (Deterministic, subject-specific). 
    Each subject gets their own row means as diagonal values.
    """
    result = matrices.copy()
    n_subjects = result.shape[0]
    n_regions = result.shape[1]

    # create mask for off-diagonal elements
    off_diag_mask = ~np.eye(n_regions, dtype=bool)

    for s in range(n_subjects):
        matrix = result[s]

        # compute row means excluding diagonal
        row_means = np.zeros(n_regions)                     # Initialize array to hold row means
        for r in range(n_regions):                          # Iterate over each row
            row_vals = matrix[r][off_diag_mask[r]]          # Extract off-diagonal values
            row_means[r] = np.mean(row_vals) if len(row_vals) > 0 else 0.0

        np.fill_diagonal(result[s], row_means)

    return result

def impute_diagonal_network_mean(
        matrices: np.ndarray,
        region_list: List[str]
) -> np.ndarray:
    """ 
    Impute diagonal with network-wise mean connectivity (Deterministic, subject-specific). 
    For each region, use mean of connections to same-network regions.
    Fall back to row mean if no same-network connections exist.
    """

    result = matrices.copy()
    n_subjects = result.shape[0]
    n_regions = result.shape[1]

    # Parse networks
    network_map = parse_networks(region_list)
    
    # Build same-netowrk indices 
    same_network_indices = []
    for r in range(n_regions):  # 
        r_name = region_list[r]
        r_network = network_map.get(r_name, 'unknown') # Get network of region r 

        idx = [
            j for j in range(n_regions)
            if j != r and network_map.get(region_list[j], 'unknown') == r_network
        ]                                                       # Indices of regions in the same network excluding self
        same_network_indices.append(np.array(idx, dtype=int))  

    # create mask for off-diagonal elements
    off_diag_mask = ~np.eye(n_regions, dtype=bool)

    # Impute for each subject
    for s in range(n_subjects):
        matrix = result[s]
        diag_vals = np.zeros(n_regions)

        for r in range(n_regions):
            # Try network mean first
            if len(same_network_indices[r]) > 0:
                net_vals = matrix[r][same_network_indices[r]]   # Values connecting to same-network regions
                network_mean = np.nanmean(net_vals)             # Compute network mean

                if not np.isnan(network_mean):                  # Valid mean found  
                    diag_vals[r] = network_mean                 # Assign network mean to diagonal
                    continue

            # Fall back to row mean
            row_vals = matrix[r][off_diag_mask[r]]
            diag_vals[r] = np.nanmean(row_vals) if len(row_vals) > 0 else 0.0

        np.fill_diagonal(result[s], diag_vals)
    return result

def impute_diagonal_sample_from_row(
        matrices: np.ndarray
):
    """
    Impute diagonal by sampling from each row (Stochastic - TRUE randomness per call).
    """
    result = matrices.copy()
    n_subjects = result.shape[0]
    n_regions = result.shape[1]

    # create mask for off-diagonal elements
    off_diag_mask = ~np.eye(n_regions, dtype=bool)

    for s in range(n_subjects):
        matrix = result[s]

        for r in range(n_regions):
           # Get off-diagonal values for this row
           candidates = matrix[r][off_diag_mask[r]]        # Extract off-diagonal values
           candidates = candidates[~np.isnan(candidates)]  # Remove NaNs

           # Sample one value (NEW random sample every call, no fixed seed)
           if len(candidates) > 0:
               val = np.random.choice(candidates)
           else:
               val = 0.0  
           
           matrix[r, r] = val

    return result

def impute_diagonal_sample_from_matrix(
        matrices: np.ndarray
):
    """
    Impute diagonal by sampling from entire matrix (Stochastic - TRUE randomness per call).
    """
    result = matrices.copy()
    n_subjects = result.shape[0]
    n_regions = result.shape[1]

    # create mask for off-diagonal elements
    off_diag_mask = ~np.eye(n_regions, dtype=bool)

    for s in range(n_subjects):
        matrix = result[s]

        # Get all off-diagonal values
        candidates = matrix[off_diag_mask]               # Extract off-diagonal values
        candidates = candidates[~np.isnan(candidates)]   # Remove NaNs

        for r in range(n_regions):
            # Sample one value (NEW random sample every call, no fixed seed)
            if len(candidates) > 0:
                val = np.random.choice(candidates)
            else:
                val = 0.0  
           
            matrix[r, r] = val

    return result   

def impute_diagonal(
        matrices: np.ndarray,
        Strategy: str,
        region_list: Optional[List[str]] = None
) -> np.ndarray:
    """
    Impute diagonal values in connectivity matrices using specified method.

    Parameters:
    matrices : 3D array (n_subjects x n_regions x n_regions).
    region_list : List of region names corresponding to matrix indices.
    method : str
        Imputation method. Options:
        -- Deterministic --
        - 'zero': Fill diagonal with zeros.
        - 'region_mean': Fill diagonal with region-wise mean connectivity.
        - 'network_mean': Fill diagonal with network-wise mean connectivity.

        -- Stochastic (TRUE randomness - different values per call) --
        - 'random': Fill diagonal with random values.
        - 'sample_row': Sample diagonal values from each row.
        - 'sample_matrix': Sample diagonal values from entire matrix.

    Returns: imputed_matrices : Matrices with imputed diagonal values.
    """
    # Deterministic Methods
    if Strategy == 'zero':
        return impute_diagonal_zero(matrices)
    elif Strategy == 'region_mean':
        return impute_diagonal_region_mean(matrices)
    elif Strategy == 'network_mean':
        if region_list is None:
            raise ValueError("region_list must be provided for network_mean imputation.")
        return impute_diagonal_network_mean(matrices, region_list)
    
    # Stochastic Methods (TRUE randomness - no fixed seed)
    elif Strategy == 'random':
        return impute_diagonal_random(matrices)
    elif Strategy == 'sample_row':
        return impute_diagonal_sample_from_row(matrices)
    elif Strategy == 'sample_matrix':
        return impute_diagonal_sample_from_matrix(matrices)
    else:
        raise ValueError(f"Unknown imputation method: {Strategy}")
    

def extract_features_for_classification(
        matrices : np.ndarray,
        include_diagonal: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     Extract per-region connectivity patterns for classification.
    
    Args:
        matrices: 3D array (n_subjects, n_regions, n_regions).
        include_diagonal (bool): Whether to include diagonal elements in the features.

    Returns:
        X_features : 2D array of shape (n_subjects*n_regions, n_features).
        labels: Region labels for each sample.
        subjects: Subject indices for each sample.
    """
    # Determine dimensions
    n_subjects = matrices.shape[0]
    n_regions = matrices.shape[1]
    n_samples = n_subjects * n_regions
    n_features = n_regions if include_diagonal else n_regions - 1

    # Initialize arrays
    X_features = np.zeros((n_samples, n_features), dtype=float)
    labels = np.zeros(n_samples, dtype=int)
    subjects = np.zeros(n_samples, dtype=int)
    
    sample_idx = 0
    # Iterate over subjects and regions
    for subj in range(n_subjects):
        # Iterate over regions
        for region_idx in range(n_regions):
            # Get connectivity vector for this region
            conn_vector = matrices[subj, region_idx, :]

            # Extract features (exclude diagonal unless specified)
            if include_diagonal:
                features = conn_vector
            else:
                features = np.delete(conn_vector, region_idx)  # Exclude diagonal element
                
            X_features[sample_idx, :] = features
            labels[sample_idx] = region_idx
            subjects[sample_idx] = subj
            
            sample_idx += 1

    return X_features, labels, subjects 

# ==============================
# Main Preprocessor Class
# ==============================

class BrainConnectivityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for brain connectivity data.
    
    Steps:
    1. Reconstruct connectivity matrices from flattened DataFrame.
    2. Impute diagonal values using specified strategy.
    3. Extract per-region connectivity patterns for classification.
    Parameters:
    ----------
    connection_columns : List[str]
        List of connection column names in the format 'RegionA~RegionB'.
    diagonal_imputation_strategy : str
        Strategy for diagonal imputation. Options:
        - 'zero'
        - 'region_mean'
        - 'network_mean'
        - 'random'
        - 'sample_row'
        - 'sample_matrix'
    include_diagonal_in_features : bool
        Whether to include diagonal elements in the extracted features.
    Returns:
    -------
    X_features : np.ndarray
        2D array of shape (n_subjects*n_regions, n_features).
    labels : np.ndarray
        Region labels for each sample.
    subjects : np.ndarray
        Subject indices for each sample.
    """

    def __init__(
        self,
        connection_columns: Optional[List[str]] = None,
        diagonal_strategy: str = 'zero',
        region_list: Optional[List[str]] = None,
        include_diagonal: bool = False,
        apply_fisher_z: bool = True,
        random_state: int = 42,
        enable_diagnostics: bool = False
    ):
        """
        Args:
            connection_columns (List[str]): List of connection column names.
            diagonal_strategy (str): Diagonal imputation strategy.
            region_list (List[str]): List of region names.
            include_diagonal (bool): Include diagonal in features.
            apply_fisher_z (bool): Apply Fisher Z-transformation.
            random_state (int): Random seed (ONLY for deterministic strategies and reproducibility testing).
                                NOTE: Stochastic strategies (random, sample_row, sample_matrix) 
                                      IGNORE this parameter to ensure true randomness.
            enable_diagnostics (bool): Enable diagnostic logging.
        """
        # Store parameters
        self.connection_columns = connection_columns
        self.diagonal_strategy = diagonal_strategy
        self.region_list = region_list
        self.include_diagonal = include_diagonal
        self.apply_fisher_z = apply_fisher_z
        self.random_state = random_state
        self.enable_diagnostics = enable_diagnostics

        # will be set during fit
        self.region_list_ = None
        self.region_to_idx_ = None
        self.n_regions_ = None

        # Diagnostic tracking
        self._transform_count = 0

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit the preprocessor to the data.

        Fit just extracts region information from the connection columns - no statistics are learned.

        Args:
            X (pd.DataFrame): Input DataFrame with flattened connectivity data.
            y (np.ndarray, optional): Target labels (not used).
        """

        # Extract connection columns if not provided
        if self.connection_columns is None:
            if isinstance(X, pd.DataFrame):
                self.connection_columns = [col for col in X.columns if '~' in str(col)]
            else:
                raise ValueError("connection_columns must be provided for a non dataframe input.")
        
        if len(self.connection_columns) == 0:
            raise ValueError("No connection columns found in the input data.")
        
        # Extract region information
        self.region_list_, self.region_to_idx_, self.n_regions_ = extract_regions(self.connection_columns)
        
        # Reset diagnostic counter
        self._transform_count = 0
        
        return self
    
    def transform(self, X: pd.DataFrame):
        """
        Transform the data using the preprocessing pipeline.

        Args:
            X (pd.DataFrame): Input DataFrame with flattened connectivity data.
        Returns:
        -------
        X_features : np.ndarray
            2D array of shape (n_subjects*n_regions, n_features).
        
        Pipeline Steps:
        ---------------
        1. Reconstruct connectivity matrices from flattened DataFrame.
        2. Impute diagonal values using specified strategy.
           NOTE: For stochastic strategies (random, sample_row, sample_matrix),
                 NEW random values are generated on EVERY transform() call.
        3. Extract per-region connectivity patterns for classification.
        4. Apply Fisher Z-transformation to connectivity values.
        """
        self._transform_count += 1
        
        if self.enable_diagnostics:
            print(f"\n[Preprocessor Transform call count: {self._transform_count}]")
            if self.diagonal_strategy in ['random', 'sample_row', 'sample_matrix']:
                print(f"  -> Using STOCHASTIC imputation: '{self.diagonal_strategy}'")
                print(f"     (Diagonal values will be DIFFERENT on each transform call)")
        
        # Step 1: Reconstruct connectivity matrices
        matrices = reconstruct_matrices_from_dataframe(
            X,
            self.connection_columns,
            self.region_to_idx_,
            self.n_regions_
        )

        if self.enable_diagnostics:
            print(f"  -> Reconstructed {matrices.shape[0]} connectivity matrices of size {matrices.shape[1]}x{matrices.shape[2]}.")
        
        # Step 2: Impute diagonal values
        # CRITICAL: For stochastic strategies, this generates NEW random values every time
        matrices = impute_diagonal(
            matrices,
            self.diagonal_strategy,
            self.region_list_
        )
        if self.enable_diagnostics:
            print(f"  -> Applied diagonal imputation using strategy: '{self.diagonal_strategy}'.")
            print(f"    Subject 0 diagonal sample after imputation: {matrices[0].diagonal()[:5]} ...")

        # Step 3: Extract features or per-region connectivity patterns
        X_features, self.labels_, self.subjects_ = extract_features_for_classification(
            matrices,
            self.include_diagonal
        )

        if self.enable_diagnostics:
            print(f"  -> Extracted features for classification: {X_features.shape[0]} samples with {X_features.shape[1]} features each.")            

        # Step 4: Apply Fisher Z-transformation
        if self.apply_fisher_z:
            eps = 1e-6
            X_clipped = np.clip(X_features, -1 + eps, 1 - eps)
            X_features = np.arctanh(X_clipped)
            if self.enable_diagnostics:
                print(f"  -> Applied Fisher Z-transformation to connectivity values.")

        return X_features

    def get_labels(self) -> np.ndarray:
        """ Get region labels for each sample after transform. """
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Must call transform() before get_labels()")
        return self.labels_

    def get_subjects(self) -> np.ndarray:
        """ Get subject indices for each sample after transform. """
        if not hasattr(self, 'subjects_'):
            raise RuntimeError("Must call transform() before get_subjects()")
        return self.subjects_