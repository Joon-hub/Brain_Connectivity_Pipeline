"""
Feature Engineering: Connectivity Matrix Preprocessing 
======================================================
All preprocessing is now fold-aware and compatible with sklearn Pipeline.
No statistics are computed globally - everything happens within fit/transform.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
import warnings
import pickle
from pathlib import Path


# ======================
# UTILITY FUNCTIONS 
# ======================

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
    
    # Extract unique regions
    for col in connection_columns:
        region_a, region_b = col.split('~') 
        
        for region in [region_a, region_b]: # Add both regions
            if region not in seen:
                seen.add(region)
                unique_regions.append(region)
    
    region_to_idx = {region: idx for idx, region in enumerate(unique_regions)} # Create mapping from name to index
    
    return unique_regions, region_to_idx, len(unique_regions)


def reconstruct_connectivity_matrix(
    subject_values: np.ndarray,       # [0.4, 0.3, 0.2]
    connection_columns: List[str],    # ['A~B', 'B~C', 'C~D']
    region_to_idx: Dict[str, int],    # {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    n_regions: int                    # 232
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
    # Initialize connectivity matrix with zeros (n_regions x n_regions) 
    matrix = np.zeros((n_regions, n_regions), dtype=float)
    
    # Fill off-diagonal elements
    for col, value in zip(connection_columns, subject_values): # col = 'A~B', value = 0.4
        region_a, region_b = col.split('~')
        idx_a = region_to_idx[region_a]             # idx_a = 0
        idx_b = region_to_idx[region_b]             # idx_b = 1
        
        matrix[idx_a, idx_b] = value                # matrix[0, 1] = 0.4
        matrix[idx_b, idx_a] = value                # matrix[1, 0] = 0.4    
    
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
        network = 'Unknown'             # Default to 'Unknown'
        
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
        
        network_map[region] = network   # {region: network}
    
    return network_map    #  e.g. {'LH_VisCent': 'VisCent'}


# ============================================================================
# SKLEARN-COMPATIBLE TRANSFORMERS 
# ============================================================================

class ConnectivityMatrixReconstructor(BaseEstimator, TransformerMixin):
    """
    Reconstruct connectivity matrices from flattened DataFrame.
    """
    
    def __init__(self, connection_columns: Optional[List[str]] = None):
        self.connection_columns = connection_columns
    
    def fit(self, X, y=None):                  # metadata extraction
        """Extract region information from columns."""
        if self.connection_columns is None:
            if isinstance(X, pd.DataFrame):
                self.connection_columns = [col for col in X.columns if '~' in str(col)] # e.g. ['LH_VisCent~RH_VisCent']
            else:
                raise ValueError("connection_columns must be provided for non-DataFrame input")
        
        # Extract regions
        self.region_list_, self.region_to_idx_, self.n_regions_ = extract_regions(self.connection_columns)           
        # e.g. ['LH_VisCent', 'RH_VisCent'], {'LH_VisCent': 0, 'RH_VisCent': 1}, 2 
        return self
    
    def transform(self, X):                       # reconstruction of matrix for every subject
        """Reconstruct connectivity matrices."""
        if isinstance(X, pd.DataFrame):
            X_values = X[self.connection_columns].values

            # e.g.
            # X_values = X['A~B','B~C','C~D'].values
            # X_values = [
            # [0.40, 0.70, 0.55],  # subject S001
            # [0.10, 0.20, 0.30]   # subject S002
            # ]
        
        else:
            X_values = X
        
        n_subjects = X_values.shape[0]                         # e.g. 2
        matrices = np.zeros((n_subjects, self.n_regions_, self.n_regions_))
        
        for i in range(n_subjects):
            matrices[i] = reconstruct_connectivity_matrix(
                X_values[i],
                self.connection_columns,
                self.region_to_idx_,
                self.n_regions_
            )
            
            # e.g.
            # matrices[subject] = [
            # [0.40, 0.70, 0.55],  # subject S001
            # [0.10, 0.20, 0.30]   # subject S002
            # ]
        
        return matrices


class FoldAwareDiagonalImputer(BaseEstimator, TransformerMixin):
    """
    Impute the diagonal of connectivity matrices in a fold-aware.

    Strategies:
      - "zero":                fill 0 on the diagonal (deterministic)
      - "random":              fill U(-1,1) per subject/region (deterministic via random_state)
      - "region_mean":         SUBJECT-SPECIFIC row mean (exclude diagonal)
      - "network_mean":        SUBJECT-SPECIFIC mean over same-network targets; fallback to row mean
      - "sample_from_matrix":  SUBJECT-SPECIFIC sample from that subject's off-diagonal entries
    """
    
    def __init__(
        self, 
        strategy: str = "zero",
        region_list: Optional[List[str]] = None,
        k_neighbors: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            strategy: 'zero', 'region_mean', 'network_mean', 'random', 'sample_from_matrix'
            region_list: Required for 'network_mean'
            k_neighbors: For future KNN strategy
            random_state: For 'random' strategy
        """
        self.strategy = strategy
        self.region_list = region_list
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.same_net_idx_: Optional[List[np.ndarray]] = None
        self.offdiag_mask_: Optional[np.ndarray] = None
        self.offdiag_den_: Optional[np.ndarray] = None
        
        # Statistics computed during fit (ONLY on training data)
        self.statistics_ = None
        self.network_map_ = None
        self.n_regions_ = None
    def _build_same_net_indices(self):
        """Precompute index lists of same-network partners for each region (exclude self)."""
        assert self.region_list is not None and self.network_map_ is not None
        R = self.n_regions_
        same_net_idx = []
        for r in range(R):
            r_name = self.region_list[r]
            r_net = self.network_map_.get(r_name, "Unknown")
            idx = [
                j for j in range(R)
                if j != r and self.network_map_.get(self.region_list[j], "Unknown") == r_net
            ]
            same_net_idx.append(np.asarray(idx, dtype=int))
        self.same_net_idx_ = same_net_idx
    def fit(self, X, y=None):
        """
        X: array of shape (n_subjects, n_regions, n_regions)
        Only metadata is prepared here; no cross-subject statistics are computed.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        _, R, C = X.shape
        if R != C:
            raise ValueError("Connectivity matrices must be square.")
        self.n_regions_ = R

        # Prepare a mask that excludes the diagonal (useful for fast row means).
        offdiag_mask = np.ones((R, R), dtype=bool)
        np.fill_diagonal(offdiag_mask, False)
        self.offdiag_mask_ = offdiag_mask
        self.offdiag_den_ = offdiag_mask.sum(axis=1).astype(float)  # == R-1

        # For subject-specific region/network means we don't keep global stats.
        self.statistics_ = None

        # For subject-specific network_mean we only need network membership & same-net indices.
        if self.strategy == "network_mean":
            if self.region_list is None:
                raise ValueError("region_list is required for 'network_mean' strategy.")
            self.network_map_ = parse_networks(self.region_list)
            self._build_same_net_indices()

        return self

    def transform(self, X):
        """
        Apply the selected strategy per subject.
        Returns X_imputed with diagonals replaced.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        if X.shape[1] != self.n_regions_ or X.shape[2] != self.n_regions_:
            raise ValueError("Input shape does not match fit() dimensions.")

        n_subjects = X.shape[0]
        R = self.n_regions_
        X_imputed = X.copy()

        if self.strategy == "zero":
            for s in range(n_subjects):
                np.fill_diagonal(X_imputed[s], 0.0)
            return X_imputed

        if self.strategy == "random":
            rng = np.random.RandomState(self.random_state)
            for s in range(n_subjects):
                np.fill_diagonal(X_imputed[s], rng.uniform(-1, 1, R))
            return X_imputed

        if self.strategy == "sample_from_matrix":
            rng = np.random.RandomState(self.random_state)
            # one global off-diagonal mask works for every subject
            mask = self.offdiag_mask_
            for s in range(n_subjects):
                M = X_imputed[s]
                off_diag_vals = M[mask]
                for r in range(R):
                    M[r, r] = rng.choice(off_diag_vals)
            return X_imputed

        if self.strategy == "region_mean":
            # SUBJECT-SPECIFIC row means (exclude diagonal)
            mask = self.offdiag_mask_
            den = self.offdiag_den_
            for s in range(n_subjects):
                M = X_imputed[s]
                # use nan-safe ops; if your data has no NaNs, np.sum/np.mean is fine too
                row_sums = np.nansum(M * mask, axis=1)
                row_means = row_sums / den
                # Optional: guard against rows that are all-NaN
                row_means = np.nan_to_num(row_means, nan=0.0)
                np.fill_diagonal(M, row_means)
            return X_imputed

        if self.strategy == "network_mean":
            if self.same_net_idx_ is None:
                raise RuntimeError("fit() must be called before transform() for 'network_mean'.")

            mask = self.offdiag_mask_
            den = self.offdiag_den_

            for s in range(n_subjects):
                M = X_imputed[s]
                # Precompute per-subject row means for fallback
                row_sums = np.nansum(M * mask, axis=1)
                row_means = row_sums / den
                row_means = np.nan_to_num(row_means, nan=0.0)

                diag_vals = np.empty(R, dtype=float)
                for r in range(R):
                    idx = self.same_net_idx_[r]
                    if idx.size > 0:
                        v = np.nanmean(M[r, idx])
                        # Fallback if same-network entries are all NaN
                        if np.isnan(v):
                            v = row_means[r]
                    else:
                        v = row_means[r]
                    diag_vals[r] = v

                np.fill_diagonal(M, diag_vals)

            return X_imputed

        raise ValueError(f"Unknown strategy: {self.strategy}")


class RegionConnectivityExtractor(BaseEstimator, TransformerMixin):
    """
    Extract per-region connectivity patterns for classification.
    """
    
    def __init__(self, include_diagonal: bool = False):
        self.include_diagonal = include_diagonal
    
    def fit(self, X, y=None):
        """Store region count."""
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        
        self.n_regions_ = X.shape[1]
        return self
    
    def transform(self, X):
        """
        Extract connectivity patterns.
        
        Args:
            X: 3D array (n_subjects × n_regions × n_regions)
        
        Returns:
            X_features: 2D array (n_subjects*n_regions × n_features)
            Also stores labels and subjects as attributes
        """
        n_subjects, n_regions, _ = X.shape
        n_samples = n_subjects * n_regions
        n_features = n_regions if self.include_diagonal else n_regions - 1
        
        X_features = np.zeros((n_samples, n_features), dtype=float)
        self.labels_ = np.zeros(n_samples, dtype=int)
        self.subjects_ = np.zeros(n_samples, dtype=int)
        
        sample_idx = 0
        for subj_idx in range(n_subjects):
            for region_idx in range(n_regions):
                row = X[subj_idx, region_idx, :]
                
                if self.include_diagonal:
                    features = row
                else:
                    features = np.delete(row, region_idx)
                
                X_features[sample_idx] = features
                self.labels_[sample_idx] = region_idx
                self.subjects_[sample_idx] = subj_idx
                
                sample_idx += 1
        
        return X_features
    
    def get_labels(self):
        """Get region labels (must call after transform)."""
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Must call transform() before get_labels()")
        return self.labels_
    
    def get_subjects(self):
        """Get subject indices (must call after transform)."""
        if not hasattr(self, 'subjects_'):
            raise RuntimeError("Must call transform() before get_subjects()")
        return self.subjects_


class BrainConnectivityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline.
    
    All statistics are computed during fit() using only training data,
    then applied during transform() to any dataset.
    """
    
    def __init__(
        self,
        connection_columns: Optional[List[str]] = None,
        diagonal_strategy: str = "zero",
        region_list: Optional[List[str]] = None,
        include_diagonal: bool = False,
        random_state: int = 42
    ):
        self.connection_columns = connection_columns
        self.diagonal_strategy = diagonal_strategy
        self.region_list = region_list
        self.include_diagonal = include_diagonal
        self.random_state = random_state
        
        # Component transformers (will be created in fit)
        self.reconstructor_ = None
        self.imputer_ = None
        self.extractor_ = None
    
    def fit(self, X, y=None):
        """
        Fit all preprocessing components on training data.
        
        Args:
            X: DataFrame with connectivity data (training fold only)
        """
        # Step 1: Reconstruct matrices
        self.reconstructor_ = ConnectivityMatrixReconstructor(self.connection_columns)
        X_matrices = self.reconstructor_.fit_transform(X)
        
        # Store region info for convenience
        self.region_list_ = self.reconstructor_.region_list_
        self.n_regions_ = self.reconstructor_.n_regions_
        
        # Step 2: Fit imputer on training matrices
        self.imputer_ = FoldAwareDiagonalImputer(
            strategy=self.diagonal_strategy,
            region_list=self.region_list_,
            random_state=self.random_state
        )
        X_imputed = self.imputer_.fit_transform(X_matrices)
        
        # Step 3: Extractor (no fitting needed, but we initialize)
        self.extractor_ = RegionConnectivityExtractor(
            include_diagonal=self.include_diagonal
        )
        self.extractor_.fit(X_imputed)
        
        return self
    
    def transform(self, X):
        """
        Apply preprocessing using statistics from training data.
        
        Args:
            X: DataFrame with connectivity data (can be train or test)
        
        Returns:
            X_features: 2D feature array
        """
        X_matrices = self.reconstructor_.transform(X)
        X_imputed = self.imputer_.transform(X_matrices)
        X_features = self.extractor_.transform(X_imputed)
        
        return X_features
    
    def get_labels(self):
        """Get region labels for samples (after transform)."""
        return self.extractor_.get_labels()
    
    def get_subjects(self):
        """Get subject indices for samples (after transform)."""
        return self.extractor_.get_subjects()
