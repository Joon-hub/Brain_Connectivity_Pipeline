"""
connectivity_preprocessor.py

Preprocessing pipeline for functional connectivity data.
Handles Fisher Z-transformation, diagonal imputation, and standardization
with support for cross-validation (fit on train, transform on test).

Author: Joon
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class ConnectivityPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocess functional connectivity data for classification.
    
    Pipeline:
    1. Fisher Z-transformation (optional)
    2. Diagonal imputation (various strategies)
    3. Standardization (optional)
    
    Designed for use in cross-validation with proper fit/transform separation
    to prevent data leakage.
    
    Parameters
    ----------
    diagonal_strategy : str, default='region_mean'
        Strategy for handling diagonal values:
        - 'zero': Set diagonal to 0
        - 'region_mean': Set diagonal to mean of off-diagonal values for that region
        - 'network_mean': Set diagonal to mean within functional network
        - 'global_mean': Set diagonal to global mean of all off-diagonal values
        - 'keep': Keep existing diagonal values (usually 1.0)
    apply_fisher_z : bool, default=True
        Apply Fisher Z-transformation to connectivity values
    standardize : bool, default=True
        Apply standardization (z-score) after other preprocessing
    clip_values : bool, default=True
        Clip extreme values after Fisher Z (prevents inf from r=±1)
    clip_range : tuple, default=(-5, 5)
        Range for clipping after Fisher Z-transformation
    
    Attributes
    ----------
    diagonal_values_ : np.ndarray
        Computed diagonal values from training data
    scaler_ : StandardScaler
        Fitted scaler (if standardize=True)
    n_features_ : int
        Number of features after preprocessing
    
    Examples
    --------
    >>> from sklearn.model_selection import GroupKFold
    >>> preprocessor = ConnectivityPreprocessor(diagonal_strategy='region_mean')
    >>> 
    >>> for train_idx, test_idx in gkf.split(X, y, groups):
    >>>     X_train, X_test = X[train_idx], X[test_idx]
    >>>     
    >>>     # Fit on training data only
    >>>     preprocessor.fit(X_train)
    >>>     
    >>>     # Transform both
    >>>     X_train_processed = preprocessor.transform(X_train)
    >>>     X_test_processed = preprocessor.transform(X_test)
    """
    
    def __init__(
        self,
        diagonal_strategy: str = 'region_mean',
        apply_fisher_z: bool = True,
        standardize: bool = True,
        clip_values: bool = True,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
        region_info: Optional[pd.DataFrame] = None
    ):
        self.diagonal_strategy = diagonal_strategy
        self.apply_fisher_z = apply_fisher_z
        self.standardize = standardize
        self.clip_values = clip_values
        self.clip_range = clip_range
        self.region_info = region_info
        
        # Validate parameters
        valid_strategies = ['zero', 'region_mean', 'network_mean', 'global_mean', 'keep']
        if diagonal_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid diagonal_strategy: {diagonal_strategy}. "
                f"Must be one of {valid_strategies}"
            )
        
        if diagonal_strategy == 'network_mean' and region_info is None:
            raise ValueError(
                "region_info must be provided when using diagonal_strategy='network_mean'"
            )
        
        # Fitted attributes (set during fit())
        self.diagonal_values_ = None
        self.scaler_ = None
        self.n_features_ = None
        self.n_regions_ = None
    
    def fit(self, X: np.ndarray, y=None, region_info: Optional[pd.DataFrame] = None):
        """
        Fit preprocessor on training data.
        
        Computes statistics needed for preprocessing (diagonal values, 
        standardization parameters) based on training data only.
        
        Parameters
        ----------
        X : np.ndarray
            Connectivity data, shape (n_samples, n_regions) where each row
            is a flattened connectivity vector for one region
        y : array-like, optional
            Target labels (not used, for sklearn compatibility)
        region_info : pd.DataFrame, optional
            Region information with network assignments
            Required if diagonal_strategy='network_mean'
        
        Returns
        -------
        self : ConnectivityPreprocessor
            Fitted preprocessor
        """
        
        # Update region_info if provided
        if region_info is not None:
            self.region_info = region_info
        
        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.n_regions_ = n_features
        
        logger.info(f"Fitting preprocessor on {n_samples} samples, {n_features} features")
        logger.info(f"  Diagonal strategy: {self.diagonal_strategy}")
        logger.info(f"  Fisher Z: {self.apply_fisher_z}")
        logger.info(f"  Standardize: {self.standardize}")
        
        # Make a copy for computing statistics (don't modify original)
        X_work = X.copy()
        
        # Step 1: Apply Fisher Z if requested
        if self.apply_fisher_z:
            X_work = self._apply_fisher_z_transform(X_work)
        
        # Step 2: Compute diagonal imputation values
        self.diagonal_values_ = self._compute_diagonal_values(X_work)
        
        # Step 3: Apply diagonal imputation
        X_work = self._impute_diagonal(X_work, self.diagonal_values_)
        
        # Step 4: Fit standardization if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_work)
            logger.info(f"  Fitted StandardScaler: mean={self.scaler_.mean_[:5]}, std={self.scaler_.scale_[:5]}")
        
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform connectivity data using fitted parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Connectivity data, shape (n_samples, n_regions)
        
        Returns
        -------
        X_transformed : np.ndarray
            Preprocessed connectivity data, same shape as input
        """
        
        # Check if fitted
        if self.diagonal_values_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_}, "
                f"got {X.shape[1]}"
            )
        
        # Make a copy to avoid modifying original
        X_transformed = X.copy()
        
        # Step 1: Apply Fisher Z if requested
        if self.apply_fisher_z:
            X_transformed = self._apply_fisher_z_transform(X_transformed)
        
        # Step 2: Apply diagonal imputation using fitted values
        X_transformed = self._impute_diagonal(X_transformed, self.diagonal_values_)
        
        # Step 3: Apply standardization if fitted
        if self.standardize and self.scaler_ is not None:
            X_transformed = self.scaler_.transform(X_transformed)
        
        # Final validation
        if np.any(np.isnan(X_transformed)):
            n_nan = np.sum(np.isnan(X_transformed))
            logger.warning(f"Transform produced {n_nan} NaN values")
        
        if np.any(np.isinf(X_transformed)):
            n_inf = np.sum(np.isinf(X_transformed))
            logger.warning(f"Transform produced {n_inf} Inf values")
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> np.ndarray:
        """
        Fit preprocessor and transform in one step.
        
        Parameters
        ----------
        X : np.ndarray
            Connectivity data
        y : array-like, optional
            Target labels
        **fit_params : dict
            Additional parameters for fit()
        
        Returns
        -------
        X_transformed : np.ndarray
            Preprocessed connectivity data
        """
        return self.fit(X, y, **fit_params).transform(X)
    
    def _apply_fisher_z_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Fisher Z-transformation: z = 0.5 * ln((1+r)/(1-r))
        
        Also known as arctanh transformation.
        Stabilizes variance and makes distribution more normal.
        """
        
        # Clip to valid range for correlation: (-1, 1)
        # Avoid exact ±1 which would give ±inf
        X_clipped = np.clip(X, -0.9999, 0.9999)
        
        # Apply Fisher Z
        X_transformed = np.arctanh(X_clipped)
        
        # Clip extreme values if requested
        if self.clip_values:
            X_transformed = np.clip(
                X_transformed,
                self.clip_range[0],
                self.clip_range[1]
            )
        
        return X_transformed
    
    def _compute_diagonal_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute diagonal imputation values based on training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training connectivity data (after Fisher Z if applicable)
        
        Returns
        -------
        diagonal_values : np.ndarray
            Values to use for diagonal imputation, shape (n_regions,)
        """
        
        n_regions = self.n_regions_
        
        if self.diagonal_strategy == 'zero':
            # Set all diagonal to 0
            diagonal_values = np.zeros(n_regions)
        
        elif self.diagonal_strategy == 'region_mean':
            # Each diagonal = mean of that region's off-diagonal values
            diagonal_values = np.mean(X, axis=0)
        
        elif self.diagonal_strategy == 'global_mean':
            # All diagonals = global mean of all values
            global_mean = np.mean(X)
            diagonal_values = np.full(n_regions, global_mean)
        
        elif self.diagonal_strategy == 'network_mean':
            # Each diagonal = mean within its functional network
            if self.region_info is None:
                raise ValueError(
                    "region_info required for network_mean diagonal strategy"
                )
            
            diagonal_values = self._compute_network_mean_diagonal(X)
        
        elif self.diagonal_strategy == 'keep':
            # Keep existing diagonal values (usually 1.0)
            # In transformed space, this is Fisher Z of 1.0 if Fisher Z was applied
            if self.apply_fisher_z:
                # Fisher Z of 1.0 is undefined, use high value
                diagonal_values = np.full(n_regions, self.clip_range[1])
            else:
                diagonal_values = np.ones(n_regions)
        
        else:
            raise ValueError(f"Unknown diagonal strategy: {self.diagonal_strategy}")
        
        logger.info(
            f"Computed diagonal values: "
            f"min={np.min(diagonal_values):.4f}, "
            f"max={np.max(diagonal_values):.4f}, "
            f"mean={np.mean(diagonal_values):.4f}"
        )
        
        return diagonal_values
    
    def _compute_network_mean_diagonal(self, X: np.ndarray) -> np.ndarray:
        """
        Compute diagonal values as mean within each functional network.
        
        Parameters
        ----------
        X : np.ndarray
            Training connectivity data
        
        Returns
        -------
        diagonal_values : np.ndarray
            Network-specific diagonal values
        """
        
        n_regions = self.n_regions_
        diagonal_values = np.zeros(n_regions)
        
        # Get network assignments
        if 'network' not in self.region_info.columns:
            logger.warning(
                "No 'network' column in region_info, falling back to region_mean"
            )
            return np.mean(X, axis=0)
        
        networks = self.region_info['network'].values
        
        # Compute mean for each network
        for network in np.unique(networks):
            # Get regions in this network
            network_mask = networks == network
            network_indices = np.where(network_mask)[0]
            
            # Compute mean connectivity within this network
            network_values = X[:, network_indices]
            network_mean = np.mean(network_values)
            
            # Assign to all regions in this network
            diagonal_values[network_indices] = network_mean
            
            logger.debug(
                f"Network {network}: {len(network_indices)} regions, "
                f"mean={network_mean:.4f}"
            )
        
        return diagonal_values
    
    def _impute_diagonal(
        self, 
        X: np.ndarray, 
        diagonal_values: np.ndarray
    ) -> np.ndarray:
        """
        Replace diagonal values with imputed values.
        
        Note: In our data format, each row is a connectivity vector for one region.
        The "diagonal" refers to the self-connection, which is typically at a 
        specific position depending on the region index.
        
        For now, this implementation assumes X represents flattened connectivity
        where we want to replace certain positions with diagonal_values.
        
        Parameters
        ----------
        X : np.ndarray
            Connectivity data, shape (n_samples, n_regions)
        diagonal_values : np.ndarray
            Values to impute, shape (n_regions,)
        
        Returns
        -------
        X_imputed : np.ndarray
            Data with diagonal imputed
        """
        
        # In your classification setup:
        # - Each sample is a connectivity vector for one region
        # - Shape: (n_subjects * n_regions, n_regions)
        # - The "diagonal" is the position corresponding to self-connection
        
        # Create a copy
        X_imputed = X.copy()
        
        # Strategy: For each row (representing region i's connectivity),
        # replace the value at position i with diagonal_values[i]
        
        # Determine which region each row corresponds to
        n_samples, n_features = X.shape
        n_regions = len(diagonal_values)
        
        if n_features != n_regions:
            logger.warning(
                f"Feature dimension ({n_features}) doesn't match "
                f"diagonal_values length ({n_regions}). "
                f"Skipping diagonal imputation."
            )
            return X_imputed
        
        # Each block of n_regions rows corresponds to one subject
        # Within each block, row i corresponds to region i
        for sample_idx in range(n_samples):
            region_idx = sample_idx % n_regions
            X_imputed[sample_idx, region_idx] = diagonal_values[region_idx]
        
        return X_imputed
    
    def get_params(self, deep: bool = True) -> Dict:
        """
        Get parameters for this estimator (sklearn compatibility).
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of sub-objects
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        return {
            'diagonal_strategy': self.diagonal_strategy,
            'apply_fisher_z': self.apply_fisher_z,
            'standardize': self.standardize,
            'clip_values': self.clip_values,
            'clip_range': self.clip_range,
            'region_info': self.region_info
        }
    
    def set_params(self, **params) -> 'ConnectivityPreprocessor':
        """
        Set parameters for this estimator (sklearn compatibility).
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
        
        Returns
        -------
        self : ConnectivityPreprocessor
            Estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ConnectivityMatrixPreprocessor:
    """
    Preprocessor for full connectivity matrices (n_subjects, n_regions, n_regions).
    
    This is a helper class for preprocessing before reshaping into classification format.
    
    Parameters
    ----------
    apply_fisher_z : bool, default=True
        Apply Fisher Z-transformation
    clip_values : bool, default=True
        Clip extreme values
    clip_range : tuple, default=(-5, 5)
        Range for clipping
    """
    
    def __init__(
        self,
        apply_fisher_z: bool = True,
        clip_values: bool = True,
        clip_range: Tuple[float, float] = (-5.0, 5.0)
    ):
        self.apply_fisher_z = apply_fisher_z
        self.clip_values = clip_values
        self.clip_range = clip_range
    
    def transform(self, connectivity: np.ndarray) -> np.ndarray:
        """
        Transform connectivity matrices.
        
        Parameters
        ----------
        connectivity : np.ndarray
            Connectivity matrices, shape (n_subjects, n_regions, n_regions)
        
        Returns
        -------
        connectivity_transformed : np.ndarray
            Transformed connectivity matrices
        """
        
        if connectivity.ndim != 3:
            raise ValueError(
                f"Expected 3D array (subjects, regions, regions), "
                f"got shape {connectivity.shape}"
            )
        
        connectivity_transformed = connectivity.copy()
        
        if self.apply_fisher_z:
            # Clip to valid correlation range
            connectivity_transformed = np.clip(
                connectivity_transformed, 
                -0.9999, 
                0.9999
            )
            
            # Apply Fisher Z
            connectivity_transformed = np.arctanh(connectivity_transformed)
            
            # Clip extreme values
            if self.clip_values:
                connectivity_transformed = np.clip(
                    connectivity_transformed,
                    self.clip_range[0],
                    self.clip_range[1]
                )
        
        logger.info(f"Transformed connectivity matrices: {connectivity_transformed.shape}")
        
        return connectivity_transformed


def preprocess_connectivity_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    diagonal_strategy: str = 'region_mean',
    apply_fisher_z: bool = True,
    standardize: bool = True,
    region_info: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, ConnectivityPreprocessor]:
    """
    Convenience function for preprocessing train and test sets.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training connectivity data
    X_test : np.ndarray
        Test connectivity data
    diagonal_strategy : str
        Diagonal imputation strategy
    apply_fisher_z : bool
        Apply Fisher Z-transformation
    standardize : bool
        Apply standardization
    region_info : pd.DataFrame, optional
        Region information
    verbose : bool
        Print progress
    
    Returns
    -------
    X_train_processed : np.ndarray
        Preprocessed training data
    X_test_processed : np.ndarray
        Preprocessed test data
    preprocessor : ConnectivityPreprocessor
        Fitted preprocessor
    """
    
    if verbose:
        logger.info("Preprocessing connectivity data...")
        logger.info(f"  Train shape: {X_train.shape}")
        logger.info(f"  Test shape: {X_test.shape}")
    
    # Initialize preprocessor
    preprocessor = ConnectivityPreprocessor(
        diagonal_strategy=diagonal_strategy,
        apply_fisher_z=apply_fisher_z,
        standardize=standardize,
        region_info=region_info
    )
    
    # Fit on training data
    preprocessor.fit(X_train)
    
    # Transform both sets
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    if verbose:
        logger.info("Preprocessing complete")
        logger.info(f"  Train processed: {X_train_processed.shape}")
        logger.info(f"  Test processed: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, preprocessor


# Example usage and testing
if __name__ == "__main__":
    """Test connectivity preprocessor."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Testing ConnectivityPreprocessor")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_subjects = 10
    n_regions = 116
    
    # Simulate connectivity matrices
    connectivity = np.random.rand(n_subjects, n_regions, n_regions)
    connectivity = (connectivity + connectivity.transpose(0, 2, 1)) / 2  # Make symmetric
    np.fill_diagonal(connectivity[0], 1.0)  # Set diagonal to 1
    
    # Reshape for classification (each region is a sample)
    X = connectivity.reshape(n_subjects * n_regions, n_regions)
    
    print(f"Synthetic data generated:")
    print(f"  Connectivity shape: {connectivity.shape}")
    print(f"  Classification X shape: {X.shape}")
    
    # Test preprocessing
    print("\nTesting preprocessing strategies...")
    
    strategies = ['zero', 'region_mean', 'global_mean']
    
    for strategy in strategies:
        print(f"\n{strategy.upper()}:")
        preprocessor = ConnectivityPreprocessor(
            diagonal_strategy=strategy,
            apply_fisher_z=True,
            standardize=True
        )
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)
        
        print(f"  Output shape: {X_transformed.shape}")
        print(f"  Output range: [{X_transformed.min():.3f}, {X_transformed.max():.3f}]")
        print(f"  Output mean: {X_transformed.mean():.3f}")
        print(f"  Output std: {X_transformed.std():.3f}")
    
    print("\n" + "="*60)
    print("Testing complete!")