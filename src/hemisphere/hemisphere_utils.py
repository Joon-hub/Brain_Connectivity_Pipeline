"""
hemisphere_utils.py

Utility functions for hemisphere-specific data handling and manipulation.
Works with CSV data in wide format (subject × upper_triangle_connections).

CRITICAL UPDATE: Diagonal values (self-connections) are now removed before
reshaping to prevent them from being perfect predictors in classification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def load_region_info_from_csv(
    csv_path: Union[str, Path],
    hemisphere: str
) -> pd.DataFrame:
    """
    Extract region information from hemisphere-specific CSV file.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to hemisphere CSV file (e.g., LH_PIOP2_RestingState.csv)
    hemisphere : str
        'left' or 'right'
    
    Returns
    -------
    region_info : pd.DataFrame
        DataFrame with columns:
        - region_id: int (0 to n-1)
        - region_name: str
        - hemisphere: str ('L' or 'R')
    """
    
    csv_path = Path(csv_path)
    hemisphere = hemisphere.lower()
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Extracting region info from: {csv_path}")
    
    # Load CSV to get column names
    df = pd.read_csv(csv_path, nrows=0)  # Only load header
    
    # Get connectivity columns (format: region1~region2)
    conn_columns = [col for col in df.columns if '~' in col]
    
    # Extract unique regions
    regions = []
    seen = set()
    
    for col in conn_columns:
        region1, region2 = col.split('~')
        if region1 not in seen:
            regions.append(region1)
            seen.add(region1)
        if region2 not in seen:
            regions.append(region2)
            seen.add(region2)
    
    # Create region_info DataFrame
    region_info = pd.DataFrame({
        'region_id': range(len(regions)),
        'region_name': regions,
        'hemisphere': 'L' if hemisphere == 'left' else 'R'
    })
    
    # Add network information if possible (parse from region names)
    region_info['network'] = region_info['region_name'].apply(
        lambda x: _extract_network_from_name(x)
    )
    
    # Add atlas source
    region_info['atlas_source'] = region_info['region_name'].apply(
        lambda x: 'Schaefer' if x.startswith(('LH_', 'RH_')) else 'Tian'
    )
    
    logger.info(f"Extracted {len(region_info)} regions for {hemisphere} hemisphere")
    
    return region_info


def _extract_network_from_name(region_name: str) -> str:
    """
    Extract network name from region name.
    
    Examples:
    - LH_Vis_1 -> Vis
    - LH_Default_PFC_1 -> Default
    - striatum-lh -> subcortical
    """
    
    # Subcortical regions
    if region_name.endswith(('-lh', '-rh')):
        return 'Subcortical'
    
    # Cortical regions (Schaefer format: LH_Network_*)
    if region_name.startswith(('LH_', 'RH_')):
        parts = region_name.split('_')
        if len(parts) >= 2:
            return parts[1]  # Network is second part
    
    return 'Unknown'


def load_hemisphere_data_from_csv(
    csv_path: Union[str, Path],
    hemisphere: str,
    return_matrix: bool = True,
    validate: bool = True
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Load hemisphere-specific data from CSV in wide format.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to hemisphere CSV file
    hemisphere : str
        'left' or 'right'
    return_matrix : bool, default=True
        If True, convert to full connectivity matrices (n_subjects, n_regions, n_regions)
        If False, keep as upper triangle format
    validate : bool, default=True
        Whether to validate loaded data
    
    Returns
    -------
    data : dict
        Dictionary containing:
        - 'connectivity': np.ndarray, shape depends on return_matrix
        - 'subject_ids': np.ndarray, subject identifiers
        - 'region_info': pd.DataFrame, region metadata
        - 'hemisphere': str
        - 'n_subjects': int
        - 'n_regions': int
    """
    
    csv_path = Path(csv_path)
    hemisphere = hemisphere.lower()
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Loading {hemisphere} hemisphere data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Extract subject IDs
    subject_ids = df['subject'].values
    n_subjects = len(subject_ids)
    
    # Get connectivity columns
    conn_columns = [col for col in df.columns if '~' in col]
    n_connections = len(conn_columns)
    
    logger.info(f"  Subjects: {n_subjects}")
    logger.info(f"  Connections: {n_connections}")
    
    # Extract connectivity values
    connectivity_upper = df[conn_columns].values  # (n_subjects, n_connections)
    
    # Get region info
    region_info = load_region_info_from_csv(csv_path, hemisphere)
    n_regions = len(region_info)
    
    logger.info(f"  Regions: {n_regions}")
    
    # Validate upper triangle size
    expected_connections = n_regions * (n_regions - 1) // 2
    if n_connections != expected_connections:
        logger.warning(
            f"Connection count mismatch: expected {expected_connections}, "
            f"found {n_connections}"
        )
    
    if return_matrix:
        # Convert upper triangle to full matrices
        logger.info("Converting to full connectivity matrices...")
        connectivity = upper_triangle_to_matrix(
            connectivity_upper, 
            n_regions,
            conn_columns
        )
        logger.info(f"  Matrix shape: {connectivity.shape}")
    else:
        connectivity = connectivity_upper
    
    # Validate if requested
    if validate and return_matrix:
        _validate_connectivity_matrices(connectivity, n_regions)
    
    # Package data
    data = {
        'connectivity': connectivity,
        'subject_ids': subject_ids,
        'region_info': region_info,
        'hemisphere': hemisphere,
        'n_subjects': n_subjects,
        'n_regions': n_regions
    }
    
    logger.info(f"Successfully loaded {hemisphere} hemisphere data")
    
    return data


def upper_triangle_to_matrix(
    upper_triangle: np.ndarray,
    n_regions: int,
    conn_columns: List[str]
) -> np.ndarray:
    """
    Convert upper triangle connectivity to full symmetric matrices.
    
    Parameters
    ----------
    upper_triangle : np.ndarray
        Upper triangle values, shape (n_subjects, n_connections)
    n_regions : int
        Number of regions
    conn_columns : List[str]
        Column names in format "region1~region2"
    
    Returns
    -------
    matrices : np.ndarray
        Full connectivity matrices, shape (n_subjects, n_regions, n_regions)
    """
    
    n_subjects = upper_triangle.shape[0]
    matrices = np.zeros((n_subjects, n_regions, n_regions))
    
    # Create region name to index mapping
    region_names = []
    for col in conn_columns:
        r1, r2 = col.split('~')
        if r1 not in region_names:
            region_names.append(r1)
        if r2 not in region_names:
            region_names.append(r2)
    
    region_to_idx = {name: idx for idx, name in enumerate(region_names)}
    
    # Fill matrices
    for conn_idx, col in enumerate(conn_columns):
        region1, region2 = col.split('~')
        i = region_to_idx[region1]
        j = region_to_idx[region2]
        
        # Fill upper triangle
        matrices[:, i, j] = upper_triangle[:, conn_idx]
        # Make symmetric
        matrices[:, j, i] = upper_triangle[:, conn_idx]
    
    # Set diagonal to 1 (self-connections)
    for i in range(n_regions):
        matrices[:, i, i] = 1.0
    
    return matrices


def matrix_to_upper_triangle(
    matrices: np.ndarray,
    region_info: pd.DataFrame
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert full connectivity matrices to upper triangle format.
    
    Parameters
    ----------
    matrices : np.ndarray
        Full connectivity matrices, shape (n_subjects, n_regions, n_regions)
    region_info : pd.DataFrame
        Region information with region names
    
    Returns
    -------
    upper_triangle : np.ndarray
        Upper triangle values, shape (n_subjects, n_connections)
    conn_columns : List[str]
        Column names in format "region1~region2"
    """
    
    n_subjects, n_regions, _ = matrices.shape
    n_connections = n_regions * (n_regions - 1) // 2
    
    upper_triangle = np.zeros((n_subjects, n_connections))
    conn_columns = []
    
    region_names = region_info['region_name'].tolist()
    
    conn_idx = 0
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            upper_triangle[:, conn_idx] = matrices[:, i, j]
            conn_columns.append(f"{region_names[i]}~{region_names[j]}")
            conn_idx += 1
    
    return upper_triangle, conn_columns


def _validate_connectivity_matrices(
    connectivity: np.ndarray,
    n_regions: int
):
    """Validate connectivity matrices."""
    
    n_subjects, n_rows, n_cols = connectivity.shape
    
    # Check shape
    if n_rows != n_regions or n_cols != n_regions:
        raise ValueError(
            f"Shape mismatch: expected ({n_subjects}, {n_regions}, {n_regions}), "
            f"got {connectivity.shape}"
        )
    
    # Check for NaN/Inf
    if np.any(np.isnan(connectivity)):
        n_nan = np.sum(np.isnan(connectivity))
        logger.warning(f"Found {n_nan} NaN values in connectivity matrices")
    
    if np.any(np.isinf(connectivity)):
        raise ValueError("Inf values detected in connectivity matrices")
    
    # Check symmetry
    is_symmetric = np.allclose(
        connectivity, 
        connectivity.transpose(0, 2, 1), 
        atol=1e-6
    )
    if not is_symmetric:
        logger.warning("Connectivity matrices are not perfectly symmetric")
    
    # Check diagonal
    diag_values = np.array([connectivity[i, :, :].diagonal() for i in range(n_subjects)])
    if not np.allclose(diag_values, 1.0, atol=1e-6):
        logger.warning("Diagonal values are not all 1.0")
    
    logger.info("Connectivity validation completed")


def load_hemisphere_data(
    data_dir: Union[str, Path],
    hemisphere: str,
    dataset: str = 'rest',
    return_matrix: bool = True,
    validate: bool = True
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Load hemisphere-specific data with automatic file detection.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing hemisphere CSV files
    hemisphere : str
        'left' or 'right'
    dataset : str, default='rest'
        'rest' or 'task' (maps to PIOP2 or PIOP1)
    return_matrix : bool, default=True
        Convert to full matrices if True
    validate : bool, default=True
        Validate loaded data
    
    Returns
    -------
    data : dict
        Dictionary with connectivity, subject_ids, region_info, etc.
    """
    
    data_dir = Path(data_dir)
    hemisphere = hemisphere.lower()
    
    # Map dataset to filename patterns
    if dataset == 'rest':
        pattern = '*PIOP2*Resting*.csv'
    elif dataset == 'task':
        pattern = '*PIOP1*Gstroop*.csv'
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'rest' or 'task'")
    
    # Find matching file
    hemi_prefix = 'LH' if hemisphere == 'left' else 'RH'
    matching_files = list(data_dir.glob(f"{hemi_prefix}_{pattern}"))
    
    if not matching_files:
        # Try alternative patterns
        matching_files = list(data_dir.glob(f"{hemi_prefix}_*.csv"))
    
    if not matching_files:
        raise FileNotFoundError(
            f"No {hemisphere} hemisphere file found in {data_dir}\n"
            f"Looking for pattern: {hemi_prefix}_{pattern}"
        )
    
    if len(matching_files) > 1:
        logger.warning(
            f"Multiple files found for {hemisphere}/{dataset}, using first: "
            f"{matching_files[0]}"
        )
    
    csv_path = matching_files[0]
    
    return load_hemisphere_data_from_csv(
        csv_path=csv_path,
        hemisphere=hemisphere,
        return_matrix=return_matrix,
        validate=validate
    )


def get_hemisphere_indices(
    region_info: pd.DataFrame,
    hemisphere: str
) -> np.ndarray:
    """
    Get indices of regions (already filtered by hemisphere in region_info).
    
    For hemisphere-specific data, this just returns all indices.
    """
    return np.arange(len(region_info))


def create_labels_from_connectivity(
    connectivity: np.ndarray,
    n_regions: int
) -> np.ndarray:
    """
    Create labels array for classification (each sample labeled by its target region).
    
    Parameters
    ----------
    connectivity : np.ndarray
        Connectivity matrices (n_subjects, n_regions, n_regions)
    n_regions : int
        Number of regions
    
    Returns
    -------
    labels : np.ndarray
        Labels array, shape (n_subjects * n_regions,)
        Each subject contributes n_regions samples, labeled 0 to n_regions-1
    """
    
    n_subjects = connectivity.shape[0]
    labels = np.tile(np.arange(n_regions), n_subjects)
    
    return labels


def prepare_classification_data(
    connectivity: np.ndarray,
    region_info: pd.DataFrame,
    subject_ids: np.ndarray,
    remove_diagonal: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for classification by creating samples and labels.
    
    CRITICAL: Removes/zeros diagonal values BEFORE reshaping to prevent
    self-connections from being perfect predictors of region identity.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Connectivity matrices (n_subjects, n_regions, n_regions)
    region_info : pd.DataFrame
        Region information
    subject_ids : np.ndarray
        Subject identifiers
    remove_diagonal : bool, default=True
        If True, set diagonal values to 0 before reshaping.
        This prevents self-connections (=1.0) from being perfect predictors
        of region identity, which would artificially inflate accuracy to 100%.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_subjects * n_regions, n_regions)
        Each row is one region's connectivity pattern
    y : np.ndarray
        Labels (n_subjects * n_regions,)
        Region ID for each sample
    groups : np.ndarray
        Subject IDs for GroupKFold (n_subjects * n_regions,)
    
    Notes
    -----
    Without removing diagonals:
    - Self-connections are always 1.0 (by construction)
    - Feature vector for region i has 1.0 at position i
    - This is a perfect predictor → 100% accuracy (trivial solution)
    
    After removing diagonals:
    - Model must learn from actual connectivity patterns
    - Expected accuracy: 90-95% (realistic performance)
    - Classification errors reveal genuine confusion patterns
    
    Examples
    --------
    >>> # Without diagonal removal (WRONG - leads to 100% accuracy)
    >>> Region_0_features = [1.0, 0.3, 0.5, ...]  # 1.0 at position 0
    >>> Region_5_features = [0.2, 0.4, 0.6, 0.3, 0.5, 1.0, ...]  # 1.0 at position 5
    >>> # Model learns: "If feature[i]=1.0, predict region i" → trivial!
    
    >>> # With diagonal removal (CORRECT - realistic classification)
    >>> Region_0_features = [0.0, 0.3, 0.5, ...]  # 0.0 at position 0
    >>> Region_5_features = [0.2, 0.4, 0.6, 0.3, 0.5, 0.0, ...]  # 0.0 at position 5
    >>> # Model learns from connectivity patterns → realistic performance
    """
    
    n_subjects, n_regions, _ = connectivity.shape
    n_samples = n_subjects * n_regions
    
    # CRITICAL: Remove diagonal values BEFORE reshaping
    # Without this, diagonal=1.0 acts as perfect predictor of region identity
    if remove_diagonal:
        logger.info("  Removing diagonal values (self-connections) before classification")
        logger.info(f"  Original diagonal values (first subject, first 5 regions): "
                   f"{[connectivity[0, i, i] for i in range(min(5, n_regions))]}")
        
        connectivity = connectivity.copy()  # Don't modify original
        for i in range(n_subjects):
            np.fill_diagonal(connectivity[i], 0.0)
        
        logger.info(f"  After removal (should be all zeros): "
                   f"{[connectivity[0, i, i] for i in range(min(5, n_regions))]}")
    else:
        logger.warning("  ⚠️  Diagonal values NOT removed - self-connections will be perfect predictors!")
        logger.warning("  ⚠️  This will lead to artificially inflated 100% accuracy!")
    
    # Reshape connectivity: (n_subjects, n_regions, n_regions) -> (n_samples, n_regions)
    X = connectivity.reshape(n_samples, n_regions)
    
    # Verify no feature is constant at 1.0 (would indicate diagonal still present)
    if remove_diagonal:
        problematic_features = []
        for feature_idx in range(n_regions):
            unique_vals = np.unique(X[:, feature_idx])
            if len(unique_vals) == 1 and abs(unique_vals[0] - 1.0) < 1e-6:
                problematic_features.append(feature_idx)
        
        if problematic_features:
            logger.error(f"  ❌ {len(problematic_features)} features are constant at 1.0 - "
                        f"diagonal removal failed!")
            logger.error(f"  Problematic features: {problematic_features[:10]}")
        else:
            logger.info("  ✅ Diagonal removal verified - no features constant at 1.0")
    
    # Create labels: each sample labeled by its target region
    y = np.tile(np.arange(n_regions), n_subjects)
    
    # Create groups: repeat each subject_id n_regions times
    groups = np.repeat(subject_ids, n_regions)
    
    logger.info(f"Prepared classification data:")
    logger.info(f"  Features (X): {X.shape}")
    logger.info(f"  Labels (y): {y.shape}")
    logger.info(f"  Groups: {groups.shape}")
    logger.info(f"  Unique labels: {len(np.unique(y))}")
    logger.info(f"  Unique subjects: {len(np.unique(groups))}")
    logger.info(f"  Feature range: [{X.min():.3f}, {X.max():.3f}]")
    logger.info(f"  Feature mean: {X.mean():.3f}, std: {X.std():.3f}")
    
    return X, y, groups


def save_hemisphere_data_npy(
    connectivity: np.ndarray,
    subject_ids: np.ndarray,
    region_info: pd.DataFrame,
    output_dir: Union[str, Path],
    hemisphere: str,
    dataset: str = 'rest'
):
    """
    Save hemisphere data in numpy format for faster loading.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Connectivity matrices
    subject_ids : np.ndarray
        Subject identifiers
    region_info : pd.DataFrame
        Region information
    output_dir : str or Path
        Output directory
    hemisphere : str
        'left' or 'right'
    dataset : str
        'rest' or 'task'
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(
        output_dir / f"connectivity_{dataset}_{hemisphere}.npy",
        connectivity
    )
    np.save(
        output_dir / f"subject_ids_{dataset}_{hemisphere}.npy",
        subject_ids
    )
    
    # Save region info
    region_info.to_csv(
        output_dir / f"region_info_{hemisphere}.csv",
        index=False
    )
    
    logger.info(f"Saved {hemisphere} hemisphere data to {output_dir}")


def print_hemisphere_summary(data: Dict):
    """Print summary of loaded hemisphere data."""
    
    print("\n" + "="*60)
    print(f"HEMISPHERE DATA SUMMARY - {data['hemisphere'].upper()}")
    print("="*60)
    print(f"Subjects: {data['n_subjects']}")
    print(f"Regions: {data['n_regions']}")
    print(f"Connectivity shape: {data['connectivity'].shape}")
    
    if 'region_info' in data:
        region_info = data['region_info']
        print(f"\nRegion breakdown:")
        if 'network' in region_info.columns:
            network_counts = region_info['network'].value_counts()
            for network, count in network_counts.items():
                print(f"  {network}: {count}")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    """Test hemisphere utilities."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load left hemisphere resting-state data
    data_dir = Path("data/processed/hemispheres")
    
    if data_dir.exists():
        try:
            data = load_hemisphere_data(
                data_dir=data_dir,
                hemisphere='left',
                dataset='rest'
            )
            
            print_hemisphere_summary(data)
            
            # Prepare for classification
            X, y, groups = prepare_classification_data(
                connectivity=data['connectivity'],
                region_info=data['region_info'],
                subject_ids=data['subject_ids'],
                remove_diagonal=True  # CRITICAL: Set to True!
            )
            
            print(f"Classification data ready:")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  groups shape: {groups.shape}")
            
        except FileNotFoundError as e:
            print(f"Data not found: {e}")
    else:
        print(f"Data directory not found: {data_dir}")