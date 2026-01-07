"""
Compute analysis-specific metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def compute_reorganization_index(
    rest_accuracy: float,
    task_accuracy: float
) -> float:
    """
    Compute reorganization index for a region/network.
    
    RI = 1 - (task_accuracy / rest_accuracy)
    
    High RI = Large reorganization
    Low RI = Stable connectivity
    """
    
    if rest_accuracy == 0:
        return np.nan
    
    return 1.0 - (task_accuracy / rest_accuracy)


def compute_per_region_reorganization(
    per_region_rest: pd.DataFrame,
    per_region_task: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute reorganization index for each region.
    
    Parameters
    ----------
    per_region_rest : pd.DataFrame
        Per-region metrics from rest CV
    per_region_task : pd.DataFrame
        Per-region metrics from task testing
    
    Returns
    -------
    reorg_df : pd.DataFrame
        DataFrame with reorganization metrics
    """
    
    merged = per_region_rest.merge(
        per_region_task,
        on='region_name',
        suffixes=('_rest', '_task')
    )
    
    merged['reorganization_index'] = merged.apply(
        lambda row: compute_reorganization_index(
            row['accuracy_rest'],
            row['accuracy_task']
        ),
        axis=1
    )
    
    merged['accuracy_drop'] = merged['accuracy_rest'] - merged['accuracy_task']
    merged['accuracy_drop_pct'] = (merged['accuracy_drop'] / merged['accuracy_rest']) * 100
    
    return merged


def classify_error_types(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Classify each misclassification into error types.
    
    Error Types:
    1. within_network: True and predicted regions in same network
    2. cross_network: Different networks
    3. within_hemisphere: Same hemisphere
    4. cross_hemisphere: Different hemispheres
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (region indices)
    y_pred : np.ndarray
        Predicted labels
    region_info : pd.DataFrame
        Region metadata with 'network' and 'hemisphere' columns
    
    Returns
    -------
    error_df : pd.DataFrame
        DataFrame with error classifications
    """
    
    errors = []
    
    for i, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)):
        if true_idx == pred_idx:
            continue  # Skip correct predictions
        
        true_region = region_info.iloc[true_idx]
        pred_region = region_info.iloc[pred_idx]
        
        error_type = {
            'sample_idx': i,
            'true_region': true_region['region_name'],
            'pred_region': pred_region['region_name'],
            'true_network': true_region['network'],
            'pred_network': pred_region['network'],
            'true_hemisphere': true_region['hemisphere'],
            'pred_hemisphere': pred_region['hemisphere'],
            'within_network': true_region['network'] == pred_region['network'],
            'cross_network': true_region['network'] != pred_region['network'],
            'within_hemisphere': true_region['hemisphere'] == pred_region['hemisphere'],
            'cross_hemisphere': true_region['hemisphere'] != pred_region['hemisphere']
        }
        
        errors.append(error_type)
    
    return pd.DataFrame(errors)


def compute_error_type_summary(error_df: pd.DataFrame) -> Dict:
    """
    Summarize error types.
    
    Returns
    -------
    summary : dict
        Error type counts and percentages
    """
    
    total_errors = len(error_df)
    
    summary = {
        'total_errors': total_errors,
        'within_network_count': error_df['within_network'].sum(),
        'cross_network_count': error_df['cross_network'].sum(),
        'within_hemisphere_count': error_df['within_hemisphere'].sum(),
        'cross_hemisphere_count': error_df['cross_hemisphere'].sum(),
    }
    
    # Percentages
    summary['within_network_pct'] = (summary['within_network_count'] / total_errors) * 100
    summary['cross_network_pct'] = (summary['cross_network_count'] / total_errors) * 100
    summary['within_hemisphere_pct'] = (summary['within_hemisphere_count'] / total_errors) * 100
    summary['cross_hemisphere_pct'] = (summary['cross_hemisphere_count'] / total_errors) * 100
    
    return summary


def compute_network_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute confusion matrix at network level.
    
    Parameters
    ----------
    y_true : np.ndarray
        True region labels
    y_pred : np.ndarray
        Predicted region labels
    region_info : pd.DataFrame
        Region metadata
    
    Returns
    -------
    network_cm : pd.DataFrame
        Network-level confusion matrix
    """
    
    # Map region indices to networks
    true_networks = [region_info.iloc[idx]['network'] for idx in y_true]
    pred_networks = [region_info.iloc[idx]['network'] for idx in y_pred]
    
    # Get unique networks
    networks = sorted(region_info['network'].unique())
    
    # Create confusion matrix
    network_cm = pd.DataFrame(
        0,
        index=networks,
        columns=networks
    )
    
    for true_net, pred_net in zip(true_networks, pred_networks):
        network_cm.loc[true_net, pred_net] += 1
    
    return network_cm


def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 5
) -> float:
    """
    Compute top-k accuracy.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Prediction probabilities (n_samples, n_classes)
    k : int
        Number of top predictions to consider
    
    Returns
    -------
    top_k_acc : float
        Top-k accuracy
    """
    
    n_samples = len(y_true)
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / n_samples