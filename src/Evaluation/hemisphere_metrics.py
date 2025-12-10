"""
hemisphere_metrics.py

Comprehensive metrics computation for hemisphere-specific brain region classification.
Provides overall, per-region, and network-level performance evaluation.

Author: Joon
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    top_k_accuracy_score
)


logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities, shape (n_samples, n_classes)
    average : str, default='weighted'
        Averaging method for multi-class metrics
    
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - accuracy: Overall accuracy
        - balanced_accuracy: Balanced accuracy (accounts for class imbalance)
        - precision: Weighted precision
        - recall: Weighted recall
        - f1_score: Weighted F1 score
        - cohen_kappa: Cohen's kappa coefficient
        - matthews_corrcoef: Matthews correlation coefficient
        - top_5_accuracy: Top-5 accuracy (if y_proba provided)
        - top_10_accuracy: Top-10 accuracy (if y_proba provided)
        - error_rate: Overall error rate
    """
    
    logger.info("Computing classification metrics...")
    
    # Basic validation
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    # Core metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'error_rate': 1.0 - accuracy_score(y_true, y_pred)
    }
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1_score'] = float(f1)
    
    # Cohen's Kappa (agreement beyond chance)
    try:
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Could not compute Cohen's kappa: {e}")
        metrics['cohen_kappa'] = np.nan
    
    # Matthews Correlation Coefficient (for multiclass)
    try:
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Could not compute Matthews correlation: {e}")
        metrics['matthews_corrcoef'] = np.nan
    
    # Top-K accuracy (if probabilities provided)
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        
        # Top-5 accuracy
        if n_classes >= 5:
            try:
                metrics['top_5_accuracy'] = top_k_accuracy_score(
                    y_true, y_proba, k=5
                )
            except Exception as e:
                logger.warning(f"Could not compute top-5 accuracy: {e}")
                metrics['top_5_accuracy'] = np.nan
        
        # Top-10 accuracy
        if n_classes >= 10:
            try:
                metrics['top_10_accuracy'] = top_k_accuracy_score(
                    y_true, y_proba, k=10
                )
            except Exception as e:
                logger.warning(f"Could not compute top-10 accuracy: {e}")
                metrics['top_10_accuracy'] = np.nan
    
    # Log summary
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics


def compute_per_region_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_info: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute per-region classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    region_info : pd.DataFrame, optional
        Region information with columns: region_id, region_name, network
    
    Returns
    -------
    per_region_metrics : pd.DataFrame
        DataFrame with columns:
        - region_id: Region identifier
        - region_name: Region name (if region_info provided)
        - network: Network assignment (if region_info provided)
        - n_samples: Number of samples for this region
        - accuracy: Per-region accuracy (recall)
        - precision: Precision for this region
        - recall: Recall for this region
        - f1_score: F1 score for this region
        - support: Number of true samples
    """
    
    logger.info("Computing per-region metrics...")
    
    # Get unique labels
    unique_labels = np.unique(y_true)
    n_regions = len(unique_labels)
    
    logger.info(f"  Computing metrics for {n_regions} regions")
    
    # Compute precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )
    
    # Create DataFrame
    per_region_data = {
        'region_id': unique_labels,
        'n_samples': support,
        'accuracy': recall,  # Recall = per-class accuracy
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }
    
    per_region_metrics = pd.DataFrame(per_region_data)
    
    # Add region names and network info if available
    if region_info is not None:
        # Ensure region_info has region_id column
        if 'region_id' in region_info.columns:
            # Merge with region_info
            per_region_metrics = per_region_metrics.merge(
                region_info[['region_id', 'region_name', 'network']]
                if 'network' in region_info.columns
                else region_info[['region_id', 'region_name']],
                on='region_id',
                how='left'
            )
            
            # Reorder columns
            base_cols = ['region_id', 'region_name']
            if 'network' in per_region_metrics.columns:
                base_cols.append('network')
            metric_cols = ['n_samples', 'accuracy', 'precision', 'recall', 'f1_score', 'support']
            per_region_metrics = per_region_metrics[base_cols + metric_cols]
    
    # Sort by accuracy (descending)
    per_region_metrics = per_region_metrics.sort_values('accuracy', ascending=False)
    
    # Summary statistics
    logger.info(f"  Mean per-region accuracy: {per_region_metrics['accuracy'].mean():.4f}")
    logger.info(f"  Std per-region accuracy: {per_region_metrics['accuracy'].std():.4f}")
    logger.info(f"  Min per-region accuracy: {per_region_metrics['accuracy'].min():.4f}")
    logger.info(f"  Max per-region accuracy: {per_region_metrics['accuracy'].max():.4f}")
    
    return per_region_metrics


def compute_network_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute network-level classification metrics by aggregating regions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (region IDs)
    y_pred : np.ndarray
        Predicted labels (region IDs)
    region_info : pd.DataFrame
        Region information with 'region_id' and 'network' columns
    
    Returns
    -------
    network_metrics : pd.DataFrame
        DataFrame with columns:
        - network: Network name
        - n_regions: Number of regions in this network
        - n_samples: Total samples for this network
        - accuracy: Network-level accuracy
        - precision: Network-level precision
        - recall: Network-level recall
        - f1_score: Network-level F1 score
    """
    
    logger.info("Computing network-level metrics...")
    
    # Check if network column exists
    if 'network' not in region_info.columns:
        logger.warning("No 'network' column in region_info, cannot compute network metrics")
        return pd.DataFrame()
    
    # Get network assignment for each region
    region_to_network = dict(zip(
        region_info['region_id'],
        region_info['network']
    ))
    
    # Map labels to networks
    y_true_network = np.array([region_to_network.get(label, 'Unknown') for label in y_true])
    y_pred_network = np.array([region_to_network.get(label, 'Unknown') for label in y_pred])
    
    # Get unique networks
    unique_networks = region_info['network'].unique()
    n_networks = len(unique_networks)
    
    logger.info(f"  Computing metrics for {n_networks} networks")
    
    # Compute metrics for each network
    network_data = []
    
    for network in unique_networks:
        # Get regions in this network
        network_regions = region_info[region_info['network'] == network]['region_id'].values
        n_regions = len(network_regions)
        
        # Get samples for this network
        network_mask = np.isin(y_true, network_regions)
        
        if not np.any(network_mask):
            logger.warning(f"No samples found for network: {network}")
            continue
        
        y_true_net = y_true[network_mask]
        y_pred_net = y_pred[network_mask]
        n_samples = len(y_true_net)
        
        # Compute metrics
        accuracy = accuracy_score(y_true_net, y_pred_net)
        
        # Precision, recall, F1 (macro average across regions in network)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_net, y_pred_net,
            labels=network_regions,
            average='macro',
            zero_division=0
        )
        
        network_data.append({
            'network': network,
            'n_regions': n_regions,
            'n_samples': n_samples,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    network_metrics = pd.DataFrame(network_data)
    
    # Sort by accuracy (descending)
    network_metrics = network_metrics.sort_values('accuracy', ascending=False)
    
    # Log summary
    if len(network_metrics) > 0:
        logger.info(f"  Mean network accuracy: {network_metrics['accuracy'].mean():.4f}")
        logger.info(f"  Best network: {network_metrics.iloc[0]['network']} "
                   f"(accuracy={network_metrics.iloc[0]['accuracy']:.4f})")
        logger.info(f"  Worst network: {network_metrics.iloc[-1]['network']} "
                   f"(accuracy={network_metrics.iloc[-1]['accuracy']:.4f})")
    
    return network_metrics


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Create confusion matrix for classification results.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    n_classes : int, optional
        Number of classes (if not provided, inferred from labels)
    normalize : str, optional
        Normalization mode: 'true', 'pred', 'all', or None
        - 'true': Normalize by true class (row sums to 1)
        - 'pred': Normalize by predicted class (column sums to 1)
        - 'all': Normalize by total count
        - None: Raw counts
    
    Returns
    -------
    cm : np.ndarray
        Confusion matrix, shape (n_classes, n_classes)
    """
    
    logger.info("Creating confusion matrix...")
    
    # Infer n_classes if not provided
    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    # Create confusion matrix
    cm = confusion_matrix(
        y_true, y_pred,
        labels=np.arange(n_classes)
    )
    
    logger.info(f"  Confusion matrix shape: {cm.shape}")
    logger.info(f"  Total predictions: {np.sum(cm)}")
    logger.info(f"  Correct predictions (diagonal): {np.trace(cm)}")
    
    # Normalize if requested
    if normalize is not None:
        if normalize == 'true':
            # Normalize by true class (rows)
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            logger.info("  Normalized by true class (rows sum to 1)")
        elif normalize == 'pred':
            # Normalize by predicted class (columns)
            cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
            logger.info("  Normalized by predicted class (columns sum to 1)")
        elif normalize == 'all':
            # Normalize by total
            cm = cm.astype('float') / cm.sum()
            logger.info("  Normalized by total count")
        else:
            raise ValueError(f"Invalid normalize value: {normalize}")
    
    return cm


def analyze_confusion_patterns(
    cm: np.ndarray,
    region_info: Optional[pd.DataFrame] = None,
    top_k: int = 10
) -> Dict[str, Union[List, pd.DataFrame]]:
    """
    Analyze confusion patterns from confusion matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    region_info : pd.DataFrame, optional
        Region information for labeling
    top_k : int, default=10
        Number of top confusions to return
    
    Returns
    -------
    analysis : dict
        Dictionary containing:
        - 'most_confused_pairs': List of (region_i, region_j, count) tuples
        - 'per_region_confusion': DataFrame with confusion statistics per region
        - 'within_network_errors': Percentage of errors within same network
    """
    
    logger.info("Analyzing confusion patterns...")
    
    n_classes = cm.shape[0]
    analysis = {}
    
    # Find most confused pairs (off-diagonal elements)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    # Get top-k confused pairs
    flat_indices = np.argsort(cm_no_diag.ravel())[::-1][:top_k]
    confused_pairs = []
    
    for flat_idx in flat_indices:
        i, j = np.unravel_index(flat_idx, cm_no_diag.shape)
        count = cm_no_diag[i, j]
        
        if count > 0:
            pair_info = {
                'true_region': int(i),
                'predicted_region': int(j),
                'count': int(count),
                'percentage': float(count / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
            }
            
            # Add region names if available
            if region_info is not None and 'region_name' in region_info.columns:
                region_names = region_info['region_name'].values
                if i < len(region_names) and j < len(region_names):
                    pair_info['true_region_name'] = region_names[i]
                    pair_info['predicted_region_name'] = region_names[j]
            
            confused_pairs.append(pair_info)
    
    analysis['most_confused_pairs'] = confused_pairs
    
    # Per-region confusion statistics
    per_region_confusion = []
    
    for i in range(n_classes):
        total_samples = cm[i].sum()
        correct = cm[i, i]
        incorrect = total_samples - correct
        
        if total_samples > 0:
            per_region_confusion.append({
                'region_id': i,
                'total_samples': int(total_samples),
                'correct': int(correct),
                'incorrect': int(incorrect),
                'accuracy': float(correct / total_samples),
                'error_rate': float(incorrect / total_samples)
            })
    
    analysis['per_region_confusion'] = pd.DataFrame(per_region_confusion)
    
    # Within-network vs between-network errors
    if region_info is not None and 'network' in region_info.columns:
        within_network_errors = 0
        between_network_errors = 0
        
        region_to_network = dict(zip(
            region_info['region_id'],
            region_info['network']
        ))
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    network_i = region_to_network.get(i, 'Unknown')
                    network_j = region_to_network.get(j, 'Unknown')
                    
                    if network_i == network_j:
                        within_network_errors += cm[i, j]
                    else:
                        between_network_errors += cm[i, j]
        
        total_errors = within_network_errors + between_network_errors
        
        if total_errors > 0:
            analysis['within_network_error_pct'] = float(
                within_network_errors / total_errors * 100
            )
            analysis['between_network_error_pct'] = float(
                between_network_errors / total_errors * 100
            )
            
            logger.info(f"  Within-network errors: {analysis['within_network_error_pct']:.1f}%")
            logger.info(f"  Between-network errors: {analysis['between_network_error_pct']:.1f}%")
    
    logger.info(f"  Found {len(confused_pairs)} most confused pairs")
    
    return analysis


def compare_hemisphere_performance(
    left_metrics: Dict[str, float],
    right_metrics: Dict[str, float],
    left_per_region: pd.DataFrame,
    right_per_region: pd.DataFrame
) -> Dict[str, Union[float, Dict]]:
    """
    Compare performance between left and right hemispheres.
    
    Parameters
    ----------
    left_metrics : dict
        Overall metrics for left hemisphere
    right_metrics : dict
        Overall metrics for right hemisphere
    left_per_region : pd.DataFrame
        Per-region metrics for left hemisphere
    right_per_region : pd.DataFrame
        Per-region metrics for right hemisphere
    
    Returns
    -------
    comparison : dict
        Dictionary containing:
        - 'accuracy_difference': Difference in accuracy (left - right)
        - 'better_hemisphere': 'left', 'right', or 'equal'
        - 'statistical_test': Results of paired t-test on per-region accuracy
        - 'correlation': Correlation of per-region accuracies
    """
    
    logger.info("Comparing hemisphere performance...")
    
    comparison = {}
    
    # Overall accuracy comparison
    left_acc = left_metrics.get('accuracy', 0)
    right_acc = right_metrics.get('accuracy', 0)
    
    comparison['left_accuracy'] = left_acc
    comparison['right_accuracy'] = right_acc
    comparison['accuracy_difference'] = left_acc - right_acc
    
    if left_acc > right_acc + 0.001:
        comparison['better_hemisphere'] = 'left'
    elif right_acc > left_acc + 0.001:
        comparison['better_hemisphere'] = 'right'
    else:
        comparison['better_hemisphere'] = 'equal'
    
    logger.info(f"  Left accuracy: {left_acc:.4f}")
    logger.info(f"  Right accuracy: {right_acc:.4f}")
    logger.info(f"  Difference: {comparison['accuracy_difference']:.4f}")
    
    # Statistical test on per-region accuracies
    if len(left_per_region) == len(right_per_region):
        left_accuracies = left_per_region.sort_values('region_id')['accuracy'].values
        right_accuracies = right_per_region.sort_values('region_id')['accuracy'].values
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(left_accuracies, right_accuracies)
        
        comparison['statistical_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
        
        # Correlation
        correlation, corr_p = stats.pearsonr(left_accuracies, right_accuracies)
        
        comparison['per_region_correlation'] = {
            'correlation': float(correlation),
            'p_value': float(corr_p)
        }
        
        logger.info(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        logger.info(f"  Per-region correlation: r={correlation:.4f}")
    else:
        logger.warning("Cannot perform statistical tests: different number of regions")
    
    return comparison


def compute_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_info: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute distribution of prediction errors.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    region_info : pd.DataFrame, optional
        Region information
    
    Returns
    -------
    error_dist : pd.DataFrame
        DataFrame with error distribution statistics
    """
    
    logger.info("Computing error distribution...")
    
    # Find misclassified samples
    errors = y_true != y_pred
    n_errors = np.sum(errors)
    n_total = len(y_true)
    
    logger.info(f"  Total errors: {n_errors} / {n_total} ({n_errors/n_total*100:.2f}%)")
    
    if n_errors == 0:
        logger.info("  Perfect classification - no errors to analyze!")
        return pd.DataFrame()
    
    # Analyze error types
    error_data = {
        'true_label': y_true[errors],
        'predicted_label': y_pred[errors]
    }
    
    error_df = pd.DataFrame(error_data)
    
    # Add region names if available
    if region_info is not None and 'region_name' in region_info.columns:
        region_names = dict(zip(region_info['region_id'], region_info['region_name']))
        error_df['true_region_name'] = error_df['true_label'].map(region_names)
        error_df['predicted_region_name'] = error_df['predicted_label'].map(region_names)
        
        # Add network info
        if 'network' in region_info.columns:
            region_networks = dict(zip(region_info['region_id'], region_info['network']))
            error_df['true_network'] = error_df['true_label'].map(region_networks)
            error_df['predicted_network'] = error_df['predicted_label'].map(region_networks)
            
            # Within vs between network errors
            error_df['within_network'] = (
                error_df['true_network'] == error_df['predicted_network']
            )
            
            within_net = error_df['within_network'].sum()
            logger.info(f"  Within-network errors: {within_net} ({within_net/n_errors*100:.1f}%)")
    
    return error_df


# Example usage and testing
if __name__ == "__main__":
    """Test hemisphere metrics computation."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Hemisphere Metrics")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 116
    
    # Simulate predictions with 90% accuracy
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    
    # Introduce 10% errors
    n_errors = int(0.1 * n_samples)
    error_indices = np.random.choice(n_samples, n_errors, replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, n_errors)
    
    # Simulate probabilities
    y_proba = np.random.rand(n_samples, n_classes)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    print(f"\nSynthetic data generated:")
    print(f"  Samples: {n_samples}")
    print(f"  Classes: {n_classes}")
    print(f"  True accuracy: ~90%")
    
    # Test overall metrics
    print("\n" + "-"*60)
    print("Testing compute_classification_metrics:")
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Create synthetic region info
    region_info = pd.DataFrame({
        'region_id': range(n_classes),
        'region_name': [f'Region_{i}' for i in range(n_classes)],
        'network': [f'Network_{i%7}' for i in range(n_classes)]
    })
    
    # Test per-region metrics
    print("\n" + "-"*60)
    print("Testing compute_per_region_metrics:")
    per_region = compute_per_region_metrics(y_true, y_pred, region_info)
    print(per_region.head(10))
    
    # Test network-level metrics
    print("\n" + "-"*60)
    print("Testing compute_network_level_metrics:")
    network_metrics = compute_network_level_metrics(y_true, y_pred, region_info)
    print(network_metrics)
    
    # Test confusion matrix
    print("\n" + "-"*60)
    print("Testing create_confusion_matrix:")
    cm = create_confusion_matrix(y_true, y_pred, n_classes)
    print(f"  Shape: {cm.shape}")
    print(f"  Diagonal sum: {np.trace(cm)}")
    print(f"  Total: {np.sum(cm)}")
    
    # Test confusion analysis
    print("\n" + "-"*60)
    print("Testing analyze_confusion_patterns:")
    analysis = analyze_confusion_patterns(cm, region_info, top_k=5)
    print(f"  Most confused pairs: {len(analysis['most_confused_pairs'])}")
    if analysis['most_confused_pairs']:
        print("  Top confusion:")
        print(f"    {analysis['most_confused_pairs'][0]}")
    
    print("\n" + "="*60)
    print("Testing complete!")