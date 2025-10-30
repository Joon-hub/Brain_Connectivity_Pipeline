"""
Model Evaluation and Metrics
=============================
Calculate error maps and save results to CSV.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
from typing import Dict, List, Tuple


def calculate_error_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_list: List[str]
) -> pd.DataFrame:
    """
    Calculate per-region misclassification rates.
    
    Args:
        y_true: True region labels
        y_pred: Predicted labels
        region_list: Region names
        
    Returns:
        DataFrame with columns: region_index, region_name, misclassification_rate, n_samples
    """
    n_regions = len(region_list)
    error_rates = np.zeros(n_regions)
    sample_counts = np.zeros(n_regions, dtype=int)
    
    for region_idx in range(n_regions):
        mask = (y_true == region_idx)
        if mask.any():
            region_true = y_true[mask]
            region_pred = y_pred[mask]
            
            accuracy = accuracy_score(region_true, region_pred)
            error_rates[region_idx] = 1.0 - accuracy
            sample_counts[region_idx] = mask.sum()
    
    df = pd.DataFrame({
        'region_index': range(n_regions),
        'region_name': region_list,
        'misclassification_rate': error_rates,
        'n_samples': sample_counts
    })
    
    # Sort by error rate (descending)
    df = df.sort_values('misclassification_rate', ascending=False).reset_index(drop=True)
    
    return df


def calculate_global_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "unknown"
) -> Dict:
    """
    Calculate overall classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        dataset_name: Name for logging
        
    Returns:
        Dictionary of metrics
    """
    overall_acc = accuracy_score(y_true, y_pred)
    
    # Random baseline (1/n_classes)
    n_classes = len(np.unique(y_true))
    random_baseline = 1.0 / n_classes
    improvement = overall_acc / random_baseline
    
    metrics = {
        'dataset': dataset_name,
        'accuracy': overall_acc,
        'n_samples': len(y_true),
        'n_classes': n_classes,
        'random_baseline': random_baseline,
        'improvement_over_random': improvement
    }
    
    return metrics


def compare_error_maps(
    error_map_rest: pd.DataFrame,
    error_map_task: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare error rates between rest and task conditions.
    
    Args:
        error_map_rest: Error map from resting-state
        error_map_task: Error map from task
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison = pd.merge(
        error_map_rest[['region_name', 'misclassification_rate']],
        error_map_task[['region_name', 'misclassification_rate']],
        on='region_name',
        suffixes=('_rest', '_task')
    )
    
    comparison['error_increase'] = (
        comparison['misclassification_rate_task'] - 
        comparison['misclassification_rate_rest']
    )
    
    # Sort by error increase (descending)
    comparison = comparison.sort_values('error_increase', ascending=False).reset_index(drop=True)
    
    return comparison


def aggregate_by_network(
    error_map: pd.DataFrame,
    region_list: List[str]
) -> pd.DataFrame:
    """
    Aggregate error rates by brain network.
    
    Args:
        error_map: Per-region error map
        region_list: Region names
        
    Returns:
        DataFrame with network-level statistics
    """
    from features import parse_networks
    
    # Parse networks
    networks = parse_networks(region_list)
    
    # Add network column
    error_map_copy = error_map.copy()
    error_map_copy['network'] = [networks[idx] for idx in error_map_copy['region_index']]
    
    # Aggregate
    network_stats = error_map_copy.groupby('network').agg({
        'misclassification_rate': ['mean', 'std', 'min', 'max'],
        'n_samples': 'sum'
    }).reset_index()
    
    # Flatten column names
    network_stats.columns = ['network', 'mean_error', 'std_error', 'min_error', 'max_error', 'total_samples']
    
    # Add number of regions per network
    network_stats['n_regions'] = error_map_copy.groupby('network')['region_index'].nunique().values
    
    # Sort by mean error
    network_stats = network_stats.sort_values('mean_error', ascending=False).reset_index(drop=True)
    
    return network_stats


def save_results_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV in reports/tables/."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    
    n_samples = 1000
    n_regions = 232
    
    y_true = np.random.randint(0, n_regions, n_samples)
    y_pred = y_true.copy()
    y_pred[np.random.rand(n_samples) < 0.3] = np.random.randint(0, n_regions, (y_pred != y_true).sum())
    
    region_list = [f"Region_{i}" for i in range(n_regions)]
    
    # Calculate error map
    error_map = calculate_error_map(y_true, y_pred, region_list)
    print(f"Error map shape: {error_map.shape}")
    print(f"Mean error: {error_map['misclassification_rate'].mean():.4f}")
    
    # Global metrics
    metrics = calculate_global_metrics(y_true, y_pred, "test")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"Improvement over random: {metrics['improvement_over_random']:.1f}x")