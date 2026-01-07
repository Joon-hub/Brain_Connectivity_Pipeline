"""
Compare results across multiple models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import spearmanr, pearsonr


def create_performance_summary_table(
    all_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Create comprehensive performance summary table (Table 1).
    
    Parameters
    ----------
    all_results : dict
        Results from all models
    
    Returns
    -------
    summary_df : pd.DataFrame
        Performance summary table
    """
    
    rows = []
    
    for model_name, results in all_results.items():
        summary = results['summary']
        
        row = {
            'model': model_name,
            'scope': results['scope'],
            'strategy': results['strategy'],
            'n_regions': results['n_regions'],
            'rest_cv_accuracy': summary['rest_train_accuracy'],
            'task_test_accuracy': summary['task_test_accuracy'],
            'balanced_accuracy': summary['task_balanced_accuracy'],
            'accuracy_drop': summary['accuracy_drop'],
            'accuracy_drop_pct': (summary['accuracy_drop'] / summary['rest_train_accuracy']) * 100,
            'top_5_accuracy': summary.get('task_top_5_accuracy', np.nan),
            'n_rest_subjects': summary['n_rest_subjects'],
            'n_task_subjects': summary['n_task_subjects']
        }
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    # Sort by scope and strategy
    summary_df['scope_order'] = summary_df['scope'].map({'full': 1, 'left': 2, 'right': 3})
    summary_df['strategy_order'] = summary_df['strategy'].map({'multinomial': 1, 'ovr': 2, 'ovo': 3})
    summary_df = summary_df.sort_values(['scope_order', 'strategy_order'])
    summary_df = summary_df.drop(['scope_order', 'strategy_order'], axis=1)
    
    return summary_df


def create_network_performance_table(
    all_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Create network-level performance table (Table 2).
    
    Parameters
    ----------
    all_results : dict
        Results from all models
    
    Returns
    -------
    network_df : pd.DataFrame
        Network performance across models
    """
    
    # Collect network metrics from all models
    network_data = []
    
    for model_name, results in all_results.items():
        if 'network_metrics' not in results:
            continue
        
        net_metrics = results['network_metrics'].copy()
        net_metrics['model'] = model_name
        net_metrics['scope'] = results['scope']
        net_metrics['strategy'] = results['strategy']
        
        network_data.append(net_metrics)
    
    # Combine
    combined = pd.concat(network_data, ignore_index=True)
    
    # Pivot for easier reading
    pivot = combined.pivot_table(
        index='network',
        columns=['scope', 'strategy'],
        values='accuracy'
    )
    
    # Compute average drop per network
    pivot['mean_accuracy'] = pivot.mean(axis=1)
    pivot['std_accuracy'] = pivot.std(axis=1)
    
    return pivot


def compare_strategies(
    all_results: Dict[str, Dict],
    scope: str = 'full'
) -> pd.DataFrame:
    """
    Compare different strategies within a scope.
    
    Parameters
    ----------
    all_results : dict
        Results from all models
    scope : str
        'full', 'left', or 'right'
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Strategy comparison
    """
    
    # Filter by scope
    scope_models = {
        name: results 
        for name, results in all_results.items() 
        if results['scope'] == scope
    }
    
    rows = []
    
    for model_name, results in scope_models.items():
        summary = results['summary']
        
        row = {
            'strategy': results['strategy'],
            'task_accuracy': summary['task_test_accuracy'],
            'accuracy_drop': summary['accuracy_drop'],
            'top_5_accuracy': summary.get('task_top_5_accuracy', np.nan)
        }
        
        rows.append(row)
    
    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values('task_accuracy', ascending=False)
    
    return comparison_df


def compare_hemispheres(
    all_results: Dict[str, Dict],
    strategy: str = 'multinomial'
) -> pd.DataFrame:
    """
    Compare left vs right hemisphere for a given strategy.
    
    Parameters
    ----------
    all_results : dict
        Results from all models
    strategy : str
        'multinomial', 'ovr', or 'ovo'
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Hemisphere comparison
    """
    
    left_model = f'left_{strategy}'
    right_model = f'right_{strategy}'
    
    if left_model not in all_results or right_model not in all_results:
        raise ValueError(f"Models not found: {left_model}, {right_model}")
    
    left_summary = all_results[left_model]['summary']
    right_summary = all_results[right_model]['summary']
    
    comparison = {
        'metric': [
            'Rest CV Accuracy',
            'Task Test Accuracy',
            'Accuracy Drop',
            'Accuracy Drop %',
            'Top-5 Accuracy'
        ],
        'left_hemisphere': [
            left_summary['rest_train_accuracy'],
            left_summary['task_test_accuracy'],
            left_summary['accuracy_drop'],
            (left_summary['accuracy_drop'] / left_summary['rest_train_accuracy']) * 100,
            left_summary.get('task_top_5_accuracy', np.nan)
        ],
        'right_hemisphere': [
            right_summary['rest_train_accuracy'],
            right_summary['task_test_accuracy'],
            right_summary['accuracy_drop'],
            (right_summary['accuracy_drop'] / right_summary['rest_train_accuracy']) * 100,
            right_summary.get('task_top_5_accuracy', np.nan)
        ]
    }
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df['difference'] = comparison_df['left_hemisphere'] - comparison_df['right_hemisphere']
    
    return comparison_df


def compute_cross_model_agreement(
    all_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Compute agreement between models on region-level errors.
    
    Returns correlation matrix of reorganization patterns.
    
    Parameters
    ----------
    all_results : dict
        Results from all models
    
    Returns
    -------
    agreement_matrix : pd.DataFrame
        Correlation matrix between models
    """
    
    # Collect per-region accuracies
    model_accuracies = {}
    
    for model_name, results in all_results.items():
        if 'per_region_metrics' not in results:
            continue
        
        per_region = results['per_region_metrics']
        
        # Sort by region name for alignment
        per_region = per_region.sort_values('region_name')
        
        model_accuracies[model_name] = per_region['accuracy'].values
    
    # Create correlation matrix
    model_names = list(model_accuracies.keys())
    n_models = len(model_names)
    
    agreement_matrix = pd.DataFrame(
        np.zeros((n_models, n_models)),
        index=model_names,
        columns=model_names
    )
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                agreement_matrix.iloc[i, j] = 1.0
            else:
                # Check if regions align
                if len(model_accuracies[model1]) == len(model_accuracies[model2]):
                    corr, _ = spearmanr(model_accuracies[model1], model_accuracies[model2])
                    agreement_matrix.iloc[i, j] = corr
                else:
                    agreement_matrix.iloc[i, j] = np.nan
    
    return agreement_matrix