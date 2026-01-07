"""
Statistical tests for model comparisons.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, Tuple


def friedman_test_strategies(
    all_results: Dict[str, Dict],
    scope: str = 'full'
) -> Tuple[float, float, pd.DataFrame]:
    """
    Friedman test comparing 3 strategies across networks.
    
    H0: All strategies perform equally across networks
    H1: At least one strategy differs
    
    Parameters
    ----------
    all_results : dict
        Results from all models
    scope : str
        Scope to test
    
    Returns
    -------
    statistic : float
        Friedman test statistic
    p_value : float
        P-value
    data_df : pd.DataFrame
        Data used for test
    """
    
    strategies = ['multinomial', 'ovr', 'ovo']
    
    # Collect network accuracies for each strategy
    network_data = {strategy: [] for strategy in strategies}
    
    for strategy in strategies:
        model_name = f'{scope}_{strategy}'
        
        if model_name in all_results and 'network_metrics' in all_results[model_name]:
            net_metrics = all_results[model_name]['network_metrics']
            net_metrics = net_metrics.sort_values('network')  # Ensure alignment
            network_data[strategy] = net_metrics['accuracy'].values
    
    # Verify all strategies have same networks
    lengths = [len(v) for v in network_data.values()]
    if len(set(lengths)) > 1:
        raise ValueError("Networks don't align across strategies")
    
    # Prepare data for Friedman test
    data_matrix = np.column_stack([network_data[s] for s in strategies])
    
    # Friedman test
    statistic, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(3)])
    
    # Create DataFrame for reporting
    data_df = pd.DataFrame(network_data)
    data_df['network'] = all_results[f'{scope}_multinomial']['network_metrics']['network'].values
    
    return statistic, p_value, data_df


def paired_ttest_hemispheres(
    all_results: Dict[str, Dict],
    strategy: str = 'multinomial'
) -> Tuple[float, float]:
    """
    Paired t-test comparing left vs right hemisphere.
    
    H0: No difference between hemispheres
    H1: Hemispheres differ
    
    Parameters
    ----------
    all_results : dict
        Results from all models
    strategy : str
        Strategy to compare
    
    Returns
    -------
    t_statistic : float
        T-test statistic
    p_value : float
        P-value
    """
    
    left_model = f'left_{strategy}'
    right_model = f'right_{strategy}'
    
    # Get network-level accuracies
    left_acc = all_results[left_model]['network_metrics']['accuracy'].values
    right_acc = all_results[right_model]['network_metrics']['accuracy'].values
    
    # Paired t-test
    t_statistic, p_value = stats.ttest_rel(left_acc, right_acc)
    
    return t_statistic, p_value


def anova_network_reorganization(
    reorganization_df: pd.DataFrame
) -> Tuple[float, float]:
    """
    One-way ANOVA testing if reorganization differs across networks.
    
    H0: All networks reorganize equally
    H1: At least one network differs
    
    Parameters
    ----------
    reorganization_df : pd.DataFrame
        DataFrame with 'network' and 'reorganization_index' columns
    
    Returns
    -------
    f_statistic : float
        F-statistic
    p_value : float
        P-value
    """
    
    # Group by network
    groups = []
    networks = reorganization_df['network'].unique()
    
    for network in networks:
        network_data = reorganization_df[
            reorganization_df['network'] == network
        ]['reorganization_index'].values
        groups.append(network_data)
    
    # One-way ANOVA
    f_statistic, p_value = stats.f_oneway(*groups)
    
    return f_statistic, p_value


def chi_square_error_types(
    error_summaries: Dict[str, Dict]
) -> Tuple[float, float, pd.DataFrame]:
    """
    Chi-square test for independence of error types across models.
    
    H0: Error type distribution is independent of model
    H1: Error types depend on model
    
    Parameters
    ----------
    error_summaries : dict
        Dictionary mapping model names to error type summaries
    
    Returns
    -------
    chi2_statistic : float
        Chi-square statistic
    p_value : float
        P-value
    contingency_table : pd.DataFrame
        Contingency table used
    """
    
    # Build contingency table
    models = list(error_summaries.keys())
    error_types = ['within_network', 'cross_network', 'within_hemisphere', 'cross_hemisphere']
    
    table_data = []
    
    for model in models:
        row = [
            error_summaries[model]['within_network_count'],
            error_summaries[model]['cross_network_count'],
            error_summaries[model]['within_hemisphere_count'],
            error_summaries[model]['cross_hemisphere_count']
        ]
        table_data.append(row)
    
    contingency_table = pd.DataFrame(
        table_data,
        index=models,
        columns=error_types
    )
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return chi2, p_value, contingency_table


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply FDR correction for multiple comparisons.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
    alpha : float
        Significance level
    
    Returns
    -------
    p_corrected : np.ndarray
        FDR-corrected p-values
    """
    
    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    return p_corrected


def permutation_test_reorganization(
    reorganization_scores: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42
) -> float:
    """
    Permutation test for significance of reorganization scores.
    
    H0: Reorganization scores are random
    H1: Scores are significantly different from random
    
    Parameters
    ----------
    reorganization_scores : np.ndarray
        Observed reorganization scores
    n_permutations : int
        Number of permutations
    random_state : int
        Random seed
    
    Returns
    -------
    p_value : float
        Permutation test p-value
    """
    
    rng = np.random.RandomState(random_state)
    
    # Observed mean
    observed_mean = np.mean(reorganization_scores)
    
    # Permutation distribution
    permuted_means = []
    
    for _ in range(n_permutations):
        permuted = rng.permutation(reorganization_scores)
        permuted_means.append(np.mean(permuted))
    
    permuted_means = np.array(permuted_means)
    
    # P-value: proportion of permuted means >= observed
    p_value = np.sum(permuted_means >= observed_mean) / n_permutations
    
    return p_value