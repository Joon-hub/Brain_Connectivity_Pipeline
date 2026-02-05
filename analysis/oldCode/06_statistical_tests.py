"""
Run all statistical tests.

Tests:
1. Friedman test: Strategy comparison
2. Paired t-test: Hemisphere comparison
3. ANOVA: Network reorganization
4. Chi-square: Error type distribution

Usage:
    python analysis/06_statistical_tests.py
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import logging

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.analysis.statistics import (
    friedman_test_strategies,
    paired_ttest_hemispheres,
    anova_network_reorganization,
    chi_square_error_types,
    fdr_correction
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run statistical tests."""
    
    logger.info("="*80)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("="*80)
    
    # Load results
    compiled_path = project_root / 'outputs' / 'compiled' / 'all_models_results.pkl'
    
    with open(compiled_path, 'rb') as f:
        all_results = pickle.load(f)
    
    output_dir = project_root / 'outputs' / 'tables'
    
    # Collect all test results
    test_results = []
    
    # Test 1: Friedman test for strategies
    logger.info("\n1. Friedman Test: Strategy Comparison")
    
    for scope in ['full', 'left', 'right']:
        statistic, p_value, data_df = friedman_test_strategies(all_results, scope=scope)
        
        logger.info(f"  {scope.capitalize()} scope:")
        logger.info(f"    Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
        
        test_results.append({
            'test': 'Friedman',
            'comparison': f'Strategies ({scope})',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    # Test 2: Paired t-test for hemispheres
    logger.info("\n2. Paired T-Test: Hemisphere Comparison")
    
    for strategy in ['multinomial', 'ovr', 'ovo']:
        t_stat, p_value = paired_ttest_hemispheres(all_results, strategy=strategy)
        
        logger.info(f"  {strategy.capitalize()} strategy:")
        logger.info(f"    T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        
        test_results.append({
            'test': 'Paired t-test',
            'comparison': f'Hemispheres ({strategy})',
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    # Test 3: ANOVA for network reorganization
    logger.info("\n3. ANOVA: Network Reorganization")
    
    # Load reorganization data
    reorg_df = pd.read_csv(
        project_root / 'outputs' / 'compiled' / 'full_multinomial_reorganization.csv'
    )
    
    f_stat, p_value = anova_network_reorganization(reorg_df)
    
    logger.info(f"  F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
    
    test_results.append({
        'test': 'One-way ANOVA',
        'comparison': 'Network reorganization',
        'statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    })
    
    # Test 4: Chi-square for error types
    logger.info("\n4. Chi-Square: Error Type Distribution")
    
    error_summaries_df = pd.read_csv(
        project_root / 'outputs' / 'compiled' / 'error_type_summaries.csv',
        index_col=0
    )
    
    error_summaries = error_summaries_df.to_dict('index')
    
    chi2, p_value, contingency = chi_square_error_types(error_summaries)
    
    logger.info(f"  Chi-square: {chi2:.4f}, p-value: {p_value:.4f}")
    
    test_results.append({
        'test': 'Chi-square',
        'comparison': 'Error types across models',
        'statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    })
    
    # FDR correction
    logger.info("\n5. FDR Correction")
    
    test_results_df = pd.DataFrame(test_results)
    p_values = test_results_df['p_value'].values
    p_corrected = fdr_correction(p_values)
    test_results_df['p_corrected'] = p_corrected
    test_results_df['significant_corrected'] = p_corrected < 0.05
    
    # Save results
    test_results_df.to_csv(output_dir / 'table3_statistical_tests.csv', index=False)
    
    logger.info(f"\n✓ Statistical test results saved to: {output_dir}/table3_statistical_tests.csv")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL SUMMARY")
    logger.info("="*80)
    
    print(test_results_df.to_string())
    
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()