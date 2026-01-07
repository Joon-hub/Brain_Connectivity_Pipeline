"""
Analyze overall performance metrics.

Generates:
- Table 1: Overall performance summary
- Table 2: Network-level performance
- Basic comparison plots

Usage:
    python analysis/02_performance_analysis.py
"""

import sys
from pathlib import Path
import pickle
import logging

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.analysis.comparisons import (
    create_performance_summary_table,
    create_network_performance_table,
    compare_strategies,
    compare_hemispheres
)
from src.analysis.visualizations import plot_performance_overview

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run performance analysis."""
    
    logger.info("="*80)
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("="*80)
    
    # Load compiled results
    compiled_path = project_root / 'outputs' / 'compiled' / 'all_models_results.pkl'
    
    with open(compiled_path, 'rb') as f:
        all_results = pickle.load(f)
    
    logger.info(f"Loaded {len(all_results)} models")
    
    # Create output directories
    tables_dir = project_root / 'outputs' / 'tables'
    figures_dir = project_root / 'outputs' / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Overall performance
    logger.info("\nGenerating Table 1: Overall Performance Summary...")
    summary_df = create_performance_summary_table(all_results)
    summary_df.to_csv(tables_dir / 'table1_overall_performance.csv', index=False)
    logger.info(f"✓ Saved to: {tables_dir}/table1_overall_performance.csv")
    
    # Table 2: Network performance
    logger.info("\nGenerating Table 2: Network-Level Performance...")
    network_df = create_network_performance_table(all_results)
    network_df.to_csv(tables_dir / 'table2_network_performance.csv')
    logger.info(f"✓ Saved to: {tables_dir}/table2_network_performance.csv")
    
    # Strategy comparison
    logger.info("\nComparing strategies...")
    for scope in ['full', 'left', 'right']:
        strategy_comp = compare_strategies(all_results, scope=scope)
        strategy_comp.to_csv(tables_dir / f'strategy_comparison_{scope}.csv', index=False)
        logger.info(f"✓ {scope.capitalize()} scope comparison saved")
    
    # Hemisphere comparison
    logger.info("\nComparing hemispheres...")
    for strategy in ['multinomial', 'ovr', 'ovo']:
        hem_comp = compare_hemispheres(all_results, strategy=strategy)
        hem_comp.to_csv(tables_dir / f'hemisphere_comparison_{strategy}.csv', index=False)
        logger.info(f"✓ {strategy.capitalize()} strategy comparison saved")
    
    # Generate Figure 1
    logger.info("\nGenerating Figure 1: Performance Overview...")
    plot_performance_overview(
        summary_df,
        save_path=figures_dir / 'fig1_performance_overview.png'
    )
    
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()