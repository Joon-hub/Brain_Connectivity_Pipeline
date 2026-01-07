"""
Analyze functional reorganization patterns.

Generates:
- Region-level reorganization indices
- Network-level reorganization
- Reorganization maps

Usage:
    python analysis/04_reorganization_analysis.py
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import logging

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.analysis.metrics import compute_per_region_reorganization
from src.analysis.visualizations import plot_reorganization_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run reorganization analysis."""
    
    logger.info("="*80)
    logger.info("REORGANIZATION ANALYSIS")
    logger.info("="*80)
    
    # Load results
    compiled_path = project_root / 'outputs' / 'compiled' / 'all_models_results.pkl'
    
    with open(compiled_path, 'rb') as f:
        all_results = pickle.load(f)
    
    output_dir = project_root / 'outputs' / 'compiled'
    
    # Analyze for each model
    logger.info("\nComputing reorganization indices...")
    
    reorganization_dfs = {}
    
    for model_name, results in all_results.items():
        logger.info(f"Processing {model_name}...")
        
        # Note: You'll need CV per-region metrics too
        # This is a simplified version
        per_region_task = results['per_region_metrics']
        
        # For now, use task accuracy and assume rest accuracy from summary
        rest_acc = results['summary']['rest_train_accuracy']
        
        per_region_task['rest_accuracy'] = rest_acc  # Simplified
        per_region_task['reorganization_index'] = (
            1 - (per_region_task['accuracy'] / per_region_task['rest_accuracy'])
        )
        
        reorganization_dfs[model_name] = per_region_task
        
        # Save
        per_region_task.to_csv(
            output_dir / f'{model_name}_reorganization.csv',
            index=False
        )
    
    logger.info(f"✓ Reorganization indices saved to: {output_dir}")
    
    # Generate reorganization map for full multinomial
    logger.info("\nGenerating reorganization map...")
    figures_dir = project_root / 'outputs' / 'figures'
    
    plot_reorganization_map(
        reorganization_dfs['full_multinomial'],
        save_path=figures_dir / 'fig3_reorganization_map.png'
    )
    
    # Network-level aggregation
    logger.info("\nAggregating at network level...")
    
    for model_name, reorg_df in reorganization_dfs.items():
        network_reorg = reorg_df.groupby('network').agg({
            'reorganization_index': ['mean', 'std', 'min', 'max'],
            'accuracy': 'mean'
        }).reset_index()
        
        network_reorg.to_csv(
            output_dir / f'{model_name}_network_reorganization.csv',
            index=False
        )
    
    logger.info("\n" + "="*80)
    logger.info("REORGANIZATION ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()