"""
Analyze error patterns across models.

Generates:
- Error type classifications
- Confusion matrix analysis
- Cross-model agreement

Usage:
    python analysis/03_error_analysis.py
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import logging

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.analysis.metrics import (
    classify_error_types,
    compute_error_type_summary,
    compute_network_confusion_matrix
)
from src.analysis.comparisons import compute_cross_model_agreement
from src.analysis.visualizations import plot_confusion_matrices_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run error analysis."""
    
    logger.info("="*80)
    logger.info("ERROR PATTERN ANALYSIS")
    logger.info("="*80)
    
    # Load results
    compiled_path = project_root / 'outputs' / 'compiled' / 'all_models_results.pkl'
    
    with open(compiled_path, 'rb') as f:
        all_results = pickle.load(f)
    
    # Create output directory
    output_dir = project_root / 'outputs' / 'compiled'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze error types for each model
    logger.info("\nClassifying error types...")
    
    error_summaries = {}
    
    for model_name, results in all_results.items():
        logger.info(f"Processing {model_name}...")
        
        y_true = results['true_labels']
        y_pred = results['predictions']
        region_info = results['per_region_metrics'][['region_name', 'network']].copy()
        region_info['region_idx'] = range(len(region_info))
        
        # Classify errors
        error_df = classify_error_types(y_true, y_pred, region_info)
        
        # Summarize
        error_summary = compute_error_type_summary(error_df)
        error_summaries[model_name] = error_summary
        
        # Save detailed errors
        error_df.to_csv(
            output_dir / f'{model_name}_error_types.csv',
            index=False
        )
    
    # Save error summaries
    error_summary_df = pd.DataFrame(error_summaries).T
    error_summary_df.to_csv(output_dir / 'error_type_summaries.csv')
    
    logger.info(f"✓ Error classifications saved to: {output_dir}")
    
    # Cross-model agreement
    logger.info("\nComputing cross-model agreement...")
    agreement_matrix = compute_cross_model_agreement(all_results)
    agreement_matrix.to_csv(output_dir / 'cross_model_agreement.csv')
    logger.info(f"✓ Agreement matrix saved")
    
    # Generate confusion matrix grid
    logger.info("\nGenerating confusion matrix visualization...")
    figures_dir = project_root / 'outputs' / 'figures'
    plot_confusion_matrices_grid(
        all_results,
        save_path=figures_dir / 'fig4_confusion_matrices.png'
    )
    
    logger.info("\n" + "="*80)
    logger.info("ERROR ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()