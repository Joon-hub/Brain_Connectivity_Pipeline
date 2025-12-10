"""
05_compare_strategies.py

Compare classification performance across different strategies:
- Multinomial logistic regression (baseline)
- One-vs-Rest (OvR) - region discriminability
- One-vs-One (OvO) - pairwise confusability (if available)

Generates comprehensive comparison tables, figures, and statistical analyses.

Usage:
    python scripts/hemisphere/05_compare_strategies.py --hemisphere left
    python scripts/hemisphere/05_compare_strategies.py --hemisphere right
    python scripts/hemisphere/05_compare_strategies.py --hemisphere both

Author: Joon
Date: 2024
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging configuration."""
    log_file = output_dir / "comparison_analysis.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare classification strategies across hemispheres'
    )
    
    parser.add_argument(
        '--hemisphere',
        type=str,
        required=True,
        choices=['left', 'right', 'both'],
        help='Which hemisphere to analyze (left, right, or both)'
    )
    
    parser.add_argument(
        '--results_dir',
        type=Path,
        default=project_root / 'data' / 'results' / 'hemisphere_analysis',
        help='Directory containing analysis results'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Output directory (default: results_dir/comparison)'
    )
    
    parser.add_argument(
        '--include_ovo',
        action='store_true',
        help='Include OvO analysis if available'
    )
    
    parser.add_argument(
        '--significance_level',
        type=float,
        default=0.05,
        help='Significance level for statistical tests'
    )
    
    return parser.parse_args()


def load_strategy_results(
    results_dir: Path,
    hemisphere: str,
    strategy: str,
    logger: logging.Logger
) -> Optional[Dict]:
    """
    Load results for a specific strategy and hemisphere.
    
    Parameters
    ----------
    results_dir : Path
        Base results directory
    hemisphere : str
        'left' or 'right'
    strategy : str
        'multinomial', 'ovr', or 'ovo'
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    results : dict or None
        Dictionary containing loaded results, or None if not found
    """
    
    strategy_dir = results_dir / f"{hemisphere}_hemisphere" / strategy
    
    if not strategy_dir.exists():
        logger.warning(f"  {strategy.upper()} results not found: {strategy_dir}")
        return None
    
    logger.info(f"  Loading {strategy.upper()} results from: {strategy_dir}")
    
    results = {'strategy': strategy, 'hemisphere': hemisphere}
    
    try:
        # Load overall metrics
        metrics_file = strategy_dir / f"{strategy}_overall_metrics.json" if strategy != 'multinomial' else strategy_dir / "overall_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                results['overall_metrics'] = json.load(f)
        
        # Load per-region metrics
        if strategy == 'multinomial':
            per_region_file = strategy_dir / "per_region_metrics.csv"
        elif strategy == 'ovr':
            per_region_file = strategy_dir / "ovr_per_region_metrics.csv"
        else:  # ovo
            per_region_file = strategy_dir / "ovo_pair_summary.csv"
        
        if per_region_file.exists():
            results['per_region_metrics'] = pd.read_csv(per_region_file)
        
        # Load predictions
        if strategy == 'multinomial':
            pred_file = strategy_dir / "cv_predictions.npy"
        else:
            pred_file = strategy_dir / f"{strategy}_predictions.npy"
        
        if pred_file.exists():
            results['predictions'] = np.load(pred_file)
        
        # Load true labels
        if strategy == 'multinomial':
            labels_file = strategy_dir / "cv_true_labels.npy"
        else:
            labels_file = strategy_dir / f"{strategy}_true_labels.npy"
        
        if labels_file.exists():
            results['true_labels'] = np.load(labels_file)
        
        # Load confusion matrix (multinomial only)
        if strategy == 'multinomial':
            cm_file = strategy_dir / "confusion_matrix.npy"
            if cm_file.exists():
                results['confusion_matrix'] = np.load(cm_file)
        
        logger.info(f"    Successfully loaded {strategy.upper()} results")
        
    except Exception as e:
        logger.error(f"    Error loading {strategy.upper()} results: {e}")
        return None
    
    return results


def compare_overall_performance(
    multinomial_results: Dict,
    ovr_results: Dict,
    ovo_results: Optional[Dict],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compare overall performance metrics across strategies.
    
    Parameters
    ----------
    multinomial_results : dict
        Multinomial results
    ovr_results : dict
        OvR results
    ovo_results : dict, optional
        OvO results
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table
    """
    
    logger.info("\nComparing overall performance metrics...")
    
    # Extract metrics
    metrics_to_compare = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
    
    comparison_data = []
    
    # Multinomial
    multi_metrics = multinomial_results['overall_metrics']
    for metric in metrics_to_compare:
        if metric in multi_metrics:
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Multinomial': multi_metrics[metric]
            })
    
    # OvR
    ovr_metrics = ovr_results['overall_metrics']
    for i, item in enumerate(comparison_data):
        metric_key = metrics_to_compare[i]
        if metric_key in ovr_metrics:
            item['OvR'] = ovr_metrics[metric_key]
    
    # OvO (if available)
    if ovo_results is not None and 'overall_metrics' in ovo_results:
        ovo_metrics = ovo_results['overall_metrics']
        for i, item in enumerate(comparison_data):
            metric_key = metrics_to_compare[i]
            if metric_key in ovo_metrics:
                item['OvO'] = ovo_metrics[metric_key]
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate differences
    comparison_df['OvR - Multinomial'] = comparison_df['OvR'] - comparison_df['Multinomial']
    
    if 'OvO' in comparison_df.columns:
        comparison_df['OvO - Multinomial'] = comparison_df['OvO'] - comparison_df['Multinomial']
    
    logger.info("\nOverall Performance Comparison:")
    logger.info("\n" + comparison_df.to_string(index=False))
    
    return comparison_df


def compare_per_region_performance(
    multinomial_results: Dict,
    ovr_results: Dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compare per-region performance between multinomial and OvR.
    
    Parameters
    ----------
    multinomial_results : dict
        Multinomial results
    ovr_results : dict
        OvR results
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Per-region comparison
    """
    
    logger.info("\nComparing per-region performance...")
    
    # Get per-region metrics
    multi_per_region = multinomial_results['per_region_metrics'].copy()
    ovr_per_region = ovr_results['per_region_metrics'].copy()
    
    # Merge on region_id
    comparison = pd.merge(
        multi_per_region[['region_id', 'region_name', 'network', 'accuracy']],
        ovr_per_region[['region_id', 'mean_accuracy']],
        on='region_id',
        how='inner',
        suffixes=('_multinomial', '_ovr')
    )
    
    # Rename OvR column
    comparison = comparison.rename(columns={'mean_accuracy': 'accuracy_ovr'})
    
    # Calculate difference
    comparison['ovr_advantage'] = comparison['accuracy_ovr'] - comparison['accuracy_multinomial']
    
    # Correlation
    corr, p_value = stats.pearsonr(
        comparison['accuracy_multinomial'],
        comparison['accuracy_ovr']
    )
    
    logger.info(f"  Pearson correlation: r = {corr:.4f}, p = {p_value:.4f}")
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(
        comparison['accuracy_ovr'],
        comparison['accuracy_multinomial']
    )
    
    logger.info(f"  Paired t-test: t = {t_stat:.4f}, p = {t_pvalue:.4f}")
    
    # Summary statistics
    logger.info(f"  Mean multinomial accuracy: {comparison['accuracy_multinomial'].mean():.4f}")
    logger.info(f"  Mean OvR accuracy: {comparison['accuracy_ovr'].mean():.4f}")
    logger.info(f"  Mean OvR advantage: {comparison['ovr_advantage'].mean():.4f}")
    
    return comparison


def analyze_error_patterns(
    multinomial_results: Dict,
    ovr_results: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Analyze error pattern differences between strategies.
    
    Parameters
    ----------
    multinomial_results : dict
        Multinomial results
    ovr_results : dict
        OvR results
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    error_analysis : dict
        Error pattern analysis results
    """
    
    logger.info("\nAnalyzing error patterns...")
    
    # Get predictions
    y_true = multinomial_results['true_labels']
    y_pred_multi = multinomial_results['predictions']
    y_pred_ovr = ovr_results['predictions']
    
    # Identify errors
    errors_multi = y_true != y_pred_multi
    errors_ovr = y_true != y_pred_ovr
    
    # Error counts
    n_errors_multi = np.sum(errors_multi)
    n_errors_ovr = np.sum(errors_ovr)
    
    logger.info(f"  Multinomial errors: {n_errors_multi} ({n_errors_multi/len(y_true)*100:.2f}%)")
    logger.info(f"  OvR errors: {n_errors_ovr} ({n_errors_ovr/len(y_true)*100:.2f}%)")
    
    # Error overlap analysis
    errors_both = errors_multi & errors_ovr
    errors_only_multi = errors_multi & ~errors_ovr
    errors_only_ovr = ~errors_multi & errors_ovr
    correct_both = ~errors_multi & ~errors_ovr
    
    n_errors_both = np.sum(errors_both)
    n_errors_only_multi = np.sum(errors_only_multi)
    n_errors_only_ovr = np.sum(errors_only_ovr)
    n_correct_both = np.sum(correct_both)
    
    logger.info(f"\nError Overlap:")
    logger.info(f"  Errors in both: {n_errors_both} ({n_errors_both/len(y_true)*100:.2f}%)")
    logger.info(f"  Errors only in multinomial: {n_errors_only_multi} ({n_errors_only_multi/len(y_true)*100:.2f}%)")
    logger.info(f"  Errors only in OvR: {n_errors_only_ovr} ({n_errors_only_ovr/len(y_true)*100:.2f}%)")
    logger.info(f"  Correct in both: {n_correct_both} ({n_correct_both/len(y_true)*100:.2f}%)")
    
    # McNemar's test
    # Contingency table: [[both_correct, ovr_error_multi_correct], [multi_error_ovr_correct, both_error]]
    contingency = np.array([
        [n_correct_both, n_errors_only_ovr],
        [n_errors_only_multi, n_errors_both]
    ])
    
    from statsmodels.stats.contingency_tables import mcnemar
    result = mcnemar(contingency, exact=True)
    
    logger.info(f"\nMcNemar's Test:")
    logger.info(f"  Statistic: {result.statistic:.4f}")
    logger.info(f"  p-value: {result.pvalue:.4f}")
    
    error_analysis = {
        'n_errors_multi': int(n_errors_multi),
        'n_errors_ovr': int(n_errors_ovr),
        'n_errors_both': int(n_errors_both),
        'n_errors_only_multi': int(n_errors_only_multi),
        'n_errors_only_ovr': int(n_errors_only_ovr),
        'n_correct_both': int(n_correct_both),
        'mcnemar_statistic': float(result.statistic),
        'mcnemar_pvalue': float(result.pvalue),
        'contingency_table': contingency.tolist()
    }
    
    return error_analysis


def create_comparison_visualizations(
    multinomial_results: Dict,
    ovr_results: Dict,
    per_region_comparison: pd.DataFrame,
    error_analysis: Dict,
    output_dir: Path,
    hemisphere: str,
    logger: logging.Logger
):
    """
    Create comprehensive comparison visualizations.
    
    Parameters
    ----------
    multinomial_results : dict
        Multinomial results
    ovr_results : dict
        OvR results
    per_region_comparison : pd.DataFrame
        Per-region comparison data
    error_analysis : dict
        Error analysis results
    output_dir : Path
        Output directory
    hemisphere : str
        Hemisphere name
    logger : logging.Logger
        Logger instance
    """
    
    logger.info("\nCreating comparison visualizations...")
    
    # 1. Overall metrics comparison bar chart
    logger.info("  Creating overall metrics comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score']
    metric_labels = ['Accuracy', 'Balanced Accuracy', 'F1 Score']
    
    multi_metrics = multinomial_results['overall_metrics']
    ovr_metrics = ovr_results['overall_metrics']
    
    multi_values = [multi_metrics.get(m, 0) for m in metrics]
    ovr_values = [ovr_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, multi_values, width, label='Multinomial',
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, ovr_values, width, label='One-vs-Rest',
                   color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_title(f'{hemisphere.capitalize()} Hemisphere - Strategy Comparison',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim([0.8, 1.0])
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-region scatter plot
    logger.info("  Creating per-region scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color by network if available
    if 'network' in per_region_comparison.columns:
        networks = per_region_comparison['network'].unique()
        network_colors = dict(zip(networks, plt.cm.tab10(np.linspace(0, 1, len(networks)))))
        colors = [network_colors[net] for net in per_region_comparison['network']]
        
        # Create legend
        handles = [mpatches.Patch(color=network_colors[net], label=net) for net in networks]
        ax.legend(handles=handles, title='Network', loc='lower right', frameon=True)
    else:
        colors = 'steelblue'
    
    ax.scatter(per_region_comparison['accuracy_multinomial'],
               per_region_comparison['accuracy_ovr'],
               c=colors, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Agreement')
    
    # Correlation
    corr, p_value = stats.pearsonr(
        per_region_comparison['accuracy_multinomial'],
        per_region_comparison['accuracy_ovr']
    )
    
    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontweight='bold')
    
    ax.set_xlabel('Multinomial Accuracy', fontweight='bold')
    ax.set_ylabel('One-vs-Rest Accuracy', fontweight='bold')
    ax.set_title(f'{hemisphere.capitalize()} Hemisphere - Per-Region Performance Correlation',
                 fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_region_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. OvR advantage distribution
    logger.info("  Creating OvR advantage distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(per_region_comparison['ovr_advantage'], bins=30, 
            color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
    ax.axvline(per_region_comparison['ovr_advantage'].mean(), color='blue',
               linestyle='--', linewidth=2, 
               label=f"Mean: {per_region_comparison['ovr_advantage'].mean():.4f}")
    
    ax.set_xlabel('OvR Accuracy - Multinomial Accuracy', fontweight='bold')
    ax.set_ylabel('Number of Regions', fontweight='bold')
    ax.set_title(f'{hemisphere.capitalize()} Hemisphere - OvR Performance Advantage',
                 fontweight='bold', pad=20)
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ovr_advantage_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Error overlap Venn diagram (as bar chart)
    logger.info("  Creating error overlap visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = [
        'Correct in Both',
        'Error Only\nin Multinomial',
        'Error Only\nin OvR',
        'Error in Both'
    ]
    
    counts = [
        error_analysis['n_correct_both'],
        error_analysis['n_errors_only_multi'],
        error_analysis['n_errors_only_ovr'],
        error_analysis['n_errors_both']
    ]
    
    colors_venn = ['lightgreen', 'lightcoral', 'lightblue', 'orange']
    
    bars = ax.bar(categories, counts, color=colors_venn, alpha=0.7,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        total = sum(counts)
        percentage = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
               f'{count}\n({percentage:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title(f'{hemisphere.capitalize()} Hemisphere - Error Pattern Overlap',
                 fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add McNemar test result
    ax.text(0.98, 0.98,
            f"McNemar's test: p = {error_analysis['mcnemar_pvalue']:.4f}",
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_overlap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Top regions with largest OvR advantage
    logger.info("  Creating top OvR advantage regions...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top 20 regions with largest OvR advantage
    top_regions = per_region_comparison.nlargest(20, 'ovr_advantage')
    
    y_pos = np.arange(len(top_regions))
    colors_advantage = ['green' if x > 0 else 'red' for x in top_regions['ovr_advantage']]
    
    ax.barh(y_pos, top_regions['ovr_advantage'], color=colors_advantage,
            alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_regions['region_name'], fontsize=8)
    ax.set_xlabel('OvR Advantage (OvR - Multinomial)', fontweight='bold')
    ax.set_title(f'{hemisphere.capitalize()} Hemisphere - Top 20 Regions by OvR Advantage',
                 fontweight='bold', pad=20)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_ovr_advantage_regions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Network-level comparison (if available)
    if 'network' in per_region_comparison.columns:
        logger.info("  Creating network-level comparison...")
        
        network_comparison = per_region_comparison.groupby('network').agg({
            'accuracy_multinomial': 'mean',
            'accuracy_ovr': 'mean',
            'ovr_advantage': 'mean'
        }).reset_index()
        
        network_comparison = network_comparison.sort_values('ovr_advantage', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(network_comparison))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, network_comparison['accuracy_multinomial'],
                      width, label='Multinomial', color='steelblue',
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, network_comparison['accuracy_ovr'],
                      width, label='One-vs-Rest', color='coral',
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(network_comparison['network'], rotation=45, ha='right')
        ax.set_ylabel('Mean Accuracy', fontweight='bold')
        ax.set_xlabel('Functional Network', fontweight='bold')
        ax.set_title(f'{hemisphere.capitalize()} Hemisphere - Network-Level Strategy Comparison',
                     fontweight='bold', pad=20)
        ax.legend(loc='lower right', frameon=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'network_level_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("  All visualizations created successfully")


def compare_single_hemisphere(
    hemisphere: str,
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict:
    """
    Compare strategies for a single hemisphere.
    
    Parameters
    ----------
    hemisphere : str
        'left' or 'right'
    args : argparse.Namespace
        Command line arguments
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    comparison_results : dict
        Dictionary containing all comparison results
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPARING STRATEGIES - {hemisphere.upper()} HEMISPHERE")
    logger.info(f"{'='*80}\n")
    
    # Set up output directory
    if args.output_dir is None:
        output_dir = args.results_dir / "comparison" / f"{hemisphere}_hemisphere"
    else:
        output_dir = args.output_dir / f"{hemisphere}_hemisphere"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results for each strategy
    logger.info("Loading strategy results...")
    
    multinomial_results = load_strategy_results(
        args.results_dir, hemisphere, 'multinomial', logger
    )
    
    ovr_results = load_strategy_results(
        args.results_dir, hemisphere, 'ovr', logger
    )
    
    ovo_results = None
    if args.include_ovo:
        ovo_results = load_strategy_results(
            args.results_dir, hemisphere, 'ovo', logger
        )
    
    # Check if we have minimum required results
    if multinomial_results is None:
        logger.error("Multinomial results not found. Cannot perform comparison.")
        return None
    
    if ovr_results is None:
        logger.error("OvR results not found. Cannot perform comparison.")
        return None
    
    # Overall performance comparison
    overall_comparison = compare_overall_performance(
        multinomial_results, ovr_results, ovo_results, logger
    )
    
    # Per-region performance comparison
    per_region_comparison = compare_per_region_performance(
        multinomial_results, ovr_results, logger
    )
    
    # Error pattern analysis
    error_analysis = analyze_error_patterns(
        multinomial_results, ovr_results, logger
    )
    
    # Save comparison results
    logger.info("\nSaving comparison results...")
    
    # Save overall comparison
    overall_comparison.to_csv(
        output_dir / 'overall_strategy_comparison.csv',
        index=False
    )
    
    # Save per-region comparison
    per_region_comparison.to_csv(
        output_dir / 'per_region_strategy_comparison.csv',
        index=False
    )
    
    # Save error analysis
    with open(output_dir / 'error_pattern_analysis.json', 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    # Create visualizations
    create_comparison_visualizations(
        multinomial_results,
        ovr_results,
        per_region_comparison,
        error_analysis,
        output_dir,
        hemisphere,
        logger
    )
    
    # Generate summary report
    logger.info("\nGenerating summary report...")
    create_summary_report(
        hemisphere,
        overall_comparison,
        per_region_comparison,
        error_analysis,
        output_dir,
        logger
    )
    
    logger.info(f"\nAll comparison results saved to: {output_dir}")
    
    comparison_results = {
        'hemisphere': hemisphere,
        'overall_comparison': overall_comparison,
        'per_region_comparison': per_region_comparison,
        'error_analysis': error_analysis,
        'output_dir': output_dir
    }
    
    return comparison_results


def create_summary_report(
    hemisphere: str,
    overall_comparison: pd.DataFrame,
    per_region_comparison: pd.DataFrame,
    error_analysis: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Create a text summary report of the comparison.
    
    Parameters
    ----------
    hemisphere : str
        Hemisphere name
    overall_comparison : pd.DataFrame
        Overall metrics comparison
    per_region_comparison : pd.DataFrame
        Per-region comparison
    error_analysis : dict
        Error analysis results
    output_dir : Path
        Output directory
    logger : logging.Logger
        Logger instance
    """
    
    report_file = output_dir / 'comparison_summary_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"STRATEGY COMPARISON SUMMARY REPORT\n")
        f.write(f"{hemisphere.upper()} HEMISPHERE\n")
        f.write("="*80 + "\n\n")
        
        # Overall metrics
        f.write("1. OVERALL PERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(overall_comparison.to_string(index=False))
        f.write("\n\n")
        
        # Key findings
        f.write("2. KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        
        multi_acc = overall_comparison[overall_comparison['Metric'] == 'Accuracy']['Multinomial'].values[0]
        ovr_acc = overall_comparison[overall_comparison['Metric'] == 'Accuracy']['OvR'].values[0]
        diff = ovr_acc - multi_acc
        
        f.write(f"Overall Accuracy:\n")
        f.write(f"  - Multinomial: {multi_acc:.4f}\n")
        f.write(f"  - One-vs-Rest: {ovr_acc:.4f}\n")
        f.write(f"  - Difference: {diff:+.4f}\n\n")
        
        # Per-region statistics
        f.write("Per-Region Analysis:\n")
        f.write(f"  - Regions analyzed: {len(per_region_comparison)}\n")
        f.write(f"  - Mean multinomial accuracy: {per_region_comparison['accuracy_multinomial'].mean():.4f}\n")
        f.write(f"  - Mean OvR accuracy: {per_region_comparison['accuracy_ovr'].mean():.4f}\n")
        f.write(f"  - Mean OvR advantage: {per_region_comparison['ovr_advantage'].mean():.4f}\n")
        
        # Correlation
        corr, p_value = stats.pearsonr(
            per_region_comparison['accuracy_multinomial'],
            per_region_comparison['accuracy_ovr']
        )
        f.write(f"  - Correlation: r = {corr:.4f}, p = {p_value:.4f}\n\n")
        
        # Error patterns
        f.write("3. ERROR PATTERN ANALYSIS\n")
        f.write("-"*80 + "\n")
        
        total_samples = sum([
            error_analysis['n_correct_both'],
            error_analysis['n_errors_only_multi'],
            error_analysis['n_errors_only_ovr'],
            error_analysis['n_errors_both']
        ])
        
        f.write(f"Total samples: {total_samples}\n\n")
        f.write(f"Error Distribution:\n")
        f.write(f"  - Correct in both: {error_analysis['n_correct_both']} "
               f"({error_analysis['n_correct_both']/total_samples*100:.2f}%)\n")
        f.write(f"  - Error only in Multinomial: {error_analysis['n_errors_only_multi']} "
               f"({error_analysis['n_errors_only_multi']/total_samples*100:.2f}%)\n")
        f.write(f"  - Error only in OvR: {error_analysis['n_errors_only_ovr']} "
               f"({error_analysis['n_errors_only_ovr']/total_samples*100:.2f}%)\n")
        f.write(f"  - Error in both: {error_analysis['n_errors_both']} "
               f"({error_analysis['n_errors_both']/total_samples*100:.2f}%)\n\n")
        
        f.write(f"McNemar's Test (Strategy Difference):\n")
        f.write(f"  - Statistic: {error_analysis['mcnemar_statistic']:.4f}\n")
        f.write(f"  - p-value: {error_analysis['mcnemar_pvalue']:.4f}\n")
        
        if error_analysis['mcnemar_pvalue'] < 0.05:
            f.write(f"  - Result: Strategies differ significantly (p < 0.05)\n")
        else:
            f.write(f"  - Result: No significant difference between strategies\n")
        
        f.write("\n")
        
        # Top regions with OvR advantage
        f.write("4. TOP REGIONS WITH OVR ADVANTAGE\n")
        f.write("-"*80 + "\n")
        
        top_10 = per_region_comparison.nlargest(10, 'ovr_advantage')
        for idx, row in top_10.iterrows():
            f.write(f"{row['region_name']}: {row['ovr_advantage']:+.4f} "
                   f"(Multi: {row['accuracy_multinomial']:.4f}, OvR: {row['accuracy_ovr']:.4f})\n")
        
        f.write("\n")
        
        # Bottom regions (OvR disadvantage)
        f.write("5. TOP REGIONS WITH OVR DISADVANTAGE\n")
        f.write("-"*80 + "\n")
        
        bottom_10 = per_region_comparison.nsmallest(10, 'ovr_advantage')
        for idx, row in bottom_10.iterrows():
            f.write(f"{row['region_name']}: {row['ovr_advantage']:+.4f} "
                   f"(Multi: {row['accuracy_multinomial']:.4f}, OvR: {row['accuracy_ovr']:.4f})\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"  Summary report saved: {report_file}")


def compare_hemispheres(
    left_results: Dict,
    right_results: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Compare results between left and right hemispheres.
    
    Parameters
    ----------
    left_results : dict
        Left hemisphere comparison results
    right_results : dict
        Right hemisphere comparison results
    output_dir : Path
        Output directory
    logger : logging.Logger
        Logger instance
    """
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPARING LEFT VS RIGHT HEMISPHERES")
    logger.info(f"{'='*80}\n")
    
    comparison_dir = output_dir / "hemisphere_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract overall accuracies
    left_multi_acc = left_results['overall_comparison'][
        left_results['overall_comparison']['Metric'] == 'Accuracy'
    ]['Multinomial'].values[0]
    
    left_ovr_acc = left_results['overall_comparison'][
        left_results['overall_comparison']['Metric'] == 'Accuracy'
    ]['OvR'].values[0]
    
    right_multi_acc = right_results['overall_comparison'][
        right_results['overall_comparison']['Metric'] == 'Accuracy'
    ]['Multinomial'].values[0]
    
    right_ovr_acc = right_results['overall_comparison'][
        right_results['overall_comparison']['Metric'] == 'Accuracy'
    ]['OvR'].values[0]
    
    logger.info("Hemisphere Performance Comparison:")
    logger.info(f"  Left Multinomial: {left_multi_acc:.4f}")
    logger.info(f"  Right Multinomial: {right_multi_acc:.4f}")
    logger.info(f"  Left OvR: {left_ovr_acc:.4f}")
    logger.info(f"  Right OvR: {right_ovr_acc:.4f}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['Multinomial', 'One-vs-Rest']
    left_values = [left_multi_acc, left_ovr_acc]
    right_values = [right_multi_acc, right_ovr_acc]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, left_values, width, label='Left Hemisphere',
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, right_values, width, label='Right Hemisphere',
                   color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Hemisphere Comparison Across Strategies', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim([0.85, 1.0])
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'hemisphere_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Hemisphere comparison saved to: {comparison_dir}")


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = args.results_dir / "comparison"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    
    logger.info("="*80)
    logger.info("STRATEGY COMPARISON ANALYSIS")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Hemisphere: {args.hemisphere}")
    logger.info(f"  Results directory: {args.results_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Include OvO: {args.include_ovo}")
    
    try:
        # Compare strategies
        if args.hemisphere == 'both':
            # Compare both hemispheres
            left_results = compare_single_hemisphere('left', args, logger)
            right_results = compare_single_hemisphere('right', args, logger)
            
            # Cross-hemisphere comparison
            if left_results is not None and right_results is not None:
                compare_hemispheres(left_results, right_results, args.output_dir, logger)
        else:
            # Compare single hemisphere
            results = compare_single_hemisphere(args.hemisphere, args, logger)
        
        logger.info("\n" + "="*80)
        logger.info("STRATEGY COMPARISON COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()