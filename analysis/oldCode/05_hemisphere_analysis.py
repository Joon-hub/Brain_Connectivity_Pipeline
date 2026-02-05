"""
Analyze hemisphere-specific patterns.

Generates:
- Left vs Right hemisphere comparisons
- Lateralization indices
- Hemisphere-specific network patterns
- Asymmetry visualizations

Usage:
    python analysis/05_hemisphere_analysis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from analysis.data_loader import DataLoader
from src.analysis.statistics import paired_ttest_hemispheres, fdr_correction
from src.analysis.comparisons import compare_hemispheres

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_lateralization_index(
    left_value: float,
    right_value: float
) -> float:
    """
    Compute lateralization index.
    
    LI = (Left - Right) / (Left + Right)
    
    LI > 0: Left-lateralized
    LI < 0: Right-lateralized
    LI ≈ 0: Bilateral
    
    Parameters
    ----------
    left_value : float
        Left hemisphere value (e.g., accuracy, reorganization)
    right_value : float
        Right hemisphere value
    
    Returns
    -------
    li : float
        Lateralization index
    """
    
    if (left_value + right_value) == 0:
        return 0.0
    
    return (left_value - right_value) / (left_value + right_value)


def analyze_hemisphere_performance(
    all_results: dict,
    strategy: str = 'multinomial'
) -> pd.DataFrame:
    """
    Analyze performance differences between hemispheres.
    
    Parameters
    ----------
    all_results : dict
        All model results
    strategy : str
        Strategy to analyze
    
    Returns
    -------
    analysis_df : pd.DataFrame
        Hemisphere comparison analysis
    """
    
    left_model = f'left_{strategy}'
    right_model = f'right_{strategy}'
    
    if left_model not in all_results or right_model not in all_results:
        raise ValueError(f"Models not found: {left_model}, {right_model}")
    
    left_summary = all_results[left_model]['summary']
    right_summary = all_results[right_model]['summary']
    
    # Create comparison DataFrame
    metrics = [
        'rest_train_accuracy',
        'task_test_accuracy',
        'accuracy_drop',
        'task_balanced_accuracy',
        'task_top_5_accuracy'
    ]
    
    comparison_data = []
    
    for metric in metrics:
        left_val = left_summary.get(metric, np.nan)
        right_val = right_summary.get(metric, np.nan)
        
        if metric == 'task_top_5_accuracy':
            left_val = left_summary.get(metric, np.nan) if metric in left_summary else np.nan
            right_val = right_summary.get(metric, np.nan) if metric in right_summary else np.nan
        
        difference = left_val - right_val
        
        if metric.endswith('accuracy'):
            # For accuracy, positive LI means left is better
            li = compute_lateralization_index(left_val, right_val)
        else:
            # For drops, positive LI means left drops more
            li = compute_lateralization_index(left_val, right_val)
        
        comparison_data.append({
            'metric': metric,
            'left_hemisphere': left_val,
            'right_hemisphere': right_val,
            'difference': difference,
            'lateralization_index': li,
            'abs_difference': abs(difference)
        })
    
    analysis_df = pd.DataFrame(comparison_data)
    
    return analysis_df


def analyze_network_lateralization(
    all_results: dict,
    strategy: str = 'multinomial'
) -> pd.DataFrame:
    """
    Analyze lateralization at network level.
    
    Parameters
    ----------
    all_results : dict
        All model results
    strategy : str
        Strategy to analyze
    
    Returns
    -------
    network_lat_df : pd.DataFrame
        Network-level lateralization analysis
    """
    
    left_model = f'left_{strategy}'
    right_model = f'right_{strategy}'
    
    left_networks = all_results[left_model]['network_metrics']
    right_networks = all_results[right_model]['network_metrics']
    
    # Merge on network name
    merged = left_networks.merge(
        right_networks,
        on='network',
        suffixes=('_left', '_right')
    )
    
    # Compute lateralization indices
    merged['lateralization_index'] = merged.apply(
        lambda row: compute_lateralization_index(
            row['accuracy_left'],
            row['accuracy_right']
        ),
        axis=1
    )
    
    merged['accuracy_difference'] = merged['accuracy_left'] - merged['accuracy_right']
    merged['abs_difference'] = merged['accuracy_difference'].abs()
    
    # Sort by absolute lateralization
    merged = merged.sort_values('abs_difference', ascending=False)
    
    return merged


def analyze_region_lateralization(
    all_results: dict,
    strategy: str = 'multinomial'
) -> pd.DataFrame:
    """
    Analyze lateralization at region level.
    
    This is tricky because left/right models have different regions.
    We'll focus on network-level patterns instead.
    
    Parameters
    ----------
    all_results : dict
        All model results
    strategy : str
        Strategy to analyze
    
    Returns
    -------
    region_lat_df : pd.DataFrame
        Region-level lateralization (if possible)
    """
    
    logger.info("Region-level lateralization analysis...")
    logger.info("Note: Left and right models have different regions")
    logger.info("Analysis performed at network level instead")
    
    return analyze_network_lateralization(all_results, strategy)


def compute_hemisphere_asymmetry_index(
    left_reorg: pd.DataFrame,
    right_reorg: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute asymmetry index for reorganization patterns.
    
    AI = (Left_reorg - Right_reorg) / (Left_reorg + Right_reorg)
    
    Parameters
    ----------
    left_reorg : pd.DataFrame
        Left hemisphere reorganization data
    right_reorg : pd.DataFrame
        Right hemisphere reorganization data
    
    Returns
    -------
    asymmetry_df : pd.DataFrame
        Asymmetry analysis
    """
    
    # Aggregate by network
    left_net = left_reorg.groupby('network')['reorganization_index'].mean()
    right_net = right_reorg.groupby('network')['reorganization_index'].mean()
    
    # Create asymmetry DataFrame
    asymmetry_data = []
    
    for network in left_net.index:
        if network in right_net.index:
            left_val = left_net[network]
            right_val = right_net[network]
            
            ai = compute_lateralization_index(left_val, right_val)
            
            asymmetry_data.append({
                'network': network,
                'left_reorganization': left_val,
                'right_reorganization': right_val,
                'asymmetry_index': ai,
                'difference': left_val - right_val
            })
    
    asymmetry_df = pd.DataFrame(asymmetry_data)
    asymmetry_df = asymmetry_df.sort_values('asymmetry_index', key=abs, ascending=False)
    
    return asymmetry_df


def plot_hemisphere_comparison_overview(
    analysis_df: pd.DataFrame,
    save_path: Path,
    strategy: str = 'multinomial'
):
    """
    Plot 4-panel hemisphere comparison overview.
    
    A) Overall performance comparison
    B) Lateralization indices
    C) Metric-wise differences
    D) Statistical significance
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Overall performance
    ax = axes[0, 0]
    
    metrics_to_plot = ['rest_train_accuracy', 'task_test_accuracy', 'task_balanced_accuracy']
    plot_data = analysis_df[analysis_df['metric'].isin(metrics_to_plot)]
    
    x = np.arange(len(plot_data))
    width = 0.35
    
    ax.bar(x - width/2, plot_data['left_hemisphere'], width, label='Left', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, plot_data['right_hemisphere'], width, label='Right', alpha=0.8, color='coral')
    
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'A) Performance Comparison ({strategy.capitalize()})', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Rest CV', 'Task Test', 'Balanced'], rotation=0)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Lateralization indices
    ax = axes[0, 1]
    
    colors = ['steelblue' if li > 0 else 'coral' for li in analysis_df['lateralization_index']]
    
    ax.barh(range(len(analysis_df)), analysis_df['lateralization_index'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(analysis_df)))
    ax.set_yticklabels(analysis_df['metric'], fontsize=9)
    ax.set_xlabel('Lateralization Index', fontsize=11)
    ax.set_title('B) Lateralization Patterns', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.text(0.05, len(analysis_df)-0.5, 'Left >', fontsize=10, color='steelblue', weight='bold')
    ax.text(-0.05, len(analysis_df)-0.5, '< Right', fontsize=10, color='coral', weight='bold', ha='right')
    ax.grid(axis='x', alpha=0.3)
    
    # Panel C: Metric-wise differences
    ax = axes[1, 0]
    
    ax.barh(range(len(analysis_df)), analysis_df['abs_difference'], color='teal', alpha=0.7)
    ax.set_yticks(range(len(analysis_df)))
    ax.set_yticklabels(analysis_df['metric'], fontsize=9)
    ax.set_xlabel('Absolute Difference', fontsize=11)
    ax.set_title('C) Magnitude of Differences', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Panel D: Text summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Hemisphere Comparison Summary
    Strategy: {strategy.capitalize()}
    
    Key Findings:
    
    • Largest difference: {analysis_df.iloc[analysis_df['abs_difference'].idxmax()]['metric']}
      ({analysis_df['abs_difference'].max():.4f})
    
    • Strongest lateralization: {analysis_df.iloc[analysis_df['lateralization_index'].abs().idxmax()]['metric']}
      (LI = {analysis_df['lateralization_index'].abs().max():.4f})
    
    • Mean |difference|: {analysis_df['abs_difference'].mean():.4f}
    
    • Left-dominant metrics: {(analysis_df['lateralization_index'] > 0).sum()}
    • Right-dominant metrics: {(analysis_df['lateralization_index'] < 0).sum()}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def plot_network_lateralization(
    network_lat_df: pd.DataFrame,
    save_path: Path,
    strategy: str = 'multinomial'
):
    """
    Plot network-level lateralization patterns.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Network accuracies comparison
    ax = axes[0, 0]
    
    x = np.arange(len(network_lat_df))
    width = 0.35
    
    ax.bar(x - width/2, network_lat_df['accuracy_left'], width, label='Left', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, network_lat_df['accuracy_right'], width, label='Right', alpha=0.8, color='coral')
    
    ax.set_ylabel('Task Accuracy', fontsize=11)
    ax.set_title(f'A) Network Accuracy by Hemisphere ({strategy.capitalize()})', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(network_lat_df['network'], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Lateralization indices
    ax = axes[0, 1]
    
    colors = ['steelblue' if li > 0 else 'coral' for li in network_lat_df['lateralization_index']]
    
    sorted_df = network_lat_df.sort_values('lateralization_index')
    colors_sorted = ['steelblue' if li > 0 else 'coral' for li in sorted_df['lateralization_index']]
    
    ax.barh(range(len(sorted_df)), sorted_df['lateralization_index'], color=colors_sorted, alpha=0.7)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['network'], fontsize=9)
    ax.set_xlabel('Lateralization Index', fontsize=11)
    ax.set_title('B) Network Lateralization Patterns', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Panel C: Scatter - Left vs Right accuracy
    ax = axes[1, 0]
    
    ax.scatter(network_lat_df['accuracy_left'], network_lat_df['accuracy_right'],
               s=100, alpha=0.6, c='teal')
    
    # Add network labels
    for _, row in network_lat_df.iterrows():
        ax.annotate(row['network'], 
                   (row['accuracy_left'], row['accuracy_right']),
                   fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points')
    
    # Add identity line
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Identity')
    
    ax.set_xlabel('Left Hemisphere Accuracy', fontsize=11)
    ax.set_ylabel('Right Hemisphere Accuracy', fontsize=11)
    ax.set_title('C) Left vs Right Network Performance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Panel D: Absolute differences
    ax = axes[1, 1]
    
    sorted_df_abs = network_lat_df.sort_values('abs_difference', ascending=True)
    
    colors_diff = plt.cm.Reds(sorted_df_abs['abs_difference'] / sorted_df_abs['abs_difference'].max())
    
    ax.barh(range(len(sorted_df_abs)), sorted_df_abs['abs_difference'], color=colors_diff, alpha=0.8)
    ax.set_yticks(range(len(sorted_df_abs)))
    ax.set_yticklabels(sorted_df_abs['network'], fontsize=9)
    ax.set_xlabel('|Accuracy Difference|', fontsize=11)
    ax.set_title('D) Magnitude of Network Asymmetry', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def plot_reorganization_asymmetry(
    asymmetry_df: pd.DataFrame,
    save_path: Path,
    strategy: str = 'multinomial'
):
    """
    Plot reorganization asymmetry between hemispheres.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Reorganization comparison
    ax = axes[0]
    
    x = np.arange(len(asymmetry_df))
    width = 0.35
    
    ax.bar(x - width/2, asymmetry_df['left_reorganization'], width, 
           label='Left', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, asymmetry_df['right_reorganization'], width,
           label='Right', alpha=0.8, color='coral')
    
    ax.set_ylabel('Reorganization Index', fontsize=11)
    ax.set_title(f'A) Network Reorganization by Hemisphere ({strategy.capitalize()})', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(asymmetry_df['network'], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Asymmetry indices
    ax = axes[1]
    
    sorted_asymm = asymmetry_df.sort_values('asymmetry_index')
    colors = ['steelblue' if ai > 0 else 'coral' for ai in sorted_asymm['asymmetry_index']]
    
    ax.barh(range(len(sorted_asymm)), sorted_asymm['asymmetry_index'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_asymm)))
    ax.set_yticklabels(sorted_asymm['network'], fontsize=9)
    ax.set_xlabel('Asymmetry Index', fontsize=11)
    ax.set_title('B) Reorganization Asymmetry', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Panel C: Scatter plot
    ax = axes[2]
    
    ax.scatter(asymmetry_df['left_reorganization'], asymmetry_df['right_reorganization'],
               s=100, alpha=0.6, c='purple')
    
    for _, row in asymmetry_df.iterrows():
        ax.annotate(row['network'],
                   (row['left_reorganization'], row['right_reorganization']),
                   fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points')
    
    # Identity line
    max_val = max(asymmetry_df['left_reorganization'].max(), 
                  asymmetry_df['right_reorganization'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1, label='Identity')
    
    ax.set_xlabel('Left Hemisphere Reorganization', fontsize=11)
    ax.set_ylabel('Right Hemisphere Reorganization', fontsize=11)
    ax.set_title('C) Left vs Right Reorganization', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def run_statistical_tests(
    all_results: dict,
    strategies: list = ['multinomial', 'ovr', 'ovo']
) -> pd.DataFrame:
    """
    Run statistical tests for hemisphere comparisons.
    
    Parameters
    ----------
    all_results : dict
        All model results
    strategies : list
        Strategies to test
    
    Returns
    -------
    test_results_df : pd.DataFrame
        Statistical test results
    """
    
    test_results = []
    
    for strategy in strategies:
        logger.info(f"\nTesting {strategy}...")
        
        try:
            t_stat, p_value = paired_ttest_hemispheres(all_results, strategy=strategy)
            
            test_results.append({
                'strategy': strategy,
                'test': 'Paired t-test',
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': 'Left > Right' if t_stat > 0 else 'Right > Left'
            })
            
            logger.info(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
            
        except Exception as e:
            logger.error(f"  Error: {str(e)}")
    
    test_results_df = pd.DataFrame(test_results)
    
    # Apply FDR correction
    if len(test_results_df) > 0:
        p_corrected = fdr_correction(test_results_df['p_value'].values)
        test_results_df['p_corrected'] = p_corrected
        test_results_df['significant_corrected'] = p_corrected < 0.05
    
    return test_results_df


def main():
    """Run hemisphere analysis."""
    
    logger.info("="*80)
    logger.info("HEMISPHERE ANALYSIS")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    all_results = loader.load_all()
    
    logger.info(f"Loaded {len(all_results)} models")
    
    # Create output directories
    output_dir = project_root / 'outputs' / 'compiled'
    figures_dir = project_root / 'outputs' / 'figures'
    tables_dir = project_root / 'outputs' / 'tables'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each strategy
    strategies = ['multinomial', 'ovr', 'ovo']
    
    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {strategy.upper()} strategy")
        logger.info(f"{'='*60}")
        
        # Overall performance comparison
        logger.info("\n1. Overall performance analysis...")
        analysis_df = analyze_hemisphere_performance(all_results, strategy=strategy)
        analysis_df.to_csv(
            output_dir / f'hemisphere_performance_{strategy}.csv',
            index=False
        )
        logger.info(f"✓ Saved performance comparison")
        
        # Network-level lateralization
        logger.info("\n2. Network lateralization analysis...")
        network_lat_df = analyze_network_lateralization(all_results, strategy=strategy)
        network_lat_df.to_csv(
            output_dir / f'network_lateralization_{strategy}.csv',
            index=False
        )
        logger.info(f"✓ Saved network lateralization")
        
        # Reorganization asymmetry (if data available)
        logger.info("\n3. Reorganization asymmetry...")
        try:
            reorg_data = loader.load_reorganization_data()
            
            if f'left_{strategy}' in reorg_data and f'right_{strategy}' in reorg_data:
                asymmetry_df = compute_hemisphere_asymmetry_index(
                    reorg_data[f'left_{strategy}'],
                    reorg_data[f'right_{strategy}']
                )
                asymmetry_df.to_csv(
                    output_dir / f'reorganization_asymmetry_{strategy}.csv',
                    index=False
                )
                logger.info(f"✓ Saved reorganization asymmetry")
                
                # Plot reorganization asymmetry
                plot_reorganization_asymmetry(
                    asymmetry_df,
                    save_path=figures_dir / f'hemisphere_reorganization_asymmetry_{strategy}.png',
                    strategy=strategy
                )
            else:
                logger.warning("Reorganization data not available")
        
        except Exception as e:
            logger.warning(f"Could not analyze reorganization asymmetry: {str(e)}")
        
        # Generate visualizations
        logger.info("\n4. Generating visualizations...")
        
        plot_hemisphere_comparison_overview(
            analysis_df,
            save_path=figures_dir / f'hemisphere_comparison_{strategy}.png',
            strategy=strategy
        )
        
        plot_network_lateralization(
            network_lat_df,
            save_path=figures_dir / f'network_lateralization_{strategy}.png',
            strategy=strategy
        )
    
    # Statistical tests
    logger.info("\n" + "="*60)
    logger.info("STATISTICAL TESTS")
    logger.info("="*60)
    
    test_results_df = run_statistical_tests(all_results, strategies=strategies)
    test_results_df.to_csv(tables_dir / 'hemisphere_statistical_tests.csv', index=False)
    
    logger.info("\nTest Results:")
    print(test_results_df.to_string(index=False))
    
    # Summary report
    logger.info("\n" + "="*80)
    logger.info("HEMISPHERE ANALYSIS SUMMARY")
    logger.info("="*80)
    
    for strategy in strategies:
        analysis_df = pd.read_csv(output_dir / f'hemisphere_performance_{strategy}.csv')
        
        logger.info(f"\n{strategy.upper()} Strategy:")
        logger.info(f"  Mean |difference|: {analysis_df['abs_difference'].mean():.4f}")
        logger.info(f"  Max |difference|: {analysis_df['abs_difference'].max():.4f}")
        logger.info(f"  Mean LI: {analysis_df['lateralization_index'].mean():.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("HEMISPHERE ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to:")
    logger.info(f"  • Tables: {output_dir}")
    logger.info(f"  • Figures: {figures_dir}")
    logger.info(f"  • Statistical tests: {tables_dir}")


if __name__ == "__main__":
    main()