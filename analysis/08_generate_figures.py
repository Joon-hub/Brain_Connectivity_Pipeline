"""
Generate all publication-quality figures for thesis.

Creates:
- Figure 1: Performance overview
- Figure 2: Model comparisons
- Figure 3: Reorganization maps
- Figure 4: Confusion matrices
- Figure 5: Error analysis
- Figure 6: Hemisphere analysis
- Supplementary figures

Usage:
    python analysis/08_generate_figures.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from analysis.data_loader import DataLoader
from src.analysis.visualizations import (
    plot_performance_overview,
    plot_strategy_comparison,
    plot_hemisphere_comparison,
    plot_confusion_matrices_grid,
    plot_reorganization_map
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def generate_figure_1(
    all_results: dict,
    summary_df: pd.DataFrame,
    network_df: pd.DataFrame,
    save_path: Path
):
    """
    Figure 1: Overall Performance Overview.
    
    4-panel figure showing:
    A) Rest vs Task accuracy comparison
    B) Accuracy drops
    C) Network-level performance heatmap
    D) Top-5 accuracy
    """
    
    logger.info("\nGenerating Figure 1: Performance Overview...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Rest vs Task comparison
    ax_a = fig.add_subplot(gs[0, 0])
    
    x = np.arange(len(summary_df))
    width = 0.35
    
    ax_a.bar(x - width/2, summary_df['rest_cv_accuracy'], width, 
            label='Rest CV', alpha=0.8, color='steelblue')
    ax_a.bar(x + width/2, summary_df['task_test_accuracy'], width,
            label='Task Test', alpha=0.8, color='coral')
    
    ax_a.set_ylabel('Accuracy', fontsize=11)
    ax_a.set_title('A) Rest CV vs Task Test Accuracy', fontsize=12, fontweight='bold')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(summary_df['model'], rotation=45, ha='right', fontsize=8)
    ax_a.legend(loc='lower left')
    ax_a.set_ylim([0, 1])
    ax_a.grid(axis='y', alpha=0.3)
    ax_a.axhline(y=1/232, color='gray', linestyle=':', linewidth=1, label='Chance (1/232)')
    
    # Panel B: Accuracy drops
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Group by scope for better visualization
    scope_order = ['full', 'left', 'right']
    strategy_order = ['multinomial', 'ovr', 'ovo']
    
    summary_df['scope_cat'] = pd.Categorical(summary_df['scope'], categories=scope_order, ordered=True)
    summary_df['strategy_cat'] = pd.Categorical(summary_df['strategy'], categories=strategy_order, ordered=True)
    summary_sorted = summary_df.sort_values(['scope_cat', 'strategy_cat'])
    
    colors_map = {'full': 'steelblue', 'left': 'coral', 'right': 'teal'}
    colors = [colors_map[scope] for scope in summary_sorted['scope']]
    
    ax_b.barh(range(len(summary_sorted)), summary_sorted['accuracy_drop'], color=colors, alpha=0.7)
    ax_b.set_yticks(range(len(summary_sorted)))
    ax_b.set_yticklabels(summary_sorted['model'], fontsize=8)
    ax_b.set_xlabel('Accuracy Drop', fontsize=11)
    ax_b.set_title('B) Task Generalization Drop', fontsize=12, fontweight='bold')
    ax_b.axvline(summary_sorted['accuracy_drop'].mean(), color='red', linestyle='--',
                linewidth=1, label=f'Mean: {summary_sorted["accuracy_drop"].mean():.3f}')
    ax_b.legend()
    ax_b.grid(axis='x', alpha=0.3)
    
    # Create legend for scopes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[scope], alpha=0.7, label=scope.capitalize())
                      for scope in scope_order]
    ax_b.legend(handles=legend_elements, loc='lower right', title='Scope')
    
    # Panel C: Network-level heatmap
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Prepare network data for heatmap
    # Pivot: models × networks
    network_pivot = network_df.pivot_table(
        index='network',
        columns='model',
        values='accuracy'
    )
    
    # Sort networks by mean accuracy
    network_pivot['mean'] = network_pivot.mean(axis=1)
    network_pivot = network_pivot.sort_values('mean', ascending=False)
    network_pivot = network_pivot.drop('mean', axis=1)
    
    # Plot heatmap
    im = ax_c.imshow(network_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax_c.set_yticks(range(len(network_pivot)))
    ax_c.set_yticklabels(network_pivot.index, fontsize=8)
    ax_c.set_xticks(range(len(network_pivot.columns)))
    ax_c.set_xticklabels(network_pivot.columns, rotation=45, ha='right', fontsize=7)
    ax_c.set_title('C) Network-Level Task Performance', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_c, label='Accuracy')
    
    # Panel D: Top-5 accuracy
    ax_d = fig.add_subplot(gs[1, 1])
    
    if 'top_5_accuracy' in summary_df.columns:
        colors_d = [colors_map[scope] for scope in summary_sorted['scope']]
        
        ax_d.barh(range(len(summary_sorted)), summary_sorted['top_5_accuracy'], 
                 color=colors_d, alpha=0.7)
        ax_d.set_yticks(range(len(summary_sorted)))
        ax_d.set_yticklabels(summary_sorted['model'], fontsize=8)
        ax_d.set_xlabel('Top-5 Accuracy', fontsize=11)
        ax_d.set_title('D) Top-5 Prediction Accuracy', fontsize=12, fontweight='bold')
        ax_d.axvline(summary_sorted['top_5_accuracy'].mean(), color='red', linestyle='--',
                    linewidth=1, label=f'Mean: {summary_sorted["top_5_accuracy"].mean():.3f}')
        ax_d.set_xlim([0, 1])
        ax_d.legend()
        ax_d.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Figure 1: Overall Model Performance', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def generate_figure_2(
    all_results: dict,
    summary_df: pd.DataFrame,
    save_path: Path
):
    """
    Figure 2: Strategy and Scope Comparisons.
    
    3-panel figure showing:
    A) Strategy comparison (Multinomial vs OvR vs OvO)
    B) Hemisphere comparison (Left vs Right)
    C) Scope comparison (Full vs Hemisphere)
    """
    
    logger.info("\nGenerating Figure 2: Model Comparisons...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: Strategy comparison
    ax = axes[0]
    
    scopes = ['full', 'left', 'right']
    strategies = ['multinomial', 'ovr', 'ovo']
    
    drops_by_strategy = {s: [] for s in strategies}
    
    for scope in scopes:
        for strategy in strategies:
            model_name = f'{scope}_{strategy}'
            if model_name in summary_df['model'].values:
                drop = summary_df[summary_df['model'] == model_name]['accuracy_drop'].values[0]
                drops_by_strategy[strategy].append(drop)
    
    x = np.arange(len(scopes))
    width = 0.25
    
    for i, strategy in enumerate(strategies):
        ax.bar(x + i*width - width, drops_by_strategy[strategy], width, 
              label=strategy.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Scope', fontsize=11)
    ax.set_ylabel('Accuracy Drop', fontsize=11)
    ax.set_title('A) Strategy Comparison Across Scopes', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scopes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Hemisphere comparison
    ax = axes[1]
    
    hemisphere_data = []
    
    for strategy in strategies:
        left_model = f'left_{strategy}'
        right_model = f'right_{strategy}'
        
        if left_model in summary_df['model'].values and right_model in summary_df['model'].values:
            left_drop = summary_df[summary_df['model'] == left_model]['accuracy_drop'].values[0]
            right_drop = summary_df[summary_df['model'] == right_model]['accuracy_drop'].values[0]
            
            hemisphere_data.append({
                'strategy': strategy,
                'left': left_drop,
                'right': right_drop
            })
    
    hem_df = pd.DataFrame(hemisphere_data)
    
    x = np.arange(len(hem_df))
    width = 0.35
    
    ax.bar(x - width/2, hem_df['left'], width, label='Left', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, hem_df['right'], width, label='Right', alpha=0.8, color='coral')
    
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_ylabel('Accuracy Drop', fontsize=11)
    ax.set_title('B) Hemisphere Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in hem_df['strategy']])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Scope comparison
    ax = axes[2]
    
    scope_comparison = []
    
    for strategy in strategies:
        full_model = f'full_{strategy}'
        
        if full_model in summary_df['model'].values:
            full_acc = summary_df[summary_df['model'] == full_model]['task_test_accuracy'].values[0]
            full_drop = summary_df[summary_df['model'] == full_model]['accuracy_drop'].values[0]
            
            scope_comparison.append({
                'strategy': strategy,
                'accuracy': full_acc,
                'drop': full_drop,
                'scope': 'full'
            })
        
        # Average of hemispheres
        left_model = f'left_{strategy}'
        right_model = f'right_{strategy}'
        
        if left_model in summary_df['model'].values and right_model in summary_df['model'].values:
            left_acc = summary_df[summary_df['model'] == left_model]['task_test_accuracy'].values[0]
            right_acc = summary_df[summary_df['model'] == right_model]['task_test_accuracy'].values[0]
            avg_acc = (left_acc + right_acc) / 2
            
            left_drop = summary_df[summary_df['model'] == left_model]['accuracy_drop'].values[0]
            right_drop = summary_df[summary_df['model'] == right_model]['accuracy_drop'].values[0]
            avg_drop = (left_drop + right_drop) / 2
            
            scope_comparison.append({
                'strategy': strategy,
                'accuracy': avg_acc,
                'drop': avg_drop,
                'scope': 'hemisphere'
            })
    
    scope_df = pd.DataFrame(scope_comparison)
    
    # Plot grouped bars
    full_data = scope_df[scope_df['scope'] == 'full']
    hem_data = scope_df[scope_df['scope'] == 'hemisphere']
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax.bar(x - width/2, full_data['accuracy'], width, label='Full Brain', alpha=0.8, color='teal')
    ax.bar(x + width/2, hem_data['accuracy'], width, label='Hemisphere (avg)', alpha=0.8, color='orange')
    
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_ylabel('Task Test Accuracy', fontsize=11)
    ax.set_title('C) Full Brain vs Hemisphere Models', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Figure 2: Model Strategy and Scope Comparisons', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def generate_figure_3(
    reorg_data: dict,
    save_path: Path,
    model_name: str = 'full_multinomial'
):
    """
    Figure 3: Reorganization Patterns.
    
    Shows region-level and network-level reorganization.
    """
    
    logger.info("\nGenerating Figure 3: Reorganization Maps...")
    
    if model_name not in reorg_data:
        logger.warning(f"Reorganization data not found for {model_name}")
        return
    
    reorg_df = reorg_data[model_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Panel A: Region-level reorganization
    ax = axes[0]
    
    sorted_df = reorg_df.sort_values('reorganization_index', ascending=True)
    
    # Color by network
    networks = sorted_df['network'].unique()
    network_colors = {net: plt.cm.tab10(i) for i, net in enumerate(networks)}
    colors = [network_colors[net] for net in sorted_df['network']]
    
    y_pos = np.arange(len(sorted_df))
    
    ax.barh(y_pos, sorted_df['reorganization_index'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos[::10])  # Show every 10th region
    ax.set_yticklabels(sorted_df['region_name'].iloc[::10], fontsize=6)
    ax.set_xlabel('Reorganization Index', fontsize=11)
    ax.set_title('A) Region-Level Functional Reorganization', fontsize=12, fontweight='bold')
    ax.axvline(sorted_df['reorganization_index'].mean(), color='red', linestyle='--',
              linewidth=1, label=f'Mean: {sorted_df["reorganization_index"].mean():.3f}')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Panel B: Network-level reorganization
    ax = axes[1]
    
    network_reorg = reorg_df.groupby('network')['reorganization_index'].agg(['mean', 'std']).reset_index()
    network_reorg = network_reorg.sort_values('mean', ascending=False)
    
    x = np.arange(len(network_reorg))
    
    bars = ax.bar(x, network_reorg['mean'], alpha=0.8, color='teal',
                 yerr=network_reorg['std'], capsize=5, error_kw={'linewidth': 1})
    
    ax.set_xticks(x)
    ax.set_xticklabels(network_reorg['network'], rotation=45, ha='right')
    ax.set_ylabel('Mean Reorganization Index', fontsize=11)
    ax.set_title('B) Network-Level Reorganization', fontsize=12, fontweight='bold')
    ax.axhline(network_reorg['mean'].mean(), color='red', linestyle='--',
              linewidth=1, label=f'Overall Mean: {network_reorg["mean"].mean():.3f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Figure 3: Functional Reorganization Patterns ({model_name})',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def generate_supplementary_figures(
    all_results: dict,
    output_dir: Path
):
    """
    Generate supplementary figures.
    
    - Detailed confusion matrices (one per model)
    - Per-region accuracy plots
    - Error type distributions
    - etc.
    """
    
    logger.info("\nGenerating supplementary figures...")
    
    supp_dir = output_dir / 'supplementary'
    supp_dir.mkdir(exist_ok=True)
    
    # Supplementary Figure S1: Individual confusion matrices
    logger.info("  • Confusion matrices (3x3 grid)...")
    plot_confusion_matrices_grid(
        all_results,
        save_path=supp_dir / 'figS1_confusion_matrices_grid.png'
    )
    
    # Supplementary Figure S2: Per-region accuracy for each model
    logger.info("  • Per-region accuracy plots...")
    for model_name, results in all_results.items():
        if 'per_region_metrics' not in results:
            continue
        
        per_region = results['per_region_metrics'].sort_values('accuracy', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, max(8, len(per_region) * 0.12)))
        
        colors = plt.cm.RdYlGn(per_region['accuracy'])
        
        ax.barh(range(len(per_region)), per_region['accuracy'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(per_region)))
        ax.set_yticklabels(per_region['region_name'], fontsize=6)
        ax.set_xlabel('Task Test Accuracy', fontsize=11)
        ax.set_title(f'Per-Region Accuracy: {model_name}', fontsize=12, fontweight='bold')
        ax.axvline(per_region['accuracy'].mean(), color='red', linestyle='--',
                  linewidth=1, label=f'Mean: {per_region["accuracy"].mean():.3f}')
        ax.legend()
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(supp_dir / f'figS2_{model_name}_per_region.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"✓ Supplementary figures saved to: {supp_dir}")


def main():
    """Generate all figures."""
    
    logger.info("="*80)
    logger.info("GENERATING ALL THESIS FIGURES")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    all_results = loader.load_all()
    
    logger.info(f"Loaded {len(all_results)} models")
    
    # Load compiled data
    compiled_dir = project_root / 'outputs' / 'compiled'
    
    summary_df = pd.read_csv(compiled_dir.parent / 'tables' / 'table1_overall_performance.csv')
    
    # Load network data
    network_files = list(compiled_dir.glob('*_network_reorganization.csv'))
    if network_files:
        network_dfs = []
        for nf in network_files:
            df = pd.read_csv(nf)
            model_name = nf.stem.replace('_network_reorganization', '')
            df['model'] = model_name
            network_dfs.append(df)
        network_df = pd.concat(network_dfs, ignore_index=True)
    else:
        # Fallback: use network_metrics from results
        network_dfs = []
        for model_name, results in all_results.items():
            if 'network_metrics' in results:
                df = results['network_metrics'].copy()
                df['model'] = model_name
                network_dfs.append(df)
        network_dfs = pd.concat(network_dfs, ignore_index=True)

    # Load reorganization data
    reorg_data = loader.load_reorganization_data()

    # Create figures directory
    figures_dir = project_root / 'outputs' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate main figures
    logger.info("\n" + "="*60)
    logger.info("MAIN FIGURES")
    logger.info("="*60)

    # Figure 1
    generate_figure_1(
        all_results,
        summary_df,
        network_df,
        save_path=figures_dir / 'figure1_performance_overview.png'
    )

    # Figure 2
    generate_figure_2(
        all_results,
        summary_df,
        save_path=figures_dir / 'figure2_model_comparisons.png'
    )

    # Figure 3
    generate_figure_3(
        reorg_data,
        save_path=figures_dir / 'figure3_reorganization_patterns.png',
        model_name='full_multinomial'
    )

    # Figure 4 (already generated by 03_error_analysis.py)
    logger.info("\nFigure 4: Confusion matrices (see fig4_confusion_matrices.png)")

    # Figure 5 (error analysis - already generated)
    logger.info("Figure 5: Error analysis (see error analysis outputs)")

    # Figure 6 (hemisphere analysis - already generated)
    logger.info("Figure 6: Hemisphere analysis (see hemisphere_comparison_*.png)")

    # Generate supplementary figures
    logger.info("\n" + "="*60)
    logger.info("SUPPLEMENTARY FIGURES")
    logger.info("="*60)

    generate_supplementary_figures(all_results, figures_dir)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("FIGURE GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll figures saved to: {figures_dir}")
    logger.info("\nMain Figures:")
    logger.info("  • Figure 1: Performance Overview")
    logger.info("  • Figure 2: Model Comparisons")
    logger.info("  • Figure 3: Reorganization Patterns")
    logger.info("  • Figure 4: Confusion Matrices")
    logger.info("  • Figure 5: Error Analysis")
    logger.info("  • Figure 6: Hemisphere Analysis")
    logger.info("\nSupplementary Figures:")
    logger.info("  • Individual confusion matrices")
    logger.info("  • Per-region accuracy plots")
    logger.info("  • Additional visualizations")