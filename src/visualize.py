"""
Visualization Module
====================
Create publication-quality figures for thesis.

Generates 4 key figures:
1. Training error map (resting-state)
2. Task error map
3. Rest vs task comparison
4. Network-level analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def plot_error_map(error_df: pd.DataFrame, 
                   title: str,
                   output_path: str,
                   dpi: int = 300):
    """
    Create 4-panel error map visualization.
    
    Args:
        error_df: Error map DataFrame
        title: Figure title
        output_path: Save path
        dpi: Resolution
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: All regions bar plot
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn_r(error_df['misclassification_rate'] / error_df['misclassification_rate'].max())
    ax1.bar(range(len(error_df)), error_df['misclassification_rate'], 
            color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax1.axhline(error_df['misclassification_rate'].mean(), 
                color='blue', linestyle='--', linewidth=2, label='Mean')
    ax1.set_xlabel('Region Index', fontweight='bold')
    ax1.set_ylabel('Misclassification Rate', fontweight='bold')
    ax1.set_title('All Regions', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Top 20 worst regions
    ax2 = axes[0, 1]
    top_20 = error_df.head(20)
    ax2.barh(range(20), top_20['misclassification_rate'], 
             color='red', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([name[:35] for name in top_20['region_name']], fontsize=8)
    ax2.set_xlabel('Misclassification Rate', fontweight='bold')
    ax2.set_title('Top 20 Misclassified Regions', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Panel 3: Distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(error_df['misclassification_rate'], bins=50, 
             alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(error_df['misclassification_rate'].mean(), 
                color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.set_xlabel('Misclassification Rate', fontweight='bold')
    ax3.set_ylabel('Number of Regions', fontweight='bold')
    ax3.set_title('Distribution of Error Rates', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    SUMMARY STATISTICS
    ==================
    
    Total Regions: {len(error_df)}
    Mean Error: {error_df['misclassification_rate'].mean():.4f}
    Median Error: {error_df['misclassification_rate'].median():.4f}
    Std Error: {error_df['misclassification_rate'].std():.4f}
    Min Error: {error_df['misclassification_rate'].min():.4f}
    Max Error: {error_df['misclassification_rate'].max():.4f}
    
    High Error (>30%): {(error_df['misclassification_rate'] > 0.3).sum()}
    Low Error (<10%): {(error_df['misclassification_rate'] < 0.1).sum()}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure: {output_path}")


def plot_rest_vs_task_comparison(error_rest: pd.DataFrame,
                                 error_task: pd.DataFrame,
                                 comparison: pd.DataFrame,
                                 output_path: str,
                                 dpi: int = 300):
    """
    Create rest vs task comparison figure.
    
    Args:
        error_rest: Resting-state errors
        error_task: Task errors
        comparison: Comparison DataFrame
        output_path: Save path
        dpi: Resolution
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Scatter plot
    ax1 = axes[0, 0]
    merged = pd.merge(
        error_rest[['region_name', 'misclassification_rate']],
        error_task[['region_name', 'misclassification_rate']],
        on='region_name',
        suffixes=('_rest', '_task')
    )
    ax1.scatter(merged['misclassification_rate_rest'],
                merged['misclassification_rate_task'],
                alpha=0.5, s=50)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Equal Error')
    ax1.set_xlabel('Resting-State Error', fontweight='bold')
    ax1.set_ylabel('Task Error', fontweight='bold')
    ax1.set_title('Error Rate Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Top altered regions
    ax2 = axes[0, 1]
    top_20 = comparison.head(20)
    colors = ['red' if x > 0.1 else 'orange' for x in top_20['error_increase']]
    ax2.barh(range(20), top_20['error_increase'], color=colors,
             edgecolor='black', alpha=0.7)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([name[:35] for name in top_20['region_name']], fontsize=8)
    ax2.set_xlabel('Error Increase (Task - Rest)', fontweight='bold')
    ax2.set_title('Top 20 Task-Altered Regions', fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(0, color='black', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Panel 3: Error increase distribution
    ax3 = axes[1, 0]
    ax3.hist(comparison['error_increase'], bins=50, alpha=0.7,
             color='steelblue', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax3.axvline(comparison['error_increase'].mean(), color='green',
                linestyle='--', linewidth=2, label='Mean')
    ax3.set_xlabel('Error Increase', fontweight='bold')
    ax3.set_ylabel('Number of Regions', fontweight='bold')
    ax3.set_title('Distribution of Error Changes', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    n_increased = (comparison['error_increase'] > 0.05).sum()
    n_decreased = (comparison['error_increase'] < -0.05).sum()
    n_stable = len(comparison) - n_increased - n_decreased
    
    stats_text = f"""
    COMPARISON SUMMARY
    ==================
    
    Regions Analyzed: {len(comparison)}
    
    Error Changes:
      Increased (>5%):  {n_increased}
      Decreased (>5%):  {n_decreased}
      Stable:           {n_stable}
    
    Mean Error Increase: {comparison['error_increase'].mean():.4f}
    Max Error Increase:  {comparison['error_increase'].max():.4f}
    
    Rest Mean Error: {error_rest['misclassification_rate'].mean():.4f}
    Task Mean Error: {error_task['misclassification_rate'].mean():.4f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Rest vs Task Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure: {output_path}")


def plot_network_analysis(error_df: pd.DataFrame,
                         network_stats: pd.DataFrame,
                         output_path: str,
                         dpi: int = 300):
    """
    Create network-level analysis figure.
    
    Args:
        error_df: Per-region error DataFrame
        network_stats: Network statistics DataFrame
        output_path: Save path
        dpi: Resolution
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Network mean errors
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn_r(network_stats['mean_error'] / network_stats['mean_error'].max())
    bars = ax1.barh(range(len(network_stats)), network_stats['mean_error'],
                    color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_yticks(range(len(network_stats)))
    ax1.set_yticklabels(network_stats['network'], fontsize=9)
    ax1.set_xlabel('Mean Misclassification Rate', fontweight='bold')
    ax1.set_title('Error Rate by Network', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Panel 2: Network error bars with std
    ax2 = axes[0, 1]
    ax2.barh(range(len(network_stats)), network_stats['mean_error'],
             xerr=network_stats['std_error'], color='steelblue',
             edgecolor='black', linewidth=1.5, alpha=0.7, capsize=5)
    ax2.set_yticks(range(len(network_stats)))
    ax2.set_yticklabels(network_stats['network'], fontsize=9)
    ax2.set_xlabel('Mean Error ± Std', fontweight='bold')
    ax2.set_title('Network Variability', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Panel 3: Number of regions per network
    ax3 = axes[1, 0]
    ax3.bar(range(len(network_stats)), network_stats['n_regions'],
            color='coral', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax3.set_xticks(range(len(network_stats)))
    ax3.set_xticklabels(network_stats['network'], rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Number of Regions', fontweight='bold')
    ax3.set_title('Regions per Network', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_network = network_stats.iloc[-1]  # Lowest error (sorted ascending False in aggregate)
    worst_network = network_stats.iloc[0]  # Highest error
    
    stats_text = f"""
    NETWORK ANALYSIS
    ================
    
    Total Networks: {len(network_stats)}
    Total Regions: {network_stats['n_regions'].sum()}
    
    Best Network:
      {best_network['network']}
      Mean Error: {best_network['mean_error']:.4f}
      N Regions: {int(best_network['n_regions'])}
    
    Worst Network:
      {worst_network['network']}
      Mean Error: {worst_network['mean_error']:.4f}
      N Regions: {int(worst_network['n_regions'])}
    
    Overall Mean: {network_stats['mean_error'].mean():.4f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Network-Level Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure: {output_path}")