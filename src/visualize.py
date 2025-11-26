# src/visualize.py
"""
Visualization Module (Updated for Phase 2 Pipeline)
==================================================
Creates publication-quality 4-panel figures with proper region names.
Now compatible with new pipeline outputs (numpy arrays + region_list).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def plot_error_map(
    error_rates: Union[np.ndarray, List[float]],
    title: str,
    output_path: str,
    region_list: Optional[List[str]] = None,
    dpi: int = 300
):
    """
    Create 4-panel error map visualization from raw error rates + region names.
    
    Args:
        error_rates: 1D array/list of misclassification rates (length = n_regions)
        title: Figure title
        output_path: Save path
        region_list: List of region names (same order as error_rates)
        dpi: Resolution
    """
    error_rates = np.array(error_rates)
    n_regions = len(error_rates)
    
    if region_list is None:
        region_list = [f"Region_{i:02d}" for i in range(n_regions)]
    
    # Create DataFrame for easier handling
    error_df = pd.DataFrame({
        'region_name': region_list,
        'misclassification_rate': error_rates
    }).sort_values('misclassification_rate', ascending=False).reset_index(drop=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: All regions bar plot (sorted)
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn_r(error_df['misclassification_rate'] / error_df['misclassification_rate'].max())
    bars1 = ax1.bar(range(n_regions), error_df['misclassification_rate'], 
                    color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    mean_err = error_df['misclassification_rate'].mean()
    ax1.axhline(mean_err, color='blue', linestyle='--', linewidth=2, label=f'Mean = {mean_err:.3f}')
    ax1.set_xlabel('Region (sorted by error)', fontweight='bold')
    ax1.set_ylabel('Misclassification Rate', fontweight='bold')
    ax1.set_title('All Regions - Sorted by Error', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Top 20 worst regions
    ax2 = axes[0, 1]
    top_20 = error_df.head(20)
    bars2 = ax2.barh(range(20), top_20['misclassification_rate'], 
                     color='red', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([name[:40] + "..." if len(name) > 40 else name 
                         for name in top_20['region_name']], fontsize=8.5)
    ax2.set_xlabel('Misclassification Rate', fontweight='bold')
    ax2.set_title('Top 20 Most Confused Regions', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Panel 3: Distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(error_df['misclassification_rate'], bins=min(50, n_regions//2), 
             alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)
    ax3.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_err:.3f}')
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
    {'='*25}
    
    Total Regions        → {n_regions}
    Mean Error           → {mean_err:.4f}
    Median Error         → {error_df['misclassification_rate'].median():.4f}
    Std Deviation        → {error_df['misclassification_rate'].std():.4f}
    Min Error            → {error_df['misclassification_rate'].min():.4f}
    Max Error            → {error_df['misclassification_rate'].max():.4f}
    
    High Error (>0.30)   → {(error_rates > 0.30).sum()}
    Low Error (<0.10)    → {(error_rates < 0.10).sum()}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11, family='monospace',
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.9))
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved: {Path(output_path).name}")

def plot_rest_vs_task_comparison(
    error_rest: Union[np.ndarray, List[float]],
    error_task: Union[np.ndarray, List[float]],
    comparison_df: Optional[pd.DataFrame] = None,
    output_path: str = None,
    region_list: Optional[List[str]] = None,
    dpi: int = 300
):
    """
    Create rest vs task comparison figure.
    Now robust: uses comparison_df if provided, otherwise builds it.
    """
    error_rest = np.array(error_rest)
    error_task = np.array(error_task)

    if region_list is None:
        region_list = [f"Region_{i:02d}" for i in range(len(error_rest))]

    # Use provided comparison_df if valid, otherwise build it
    if comparison_df is not None and not comparison_df.empty:
        comp_df = comparison_df.copy()
        # Ensure required columns exist — standardize them
        if 'error_increase' not in comp_df.columns:
            print("Warning: comparison_df missing 'error_increase', rebuilding...")
            comp_df = None
        elif 'region_name' not in comp_df.columns:
            comp_df = None
    else:
        comp_df = None

    if comp_df is None:
        # Rebuild from raw errors
        comp_df = pd.DataFrame({
            'region_name': region_list,
            'error_rest': error_rest,
            'error_task': error_task,
            'error_increase': error_task - error_rest
        }).sort_values('error_increase', ascending=False).reset_index(drop=True)
    else:
        # Standardize column names for consistency
        comp_df = comp_df.rename(columns={
            'error_diff': 'error_increase',
            'diff': 'error_increase',
            'difference': 'error_increase'
        })
        if 'region_name' not in comp_df.columns:
            comp_df['region_name'] = region_list

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(comp_df['error_rest'] if 'error_rest' in comp_df.columns else error_rest,
                comp_df['error_task'] if 'error_task' in comp_df.columns else error_task,
                alpha=0.6, s=40, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No Change')
    ax1.set_xlabel('Resting-State Error Rate', fontweight='bold')
    ax1.set_ylabel('Task Error Rate', fontweight='bold')
    ax1.set_title('Rest vs Task: Classification Stability', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: Top 20 most disrupted regions
    top_20 = comp_df.sort_values('error_increase', ascending=False).head(20)
    ax2 = axes[0, 1]
    colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'lightgray' for x in top_20['error_increase']]
    bars = ax2.barh(range(20), top_20['error_increase'], color=colors, edgecolor='black', alpha=0.8)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([name[:40] + "..." if len(name) > 40 else name 
                         for name in top_20['region_name']], fontsize=8.5)
    ax2.set_xlabel('Error Increase (Task − Rest)', fontweight='bold')
    ax2.set_title('Top 20 Regions Most Affected by Task', fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(0, color='black', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    # Panel 3: Distribution of changes
    changes = comp_df['error_increase']
    ax3 = axes[1, 0]
    ax3.hist(changes, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax3.axvline(changes.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean Δ = {changes.mean():+.4f}')
    ax3.set_xlabel('Error Change (Task − Rest)', fontweight='bold')
    ax3.set_ylabel('Number of Regions', fontweight='bold')
    ax3.set_title('Distribution of Task-Induced Changes', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    n_inc = (changes > 0.05).sum()
    n_dec = (changes < -0.05).sum()
    n_stable = len(changes) - n_inc - n_dec

    stats_text = f"""
    REST vs TASK COMPARISON
    {'='*28}

    Regions Analyzed     → {len(comp_df)}

    Significant Changes (>5%):
      Increased          → {n_inc}
      Decreased          → {n_dec}
      Stable             → {n_stable}

    Mean Change          → {changes.mean():+.4f}
    Max Increase         → {changes.max():+.4f}
    Max Decrease         → {changes.min():+.4f}

    Rest Mean Error      → {error_rest.mean():.4f}
    Task Mean Error      → {error_task.mean():.4f}
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11, family='monospace',
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.9))

    plt.suptitle('Resting-State vs Task: How Task Affects Brain Region Decoding', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved: {Path(output_path).name}")
    else:
        plt.show()