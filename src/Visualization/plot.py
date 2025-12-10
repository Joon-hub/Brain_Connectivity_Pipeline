# src/visualize.py
"""
Visualization Module (Updated for Phase 2 Pipeline - WITH ACCURACY METRICS)
===========================================================================
Creates publication-quality 4-panel figures with proper region names.
NOW INCLUDES ACCURACY alongside error rates for better interpretation.
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
    Create 4-panel error map visualization with ACCURACY METRICS.
    
    Args:
        error_rates: 1D array/list of misclassification rates (length = n_regions)
        title: Figure title
        output_path: Save path
        region_list: List of region names (same order as error_rates)
        dpi: Resolution
    """
    error_rates = np.array(error_rates)
    n_regions = len(error_rates)
    
    # MODIFIED: Calculate accuracy from error rates
    accuracy_rates = 1.0 - error_rates
    
    if region_list is None:
        region_list = [f"Region_{i:02d}" for i in range(n_regions)]
    
    # Create DataFrame for easier handling
    error_df = pd.DataFrame({
        'region_name': region_list,
        'misclassification_rate': error_rates,
        'accuracy': accuracy_rates  # MODIFIED: Added accuracy column
    }).sort_values('misclassification_rate', ascending=False).reset_index(drop=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: All regions bar plot (sorted) - MODIFIED: Show accuracy reference
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn_r(error_df['misclassification_rate'] / error_df['misclassification_rate'].max())
    bars1 = ax1.bar(range(n_regions), error_df['misclassification_rate'], 
                    color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    mean_err = error_df['misclassification_rate'].mean()
    mean_acc = error_df['accuracy'].mean()  # MODIFIED
    
    # MODIFIED: Add both error and accuracy reference lines
    ax1.axhline(mean_err, color='red', linestyle='--', linewidth=2, 
                label=f'Mean Error = {mean_err:.3f}')
    
    # MODIFIED: Add secondary axis for accuracy
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.invert_yaxis()  # Invert so high accuracy is at top
    ax1_right.set_ylabel('Accuracy', fontweight='bold', color='blue')
    ax1_right.tick_params(axis='y', labelcolor='blue')
    
    # Add accuracy reference line on twin axis
    ax1_right.axhline(mean_acc, color='blue', linestyle=':', linewidth=2, 
                      label=f'Mean Accuracy = {mean_acc:.3f}')
    
    ax1.set_xlabel('Region (sorted by error)', fontweight='bold')
    ax1.set_ylabel('Misclassification Rate', fontweight='bold', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_title('All Regions - Sorted by Error', fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Top 20 worst regions (unchanged)
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
    
    # Panel 3: Distribution histogram (unchanged)
    ax3 = axes[1, 0]
    ax3.hist(error_df['misclassification_rate'], bins=min(50, n_regions//2), 
             alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)
    ax3.axvline(mean_err, color='red', linestyle='--', linewidth=2, 
                label=f'Mean Error = {mean_err:.3f}')
    ax3.set_xlabel('Misclassification Rate', fontweight='bold')
    ax3.set_ylabel('Number of Regions', fontweight='bold')
    ax3.set_title('Distribution of Error Rates', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Summary statistics - MODIFIED: Added accuracy metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # MODIFIED: Calculate accuracy statistics
    median_acc = error_df['accuracy'].median()
    std_acc = error_df['accuracy'].std()
    min_acc = error_df['accuracy'].min()
    max_acc = error_df['accuracy'].max()
    high_acc_count = (accuracy_rates > 0.70).sum()
    med_acc_count = ((accuracy_rates >= 0.50) & (accuracy_rates <= 0.70)).sum()
    low_acc_count = (accuracy_rates < 0.50).sum()
    
    stats_text = f"""
    SUMMARY STATISTICS
    {'='*25}
    
    Total Regions        → {n_regions}
    
    ACCURACY METRICS:
    Mean Accuracy        → {mean_acc:.4f}
    Median Accuracy      → {median_acc:.4f}
    Std Deviation        → {std_acc:.4f}
    Min Accuracy         → {min_acc:.4f}
    Max Accuracy         → {max_acc:.4f}
    
    High Accuracy (>70%) → {high_acc_count}
    Med Accuracy (50-70%) → {med_acc_count}
    Low Accuracy (<50%)  → {low_acc_count}
    
    ERROR METRICS:
    Mean Error           → {mean_err:.4f}
    Median Error         → {error_df['misclassification_rate'].median():.4f}
    Std Deviation        → {error_df['misclassification_rate'].std():.4f}
    Min Error            → {error_df['misclassification_rate'].min():.4f}
    Max Error            → {error_df['misclassification_rate'].max():.4f}
    
    High Error (>30%)    → {(error_rates > 0.30).sum()}
    Low Error (<10%)     → {(error_rates < 0.10).sum()}
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
    Create rest vs task comparison figure with ACCURACY METRICS.
    
    Args:
        error_rest: Rest error rates
        error_task: Task error rates
        comparison_df: Optional pre-computed comparison
        output_path: Save path
        region_list: Region names
        dpi: Resolution
    """
    error_rest = np.array(error_rest)
    error_task = np.array(error_task)
    
    # MODIFIED: Calculate accuracies
    acc_rest = 1.0 - error_rest
    acc_task = 1.0 - error_task

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
            'error_increase': error_task - error_rest,
            'acc_rest': acc_rest,  # MODIFIED: Added accuracy columns
            'acc_task': acc_task,
            'acc_change': acc_task - acc_rest
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
        # MODIFIED: Add accuracy columns if not present
        if 'acc_rest' not in comp_df.columns:
            comp_df['acc_rest'] = 1.0 - comp_df['error_rest']
        if 'acc_task' not in comp_df.columns:
            comp_df['acc_task'] = 1.0 - comp_df['error_task']
        if 'acc_change' not in comp_df.columns:
            comp_df['acc_change'] = comp_df['acc_task'] - comp_df['acc_rest']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Scatter plot - MODIFIED: Show accuracy on axes
    ax1 = axes[0, 0]
    ax1.scatter(comp_df['error_rest'] if 'error_rest' in comp_df.columns else error_rest,
                comp_df['error_task'] if 'error_task' in comp_df.columns else error_task,
                alpha=0.6, s=40, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No Change')
    
    # MODIFIED: Add accuracy values to axis labels
    mean_err_rest = error_rest.mean()
    mean_err_task = error_task.mean()
    mean_acc_rest = acc_rest.mean()
    mean_acc_task = acc_task.mean()
    
    ax1.set_xlabel(f'Rest Error Rate (Accuracy: {mean_acc_rest:.3f})', fontweight='bold')
    ax1.set_ylabel(f'Task Error Rate (Accuracy: {mean_acc_task:.3f})', fontweight='bold')
    ax1.set_title('Rest vs Task: Classification Stability', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: Top 20 most disrupted regions (unchanged)
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

    # Panel 3: Distribution of changes (unchanged)
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

    # Panel 4: Summary - MODIFIED: Added comprehensive accuracy metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    n_inc = (changes > 0.05).sum()
    n_dec = (changes < -0.05).sum()
    n_stable = len(changes) - n_inc - n_dec

    # MODIFIED: Calculate accuracy changes
    acc_changes = comp_df['acc_change'] if 'acc_change' in comp_df.columns else (acc_task - acc_rest)
    n_acc_improved = (acc_changes > 0.05).sum()
    n_acc_degraded = (acc_changes < -0.05).sum()

    stats_text = f"""
    REST vs TASK COMPARISON
    {'='*28}

    Regions Analyzed     → {len(comp_df)}

    ACCURACY COMPARISON:
    Rest Mean Accuracy   → {mean_acc_rest:.4f}
    Task Mean Accuracy   → {mean_acc_task:.4f}
    Accuracy Change      → {(mean_acc_task - mean_acc_rest):+.4f}
    
    Regions Improved     → {n_acc_improved}
    Regions Degraded     → {n_acc_degraded}
    
    ERROR COMPARISON:
    Rest Mean Error      → {mean_err_rest:.4f}
    Task Mean Error      → {mean_err_task:.4f}
    Error Change         → {(mean_err_task - mean_err_rest):+.4f}

    Significant Changes (>5%):
      Error Increased    → {n_inc}
      Error Decreased    → {n_dec}
      Stable             → {n_stable}

    Extreme Changes:
    Max Error Increase   → {changes.max():+.4f}
    Max Error Decrease   → {changes.min():+.4f}
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