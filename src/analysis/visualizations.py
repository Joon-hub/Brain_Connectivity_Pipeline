"""
Visualization functions for thesis figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_performance_overview(
    summary_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (15, 10)
):
    """
    Figure 1: Performance overview (4 panels).
    
    A) Accuracy comparison (rest vs task)
    B) Accuracy drop by model
    C) Network-level performance heatmap
    D) Top-5 accuracy comparison
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Panel A: Accuracy comparison
    ax = axes[0, 0]
    x = np.arange(len(summary_df))
    width = 0.35
    
    ax.bar(x - width/2, summary_df['rest_cv_accuracy'], width, label='Rest CV', alpha=0.8)
    ax.bar(x + width/2, summary_df['task_test_accuracy'], width, label='Task Test', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('A) Rest CV vs Task Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Accuracy drop
    ax = axes[0, 1]
    colors = plt.cm.Reds(summary_df['accuracy_drop_pct'] / summary_df['accuracy_drop_pct'].max())
    
    ax.barh(range(len(summary_df)), summary_df['accuracy_drop'], color=colors)
    ax.set_yticks(range(len(summary_df)))
    ax.set_yticklabels(summary_df['model'])
    ax.set_xlabel('Accuracy Drop')
    ax.set_title('B) Accuracy Drop by Model')
    ax.axvline(summary_df['accuracy_drop'].mean(), color='red', linestyle='--', 
               label=f'Mean: {summary_df["accuracy_drop"].mean():.3f}')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Panel C: Network heatmap (placeholder - needs network data)
    ax = axes[1, 0]
    ax.text(0.5, 0.5, 'Network-Level\nPerformance Heatmap', 
            ha='center', va='center', fontsize=12)
    ax.set_title('C) Network-Level Performance')
    ax.axis('off')
    
    # Panel D: Top-5 accuracy
    ax = axes[1, 1]
    
    if 'top_5_accuracy' in summary_df.columns:
        x = np.arange(len(summary_df))
        ax.bar(x, summary_df['top_5_accuracy'], alpha=0.8, color='teal')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['model'], rotation=45, ha='right')
        ax.set_ylabel('Top-5 Accuracy')
        ax.set_title('D) Top-5 Accuracy Comparison')
        ax.set_ylim([0, 1])
        ax.axhline(summary_df['top_5_accuracy'].mean(), color='red', linestyle='--',
                   label=f'Mean: {summary_df["top_5_accuracy"].mean():.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_strategy_comparison(
    all_results: Dict[str, Dict],
    save_path: Optional[Path] = None,
    figsize: tuple = (18, 5)
):
    """
    Figure 2: Strategy comparison (3 panels).
    
    A) Accuracy drops by strategy
    B) Network sensitivity by strategy
    C) Strategy correlation
    """
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Collect data
    scopes = ['full', 'left', 'right']
    strategies = ['multinomial', 'ovr', 'ovo']
    
    # Panel A: Accuracy drops
    ax = axes[0]
    
    drops_by_strategy = {s: [] for s in strategies}
    
    for scope in scopes:
        for strategy in strategies:
            model_name = f'{scope}_{strategy}'
            if model_name in all_results:
                drop = all_results[model_name]['summary']['accuracy_drop']
                drops_by_strategy[strategy].append(drop)
    
    x = np.arange(len(scopes))
    width = 0.25
    
    for i, strategy in enumerate(strategies):
        ax.bar(x + i*width, drops_by_strategy[strategy], width, label=strategy.capitalize())
    
    ax.set_xlabel('Scope')
    ax.set_ylabel('Accuracy Drop')
    ax.set_title('A) Accuracy Drop by Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in scopes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B & C: Add your specific visualizations
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_hemisphere_comparison(
    all_results: Dict[str, Dict],
    strategy: str = 'multinomial',
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10)
):
    """
    Figure 3: Hemisphere asymmetry.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get hemisphere data
    left_model = f'left_{strategy}'
    right_model = f'right_{strategy}'
    
    left_summary = all_results[left_model]['summary']
    right_summary = all_results[right_model]['summary']
    
    # Panel A: Overall comparison
    ax = axes[0, 0]
    hemispheres = ['Left', 'Right']
    drops = [left_summary['accuracy_drop'], right_summary['accuracy_drop']]
    
    ax.bar(hemispheres, drops, color=['steelblue', 'coral'], alpha=0.8)
    ax.set_ylabel('Accuracy Drop')
    ax.set_title('A) Overall Hemisphere Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Additional panels...
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_confusion_matrices_grid(
    all_results: Dict[str, Dict],
    save_path: Optional[Path] = None,
    figsize: tuple = (20, 20)
):
    """
    Figure 4: 3x3 grid of confusion matrices.
    """
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    scopes = ['full', 'left', 'right']
    strategies = ['multinomial', 'ovr', 'ovo']
    
    for i, scope in enumerate(scopes):
        for j, strategy in enumerate(strategies):
            model_name = f'{scope}_{strategy}'
            ax = axes[i, j]
            
            if model_name in all_results and 'confusion_matrix' in all_results[model_name]:
                cm = all_results[model_name]['confusion_matrix']
                
                # Normalize
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cm_norm = cm / row_sums
                
                # Plot
                im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
                ax.set_title(f'{scope.capitalize()} - {strategy.capitalize()}')
                
                if j == 0:
                    ax.set_ylabel('True Region')
                if i == 2:
                    ax.set_xlabel('Predicted Region')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_reorganization_map(
    reorganization_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 8)
):
    """
    Figure: Brain reorganization map (ranked bar plot).
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by reorganization index
    sorted_df = reorganization_df.sort_values('reorganization_index', ascending=True)
    
    # Color by network
    network_colors = {net: plt.cm.tab10(i) for i, net in enumerate(sorted_df['network'].unique())}
    colors = [network_colors[net] for net in sorted_df['network']]
    
    y_pos = np.arange(len(sorted_df))
    
    ax.barh(y_pos, sorted_df['reorganization_index'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df['region_name'], fontsize=6)
    ax.set_xlabel('Reorganization Index')
    ax.set_title('Region-Level Functional Reorganization')
    ax.axvline(sorted_df['reorganization_index'].mean(), color='red', linestyle='--',
               label=f'Mean: {sorted_df["reorganization_index"].mean():.3f}')
    
    # Legend for networks
    handles = [plt.Rectangle((0,0),1,1, color=network_colors[net]) 
               for net in network_colors.keys()]
    ax.legend(handles, network_colors.keys(), loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()