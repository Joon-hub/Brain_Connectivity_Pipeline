"""
hemisphere_viz.py

Visualization functions for hemisphere-specific brain region classification results.
Generates publication-quality figures for confusion matrices, per-region performance,
and network-level analyses.

Author: Joon
Date: 2024
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    region_info: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Confusion Matrix',
    normalize: bool = False,
    show_values: bool = False,
    cmap: str = 'Blues',
    figsize: Tuple[float, float] = (12, 10),
    network_boundaries: bool = True
):
    """
    Plot confusion matrix with optional network boundaries.
    
    Parameters
    ----------
    confusion_mat : np.ndarray
        Confusion matrix, shape (n_classes, n_classes)
    region_info : pd.DataFrame, optional
        Region information with 'region_id', 'region_name', 'network' columns
    save_path : str or Path, optional
        Path to save figure
    title : str, default='Confusion Matrix'
        Figure title
    normalize : bool, default=False
        Normalize confusion matrix by row (true class)
    show_values : bool, default=False
        Show numerical values in cells (not recommended for large matrices)
    cmap : str, default='Blues'
        Colormap name
    figsize : tuple, default=(12, 10)
        Figure size in inches
    network_boundaries : bool, default=True
        Draw lines separating functional networks
    """
    
    logger.info(f"Plotting confusion matrix: {confusion_mat.shape}")
    
    # Normalize if requested
    if normalize:
        confusion_mat = confusion_mat.astype('float')
        row_sums = confusion_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        confusion_mat = confusion_mat / row_sums
        logger.info("  Normalized by row (true class)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(confusion_mat, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if normalize:
        cbar.set_label('Proportion of True Class', rotation=270, labelpad=20)
    else:
        cbar.set_label('Number of Samples', rotation=270, labelpad=20)
    
    # Add network boundaries if requested and info available
    if network_boundaries and region_info is not None and 'network' in region_info.columns:
        _add_network_boundaries(ax, region_info)
    
    # Labels
    n_classes = confusion_mat.shape[0]
    
    # For large matrices, reduce tick density
    if n_classes > 50:
        tick_spacing = max(10, n_classes // 10)
        tick_positions = np.arange(0, n_classes, tick_spacing)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions, rotation=90)
        ax.set_yticklabels(tick_positions)
    else:
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        if region_info is not None and 'region_name' in region_info.columns:
            region_names = region_info.sort_values('region_id')['region_name'].values[:n_classes]
            ax.set_xticklabels(region_names, rotation=90, ha='right')
            ax.set_yticklabels(region_names)
        else:
            ax.set_xticklabels(np.arange(n_classes), rotation=90)
            ax.set_yticklabels(np.arange(n_classes))
    
    # Axis labels
    ax.set_xlabel('Predicted Region', fontweight='bold')
    ax.set_ylabel('True Region', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Show values in cells (only for small matrices)
    if show_values and n_classes <= 20:
        for i in range(n_classes):
            for j in range(n_classes):
                value = confusion_mat[i, j]
                if normalize:
                    text = f'{value:.2f}'
                else:
                    text = f'{int(value)}'
                color = 'white' if value > confusion_mat.max() * 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    # Add accuracy on diagonal annotation
    accuracy = np.trace(confusion_mat) / np.sum(confusion_mat)
    ax.text(
        0.02, 0.98, f'Accuracy: {accuracy:.3f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {save_path}")
    
    plt.close()


def _add_network_boundaries(ax, region_info: pd.DataFrame):
    """Add lines to separate functional networks in confusion matrix."""
    
    # Get network boundaries
    region_info_sorted = region_info.sort_values('region_id')
    networks = region_info_sorted['network'].values
    
    # Find where network changes
    boundaries = []
    current_network = networks[0]
    
    for i, network in enumerate(networks[1:], start=1):
        if network != current_network:
            boundaries.append(i)
            current_network = network
    
    # Draw lines
    for boundary in boundaries:
        ax.axhline(y=boundary - 0.5, color='red', linewidth=1.5, alpha=0.7)
        ax.axvline(x=boundary - 0.5, color='red', linewidth=1.5, alpha=0.7)
    
    logger.info(f"  Added {len(boundaries)} network boundaries")


def plot_per_region_accuracy(
    per_region_metrics: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Per-Region Classification Accuracy',
    figsize: Tuple[float, float] = (14, 8),
    color_by_network: bool = True,
    show_threshold: bool = True,
    threshold: float = 0.8
):
    """
    Plot per-region classification accuracy as a bar chart.
    
    Parameters
    ----------
    per_region_metrics : pd.DataFrame
        Per-region metrics with 'region_id', 'accuracy', and optionally 'network'
    save_path : str or Path, optional
        Path to save figure
    title : str
        Figure title
    figsize : tuple
        Figure size
    color_by_network : bool, default=True
        Color bars by network if available
    show_threshold : bool, default=True
        Show horizontal line at threshold
    threshold : float, default=0.8
        Threshold value for reference line
    """
    
    logger.info(f"Plotting per-region accuracy for {len(per_region_metrics)} regions")
    
    # Sort by accuracy (descending)
    df = per_region_metrics.sort_values('accuracy', ascending=True).reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine colors
    if color_by_network and 'network' in df.columns:
        # Get unique networks and assign colors
        networks = df['network'].unique()
        network_colors = plt.cm.tab10(np.linspace(0, 1, len(networks)))
        color_map = dict(zip(networks, network_colors))
        colors = [color_map[net] for net in df['network']]
        
        # Create legend
        handles = [
            mpatches.Patch(color=color_map[net], label=net)
            for net in networks
        ]
        ax.legend(
            handles=handles,
            title='Network',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True
        )
    else:
        colors = 'steelblue'
    
    # Create bar chart
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['accuracy'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Threshold line
    if show_threshold:
        ax.axvline(
            x=threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label=f'Threshold ({threshold:.2f})'
        )
    
    # Labels
    ax.set_yticks(y_pos)
    
    # For large number of regions, show every nth label
    if len(df) > 30:
        tick_spacing = max(5, len(df) // 20)
        tick_positions = y_pos[::tick_spacing]
        ax.set_yticks(tick_positions)
        
        if 'region_name' in df.columns:
            labels = df['region_name'].values[::tick_spacing]
        else:
            labels = df['region_id'].values[::tick_spacing]
        
        ax.set_yticklabels(labels, fontsize=8)
    else:
        if 'region_name' in df.columns:
            ax.set_yticklabels(df['region_name'], fontsize=8)
        else:
            ax.set_yticklabels(df['region_id'], fontsize=8)
    
    ax.set_xlabel('Classification Accuracy', fontweight='bold')
    ax.set_ylabel('Brain Region', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add summary statistics
    mean_acc = df['accuracy'].mean()
    std_acc = df['accuracy'].std()
    summary_text = f'Mean: {mean_acc:.3f} ± {std_acc:.3f}'
    
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {save_path}")
    
    plt.close()


def plot_network_accuracy(
    network_metrics: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Network-Level Classification Accuracy',
    figsize: Tuple[float, float] = (10, 6),
    show_error_bars: bool = True
):
    """
    Plot network-level classification accuracy.
    
    Parameters
    ----------
    network_metrics : pd.DataFrame
        Network metrics with 'network', 'accuracy' columns
    save_path : str or Path, optional
        Path to save figure
    title : str
        Figure title
    figsize : tuple
        Figure size
    show_error_bars : bool, default=True
        Show error bars if std/sem available
    """
    
    logger.info(f"Plotting network accuracy for {len(network_metrics)} networks")
    
    # Sort by accuracy (descending)
    df = network_metrics.sort_values('accuracy', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    x_pos = np.arange(len(df))
    
    # Use distinct colors for each network
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    
    bars = ax.bar(
        x_pos,
        df['accuracy'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, df['accuracy'])):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{acc:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=9
        )
    
    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['network'], rotation=45, ha='right')
    ax.set_ylabel('Classification Accuracy', fontweight='bold')
    ax.set_xlabel('Functional Network', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line at mean accuracy
    mean_acc = df['accuracy'].mean()
    ax.axhline(
        y=mean_acc,
        color='red',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label=f'Mean ({mean_acc:.3f})'
    )
    ax.legend()
    
    # Add summary statistics
    std_acc = df['accuracy'].std()
    summary_text = f'Mean: {mean_acc:.3f} ± {std_acc:.3f}'
    
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {save_path}")
    
    plt.close()


def plot_hemisphere_comparison(
    left_metrics: Dict[str, float],
    right_metrics: Dict[str, float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Hemisphere Performance Comparison',
    figsize: Tuple[float, float] = (10, 6)
):
    """
    Plot comparison of left vs right hemisphere performance.
    
    Parameters
    ----------
    left_metrics : dict
        Metrics for left hemisphere
    right_metrics : dict
        Metrics for right hemisphere
    save_path : str or Path, optional
        Path to save figure
    title : str
        Figure title
    figsize : tuple
        Figure size
    """
    
    logger.info("Plotting hemisphere comparison")
    
    # Extract common metrics
    metric_names = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'Precision', 'Recall', 'F1 Score']
    
    left_values = [left_metrics.get(m, 0) for m in metric_names]
    right_values = [right_metrics.get(m, 0) for m in metric_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(metric_labels))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(
        x_pos - width/2,
        left_values,
        width,
        label='Left Hemisphere',
        color='steelblue',
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    bars2 = ax.bar(
        x_pos + width/2,
        right_values,
        width,
        label='Right Hemisphere',
        color='coral',
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
    
    # Labels and formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {save_path}")
    
    plt.close()


def plot_per_region_comparison(
    left_per_region: pd.DataFrame,
    right_per_region: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Per-Region Accuracy: Left vs Right Hemisphere',
    figsize: Tuple[float, float] = (10, 10)
):
    """
    Plot scatter plot comparing per-region accuracy between hemispheres.
    
    Parameters
    ----------
    left_per_region : pd.DataFrame
        Per-region metrics for left hemisphere
    right_per_region : pd.DataFrame
        Per-region metrics for right hemisphere
    save_path : str or Path, optional
        Path to save figure
    title : str
        Figure title
    figsize : tuple
        Figure size
    """
    
    logger.info("Plotting per-region hemisphere comparison")
    
    # Align by region_id
    merged = pd.merge(
        left_per_region[['region_id', 'accuracy']],
        right_per_region[['region_id', 'accuracy']],
        on='region_id',
        suffixes=('_left', '_right')
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    scatter = ax.scatter(
        merged['accuracy_left'],
        merged['accuracy_right'],
        alpha=0.6,
        s=50,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Diagonal line (perfect correlation)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Correlation')
    
    # Calculate correlation
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(merged['accuracy_left'], merged['accuracy_right'])
    
    # Add correlation text
    corr_text = f'r = {corr:.3f}\np = {p_value:.4f}'
    ax.text(
        0.05, 0.95, corr_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontweight='bold'
    )
    
    # Labels
    ax.set_xlabel('Left Hemisphere Accuracy', fontweight='bold')
    ax.set_ylabel('Right Hemisphere Accuracy', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {save_path}")
    
    plt.close()


def plot_error_distribution(
    confusion_mat: np.ndarray,
    region_info: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Error Distribution Analysis',
    figsize: Tuple[float, float] = (14, 6)
):
    """
    Plot distribution of classification errors.
    
    Parameters
    ----------
    confusion_mat : np.ndarray
        Confusion matrix
    region_info : pd.DataFrame, optional
        Region information
    save_path : str or Path, optional
        Path to save figure
    title : str
        Figure title
    figsize : tuple
        Figure size
    """
    
    logger.info("Plotting error distribution")
    
    # Calculate per-region error rates
    n_classes = confusion_mat.shape[0]
    error_rates = []
    
    for i in range(n_classes):
        total = confusion_mat[i].sum()
        correct = confusion_mat[i, i]
        error_rate = (total - correct) / total if total > 0 else 0
        error_rates.append(error_rate)
    
    error_rates = np.array(error_rates)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Histogram of error rates
    ax1.hist(error_rates, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(
        error_rates.mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {error_rates.mean():.3f}'
    )
    ax1.set_xlabel('Error Rate', fontweight='bold')
    ax1.set_ylabel('Number of Regions', fontweight='bold')
    ax1.set_title('Distribution of Per-Region Error Rates', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Plot 2: Error rate by network (if available)
    if region_info is not None and 'network' in region_info.columns:
        # Group by network
        region_info_sorted = region_info.sort_values('region_id')
        networks = region_info_sorted['network'].values[:n_classes]
        
        # Calculate mean error rate per network
        network_errors = {}
        for network in np.unique(networks):
            mask = networks == network
            network_errors[network] = error_rates[mask].mean()
        
        # Plot
        network_names = list(network_errors.keys())
        network_error_values = list(network_errors.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(network_names)))
        
        bars = ax2.bar(
            range(len(network_names)),
            network_error_values,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        
        # Add value labels
        for bar, val in zip(bars, network_error_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
        
        ax2.set_xticks(range(len(network_names)))
        ax2.set_xticklabels(network_names, rotation=45, ha='right')
        ax2.set_ylabel('Mean Error Rate', fontweight='bold')
        ax2.set_xlabel('Functional Network', fontweight='bold')
        ax2.set_title('Error Rate by Network', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        # Just show top errors
        top_errors_idx = np.argsort(error_rates)[-10:]
        top_errors = error_rates[top_errors_idx]
        
        ax2.barh(range(10), top_errors, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(10))
        ax2.set_yticklabels([f'Region {i}' for i in top_errors_idx])
        ax2.set_xlabel('Error Rate', fontweight='bold')
        ax2.set_title('Top 10 Regions with Highest Error Rate', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {save_path}")
    
    plt.close()


def create_summary_figure(
    confusion_mat: np.ndarray,
    per_region_metrics: pd.DataFrame,
    network_metrics: pd.DataFrame,
    overall_metrics: Dict[str, float],
    region_info: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = 'Classification Results Summary',
    figsize: Tuple[float, float] = (16, 12)
):
    """
    Create a comprehensive summary figure with multiple panels.
    
    Parameters
    ----------
    confusion_mat : np.ndarray
        Confusion matrix
    per_region_metrics : pd.DataFrame
        Per-region metrics
    network_metrics : pd.DataFrame
        Network-level metrics
    overall_metrics : dict
        Overall classification metrics
    region_info : pd.DataFrame, optional
        Region information
    save_path : str or Path, optional
        Path to save figure
    title : str
        Figure title
    figsize : tuple
        Figure size
    """
    
    logger.info("Creating comprehensive summary figure")
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Confusion matrix (top left, spans 2 rows)
    ax1 = fig.add_subplot(gs[0:2, 0])
    im = ax1.imshow(confusion_mat, cmap='Blues', aspect='auto')
    ax1.set_title('Confusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    plt.colorbar(im, ax=ax1, fraction=0.046)
    
    # Panel 2: Per-region accuracy (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    accuracies = per_region_metrics.sort_values('accuracy')['accuracy'].values
    ax2.hist(accuracies, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(accuracies.mean(), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Count')
    ax2.set_title('Per-Region Accuracy Distribution', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Panel 3: Network accuracy (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    if len(network_metrics) > 0:
        network_metrics_sorted = network_metrics.sort_values('accuracy', ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(network_metrics_sorted)))
        ax3.bar(
            range(len(network_metrics_sorted)),
            network_metrics_sorted['accuracy'],
            color=colors,
            alpha=0.8,
            edgecolor='black'
        )
        ax3.set_xticks(range(len(network_metrics_sorted)))
        ax3.set_xticklabels(network_metrics_sorted['network'], rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Network-Level Accuracy', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Overall metrics (bottom, spans both columns)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create metrics table
    metrics_text = "OVERALL METRICS\n" + "="*60 + "\n"
    for key, value in overall_metrics.items():
        if isinstance(value, float):
            metrics_text += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
    
    ax4.text(
        0.5, 0.5, metrics_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace'
    )
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {save_path}")
    
    plt.close()


# Example usage and testing
if __name__ == "__main__":
    """Test hemisphere visualization functions."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Hemisphere Visualization Functions")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_classes = 116
    
    # Confusion matrix
    cm = np.random.randint(0, 100, (n_classes, n_classes))
    cm = cm + cm.T  # Make symmetric
    np.fill_diagonal(cm, cm.diagonal() * 5)  # Higher diagonal
    
    # Region info
    region_info = pd.DataFrame({
        'region_id': range(n_classes),
        'region_name': [f'Region_{i}' for i in range(n_classes)],
        'network': [f'Network_{i%7}' for i in range(n_classes)]
    })
    
    # Per-region metrics
    per_region_metrics = pd.DataFrame({
        'region_id': range(n_classes),
        'accuracy': np.random.uniform(0.7, 0.98, n_classes),
        'network': [f'Network_{i%7}' for i in range(n_classes)]
    })
    
    # Network metrics
    network_metrics = pd.DataFrame({
        'network': [f'Network_{i}' for i in range(7)],
        'accuracy': np.random.uniform(0.8, 0.95, 7)
    })
    
    # Overall metrics
    overall_metrics = {
        'accuracy': 0.92,
        'balanced_accuracy': 0.91,
        'f1_score': 0.90
    }
    
    # Test plots
    print("\nTesting plot_confusion_matrix...")
    plot_confusion_matrix(cm, region_info, title='Test Confusion Matrix')
    
    print("\nTesting plot_per_region_accuracy...")
    plot_per_region_accuracy(per_region_metrics, title='Test Per-Region Accuracy')
    
    print("\nTesting plot_network_accuracy...")
    plot_network_accuracy(network_metrics, title='Test Network Accuracy')
    
    print("\nTesting plot_error_distribution...")
    plot_error_distribution(cm, region_info, title='Test Error Distribution')
    
    print("\n" + "="*60)
    print("Testing complete! (Figures not saved in test mode)")