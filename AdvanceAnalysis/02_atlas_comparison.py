#!/usr/bin/env python3
"""
Atlas Comparison Analysis - Simplified with Outlier Annotations
Creates 3 multi-panel figures comparing atlas performance.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# =============================================================================
# LOAD DATA
# =============================================================================

def load_error_rates(tables_dir):
    """Load all error rate CSV files."""
    data = {}
    
    files = {
        'N7_rest': 'error_rates_N7_cortical_rest.csv',
        'N7_task': 'error_rates_N7_cortical_task.csv',
        'N17_rest': 'error_rates_N17_cortical_rest.csv',
        'N17_task': 'error_rates_N17_cortical_task.csv',
        'TianI_rest': 'error_rates_TianI_subcortical_rest.csv',
        'TianI_task': 'error_rates_TianI_subcortical_task.csv',
        'TianII_rest': 'error_rates_TianII_subcortical_rest.csv',
        'TianII_task': 'error_rates_TianII_subcortical_task.csv',
        'Combined_rest': 'error_rates_N7_TianI_combined_rest.csv',
        'Combined_task': 'error_rates_N7_TianI_combined_task.csv'
    }
    
    for key, filename in files.items():
        filepath = tables_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
            print(f"  ✓ {filename} ({len(data[key])} networks)")
    
    return data


# =============================================================================
# STATISTICS
# =============================================================================

def compare_resolution(error_data):
    """Compare coarse vs fine parcellations."""
    results = []
    
    # N7 vs N17
    for cond, n7_key, n17_key in [
        ('Rest', 'N7_rest', 'N17_rest'),
        ('Task', 'N7_task', 'N17_task')
    ]:
        if n7_key in error_data and n17_key in error_data:
            n7 = error_data[n7_key]['error_rate']
            n17 = error_data[n17_key]['error_rate']
            
            results.append({
                'comparison': f'Cortical N7 vs N17 ({cond})',
                'coarse_mean': n7.mean(),
                'fine_mean': n17.mean(),
                'difference': n17.mean() - n7.mean(),
                'pct_change': (n17.mean() - n7.mean()) / n7.mean() * 100
            })
    
    # Tian I vs II
    for cond, t1_key, t2_key in [
        ('Rest', 'TianI_rest', 'TianII_rest'),
        ('Task', 'TianI_task', 'TianII_task')
    ]:
        if t1_key in error_data and t2_key in error_data:
            t1 = error_data[t1_key]['error_rate']
            t2 = error_data[t2_key]['error_rate']
            
            results.append({
                'comparison': f'Subcortical Tian I vs II ({cond})',
                'coarse_mean': t1.mean(),
                'fine_mean': t2.mean(),
                'difference': t2.mean() - t1.mean(),
                'pct_change': (t2.mean() - t1.mean()) / t1.mean() * 100
            })
    
    return pd.DataFrame(results)


def compare_cortical_subcortical(error_data):
    """Compare cortical vs subcortical performance."""
    results = []
    
    for cond, n7_key, tian_key in [
        ('Rest', 'N7_rest', 'TianI_rest'),
        ('Task', 'N7_task', 'TianI_task')
    ]:
        if n7_key in error_data and tian_key in error_data:
            cort = error_data[n7_key]['error_rate']
            subcort = error_data[tian_key]['error_rate']
            
            t_stat, p_val = stats.ttest_ind(cort, subcort)
            
            pooled_std = np.sqrt(((len(cort)-1)*cort.std()**2 + 
                                 (len(subcort)-1)*subcort.std()**2) / 
                                (len(cort) + len(subcort) - 2))
            cohens_d = (subcort.mean() - cort.mean()) / pooled_std
            
            results.append({
                'condition': cond,
                'cortical_mean': cort.mean(),
                'subcortical_mean': subcort.mean(),
                'difference': subcort.mean() - cort.mean(),
                'pct_difference': (subcort.mean() - cort.mean()) / cort.mean() * 100,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d
            })
    
    return pd.DataFrame(results)


def compare_rest_task(error_data):
    """Compare rest vs task conditions."""
    results = []
    
    comparisons = [
        ('N7 Cortical', 'N7_rest', 'N7_task'),
        ('N17 Cortical', 'N17_rest', 'N17_task'),
        ('Tian I Subcortical', 'TianI_rest', 'TianI_task'),
        ('Tian II Subcortical', 'TianII_rest', 'TianII_task'),
        ('Combined', 'Combined_rest', 'Combined_task')
    ]
    
    for name, rest_key, task_key in comparisons:
        if rest_key in error_data and task_key in error_data:
            rest_df = error_data[rest_key]
            task_df = error_data[task_key]
            
            merged = pd.merge(
                rest_df[['network', 'error_rate']],
                task_df[['network', 'error_rate']],
                on='network',
                suffixes=('_rest', '_task')
            )
            
            if len(merged) > 0:
                t_stat, p_val = stats.ttest_rel(
                    merged['error_rate_rest'],
                    merged['error_rate_task']
                )
                
                diff = merged['error_rate_task'] - merged['error_rate_rest']
                cohens_d = diff.mean() / diff.std()
                
                results.append({
                    'atlas': name,
                    'rest_mean': merged['error_rate_rest'].mean(),
                    'task_mean': merged['error_rate_task'].mean(),
                    'mean_increase': diff.mean(),
                    'pct_increase': (diff.mean() / merged['error_rate_rest'].mean()) * 100,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'n_networks': len(merged)
                })
    
    return pd.DataFrame(results)


# =============================================================================
# PLOTTING WITH OUTLIER ANNOTATIONS
# =============================================================================

def annotate_outliers(ax, data, positions, labels, n_outliers=3):
    """
    Add text labels for top N outliers in boxplot.
    
    Args:
        ax: matplotlib axis
        data: list of arrays (one per box)
        positions: x positions of boxes
        labels: network names for each data point
        n_outliers: number of outliers to label (default 3)
    """
    for i, (arr, pos) in enumerate(zip(data, positions)):
        if len(arr) == 0:
            continue
        
        # Calculate outlier threshold (1.5 * IQR)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Get network names for this box
        if isinstance(labels[i], pd.Series):
            net_names = labels[i].values
        else:
            net_names = labels[i]
        
        # Find outliers
        outlier_mask = (arr < lower_bound) | (arr > upper_bound)
        outlier_vals = arr[outlier_mask]
        outlier_names = net_names[outlier_mask]
        
        if len(outlier_vals) == 0:
            continue
        
        # Get top N outliers (most extreme)
        outlier_df = pd.DataFrame({
            'value': outlier_vals,
            'name': outlier_names,
            'distance': np.abs(outlier_vals - arr.median())
        })
        outlier_df = outlier_df.nlargest(n_outliers, 'distance')
        
        # Annotate
        for _, row in outlier_df.iterrows():
            ax.annotate(
                row['name'],
                xy=(pos, row['value']),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=8,
                color='darkred',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                         alpha=0.7, edgecolor='darkred'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                              color='darkred', lw=1.5)
            )


def plot_figure1_resolution_systems(error_data, output_path):
    """
    Figure 1: Resolution & Brain System Comparisons (2x2).
    Clean boxplots with outlier annotations, no significance stars.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Figure 1: Parcellation Resolution & Brain System Comparisons', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: N7 vs N17 (Rest)
    ax = axes[0, 0]
    if 'N7_rest' in error_data and 'N17_rest' in error_data:
        n7 = error_data['N7_rest']
        n17 = error_data['N17_rest']
        
        data = [n7['error_rate'], n17['error_rate']]
        labels_list = [n7['network'], n17['network']]
        
        bp = ax.boxplot(data, labels=['N7\n(7 nets)', 'N17\n(17 nets)'], 
                       patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['#4A90E2', '#E24A4A']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add means
        means = [d.mean() for d in data]
        ax.plot(range(1, len(means)+1), means, 'D', color='darkred', 
               markersize=8, label='Mean', zorder=3)
        
        # Annotate outliers
        annotate_outliers(ax, data, [1, 2], labels_list, n_outliers=3)
        
        ax.set_ylabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('A) Cortical Resolution: Coarse vs Fine (Rest)', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel B: N7 vs N17 (Task)
    ax = axes[0, 1]
    if 'N7_task' in error_data and 'N17_task' in error_data:
        n7 = error_data['N7_task']
        n17 = error_data['N17_task']
        
        data = [n7['error_rate'], n17['error_rate']]
        labels_list = [n7['network'], n17['network']]
        
        bp = ax.boxplot(data, labels=['N7\n(7 nets)', 'N17\n(17 nets)'], 
                       patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['#4A90E2', '#E24A4A']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        means = [d.mean() for d in data]
        ax.plot(range(1, len(means)+1), means, 'D', color='darkred', 
               markersize=8, label='Mean', zorder=3)
        
        annotate_outliers(ax, data, [1, 2], labels_list, n_outliers=3)
        
        ax.set_ylabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('B) Cortical Resolution: Coarse vs Fine (Task)', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Cortical vs Subcortical (Rest)
    ax = axes[1, 0]
    if 'N7_rest' in error_data and 'TianI_rest' in error_data:
        cort = error_data['N7_rest']
        subcort = error_data['TianI_rest']
        
        data = [cort['error_rate'], subcort['error_rate']]
        labels_list = [cort['network'], subcort['network']]
        
        bp = ax.boxplot(data, labels=['Cortical\n(7 nets)', 'Subcortical\n(8 regions)'], 
                       patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['#2ECC71', '#F39C12']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        means = [d.mean() for d in data]
        ax.plot(range(1, len(means)+1), means, 'D', color='darkred', 
               markersize=8, label='Mean', zorder=3)
        
        annotate_outliers(ax, data, [1, 2], labels_list, n_outliers=3)
        
        ax.set_ylabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('C) Brain Systems: Cortical vs Subcortical (Rest)', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel D: Cortical vs Subcortical (Task)
    ax = axes[1, 1]
    if 'N7_task' in error_data and 'TianI_task' in error_data:
        cort = error_data['N7_task']
        subcort = error_data['TianI_task']
        
        data = [cort['error_rate'], subcort['error_rate']]
        labels_list = [cort['network'], subcort['network']]
        
        bp = ax.boxplot(data, labels=['Cortical\n(7 nets)', 'Subcortical\n(8 regions)'], 
                       patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['#2ECC71', '#F39C12']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        means = [d.mean() for d in data]
        ax.plot(range(1, len(means)+1), means, 'D', color='darkred', 
               markersize=8, label='Mean', zorder=3)
        
        annotate_outliers(ax, data, [1, 2], labels_list, n_outliers=3)
        
        ax.set_ylabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('D) Brain Systems: Cortical vs Subcortical (Task)', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_figure2_rest_vs_task(error_data, output_path):
    """Figure 2: Rest vs Task Effects (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Figure 2: Rest vs Task Effects', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: Overall comparison
    ax = axes[0, 0]
    rest_task_data = []
    atlas_labels = []
    
    for name, rest_key, task_key in [
        ('N7', 'N7_rest', 'N7_task'),
        ('N17', 'N17_rest', 'N17_task'),
        ('Tian I', 'TianI_rest', 'TianI_task')
    ]:
        if rest_key in error_data and task_key in error_data:
            rest_mean = error_data[rest_key]['error_rate'].mean()
            task_mean = error_data[task_key]['error_rate'].mean()
            rest_task_data.append([rest_mean, task_mean])
            atlas_labels.append(name)
    
    if rest_task_data:
        arr = np.array(rest_task_data)
        x = np.arange(len(atlas_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, arr[:, 0], width, label='Rest', 
                      color='#3498DB', alpha=0.85, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, arr[:, 1], width, label='Task',
                      color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Mean Error Rate', fontweight='bold', fontsize=11)
        ax.set_xlabel('Atlas', fontweight='bold', fontsize=11)
        ax.set_title('A) Rest vs Task: Overall Comparison', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(atlas_labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel B: N7 changes
    ax = axes[0, 1]
    if 'N7_rest' in error_data and 'N7_task' in error_data:
        rest_df = error_data['N7_rest']
        task_df = error_data['N7_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network', suffixes=('_rest', '_task')
        )
        
        merged['change'] = merged['error_rate_task'] - merged['error_rate_rest']
        merged['pct'] = (merged['change'] / merged['error_rate_rest'] * 100)
        merged = merged.sort_values('change', ascending=False)
        
        colors = ['#E74C3C' if x > 0 else '#2ECC71' for x in merged['change']]
        bars = ax.barh(range(len(merged)), merged['change'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for i, (val, pct) in enumerate(zip(merged['change'], merged['pct'])):
            x_pos = val + (0.006 if val > 0 else -0.006)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{pct:+.0f}%', va='center', ha=ha, 
                   fontsize=8, fontweight='bold')
        
        ax.set_yticks(range(len(merged)))
        ax.set_yticklabels(merged['network'], fontsize=9)
        ax.set_xlabel('Error Change (Task - Rest)', fontweight='bold', fontsize=11)
        ax.set_title('B) N7: Task-Induced Changes', fontweight='bold', fontsize=12, pad=10)
        ax.axvline(0, color='black', linewidth=2)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    # Panel C: N17 changes
    ax = axes[1, 0]
    if 'N17_rest' in error_data and 'N17_task' in error_data:
        rest_df = error_data['N17_rest']
        task_df = error_data['N17_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network', suffixes=('_rest', '_task')
        )
        
        merged['change'] = merged['error_rate_task'] - merged['error_rate_rest']
        merged['pct'] = (merged['change'] / merged['error_rate_rest'] * 100)
        merged = merged.sort_values('change', ascending=False)
        
        colors = ['#E74C3C' if x > 0 else '#2ECC71' for x in merged['change']]
        bars = ax.barh(range(len(merged)), merged['change'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for i, (val, pct) in enumerate(zip(merged['change'], merged['pct'])):
            x_pos = val + (0.006 if val > 0 else -0.006)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{pct:+.0f}%', va='center', ha=ha, 
                   fontsize=7, fontweight='bold')
        
        ax.set_yticks(range(len(merged)))
        ax.set_yticklabels(merged['network'], fontsize=8)
        ax.set_xlabel('Error Change (Task - Rest)', fontweight='bold', fontsize=11)
        ax.set_title('C) N17: Task-Induced Changes', fontweight='bold', fontsize=12, pad=10)
        ax.axvline(0, color='black', linewidth=2)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    # Panel D: Tian I changes
    ax = axes[1, 1]
    if 'TianI_rest' in error_data and 'TianI_task' in error_data:
        rest_df = error_data['TianI_rest']
        task_df = error_data['TianI_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network', suffixes=('_rest', '_task')
        )
        
        merged['change'] = merged['error_rate_task'] - merged['error_rate_rest']
        merged['pct'] = (merged['change'] / merged['error_rate_rest'] * 100)
        merged = merged.sort_values('change', ascending=False)
        
        colors = ['#E74C3C' if x > 0 else '#2ECC71' for x in merged['change']]
        bars = ax.barh(range(len(merged)), merged['change'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for i, (val, pct) in enumerate(zip(merged['change'], merged['pct'])):
            x_pos = val + (0.006 if val > 0 else -0.006)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{pct:+.0f}%', va='center', ha=ha, 
                   fontsize=8, fontweight='bold')
        
        ax.set_yticks(range(len(merged)))
        ax.set_yticklabels(merged['network'], fontsize=9)
        ax.set_xlabel('Error Change (Task - Rest)', fontweight='bold', fontsize=11)
        ax.set_title('D) Tian I: Task-Induced Changes', fontweight='bold', fontsize=12, pad=10)
        ax.axvline(0, color='black', linewidth=2)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_figure3_network_detail(error_data, output_path):
    """Figure 3: Network-Level Error Rates (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Figure 3: Network-Level Error Rates (Rest vs Task)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: N7
    ax = axes[0, 0]
    if 'N7_rest' in error_data and 'N7_task' in error_data:
        rest_df = error_data['N7_rest']
        task_df = error_data['N7_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network', suffixes=('_rest', '_task')
        )
        
        merged['avg'] = (merged['error_rate_rest'] + merged['error_rate_task']) / 2
        merged = merged.sort_values('avg', ascending=False)
        
        y_pos = np.arange(len(merged))
        
        ax.barh(y_pos - 0.2, merged['error_rate_rest'], 0.4, label='Rest',
               color='#3498DB', alpha=0.85, edgecolor='black')
        ax.barh(y_pos + 0.2, merged['error_rate_task'], 0.4, label='Task',
               color='#E74C3C', alpha=0.85, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(merged['network'], fontsize=10)
        ax.set_xlabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('A) N7 Cortical Networks', fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    # Panel B: N17
    ax = axes[0, 1]
    if 'N17_rest' in error_data and 'N17_task' in error_data:
        rest_df = error_data['N17_rest']
        task_df = error_data['N17_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network', suffixes=('_rest', '_task')
        )
        
        merged['avg'] = (merged['error_rate_rest'] + merged['error_rate_task']) / 2
        merged = merged.sort_values('avg', ascending=False)
        
        y_pos = np.arange(len(merged))
        
        ax.barh(y_pos - 0.2, merged['error_rate_rest'], 0.4, label='Rest',
               color='#3498DB', alpha=0.85, edgecolor='black')
        ax.barh(y_pos + 0.2, merged['error_rate_task'], 0.4, label='Task',
               color='#E74C3C', alpha=0.85, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(merged['network'], fontsize=8)
        ax.set_xlabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('B) N17 Cortical Networks', fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    # Panel C: Tian I
    ax = axes[1, 0]
    if 'TianI_rest' in error_data and 'TianI_task' in error_data:
        rest_df = error_data['TianI_rest']
        task_df = error_data['TianI_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network', suffixes=('_rest', '_task')
        )
        
        merged['avg'] = (merged['error_rate_rest'] + merged['error_rate_task']) / 2
        merged = merged.sort_values('avg', ascending=False)
        
        y_pos = np.arange(len(merged))
        
        ax.barh(y_pos - 0.2, merged['error_rate_rest'], 0.4, label='Rest',
               color='#3498DB', alpha=0.85, edgecolor='black')
        ax.barh(y_pos + 0.2, merged['error_rate_task'], 0.4, label='Task',
               color='#E74C3C', alpha=0.85, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(merged['network'], fontsize=10)
        ax.set_xlabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('C) Tian I Subcortical', fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    # Panel D: Tian II
    ax = axes[1, 1]
    if 'TianII_rest' in error_data and 'TianII_task' in error_data:
        rest_df = error_data['TianII_rest']
        task_df = error_data['TianII_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network', suffixes=('_rest', '_task')
        )
        
        if len(merged) > 0:
            merged['avg'] = (merged['error_rate_rest'] + merged['error_rate_task']) / 2
            merged = merged.sort_values('avg', ascending=False)
            
            y_pos = np.arange(len(merged))
            
            ax.barh(y_pos - 0.2, merged['error_rate_rest'], 0.4, label='Rest',
                   color='#3498DB', alpha=0.85, edgecolor='black')
            ax.barh(y_pos + 0.2, merged['error_rate_task'], 0.4, label='Task',
                   color='#E74C3C', alpha=0.85, edgecolor='black')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(merged['network'], fontsize=8)
            ax.set_xlabel('Error Rate', fontweight='bold', fontsize=11)
            ax.set_title('D) Tian II Subcortical', fontweight='bold', fontsize=12, pad=10)
            ax.legend(fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No Tian II data', ha='center', va='center', fontsize=12)
            ax.set_title('D) Tian II Subcortical', fontweight='bold', fontsize=12, pad=10)
    else:
        ax.text(0.5, 0.5, 'No Tian II data', ha='center', va='center', fontsize=12)
        ax.set_title('D) Tian II Subcortical', fontweight='bold', fontsize=12, pad=10)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("ATLAS COMPARISON ANALYSIS - SIMPLIFIED")
    print("="*80)
    
    # Load data
    tables_dir = Path('reports/tables/atlas_analysis')
    
    if not tables_dir.exists():
        print(f"\n✗ Error: {tables_dir} not found")
        print("Run 01_atlas_performance_analysis first!")
        return 1
    
    print("\nLoading error rates...")
    error_data = load_error_rates(tables_dir)
    
    if not error_data:
        print("✗ No data found!")
        return 1
    
    print(f"\n✓ Loaded {len(error_data)} files")
    
    # Output directories
    output_tables = Path('reports/tables/atlas_comparison')
    output_figures = Path('reports/figures/atlas_comparison')
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print("\n1. Resolution effects...")
    resolution = compare_resolution(error_data)
    if not resolution.empty:
        resolution.to_csv(output_tables / 'resolution_comparison.csv', index=False)
        print("✓ Resolution comparison saved")
    
    print("\n2. Cortical vs subcortical...")
    cort_subcort = compare_cortical_subcortical(error_data)
    if not cort_subcort.empty:
        cort_subcort.to_csv(output_tables / 'cortical_vs_subcortical.csv', index=False)
        print("✓ Cortical vs subcortical saved")
    
    print("\n3. Rest vs task...")
    rest_task = compare_rest_task(error_data)
    if not rest_task.empty:
        rest_task.to_csv(output_tables / 'rest_vs_task_comparison.csv', index=False)
        print("✓ Rest vs task saved")
    
    # Figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    print("\nFigure 1: Resolution & Systems (with outlier annotations)...")
    plot_figure1_resolution_systems(error_data, 
                                   output_figures / 'figure1_resolution_systems.png')
    
    print("\nFigure 2: Rest vs Task...")
    plot_figure2_rest_vs_task(error_data, 
                             output_figures / 'figure2_rest_vs_task.png')
    
    print("\nFigure 3: Network Detail...")
    plot_figure3_network_detail(error_data, 
                                output_figures / 'figure3_network_detail.png')
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if not resolution.empty:
        print("\nResolution Effects:")
        for _, row in resolution.iterrows():
            print(f"  {row['comparison']}: {row['difference']:+.4f} ({row['pct_change']:+.1f}%)")
    
    if not cort_subcort.empty:
        print("\nCortical vs Subcortical:")
        for _, row in cort_subcort.iterrows():
            print(f"  {row['condition']}: {row['difference']:+.4f} (p={row['p_value']:.4f}, d={row['cohens_d']:.2f})")
    
    if not rest_task.empty:
        print("\nRest vs Task:")
        for _, row in rest_task.iterrows():
            print(f"  {row['atlas']}: {row['mean_increase']:+.4f} ({row['pct_increase']:+.1f}%, p={row['p_value']:.4f})")
    
    print(f"""
Generated Files:
===============
Tables: {output_tables}
  • resolution_comparison.csv
  • cortical_vs_subcortical.csv
  • rest_vs_task_comparison.csv

Figures: {output_figures}
  • figure1_resolution_systems.png (with outlier annotations)
  • figure2_rest_vs_task.png
  • figure3_network_detail.png

✓ All boxplots show top 3 outliers with network labels
✓ No significance stars or p-values on plots
✓ Clean, easy-to-read visualizations
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())