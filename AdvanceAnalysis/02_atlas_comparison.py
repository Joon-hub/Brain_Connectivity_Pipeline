#!/usr/bin/env python3
"""
Atlas Comparison Analysis - Combined Multi-Panel Figures
=========================================================
Compares model performance across different atlas configurations with
combined visualization figures organized by logical groupings.

Generates 3 comprehensive multi-panel figures:
1. Figure 1 (2x2): Resolution & Brain System Comparisons
   - Panel A: N7 vs N17 (Rest)
   - Panel B: N7 vs N17 (Task)
   - Panel C: Cortical vs Subcortical (Rest)
   - Panel D: Cortical vs Subcortical (Task)

2. Figure 2 (2x2): Rest vs Task Overview
   - Panel A: Mean error rates across atlases
   - Panel B: N7 network changes
   - Panel C: N17 network changes
   - Panel D: Tian I network changes

3. Figure 3 (2x2): Detailed Network-Level Analysis
   - Panel A: N7 error bars (Rest vs Task)
   - Panel B: N17 error bars (Rest vs Task)
   - Panel C: Tian I error bars (Rest vs Task)
   - Panel D: Tian II error bars (Rest vs Task)

Usage:
    python 02_atlas_comparison_combined.py

Requirements:
    - Must run 01_atlas_performance_analysis_refactored.py first
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 9


def load_error_rates(tables_dir):
    """Load all error rate CSV files from refactored script."""
    error_data = {}
    
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
            error_data[key] = pd.read_csv(filepath)
            print(f"  ✓ Loaded {filename} ({len(error_data[key])} networks)")
        else:
            print(f"  ⚠ Warning: {filename} not found")
    
    return error_data


# =============================================================================
# STATISTICAL COMPARISON FUNCTIONS
# =============================================================================

def compare_resolution_effects(error_data):
    """Compare performance between coarse and fine parcellations."""
    results = []
    
    for condition, rest_n7, task_n7, rest_n17, task_n17 in [
        ('Rest', 'N7_rest', None, 'N17_rest', None),
        ('Task', None, 'N7_task', None, 'N17_task')
    ]:
        n7_key = rest_n7 if rest_n7 else task_n7
        n17_key = rest_n17 if rest_n17 else task_n17
        
        if n7_key in error_data and n17_key in error_data:
            n7_mean = error_data[n7_key]['error_rate'].mean()
            n7_std = error_data[n7_key]['error_rate'].std()
            n17_mean = error_data[n17_key]['error_rate'].mean()
            n17_std = error_data[n17_key]['error_rate'].std()
            
            results.append({
                'comparison': f'Cortical: N7 vs N17 ({condition})',
                'coarse_mean': n7_mean,
                'coarse_std': n7_std,
                'fine_mean': n17_mean,
                'fine_std': n17_std,
                'difference': n17_mean - n7_mean,
                'pct_change': ((n17_mean - n7_mean) / n7_mean * 100) if n7_mean > 0 else 0,
                'coarse_n': len(error_data[n7_key]),
                'fine_n': len(error_data[n17_key])
            })
    
    # Tian I vs II
    for condition, tian1_key, tian2_key in [
        ('Rest', 'TianI_rest', 'TianII_rest'),
        ('Task', 'TianI_task', 'TianII_task')
    ]:
        if tian1_key in error_data and tian2_key in error_data:
            t1_mean = error_data[tian1_key]['error_rate'].mean()
            t1_std = error_data[tian1_key]['error_rate'].std()
            t2_mean = error_data[tian2_key]['error_rate'].mean()
            t2_std = error_data[tian2_key]['error_rate'].std()
            
            results.append({
                'comparison': f'Subcortical: Tian I vs II ({condition})',
                'coarse_mean': t1_mean,
                'coarse_std': t1_std,
                'fine_mean': t2_mean,
                'fine_std': t2_std,
                'difference': t2_mean - t1_mean,
                'pct_change': ((t2_mean - t1_mean) / t1_mean * 100) if t1_mean > 0 else 0,
                'coarse_n': len(error_data[tian1_key]),
                'fine_n': len(error_data[tian2_key])
            })
    
    return pd.DataFrame(results)


def compare_cortical_vs_subcortical(error_data):
    """Compare error rates between cortical (N7) and subcortical (Tian I) regions."""
    results = []
    
    for condition, n7_key, tian_key in [
        ('Rest', 'N7_rest', 'TianI_rest'),
        ('Task', 'N7_task', 'TianI_task')
    ]:
        if n7_key in error_data and tian_key in error_data:
            cortical = error_data[n7_key]
            subcortical = error_data[tian_key]
            
            if len(cortical) > 0 and len(subcortical) > 0:
                cort_mean = cortical['error_rate'].mean()
                cort_std = cortical['error_rate'].std()
                subcort_mean = subcortical['error_rate'].mean()
                subcort_std = subcortical['error_rate'].std()
                
                t_stat, p_val = stats.ttest_ind(
                    cortical['error_rate'],
                    subcortical['error_rate']
                )
                
                pooled_std = np.sqrt(((len(cortical)-1)*cort_std**2 + (len(subcortical)-1)*subcort_std**2) / 
                                    (len(cortical) + len(subcortical) - 2))
                cohens_d = (subcort_mean - cort_mean) / pooled_std if pooled_std > 0 else 0
                
                results.append({
                    'condition': condition,
                    'cortical_mean': cort_mean,
                    'cortical_std': cort_std,
                    'subcortical_mean': subcort_mean,
                    'subcortical_std': subcort_std,
                    'difference': subcort_mean - cort_mean,
                    'pct_difference': ((subcort_mean - cort_mean) / cort_mean * 100) if cort_mean > 0 else 0,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'n_cortical': len(cortical),
                    'n_subcortical': len(subcortical)
                })
    
    return pd.DataFrame(results)


def compare_rest_vs_task(error_data):
    """Compare error rates between rest and task conditions across all atlases."""
    results = []
    
    comparisons = [
        ('N7 Cortical', 'N7_rest', 'N7_task'),
        ('N17 Cortical', 'N17_rest', 'N17_task'),
        ('Tian I Subcortical', 'TianI_rest', 'TianI_task'),
        ('Tian II Subcortical', 'TianII_rest', 'TianII_task'),
        ('Combined (N7+TianI)', 'Combined_rest', 'Combined_task')
    ]
    
    for atlas_name, rest_key, task_key in comparisons:
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
                mean_rest = merged['error_rate_rest'].mean()
                mean_task = merged['error_rate_task'].mean()
                mean_increase = mean_task - mean_rest
                
                t_stat, p_val = stats.ttest_rel(
                    merged['error_rate_rest'],
                    merged['error_rate_task']
                )
                
                diff = merged['error_rate_task'] - merged['error_rate_rest']
                cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
                
                results.append({
                    'atlas': atlas_name,
                    'rest_mean': mean_rest,
                    'task_mean': mean_task,
                    'mean_increase': mean_increase,
                    'pct_increase': (mean_increase / mean_rest * 100) if mean_rest > 0 else 0,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'n_networks': len(merged)
                })
    
    return pd.DataFrame(results)


# =============================================================================
# COMBINED MULTI-PANEL FIGURES
# =============================================================================

def plot_figure1_resolution_and_systems(error_data, output_path):
    """
    Figure 1: Resolution & Brain System Comparisons (2x2 grid)
    Panel A: N7 vs N17 (Rest)
    Panel B: N7 vs N17 (Task)
    Panel C: Cortical vs Subcortical (Rest)
    Panel D: Cortical vs Subcortical (Task)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Figure 1: Parcellation Resolution & Brain System Comparisons', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: N7 vs N17 (Rest)
    ax = axes[0, 0]
    if 'N7_rest' in error_data and 'N17_rest' in error_data:
        data = [error_data['N7_rest']['error_rate'], error_data['N17_rest']['error_rate']]
        labels = [f'N7\n({len(error_data["N7_rest"])} nets)', 
                  f'N17\n({len(error_data["N17_rest"])} nets)']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['#4A90E2', '#E24A4A']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        means = [d.mean() for d in data]
        ax.plot(range(1, len(means)+1), means, 'D', color='darkred', markersize=8, 
                label='Mean', zorder=3, markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_ylabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('A) Cortical Resolution: Coarse vs Fine (Rest)', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel B: N7 vs N17 (Task)
    ax = axes[0, 1]
    if 'N7_task' in error_data and 'N17_task' in error_data:
        data = [error_data['N7_task']['error_rate'], error_data['N17_task']['error_rate']]
        labels = [f'N7\n({len(error_data["N7_task"])} nets)', 
                  f'N17\n({len(error_data["N17_task"])} nets)']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['#4A90E2', '#E24A4A']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        means = [d.mean() for d in data]
        ax.plot(range(1, len(means)+1), means, 'D', color='darkred', markersize=8,
                label='Mean', zorder=3, markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_ylabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('B) Cortical Resolution: Coarse vs Fine (Task)', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel C: Cortical vs Subcortical (Rest)
    ax = axes[1, 0]
    if 'N7_rest' in error_data and 'TianI_rest' in error_data:
        cortical = error_data['N7_rest']
        subcortical = error_data['TianI_rest']
        
        data = [cortical['error_rate'], subcortical['error_rate']]
        labels = [f'Cortical\n({len(cortical)} nets)', 
                  f'Subcortical\n({len(subcortical)} regions)']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['#2ECC71', '#F39C12']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        means = [d.mean() for d in data]
        ax.plot(range(1, len(means)+1), means, 'D', color='darkred', markersize=8,
                label='Mean', zorder=3, markeredgecolor='white', markeredgewidth=1.5)
        
        # Add significance
        t_stat, p_val = stats.ttest_ind(data[0], data[1])
        y_max = max(d.max() for d in data)
        y_range = y_max - min(d.min() for d in data)
        y_pos = y_max + 0.08 * y_range
        
        ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=2)
        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.text(1.5, y_pos + 0.01 * y_range, sig_text, ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
        ax.text(1.5, y_pos - 0.03 * y_range, f'p = {p_val:.4f}', ha='center', 
                va='top', fontsize=8, style='italic')
        
        ax.set_ylabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('C) Brain Systems: Cortical vs Subcortical (Rest)', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel D: Cortical vs Subcortical (Task)
    ax = axes[1, 1]
    if 'N7_task' in error_data and 'TianI_task' in error_data:
        cortical = error_data['N7_task']
        subcortical = error_data['TianI_task']
        
        data = [cortical['error_rate'], subcortical['error_rate']]
        labels = [f'Cortical\n({len(cortical)} nets)', 
                  f'Subcortical\n({len(subcortical)} regions)']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['#2ECC71', '#F39C12']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        means = [d.mean() for d in data]
        ax.plot(range(1, len(means)+1), means, 'D', color='darkred', markersize=8,
                label='Mean', zorder=3, markeredgecolor='white', markeredgewidth=1.5)
        
        # Add significance
        t_stat, p_val = stats.ttest_ind(data[0], data[1])
        y_max = max(d.max() for d in data)
        y_range = y_max - min(d.min() for d in data)
        y_pos = y_max + 0.08 * y_range
        
        ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=2)
        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.text(1.5, y_pos + 0.01 * y_range, sig_text, ha='center', va='bottom',
                fontsize=14, fontweight='bold')
        ax.text(1.5, y_pos - 0.03 * y_range, f'p = {p_val:.4f}', ha='center',
                va='top', fontsize=8, style='italic')
        
        ax.set_ylabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('D) Brain Systems: Cortical vs Subcortical (Task)', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_figure2_rest_vs_task_overview(error_data, output_path):
    """
    Figure 2: Rest vs Task Effects Overview (2x2 grid)
    Panel A: Mean error rates across atlases (bar chart)
    Panel B: N7 network task-induced changes
    Panel C: N17 network task-induced changes
    Panel D: Tian I network task-induced changes
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Figure 2: Rest vs Task Effects Across Atlas Configurations', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: Overall comparison
    ax = axes[0, 0]
    rest_task_data = []
    atlas_labels = []
    
    for atlas_name, rest_key, task_key in [
        ('N7', 'N7_rest', 'N7_task'),
        ('N17', 'N17_rest', 'N17_task'),
        ('Tian I', 'TianI_rest', 'TianI_task')
    ]:
        if rest_key in error_data and task_key in error_data:
            rest_mean = error_data[rest_key]['error_rate'].mean()
            task_mean = error_data[task_key]['error_rate'].mean()
            rest_task_data.append([rest_mean, task_mean])
            atlas_labels.append(atlas_name)
    
    if rest_task_data:
        rest_task_array = np.array(rest_task_data)
        x = np.arange(len(atlas_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rest_task_array[:, 0], width, 
                       label='Rest', color='#3498DB', alpha=0.85, 
                       edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, rest_task_array[:, 1], width, 
                       label='Task', color='#E74C3C', alpha=0.85, 
                       edgecolor='black', linewidth=1.5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Mean Error Rate', fontweight='bold', fontsize=11)
        ax.set_xlabel('Atlas Configuration', fontweight='bold', fontsize=11)
        ax.set_title('A) Rest vs Task: Overall Comparison', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(atlas_labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel B: N7 task-induced changes
    ax = axes[0, 1]
    if 'N7_rest' in error_data and 'N7_task' in error_data:
        rest_df = error_data['N7_rest']
        task_df = error_data['N7_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network',
            suffixes=('_rest', '_task')
        )
        
        merged['increase'] = merged['error_rate_task'] - merged['error_rate_rest']
        merged['increase_pct'] = (merged['increase'] / merged['error_rate_rest'] * 100)
        merged = merged.sort_values('increase', ascending=False)
        
        colors = ['#E74C3C' if x > 0 else '#2ECC71' for x in merged['increase']]
        bars = ax.barh(range(len(merged)), merged['increase'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for i, (bar, val, pct) in enumerate(zip(bars, merged['increase'], 
                                                 merged['increase_pct'])):
            x_pos = val + (0.006 if val > 0 else -0.006)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{pct:+.0f}%', 
                    va='center', ha=ha, fontsize=8, fontweight='bold')
        
        ax.set_yticks(range(len(merged)))
        ax.set_yticklabels(merged['network'], fontsize=9)
        ax.set_xlabel('Error Change (Task - Rest)', fontweight='bold', fontsize=11)
        ax.set_title('B) N7: Task-Induced Changes', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.axvline(0, color='black', linewidth=2, zorder=1)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Panel C: N17 task-induced changes
    ax = axes[1, 0]
    if 'N17_rest' in error_data and 'N17_task' in error_data:
        rest_df = error_data['N17_rest']
        task_df = error_data['N17_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network',
            suffixes=('_rest', '_task')
        )
        
        merged['increase'] = merged['error_rate_task'] - merged['error_rate_rest']
        merged['increase_pct'] = (merged['increase'] / merged['error_rate_rest'] * 100)
        merged = merged.sort_values('increase', ascending=False)
        
        colors = ['#E74C3C' if x > 0 else '#2ECC71' for x in merged['increase']]
        bars = ax.barh(range(len(merged)), merged['increase'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for i, (bar, val, pct) in enumerate(zip(bars, merged['increase'], 
                                                 merged['increase_pct'])):
            x_pos = val + (0.006 if val > 0 else -0.006)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{pct:+.0f}%', 
                    va='center', ha=ha, fontsize=7, fontweight='bold')
        
        ax.set_yticks(range(len(merged)))
        ax.set_yticklabels(merged['network'], fontsize=8)
        ax.set_xlabel('Error Change (Task - Rest)', fontweight='bold', fontsize=11)
        ax.set_title('C) N17: Task-Induced Changes', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.axvline(0, color='black', linewidth=2, zorder=1)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Panel D: Tian I task-induced changes
    ax = axes[1, 1]
    if 'TianI_rest' in error_data and 'TianI_task' in error_data:
        rest_df = error_data['TianI_rest']
        task_df = error_data['TianI_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network',
            suffixes=('_rest', '_task')
        )
        
        merged['increase'] = merged['error_rate_task'] - merged['error_rate_rest']
        merged['increase_pct'] = (merged['increase'] / merged['error_rate_rest'] * 100)
        merged = merged.sort_values('increase', ascending=False)
        
        colors = ['#E74C3C' if x > 0 else '#2ECC71' for x in merged['increase']]
        bars = ax.barh(range(len(merged)), merged['increase'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for i, (bar, val, pct) in enumerate(zip(bars, merged['increase'], 
                                                 merged['increase_pct'])):
            x_pos = val + (0.006 if val > 0 else -0.006)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{pct:+.0f}%', 
                    va='center', ha=ha, fontsize=8, fontweight='bold')
        
        ax.set_yticks(range(len(merged)))
        ax.set_yticklabels(merged['network'], fontsize=9)
        ax.set_xlabel('Error Change (Task - Rest)', fontweight='bold', fontsize=11)
        ax.set_title('D) Tian I: Task-Induced Changes', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.axvline(0, color='black', linewidth=2, zorder=1)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add legend for increase/decrease
        legend_elements = [
            Patch(facecolor='#E74C3C', alpha=0.8, edgecolor='black', label='Increased'),
            Patch(facecolor='#2ECC71', alpha=0.8, edgecolor='black', label='Decreased')
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def plot_figure3_network_level_detail(error_data, output_path):
    """
    Figure 3: Detailed Network-Level Error Rates (2x2 grid)
    Panel A: N7 error bars (Rest vs Task)
    Panel B: N17 error bars (Rest vs Task)
    Panel C: Tian I error bars (Rest vs Task)
    Panel D: Tian II error bars (Rest vs Task)
    """
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
            on='network',
            suffixes=('_rest', '_task')
        )
        
        merged['avg_error'] = (merged['error_rate_rest'] + merged['error_rate_task']) / 2
        merged = merged.sort_values('avg_error', ascending=False)
        
        y_pos = np.arange(len(merged))
        
        bars_rest = ax.barh(y_pos - 0.2, merged['error_rate_rest'], 0.4,
                            label='Rest', color='#3498DB', alpha=0.85, 
                            edgecolor='black', linewidth=1)
        bars_task = ax.barh(y_pos + 0.2, merged['error_rate_task'], 0.4,
                            label='Task', color='#E74C3C', alpha=0.85,
                            edgecolor='black', linewidth=1)
        
        for bars in [bars_rest, bars_task]:
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}',
                        va='center', ha='left', fontsize=8, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(merged['network'], fontsize=10)
        ax.set_xlabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('A) N7 Cortical Networks', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
    
    # Panel B: N17
    ax = axes[0, 1]
    if 'N17_rest' in error_data and 'N17_task' in error_data:
        rest_df = error_data['N17_rest']
        task_df = error_data['N17_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network',
            suffixes=('_rest', '_task')
        )
        
        merged['avg_error'] = (merged['error_rate_rest'] + merged['error_rate_task']) / 2
        merged = merged.sort_values('avg_error', ascending=False)
        
        y_pos = np.arange(len(merged))
        
        bars_rest = ax.barh(y_pos - 0.2, merged['error_rate_rest'], 0.4,
                            label='Rest', color='#3498DB', alpha=0.85, 
                            edgecolor='black', linewidth=1)
        bars_task = ax.barh(y_pos + 0.2, merged['error_rate_task'], 0.4,
                            label='Task', color='#E74C3C', alpha=0.85,
                            edgecolor='black', linewidth=1)
        
        for bars in [bars_rest, bars_task]:
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}',
                        va='center', ha='left', fontsize=7, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(merged['network'], fontsize=8)
        ax.set_xlabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('B) N17 Cortical Networks', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
    
    # Panel C: Tian I
    ax = axes[1, 0]
    if 'TianI_rest' in error_data and 'TianI_task' in error_data:
        rest_df = error_data['TianI_rest']
        task_df = error_data['TianI_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network',
            suffixes=('_rest', '_task')
        )
        
        merged['avg_error'] = (merged['error_rate_rest'] + merged['error_rate_task']) / 2
        merged = merged.sort_values('avg_error', ascending=False)
        
        y_pos = np.arange(len(merged))
        
        bars_rest = ax.barh(y_pos - 0.2, merged['error_rate_rest'], 0.4,
                            label='Rest', color='#3498DB', alpha=0.85, 
                            edgecolor='black', linewidth=1)
        bars_task = ax.barh(y_pos + 0.2, merged['error_rate_task'], 0.4,
                            label='Task', color='#E74C3C', alpha=0.85,
                            edgecolor='black', linewidth=1)
        
        for bars in [bars_rest, bars_task]:
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}',
                        va='center', ha='left', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(merged['network'], fontsize=10)
        ax.set_xlabel('Error Rate', fontweight='bold', fontsize=11)
        ax.set_title('C) Tian Scale I Subcortical Regions', 
                     fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
    
    # Panel D: Tian II
    ax = axes[1, 1]
    if 'TianII_rest' in error_data and 'TianII_task' in error_data:
        rest_df = error_data['TianII_rest']
        task_df = error_data['TianII_task']
        
        merged = pd.merge(
            rest_df[['network', 'error_rate']],
            task_df[['network', 'error_rate']],
            on='network',
            suffixes=('_rest', '_task')
        )
        
        if len(merged) > 0:
            merged['avg_error'] = (merged['error_rate_rest'] + merged['error_rate_task']) / 2
            merged = merged.sort_values('avg_error', ascending=False)
            
            y_pos = np.arange(len(merged))
            
            bars_rest = ax.barh(y_pos - 0.2, merged['error_rate_rest'], 0.4,
                                label='Rest', color='#3498DB', alpha=0.85, 
                                edgecolor='black', linewidth=1)
            bars_task = ax.barh(y_pos + 0.2, merged['error_rate_task'], 0.4,
                                label='Task', color='#E74C3C', alpha=0.85,
                                edgecolor='black', linewidth=1)
            
            for bars in [bars_rest, bars_task]:
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{width:.3f}',
                            va='center', ha='left', fontsize=7, fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(merged['network'], fontsize=8)
            ax.set_xlabel('Error Rate', fontweight='bold', fontsize=11)
            ax.set_title('D) Tian Scale II Subcortical Subdivisions', 
                         fontweight='bold', fontsize=12, pad=10)
            ax.legend(fontsize=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No Tian Scale II data available', 
                   ha='center', va='center', fontsize=12, style='italic')
            ax.set_title('D) Tian Scale II Subcortical Subdivisions', 
                        fontweight='bold', fontsize=12, pad=10)
    else:
        ax.text(0.5, 0.5, 'No Tian Scale II data available', 
               ha='center', va='center', fontsize=12, style='italic')
        ax.set_title('D) Tian Scale II Subcortical Subdivisions', 
                    fontweight='bold', fontsize=12, pad=10)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


def print_detailed_summary(resolution_comp, cortical_subcortical, rest_task):
    """Print comprehensive statistical summary."""
    
    print("\n" + "="*80)
    print("DETAILED RESULTS SUMMARY")
    print("="*80)
    
    if not resolution_comp.empty:
        print("\n1. PARCELLATION RESOLUTION EFFECTS:")
        print("-" * 80)
        for _, row in resolution_comp.iterrows():
            direction = "increases" if row['difference'] > 0 else "decreases"
            print(f"\n{row['comparison']}:")
            print(f"  Coarse ({row['coarse_n']} networks): {row['coarse_mean']:.4f} ± {row['coarse_std']:.4f}")
            print(f"  Fine ({row['fine_n']} networks):     {row['fine_mean']:.4f} ± {row['fine_std']:.4f}")
            print(f"  → Fine parcellation {direction} error by {abs(row['difference']):.4f} ({row['pct_change']:+.1f}%)")
    
    if not cortical_subcortical.empty:
        print("\n2. CORTICAL VS SUBCORTICAL PERFORMANCE:")
        print("-" * 80)
        for _, row in cortical_subcortical.iterrows():
            sig_stars = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'n.s.'
            better = "cortical" if row['difference'] < 0 else "subcortical"
            worse = "subcortical" if row['difference'] < 0 else "cortical"
            
            print(f"\n{row['condition']} Condition:")
            print(f"  Cortical:    {row['cortical_mean']:.4f} ± {row['cortical_std']:.4f} ({row['n_cortical']} networks)")
            print(f"  Subcortical: {row['subcortical_mean']:.4f} ± {row['subcortical_std']:.4f} ({row['n_subcortical']} regions)")
            print(f"  → {worse.capitalize()} has {abs(row['pct_difference']):.1f}% higher error than {better}")
            print(f"  → t({row['n_cortical']+row['n_subcortical']-2}) = {row['t_statistic']:.3f}, p = {row['p_value']:.4f} {sig_stars}")
            print(f"  → Effect size (Cohen's d) = {row['cohens_d']:.3f}")
    
    if not rest_task.empty:
        print("\n3. REST VS TASK EFFECTS:")
        print("-" * 80)
        for _, row in rest_task.iterrows():
            sig_stars = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'n.s.'
            
            print(f"\n{row['atlas']}:")
            print(f"  Rest: {row['rest_mean']:.4f}")
            print(f"  Task: {row['task_mean']:.4f}")
            print(f"  → Task increases error by {row['mean_increase']:.4f} ({row['pct_increase']:+.1f}%)")
            print(f"  → t({row['n_networks']-1}) = {row['t_statistic']:.3f}, p = {row['p_value']:.4f} {sig_stars}")
            print(f"  → Effect size (Cohen's d) = {row['cohens_d']:.3f}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("="*80)
    print("ATLAS COMPARISON ANALYSIS - COMBINED MULTI-PANEL FIGURES")
    print("="*80)
    
    # Load error rates
    tables_dir = Path('reports/tables/atlas_analysis')
    
    if not tables_dir.exists():
        print(f"\n✗ Error: {tables_dir} not found")
        print("Please run 01_atlas_performance_analysis_refactored.py first!")
        return 1
    
    print("\nLoading error rate data...")
    error_data = load_error_rates(tables_dir)
    
    if not error_data:
        print("✗ No error rate files found!")
        return 1
    
    print(f"\n✓ Successfully loaded {len(error_data)} error rate files")
    
    # Create output directories
    output_tables = Path('reports/tables/atlas_comparison')
    output_figures = Path('reports/figures/atlas_comparison')
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Statistical Comparisons
    # =========================================================================
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS")
    print("="*80)
    
    print("\nAnalysis 1: Resolution Effects...")
    resolution_comparison = compare_resolution_effects(error_data)
    if not resolution_comparison.empty:
        resolution_comparison.to_csv(
            output_tables / 'resolution_comparison.csv', index=False
        )
        print("✓ Resolution comparison completed")
    
    print("\nAnalysis 2: Cortical vs Subcortical...")
    cortical_subcortical = compare_cortical_vs_subcortical(error_data)
    if not cortical_subcortical.empty:
        cortical_subcortical.to_csv(
            output_tables / 'cortical_vs_subcortical.csv', index=False
        )
        print("✓ Cortical vs subcortical comparison completed")
    
    print("\nAnalysis 3: Rest vs Task...")
    rest_task_comparison = compare_rest_vs_task(error_data)
    if not rest_task_comparison.empty:
        rest_task_comparison.to_csv(
            output_tables / 'rest_vs_task_comparison.csv', index=False
        )
        print("✓ Rest vs task comparison completed")
    
    # =========================================================================
    # Generate Combined Multi-Panel Figures
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING COMBINED MULTI-PANEL FIGURES")
    print("="*80)
    
    print("\nFigure 1: Resolution & Brain System Comparisons (2x2)...")
    plot_figure1_resolution_and_systems(error_data, 
                                        output_figures / 'figure1_resolution_and_systems.png')
    
    print("\nFigure 2: Rest vs Task Effects Overview (2x2)...")
    plot_figure2_rest_vs_task_overview(error_data, 
                                       output_figures / 'figure2_rest_vs_task_overview.png')
    
    print("\nFigure 3: Detailed Network-Level Analysis (2x2)...")
    plot_figure3_network_level_detail(error_data, 
                                      output_figures / 'figure3_network_level_detail.png')
    
    # =========================================================================
    # Print Detailed Summary
    # =========================================================================
    print_detailed_summary(resolution_comparison, cortical_subcortical, rest_task_comparison)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"""
Generated Files:
================

Statistical Tables (CSV):
  • {output_tables}/resolution_comparison.csv
  • {output_tables}/cortical_vs_subcortical.csv
  • {output_tables}/rest_vs_task_comparison.csv

Combined Multi-Panel Figures (PNG) - 3 Total:
  
  Figure 1 (2x2): Resolution & Brain System Comparisons
  • {output_figures}/figure1_resolution_and_systems.png
    - Panel A: N7 vs N17 (Rest)
    - Panel B: N7 vs N17 (Task)
    - Panel C: Cortical vs Subcortical (Rest)
    - Panel D: Cortical vs Subcortical (Task)
  
  Figure 2 (2x2): Rest vs Task Effects Overview
  • {output_figures}/figure2_rest_vs_task_overview.png
    - Panel A: Overall comparison (bar chart)
    - Panel B: N7 task-induced changes
    - Panel C: N17 task-induced changes
    - Panel D: Tian I task-induced changes
  
  Figure 3 (2x2): Detailed Network-Level Analysis
  • {output_figures}/figure3_network_level_detail.png
    - Panel A: N7 error bars (Rest vs Task)
    - Panel B: N17 error bars (Rest vs Task)
    - Panel C: Tian I error bars (Rest vs Task)
    - Panel D: Tian II error bars (Rest vs Task)

✓ All analyses and visualizations completed successfully!
✓ 3 combined figures generated instead of 13 separate plots
✓ Each figure is a 2x2 grid organized by logical comparison themes
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())