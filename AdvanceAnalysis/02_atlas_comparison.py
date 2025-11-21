#!/usr/bin/env python3
"""
Atlas Comparison Analysis - IMPROVED VERSION

IMPROVEMENTS:
- Standardized color scheme (rest: #2E86AB blue, task: #A23B72 purple)
- Simplified hybrid violin/box plots using inner='box' parameter
- Added error bars to bar charts
- Consistent font sizes (18pt titles, 14pt labels, 11pt ticks)
- Better p-value formatting (space before stars)
- Improved legend positioning
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib 
import seaborn as sns
from scipy import stats

# =============================================================================
# GLOBAL PLOTTING STYLE
# =============================================================================
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.dpi': 300
})

# IMPROVED: Consistent professional color scheme across all plots
COLORS = {
    'rest': '#2E86AB',        # Blue (consistent)
    'task': '#A23B72',        # Purple/Red (consistent)
    'cortical': '#06A77D',
    'subcortical': '#F18F01',
    'n7': '#4A90E2',          # Blue
    'n17': '#E24A4A',         # Red
    'tian1': '#F39C12',       # Orange
    'tian2': '#9B59B6',       # Purple
    'increase': '#E74C3C',    # Red for increase
    'decrease': '#2ECC71',    # Green for decrease
}


# =============================================================================
# STATISTICAL HELPER FUNCTIONS
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
            print(f"  ✓ {filename}")
    
    return data

def compute_statistics(group1, group2, name1="Group1", name2="Group2"):
    """Compute comprehensive statistics comparing two groups."""
    stats_dict = {
        'group1_name': name1, 'group2_name': name2,
        'n1': len(group1), 'n2': len(group2),
        'mean1': np.mean(group1), 'mean2': np.mean(group2),
        'std1': np.std(group1, ddof=1), 'std2': np.std(group2, ddof=1),
        'median1': np.median(group1), 'median2': np.median(group2),
        'min1': np.min(group1), 'min2': np.min(group2),
        'max1': np.max(group1), 'max2': np.max(group2),
    }
    
    # Difference measures
    stats_dict['mean_diff'] = stats_dict['mean2'] - stats_dict['mean1']
    stats_dict['median_diff'] = stats_dict['median2'] - stats_dict['median1']
    stats_dict['pct_change'] = (stats_dict['mean_diff'] / stats_dict['mean1']) * 100
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group1)-1)*stats_dict['std1']**2 + 
                          (len(group2)-1)*stats_dict['std2']**2) / 
                         (len(group1) + len(group2) - 2))
    stats_dict['cohens_d'] = stats_dict['mean_diff'] / pooled_std
    
    abs_d = abs(stats_dict['cohens_d'])
    if abs_d < 0.2: stats_dict['effect_interpretation'] = 'negligible'
    elif abs_d < 0.5: stats_dict['effect_interpretation'] = 'small'
    elif abs_d < 0.8: stats_dict['effect_interpretation'] = 'medium'
    else: stats_dict['effect_interpretation'] = 'large'
    
    # Statistical tests
    t_stat, p_val = stats.ttest_ind(group1, group2)
    stats_dict['t_statistic'] = t_stat
    stats_dict['t_pvalue'] = p_val
    t_welch, p_welch = stats.ttest_ind(group1, group2, equal_var=False)
    stats_dict['welch_t'] = t_welch
    stats_dict['welch_p'] = p_welch
    u_stat, p_mann = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    stats_dict['mann_whitney_u'] = u_stat
    stats_dict['mann_whitney_p'] = p_mann
    levene_stat, levene_p = stats.levene(group1, group2)
    stats_dict['levene_statistic'] = levene_stat
    stats_dict['levene_p'] = levene_p
    
    # Significance interpretation
    stats_dict['significant_005'] = p_val < 0.05
    
    # Confidence intervals (95%)
    ci1 = stats.t.interval(0.95, len(group1)-1, loc=stats_dict['mean1'], scale=stats.sem(group1))
    ci2 = stats.t.interval(0.95, len(group2)-1, loc=stats_dict['mean2'], scale=stats.sem(group2))
    stats_dict['ci95_lower1'] = ci1[0]
    stats_dict['ci95_upper1'] = ci1[1]
    stats_dict['ci95_lower2'] = ci2[0]
    stats_dict['ci95_upper2'] = ci2[1]
    
    return stats_dict


def paired_statistics(group1, group2, name1="Pre", name2="Post"):
    """Compute paired statistics (for rest vs task within same networks)."""
    if len(group1) != len(group2):
        return compute_statistics(group1, group2, name1, name2)
    
    stats_dict = compute_statistics(group1, group2, name1, name2)
    
    # Paired t-test
    t_paired, p_paired = stats.ttest_rel(group1, group2)
    stats_dict['paired_t'] = t_paired
    stats_dict['paired_p'] = p_paired
    
    # Wilcoxon signed-rank test
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(group1, group2)
    stats_dict['wilcoxon_statistic'] = wilcoxon_stat
    stats_dict['wilcoxon_p'] = wilcoxon_p
    
    return stats_dict


def format_pvalue(p):
    """IMPROVED: Format p-value for display with space before stars."""
    if p < 0.001: return "p < 0.001 ***"
    elif p < 0.01: return f"p = {p:.3f} **"
    elif p < 0.05: return f"p = {p:.3f} *"
    else: return f"p = {p:.3f} ns"


def get_significance_stars(p):
    """Get significance stars."""
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"


def log_statistics(stat, comparison_name):
    """Log comprehensive statistics to console."""
    print(f"\n{'='*70}")
    print(f"  {comparison_name}")
    print(f"{'='*70}")
    
    print(f"\nDESCRIPTIVE STATISTICS:")
    print(f"  {stat['group1_name']:12s}: n={stat['n1']:2d}, μ={stat['mean1']:.4f}, σ={stat['std1']:.4f}, median={stat['median1']:.4f}")
    print(f"  {stat['group2_name']:12s}: n={stat['n2']:2d}, μ={stat['mean2']:.4f}, σ={stat['std2']:.4f}, median={stat['median2']:.4f}")
    
    print(f"\nDIFFERENCE MEASURES:")
    print(f"  Mean difference:     {stat['mean_diff']:+.4f} ({stat['pct_change']:+.1f}%)")
    
    print(f"\nEFFECT SIZE:")
    print(f"  Cohen's d:           {stat['cohens_d']:.3f} ({stat['effect_interpretation']})")
    
    print(f"\nSTATISTICAL TESTS:")
    print(f"  Independent t-test:  t({stat['n1']+stat['n2']-2}) = {stat['t_statistic']:.3f}, {format_pvalue(stat['t_pvalue'])}")
    print(f"  Mann-Whitney U:      U = {stat['mann_whitney_u']:.1f}, {format_pvalue(stat['mann_whitney_p'])}")
    
    if 'paired_t' in stat:
        print(f"\nPAIRED TESTS:")
        print(f"  Paired t-test:       t({stat['n1']-1}) = {stat['paired_t']:.3f}, {format_pvalue(stat['paired_p'])}")
        print(f"  Wilcoxon signed-rank: W = {stat['wilcoxon_statistic']:.1f}, {format_pvalue(stat['wilcoxon_p'])}")


# =============================================================================
# IMPROVED PLOTTING HELPER FUNCTIONS
# =============================================================================

def compute_basic_stats(group1, group2):
    """Compute t-test and means for plotting."""
    t_stat, p_val = stats.ttest_ind(group1, group2)
    return {'p_val': p_val}

def get_stars(p):
    """Get significance stars for annotation."""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def add_stat_annotation(ax, x1, x2, y, p_val, text_offset=0.002):
    """Adds a professional statistical bracket and p-value/stars."""
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = y_range * 0.02
    col = 'k'
    
    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c=col)
    
    star_text = get_stars(p_val)
    text_content = f"{star_text}\n(p={p_val:.3f})" if p_val >= 0.001 else f"{star_text}\n(p<0.001)"
    
    # Add text annotation
    ax.text((x1+x2)*.5, y+h + text_offset, text_content, 
            ha='center', va='bottom', color=col, fontsize=10, fontweight='bold')

def plot_hybrid_violin_box(ax, data1, data2, label1, label2, color1, color2, title):
    """
    IMPROVED: Simplified violin plot using inner='box' parameter.
    Creates a cleaner visualization with violin + embedded box plot.
    """
    df = pd.DataFrame({
        'Error Rate': np.concatenate([data1, data2]),
        'Group': [label1] * len(data1) + [label2] * len(data2)
    })
    palette = {label1: color1, label2: color2}

    # IMPROVED: Use inner='box' for cleaner look (combines violin + box in one call)
    sns.violinplot(data=df, x='Group', y='Error Rate', ax=ax, 
                   palette=palette, inner='box', linewidth=1.5, saturation=0.75)
    
    # Add mean markers
    means = df.groupby('Group')['Error Rate'].mean()
    for i, (group, mean_val) in enumerate(means.items()):
        ax.plot(i, mean_val, marker='D', markersize=10, color=COLORS['task'], 
                markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    # Statistics Calculation & Annotation
    stats_res = compute_basic_stats(data1, data2) 
    y_max = ax.get_ylim()[1]
    add_stat_annotation(ax, 0, 1, y_max * 0.9, stats_res['p_val'])

    # Styling
    ax.set_title(title, fontweight='bold', fontsize=18, pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('Error Rate', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=11)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    sns.despine(ax=ax, offset=5, trim=True)
    
    # Custom Legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=2, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['task'], 
                   markersize=8, label='Mean', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)

    return stats_res

# =============================================================================
# FIGURE 1 & 2 GENERATORS (IMPROVED HYBRID PLOTS)
# =============================================================================

def plot_resolution_effects(error_data, output_path):
    """Parcellation Resolution Effects (Improved Hybrid Violin/Box)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Impact of Parcellation Resolution on Model Performance', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # (A) N7 vs N17 (Rest)
    if 'N7_rest' in error_data and 'N17_rest' in error_data:
        plot_hybrid_violin_box(axes[0,0], error_data['N7_rest']['error_rate'].values,
                              error_data['N17_rest']['error_rate'].values,
                              'N7 (Coarse)', 'N17 (Fine)', COLORS['n7'], COLORS['n17'],
                              '(A) Cortical Networks - Rest')

    # (B) N7 vs N17 (Task)
    if 'N7_task' in error_data and 'N17_task' in error_data:
        plot_hybrid_violin_box(axes[0,1], error_data['N7_task']['error_rate'].values,
                              error_data['N17_task']['error_rate'].values,
                              'N7 (Coarse)', 'N17 (Fine)', COLORS['n7'], COLORS['n17'],
                              '(B) Cortical Networks - Task')

    # (C) Tian I vs II (Rest)
    if 'TianI_rest' in error_data and 'TianII_rest' in error_data:
        plot_hybrid_violin_box(axes[1,0], error_data['TianI_rest']['error_rate'].values,
                              error_data['TianII_rest']['error_rate'].values,
                              'Tian I', 'Tian II', COLORS['tian1'], COLORS['tian2'],
                              '(C) Subcortical Regions - Rest')

    # (D) Tian I vs II (Task)
    if 'TianI_task' in error_data and 'TianII_task' in error_data:
        plot_hybrid_violin_box(axes[1,1], error_data['TianI_task']['error_rate'].values,
                              error_data['TianII_task']['error_rate'].values,
                              'Tian I', 'Tian II', COLORS['tian1'], COLORS['tian2'],
                              '(D) Subcortical Regions - Task')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 1 (Resolution Analysis - Improved): {output_path.name}")
    return None

def plot_cortical_vs_subcortical(error_data, output_path):
    """Cortical vs. Subcortical Performance (Improved Hybrid Violin/Box)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Cortical vs. Subcortical Performance', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    comparisons = [
        ('N7_rest', 'TianI_rest', 'Coarse Resolution - Rest', axes[0,0], '(A)'),
        ('N7_task', 'TianI_task', 'Coarse Resolution - Task', axes[0,1], '(B)'),
        ('N17_rest', 'TianII_rest', 'Fine Resolution - Rest', axes[1,0], '(C)'),
        ('N17_task', 'TianII_task', 'Fine Resolution - Task', axes[1,1], '(D)'),
    ]

    for cort_key, sub_key, title, ax, panel in comparisons:
        if cort_key in error_data and sub_key in error_data:
            plot_hybrid_violin_box(ax, error_data[cort_key]['error_rate'].values,
                                  error_data[sub_key]['error_rate'].values,
                                  'Cortical', 'Subcortical', 
                                  COLORS['cortical'], COLORS['subcortical'],
                                  f'{panel} {title}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 2 (System Comparison - Improved): {output_path.name}")
    return None

# =============================================================================
# FIGURE 3 (IMPROVED HYBRID VIOLIN/BOX)
# =============================================================================

def plot_task_effects(error_data, output_path):
    """Rest vs. Task State Comparison (Improved Hybrid Violin/Box)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Rest vs. Task State Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)

    configs = [
        ('N7_rest', 'N7_task', 'N7 Cortical', axes[0,0], '(A)'),
        ('N17_rest', 'N17_task', 'N17 Cortical', axes[0,1], '(B)'),
        ('TianI_rest', 'TianI_task', 'Tian I Subcortical', axes[1,0], '(C)'),
        ('TianII_rest', 'TianII_task', 'Tian II Subcortical', axes[1,1], '(D)'),
    ]

    for rest_key, task_key, title, ax, panel in configs:
        if rest_key in error_data and task_key in error_data:
            plot_hybrid_violin_box(ax, error_data[rest_key]['error_rate'].values,
                                  error_data[task_key]['error_rate'].values,
                                  'Rest', 'Task', 
                                  COLORS['rest'], COLORS['task'],
                                  f'{panel} {title}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 3 (Task Effects - Improved): {output_path.name}")
    return None 

# =============================================================================
# FIGURE 3a (IMPROVED BAR CHART WITH ERROR BARS)
# =============================================================================

def plot_task_effects_bar_chart(error_data, output_path):
    """IMPROVED: Bar chart with error bars showing Task effects."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Task-Induced Changes in Classification Performance',
                 fontsize=20, fontweight='bold', y=0.995)
    
    all_stats = []
    
    atlas_configs = [
        ('N7 Cortical', 'N7_rest', 'N7_task', axes[0, 0], '(A)'),
        ('N17 Cortical', 'N17_rest', 'N17_task', axes[0, 1], '(B)'),
        ('Tian I Subcortical', 'TianI_rest', 'TianI_task', axes[1, 0], '(C)'),
        ('Tian II Subcortical', 'TianII_rest', 'TianII_task', axes[1, 1], '(D)'),
    ]
    
    for name, rest_key, task_key, ax, panel in atlas_configs:
        if rest_key in error_data and task_key in error_data:
            rest_df = error_data[rest_key]
            task_df = error_data[task_key]
            
            merged = pd.merge(
                rest_df[['network', 'error_rate']],
                task_df[['network', 'error_rate']],
                on='network', suffixes=('_rest', '_task')
            )
            
            # Calculate difference and sort
            merged['diff'] = merged['error_rate_rest'] - merged['error_rate_task']
            merged = merged.sort_values(by='diff', ascending=True) 
            
            if len(merged) > 0:
                rest_vals = merged['error_rate_rest'].values
                task_vals = merged['error_rate_task'].values
                
                # IMPROVED: Calculate standard errors for error bars
                rest_sem = rest_df.groupby('network')['error_rate'].sem().reindex(merged['network']).fillna(0).values
                task_sem = task_df.groupby('network')['error_rate'].sem().reindex(merged['network']).fillna(0).values
                
                stat = paired_statistics(rest_vals, task_vals, "Rest", "Task")
                all_stats.append({**stat, 'atlas': name})
                
                y_pos = np.arange(len(merged))
                
                # IMPROVED: Add error bars
                ax.barh(y_pos - 0.2, rest_vals, 0.4, xerr=rest_sem, label='Rest',
                       color=COLORS['rest'], alpha=0.8, edgecolor='black', linewidth=1,
                       error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})
                ax.barh(y_pos + 0.2, task_vals, 0.4, xerr=task_sem, label='Task',
                       color=COLORS['task'], alpha=0.8, edgecolor='black', linewidth=1,
                       error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})
                
                for i, (rest_val, task_val) in enumerate(zip(rest_vals, task_vals)):
                    ax.text(rest_val, i - 0.2, f'{rest_val:.3f}', 
                           va='center', ha='left' if rest_val > 0.02 else 'right',
                           fontsize=8, fontweight='bold', color='black')
                    ax.text(task_val, i + 0.2, f'{task_val:.3f}',
                           va='center', ha='left' if task_val > 0.02 else 'right',
                           fontsize=8, fontweight='bold', color='black')
                
                # Connect pairs with lines
                for i, (r, t) in enumerate(zip(rest_vals, task_vals)):
                    ax.plot([r, t], [i-0.2, i+0.2], 
                            color=COLORS['decrease'] if t < r else COLORS['increase'], 
                            alpha=0.6, linewidth=1.5)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(merged['network'], fontsize=10 if len(merged) < 10 else 8)
                ax.set_xlabel('Error Rate', fontsize=14, fontweight='bold')
                ax.set_title(f'{panel} {name}', fontsize=18, fontweight='bold', pad=10)
                ax.legend(fontsize=11)
                ax.grid(axis='x', alpha=0.3)
                ax.invert_yaxis()
                ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 3a (Task Effects Bar Chart - Improved with Error Bars): {output_path.name}")
    return pd.DataFrame(all_stats)


# =============================================================================
# FIGURE 4 (DISTRIBUTION OVERVIEW - IMPROVED)
# =============================================================================
def plot_distribution_overview_orig(error_data, output_path):
    """IMPROVED: Distribution overview with better annotations."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Error Rate Distributions Across Parcellation Schemes',
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Data Preparation
    atlas_keys = [
        ('N7_rest', 'N7 Cortical', 'n7'), ('N17_rest', 'N17 Cortical', 'n17'),
        ('TianI_rest', 'Tian I Subcort', 'tian1'), ('TianII_rest', 'Tian II Subcort', 'tian2'),
    ]
    
    # Prepare data for Rest (Panel A & C)
    data_rest = []
    labels_rest = []
    colors_rest = []
    means_rest = []
    
    for key, label, color_key in atlas_keys:
        if key in error_data:
            values = error_data[key]['error_rate'].values
            data_rest.append(values)
            labels_rest.append(label)
            colors_rest.append(COLORS[color_key])
            means_rest.append(np.mean(values))
            
    # Prepare data for Task (Panel B & D)
    data_task = []
    labels_task = []
    colors_task = []
    means_task = []
    
    for key, label, color_key in atlas_keys:
        key = key.replace('_rest', '_task')
        if key in error_data:
            values = error_data[key]['error_rate'].values
            data_task.append(values)
            labels_task.append(label)
            colors_task.append(COLORS[color_key])
            means_task.append(np.mean(values))

    # Panel A: Violin plots (Rest)
    ax = fig.add_subplot(gs[0, 0])
    
    if data_rest:
        parts = ax.violinplot(data_rest, positions=range(len(data_rest)), widths=0.7, 
                             showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(parts['bodies'], colors_rest)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        if 'cmeans' in parts:
            parts['cmeans'].set_color(COLORS['task'])
            parts['cmeans'].set_linestyle('--')
            parts['cmeans'].set_linewidth(2)
        
        if 'cmedians' in parts:
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(2)
        
        # Add Mean Values
        y_max = max(ax.get_ylim()[1], np.max([np.max(d) for d in data_rest]) + 0.01)
        ax.set_ylim(top=y_max + (y_max * 0.1))
        for i, mean_val in enumerate(means_rest):
            ax.text(i, y_max * 0.95, f'μ={mean_val:.4f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['task'])
            
        ax.set_xticks(range(len(labels_rest)))
        ax.set_xticklabels(labels_rest, fontsize=11)
        ax.set_ylabel('Error Rate', fontsize=14, fontweight='bold')
        ax.set_title('(A) Distribution Comparison - Rest', fontsize=18, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=11)
        
        legend_elements = [
            plt.Line2D([0], [0], color='black', lw=2, label='Median'),
            plt.Line2D([0], [0], color=COLORS['task'], lw=2, ls='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)
    
    # Panel B: Violin plots (Task)
    ax = fig.add_subplot(gs[0, 1])
    
    if data_task:
        parts = ax.violinplot(data_task, positions=range(len(data_task)), widths=0.7, 
                             showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(parts['bodies'], colors_task)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        if 'cmeans' in parts:
            parts['cmeans'].set_color(COLORS['task']) 
            parts['cmeans'].set_linestyle('--')
            parts['cmeans'].set_linewidth(2)
        
        if 'cmedians' in parts:
            parts['cmedians'].set_color('black') 
            parts['cmedians'].set_linewidth(2)
        
        y_max = max(ax.get_ylim()[1], np.max([np.max(d) for d in data_task]) + 0.01)
        ax.set_ylim(top=y_max + (y_max * 0.1))
        for i, mean_val in enumerate(means_task):
            ax.text(i, y_max * 0.95, f'μ={mean_val:.4f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['task'])
        
        ax.set_xticks(range(len(labels_task)))
        ax.set_xticklabels(labels_task, fontsize=11)
        ax.set_ylabel('Error Rate', fontsize=14, fontweight='bold')
        ax.set_title('(B) Distribution Comparison - Task', fontsize=18, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)
        ax.tick_params(labelsize=11)
    
    # Panel C: Histograms (Rest)
    ax = fig.add_subplot(gs[1, 0])
    for data, label, color in zip(data_rest, labels_rest, colors_rest):
        ax.hist(data, bins=15, alpha=0.7, label=label, color=color, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Error Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('(C) Error Rate Histograms - Rest', fontsize=18, fontweight='bold', pad=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Panel D: Histograms (Task)
    ax = fig.add_subplot(gs[1, 1])
    for data, label, color in zip(data_task, labels_task, colors_task):
        ax.hist(data, bins=15, alpha=0.7, label=label, color=color, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Error Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('(D) Error Rate Histograms - Task', fontsize=18, fontweight='bold', pad=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 4 (Distribution Overview - Improved): {output_path.name}")
    return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    
    print("="*60)
    print("ATLAS COMPARISON - IMPROVED VERSION")
    print("="*60)
    
    tables_dir = Path('reports/tables/atlas_analysis')
    output_figures = Path('reports/figures/atlas_comparison')
    output_tables = Path('reports/tables/atlas_comparison') 
    output_figures.mkdir(parents=True, exist_ok=True)
    output_tables.mkdir(parents=True, exist_ok=True)

    if not tables_dir.exists():
        print("✗ Data directory not found.")
        return

    print("\nLoading error rates...")
    data = load_error_rates(tables_dir) 
    if not data:
        print("✗ No CSV data found!")
        return
    print(f"✓ Loaded {len(data)} datasets\n")


    print("\n" + "▶"*40)
    print("  GENERATING IMPROVED HYBRID VIOLIN/BOX PLOTS")
    print("▶"*40)
    
    # FIGURE 1: Resolution Effects (Improved)
    plot_resolution_effects(data, output_figures / 'fig1_resolution_violin.png')
    
    # FIGURE 2: Cortical vs Subcortical (Improved)
    plot_cortical_vs_subcortical(data, output_figures / 'fig2_systems_violin.png')
    
    # FIGURE 3: Task Effects (Improved)
    plot_task_effects(data, output_figures / 'fig3_task_violin.png') 
    
    print("\n" + "▶"*40)
    print("  GENERATING IMPROVED BAR CHART & DISTRIBUTIONS")
    print("▶"*40)

    # FIGURE 3a: Bar Chart with Error Bars (Improved)
    task_stats_orig = plot_task_effects_bar_chart(data, 
                                                  output_figures / 'fig3a_task_effects_bar.png') 
    
    # FIGURE 4: Distribution Overview (Improved)
    plot_distribution_overview_orig(data, 
                                    output_figures / 'fig4_distribution_overview_orig.png')
    
    # SAVE STATISTICS
    if task_stats_orig is not None and len(task_stats_orig) > 0:
        task_stats_orig.to_csv(output_tables / 'rest_vs_task_comparison_orig.csv', index=False)
        print(f"\n✓ Saved statistics table: rest_vs_task_comparison_orig.csv")

    print("\n" + "="*60)
    print("DONE. Improved figures saved to 'reports/figures/atlas_comparison'")
    print("\nIMPROVEMENTS:")
    print("  • Consistent color scheme (rest: blue, task: purple)")
    print("  • Simplified violin/box plots using inner='box'")
    print("  • Error bars added to bar charts")
    print("  • Standardized fonts (18pt titles, 14pt labels, 11pt ticks)")
    print("  • Better p-value formatting (space before stars)")
    print("  • Improved legend positioning")
    print("="*60)

if __name__ == "__main__":
    main()