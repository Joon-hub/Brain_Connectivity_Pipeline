#!/usr/bin/env python3
"""
Atlas Comparison Analysis
All statistics are logged to console and saved in CSV tables.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Professional color scheme
COLORS = {
    'rest': '#2E86AB',
    'task': '#A23B72',
    'cortical': '#06A77D',
    'subcortical': '#F18F01',
    'n7': '#4A90E2',
    'n17': '#E24A4A',
    'tian1': '#F39C12',
    'tian2': '#9B59B6',
    'increase': '#E74C3C',
    'decrease': '#2ECC71',
}


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
            print(f"  ✓ {filename}")
    
    return data


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def compute_statistics(group1, group2, name1="Group1", name2="Group2"):
    """
    Compute comprehensive statistics comparing two groups.
    Returns dictionary with all statistical measures.
    """
    # Descriptive statistics
    stats_dict = {
        'group1_name': name1,
        'group2_name': name2,
        'n1': len(group1),
        'n2': len(group2),
        'mean1': np.mean(group1),
        'mean2': np.mean(group2),
        'std1': np.std(group1, ddof=1),
        'std2': np.std(group2, ddof=1),
        'median1': np.median(group1),
        'median2': np.median(group2),
        'min1': np.min(group1),
        'min2': np.min(group2),
        'max1': np.max(group1),
        'max2': np.max(group2),
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
    
    # Interpret effect size
    abs_d = abs(stats_dict['cohens_d'])
    if abs_d < 0.2:
        stats_dict['effect_interpretation'] = 'negligible'
    elif abs_d < 0.5:
        stats_dict['effect_interpretation'] = 'small'
    elif abs_d < 0.8:
        stats_dict['effect_interpretation'] = 'medium'
    else:
        stats_dict['effect_interpretation'] = 'large'
    
    # Statistical tests
    # 1. Independent t-test
    t_stat, p_val = stats.ttest_ind(group1, group2)
    stats_dict['t_statistic'] = t_stat
    stats_dict['t_pvalue'] = p_val
    
    # 2. Welch's t-test (unequal variances)
    t_welch, p_welch = stats.ttest_ind(group1, group2, equal_var=False)
    stats_dict['welch_t'] = t_welch
    stats_dict['welch_p'] = p_welch
    
    # 3. Mann-Whitney U test (non-parametric)
    u_stat, p_mann = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    stats_dict['mann_whitney_u'] = u_stat
    stats_dict['mann_whitney_p'] = p_mann
    
    # 4. Levene's test for equal variances
    levene_stat, levene_p = stats.levene(group1, group2)
    stats_dict['levene_statistic'] = levene_stat
    stats_dict['levene_p'] = levene_p
    
    # Significance interpretation
    stats_dict['significant_005'] = p_val < 0.05
    stats_dict['significant_001'] = p_val < 0.01
    stats_dict['significant_0001'] = p_val < 0.001
    
    # Confidence intervals (95%)
    ci1 = stats.t.interval(0.95, len(group1)-1, 
                          loc=stats_dict['mean1'], 
                          scale=stats.sem(group1))
    ci2 = stats.t.interval(0.95, len(group2)-1,
                          loc=stats_dict['mean2'],
                          scale=stats.sem(group2))
    stats_dict['ci95_lower1'] = ci1[0]
    stats_dict['ci95_upper1'] = ci1[1]
    stats_dict['ci95_lower2'] = ci2[0]
    stats_dict['ci95_upper2'] = ci2[1]
    
    return stats_dict


def paired_statistics(group1, group2, name1="Pre", name2="Post"):
    """
    Compute paired statistics (for rest vs task within same networks).
    """
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
    """Format p-value for display."""
    if p < 0.001:
        return "p < 0.001***"
    elif p < 0.01:
        return f"p = {p:.3f}**"
    elif p < 0.05:
        return f"p = {p:.3f}*"
    else:
        return f"p = {p:.3f}ns"


def get_significance_stars(p):
    """Get significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def log_statistics(stat, comparison_name):
    """Log comprehensive statistics to console."""
    print(f"\n{'='*70}")
    print(f"  {comparison_name}")
    print(f"{'='*70}")
    
    print(f"\nDESCRIPTIVE STATISTICS:")
    print(f"  {stat['group1_name']:12s}: n={stat['n1']:2d}, μ={stat['mean1']:.4f}, σ={stat['std1']:.4f}, median={stat['median1']:.4f}")
    print(f"  {stat['group2_name']:12s}: n={stat['n2']:2d}, μ={stat['mean2']:.4f}, σ={stat['std2']:.4f}, median={stat['median2']:.4f}")
    print(f"  Range {stat['group1_name']:7s}: [{stat['min1']:.4f}, {stat['max1']:.4f}]")
    print(f"  Range {stat['group2_name']:7s}: [{stat['min2']:.4f}, {stat['max2']:.4f}]")
    
    print(f"\nDIFFERENCE MEASURES:")
    print(f"  Mean difference:     {stat['mean_diff']:+.4f} ({stat['pct_change']:+.1f}%)")
    print(f"  Median difference:   {stat['median_diff']:+.4f}")
    
    print(f"\nEFFECT SIZE:")
    print(f"  Cohen's d:           {stat['cohens_d']:.3f} ({stat['effect_interpretation']})")
    
    print(f"\nSTATISTICAL TESTS:")
    print(f"  Independent t-test:  t({stat['n1']+stat['n2']-2}) = {stat['t_statistic']:.3f}, {format_pvalue(stat['t_pvalue'])}")
    print(f"  Welch's t-test:      t = {stat['welch_t']:.3f}, {format_pvalue(stat['welch_p'])}")
    print(f"  Mann-Whitney U:      U = {stat['mann_whitney_u']:.1f}, {format_pvalue(stat['mann_whitney_p'])}")
    print(f"  Levene's test:       F = {stat['levene_statistic']:.3f}, p = {stat['levene_p']:.3f}")
    
    print(f"\nCONFIDENCE INTERVALS (95%):")
    print(f"  {stat['group1_name']:12s}: [{stat['ci95_lower1']:.4f}, {stat['ci95_upper1']:.4f}]")
    print(f"  {stat['group2_name']:12s}: [{stat['ci95_lower2']:.4f}, {stat['ci95_upper2']:.4f}]")
    
    if 'paired_t' in stat:
        print(f"\nPAIRED TESTS:")
        print(f"  Paired t-test:       t({stat['n1']-1}) = {stat['paired_t']:.3f}, {format_pvalue(stat['paired_p'])}")
        print(f"  Wilcoxon signed-rank: W = {stat['wilcoxon_statistic']:.1f}, {format_pvalue(stat['wilcoxon_p'])}")


# =============================================================================
# OUTLIER DETECTION AND ANNOTATION
# =============================================================================

def identify_outliers(data, labels):
    """Identify outliers using IQR method."""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    outliers = []
    for i, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            outliers.append({
                'index': i,
                'label': labels[i],
                'value': data[i],
                'distance': abs(data[i] - np.median(data))
            })
    
    return pd.DataFrame(outliers)


def annotate_outliers(ax, data, position, labels, n_top=3, direction='right'):
    """Add outlier annotations with arrows."""
    outliers_df = identify_outliers(data, labels)
    
    if len(outliers_df) == 0:
        return
    
    # Select top N outliers by distance from median
    top_outliers = outliers_df.nlargest(min(n_top, len(outliers_df)), 'distance')
    
    for _, outlier in top_outliers.iterrows():
        if direction == 'right':
            xytext = (15, 0)
        else:
            xytext = (-15, 0)
        
        ax.annotate(
            outlier['label'],
            xy=(position, outlier['value']),
            xytext=xytext,
            textcoords='offset points',
            fontsize=8,
            color='darkred',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                     alpha=0.8, edgecolor='darkred', linewidth=1.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                          color='darkred', lw=1.5)
        )


# =============================================================================
# FIGURE 1: RESOLUTION EFFECTS 
# =============================================================================

def plot_resolution_effects(error_data, output_path):
    """
    All statistics logged to console and saved in tables.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Parcellation Resolution Effects on Classification Performance',
                 fontsize=18, fontweight='bold', y=0.995)
    
    all_stats = []
    
    # =========================================================================
    # Panel A: N7 vs N17 Rest
    # =========================================================================
    ax = fig.add_subplot(gs[0, 0])
    if 'N7_rest' in error_data and 'N17_rest' in error_data:
        n7 = error_data['N7_rest']['error_rate'].values
        n17 = error_data['N17_rest']['error_rate'].values
        n7_labels = error_data['N7_rest']['network'].values
        n17_labels = error_data['N17_rest']['network'].values
        
        stat = compute_statistics(n7, n17, "N7", "N17")
        all_stats.append({**stat, 'comparison': 'Cortical N7 vs N17', 'condition': 'Rest'})
        
        log_statistics(stat, "Cortical Resolution (Rest): N7 vs N17")
        
        bp = ax.boxplot([n7, n17], labels=['N7\n(7 networks)', 'N17\n(17 networks)'],
                       patch_artist=True, widths=0.5,
                       boxprops=dict(linewidth=2),
                       medianprops=dict(linewidth=3, color='black'),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
        
        bp['boxes'][0].set_facecolor(COLORS['n7'])
        bp['boxes'][1].set_facecolor(COLORS['n17'])
        
        # Add means
        means = [stat['mean1'], stat['mean2']]
        ax.plot([1, 2], means, 'D', color='white', markersize=14,
               markeredgecolor='black', markeredgewidth=2.5, label='Mean', zorder=5)
        
        # Annotate outliers
        annotate_outliers(ax, n7, 1, n7_labels, n_top=2, direction='left')
        annotate_outliers(ax, n17, 2, n17_labels, n_top=2, direction='right')
        
        ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax.set_title('A) Cortical Networks (Rest)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel B: N7 vs N17 Task
    # =========================================================================
    ax = fig.add_subplot(gs[0, 1])
    if 'N7_task' in error_data and 'N17_task' in error_data:
        n7 = error_data['N7_task']['error_rate'].values
        n17 = error_data['N17_task']['error_rate'].values
        n7_labels = error_data['N7_task']['network'].values
        n17_labels = error_data['N17_task']['network'].values
        
        stat = compute_statistics(n7, n17, "N7", "N17")
        all_stats.append({**stat, 'comparison': 'Cortical N7 vs N17', 'condition': 'Task'})
        
        log_statistics(stat, "Cortical Resolution (Task): N7 vs N17")
        
        bp = ax.boxplot([n7, n17], labels=['N7\n(7 networks)', 'N17\n(17 networks)'],
                       patch_artist=True, widths=0.5,
                       boxprops=dict(linewidth=2),
                       medianprops=dict(linewidth=3, color='black'),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
        
        bp['boxes'][0].set_facecolor(COLORS['n7'])
        bp['boxes'][1].set_facecolor(COLORS['n17'])
        
        means = [stat['mean1'], stat['mean2']]
        ax.plot([1, 2], means, 'D', color='white', markersize=14,
               markeredgecolor='black', markeredgewidth=2.5, label='Mean', zorder=5)
        
        annotate_outliers(ax, n7, 1, n7_labels, n_top=2, direction='left')
        annotate_outliers(ax, n17, 2, n17_labels, n_top=2, direction='right')
        
        ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax.set_title('B) Cortical Networks (Task)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel C: Tian I vs II Rest
    # =========================================================================
    ax = fig.add_subplot(gs[1, 0])
    if 'TianI_rest' in error_data and 'TianII_rest' in error_data:
        t1 = error_data['TianI_rest']['error_rate'].values
        t2 = error_data['TianII_rest']['error_rate'].values
        t1_labels = error_data['TianI_rest']['network'].values
        t2_labels = error_data['TianII_rest']['network'].values
        
        stat = compute_statistics(t1, t2, "Tian I", "Tian II")
        all_stats.append({**stat, 'comparison': 'Subcortical Tian I vs II', 'condition': 'Rest'})
        
        log_statistics(stat, "Subcortical Resolution (Rest): Tian I vs II")
        
        bp = ax.boxplot([t1, t2], labels=['Tian I\n(8 regions)', 'Tian II\n(16 regions)'],
                       patch_artist=True, widths=0.5,
                       boxprops=dict(linewidth=2),
                       medianprops=dict(linewidth=3, color='black'),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
        
        bp['boxes'][0].set_facecolor(COLORS['tian1'])
        bp['boxes'][1].set_facecolor(COLORS['tian2'])
        
        means = [stat['mean1'], stat['mean2']]
        ax.plot([1, 2], means, 'D', color='white', markersize=14,
               markeredgecolor='black', markeredgewidth=2.5, label='Mean', zorder=5)
        
        annotate_outliers(ax, t1, 1, t1_labels, n_top=2, direction='left')
        annotate_outliers(ax, t2, 2, t2_labels, n_top=2, direction='right')
        
        ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax.set_title('C) Subcortical Regions (Rest)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel D: Tian I vs II Task
    # =========================================================================
    ax = fig.add_subplot(gs[1, 1])
    if 'TianI_task' in error_data and 'TianII_task' in error_data:
        t1 = error_data['TianI_task']['error_rate'].values
        t2 = error_data['TianII_task']['error_rate'].values
        t1_labels = error_data['TianI_task']['network'].values
        t2_labels = error_data['TianII_task']['network'].values
        
        stat = compute_statistics(t1, t2, "Tian I", "Tian II")
        all_stats.append({**stat, 'comparison': 'Subcortical Tian I vs II', 'condition': 'Task'})
        
        log_statistics(stat, "Subcortical Resolution (Task): Tian I vs II")
        
        bp = ax.boxplot([t1, t2], labels=['Tian I\n(8 regions)', 'Tian II\n(16 regions)'],
                       patch_artist=True, widths=0.5,
                       boxprops=dict(linewidth=2),
                       medianprops=dict(linewidth=3, color='black'),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
        
        bp['boxes'][0].set_facecolor(COLORS['tian1'])
        bp['boxes'][1].set_facecolor(COLORS['tian2'])
        
        means = [stat['mean1'], stat['mean2']]
        ax.plot([1, 2], means, 'D', color='white', markersize=14,
               markeredgecolor='black', markeredgewidth=2.5, label='Mean', zorder=5)
        
        annotate_outliers(ax, t1, 1, t1_labels, n_top=2, direction='left')
        annotate_outliers(ax, t2, 2, t2_labels, n_top=2, direction='right')
        
        ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax.set_title('D) Subcortical Regions (Task)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path.name}")
    
    return pd.DataFrame(all_stats)


# =============================================================================
# FIGURE 2: CORTICAL VS SUBCORTICAL 
# =============================================================================

def plot_cortical_vs_subcortical(error_data, output_path):
    """
    No statistical annotations on graphs, full stats logged.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cortical vs Subcortical Classification Performance',
                 fontsize=18, fontweight='bold', y=0.995)
    
    all_stats = []
    
    comparisons = [
        ('N7_rest', 'TianI_rest', 'Coarse (Rest)', axes[0, 0], 'A'),
        ('N7_task', 'TianI_task', 'Coarse (Task)', axes[0, 1], 'B'),
        ('N17_rest', 'TianII_rest', 'Fine (Rest)', axes[1, 0], 'C'),
        ('N17_task', 'TianII_task', 'Fine (Task)', axes[1, 1], 'D'),
    ]
    
    for cort_key, subcort_key, title, ax, panel in comparisons:
        if cort_key in error_data and subcort_key in error_data:
            cort = error_data[cort_key]['error_rate'].values
            subcort = error_data[subcort_key]['error_rate'].values
            cort_labels = error_data[cort_key]['network'].values
            subcort_labels = error_data[subcort_key]['network'].values
            
            stat = compute_statistics(cort, subcort, "Cortical", "Subcortical")
            all_stats.append({**stat, 'comparison': title})
            
            log_statistics(stat, f"Cortical vs Subcortical: {title}")
            
            bp = ax.boxplot([cort, subcort],
                           labels=['Cortical', 'Subcortical'],
                           patch_artist=True, widths=0.5,
                           boxprops=dict(linewidth=2),
                           medianprops=dict(linewidth=3, color='black'),
                           whiskerprops=dict(linewidth=2),
                           capprops=dict(linewidth=2))
            
            bp['boxes'][0].set_facecolor(COLORS['cortical'])
            bp['boxes'][1].set_facecolor(COLORS['subcortical'])
            
            means = [stat['mean1'], stat['mean2']]
            ax.plot([1, 2], means, 'D', color='white', markersize=14,
                   markeredgecolor='black', markeredgewidth=2.5, label='Mean', zorder=5)
            
            annotate_outliers(ax, cort, 1, cort_labels, n_top=2, direction='left')
            annotate_outliers(ax, subcort, 2, subcort_labels, n_top=2, direction='right')
            
            ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
            ax.set_title(f'{panel}) {title}', fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path.name}")
    
    return pd.DataFrame(all_stats)


# =============================================================================
# FIGURE 3: TASK EFFECTS 
# =============================================================================

def plot_task_effects(error_data, output_path):
    """
    Full paired statistics logged to console.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Task-Induced Changes in Classification Performance',
                 fontsize=18, fontweight='bold', y=0.995)
    
    all_stats = []
    
    atlas_configs = [
        ('N7 Cortical', 'N7_rest', 'N7_task', axes[0, 0], 'A'),
        ('N17 Cortical', 'N17_rest', 'N17_task', axes[0, 1], 'B'),
        ('Tian I Subcortical', 'TianI_rest', 'TianI_task', axes[1, 0], 'C'),
        ('Tian II Subcortical', 'TianII_rest', 'TianII_task', axes[1, 1], 'D'),
    ]
    
    for name, rest_key, task_key, ax, panel in atlas_configs:
        if rest_key in error_data and task_key in error_data:
            rest_df = error_data[rest_key]
            task_df = error_data[task_key]
            
            # Merge on network names
            merged = pd.merge(
                rest_df[['network', 'error_rate']],
                task_df[['network', 'error_rate']],
                on='network', suffixes=('_rest', '_task')
            )
            
            if len(merged) > 0:
                rest_vals = merged['error_rate_rest'].values
                task_vals = merged['error_rate_task'].values
                
                stat = paired_statistics(rest_vals, task_vals, "Rest", "Task")
                all_stats.append({**stat, 'atlas': name})
                
                log_statistics(stat, f"Task Effects: {name}")
                
                # Create paired plot
                y_pos = np.arange(len(merged))
                
                bars_rest = ax.barh(y_pos - 0.2, rest_vals, 0.4, label='Rest',
                       color=COLORS['rest'], alpha=0.8, edgecolor='black', linewidth=1)
                bars_task = ax.barh(y_pos + 0.2, task_vals, 0.4, label='Task',
                       color=COLORS['task'], alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for i, (rest_val, task_val) in enumerate(zip(rest_vals, task_vals)):
                    # Rest value
                    ax.text(rest_val, i - 0.2, f'{rest_val:.3f}', 
                           va='center', ha='left' if rest_val > 0.02 else 'right',
                           fontsize=8, fontweight='bold', color='black')
                    # Task value
                    ax.text(task_val, i + 0.2, f'{task_val:.3f}',
                           va='center', ha='left' if task_val > 0.02 else 'right',
                           fontsize=8, fontweight='bold', color='black')
                
                # Connect pairs with lines
                for i, (r, t) in enumerate(zip(rest_vals, task_vals)):
                    if t > r:
                        ax.plot([r, t], [i-0.2, i+0.2], 'r-', alpha=0.3, linewidth=1)
                    else:
                        ax.plot([r, t], [i-0.2, i+0.2], 'b-', alpha=0.3, linewidth=1)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(merged['network'], fontsize=10 if len(merged) < 10 else 8)
                ax.set_xlabel('Error Rate', fontsize=11, fontweight='bold')
                ax.set_title(f'{panel}) {name}', fontsize=12, fontweight='bold', pad=10)
                ax.legend(fontsize=10)
                ax.grid(axis='x', alpha=0.3)
                ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path.name}")
    
    return pd.DataFrame(all_stats)


# =============================================================================
# FIGURE 4: DISTRIBUTION PLOTS
# =============================================================================

def plot_distributions(error_data, output_path):
    """
    Distribution comparisons across atlases.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Error Rate Distributions Across Parcellation Schemes',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: Violin plots (Rest)
    ax = fig.add_subplot(gs[0, 0])
    
    data_rest = []
    labels_rest = []
    colors_rest = []
    
    for key, label, color in [
        ('N7_rest', 'N7\nCortical', COLORS['n7']),
        ('N17_rest', 'N17\nCortical', COLORS['n17']),
        ('TianI_rest', 'Tian I\nSubcort', COLORS['tian1']),
        ('TianII_rest', 'Tian II\nSubcort', COLORS['tian2']),
    ]:
        if key in error_data:
            data_rest.append(error_data[key]['error_rate'].values)
            labels_rest.append(label)
            colors_rest.append(color)
    
    if data_rest:
        parts = ax.violinplot(data_rest, positions=range(len(data_rest)),
                             widths=0.7, showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(parts['bodies'], colors_rest)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels_rest)))
        ax.set_xticklabels(labels_rest, fontsize=11)
        ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax.set_title('A) Distribution Comparison (Rest)', fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Violin plots (Task)
    ax = fig.add_subplot(gs[0, 1])
    
    data_task = []
    labels_task = []
    colors_task = []
    
    for key, label, color in [
        ('N7_task', 'N7\nCortical', COLORS['n7']),
        ('N17_task', 'N17\nCortical', COLORS['n17']),
        ('TianI_task', 'Tian I\nSubcort', COLORS['tian1']),
        ('TianII_task', 'Tian II\nSubcort', COLORS['tian2']),
    ]:
        if key in error_data:
            data_task.append(error_data[key]['error_rate'].values)
            labels_task.append(label)
            colors_task.append(color)
    
    if data_task:
        parts = ax.violinplot(data_task, positions=range(len(data_task)),
                             widths=0.7, showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(parts['bodies'], colors_task)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels_task)))
        ax.set_xticklabels(labels_task, fontsize=11)
        ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax.set_title('B) Distribution Comparison (Task)', fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Histograms (Rest)
    ax = fig.add_subplot(gs[1, 0])
    
    for data, label, color in zip(data_rest, labels_rest, colors_rest):
        ax.hist(data, bins=15, alpha=0.5, label=label.replace('\n', ' '),
               color=color, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Error Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('C) Error Rate Histograms (Rest)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel D: Histograms (Task)
    ax = fig.add_subplot(gs[1, 1])
    
    for data, label, color in zip(data_task, labels_task, colors_task):
        ax.hist(data, bins=15, alpha=0.5, label=label.replace('\n', ' '),
               color=color, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Error Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('D) Error Rate Histograms (Task)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("ATLAS COMPARISON - ULTRA CLEAN GRAPHS")
    print("="*80)
    print("All detailed statistics are logged below and saved in CSV tables\n")
    
    # Load data
    tables_dir = Path('reports/tables/atlas_analysis')
    
    if not tables_dir.exists():
        print(f"\n✗ Error: {tables_dir} not found")
        print("Run 01_atlas_performance_analysis.py first!")
        return 1
    
    print("\nLoading error rates...")
    error_data = load_error_rates(tables_dir)
    
    if not error_data:
        print("✗ No data found!")
        return 1
    
    print(f"✓ Loaded {len(error_data)} datasets\n")
    
    # Output directories
    output_figures = Path('reports/figures/atlas_comparison')
    output_tables = Path('reports/tables/atlas_comparison')
    output_figures.mkdir(parents=True, exist_ok=True)
    output_tables.mkdir(parents=True, exist_ok=True)
    
    # Generate figures with comprehensive statistics logged
    print("\n" + "="*80)
    print("GENERATING ULTRA CLEAN FIGURES AND LOGGING STATISTICS")
    print("="*80)
    
    print("\n" + "▶"*40)
    print("  FIGURE 1: RESOLUTION EFFECTS")
    print("▶"*40)
    resolution_stats = plot_resolution_effects(error_data,
                                               output_figures / 'fig1_resolution_effects.png')
    if resolution_stats is not None and len(resolution_stats) > 0:
        resolution_stats.to_csv(output_tables / 'resolution_statistics.csv', index=False)
        print(f"\n✓ Saved statistics table: resolution_statistics.csv")
    
    print("\n" + "▶"*40)
    print("  FIGURE 2: CORTICAL VS SUBCORTICAL")
    print("▶"*40)
    system_stats = plot_cortical_vs_subcortical(error_data,
                                                output_figures / 'fig2_cortical_vs_subcortical.png')
    if system_stats is not None and len(system_stats) > 0:
        system_stats.to_csv(output_tables / 'system_statistics.csv', index=False)
        print(f"\n✓ Saved statistics table: system_statistics.csv")
    
    print("\n" + "▶"*40)
    print("  FIGURE 3: TASK EFFECTS")
    print("▶"*40)
    task_stats = plot_task_effects(error_data,
                                   output_figures / 'fig3_task_effects.png')
    if task_stats is not None and len(task_stats) > 0:
        task_stats.to_csv(output_tables / 'task_statistics.csv', index=False)
        print(f"\n✓ Saved statistics table: task_statistics.csv")
    
    print("\n" + "▶"*40)
    print("  FIGURE 4: DISTRIBUTIONS")
    print("▶"*40)
    plot_distributions(error_data,
                      output_figures / 'fig4_distributions.png')
    
    print(f"""

{"="*80}
✓ ANALYSIS COMPLETE
{"="*80}

Generated Files:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Figures ({output_figures}):
  • fig1_resolution_effects.png
  • fig2_cortical_vs_subcortical.png
  • fig3_task_effects.png
  • fig4_distributions.png

Statistical Tables ({output_tables}):
  • resolution_statistics.csv
  • system_statistics.csv
  • task_statistics.csv
{"="*80}
Perfect: graphs + Complete statistics documented!
{"="*80}
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())