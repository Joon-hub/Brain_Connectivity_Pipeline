#!/usr/bin/env python3
"""
Atlas Comparison Analysis
=========================
Compares model performance across different atlas configurations:
- N7 vs N17 (coarse vs fine cortical parcellation)
- Tian Scale I vs Scale II (coarse vs fine subcortical parcellation)
- Cortical vs Subcortical performance
- Rest vs Task differences across all atlases

Usage:
    python 02_atlas_comparison.py

Requirements:
    - Must run 01_atlas_performance_analysis.py first to generate input files

Outputs:
    - Comparative performance metrics
    - Visualization of atlas differences
    - Statistical tests of performance differences
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10


def load_error_rates(tables_dir):
    """Load all error rate CSV files."""
    error_data = {}
    
    files = {
        'N7_rest': 'error_rates_N7_rest.csv',
        'N7_task': 'error_rates_N7_task.csv',
        'N17_rest': 'error_rates_N17_rest.csv',
        'N17_task': 'error_rates_N17_task.csv',
        'TianI_rest': 'error_rates_TianI_rest.csv',
        'TianI_task': 'error_rates_TianI_task.csv',
        'TianII_rest': 'error_rates_TianII_rest.csv',
        'TianII_task': 'error_rates_TianII_task.csv'
    }
    
    for key, filename in files.items():
        filepath = tables_dir / filename
        if filepath.exists():
            error_data[key] = pd.read_csv(filepath)
        else:
            print(f"⚠ Warning: {filepath} not found")
    
    return error_data


def compare_resolution_effects(error_data):
    """
    Compare performance between coarse and fine parcellations.
    
    Returns:
        DataFrame with comparison statistics
    """
    results = []
    
    # N7 vs N17 (Cortical)
    if 'N7_rest' in error_data and 'N17_rest' in error_data:
        n7_mean = error_data['N7_rest']['error_rate'].mean()
        n17_mean = error_data['N17_rest']['error_rate'].mean()
        
        results.append({
            'comparison': 'Cortical: N7 vs N17 (Rest)',
            'coarse_mean': n7_mean,
            'fine_mean': n17_mean,
            'difference': n17_mean - n7_mean,
            'coarse_n': len(error_data['N7_rest']),
            'fine_n': len(error_data['N17_rest'])
        })
    
    if 'N7_task' in error_data and 'N17_task' in error_data:
        n7_mean = error_data['N7_task']['error_rate'].mean()
        n17_mean = error_data['N17_task']['error_rate'].mean()
        
        results.append({
            'comparison': 'Cortical: N7 vs N17 (Task)',
            'coarse_mean': n7_mean,
            'fine_mean': n17_mean,
            'difference': n17_mean - n7_mean,
            'coarse_n': len(error_data['N7_task']),
            'fine_n': len(error_data['N17_task'])
        })
    
    # Tian I vs II (Subcortical)
    if 'TianI_rest' in error_data and 'TianII_rest' in error_data:
        # Filter to only subcortical regions
        tian1_sub = error_data['TianI_rest'][error_data['TianI_rest']['network'] != 'Cortical']
        tian2_sub = error_data['TianII_rest'][error_data['TianII_rest']['network'] != 'Cortical']
        
        if len(tian1_sub) > 0 and len(tian2_sub) > 0:
            t1_mean = tian1_sub['error_rate'].mean()
            t2_mean = tian2_sub['error_rate'].mean()
            
            results.append({
                'comparison': 'Subcortical: Tian I vs II (Rest)',
                'coarse_mean': t1_mean,
                'fine_mean': t2_mean,
                'difference': t2_mean - t1_mean,
                'coarse_n': len(tian1_sub),
                'fine_n': len(tian2_sub)
            })
    
    if 'TianI_task' in error_data and 'TianII_task' in error_data:
        tian1_sub = error_data['TianI_task'][error_data['TianI_task']['network'] != 'Cortical']
        tian2_sub = error_data['TianII_task'][error_data['TianII_task']['network'] != 'Cortical']
        
        if len(tian1_sub) > 0 and len(tian2_sub) > 0:
            t1_mean = tian1_sub['error_rate'].mean()
            t2_mean = tian2_sub['error_rate'].mean()
            
            results.append({
                'comparison': 'Subcortical: Tian I vs II (Task)',
                'coarse_mean': t1_mean,
                'fine_mean': t2_mean,
                'difference': t2_mean - t1_mean,
                'coarse_n': len(tian1_sub),
                'fine_n': len(tian2_sub)
            })
    
    df = pd.DataFrame(results)
    return df


def compare_cortical_vs_subcortical(error_data):
    """
    Compare error rates between cortical and subcortical regions.
    
    Returns:
        DataFrame with comparison statistics
    """
    results = []
    
    # Use N7 for cortical (broader categories)
    if 'N7_rest' in error_data and 'TianI_rest' in error_data:
        # Cortical networks from N7
        cortical_rest = error_data['N7_rest'][
            ~error_data['N7_rest']['network'].isin(['Hippocampus', 'Amygdala', 'Thalamus', 
                                                     'Accumbens', 'Putamen', 'Pallidum', 'Caudate'])
        ]
        
        # Subcortical from Tian I
        subcortical_rest = error_data['TianI_rest'][
            error_data['TianI_rest']['network'] != 'Cortical'
        ]
        
        if len(cortical_rest) > 0 and len(subcortical_rest) > 0:
            cort_mean = cortical_rest['error_rate'].mean()
            subcort_mean = subcortical_rest['error_rate'].mean()
            
            # T-test
            t_stat, p_val = stats.ttest_ind(
                cortical_rest['error_rate'],
                subcortical_rest['error_rate']
            )
            
            results.append({
                'condition': 'Rest',
                'cortical_mean': cort_mean,
                'cortical_std': cortical_rest['error_rate'].std(),
                'subcortical_mean': subcort_mean,
                'subcortical_std': subcortical_rest['error_rate'].std(),
                'difference': subcort_mean - cort_mean,
                't_statistic': t_stat,
                'p_value': p_val,
                'n_cortical': len(cortical_rest),
                'n_subcortical': len(subcortical_rest)
            })
    
    # Same for task
    if 'N7_task' in error_data and 'TianI_task' in error_data:
        cortical_task = error_data['N7_task'][
            ~error_data['N7_task']['network'].isin(['Hippocampus', 'Amygdala', 'Thalamus',
                                                     'Accumbens', 'Putamen', 'Pallidum', 'Caudate'])
        ]
        
        subcortical_task = error_data['TianI_task'][
            error_data['TianI_task']['network'] != 'Cortical'
        ]
        
        if len(cortical_task) > 0 and len(subcortical_task) > 0:
            cort_mean = cortical_task['error_rate'].mean()
            subcort_mean = subcortical_task['error_rate'].mean()
            
            t_stat, p_val = stats.ttest_ind(
                cortical_task['error_rate'],
                subcortical_task['error_rate']
            )
            
            results.append({
                'condition': 'Task',
                'cortical_mean': cort_mean,
                'cortical_std': cortical_task['error_rate'].std(),
                'subcortical_mean': subcort_mean,
                'subcortical_std': subcortical_task['error_rate'].std(),
                'difference': subcort_mean - cort_mean,
                't_statistic': t_stat,
                'p_value': p_val,
                'n_cortical': len(cortical_task),
                'n_subcortical': len(subcortical_task)
            })
    
    df = pd.DataFrame(results)
    return df


def compare_rest_vs_task(error_data):
    """
    Compare error rates between rest and task conditions across all atlases.
    
    Returns:
        DataFrame with rest vs task comparison
    """
    results = []
    
    comparisons = [
        ('N7', 'N7_rest', 'N7_task'),
        ('N17', 'N17_rest', 'N17_task'),
        ('TianI', 'TianI_rest', 'TianI_task'),
        ('TianII', 'TianII_rest', 'TianII_task')
    ]
    
    for atlas, rest_key, task_key in comparisons:
        if rest_key in error_data and task_key in error_data:
            rest_df = error_data[rest_key]
            task_df = error_data[task_key]
            
            # Merge on network name
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
                
                # Paired t-test
                t_stat, p_val = stats.ttest_rel(
                    merged['error_rate_rest'],
                    merged['error_rate_task']
                )
                
                results.append({
                    'atlas': atlas,
                    'rest_mean': mean_rest,
                    'task_mean': mean_task,
                    'mean_increase': mean_increase,
                    'pct_increase': (mean_increase / mean_rest * 100) if mean_rest > 0 else 0,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'n_networks': len(merged)
                })
    
    df = pd.DataFrame(results)
    return df


def plot_atlas_comparison(error_data, output_path):
    """
    Create comprehensive atlas comparison figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Resolution comparison (N7 vs N17)
    ax1 = axes[0, 0]
    if 'N7_rest' in error_data and 'N17_rest' in error_data:
        data = [
            error_data['N7_rest']['error_rate'],
            error_data['N17_rest']['error_rate']
        ]
        labels = ['N7 (7 networks)', 'N17 (17 networks)']
        
        bp = ax1.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Error Rate', fontweight='bold')
        ax1.set_title('A) Cortical Parcellation Resolution (Rest)', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Cortical vs Subcortical
    ax2 = axes[0, 1]
    if 'N7_rest' in error_data and 'TianI_rest' in error_data:
        # Get cortical networks
        cortical = error_data['N7_rest'][
            ~error_data['N7_rest']['network'].isin(['Hippocampus', 'Amygdala', 'Thalamus',
                                                     'Accumbens', 'Putamen', 'Pallidum', 'Caudate'])
        ]
        
        # Get subcortical
        subcortical = error_data['TianI_rest'][
            error_data['TianI_rest']['network'] != 'Cortical'
        ]
        
        if len(cortical) > 0 and len(subcortical) > 0:
            data = [cortical['error_rate'], subcortical['error_rate']]
            labels = ['Cortical', 'Subcortical']
            
            bp = ax2.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['lightgreen', 'lightyellow']):
                patch.set_facecolor(color)
            
            ax2.set_ylabel('Error Rate', fontweight='bold')
            ax2.set_title('B) Cortical vs Subcortical (Rest)', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Rest vs Task across atlases
    ax3 = axes[1, 0]
    rest_task_data = []
    atlas_labels = []
    
    for atlas, rest_key, task_key in [('N7', 'N7_rest', 'N7_task'),
                                       ('N17', 'N17_rest', 'N17_task')]:
        if rest_key in error_data and task_key in error_data:
            rest_mean = error_data[rest_key]['error_rate'].mean()
            task_mean = error_data[task_key]['error_rate'].mean()
            rest_task_data.append([rest_mean, task_mean])
            atlas_labels.append(atlas)
    
    if rest_task_data:
        rest_task_array = np.array(rest_task_data)
        x = np.arange(len(atlas_labels))
        width = 0.35
        
        ax3.bar(x - width/2, rest_task_array[:, 0], width, label='Rest', color='steelblue', alpha=0.7)
        ax3.bar(x + width/2, rest_task_array[:, 1], width, label='Task', color='coral', alpha=0.7)
        
        ax3.set_ylabel('Mean Error Rate', fontweight='bold')
        ax3.set_xlabel('Atlas', fontweight='bold')
        ax3.set_title('C) Rest vs Task Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(atlas_labels)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Error increase from rest to task
    ax4 = axes[1, 1]
    increase_data = []
    network_labels = []
    
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
        merged = merged.sort_values('increase', ascending=False)
        
        if len(merged) > 0:
            colors = ['red' if x > 0 else 'green' for x in merged['increase']]
            ax4.barh(range(len(merged)), merged['increase'], color=colors, alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(merged)))
            ax4.set_yticklabels(merged['network'], fontsize=9)
            ax4.set_xlabel('Error Rate Increase (Task - Rest)', fontweight='bold')
            ax4.set_title('D) Task-Induced Error Changes (N7)', fontweight='bold')
            ax4.axvline(0, color='black', linewidth=1)
            ax4.invert_yaxis()
            ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Atlas Configuration Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison figure: {output_path}")


def main():
    print("="*70)
    print("ATLAS COMPARISON ANALYSIS")
    print("="*70)
    
    # Load error rates
    tables_dir = Path('reports/tables/atlas_analysis')
    
    if not tables_dir.exists():
        print(f"\n❌ Error: {tables_dir} not found")
        print("Please run 01_atlas_performance_analysis.py first!")
        return 1
    
    print("\nLoading error rate data...")
    error_data = load_error_rates(tables_dir)
    
    if not error_data:
        print("❌ No error rate files found!")
        return 1
    
    print(f"✓ Loaded {len(error_data)} error rate files")
    
    # Create output directories
    output_tables = Path('reports/tables/atlas_comparison')
    output_figures = Path('reports/figures/atlas_comparison')
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    
    # Analysis 1: Resolution effects
    print("\n" + "="*70)
    print("ANALYSIS 1: Resolution Effects (Coarse vs Fine Parcellation)")
    print("="*70)
    
    resolution_comparison = compare_resolution_effects(error_data)
    resolution_comparison.to_csv(output_tables / 'resolution_comparison.csv', index=False)
    
    print("\nResolution Comparison:")
    print(resolution_comparison.to_string(index=False))
    
    # Analysis 2: Cortical vs Subcortical
    print("\n" + "="*70)
    print("ANALYSIS 2: Cortical vs Subcortical Performance")
    print("="*70)
    
    cortical_subcortical = compare_cortical_vs_subcortical(error_data)
    cortical_subcortical.to_csv(output_tables / 'cortical_vs_subcortical.csv', index=False)
    
    print("\nCortical vs Subcortical:")
    print(cortical_subcortical.to_string(index=False))
    
    # Analysis 3: Rest vs Task
    print("\n" + "="*70)
    print("ANALYSIS 3: Rest vs Task Effects Across Atlases")
    print("="*70)
    
    rest_task_comparison = compare_rest_vs_task(error_data)
    rest_task_comparison.to_csv(output_tables / 'rest_vs_task_comparison.csv', index=False)
    
    print("\nRest vs Task Comparison:")
    print(rest_task_comparison.to_string(index=False))
    
    # Create visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)
    
    plot_atlas_comparison(error_data, output_figures / 'atlas_comparison.png')
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    print(f"""
Generated Files:
================

Tables:
  - {output_tables}/resolution_comparison.csv
  - {output_tables}/cortical_vs_subcortical.csv
  - {output_tables}/rest_vs_task_comparison.csv

Figures:
  - {output_figures}/atlas_comparison.png

Key Findings:
=============
""")
    
    # Print key insights
    if not resolution_comparison.empty:
        print("\n1. RESOLUTION EFFECTS:")
        for _, row in resolution_comparison.iterrows():
            print(f"   {row['comparison']}: Fine - Coarse = {row['difference']:.4f}")
    
    if not cortical_subcortical.empty:
        print("\n2. CORTICAL VS SUBCORTICAL:")
        for _, row in cortical_subcortical.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "n.s."
            print(f"   {row['condition']}: Subcortical - Cortical = {row['difference']:.4f} (p = {row['p_value']:.4f}) {sig}")
    
    if not rest_task_comparison.empty:
        print("\n3. REST VS TASK:")
        for _, row in rest_task_comparison.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "n.s."
            print(f"   {row['atlas']}: Task - Rest = {row['mean_increase']:.4f} ({row['pct_increase']:.1f}% increase, p = {row['p_value']:.4f}) {sig}")
    
    print("\n✅ Atlas comparison complete!\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())