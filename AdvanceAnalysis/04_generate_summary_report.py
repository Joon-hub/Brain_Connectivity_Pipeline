#!/usr/bin/env python3
"""
Summary Report Generator 
Creates summary statistics, key findings, and visualization from analysis results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results():
    """Load all CSV results from previous analyses."""
    results = {'error_rates': {}, 'comparisons': {}, 'connectivity': {}}
    
    # Load atlas performance results
    atlas_dir = Path('reports/tables/atlas_analysis')
    if atlas_dir.exists():
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
            filepath = atlas_dir / filename
            if filepath.exists():
                results['error_rates'][key] = pd.read_csv(filepath)
                print(f"  ✓ {filename}")
    
    # Load comparison results
    comp_dir = Path('reports/tables/atlas_comparison')
    if comp_dir.exists():
        files = {
            'resolution': 'resolution_comparison.csv',
            'cortical_subcortical': 'cortical_vs_subcortical.csv',
            'rest_task': 'rest_vs_task_comparison.csv'
        }
        for key, filename in files.items():
            filepath = comp_dir / filename
            if filepath.exists():
                results['comparisons'][key] = pd.read_csv(filepath)
                print(f"  ✓ {filename}")
    
    # Load connectivity results
    conn_dir = Path('reports/tables/connectivity_analysis')
    if conn_dir.exists():
        files = {
            'rest': 'inter_network_connectivity_rest.csv',
            'task': 'inter_network_connectivity_task.csv',
            'change': 'inter_network_connectivity_change.csv',
            'top_changes': 'top_changed_connections.csv'
        }
        for key, filename in files.items():
            filepath = conn_dir / filename
            if filepath.exists():
                results['connectivity'][key] = pd.read_csv(filepath, index_col=0)
                print(f"  ✓ {filename}")
    
    return results


def create_summary_stats(results):
    """Generate summary statistics table."""
    rows = []

    # Performance metrics
    for key, df in results['error_rates'].items():
        parts = key.split('_')
        if len(parts) < 2:
            continue
        atlas = parts[0]
        condition = parts[1]
        mean_error = df['error_rate'].mean()
        std_error = df['error_rate'].std()
        accuracy = 1 - mean_error
        rows.append({
            'category': 'Performance',
            'metric': f'{atlas} {condition} - Accuracy',
            'value': accuracy,
            'formatted': f'{accuracy:.3f} ± {std_error:.3f}'
        })

    # Task effect comparison
    if 'rest_task' in results['comparisons']:
        df = results['comparisons']['rest_task']
        for _, row in df.iterrows():
            rows.append({
                'category': 'Task Effect',
                'metric': f"{row['atlas']}",
                'value': row['mean_increase'],
                'formatted': f"+{row['mean_increase']:.3f} ({row['pct_increase']:+.1f}%)"
            })

    # Cortical vs Subcortical - with safe column access
    if 'cortical_subcortical' in results['comparisons']:
        df = results['comparisons']['cortical_subcortical']
        # Debug: show actual columns
        print("DEBUG: columns in cortical_vs_subcortical.csv:", df.columns.tolist())
        
        # Try common variations of p-value column
        p_col = None
        for col in ['p_value', 'pvalue', 'p_val', 'p', 'p-value', 'pval']:
            if col in df.columns:
                p_col = col
                break

        for _, row in df.iterrows():
            diff = row['difference']
            if p_col is not None and pd.notna(row[p_col]):
                p_val = row[p_col]
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            else:
                p_str = "p=n/a"
            formatted = f"{diff:+.3f} ({p_str})"
            
            rows.append({
                'category': 'Region Type',
                'metric': f"Subcortical vs Cortical ({row['condition']})",
                'value': diff,
                'formatted': formatted
            })

    return pd.DataFrame(rows)


def create_key_findings(results):
    """Generate key findings report."""
    findings = []
    
    # Finding 1: Overall performance
    if 'N7_rest' in results['error_rates']:
        df = results['error_rates']['N7_rest']
        accuracy = 1 - df['error_rate'].mean()
        n_networks = len(df)
        chance = 1.0 / n_networks
        
        findings.append({
            'finding': 'Classification Performance',
            'result': f"{accuracy:.1%} accuracy across {n_networks} networks ({accuracy/chance:.1f}× better than chance)",
            'interpretation': "Networks have unique connectivity fingerprints that can be reliably identified",
            'significance': 'High'
        })
    
    # Finding 2: Task effects
    if 'rest_task' in results['comparisons']:
        df = results['comparisons']['rest_task']
        cortical = df[df['atlas'].str.contains('Cortical')]
        if len(cortical) > 0:
            mean_increase = cortical['mean_increase'].mean()
            findings.append({
                'finding': 'Task-Induced Changes',
                'result': f"Error increased by {mean_increase:.1%} during task",
                'interpretation': "Task engagement reorganizes connectivity patterns, showing network flexibility",
                'significance': 'High'
            })
    
    # Finding 3: Network hierarchy
    if 'N7_rest' in results['error_rates']:
        df = results['error_rates']['N7_rest']
        if len(df) > 0:
            best = df.loc[df['error_rate'].idxmin()]
            worst = df.loc[df['error_rate'].idxmax()]
            findings.append({
                'finding': 'Network Variability',
                'result': f"Best: {best['network']} ({1-best['error_rate']:.1%}), Worst: {worst['network']} ({1-worst['error_rate']:.1%})",
                'interpretation': "Sensory networks more stable than cognitive networks, supporting hierarchical organization",
                'significance': 'Moderate'
            })
    
    # Finding 4: Cortical vs subcortical
    if 'cortical_subcortical' in results['comparisons']:
        df = results['comparisons']['cortical_subcortical']
        rest = df[df['condition'] == 'Rest']
        if len(rest) > 0:
            diff = rest.iloc[0]['difference']
            findings.append({
                'finding': 'Subcortical Difficulty',
                'result': f"Subcortical regions {abs(diff):.1%} harder to classify",
                'interpretation': "Smaller subcortical structures have less distinctive connectivity patterns",
                'significance': 'Moderate'
            })
    
    return pd.DataFrame(findings)


def plot_summary(results, output_path):
    """Create comprehensive summary figure (2×3 panels)."""
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    
    # Panel A: Performance across atlases
    ax1 = fig.add_subplot(gs[0, 0])
    data, labels, colors = [], [], []
    for atlas in ['N7', 'N17', 'TianI', 'TianII']:
        for cond in ['rest', 'task']:
            key = f'{atlas}_{cond}'
            if key in results['error_rates']:
                acc = 1 - results['error_rates'][key]['error_rate'].mean()
                data.append(acc)
                labels.append(f'{atlas}\n{cond}')
                colors.append('#3498DB' if cond == 'rest' else "#B3847F")
    
    if data:
        x = np.arange(len(data))
        ax1.bar(x, data, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('A) Classification Accuracy', fontweight='bold', fontsize=12)
        ax1.set_ylim([0, 1.0])
        ax1.grid(axis='y', alpha=0.3)
        # annotate accuracy
        for i, v in enumerate(data):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
       
    
    # Panel B: Rest vs Task
    ax2 = fig.add_subplot(gs[0, 1])
    if 'rest_task' in results['comparisons']:
        df = results['comparisons']['rest_task']
        df = df[df['atlas'].str.contains('Cortical|Tian I')]
        if len(df) > 0:
            x = np.arange(len(df))
            width = 0.35
            ax2.bar(x - width/2, df['rest_mean'], width, label='Rest', 
                   color='#3498DB', alpha=0.85, edgecolor='black')
            ax2.bar(x + width/2, df['task_mean'], width, label='Task',
                   color='#E74C3C', alpha=0.85, edgecolor='black')
            ax2.set_xticks(x)
            ax2.set_xticklabels(df['atlas'].str.replace(' Cortical', '').str.replace(' Subcortical', ''))
            ax2.set_ylabel('Mean Error Rate', fontweight='bold')
            ax2.set_title('B) Rest vs Task', fontweight='bold', fontsize=12)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            # annotate error rates
            for i, v in enumerate(df['rest_mean']):
                ax2.text(i - width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
            for i, v in enumerate(df['task_mean']):
                ax2.text(i + width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
            ax2.set_ylim([0, 1.0])
    
    # Panel C: Network performance
    ax3 = fig.add_subplot(gs[0, 2])
    if 'N7_rest' in results['error_rates']:
        df = results['error_rates']['N7_rest'].sort_values('error_rate')
        colors = plt.cm.RdYlGn_r(df['error_rate'] / df['error_rate'].max())
        ax3.barh(range(len(df)), 1 - df['error_rate'], color=colors, 
                alpha=0.85, edgecolor='black')
        ax3.set_yticks(range(len(df)))
        ax3.set_yticklabels(df['network'], fontsize=9)
        ax3.set_xlabel('Accuracy', fontweight='bold')
        ax3.set_title('C) Network Performance (N7)', fontweight='bold', fontsize=12)
        ax3.invert_yaxis()
        ax3.set_xlim([0, 1.0])
        ax3.grid(axis='x', alpha=0.3)
        # annotate accuracy
        for i, v in enumerate(1 - df['error_rate']):
            ax3.text(v + 0.005, i, f"{v:.3f}", ha='left', va='center', fontsize=9, color='black', weight='bold')
    
    # Panel D: Cortical vs Subcortical
    ax4 = fig.add_subplot(gs[1, 0])
    if 'cortical_subcortical' in results['comparisons']:
        df = results['comparisons']['cortical_subcortical']
        x = np.arange(len(df))
        width = 0.35
        ax4.bar(x - width/2, df['cortical_mean'], width, label='Cortical',
               color='#2ECC71', alpha=0.85, edgecolor='black')
        ax4.bar(x + width/2, df['subcortical_mean'], width, label='Subcortical',
               color='#F39C12', alpha=0.85, edgecolor='black')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['condition'])
        ax4.set_ylabel('Mean Error Rate', fontweight='bold')
        ax4.set_title('D) Cortical vs Subcortical', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        # annotate error rates
        for i, v in enumerate(df['cortical_mean']):
            ax4.text(i - width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
        for i, v in enumerate(df['subcortical_mean']):
            ax4.text(i + width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
        ax4.set_ylim([0, 1.0])
    
    # Panel E: Task changes
    ax5 = fig.add_subplot(gs[1, 1])
    if 'N7_rest' in results['error_rates'] and 'N7_task' in results['error_rates']:
        rest = results['error_rates']['N7_rest']
        task = results['error_rates']['N7_task']
        merged = pd.merge(rest[['network', 'error_rate']], 
                         task[['network', 'error_rate']], 
                         on='network', suffixes=('_rest', '_task'))
        merged['change'] = merged['error_rate_task'] - merged['error_rate_rest']
        merged = merged.sort_values('change', ascending=False)
        
        colors = ['#E74C3C' if x > 0 else '#2ECC71' for x in merged['change']]
        ax5.barh(range(len(merged)), merged['change'], color=colors, 
                alpha=0.85, edgecolor='black')
        ax5.set_yticks(range(len(merged)))
        ax5.set_yticklabels(merged['network'], fontsize=9)
        ax5.set_xlabel('Error Change (Task - Rest)', fontweight='bold')
        ax5.set_title('E) Task-Induced Changes', fontweight='bold', fontsize=12)
        ax5.axvline(0, color='black', linewidth=2)
        ax5.invert_yaxis()
        ax5.grid(axis='x', alpha=0.3)
        # annotate accuracy
        for i, v in enumerate(merged['change']):
            ax5.text(v + 0.005, i, f"{v:.3f}", ha='left', va='center', fontsize=9, color='black', weight='bold')
    
    # Panel F: Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    acc = 1 - results['error_rates']['N7_rest']['error_rate'].mean() if 'N7_rest' in results['error_rates'] else 0
    n = len(results['error_rates']['N7_rest']) if 'N7_rest' in results['error_rates'] else 0
    improvement = acc / (1.0/n) if n > 0 else 0
    
    summary = f"""
F) KEY FINDINGS

1. PERFORMANCE
   • {acc:.1%} accuracy
   • {improvement:.0f}× better than chance
   
2. TASK EFFECTS
   • Networks reorganize
   • Error increases
   
3. HIERARCHY
   • Sensory: stable
   • Cognitive: flexible
   
4. ANATOMY
   • Subcortical harder
   • Size matters
   
5. INTEGRATION
   • Whole-brain coverage
   • Cortical + subcortical
    """
    
    ax6.text(0.05, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Brain Atlas Performance - Summary', fontsize=16, fontweight='bold', y=0.98)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")

def main():
    print("="*60)
    print("SUMMARY REPORT GENERATOR")
    print("="*60)
    
    # Check directories exist
    required = ['reports/tables/atlas_analysis',
                'reports/tables/atlas_comparison',
                'reports/tables/connectivity_analysis']
    
    missing = [d for d in required if not Path(d).exists()]
    if missing:
        print("\n❌ Missing directories:")
        for d in missing:
            print(f"   - {d}")
        print("\nRun previous scripts first!")
        return 1
    
    # Load data
    print("\nLoading results...")
    results = load_results()
    
    if not any(results.values()):
        print("❌ No results found!")
        return 1
    
    print(f"\n✓ Loaded successfully")
    print(f"  - Error rates: {len(results['error_rates'])} files")
    print(f"  - Comparisons: {len(results['comparisons'])} files")
    print(f"  - Connectivity: {len(results['connectivity'])} files")
    
    # Create output directory
    output_dir = Path('reports/summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    print("\n" + "="*60)
    print("CREATING OUTPUTS")
    print("="*60)
    
    print("\n1. Summary statistics...")
    stats = create_summary_stats(results)
    stats.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"   ✓ Saved {len(stats)} statistics")
    
    print("\n2. Key findings...")
    findings = create_key_findings(results)
    findings.to_csv(output_dir / 'key_findings.csv', index=False)
    print(f"   ✓ Saved {len(findings)} findings")
    
    print("\n3. Summary figure...")
    plot_summary(results, output_dir / 'master_summary.png')
        
    # Done
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"""

""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())#!/usr/bin/env python3
"""
Summary Report Generator - Simplified Version
Creates summary statistics, key findings, and visualization from analysis results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results():
    """Load all CSV results from previous analyses."""
    results = {'error_rates': {}, 'comparisons': {}, 'connectivity': {}}
    
    # Load atlas performance results
    atlas_dir = Path('reports/tables/atlas_analysis')
    if atlas_dir.exists():
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
            filepath = atlas_dir / filename
            if filepath.exists():
                results['error_rates'][key] = pd.read_csv(filepath)
                print(f"  ✓ {filename}")
    
    # Load comparison results
    comp_dir = Path('reports/tables/atlas_comparison')
    if comp_dir.exists():
        files = {
            'resolution': 'resolution_comparison.csv',
            'cortical_subcortical': 'cortical_vs_subcortical.csv',
            'rest_task': 'rest_vs_task_comparison.csv'
        }
        for key, filename in files.items():
            filepath = comp_dir / filename
            if filepath.exists():
                results['comparisons'][key] = pd.read_csv(filepath)
                print(f"  ✓ {filename}")
    
    # Load connectivity results
    conn_dir = Path('reports/tables/connectivity_analysis')
    if conn_dir.exists():
        files = {
            'rest': 'inter_network_connectivity_rest.csv',
            'task': 'inter_network_connectivity_task.csv',
            'change': 'inter_network_connectivity_change.csv',
            'top_changes': 'top_changed_connections.csv'
        }
        for key, filename in files.items():
            filepath = conn_dir / filename
            if filepath.exists():
                results['connectivity'][key] = pd.read_csv(filepath, index_col=0)
                print(f"  ✓ {filename}")
    
    return results


def create_summary_stats(results):
    """Generate summary statistics table."""
    rows = []
    
    # Performance metrics
    for key, df in results['error_rates'].items():
        parts = key.split('_')
        if len(parts) != 2:
            continue
        atlas, condition = parts
        
        mean_error = df['error_rate'].mean()
        std_error = df['error_rate'].std()
        accuracy = 1 - mean_error
        
        rows.append({
            'category': 'Performance',
            'metric': f'{atlas} {condition} - Accuracy',
            'value': accuracy,
            'formatted': f'{accuracy:.3f} ± {std_error:.3f}'
        })
    
    # Comparison stats
    if 'rest_task' in results['comparisons']:
        df = results['comparisons']['rest_task']
        for _, row in df.iterrows():
            rows.append({
                'category': 'Task Effect',
                'metric': f"{row['atlas']}",
                'value': row['mean_increase'],
                'formatted': f"+{row['mean_increase']:.3f} ({row['pct_increase']:+.1f}%)"
            })
    
    if 'cortical_subcortical' in results['comparisons']:
        df = results['comparisons']['cortical_subcortical']
        for _, row in df.iterrows():
            rows.append({
                'category': 'Region Type',
                'metric': f"Subcortical vs Cortical ({row['condition']})",
                'value': row['difference'],
                'formatted': f"{row['difference']:+.3f} (p={row['p_value']:.3f})"
            })
    
    return pd.DataFrame(rows)


def create_key_findings(results):
    """Generate key findings report."""
    findings = []
    
    # Finding 1: Overall performance
    if 'N7_rest' in results['error_rates']:
        df = results['error_rates']['N7_rest']
        accuracy = 1 - df['error_rate'].mean()
        n_networks = len(df)
        chance = 1.0 / n_networks
        
        findings.append({
            'finding': 'Classification Performance',
            'result': f"{accuracy:.1%} accuracy across {n_networks} networks ({accuracy/chance:.1f}× better than chance)",
            'interpretation': "Networks have unique connectivity fingerprints that can be reliably identified",
            'significance': 'High'
        })
    
    # Finding 2: Task effects
    if 'rest_task' in results['comparisons']:
        df = results['comparisons']['rest_task']
        cortical = df[df['atlas'].str.contains('Cortical')]
        if len(cortical) > 0:
            mean_increase = cortical['mean_increase'].mean()
            findings.append({
                'finding': 'Task-Induced Changes',
                'result': f"Error increased by {mean_increase:.1%} during task",
                'interpretation': "Task engagement reorganizes connectivity patterns, showing network flexibility",
                'significance': 'High'
            })
    
    # Finding 3: Network hierarchy
    if 'N7_rest' in results['error_rates']:
        df = results['error_rates']['N7_rest']
        if len(df) > 0:
            best = df.loc[df['error_rate'].idxmin()]
            worst = df.loc[df['error_rate'].idxmax()]
            findings.append({
                'finding': 'Network Variability',
                'result': f"Best: {best['network']} ({1-best['error_rate']:.1%}), Worst: {worst['network']} ({1-worst['error_rate']:.1%})",
                'interpretation': "Sensory networks more stable than cognitive networks, supporting hierarchical organization",
                'significance': 'Moderate'
            })
    
    # Finding 4: Cortical vs subcortical
    if 'cortical_subcortical' in results['comparisons']:
        df = results['comparisons']['cortical_subcortical']
        rest = df[df['condition'] == 'Rest']
        if len(rest) > 0:
            diff = rest.iloc[0]['difference']
            findings.append({
                'finding': 'Subcortical Difficulty',
                'result': f"Subcortical regions {abs(diff):.1%} harder to classify",
                'interpretation': "Smaller subcortical structures have less distinctive connectivity patterns",
                'significance': 'Moderate'
            })
    
    return pd.DataFrame(findings)


def plot_summary(results, output_path):
    """Create comprehensive summary figure (2×3 panels)."""
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    
    # Panel A: Performance across atlases
    ax1 = fig.add_subplot(gs[0, 0])
    data, labels, colors = [], [], []
    for atlas in ['N7', 'N17', 'TianI', 'TianII']:
        for cond in ['rest', 'task']:
            key = f'{atlas}_{cond}'
            if key in results['error_rates']:
                acc = 1 - results['error_rates'][key]['error_rate'].mean()
                data.append(acc)
                labels.append(f'{atlas}\n{cond}')
                colors.append('#3498DB' if cond == 'rest' else "#B3847F")
    
    if data:
        x = np.arange(len(data))
        ax1.bar(x, data, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('A) Classification Accuracy', fontweight='bold', fontsize=12)
        ax1.set_ylim([0, 1.0])
        ax1.grid(axis='y', alpha=0.3)
        # annotate accuracy
        for i, v in enumerate(data):
            ax1.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
       
    
    # Panel B: Rest vs Task
    ax2 = fig.add_subplot(gs[0, 1])
    if 'rest_task' in results['comparisons']:
        df = results['comparisons']['rest_task']
        df = df[df['atlas'].str.contains('Cortical|Tian I')]
        if len(df) > 0:
            x = np.arange(len(df))
            width = 0.35
            ax2.bar(x - width/2, df['rest_mean'], width, label='Rest', 
                   color='#3498DB', alpha=0.85, edgecolor='black')
            ax2.bar(x + width/2, df['task_mean'], width, label='Task',
                   color='#E74C3C', alpha=0.85, edgecolor='black')
            ax2.set_xticks(x)
            ax2.set_xticklabels(df['atlas'].str.replace(' Cortical', '').str.replace(' Subcortical', ''))
            ax2.set_ylabel('Mean Error Rate', fontweight='bold')
            ax2.set_title('B) Rest vs Task', fontweight='bold', fontsize=12)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            # annotate error rates
            for i, v in enumerate(df['rest_mean']):
                ax2.text(i - width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
            for i, v in enumerate(df['task_mean']):
                ax2.text(i + width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
            ax2.set_ylim([0, 1.0])
    
    # Panel C: Network performance
    ax3 = fig.add_subplot(gs[0, 2])
    if 'N7_rest' in results['error_rates']:
        df = results['error_rates']['N7_rest'].sort_values('error_rate')
        colors = plt.cm.RdYlGn_r(df['error_rate'] / df['error_rate'].max())
        ax3.barh(range(len(df)), 1 - df['error_rate'], color=colors, 
                alpha=0.85, edgecolor='black')
        ax3.set_yticks(range(len(df)))
        ax3.set_yticklabels(df['network'], fontsize=9)
        ax3.set_xlabel('Accuracy', fontweight='bold')
        ax3.set_title('C) Network Performance (N7)', fontweight='bold', fontsize=12)
        ax3.invert_yaxis()
        ax3.set_xlim([0, 1.0])
        ax3.grid(axis='x', alpha=0.3)
        # annotate accuracy
        for i, v in enumerate(1 - df['error_rate']):
            ax3.text(v + 0.005, i, f"{v:.3f}", ha='left', va='center', fontsize=9, color='black', weight='bold')
    
    # Panel D: Cortical vs Subcortical
    ax4 = fig.add_subplot(gs[1, 0])
    if 'cortical_subcortical' in results['comparisons']:
        df = results['comparisons']['cortical_subcortical']
        x = np.arange(len(df))
        width = 0.35
        ax4.bar(x - width/2, df['cortical_mean'], width, label='Cortical',
               color='#2ECC71', alpha=0.85, edgecolor='black')
        ax4.bar(x + width/2, df['subcortical_mean'], width, label='Subcortical',
               color='#F39C12', alpha=0.85, edgecolor='black')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['condition'])
        ax4.set_ylabel('Mean Error Rate', fontweight='bold')
        ax4.set_title('D) Cortical vs Subcortical', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        # annotate error rates
        for i, v in enumerate(df['cortical_mean']):
            ax4.text(i - width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
        for i, v in enumerate(df['subcortical_mean']):
            ax4.text(i + width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, color='black', weight='bold')
        ax4.set_ylim([0, 1.0])
    
    # Panel E: Task changes
    ax5 = fig.add_subplot(gs[1, 1])
    if 'N7_rest' in results['error_rates'] and 'N7_task' in results['error_rates']:
        rest = results['error_rates']['N7_rest']
        task = results['error_rates']['N7_task']
        merged = pd.merge(rest[['network', 'error_rate']], 
                         task[['network', 'error_rate']], 
                         on='network', suffixes=('_rest', '_task'))
        merged['change'] = merged['error_rate_task'] - merged['error_rate_rest']
        merged = merged.sort_values('change', ascending=False)
        
        colors = ['#E74C3C' if x > 0 else '#2ECC71' for x in merged['change']]
        ax5.barh(range(len(merged)), merged['change'], color=colors, 
                alpha=0.85, edgecolor='black')
        ax5.set_yticks(range(len(merged)))
        ax5.set_yticklabels(merged['network'], fontsize=9)
        ax5.set_xlabel('Error Change (Task - Rest)', fontweight='bold')
        ax5.set_title('E) Task-Induced Changes', fontweight='bold', fontsize=12)
        ax5.axvline(0, color='black', linewidth=2)
        ax5.invert_yaxis()
        ax5.grid(axis='x', alpha=0.3)
        # annotate accuracy
        for i, v in enumerate(merged['change']):
            ax5.text(v + 0.005, i, f"{v:.3f}", ha='left', va='center', fontsize=9, color='black', weight='bold')
    
    # Panel F: Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    acc = 1 - results['error_rates']['N7_rest']['error_rate'].mean() if 'N7_rest' in results['error_rates'] else 0
    n = len(results['error_rates']['N7_rest']) if 'N7_rest' in results['error_rates'] else 0
    improvement = acc / (1.0/n) if n > 0 else 0
    
    summary = f"""
F) KEY FINDINGS

1. PERFORMANCE
   • {acc:.1%} accuracy
   • {improvement:.0f}× better than chance
   
2. TASK EFFECTS
   • Networks reorganize
   • Error increases
   
3. HIERARCHY
   • Sensory: stable
   • Cognitive: flexible
   
4. ANATOMY
   • Subcortical harder
   • Size matters
   
5. INTEGRATION
   • Whole-brain coverage
   • Cortical + subcortical
    """
    
    ax6.text(0.05, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Brain Atlas Performance - Summary', fontsize=16, fontweight='bold', y=0.98)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_guide(findings, output_path):
    """Generate interpretation guide for thesis."""
    guide = f"""
THESIS INTERPRETATION GUIDE
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

KEY FINDINGS
============
"""
    
    for i, row in findings.iterrows():
        guide += f"""
{i+1}. {row['finding'].upper()}
   Result: {row['result']}
   Meaning: {row['interpretation']}
   Importance: {row['significance']}
"""
    
    guide += """

MAIN THEMES FOR DISCUSSION
==========================

1. CONNECTIVITY FINGERPRINTS
   - Networks have unique patterns
   - Can identify regions reliably
   - Supports individual differences research

2. TASK REORGANIZATION
   - Networks flexible during tasks
   - Connectivity changes with demands
   - Novel way to detect engagement

3. BRAIN HIERARCHY
   - Sensory networks stable
   - Cognitive networks flexible
   - Matches known organization

4. METHODOLOGY
   - Validates atlas choices
   - Practical comparison tool
   - Clinical applications possible


KEY PAPERS TO CITE
==================

Fingerprinting:
- Finn et al. (2015) - Connectome fingerprinting
- Gratton et al. (2018) - Network stability

Task Effects:
- Cole et al. (2014) - Multi-task connectivity
- Bassett et al. (2011) - Dynamic reconfiguration

Atlases:
- Schaefer et al. (2018) - Cortical parcellation
- Tian et al. (2020) - Subcortical atlas

Hierarchy:
- Mesulam (1998) - Brain organization
- Margulies et al. (2016) - Gradients

"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(guide)
    print(f"✓ Saved: {output_path}")


def main():
    print("="*60)
    print("SUMMARY REPORT GENERATOR")
    print("="*60)
    
    # Check directories exist
    required = ['reports/tables/atlas_analysis',
                'reports/tables/atlas_comparison',
                'reports/tables/connectivity_analysis']
    
    missing = [d for d in required if not Path(d).exists()]
    if missing:
        print("\n❌ Missing directories:")
        for d in missing:
            print(f"   - {d}")
        print("\nRun previous scripts first!")
        return 1
    
    # Load data
    print("\nLoading results...")
    results = load_results()
    
    if not any(results.values()):
        print("❌ No results found!")
        return 1
    
    print(f"\n✓ Loaded successfully")
    print(f"  - Error rates: {len(results['error_rates'])} files")
    print(f"  - Comparisons: {len(results['comparisons'])} files")
    print(f"  - Connectivity: {len(results['connectivity'])} files")
    
    # Create output directory
    output_dir = Path('reports/summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    print("\n" + "="*60)
    print("CREATING OUTPUTS")
    print("="*60)
    
    print("\n1. Summary statistics...")
    stats = create_summary_stats(results)
    stats.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"   ✓ Saved {len(stats)} statistics")
    
    print("\n2. Key findings...")
    findings = create_key_findings(results)
    findings.to_csv(output_dir / 'key_findings.csv', index=False)
    print(f"   ✓ Saved {len(findings)} findings")
    
    print("\n3. Summary figure...")
    plot_summary(results, output_dir / 'master_summary.png')
    
    print("\n4. Interpretation guide...")
    create_guide(findings, output_dir / 'interpretation_guide.txt')
    
    # Done
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"""
Output Files:
  • {output_dir}/summary_statistics.csv
  • {output_dir}/key_findings.csv
  • {output_dir}/master_summary.png
  • {output_dir}/interpretation_guide.txt

Next Steps:
  1. Review key findings
  2. Use figure in thesis/presentations
  3. Read interpretation guide
  4. Write discussion section
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())