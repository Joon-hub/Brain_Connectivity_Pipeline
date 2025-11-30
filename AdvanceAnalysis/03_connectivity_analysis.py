#!/usr/bin/env python3
"""
Connectivity Analysis 
======================
Generates THREE separate analyses:
1. N7 Cortical (Schaefer 7-network)
2. Tian Scale I Subcortical
3. Combined N7 + Tian I

"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data import load_connectivity_data, extract_connection_columns
from features import extract_regions, reconstruct_matrices_from_dataframe
from utils import load_config, set_random_seeds, print_section

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# =============================================================================
# NETWORK MAPPING
# =============================================================================
# Note: Unlike 01_atlas_performance_analysis.py, we do NOT filter out
# unmapped regions here. This is intentional - connectivity analysis 
# includes all regions to capture full brain connectivity patterns.

def map_to_schaefer_n7_only(region_list):
    """Map cortical regions to Schaefer N7 networks only."""
    cortical_map = {
        'vis': 'Visual',
        'sommot': 'Somatomotor', 'senmot': 'Somatomotor',
        'dorsattn': 'DorsalAttention',
        'salventattn': 'VentralAttention',
        'limbic': 'Limbic',
        'cont': 'FrontoParietal',
        'default': 'DefaultMode'
    }
    
    mapping = {}
    for region in region_list:
        # Only map cortical regions (LH_ or RH_ prefix)
        if region.startswith(('LH_', 'RH_')):
            name = region.lower()
            net = next((v for k, v in cortical_map.items() if k in name), None)
            if net:
                mapping[region] = net
    
    return mapping


def map_to_tian_scale_i_only(region_list):
    """Map subcortical regions to Tian Scale I networks only."""
    subcortical_map = {
        'hip': 'Hippocampus',
        'amy': 'Amygdala',
        'tha-dp': 'Thal_post', 
        'tha_dp': 'Thal_post',  # FIXED: Added underscore version
        'tha-vp': 'Thal_post', 
        'tha_vp': 'Thal_post',  # FIXED: Added underscore version
        'tha-da': 'Thal_ant', 
        'tha_da': 'Thal_ant',   # Already present
        'tha-va': 'Thal_ant', 
        'tha_va': 'Thal_ant',   # FIXED: Added underscore version
        'nac': 'Accumbens',
        'put': 'Putamen',
        'gp': 'Pallidum',
        'cau': 'Caudate'
    }
    
    mapping = {}
    for region in region_list:
        # Only map subcortical regions (no LH_ or RH_ prefix)
        if not region.startswith(('LH_', 'RH_')):
            name = region.lower()
            net = next((v for k, v in subcortical_map.items() if k in name), None)
            if net:
                mapping[region] = net
    
    return mapping


def map_to_combined_networks(region_list):
    return {**map_to_schaefer_n7_only(region_list), **map_to_tian_scale_i_only(region_list)}


# =============================================================================
# CONNECTIVITY ANALYSIS
# =============================================================================
def calculate_inter_network_connectivity(connectivity_matrix, region_list, network_mapping):
    """Calculate mean connectivity between networks."""
    networks = sorted({v for v in network_mapping.values() if v})
    n_net = len(networks)
    net_conn = np.zeros((n_net, n_net))

    for i, net_i in enumerate(networks):
        for j, net_j in enumerate(networks):
            idx_i = [k for k, r in enumerate(region_list) if r in network_mapping and network_mapping[r] == net_i]
            idx_j = [k for k, r in enumerate(region_list) if r in network_mapping and network_mapping[r] == net_j]
            if not idx_i or not idx_j:
                continue
            submat = connectivity_matrix[np.ix_(idx_i, idx_j)]
            if i == j:
                mask = ~np.eye(len(idx_i), dtype=bool)
                net_conn[i, j] = submat[mask].mean() if mask.sum() > 0 else np.nan
            else:
                net_conn[i, j] = submat.mean()
    return pd.DataFrame(net_conn, index=networks, columns=networks)


def compute_group_connectivity(df, connection_columns, region_list, region_to_idx, network_mapping):
    """Compute group-average connectivity matrices."""
    n_regions = len(region_list)
    n_subjects = len(df)
    group_mat = np.zeros((n_regions, n_regions))

    for idx in range(n_subjects):
        values = df.iloc[idx][connection_columns].values.astype(float)
        mat = reconstruct_matrices_from_dataframe(values, connection_columns, region_to_idx, n_regions)
        group_mat += mat

    group_mat /= n_subjects
    inter_net = calculate_inter_network_connectivity(group_mat, region_list, network_mapping)
    return inter_net, group_mat


# =============================================================================
# TASK MODULATION
# =============================================================================
def identify_all_changed_connections(rest_matrix, task_matrix, region_list, network_mapping):
    """Identify all connectivity changes between rest and task."""
    n = len(region_list)
    change_matrix = task_matrix - rest_matrix
    triu = np.triu_indices(n, k=1)
    changes = []

    for i, j in zip(*triu):
        r1, r2 = region_list[i], region_list[j]
        if r1 not in network_mapping or r2 not in network_mapping:
            continue
        net1, net2 = network_mapping[r1], network_mapping[r2]
        delta = change_matrix[i, j]
        changes.append({
            'region_i': r1, 'region_j': r2,
            'network_i': net1, 'network_j': net2,
            'rest': rest_matrix[i, j], 'task': task_matrix[i, j],
            'change': delta, 'abs_change': abs(delta)
        })

    df = pd.DataFrame(changes)
    return df.sort_values('abs_change', ascending=False).reset_index(drop=True)


def categorize_network_pair_type(net_i, net_j, is_cortical_analysis=False):
    """Categorize connection type (cortical-cortical, subcortical-subcortical, or mixed)."""
    if is_cortical_analysis:
        return 'Cortical-Cortical'
    cortical_nets = {'Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention',
                     'Limbic', 'FrontoParietal', 'DefaultMode'}
    c1 = net_i in cortical_nets
    c2 = net_j in cortical_nets
    if c1 and c2:
        return 'Cortical-Cortical'
    elif not c1 and not c2:
        return 'Subcortical-Subcortical'
    else:
        return 'Cortical-Subcortical'


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_connectivity_analysis(rest_net, task_net, change_net, all_changes_df,
                               output_path, title_prefix='', is_cortical=False):
    """Create comprehensive connectivity analysis figure."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    vmin = min(rest_net.min().min(), task_net.min().min())
    vmax = max(rest_net.max().max(), task_net.max().max())
    change_vmax = max(abs(change_net.min().min()), abs(change_net.max().max()))

    # A: Rest
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(rest_net, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Connectivity'},
                square=True, ax=ax1, linewidths=0.5, annot_kws={'size': 7})
    ax1.set_title('A) Inter-Network Connectivity (Rest)', fontweight='bold', pad=10)

    # B: Task
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(task_net, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Connectivity'},
                square=True, ax=ax2, linewidths=0.5, annot_kws={'size': 7})
    ax2.set_title('B) Inter-Network Connectivity (Task)', fontweight='bold', pad=10)

    # C: Change
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(change_net, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-change_vmax, vmax=change_vmax,
                cbar_kws={'label': 'Connectivity Change'}, square=True,
                ax=ax3, linewidths=0.5, annot_kws={'size': 7})
    ax3.set_title('C) Connectivity Change (Task - Rest)', fontweight='bold', pad=10)

    # D: Top 15 Increased
    ax4 = fig.add_subplot(gs[1, 0])
    top_inc = all_changes_df[all_changes_df['change'] > 0].head(15)
    if len(top_inc) > 0:
        labels = [f"{r['network_i'][:10]}\n{r['network_j'][:10]}" for _, r in top_inc.iterrows()]
        colors = ['#E74C3C' if categorize_network_pair_type(r['network_i'], r['network_j'], is_cortical) == 'Cortical-Subcortical'
                  else '#FF6B6B' for _, r in top_inc.iterrows()]
        bars = ax4.barh(range(len(top_inc)), top_inc['change'], color=colors, edgecolor='black')
        for i, val in enumerate(top_inc['change']):
            ax4.text(val + 0.005, i, f'{val:.3f}', va='center', ha='left', fontweight='bold', fontsize=7)
        ax4.set_yticks(range(len(top_inc)))
        ax4.set_yticklabels(labels, fontsize=7)
        ax4.set_xlabel('Connectivity Increase')
        ax4.set_title('D) Top 15 Increased Connections', fontweight='bold', pad=10)
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3, linestyle='--')

    # E: Top 15 Decreased
    ax5 = fig.add_subplot(gs[1, 1])
    top_dec = all_changes_df[all_changes_df['change'] < 0].head(15)
    if len(top_dec) > 0:
        labels = [f"{r['network_i'][:10]}\n{r['network_j'][:10]}" for _, r in top_dec.iterrows()]
        colors = ['#3498DB' if categorize_network_pair_type(r['network_i'], r['network_j'], is_cortical) == 'Cortical-Subcortical'
                  else '#5DADE2' for _, r in top_dec.iterrows()]
        bars = ax5.barh(range(len(top_dec)), top_dec['change'], color=colors, edgecolor='black')
        for i, val in enumerate(top_dec['change']):
            ax5.text(val - 0.005, i, f'{val:.3f}', va='center', ha='right', fontweight='bold', fontsize=7)
        ax5.set_yticks(range(len(top_dec)))
        ax5.set_yticklabels(labels, fontsize=7)
        ax5.set_xlabel('Connectivity Decrease')
        ax5.set_title('E) Top 15 Decreased Connections', fontweight='bold', pad=10)
        ax5.invert_yaxis()
        ax5.grid(axis='x', alpha=0.3, linestyle='--')

    # F: Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    total_conn = len(all_changes_df)
    n_inc = (all_changes_df['change'] > 0.01).sum()
    n_dec = (all_changes_df['change'] < -0.01).sum()
    max_inc = all_changes_df['change'].max()
    max_dec = all_changes_df['change'].min()
    mean_abs = all_changes_df['abs_change'].mean()

    if not is_cortical:
        all_changes_df['pair_type'] = all_changes_df.apply(
            lambda row: categorize_network_pair_type(row['network_i'], row['network_j'], is_cortical), axis=1)
        top50_counts = all_changes_df.head(50)['pair_type'].value_counts()
    else:
        top50_counts = pd.Series({'Cortical-Cortical': 50})

    most_changed = all_changes_df.iloc[0]
    most_pair = f"{most_changed['network_i']} ↔ {most_changed['network_j']}"

    stats_text = f"""
F) CONNECTIVITY CHANGE SUMMARY
{'='*36}
Total Unique Connections: {total_conn:,}
Changes with |Δ| > 0.01:
  Increased: {n_inc:6,d}
  Decreased: {n_dec:6,d}
Maximum Changes:
  Largest Increase: {max_inc:+.4f}
  Largest Decrease: {max_dec:+.4f}
Mean |Δ|: {mean_abs:.4f}
Top 50 Most Changed:
  Cort-Cort: {top50_counts.get('Cortical-Cortical', 0):3d}
  Cort-Subc: {top50_counts.get('Cortical-Subcortical', 0):3d}
  Subc-Subc: {top50_counts.get('Subcortical-Subcortical', 0):3d}
Most Changed Pair:
  {most_pair}
    Δ = {most_changed['change']:+.4f}
    """

    ax6.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(f'{title_prefix}Functional Connectivity Analysis: Rest vs Task',
                 fontsize=16, fontweight='bold', y=0.98)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()

    print_section("TRIPLE CONNECTIVITY ANALYSIS: N7 | TIAN I | COMBINED")
    config = load_config(args.config)
    set_random_seeds(42)

    # Load data
    piop2_file = config['data']['piop2_file'] if not args.sample else "data/sample/sample_piop2_small.csv"
    piop1_file = config['data']['piop1_file'] if not args.sample else "data/sample/sample_piop1_small.csv"

    df_rest = load_connectivity_data(piop2_file)
    df_task = load_connectivity_data(piop1_file)
    connection_columns = extract_connection_columns(df_rest)

    print(f"Loaded: {len(df_rest)} rest, {len(df_task)} task subjects")
    print(f"Found {len(connection_columns):,} connections")

    # Use existing extract_regions function
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    print(f"Extracted {n_regions} regions")

    out_tables = Path('reports/tables/connectivity_analysis')
    out_figures = Path('reports/figures/connectivity_analysis')
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    # ===================================
    # 1. N7 CORTICAL ONLY
    # ===================================
    print_section("1. N7 CORTICAL ANALYSIS")
    n7_mapping = map_to_schaefer_n7_only(region_list)
    if not n7_mapping:
        print("No cortical regions mapped. Skipping N7.")
    else:
        n7_rest_net, n7_rest_mat = compute_group_connectivity(
            df_rest, connection_columns, region_list, region_to_idx, n7_mapping)
        n7_task_net, n7_task_mat = compute_group_connectivity(
            df_task, connection_columns, region_list, region_to_idx, n7_mapping)
        n7_change_net = n7_task_net - n7_rest_net
        n7_changes = identify_all_changed_connections(n7_rest_mat, n7_task_mat, region_list, n7_mapping)

        n7_rest_net.to_csv(out_tables / 'n7_rest.csv')
        n7_task_net.to_csv(out_tables / 'n7_task.csv')
        n7_change_net.to_csv(out_tables / 'n7_change.csv')
        n7_changes.to_csv(out_tables / 'n7_all_changes.csv', index=False)

        plot_connectivity_analysis(
            n7_rest_net, n7_task_net, n7_change_net, n7_changes,
            out_figures / 'connectivity_analysis_n7_cortical.png',
            title_prefix='N7 Cortical - ', is_cortical=True
        )

    # ===================================
    # 2. TIAN I SUBCORTICAL ONLY
    # ===================================
    print_section("2. TIAN SCALE I SUBCORTICAL ANALYSIS")
    tian_mapping = map_to_tian_scale_i_only(region_list)
    if not tian_mapping:
        print("No subcortical regions mapped. Skipping Tian I.")
    else:
        tian_rest_net, tian_rest_mat = compute_group_connectivity(
            df_rest, connection_columns, region_list, region_to_idx, tian_mapping)
        tian_task_net, tian_task_mat = compute_group_connectivity(
            df_task, connection_columns, region_list, region_to_idx, tian_mapping)
        tian_change_net = tian_task_net - tian_rest_net
        tian_changes = identify_all_changed_connections(tian_rest_mat, tian_task_mat, region_list, tian_mapping)

        tian_rest_net.to_csv(out_tables / 'tian_rest.csv')
        tian_task_net.to_csv(out_tables / 'tian_task.csv')
        tian_change_net.to_csv(out_tables / 'tian_change.csv')
        tian_changes.to_csv(out_tables / 'tian_all_changes.csv', index=False)

        plot_connectivity_analysis(
            tian_rest_net, tian_task_net, tian_change_net, tian_changes,
            out_figures / 'connectivity_analysis_tian_subcortical.png',
            title_prefix='Tian Scale I Subcortical - ', is_cortical=False
        )

    # ===================================
    # 3. COMBINED
    # ===================================
    print_section("3. COMBINED N7 + TIAN I ANALYSIS")
    combined_mapping = map_to_combined_networks(region_list)
    combined_rest_net, combined_rest_mat = compute_group_connectivity(
        df_rest, connection_columns, region_list, region_to_idx, combined_mapping)
    combined_task_net, combined_task_mat = compute_group_connectivity(
        df_task, connection_columns, region_list, region_to_idx, combined_mapping)
    combined_change_net = combined_task_net - combined_rest_net
    combined_changes = identify_all_changed_connections(combined_rest_mat, combined_task_mat, region_list, combined_mapping)

    combined_rest_net.to_csv(out_tables / 'combined_rest.csv')
    combined_task_net.to_csv(out_tables / 'combined_task.csv')
    combined_change_net.to_csv(out_tables / 'combined_change.csv')
    combined_changes.to_csv(out_tables / 'combined_all_changes.csv', index=False)

    plot_connectivity_analysis(
        combined_rest_net, combined_task_net, combined_change_net, combined_changes,
        out_figures / 'connectivity_analysis_combined.png',
        title_prefix='Combined N7 + Tian I - ', is_cortical=False
    )

    print_section("TRIPLE ANALYSIS COMPLETE!")
    print("Generated:")
    print("  • connectivity_analysis_n7_cortical.png")
    print("  • connectivity_analysis_tian_subcortical.png")
    print("  • connectivity_analysis_combined.png")
    print("  • All CSV tables in reports/tables/connectivity_analysis/")

if __name__ == '__main__':
    sys.exit(main())