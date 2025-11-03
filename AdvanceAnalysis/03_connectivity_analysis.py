#!/usr/bin/env python3
"""
Connectivity Analysis - Inter-Network Connectivity & Task Modulation
=====================================================================
Analyzes functional connectivity patterns with consistent network definitions:
1. Inter-network connectivity matrices (rest vs task)
2. Task-induced connectivity changes
3. Subcortical-cortical coupling
4. Identification of task-modulated connections

Usage:
    python 03_connectivity_analysis.py --config config.yaml

Outputs:
    - Inter-network connectivity matrices (N7 + Tian I)
    - Task modulation matrices (task - rest)
    - Top changed connections
    - Comprehensive visualization
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
from utils import load_config, set_random_seeds, print_section

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# =============================================================================
# NETWORK MAPPING (Consistent with refactored script)
# =============================================================================

def map_to_schaefer_n7(region_list):
    """Map regions to Schaefer N7 + Tian Scale I (consistent with refactored script)."""
    mapping = {}
    
    for region in region_list:
        name = region.lower()
        
        # Subcortical regions (Tian Scale I)
        if not region.startswith(('LH_', 'RH_')):
            if 'hip' in name:
                net = 'Hippocampus'
            elif 'amy' in name or 'amg' in name:
                net = 'Amygdala'
            # Thalamus - combine posterior and anterior
            elif 'tha-dp' in name or 'tha-vp' in name or 'tha_dp' in name or 'tha_vp' in name:
                net = 'Thalamus_post'
            elif 'tha-da' in name or 'tha-va' in name or 'tha_da' in name or 'tha_va' in name:
                net = 'Thalamus_ant'
            elif 'tha' in name or '_th' in name or name.startswith('th'):
                net = 'Thalamus'  # Generic thalamus if no subdivision
            elif 'nac' in name or 'accumb' in name:
                net = 'Accumbens'
            elif 'put' in name:
                net = 'Putamen'
            elif 'gp' in name or 'pallid' in name:
                net = 'Pallidum'
            elif 'cau' in name:
                net = 'Caudate'
            else:
                net = 'SubcorticalOther'
        
        # Cortical regions (Schaefer N7)
        else:
            if 'vis' in name:
                net = 'Visual'
            elif 'sommot' in name or 'senmot' in name or 'motor' in name:
                net = 'Somatomotor'
            elif 'dorsattn' in name or ('dorsal' in name and 'attn' in name):
                net = 'DorsalAttention'
            elif 'salventattn' in name or (('ventral' in name or 'salience' in name) and 'attn' in name):
                net = 'VentralAttention'
            elif 'limbic' in name or 'limb' in name:
                net = 'Limbic'
            elif 'cont' in name or 'frontoparietal' in name or 'control' in name:
                net = 'FrontoParietal'
            elif 'default' in name or 'dmn' in name:
                net = 'DefaultMode'
            else:
                net = 'CorticalOther'
        
        mapping[region] = net
    
    return mapping


def filter_out_other_networks(network_mapping):
    """Remove 'Other' networks for cleaner analysis."""
    exclude = {'CorticalOther', 'SubcorticalOther'}
    return {k: v for k, v in network_mapping.items() if v not in exclude}


# =============================================================================
# CONNECTIVITY MATRIX RECONSTRUCTION
# =============================================================================

def extract_regions_from_connections(connection_columns):
    """Extract unique regions from connection column names."""
    regions = set()
    for col in connection_columns:
        if '~' in col:
            region_a, region_b = col.split('~')
            regions.add(region_a)
            regions.add(region_b)
    
    region_list = sorted(list(regions))
    region_to_idx = {r: i for i, r in enumerate(region_list)}
    
    return region_list, region_to_idx


def reconstruct_connectivity_matrix(subject_row, connection_columns, region_to_idx, n_regions):
    """
    Reconstruct full connectivity matrix from flattened connection data.
    
    Args:
        subject_row: Single row of connectivity data (excludes subject_id)
        connection_columns: List of connection column names
        region_to_idx: Dict mapping region name to matrix index
        n_regions: Total number of regions
        
    Returns:
        Symmetric connectivity matrix (n_regions × n_regions)
    """
    matrix = np.zeros((n_regions, n_regions), dtype=float)
    
    for col, value in zip(connection_columns, subject_row):
        if '~' not in col:
            continue
        
        region_a, region_b = col.split('~')
        
        if region_a in region_to_idx and region_b in region_to_idx:
            idx_a = region_to_idx[region_a]
            idx_b = region_to_idx[region_b]
            matrix[idx_a, idx_b] = value
            matrix[idx_b, idx_a] = value  # Symmetric
    
    # Set diagonal to 1 (self-connectivity)
    np.fill_diagonal(matrix, 1.0)
    
    return matrix


# =============================================================================
# INTER-NETWORK CONNECTIVITY
# =============================================================================

def calculate_inter_network_connectivity(connectivity_matrix, region_list, network_mapping):
    """
    Calculate mean connectivity between all network pairs.
    
    Args:
        connectivity_matrix: n_regions × n_regions connectivity matrix
        region_list: List of region names
        network_mapping: Dict mapping region_name -> network_name
        
    Returns:
        DataFrame with inter-network connectivity (network × network)
    """
    # Get unique networks (sorted)
    networks = sorted(set(network_mapping.values()))
    n_networks = len(networks)
    
    # Initialize network connectivity matrix
    network_conn = np.zeros((n_networks, n_networks))
    
    # For each network pair, compute mean connectivity
    for i, net_i in enumerate(networks):
        for j, net_j in enumerate(networks):
            # Get region indices for each network
            regions_i = [k for k, r in enumerate(region_list) 
                        if r in network_mapping and network_mapping[r] == net_i]
            regions_j = [k for k, r in enumerate(region_list) 
                        if r in network_mapping and network_mapping[r] == net_j]
            
            if len(regions_i) > 0 and len(regions_j) > 0:
                # Get submatrix
                submatrix = connectivity_matrix[np.ix_(regions_i, regions_j)]
                
                if i == j:
                    # Within-network connectivity (exclude diagonal)
                    mask = ~np.eye(len(regions_i), dtype=bool)
                    if mask.sum() > 0:
                        network_conn[i, j] = submatrix[mask].mean()
                else:
                    # Between-network connectivity
                    network_conn[i, j] = submatrix.mean()
    
    return pd.DataFrame(network_conn, index=networks, columns=networks)


def compute_group_connectivity(df, connection_columns, region_list, network_mapping):
    """
    Compute mean connectivity across all subjects.
    
    Returns:
        inter_network: DataFrame with network × network connectivity
        group_connectivity: Array with region × region connectivity
    """
    region_to_idx = {r: i for i, r in enumerate(region_list)}
    n_regions = len(region_list)
    n_subjects = len(df)
    
    # Initialize group connectivity
    group_connectivity = np.zeros((n_regions, n_regions))
    
    # Average across subjects
    for idx in range(n_subjects):
        # Extract connectivity values (skip subject_id column)
        subject_values = df.iloc[idx, 1:].values
        
        # Reconstruct matrix
        matrix = reconstruct_connectivity_matrix(
            subject_values, connection_columns, region_to_idx, n_regions
        )
        
        group_connectivity += matrix
    
    # Average
    group_connectivity /= n_subjects
    
    # Compute inter-network connectivity
    inter_network = calculate_inter_network_connectivity(
        group_connectivity, region_list, network_mapping
    )
    
    return inter_network, group_connectivity


# =============================================================================
# TASK MODULATION ANALYSIS
# =============================================================================

def identify_top_changed_connections(rest_matrix, task_matrix, region_list, 
                                     network_mapping, top_k=100):
    """
    Identify connections with largest change from rest to task.
    
    Returns:
        DataFrame with top changed connections (sorted by absolute change)
    """
    n_regions = len(region_list)
    
    # Compute change matrix
    change_matrix = task_matrix - rest_matrix
    
    # Get upper triangle indices (avoid duplicates, exclude diagonal)
    triu_idx = np.triu_indices(n_regions, k=1)
    
    # Extract changes
    changes = []
    for i, j in zip(*triu_idx):
        # Skip if regions not in network mapping
        if region_list[i] not in network_mapping or region_list[j] not in network_mapping:
            continue
        
        change = change_matrix[i, j]
        rest_val = rest_matrix[i, j]
        task_val = task_matrix[i, j]
        
        changes.append({
            'region_i': region_list[i],
            'region_j': region_list[j],
            'network_i': network_mapping[region_list[i]],
            'network_j': network_mapping[region_list[j]],
            'rest_connectivity': rest_val,
            'task_connectivity': task_val,
            'change': change,
            'abs_change': abs(change),
            'pct_change': (change / abs(rest_val) * 100) if abs(rest_val) > 1e-6 else 0
        })
    
    df = pd.DataFrame(changes)
    
    # Sort by absolute change (descending)
    df = df.sort_values('abs_change', ascending=False).reset_index(drop=True)
    
    return df.head(top_k)


def categorize_network_pair_type(net_i, net_j):
    """Categorize connection as cortical-cortical, subcortical-subcortical, or cortical-subcortical."""
    cortical_nets = {'Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention', 
                     'Limbic', 'FrontoParietal', 'DefaultMode'}
    
    is_cortical_i = net_i in cortical_nets
    is_cortical_j = net_j in cortical_nets
    
    if is_cortical_i and is_cortical_j:
        return 'Cortical-Cortical'
    elif not is_cortical_i and not is_cortical_j:
        return 'Subcortical-Subcortical'
    else:
        return 'Cortical-Subcortical'


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_connectivity_analysis(rest_network, task_network, change_network, 
                               top_changes, output_path):
    """
    Create comprehensive connectivity analysis figure (2×3 layout).
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    
    # Determine color scale limits for consistency
    vmin = min(rest_network.values.min(), task_network.values.min())
    vmax = max(rest_network.values.max(), task_network.values.max())
    
    # Panel A: Resting-state inter-network connectivity
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(rest_network, annot=False, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Connectivity', 'shrink': 0.8}, 
                square=True, ax=ax1, linewidths=0.5, linecolor='gray')
    ax1.set_title('A) Inter-Network Connectivity (Rest)', 
                 fontweight='bold', fontsize=12, pad=10)
    ax1.set_xlabel('Network', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Network', fontweight='bold', fontsize=10)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    # Panel B: Task inter-network connectivity
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(task_network, annot=False, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Connectivity', 'shrink': 0.8}, 
                square=True, ax=ax2, linewidths=0.5, linecolor='gray')
    ax2.set_title('B) Inter-Network Connectivity (Task)', 
                 fontweight='bold', fontsize=12, pad=10)
    ax2.set_xlabel('Network', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Network', fontweight='bold', fontsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    # Panel C: Change matrix (Task - Rest)
    ax3 = fig.add_subplot(gs[0, 2])
    change_vmax = max(abs(change_network.values.min()), abs(change_network.values.max()))
    sns.heatmap(change_network, annot=False, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=-change_vmax, vmax=change_vmax,
                cbar_kws={'label': 'Connectivity Change', 'shrink': 0.8}, 
                square=True, ax=ax3, linewidths=0.5, linecolor='gray')
    ax3.set_title('C) Connectivity Change (Task - Rest)', 
                 fontweight='bold', fontsize=12, pad=10)
    ax3.set_xlabel('Network', fontweight='bold', fontsize=10)
    ax3.set_ylabel('Network', fontweight='bold', fontsize=10)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)
    
    # Panel D: Top increased connections
    ax4 = fig.add_subplot(gs[1, 0])
    top_increased = top_changes[top_changes['change'] > 0].head(15)
    if len(top_increased) > 0:
        labels = [f"{row['network_i'][:12]}-\n{row['network_j'][:12]}" 
                 for _, row in top_increased.iterrows()]
        colors = ['#E74C3C' if categorize_network_pair_type(row['network_i'], row['network_j']) == 'Cortical-Subcortical' 
                  else '#FF6B6B' for _, row in top_increased.iterrows()]
        
        bars = ax4.barh(range(len(top_increased)), top_increased['change'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_increased['change'])):
            ax4.text(val + 0.002, i, f'{val:.3f}', 
                    va='center', ha='left', fontsize=7, fontweight='bold')
        
        ax4.set_yticks(range(len(top_increased)))
        ax4.set_yticklabels(labels, fontsize=7)
        ax4.set_xlabel('Connectivity Increase', fontweight='bold', fontsize=10)
        ax4.set_title('D) Top 15 Increased Connections', 
                     fontweight='bold', fontsize=12, pad=10)
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Panel E: Top decreased connections
    ax5 = fig.add_subplot(gs[1, 1])
    top_decreased = top_changes[top_changes['change'] < 0].head(15)
    if len(top_decreased) > 0:
        labels = [f"{row['network_i'][:12]}-\n{row['network_j'][:12]}" 
                 for _, row in top_decreased.iterrows()]
        colors = ['#3498DB' if categorize_network_pair_type(row['network_i'], row['network_j']) == 'Cortical-Subcortical' 
                  else '#5DADE2' for _, row in top_decreased.iterrows()]
        
        bars = ax5.barh(range(len(top_decreased)), top_decreased['change'],
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_decreased['change'])):
            ax5.text(val - 0.002, i, f'{val:.3f}', 
                    va='center', ha='right', fontsize=7, fontweight='bold')
        
        ax5.set_yticks(range(len(top_decreased)))
        ax5.set_yticklabels(labels, fontsize=7)
        ax5.set_xlabel('Connectivity Decrease', fontweight='bold', fontsize=10)
        ax5.set_title('E) Top 15 Decreased Connections', 
                     fontweight='bold', fontsize=12, pad=10)
        ax5.invert_yaxis()
        ax5.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Panel F: Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate statistics
    top_changes['pair_type'] = top_changes.apply(
        lambda row: categorize_network_pair_type(row['network_i'], row['network_j']), 
        axis=1
    )
    
    n_increased = (top_changes['change'] > 0.01).sum()
    n_decreased = (top_changes['change'] < -0.01).sum()
    max_increase = top_changes['change'].max()
    max_decrease = top_changes['change'].min()
    
    # Count by pair type
    pair_type_counts = top_changes.head(50)['pair_type'].value_counts()
    
    stats_text = f"""
F) CONNECTIVITY CHANGE SUMMARY
={'='*32}

Total Connections: {len(top_changes)}

Significant Changes (|Δ| > 0.01):
  Increased: {n_increased:4d}
  Decreased: {n_decreased:4d}

Maximum Changes:
  Largest Increase:  {max_increase:7.4f}
  Largest Decrease:  {max_decrease:7.4f}

Mean Abs Change: {top_changes['abs_change'].mean():.4f}

Top 50 Connection Types:
  Cort-Cort:  {pair_type_counts.get('Cortical-Cortical', 0):2d}
  Cort-Subc:  {pair_type_counts.get('Cortical-Subcortical', 0):2d}
  Subc-Subc:  {pair_type_counts.get('Subcortical-Subcortical', 0):2d}

Most Changed Network Pair:
  {top_changes.iloc[0]['network_i'][:15]} ↔
  {top_changes.iloc[0]['network_j'][:15]}
  Δ = {top_changes.iloc[0]['change']:.4f}
    """
    
    ax6.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
             verticalalignment='center', fontweight='normal',
             bbox=dict(boxstyle='round', facecolor='lightblue', 
                      alpha=0.3, edgecolor='black', linewidth=1.5))
    
    plt.suptitle('Functional Connectivity Analysis: Rest vs Task', 
                fontsize=16, fontweight='bold', y=0.98)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved connectivity analysis: {output_path}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Connectivity Analysis')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    
    # Setup
    print_section("CONNECTIVITY ANALYSIS")
    config = load_config(args.config)
    set_random_seeds(config.get('random_seed', 42))
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print_section("Step 1: Load Data")
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']
    
    if args.sample:
        piop2_file = "data/sample/sample_piop2_small.csv"
        piop1_file = "data/sample/sample_piop1_small.csv"
    
    df_rest = load_connectivity_data(piop2_file)
    df_task = load_connectivity_data(piop1_file)
    
    connection_columns = extract_connection_columns(df_rest)
    print(f"✓ Loaded {len(df_rest)} rest subjects, {len(df_task)} task subjects")
    print(f"✓ Found {len(connection_columns)} connections")
    
    # =========================================================================
    # Step 2: Extract Regions and Map to Networks
    # =========================================================================
    print_section("Step 2: Extract Regions & Map to Networks")
    
    region_list, region_to_idx = extract_regions_from_connections(connection_columns)
    n_regions = len(region_list)
    print(f"✓ Extracted {n_regions} regions")
    
    # Map regions to networks (N7 + Tian Scale I)
    network_mapping_full = map_to_schaefer_n7(region_list)
    
    # Filter out 'Other' networks for cleaner analysis
    network_mapping = filter_out_other_networks(network_mapping_full)
    
    networks = sorted(set(network_mapping.values()))
    print(f"✓ Mapped to {len(networks)} networks (excluding 'Other' categories)")
    print(f"  Networks: {', '.join(networks)}")
    
    # =========================================================================
    # Step 3: Compute Inter-Network Connectivity
    # =========================================================================
    print_section("Step 3: Compute Inter-Network Connectivity")
    
    print("Computing resting-state connectivity...")
    rest_network, rest_matrix = compute_group_connectivity(
        df_rest, connection_columns, region_list, network_mapping
    )
    print(f"  ✓ Rest connectivity: {rest_network.shape[0]}×{rest_network.shape[1]} matrix")
    
    print("Computing task connectivity...")
    task_network, task_matrix = compute_group_connectivity(
        df_task, connection_columns, region_list, network_mapping
    )
    print(f"  ✓ Task connectivity: {task_network.shape[0]}×{task_network.shape[1]} matrix")
    
    print("✓ Computed inter-network connectivity matrices")
    
    # =========================================================================
    # Step 4: Identify Task-Modulated Connections
    # =========================================================================
    print_section("Step 4: Identify Task-Modulated Connections")
    
    change_network = task_network - rest_network
    
    # Identify top changed connections
    print("Identifying top changed connections...")
    top_changes = identify_top_changed_connections(
        rest_matrix, task_matrix, region_list, network_mapping, top_k=100
    )
    
    print(f"✓ Identified top 100 changed connections")
    print(f"  - Increased: {(top_changes['change'] > 0).sum()}")
    print(f"  - Decreased: {(top_changes['change'] < 0).sum()}")
    
    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    print_section("Step 5: Save Results")
    
    output_tables = Path('reports/tables/connectivity_analysis')
    output_figures = Path('reports/figures/connectivity_analysis')
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    
    # Save matrices
    rest_network.to_csv(output_tables / 'inter_network_connectivity_rest.csv')
    task_network.to_csv(output_tables / 'inter_network_connectivity_task.csv')
    change_network.to_csv(output_tables / 'inter_network_connectivity_change.csv')
    print("✓ Saved inter-network connectivity matrices")
    
    # Save top changes
    top_changes.to_csv(output_tables / 'top_changed_connections.csv', index=False)
    print("✓ Saved top changed connections")
    
    # =========================================================================
    # Step 6: Generate Visualization
    # =========================================================================
    print_section("Step 6: Generate Visualization")
    
    plot_connectivity_analysis(
        rest_network, task_network, change_network, top_changes,
        output_figures / 'connectivity_analysis.png'
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("ANALYSIS COMPLETE!")
    
    print(f"""
Generated Files:
================

Tables (CSV):
  • {output_tables}/inter_network_connectivity_rest.csv
  • {output_tables}/inter_network_connectivity_task.csv
  • {output_tables}/inter_network_connectivity_change.csv
  • {output_tables}/top_changed_connections.csv

Figures (PNG):
  • {output_figures}/connectivity_analysis.png
    - 2×3 grid with connectivity matrices and top changes

Key Findings:
=============
""")
    
    # Print top 10 changed connections
    print("\nTop 10 Changed Connections:")
    print("-" * 80)
    top_10 = top_changes[['network_i', 'network_j', 'change', 'pct_change']].head(10)
    for idx, row in top_10.iterrows():
        direction = "↑" if row['change'] > 0 else "↓"
        print(f"  {idx+1:2d}. {row['network_i']:20s} ↔ {row['network_j']:20s} "
              f"{direction} {abs(row['change']):7.4f} ({row['pct_change']:+7.1f}%)")
    
    # Print mean connectivity changes by network
    print("\n\nMean Connectivity Change by Network Pair:")
    print("-" * 80)
    print(change_network.round(4).to_string())
    
    # Statistics by connection type
    print("\n\nConnection Type Statistics:")
    print("-" * 80)
    top_changes['pair_type'] = top_changes.apply(
        lambda row: categorize_network_pair_type(row['network_i'], row['network_j']), 
        axis=1
    )
    
    for pair_type in ['Cortical-Cortical', 'Cortical-Subcortical', 'Subcortical-Subcortical']:
        subset = top_changes[top_changes['pair_type'] == pair_type]
        if len(subset) > 0:
            print(f"\n{pair_type}:")
            print(f"  Count: {len(subset)}")
            print(f"  Mean change: {subset['change'].mean():7.4f}")
            print(f"  Mean |change|: {subset['abs_change'].mean():7.4f}")
    
    print("\n✅ Connectivity analysis complete!\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())