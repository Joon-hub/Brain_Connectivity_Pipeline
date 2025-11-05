#!/usr/bin/env python3
"""
Connectivity Analysis - Inter-Network Connectivity & Task Modulation
=====================================================================
Analyzes functional connectivity patterns with separate analyses for:
1. N7 (Schaefer) cortical networks - 7 networks
2. Tian Scale I subcortical networks - 8 networks
3. Combined N7 + Tian I analysis
4. Task-induced connectivity changes
5. Cortical-subcortical coupling

Usage:
    python 03_connectivity_analysis_separated.py --config config.yaml

Outputs:
    - Separate connectivity matrices for N7 and Tian I
    - Combined inter-network connectivity matrices
    - Task modulation matrices (task - rest)
    - Top changed connections per analysis
    - Comprehensive visualizations (3 separate figures)
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
# NETWORK MAPPING (Separate N7 and Tian I)
# =============================================================================

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
        'tha-dp': 'Thal_post', 'tha-vp': 'Thal_post', 'tha_dp': 'Thal_post', 'tha_vp': 'Thal_post',
        'tha-da': 'Thal_ant', 'tha-va': 'Thal_ant', 'tha_da': 'Thal_ant', 'tha_va': 'Thal_ant',
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
    """Map regions to combined N7 + Tian Scale I."""
    cortical_mapping = map_to_schaefer_n7_only(region_list)
    subcortical_mapping = map_to_tian_scale_i_only(region_list)
    
    # Combine mappings
    combined = {**cortical_mapping, **subcortical_mapping}
    return combined


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
                                     network_mapping, top_k=100, analysis_name=''):
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
                               top_changes, output_path, title_prefix=''):
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
    sns.heatmap(rest_network, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Connectivity', 'shrink': 0.8}, 
                square=True, ax=ax1, linewidths=0.5, linecolor='gray',
                annot_kws={'size': 7})
    ax1.set_title('A) Inter-Network Connectivity (Rest)', 
                 fontweight='bold', fontsize=12, pad=10)
    ax1.set_xlabel('Network', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Network', fontweight='bold', fontsize=10)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    # Panel B: Task inter-network connectivity
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(task_network, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Connectivity', 'shrink': 0.8}, 
                square=True, ax=ax2, linewidths=0.5, linecolor='gray',
                annot_kws={'size': 7})
    ax2.set_title('B) Inter-Network Connectivity (Task)', 
                 fontweight='bold', fontsize=12, pad=10)
    ax2.set_xlabel('Network', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Network', fontweight='bold', fontsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    # Panel C: Change matrix (Task - Rest)
    ax3 = fig.add_subplot(gs[0, 2])
    change_vmax = max(abs(change_network.values.min()), abs(change_network.values.max()))
    sns.heatmap(change_network, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=-change_vmax, vmax=change_vmax,
                cbar_kws={'label': 'Connectivity Change', 'shrink': 0.8}, 
                square=True, ax=ax3, linewidths=0.5, linecolor='gray',
                annot_kws={'size': 7})
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
            ax4.text(val + val*0.02, i, f'{val:.3f}', 
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
            ax5.text(val - abs(val)*0.02, i, f'{val:.3f}', 
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
{'='*32}

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
    
    plt.suptitle(f'{title_prefix}Functional Connectivity Analysis: Rest vs Task', 
                fontsize=16, fontweight='bold', y=0.98)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved connectivity analysis: {output_path}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Connectivity Analysis (Separated N7 and Tian I)')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    
    # Setup
    print_section("CONNECTIVITY ANALYSIS - SEPARATED N7 & TIAN I")
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
    
    # Map regions to different network schemes
    n7_mapping = map_to_schaefer_n7_only(region_list)
    tian_mapping = map_to_tian_scale_i_only(region_list)
    combined_mapping = map_to_combined_networks(region_list)
    
    n7_networks = sorted(set(n7_mapping.values()))
    tian_networks = sorted(set(tian_mapping.values()))
    combined_networks = sorted(set(combined_mapping.values()))
    
    print(f"\n✓ Network Mappings:")
    print(f"  N7 (Cortical):     {len(n7_networks)} networks, {len(n7_mapping)} regions")
    print(f"    Networks: {', '.join(n7_networks)}")
    print(f"  Tian I (Subcort):  {len(tian_networks)} networks, {len(tian_mapping)} regions")
    print(f"    Networks: {', '.join(tian_networks)}")
    print(f"  Combined:          {len(combined_networks)} networks, {len(combined_mapping)} regions")
    
    # =========================================================================
    # Step 3: N7 Cortical Analysis
    # =========================================================================
    print_section("Step 3: N7 Cortical Network Analysis")
    
    print("Computing N7 resting-state connectivity...")
    n7_rest_network, n7_rest_matrix = compute_group_connectivity(
        df_rest, connection_columns, region_list, n7_mapping
    )
    print(f"  ✓ N7 Rest: {n7_rest_network.shape[0]}×{n7_rest_network.shape[1]} matrix")
    
    print("Computing N7 task connectivity...")
    n7_task_network, n7_task_matrix = compute_group_connectivity(
        df_task, connection_columns, region_list, n7_mapping
    )
    print(f"  ✓ N7 Task: {n7_task_network.shape[0]}×{n7_task_network.shape[1]} matrix")
    
    n7_change_network = n7_task_network - n7_rest_network
    
    print("Identifying N7 top changed connections...")
    n7_top_changes = identify_top_changed_connections(
        n7_rest_matrix, n7_task_matrix, region_list, n7_mapping, top_k=100, analysis_name='N7'
    )
    print(f"✓ N7 top 100 changed connections identified")
    print(f"  - Increased: {(n7_top_changes['change'] > 0).sum()}")
    print(f"  - Decreased: {(n7_top_changes['change'] < 0).sum()}")
    
    # =========================================================================
    # Step 4: Tian I Subcortical Analysis
    # =========================================================================
    print_section("Step 4: Tian Scale I Subcortical Network Analysis")
    
    print("Computing Tian I resting-state connectivity...")
    tian_rest_network, tian_rest_matrix = compute_group_connectivity(
        df_rest, connection_columns, region_list, tian_mapping
    )
    print(f"  ✓ Tian I Rest: {tian_rest_network.shape[0]}×{tian_rest_network.shape[1]} matrix")
    
    print("Computing Tian I task connectivity...")
    tian_task_network, tian_task_matrix = compute_group_connectivity(
        df_task, connection_columns, region_list, tian_mapping
    )
    print(f"  ✓ Tian I Task: {tian_task_network.shape[0]}×{tian_task_network.shape[1]} matrix")
    
    tian_change_network = tian_task_network - tian_rest_network
    
    print("Identifying Tian I top changed connections...")
    tian_top_changes = identify_top_changed_connections(
        tian_rest_matrix, tian_task_matrix, region_list, tian_mapping, top_k=100, analysis_name='Tian I'
    )
    print(f"✓ Tian I top 100 changed connections identified")
    print(f"  - Increased: {(tian_top_changes['change'] > 0).sum()}")
    print(f"  - Decreased: {(tian_top_changes['change'] < 0).sum()}")
    
    # =========================================================================
    # Step 5: Combined Analysis
    # =========================================================================
    print_section("Step 5: Combined N7 + Tian I Analysis")
    
    print("Computing combined resting-state connectivity...")
    combined_rest_network, combined_rest_matrix = compute_group_connectivity(
        df_rest, connection_columns, region_list, combined_mapping
    )
    print(f"  ✓ Combined Rest: {combined_rest_network.shape[0]}×{combined_rest_network.shape[1]} matrix")
    
    print("Computing combined task connectivity...")
    combined_task_network, combined_task_matrix = compute_group_connectivity(
        df_task, connection_columns, region_list, combined_mapping
    )
    print(f"  ✓ Combined Task: {combined_task_network.shape[0]}×{combined_task_network.shape[1]} matrix")
    
    combined_change_network = combined_task_network - combined_rest_network
    
    print("Identifying combined top changed connections...")
    combined_top_changes = identify_top_changed_connections(
        combined_rest_matrix, combined_task_matrix, region_list, combined_mapping, top_k=100, analysis_name='Combined'
    )
    print(f"✓ Combined top 100 changed connections identified")
    print(f"  - Increased: {(combined_top_changes['change'] > 0).sum()}")
    print(f"  - Decreased: {(combined_top_changes['change'] < 0).sum()}")
    
    # =========================================================================
    # Step 6: Save Results
    # =========================================================================
    print_section("Step 6: Save Results")
    
    output_tables = Path('reports/tables/connectivity_analysis')
    output_figures = Path('reports/figures/connectivity_analysis')
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    
    # Save N7 matrices
    n7_rest_network.to_csv(output_tables / 'n7_inter_network_connectivity_rest.csv')
    n7_task_network.to_csv(output_tables / 'n7_inter_network_connectivity_task.csv')
    n7_change_network.to_csv(output_tables / 'n7_inter_network_connectivity_change.csv')
    n7_top_changes.to_csv(output_tables / 'n7_top_changed_connections.csv', index=False)
    print("✓ Saved N7 cortical analysis results")
    
    # Save Tian I matrices
    tian_rest_network.to_csv(output_tables / 'tian_inter_network_connectivity_rest.csv')
    tian_task_network.to_csv(output_tables / 'tian_inter_network_connectivity_task.csv')
    tian_change_network.to_csv(output_tables / 'tian_inter_network_connectivity_change.csv')
    tian_top_changes.to_csv(output_tables / 'tian_top_changed_connections.csv', index=False)
    print("✓ Saved Tian I subcortical analysis results")
    
    # Save combined matrices
    combined_rest_network.to_csv(output_tables / 'combined_inter_network_connectivity_rest.csv')
    combined_task_network.to_csv(output_tables / 'combined_inter_network_connectivity_task.csv')
    combined_change_network.to_csv(output_tables / 'combined_inter_network_connectivity_change.csv')
    combined_top_changes.to_csv(output_tables / 'combined_top_changed_connections.csv', index=False)
    print("✓ Saved combined analysis results")
    
    # =========================================================================
    # Step 7: Generate Visualizations
    # =========================================================================
    print_section("Step 7: Generate Visualizations")
    
    # N7 visualization
    plot_connectivity_analysis(
        n7_rest_network, n7_task_network, n7_change_network, n7_top_changes,
        output_figures / 'connectivity_analysis_n7_cortical.png',
        title_prefix='N7 Cortical - '
    )
    
    # Tian I visualization
    plot_connectivity_analysis(
        tian_rest_network, tian_task_network, tian_change_network, tian_top_changes,
        output_figures / 'connectivity_analysis_tian_subcortical.png',
        title_prefix='Tian Scale I Subcortical - '
    )
    
    # Combined visualization
    plot_connectivity_analysis(
        combined_rest_network, combined_task_network, combined_change_network, combined_top_changes,
        output_figures / 'connectivity_analysis_combined.png',
        title_prefix='Combined N7 + Tian I - '
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("ANALYSIS COMPLETE!")
    
    print(f"""
Generated Files:
================

Tables (CSV):
  N7 Cortical Analysis:
    • {output_tables}/n7_inter_network_connectivity_rest.csv
    • {output_tables}/n7_inter_network_connectivity_task.csv
    • {output_tables}/n7_inter_network_connectivity_change.csv
    • {output_tables}/n7_top_changed_connections.csv

  Tian I Subcortical Analysis:
    • {output_tables}/tian_inter_network_connectivity_rest.csv
    • {output_tables}/tian_inter_network_connectivity_task.csv
    • {output_tables}/tian_inter_network_connectivity_change.csv
    • {output_tables}/tian_top_changed_connections.csv

  Combined Analysis:
    • {output_tables}/combined_inter_network_connectivity_rest.csv
    • {output_tables}/combined_inter_network_connectivity_task.csv
    • {output_tables}/combined_inter_network_connectivity_change.csv
    • {output_tables}/combined_top_changed_connections.csv

Figures (PNG):
  • {output_figures}/connectivity_analysis_n7_cortical.png
  • {output_figures}/connectivity_analysis_tian_subcortical.png
  • {output_figures}/connectivity_analysis_combined.png
  Each with 2×3 grid: Rest/Task/Change matrices + top increased/decreased + summary

Key Findings:
=============
""")
    
    # Print summaries for each analysis
    print("\n" + "="*80)
    print("N7 CORTICAL NETWORK ANALYSIS")
    print("="*80)
    print(f"\nNetworks: {', '.join(n7_networks)}")
    print(f"\nTop 5 Changed Connections:")
    print("-" * 80)
    for idx, row in n7_top_changes.head(5).iterrows():
        direction = "↑" if row['change'] > 0 else "↓"
        print(f"  {idx+1}. {row['network_i']:20s} ↔ {row['network_j']:20s} "
              f"{direction} {abs(row['change']):7.4f}")
    
    print(f"\nMean Connectivity Change Matrix:")
    print("-" * 80)
    print(n7_change_network.round(4).to_string())
    
    print("\n\n" + "="*80)
    print("TIAN SCALE I SUBCORTICAL NETWORK ANALYSIS")
    print("="*80)
    print(f"\nNetworks: {', '.join(tian_networks)}")
    print(f"\nTop 5 Changed Connections:")
    print("-" * 80)
    for idx, row in tian_top_changes.head(5).iterrows():
        direction = "↑" if row['change'] > 0 else "↓"
        print(f"  {idx+1}. {row['network_i']:20s} ↔ {row['network_j']:20s} "
              f"{direction} {abs(row['change']):7.4f}")
    
    print(f"\nMean Connectivity Change Matrix:")
    print("-" * 80)
    print(tian_change_network.round(4).to_string())
    
    print("\n\n" + "="*80)
    print("COMBINED N7 + TIAN I ANALYSIS")
    print("="*80)
    print(f"\nTop 5 Changed Connections:")
    print("-" * 80)
    for idx, row in combined_top_changes.head(5).iterrows():
        direction = "↑" if row['change'] > 0 else "↓"
        pair_type = categorize_network_pair_type(row['network_i'], row['network_j'])
        print(f"  {idx+1}. {row['network_i']:20s} ↔ {row['network_j']:20s} "
              f"{direction} {abs(row['change']):7.4f} [{pair_type}]")
    
    # Connection type statistics for combined analysis
    print("\n\nConnection Type Statistics (Combined):")
    print("-" * 80)
    combined_top_changes['pair_type'] = combined_top_changes.apply(
        lambda row: categorize_network_pair_type(row['network_i'], row['network_j']), 
        axis=1
    )
    
    for pair_type in ['Cortical-Cortical', 'Cortical-Subcortical', 'Subcortical-Subcortical']:
        subset = combined_top_changes[combined_top_changes['pair_type'] == pair_type]
        if len(subset) > 0:
            print(f"\n{pair_type}:")
            print(f"  Count: {len(subset)}")
            print(f"  Mean change: {subset['change'].mean():7.4f}")
            print(f"  Mean |change|: {subset['abs_change'].mean():7.4f}")
            print(f"  Increased: {(subset['change'] > 0).sum()}")
            print(f"  Decreased: {(subset['change'] < 0).sum()}")
    
    print("\n Connectivity analysis complete (separated N7 & Tian I)!\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())