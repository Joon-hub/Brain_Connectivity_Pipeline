#!/usr/bin/env python3
"""
Connectivity Analysis - Inter-Network Connectivity & Task Modulation
=====================================================================
Analyzes functional connectivity patterns:
1. Inter-network connectivity matrices (rest)
2. Task-induced connectivity changes
3. Subcortical-cortical coupling
4. Identification of task-modulated connections

Usage:
    python 03_connectivity_analysis.py --config config.yaml

Outputs:
    - Inter-network connectivity matrices
    - Task modulation matrices (task - rest)
    - Top changed connections
    - Subcortical-cortical coupling analysis
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
from features import extract_regions
from utils import load_config, set_random_seeds, print_section

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def reconstruct_connectivity_matrix(subject_values, connection_columns, region_to_idx, n_regions):
    """Reconstruct connectivity matrix from flattened data."""
    matrix = np.zeros((n_regions, n_regions), dtype=float)
    
    for col, value in zip(connection_columns, subject_values):
        if '~' not in col:
            continue
        region_a, region_b = col.split('~')
        if region_a in region_to_idx and region_b in region_to_idx:
            idx_a = region_to_idx[region_a]
            idx_b = region_to_idx[region_b]
            matrix[idx_a, idx_b] = value
            matrix[idx_b, idx_a] = value  # Symmetric
    
    np.fill_diagonal(matrix, 1.0)
    return matrix


def map_regions_to_networks(region_list, network_type='N7'):
    """Map regions to networks."""
    network_mapping = {}
    
    for region in region_list:
        name = region.lower()
        
        # Subcortical
        if not region.startswith(('LH_', 'RH_')):
            if 'hip' in name:
                network = 'Hippocampus'
            elif 'amy' in name or 'amg' in name:
                network = 'Amygdala'
            elif 'tha' in name or '_th' in name:
                network = 'Thalamus'
            elif 'nac' in name:
                network = 'Accumbens'
            elif 'put' in name:
                network = 'Putamen'
            elif 'gp' in name or 'pallid' in name:
                network = 'Pallidum'
            elif 'cau' in name:
                network = 'Caudate'
            else:
                network = 'SubcorticalOther'
        
        # Cortical (N7 only for simplicity)
        else:
            if 'vis' in name:
                network = 'Visual'
            elif 'sommot' in name:
                network = 'Somatomotor'
            elif 'dorsattn' in name:
                network = 'DorsalAttention'
            elif 'salventattn' in name:
                network = 'VentralAttention'
            elif 'limbic' in name:
                network = 'Limbic'
            elif 'cont' in name:
                network = 'FrontoParietal'
            elif 'default' in name:
                network = 'DefaultMode'
            else:
                network = 'CorticalOther'
        
        network_mapping[region] = network
    
    return network_mapping


def calculate_inter_network_connectivity(connectivity_matrix, region_list, network_mapping):
    """
    Calculate mean connectivity between all network pairs.
    
    Args:
        connectivity_matrix: n_regions × n_regions connectivity matrix
        region_list: List of region names
        network_mapping: Dict mapping region_name -> network_name
        
    Returns:
        DataFrame with inter-network connectivity
    """
    n_regions = len(region_list)
    
    # Get unique networks
    networks = sorted(set(network_mapping.values()))
    n_networks = len(networks)
    
    # Create network connectivity matrix
    network_conn = np.zeros((n_networks, n_networks))
    network_to_idx = {net: i for i, net in enumerate(networks)}
    
    # For each network pair, compute mean connectivity
    for i, net_i in enumerate(networks):
        for j, net_j in enumerate(networks):
            # Get region indices for each network
            regions_i = [k for k, r in enumerate(region_list) if network_mapping[r] == net_i]
            regions_j = [k for k, r in enumerate(region_list) if network_mapping[r] == net_j]
            
            if len(regions_i) > 0 and len(regions_j) > 0:
                # Get submatrix
                if i == j:
                    # Within-network connectivity (exclude diagonal)
                    submatrix = connectivity_matrix[np.ix_(regions_i, regions_j)]
                    mask = ~np.eye(len(regions_i), dtype=bool)
                    network_conn[i, j] = submatrix[mask].mean()
                else:
                    # Between-network connectivity
                    submatrix = connectivity_matrix[np.ix_(regions_i, regions_j)]
                    network_conn[i, j] = submatrix.mean()
    
    return pd.DataFrame(network_conn, index=networks, columns=networks)


def compute_group_connectivity(df, connection_columns, region_list, network_mapping):
    """Compute mean connectivity across all subjects."""
    region_to_idx = {r: i for i, r in enumerate(region_list)}
    n_regions = len(region_list)
    
    # Initialize group connectivity
    group_connectivity = np.zeros((n_regions, n_regions))
    
    # Average across subjects
    for idx in range(len(df)):
        subject_values = df.iloc[idx, 1:].to_numpy(dtype=float)
        matrix = reconstruct_connectivity_matrix(subject_values, connection_columns, 
                                                 region_to_idx, n_regions)
        group_connectivity += matrix
    
    group_connectivity /= len(df)
    
    # Compute inter-network connectivity
    inter_network = calculate_inter_network_connectivity(
        group_connectivity, region_list, network_mapping
    )
    
    return inter_network, group_connectivity


def identify_top_changed_connections(rest_matrix, task_matrix, region_list, 
                                     network_mapping, top_k=50):
    """
    Identify connections with largest change from rest to task.
    
    Returns:
        DataFrame with top changed connections
    """
    n_regions = len(region_list)
    
    # Compute change matrix
    change_matrix = task_matrix - rest_matrix
    
    # Get upper triangle indices (avoid duplicates)
    triu_idx = np.triu_indices(n_regions, k=1)
    
    # Extract changes
    changes = []
    for i, j in zip(*triu_idx):
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
    
    # Sort by absolute change
    df = df.sort_values('abs_change', ascending=False).reset_index(drop=True)
    
    return df.head(top_k)


def plot_connectivity_analysis(rest_network, task_network, change_network, 
                               top_changes, output_path):
    """
    Create comprehensive connectivity analysis figure.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Resting-state inter-network connectivity
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(rest_network, annot=False, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Connectivity'}, square=True, ax=ax1,
                linewidths=0.5)
    ax1.set_title('A) Inter-Network Connectivity (Rest)', fontweight='bold')
    ax1.set_xlabel('Network', fontweight='bold')
    ax1.set_ylabel('Network', fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    # Panel 2: Task inter-network connectivity
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(task_network, annot=False, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Connectivity'}, square=True, ax=ax2,
                linewidths=0.5)
    ax2.set_title('B) Inter-Network Connectivity (Task)', fontweight='bold')
    ax2.set_xlabel('Network', fontweight='bold')
    ax2.set_ylabel('Network', fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    # Panel 3: Change matrix (Task - Rest)
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(change_network, annot=False, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Change in Connectivity'}, square=True, ax=ax3,
                linewidths=0.5)
    ax3.set_title('C) Connectivity Change (Task - Rest)', fontweight='bold')
    ax3.set_xlabel('Network', fontweight='bold')
    ax3.set_ylabel('Network', fontweight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)
    
    # Panel 4: Top increased connections
    ax4 = fig.add_subplot(gs[1, 0])
    top_increased = top_changes[top_changes['change'] > 0].head(15)
    if len(top_increased) > 0:
        labels = [f"{row['network_i']}-{row['network_j']}" 
                 for _, row in top_increased.iterrows()]
        ax4.barh(range(len(top_increased)), top_increased['change'], 
                color='red', alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(top_increased)))
        ax4.set_yticklabels(labels, fontsize=7)
        ax4.set_xlabel('Connectivity Increase', fontweight='bold')
        ax4.set_title('D) Top 15 Increased Connections', fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)
    
    # Panel 5: Top decreased connections
    ax5 = fig.add_subplot(gs[1, 1])
    top_decreased = top_changes[top_changes['change'] < 0].head(15)
    if len(top_decreased) > 0:
        labels = [f"{row['network_i']}-{row['network_j']}" 
                 for _, row in top_decreased.iterrows()]
        ax5.barh(range(len(top_decreased)), top_decreased['change'],
                color='blue', alpha=0.7, edgecolor='black')
        ax5.set_yticks(range(len(top_decreased)))
        ax5.set_yticklabels(labels, fontsize=7)
        ax5.set_xlabel('Connectivity Decrease', fontweight='bold')
        ax5.set_title('E) Top 15 Decreased Connections', fontweight='bold')
        ax5.invert_yaxis()
        ax5.grid(axis='x', alpha=0.3)
    
    # Panel 6: Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate statistics
    n_increased = (top_changes['change'] > 0.01).sum()
    n_decreased = (top_changes['change'] < -0.01).sum()
    max_increase = top_changes['change'].max()
    max_decrease = top_changes['change'].min()
    
    stats_text = f"""
    F) CONNECTIVITY CHANGE SUMMARY
    ==============================
    
    Total Connections Analyzed: {len(top_changes)}
    
    Significant Changes (|Δ| > 0.01):
      Increased: {n_increased}
      Decreased: {n_decreased}
    
    Maximum Changes:
      Largest Increase: {max_increase:.4f}
      Largest Decrease: {max_decrease:.4f}
    
    Mean Absolute Change: {top_changes['abs_change'].mean():.4f}
    
    Top Network Pair (Increased):
      {top_changes.iloc[0]['network_i']} ↔ 
      {top_changes.iloc[0]['network_j']}
      Δ = {top_changes.iloc[0]['change']:.4f}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Functional Connectivity Analysis: Rest vs Task', 
                fontsize=14, fontweight='bold')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved connectivity analysis: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Connectivity Analysis')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    
    # Setup
    print_section("CONNECTIVITY ANALYSIS")
    config = load_config(args.config)
    set_random_seeds(config.get('random_seed', 42))
    
    # Load data
    print_section("Step 1: Load Data")
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']
    
    if args.sample:
        piop2_file = "data/sample/sample_piop2_small.csv"
        piop1_file = "data/sample/sample_piop1_small.csv"
    
    df_rest = load_connectivity_data(piop2_file)
    df_task = load_connectivity_data(piop1_file)
    
    connection_columns = extract_connection_columns(df_rest)
    
    # Extract regions
    print_section("Step 2: Extract Regions")
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    print(f"✓ Extracted {n_regions} regions")
    
    # Map regions to networks
    network_mapping = map_regions_to_networks(region_list, network_type='N7')
    networks = sorted(set(network_mapping.values()))
    print(f"✓ Mapped to {len(networks)} networks: {', '.join(networks)}")
    
    # Compute group-level connectivity
    print_section("Step 3: Compute Inter-Network Connectivity")
    
    print("Computing resting-state connectivity...")
    rest_network, rest_matrix = compute_group_connectivity(
        df_rest, connection_columns, region_list, network_mapping
    )
    
    print("Computing task connectivity...")
    task_network, task_matrix = compute_group_connectivity(
        df_task, connection_columns, region_list, network_mapping
    )
    
    print("✓ Computed inter-network connectivity matrices")
    
    # Compute change matrix
    print_section("Step 4: Identify Task-Modulated Connections")
    
    change_network = task_network - rest_network
    
    # Identify top changed connections
    top_changes = identify_top_changed_connections(
        rest_matrix, task_matrix, region_list, network_mapping, top_k=100
    )
    
    print(f"✓ Identified top 100 changed connections")
    
    # Save results
    print_section("Step 5: Save Results")
    
    output_tables = Path('reports/tables/connectivity_analysis')
    output_figures = Path('reports/figures/connectivity_analysis')
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    
    # Save matrices
    rest_network.to_csv(output_tables / 'inter_network_connectivity_rest.csv')
    task_network.to_csv(output_tables / 'inter_network_connectivity_task.csv')
    change_network.to_csv(output_tables / 'inter_network_connectivity_change.csv')
    
    # Save top changes
    top_changes.to_csv(output_tables / 'top_changed_connections.csv', index=False)
    
    print("✓ Saved connectivity matrices and top changes")
    
    # Create visualization
    print_section("Step 6: Generate Visualization")
    
    plot_connectivity_analysis(
        rest_network, task_network, change_network, top_changes,
        output_figures / 'connectivity_analysis.png'
    )
    
    # Summary
    print_section("ANALYSIS COMPLETE!")
    
    print(f"""
Generated Files:
================

Tables:
  - {output_tables}/inter_network_connectivity_rest.csv
  - {output_tables}/inter_network_connectivity_task.csv
  - {output_tables}/inter_network_connectivity_change.csv
  - {output_tables}/top_changed_connections.csv

Figures:
  - {output_figures}/connectivity_analysis.png

Key Findings:
=============
""")
    
    # Print top 10 changed connections
    print("\nTop 10 Changed Connections:")
    print(top_changes[['network_i', 'network_j', 'change', 'pct_change']].head(10).to_string(index=False))
    
    # Print network-level changes
    print("\n\nMean Connectivity Change per Network Pair:")
    print(change_network.to_string())
    
    print("\n✅ Connectivity analysis complete!\n")
    
    return 0


if __name__ == '__main__':
    from data import extract_connection_columns
    sys.exit(main())