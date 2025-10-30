"""
Network Analysis Module
=======================
Functions for specialized visualization and statistical analysis 
of errors grouped by functional networks (Schaefer/Tian).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

def aggregate_network_stats(error_df: pd.DataFrame, 
                            network_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Groups region-level error data by network and computes statistics.
    
    This function assumes a mapping from region_name to a Network label exists.
    
    Args:
        error_df: DataFrame with 'region_name' and 'misclassification_rate'.
        network_mapping: Dictionary where key=region_name, value=network_name.
        
    Returns:
        DataFrame of network statistics: 'network', 'mean_error', 
        'std_error', 'n_regions'.
    """
    # 1. Map regions to networks
    error_df = error_df.copy()
    error_df['network'] = error_df['region_name'].map(network_mapping)
    
    # Handle regions that might not be in the mapping (e.g., if you only map 17/100 regions)
    error_df.dropna(subset=['network'], inplace=True)
    
    # 2. Aggregate statistics
    network_stats = error_df.groupby('network')['misclassification_rate'].agg(
        mean_error='mean',
        std_error='std',
        n_regions='count'
    ).reset_index()
    
    # 3. Sort for plotting (e.g., by mean error, descending)
    network_stats = network_stats.sort_values(by='mean_error', ascending=False).reset_index(drop=True)
    
    return network_stats

def plot_network_analysis(error_df: pd.DataFrame,
                          network_stats: pd.DataFrame,
                          output_path: str,
                          title: str = 'Functional Network Error Analysis',
                          dpi: int = 300):
    """
    Creates a detailed 4-panel figure for network-level analysis.
    
    Args:
        error_df: Per-region error DataFrame (only used for region count check).
        network_stats: Network statistics DataFrame 
                       (output of aggregate_network_stats).
        output_path: Save path.
        title: Figure title.
        dpi: Resolution.
    """
    if network_stats.empty:
        print("Warning: network_stats is empty. Skipping network plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Network mean errors (Sorted by error rate)
    ax1 = axes[0, 0]
    # Use color map for visual error severity
    max_error = network_stats['mean_error'].max() if network_stats['mean_error'].max() > 0 else 1 
    colors = plt.cm.RdYlGn_r(network_stats['mean_error'] / max_error)
    
    bars = ax1.barh(range(len(network_stats)), network_stats['mean_error'],
                    color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax1.set_yticks(range(len(network_stats)))
    ax1.set_yticklabels(network_stats['network'], fontsize=9)
    ax1.set_xlabel('Mean Misclassification Rate', fontweight='bold')
    ax1.set_title('A) Mean Error Rate by Network (Sorted)', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Panel 2: Network error bars with standard deviation (Mean ± Std)
    ax2 = axes[0, 1]
    ax2.barh(range(len(network_stats)), network_stats['mean_error'],
             xerr=network_stats['std_error'].fillna(0), # fillna for single-region networks
             color='steelblue', edgecolor='black', linewidth=1.5, 
             alpha=0.7, capsize=5, label='Mean Error ± Std Dev')
             
    # Show overall mean error for context
    overall_mean = network_stats['mean_error'].mean()
    ax2.axvline(overall_mean, color='red', linestyle='--', linewidth=2, label='Overall Network Mean')
    
    ax2.set_yticks(range(len(network_stats)))
    ax2.set_yticklabels(network_stats['network'], fontsize=9)
    ax2.set_xlabel('Mean Misclassification Rate', fontweight='bold')
    ax2.set_title('B) Network Mean and Variability', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(loc='lower right', fontsize=8)

    # Panel 3: Number of regions per network
    ax3 = axes[1, 0]
    ax3.bar(range(len(network_stats)), network_stats['n_regions'],
            color='coral', edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax3.set_xticks(range(len(network_stats)))
    ax3.set_xticklabels(network_stats['network'], rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Number of Regions (Size)', fontweight='bold')
    ax3.set_title('C) Regions per Functional Network', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Summary table and key findings
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_network = network_stats.iloc[-1] # Lowest error (since sorted descending in aggregate)
    worst_network = network_stats.iloc[0] # Highest error
    
    stats_text = f"""
    D) NETWORK ANALYSIS SUMMARY
    ===========================
    
    Total Networks Analyzed: {len(network_stats)}
    Total Regions Mapped: {network_stats['n_regions'].sum()}
    
    Overall Mean Error: {network_stats['mean_error'].mean():.4f}
    
    🥇 Best Network (Lowest Error):
      {best_network['network']}
      Mean Error: {best_network['mean_error']:.4f}
      N Regions: {int(best_network['n_regions'])}
    
    💀 Worst Network (Highest Error):
      {worst_network['network']}
      Mean Error: {worst_network['mean_error']:.4f}
      N Regions: {int(worst_network['n_regions'])}
    
    Network Error Range (Max - Min): {(worst_network['mean_error'] - best_network['mean_error']):.4f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure: {output_path}")

# ----------------------------------------------------------------------
# STANDALONE EXECUTION LOGIC
# ----------------------------------------------------------------------

def main():
    """
    Simulates loading data and runs the network analysis plot.
    """
    print("--- Running Standalone Network Analysis Script ---")
    
    # 1. Simulate Input Data
    # --- Mock Region-level Error Data (e.g., 20 regions) ---
    region_names = [
        f"LH_VisCent_{i}" for i in range(1, 6)
    ] + [
        f"RH_DefaultA_{i}" for i in range(1, 6)
    ] + [
        f"L_HIP_ant_{i}" for i in range(1, 3)
    ] + [
        f"R_AMG_lat_{i}" for i in range(1, 3)
    ] + [
        f"LH_SomMotA_{i}" for i in range(1, 5)
    ]
    
    # Generate mock misclassification rates (0.0 to 0.5)
    np.random.seed(42) # for reproducible results
    misclassification_rates = np.random.uniform(0.1, 0.45, len(region_names))
    
    # Artificially make 'Visual' low and 'Limbic' high to test sorting/visualization
    for i, name in enumerate(region_names):
        if 'VisCent' in name:
            misclassification_rates[i] = np.random.uniform(0.05, 0.15)
        elif 'AMG' in name:
            misclassification_rates[i] = np.random.uniform(0.35, 0.5)

    error_data = pd.DataFrame({
        'region_name': region_names,
        'misclassification_rate': misclassification_rates
    })
    
    # 2. Define Network Mapping (Using logic from previous interaction)
    network_map = {}
    for region in region_names:
        name = region.lower()
        network = 'Unknown'
        
        if region.startswith(('LH_', 'RH_')):
            if 'viscent' in name or 'visperi' in name or 'striate' in name:
                network = 'Visual'
            elif 'sommota' in name or 'sommotb' in name:
                network = 'Somatomotor'
            elif 'defaulta' in name or 'defaultb' in name or 'defaultc' in name:
                network = 'DefaultMode'
            else:
                network = 'CorticalOther'
        else:
            if 'hip' in name:
                network = 'Hippocampus'
            elif 'amy' in name or 'amg' in name:
                network = 'Amygdala'
            else:
                network = 'SubcorticalOther'
        
        network_map[region] = network
        
    print(f"Loaded {len(error_data)} regions for analysis.")
    print(f"Mapped to {len(set(network_map.values()))} networks.")
    
    # 3. Aggregate Network Statistics
    network_stats_df = aggregate_network_stats(error_data, network_map)
    
    # Print statistics for verification
    print("\nNetwork Statistics (Top 5):")
    print(network_stats_df.head().to_markdown(index=False, floatfmt=".4f"))

    # 4. Define Output Path and Run Plotting
    output_filename = 'network_analysis_figure.png'
    output_directory = 'temp_network_figures'
    output_path = Path(output_directory) / output_filename
    
    plot_network_analysis(
        error_data,
        network_stats_df,
        output_path=str(output_path),
        title='Standalone Demo: Error Rate by Functional Network'
    )
    
    print(f"\n--- Script Finished ---")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()