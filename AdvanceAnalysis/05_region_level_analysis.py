#!/usr/bin/env python3
"""
232×232 Region Confusion Matrices - Research-Focused Design
========================================================================

Key Features:
1. Main confusion matrix with optimal size
2. Meaningful accuracy visualization
3. Readable summary statistics panel
4. Informative network visualization
5. Research-focused metrics
6. Complete network coverage (all 232 regions, 24 networks)

"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import load_connectivity_data, extract_connection_columns
from old_code.features_251125 import extract_regions
from utils import load_config, set_random_seeds, print_section

plt.style.use('seaborn-v0_8-paper')


# =============================================================================
# NETWORK COLORS 
# =============================================================================

NETWORK_COLORS = {
    'Visual_Central': '#8B0000',
    'Visual_Peripheral': '#DC143C',
    'Visual': '#FF6347',
    'Somatomotor_A': '#4169E1',
    'Somatomotor_B': '#1E90FF',
    'Somatomotor': '#87CEEB',
    'DorsalAttn_A': '#228B22',
    'DorsalAttn_B': '#32CD32',
    'DorsalAttn': '#90EE90',
    'VentralAttn_A': '#9370DB',
    'VentralAttn_B': '#BA55D3',
    'VentralAttn': '#DDA0DD',
    'Limbic_A': '#FFD700',
    'Limbic_B': '#FFA500',
    'Limbic': '#FF8C00',
    'Control_A': '#00CED1',
    'Control_B': '#20B2AA',
    'Control_C': '#48D1CC',
    'Control': '#40E0D0',
    'DefaultMode_A': '#FF1493',
    'DefaultMode_B': '#FF69B4',
    'DefaultMode_C': '#FFB6C1',
    'DefaultMode': '#FFC0CB',
    'TemporalParietal': '#CD853F',
    'Hippocampus': '#8B4513',
    'Amygdala': '#A0522D',
    'Thalamus': '#D2691E',
    'Accumbens': '#B8860B',
    'Putamen': '#DAA520',
    'Pallidum': '#F0E68C',
    'Caudate': '#BDB76B',
    'Other': '#808080'
}


# =============================================================================
# REGION SORTING & STRUCTURE
# =============================================================================

def sort_regions_by_hierarchy(region_list):
    """
    Sort regions by: Hemisphere → Network → Region name.
    COMPLETE COVERAGE: All 232 regions properly assigned.
    """
    region_info = []
    
    for idx, region in enumerate(region_list):
        name = region.lower()
        
        # ==========================================
        # HEMISPHERE DETECTION
        # ==========================================
        if region.startswith('LH_') or region.endswith('-lh'):
            hemisphere = 'LH'
            hem_order = 0
        elif region.startswith('RH_') or region.endswith('-rh'):
            hemisphere = 'RH'
            hem_order = 1
        else:
            hemisphere = 'Subcortical'
            hem_order = 2
        
        # ==========================================
        # NETWORK ASSIGNMENT - MOST SPECIFIC FIRST
        # ==========================================
        
        # Visual Networks
        if 'viscent' in name:
            network = 'Visual_Central'
            net_order = 0
        elif 'visperi' in name:
            network = 'Visual_Peripheral'
            net_order = 1
        
        # Somatomotor Networks (check full string first)
        elif 'sommota' in name:
            network = 'Somatomotor_A'
            net_order = 2
        elif 'sommotb' in name:
            network = 'Somatomotor_B'
            net_order = 3
        
        # Dorsal Attention Networks
        elif 'dorsattna' in name:
            network = 'DorsalAttn_A'
            net_order = 4
        elif 'dorsattnb' in name:
            network = 'DorsalAttn_B'
            net_order = 5
        
        # Salience/Ventral Attention Networks
        elif 'salventattna' in name:
            network = 'VentralAttn_A'
            net_order = 6
        elif 'salventattnb' in name:
            network = 'VentralAttn_B'
            net_order = 7
        
        # Limbic Networks
        elif 'limbica' in name:
            network = 'Limbic_A'
            net_order = 8
        elif 'limbicb' in name:
            network = 'Limbic_B'
            net_order = 9
        
        # Control Networks
        elif 'conta' in name and not 'contb' in name and not 'contc' in name:
            network = 'Control_A'
            net_order = 10
        elif 'contb' in name:
            network = 'Control_B'
            net_order = 11
        elif 'contc' in name:
            network = 'Control_C'
            net_order = 12
        
        # Default Mode Networks
        elif 'defaulta' in name:
            network = 'DefaultMode_A'
            net_order = 13
        elif 'defaultb' in name:
            network = 'DefaultMode_B'
            net_order = 14
        elif 'defaultc' in name:
            network = 'DefaultMode_C'
            net_order = 15
        
        # Temporal Parietal
        elif 'temppar' in name:
            network = 'TemporalParietal'
            net_order = 16
        
        # Subcortical Structures (specific matching)
        elif 'ahip' in name or 'phip' in name:
            network = 'Hippocampus'
            net_order = 17
        elif 'lamy' in name or 'mamy' in name:
            network = 'Amygdala'
            net_order = 18
        elif 'tha-' in name:  # THA-DP, THA-VP, etc.
            network = 'Thalamus'
            net_order = 19
        elif 'nac-' in name:  # NAc-shell, NAc-core
            network = 'Accumbens'
            net_order = 20
        elif 'aput' in name or 'pput' in name:
            network = 'Putamen'
            net_order = 21
        elif 'agp' in name or 'pgp' in name:
            network = 'Pallidum'
            net_order = 22
        elif 'acau' in name or 'pcau' in name:
            network = 'Caudate'
            net_order = 23
        
        # Fallback - should not happen
        else:
            network = 'Unassigned'
            net_order = 99
            print(f"  Warning: Unassigned region '{region}'")
        
        region_info.append((hem_order, net_order, network, region, idx))
    
    # Sort by hemisphere → network → region name
    region_info.sort(key=lambda x: (x[0], x[1], x[3]))
    
    sorted_indices = [x[4] for x in region_info]
    sorted_regions = [x[3] for x in region_info]
    sorted_networks = [x[2] for x in region_info]
    
    # Coverage verification
    unassigned = sorted_networks.count('Unassigned')
    if unassigned == 0:
        print(f" All {len(region_list)} regions successfully assigned")
    else:
        print(f"  {unassigned} regions remain unassigned")
    
    return sorted_indices, sorted_regions, sorted_networks


def find_hemisphere_boundaries(sorted_regions):
    """Find hemisphere boundaries."""
    boundaries = []
    current_hem = None
    
    for i, region in enumerate(sorted_regions):
        if region.startswith('LH_'):
            hem = 'LH'
        elif region.startswith('RH_'):
            hem = 'RH'
        else:
            hem = 'Subcortical'
        
        if current_hem is not None and hem != current_hem:
            boundaries.append(i)
        current_hem = hem
    
    return boundaries


def find_network_boundaries_hierarchical(sorted_networks):
    """Find network boundaries and metadata."""
    boundaries = []
    network_labels = []
    current_network = sorted_networks[0]
    start_idx = 0
    
    for i, network in enumerate(sorted_networks):
        if network != current_network:
            boundaries.append(i)
            network_labels.append({
                'name': current_network,
                'start': start_idx,
                'end': i - 1,
                'center': (start_idx + i - 1) / 2,
                'size': i - start_idx
            })
            current_network = network
            start_idx = i
    
    network_labels.append({
        'name': current_network,
        'start': start_idx,
        'end': len(sorted_networks) - 1,
        'center': (start_idx + len(sorted_networks) - 1) / 2,
        'size': len(sorted_networks) - start_idx
    })
    
    return boundaries, network_labels


# =============================================================================
# CONFUSION MATRIX - FLEXIBLE DESIGN
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, region_list, dataset_name, 
                                       output_path, show_annotations=True):
    """
    Confusion matrix with flexible, readable layout.
    
    Design priorities:
    1. Main confusion matrix is clearly visible
    2. Summary statistics are actually readable
    3. Network information is informative
    4. Layout adapts to content needs
    
    NOTE: Uses ORIGINAL region order (no sorting)
    """
    n_regions = len(region_list)
    
    # DON'T sort - use original order
    sorted_indices = list(range(n_regions))  # [0, 1, 2, ..., 231]
    sorted_regions = region_list.copy()
    
    # Assign networks for coloring only (don't reorder)
    sorted_networks = []
    for region in sorted_regions:
        name = region.lower()
        
        # Network assignment (same logic, just for colors)
        if 'viscent' in name:
            network = 'Visual_Central'
        elif 'visperi' in name:
            network = 'Visual_Peripheral'
        elif 'sommota' in name:
            network = 'Somatomotor_A'
        elif 'sommotb' in name:
            network = 'Somatomotor_B'
        elif 'dorsattna' in name:
            network = 'DorsalAttn_A'
        elif 'dorsattnb' in name:
            network = 'DorsalAttn_B'
        elif 'salventattna' in name:
            network = 'VentralAttn_A'
        elif 'salventattnb' in name:
            network = 'VentralAttn_B'
        elif 'limbica' in name:
            network = 'Limbic_A'
        elif 'limbicb' in name:
            network = 'Limbic_B'
        elif 'conta' in name and not 'contb' in name and not 'contc' in name:
            network = 'Control_A'
        elif 'contb' in name:
            network = 'Control_B'
        elif 'contc' in name:
            network = 'Control_C'
        elif 'defaulta' in name:
            network = 'DefaultMode_A'
        elif 'defaultb' in name:
            network = 'DefaultMode_B'
        elif 'defaultc' in name:
            network = 'DefaultMode_C'
        elif 'temppar' in name:
            network = 'TemporalParietal'
        elif 'ahip' in name or 'phip' in name:
            network = 'Hippocampus'
        elif 'lamy' in name or 'mamy' in name:
            network = 'Amygdala'
        elif 'tha-' in name:
            network = 'Thalamus'
        elif 'nac-' in name:
            network = 'Accumbens'
        elif 'aput' in name or 'pput' in name:
            network = 'Putamen'
        elif 'agp' in name or 'pgp' in name:
            network = 'Pallidum'
        elif 'acau' in name or 'pcau' in name:
            network = 'Caudate'
        else:
            network = 'Unassigned'
        
        sorted_networks.append(network)
    
    # NO reordering of predictions - use as-is
    y_true_sorted = y_true
    y_pred_sorted = y_pred
    
    # Create confusion matrix
    labels = np.arange(n_regions)
    cm = confusion_matrix(y_true_sorted, y_pred_sorted, labels=labels)
    
    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        cm_norm = np.nan_to_num(cm_norm)
    
    # Calculate comprehensive metrics
    region_accuracy = np.diag(cm_norm) / 100.0
    overall_accuracy = accuracy_score(y_true_sorted, y_pred_sorted)
    
    # Structure
    hem_boundaries = find_hemisphere_boundaries(sorted_regions)
    net_boundaries, network_labels = find_network_boundaries_hierarchical(sorted_networks)
    
    # FIXED: Count actual regions per network (not just from boundaries)
    # Since we're using original order, count how many times each network appears
    network_region_counts = {}
    for network in sorted_networks:
        network_region_counts[network] = network_region_counts.get(network, 0) + 1
    
    # Colors
    network_colors_hex = [NETWORK_COLORS.get(net, '#808080') for net in sorted_networks]
    network_colors_rgb = [to_rgb(color) for color in network_colors_hex]
    
    # =========================
    # CREATE FIGURE - FLEXIBLE LAYOUT
    # =========================s
    fig = plt.figure(figsize=(38, 20))
    
    # Flexible GridSpec - NO TOP NETWORK BAR
    # Columns: [AccBar(1.2), NetBar(0.4), MainPlot(18), ColorBar(0.5), Summary(6), Legend(4)]
    # Rows: [MainArea(16.5), Bottom(1)]  - Only 2 rows now (no top bar)
    gs = GridSpec(2, 6, figure=fig,
                  width_ratios=[1.2, 0.4, 18.0, 0.5, 6.0, 4.0],
                  height_ratios=[16.5, 1.0],
                  hspace=0.10, wspace=0.20,
                  left=0.03, right=0.99, top=0.96, bottom=0.03)
    
    # Define axes - NO TOP BAR
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_left_net = fig.add_subplot(gs[0, 1])
    ax_main = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[0, 3])
    ax_stats = fig.add_subplot(gs[0, 4])
    ax_legend = fig.add_subplot(gs[:, 5])  # Span both rows
    
    # ==================
    # ACCURACY BAR (LEFT)
    # ==================
    ax_acc.barh(range(n_regions), region_accuracy, height=1.0,
                color=plt.cm.RdYlGn(region_accuracy), edgecolor='none', alpha=0.88)
    ax_acc.set_ylim([-0.5, n_regions-0.5])
    ax_acc.invert_yaxis()
    ax_acc.set_xlim([0, 1])
    ax_acc.set_xlabel('Accuracy', fontsize=18, fontweight='bold')  # Increased from 14
    ax_acc.set_yticks([])
    ax_acc.grid(axis='x', alpha=0.35, linewidth=0.9)
    ax_acc.tick_params(axis='x', labelsize=14)  # Increased from 11
    
    # Add overall accuracy reference line
    ax_acc.axvline(x=overall_accuracy, color='#0066CC', linestyle='--',
                   linewidth=3.0, alpha=0.85, label=f'Overall\n{overall_accuracy:.3f}')
    ax_acc.legend(loc='upper right', fontsize=13, framealpha=0.95,  # Increased from 10
                  edgecolor='#0066CC', fancybox=True)
    
    # Hemisphere boundaries
    for boundary in hem_boundaries:
        ax_acc.axhline(y=boundary-0.5, color='black', linewidth=3.0, alpha=0.75)
    
    # Highlight extremes
    max_idx = np.argmax(region_accuracy)
    min_idx = np.argmin(region_accuracy)
    ax_acc.plot(region_accuracy[max_idx], max_idx, 'g*', markersize=16, 
                markeredgecolor='darkgreen', markeredgewidth=2.0, zorder=5)
    ax_acc.plot(region_accuracy[min_idx], min_idx, 'r*', markersize=16,
                markeredgecolor='darkred', markeredgewidth=2.0, zorder=5)
    
    # ====================
    # LEFT NETWORK BAR (Only bar now - no top bar)
    # ====================
    color_array_left = np.array([[c] for c in network_colors_rgb])
    ax_left_net.imshow(color_array_left, aspect='auto', interpolation='nearest')
    ax_left_net.set_ylim([-0.5, n_regions-0.5])
    ax_left_net.invert_yaxis()
    ax_left_net.set_xticks([])
    ax_left_net.set_yticks([])
    ax_left_net.set_xlabel('Network', fontsize=14, fontweight='bold')  # Increased from 11
    
    for boundary in hem_boundaries:
        ax_left_net.axhline(y=boundary-0.5, color='white', linewidth=4.0, alpha=0.98)
    
    # ===================
    # MAIN CONFUSION MATRIX
    # ===================
    im = ax_main.imshow(cm_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100,
                        interpolation='nearest')
    
    # Perfect diagonal line
    ax_main.plot([0, n_regions-1], [0, n_regions-1], color='#0066CC', 
                 linewidth=3.0, alpha=0.7, linestyle='-', label='Perfect Classification')
    
    # Boundaries
    if show_annotations:
        # Only show HEMISPHERE boundaries (white, thick)
        for boundary in hem_boundaries:
            ax_main.axhline(y=boundary-0.5, color='white', linewidth=4.0, alpha=0.98)
            ax_main.axvline(x=boundary-0.5, color='white', linewidth=4.0, alpha=0.98)
        
        # Network boundaries - much more subtle (optional)
        # Comment out if you want completely clean look
        for boundary in net_boundaries:
            if boundary not in hem_boundaries:
                ax_main.axhline(y=boundary-0.5, color='black', linewidth=0.3, alpha=0.15)
                ax_main.axvline(x=boundary-0.5, color='black', linewidth=0.3, alpha=0.15)
    
    # Ticks
    tick_spacing = 20
    tick_pos = np.arange(0, n_regions, tick_spacing)
    ax_main.set_xticks(tick_pos)
    ax_main.set_yticks(tick_pos)
    tick_labels = [sorted_regions[i][:35] for i in tick_pos]
    ax_main.set_xticklabels(tick_labels, rotation=90, ha='right', fontsize=11)  # Increased from 9
    ax_main.set_yticklabels(tick_labels, fontsize=11)  # Increased from 9
    
    # Labels
    ax_main.set_xlabel('Predicted Region', fontweight='bold', fontsize=20)  # Increased from 16
    ax_main.set_ylabel('True Region', fontweight='bold', fontsize=20)  # Increased from 16
    
    # Title
    title = f'{dataset_name} Region Classification Performance\n'
    title += f'Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%) | '
    title += f'{n_regions} Regions | {len(y_true):,} Samples'
    ax_main.set_title(title, fontweight='bold', fontsize=22, pad=30)  # Increased from 18, reduced pad
    
    ax_main.legend(loc='upper left', fontsize=14, framealpha=0.95, fancybox=True)  # Increased from 11
    
    # ===================
    # COLORBAR
    # ===================
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Prediction Probability (%)', fontweight='bold', fontsize=16,  # Increased from 13
                   rotation=270, labelpad=30)  # Increased labelpad
    cbar.ax.tick_params(labelsize=13)  # Increased from 11
    
    # ===================
    # SUMMARY STATISTICS - READABLE SIZE
    # ===================
    ax_stats.axis('off')
    
    n_samples = len(y_true)
    error_rate = 1 - overall_accuracy
    mean_acc = region_accuracy.mean()
    std_acc = region_accuracy.std()
    median_acc = np.median(region_accuracy)
    
    n_lh = sum(1 for r in sorted_regions if r.startswith('LH_'))
    n_rh = sum(1 for r in sorted_regions if r.startswith('RH_'))
    n_subcort = n_regions - n_lh - n_rh
    
    # Calculate network-wise performance
    network_accs = {}
    for net_info in network_labels:
        net_name = net_info['name']
        start = net_info['start']
        end = net_info['end'] + 1
        net_acc = region_accuracy[start:end].mean()
        network_accs[net_name] = net_acc
    
    # Top performers
    best_idx = np.argsort(region_accuracy)[-5:][::-1]
    worst_idx = np.argsort(region_accuracy)[:5]
    
    # Best/worst networks
    sorted_nets = sorted(network_accs.items(), key=lambda x: x[1], reverse=True)
    best_networks = sorted_nets[:3]
    worst_networks = sorted_nets[-3:]
    
    stats_text = f"""RESEARCH QUESTIONS & FINDINGS
{'='*42}

RQ1: Can connectivity patterns predict regions?
   ✓ YES - {overall_accuracy*100:.2f}% accuracy achieved
   • {(overall_accuracy / (1.0/n_regions)):.1f}× better than chance ({100/n_regions:.2f}%)
   • Error rate: {error_rate*100:.2f}%

RQ2: Which regions are most/least predictable?
   
   Most Predictable (Top 5):
"""
    
    for idx in best_idx:
        name = sorted_regions[idx].replace('LH_', '').replace('RH_', '')[:22]
        acc = region_accuracy[idx]
        stats_text += f"   • {name:22s} {acc*100:5.2f}%\n"
    
    stats_text += f"""
   Least Predictable (Bottom 5):
"""
    
    for idx in worst_idx:
        name = sorted_regions[idx].replace('LH_', '').replace('RH_', '')[:22]
        acc = region_accuracy[idx]
        stats_text += f"   • {name:22s} {acc*100:5.2f}%\n"
    
    stats_text += f"""
RQ3: Network-level performance patterns?
   
   Best Performing Networks:
"""
    
    for net_name, net_acc in best_networks:
        display_name = net_name.replace('_', ' ')[:22]
        stats_text += f"   • {display_name:22s} {net_acc*100:5.2f}%\n"
    
    stats_text += f"""
   Most Challenging Networks:
"""
    
    for net_name, net_acc in worst_networks:
        display_name = net_name.replace('_', ' ')[:22]
        stats_text += f"   • {display_name:22s} {net_acc*100:5.2f}%\n"
    
    stats_text += f"""
{'='*42}
STATISTICAL SUMMARY
{'='*42}

Dataset Composition:
   • Total Regions:      {n_regions:6,d}
   • Left Hemisphere:    {n_lh:6,d}
   • Right Hemisphere:   {n_rh:6,d}
   • Subcortical:        {n_subcort:6,d}
   • Networks:           {len(network_labels):6d}
   • Total Samples:      {n_samples:6,d}

Region-wise Accuracy:
   • Mean:               {mean_acc*100:6.2f}%
   • Median:             {median_acc*100:6.2f}%
   • Std Dev:            {std_acc*100:6.2f}%
   • Range:              [{region_accuracy.min()*100:5.2f}%, {region_accuracy.max()*100:5.2f}%]

Performance Distribution:
   • Excellent (>80%):   {(region_accuracy > 0.8).sum():6d} regions
   • Good (60-80%):      {((region_accuracy > 0.6) & (region_accuracy <= 0.8)).sum():6d} regions
   • Fair (40-60%):      {((region_accuracy > 0.4) & (region_accuracy <= 0.6)).sum():6d} regions
   • Poor (<40%):        {(region_accuracy <= 0.4).sum():6d} regions
"""
    
    ax_stats.text(0.05, 0.98, stats_text, transform=ax_stats.transAxes,
                  fontsize=12, verticalalignment='top', fontfamily='monospace',  # Increased from 10.5
                  bbox=dict(boxstyle='round,pad=1.0', facecolor='#E8F4F8', 
                           alpha=0.95, edgecolor='#0066CC', linewidth=2.5))
    
    # ===================
    # NETWORK LEGEND
    # ===================
    ax_legend.axis('off')
    
    # Title
    ax_legend.text(0.5, 0.98, 'NETWORK COLOR LEGEND', 
                   transform=ax_legend.transAxes,
                   fontsize=14, fontweight='bold',  # Increased from 11
                   horizontalalignment='center',
                   verticalalignment='top')
    
    # Create compact legend - optimized to show ALL networks
    y_pos = 0.93
    y_spacing = 0.037  # Reduced spacing to fit all networks
    seen_networks = []
    
    for net_info in network_labels:
        if net_info['name'] not in seen_networks:
            seen_networks.append(net_info['name'])
            
            color = NETWORK_COLORS.get(net_info['name'], '#808080')
            name = net_info['name'].replace('_', ' ')
            # Use actual count from network_region_counts instead of boundary size
            n_reg = network_region_counts.get(net_info['name'], 0)
            net_acc = network_accs.get(net_info['name'], 0)
            
            # Smaller color box
            rect = Rectangle((0.05, y_pos - 0.022), 0.10, 0.025,
                            facecolor=color, edgecolor='black', linewidth=0.8,
                            transform=ax_legend.transAxes)
            ax_legend.add_patch(rect)
            
            # Compact label with accuracy
            label_text = f"{name:18s}({n_reg:2d}) {net_acc*100:4.1f}%"
            ax_legend.text(0.17, y_pos - 0.0095, label_text,
                          transform=ax_legend.transAxes,
                          fontsize=10, verticalalignment='center',  # Increased from 8.5
                          fontfamily='monospace')
            
            y_pos -= y_spacing
            
            # Stop if we run out of space (shouldn't happen with 24 networks)
            if y_pos < 0.02:
                break
    
    # Border
    ax_legend.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96,
                                  fill=False, edgecolor='#0066CC',
                                  linewidth=2.0, linestyle='-',
                                  transform=ax_legend.transAxes))
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    print(f"    Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"    Mean Region Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")


# =============================================================================
# DIFFERENCE MAPS
# =============================================================================

def plot_difference(y_true_1, y_pred_1, y_true_2, y_pred_2,
                    region_list, name1, name2, output_path):
    """Difference map with flexible layout.
    
    NOTE: Uses ORIGINAL region order (no sorting)
    """
    n_regions = len(region_list)
    
    # DON'T sort - use original order
    sorted_indices = list(range(n_regions))
    sorted_regions = region_list.copy()
    
    # Assign networks for coloring only
    sorted_networks = []
    for region in sorted_regions:
        name = region.lower()
        
        if 'viscent' in name:
            network = 'Visual_Central'
        elif 'visperi' in name:
            network = 'Visual_Peripheral'
        elif 'sommota' in name:
            network = 'Somatomotor_A'
        elif 'sommotb' in name:
            network = 'Somatomotor_B'
        elif 'dorsattna' in name:
            network = 'DorsalAttn_A'
        elif 'dorsattnb' in name:
            network = 'DorsalAttn_B'
        elif 'salventattna' in name:
            network = 'VentralAttn_A'
        elif 'salventattnb' in name:
            network = 'VentralAttn_B'
        elif 'limbica' in name:
            network = 'Limbic_A'
        elif 'limbicb' in name:
            network = 'Limbic_B'
        elif 'conta' in name and not 'contb' in name and not 'contc' in name:
            network = 'Control_A'
        elif 'contb' in name:
            network = 'Control_B'
        elif 'contc' in name:
            network = 'Control_C'
        elif 'defaulta' in name:
            network = 'DefaultMode_A'
        elif 'defaultb' in name:
            network = 'DefaultMode_B'
        elif 'defaultc' in name:
            network = 'DefaultMode_C'
        elif 'temppar' in name:
            network = 'TemporalParietal'
        elif 'ahip' in name or 'phip' in name:
            network = 'Hippocampus'
        elif 'lamy' in name or 'mamy' in name:
            network = 'Amygdala'
        elif 'tha-' in name:
            network = 'Thalamus'
        elif 'nac-' in name:
            network = 'Accumbens'
        elif 'aput' in name or 'pput' in name:
            network = 'Putamen'
        elif 'agp' in name or 'pgp' in name:
            network = 'Pallidum'
        elif 'acau' in name or 'pcau' in name:
            network = 'Caudate'
        else:
            network = 'Unassigned'
        
        sorted_networks.append(network)
    
    # NO reordering
    y_true_1_sorted = y_true_1
    y_pred_1_sorted = y_pred_1
    y_true_2_sorted = y_true_2
    y_pred_2_sorted = y_pred_2
    
    labels = np.arange(n_regions)
    
    # Confusion matrices
    cm1 = confusion_matrix(y_true_1_sorted, y_pred_1_sorted, labels=labels)
    cm2 = confusion_matrix(y_true_2_sorted, y_pred_2_sorted, labels=labels)
    
    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        cm1_norm = cm1.astype('float') / cm1.sum(axis=1, keepdims=True) * 100
        cm1_norm = np.nan_to_num(cm1_norm)
        cm2_norm = cm2.astype('float') / cm2.sum(axis=1, keepdims=True) * 100
        cm2_norm = np.nan_to_num(cm2_norm)
    
    # Difference
    diff_matrix = cm2_norm - cm1_norm
    
    # Metrics
    acc1 = np.diag(cm1_norm) / 100.0
    acc2 = np.diag(cm2_norm) / 100.0
    acc_change = acc2 - acc1
    
    overall_acc1 = accuracy_score(y_true_1_sorted, y_pred_1_sorted)
    overall_acc2 = accuracy_score(y_true_2_sorted, y_pred_2_sorted)
    overall_change = overall_acc2 - overall_acc1
    
    # Structure
    hem_boundaries = find_hemisphere_boundaries(sorted_regions)
    net_boundaries, network_labels = find_network_boundaries_hierarchical(sorted_networks)
    
    # FIXED: Count actual regions per network
    network_region_counts = {}
    for network in sorted_networks:
        network_region_counts[network] = network_region_counts.get(network, 0) + 1
    
    # Colors
    network_colors_hex = [NETWORK_COLORS.get(net, '#808080') for net in sorted_networks]
    network_colors_rgb = [to_rgb(color) for color in network_colors_hex]
    
    # =========================
    # CREATE FIGURE
    # =========================
    fig = plt.figure(figsize=(38, 20))
    
    # NO TOP NETWORK BAR
    gs = GridSpec(2, 6, figure=fig,
                  width_ratios=[1.2, 0.4, 18.0, 0.5, 6.0, 4.0],
                  height_ratios=[16.5, 1.0],
                  hspace=0.10, wspace=0.20,
                  left=0.03, right=0.99, top=0.96, bottom=0.03)
    
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_left_net = fig.add_subplot(gs[0, 1])
    ax_main = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[0, 3])
    ax_stats = fig.add_subplot(gs[0, 4])
    ax_legend = fig.add_subplot(gs[:, 5])
    
    # ==================
    # ACCURACY CHANGE BAR
    # ==================
    colors = ['#E74C3C' if x < -0.03 else '#2ECC71' if x > 0.03 else '#95A5A6' 
              for x in acc_change]
    ax_acc.barh(range(n_regions), acc_change, height=1.0, color=colors, 
                edgecolor='none', alpha=0.88)
    ax_acc.set_ylim([-0.5, n_regions-0.5])
    ax_acc.invert_yaxis()
    ax_acc.set_xlabel('Δ Accuracy', fontsize=18, fontweight='bold')  # Increased
    ax_acc.set_yticks([])
    ax_acc.axvline(x=0, color='black', linestyle='-', linewidth=3.0)
    ax_acc.axvline(x=overall_change, color='#0066CC', linestyle='--', 
                   linewidth=3.0, alpha=0.85, label=f'Overall\n{overall_change:+.3f}')
    ax_acc.legend(loc='best', fontsize=13, framealpha=0.95, edgecolor='#0066CC', fancybox=True)  # Increased
    ax_acc.grid(axis='x', alpha=0.35)
    ax_acc.tick_params(axis='x', labelsize=14)  # Increased
    
    for boundary in hem_boundaries:
        ax_acc.axhline(y=boundary-0.5, color='black', linewidth=3.0, alpha=0.75)
    
    # Highlight extremes
    max_change_idx = np.argmax(acc_change)
    min_change_idx = np.argmin(acc_change)
    ax_acc.plot(acc_change[max_change_idx], max_change_idx, 'g*', markersize=16,
                markeredgecolor='darkgreen', markeredgewidth=2.0, zorder=5)
    ax_acc.plot(acc_change[min_change_idx], min_change_idx, 'r*', markersize=16,
                markeredgecolor='darkred', markeredgewidth=2.0, zorder=5)
    
    # Left network bar only (NO TOP BAR)
    color_array_left = np.array([[c] for c in network_colors_rgb])
    ax_left_net.imshow(color_array_left, aspect='auto', interpolation='nearest')
    ax_left_net.set_ylim([-0.5, n_regions-0.5])
    ax_left_net.invert_yaxis()
    ax_left_net.set_xticks([])
    ax_left_net.set_yticks([])
    ax_left_net.set_xlabel('Network', fontsize=14, fontweight='bold')  # Increased
    for boundary in hem_boundaries:
        ax_left_net.axhline(y=boundary-0.5, color='white', linewidth=4.0, alpha=0.98)
    
    # Main difference map
    vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    im = ax_main.imshow(diff_matrix, cmap='RdBu_r', aspect='auto',
                        vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    ax_main.plot([0, n_regions-1], [0, n_regions-1], 'k--', linewidth=3.0, 
                 alpha=0.6, label='Diagonal')
    
    for boundary in hem_boundaries:
        ax_main.axhline(y=boundary-0.5, color='white', linewidth=4.0, alpha=0.98)
        ax_main.axvline(x=boundary-0.5, color='white', linewidth=4.0, alpha=0.98)
    
    # Network boundaries - much more subtle (optional)
    for boundary in net_boundaries:
        if boundary not in hem_boundaries:
            ax_main.axhline(y=boundary-0.5, color='black', linewidth=0.3, alpha=0.15)
            ax_main.axvline(x=boundary-0.5, color='black', linewidth=0.3, alpha=0.15)
    
    tick_spacing = 20
    tick_pos = np.arange(0, n_regions, tick_spacing)
    ax_main.set_xticks(tick_pos)
    ax_main.set_yticks(tick_pos)
    tick_labels = [sorted_regions[i][:35] for i in tick_pos]
    ax_main.set_xticklabels(tick_labels, rotation=90, ha='right', fontsize=11)  # Increased
    ax_main.set_yticklabels(tick_labels, fontsize=11)  # Increased
    
    ax_main.set_xlabel('Predicted Region', fontweight='bold', fontsize=20)  # Increased
    ax_main.set_ylabel('True Region', fontweight='bold', fontsize=20)  # Increased
    
    title = f'Performance Difference: {name2} vs {name1}\n'
    title += f'{name1}: {overall_acc1:.4f} → {name2}: {overall_acc2:.4f} '
    title += f'(Δ = {overall_change:+.4f} / {overall_change*100:+.2f}%)'
    ax_main.set_title(title, fontweight='bold', fontsize=22, pad=30)  # Increased, reduced pad
    
    ax_main.legend(loc='upper left', fontsize=14, framealpha=0.95, fancybox=True)  # Increased
    
    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Δ Prediction Probability (%)', fontweight='bold', fontsize=16,  # Increased
                   rotation=270, labelpad=30)  # Increased
    cbar.ax.tick_params(labelsize=13)  # Increased
    
    # Statistics
    ax_stats.axis('off')
    
    mean_change = acc_change.mean()
    std_change = acc_change.std()
    median_change = np.median(acc_change)
    
    threshold = 0.05
    n_improved = (acc_change > threshold).sum()
    n_degraded = (acc_change < -threshold).sum()
    n_stable = n_regions - n_improved - n_degraded
    
    most_improved = np.argsort(acc_change)[-10:][::-1]
    most_degraded = np.argsort(acc_change)[:10]
    
    # Network-wise changes
    network_changes = {}
    for net_info in network_labels:
        net_name = net_info['name']
        start = net_info['start']
        end = net_info['end'] + 1
        net_change = acc_change[start:end].mean()
        network_changes[net_name] = net_change
    
    sorted_net_changes = sorted(network_changes.items(), key=lambda x: x[1], reverse=True)
    
    stats_text = f"""RESEARCH QUESTION: GENERALIZATION
{'='*42}

RQ: How well does the model generalize
    from {name1} to {name2}?

Overall Performance Change:
   • {name1} Accuracy:     {overall_acc1*100:6.2f}%
   • {name2} Accuracy:     {overall_acc2*100:6.2f}%
   • Absolute Change:      {overall_change*100:+6.2f}%
   • Relative Change:      {(overall_change/overall_acc1)*100:+6.1f}%

Generalization Assessment:
"""
    
    if overall_change > 0.02:
        stats_text += f"   ✓ POSITIVE - Model improves on {name2}\n"
    elif overall_change < -0.02:
        stats_text += f"   ✗ NEGATIVE - Model degrades on {name2}\n"
    else:
        stats_text += f"   ≈ STABLE - Similar performance\n"
    
    stats_text += f"""
{'='*42}
REGION-WISE GENERALIZATION
{'='*42}

Statistical Summary:
   • Mean Change:         {mean_change*100:+6.2f}%
   • Median Change:       {median_change*100:+6.2f}%
   • Std Dev:             {std_change*100:6.2f}%
   • Range:               [{acc_change.min()*100:+5.1f}%, {acc_change.max()*100:+5.1f}%]

Change Distribution (±{threshold*100:.0f}% threshold):
   • Improved:            {n_improved:6d} regions
   • Stable:              {n_stable:6d} regions  
   • Degraded:            {n_degraded:6d} regions

Top 10 Most Improved Regions:
"""
    
    for idx in most_improved:
        name = sorted_regions[idx].replace('LH_', '').replace('RH_', '')[:22]
        change = acc_change[idx]
        stats_text += f"   {name:22s} {change*100:+6.2f}%\n"
    
    stats_text += f"""
Top 10 Most Degraded Regions:
"""
    
    for idx in most_degraded:
        name = sorted_regions[idx].replace('LH_', '').replace('RH_', '')[:22]
        change = acc_change[idx]
        stats_text += f"   {name:22s} {change*100:+6.2f}%\n"
    
    stats_text += f"""
{'='*42}
NETWORK-LEVEL CHANGES
{'='*42}

Most Improved Networks:
"""
    
    for net_name, net_change in sorted_net_changes[:5]:
        display_name = net_name.replace('_', ' ')[:22]
        stats_text += f"   {display_name:22s} {net_change*100:+6.2f}%\n"
    
    stats_text += f"""
Most Degraded Networks:
"""
    
    for net_name, net_change in sorted_net_changes[-5:]:
        display_name = net_name.replace('_', ' ')[:22]
        stats_text += f"   {display_name:22s} {net_change*100:+6.2f}%\n"
    
    ax_stats.text(0.05, 0.98, stats_text, transform=ax_stats.transAxes,
                  fontsize=12, verticalalignment='top', fontfamily='monospace',  # Increased
                  bbox=dict(boxstyle='round,pad=1.0', facecolor='#FFF8E8', 
                           alpha=0.95, edgecolor='#FF8C00', linewidth=2.5))
    
    # Legend
    ax_legend.axis('off')
    
    ax_legend.text(0.5, 0.98, 'NETWORK COLOR LEGEND', 
                   transform=ax_legend.transAxes,
                   fontsize=14, fontweight='bold',  # Increased
                   horizontalalignment='center',
                   verticalalignment='top')
    
    y_pos = 0.93
    y_spacing = 0.037  # Reduced spacing to fit all networks
    seen_networks = []
    
    for net_info in network_labels:
        if net_info['name'] not in seen_networks:
            seen_networks.append(net_info['name'])
            
            color = NETWORK_COLORS.get(net_info['name'], '#808080')
            name = net_info['name'].replace('_', ' ')
            # Use actual count from network_region_counts
            n_reg = network_region_counts.get(net_info['name'], 0)
            net_change = network_changes.get(net_info['name'], 0)
            
            rect = Rectangle((0.05, y_pos - 0.022), 0.10, 0.025,
                            facecolor=color, edgecolor='black', linewidth=0.8,
                            transform=ax_legend.transAxes)
            ax_legend.add_patch(rect)
            
            label_text = f"{name:18s}({n_reg:2d}) {net_change*100:+4.1f}%"
            ax_legend.text(0.17, y_pos - 0.0095, label_text,
                          transform=ax_legend.transAxes,
                          fontsize=10, verticalalignment='center',  # Increased
                          fontfamily='monospace')
            
            y_pos -= y_spacing
            
            if y_pos < 0.02:
                break
    
    ax_legend.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96,
                                  fill=False, edgecolor='#FF8C00',
                                  linewidth=2.0, linestyle='-',
                                  transform=ax_legend.transAxes))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    print(f"    Overall Change: {overall_change:+.4f} ({overall_change*100:+.2f}%)")
    print(f"    Mean Region Change: {mean_change:+.4f} ± {std_change:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Confusion matrices with flexible, research-focused design'
    )
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    
    print_section("CONFUSION MATRICES")
    print("Design Philosophy:")
    print("  • Research question focused")
    print("  • Readable summary statistics")
    print("  • Flexible, adaptive layout")
    print("  • Publication-ready quality\n")
    
    config = load_config(args.config)
    set_random_seeds(config.get('random_seed', 42))
    
    # Load data
    print_section("Loading Data")
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']
    
    if args.sample:
        piop2_file = "data/sample/sample_piop2_small.csv"
        piop1_file = "data/sample/sample_piop1_small.csv"
    
    df_rest = load_connectivity_data(piop2_file)
    df_task = load_connectivity_data(piop1_file)
    conn_cols = extract_connection_columns(df_rest)
    
    print_section("Extracting Regions")
    region_list, region_to_idx, n_regions = extract_regions(conn_cols)
    print(f"Total Regions: {n_regions}")
    
    # Load predictions
    print_section("Loading Predictions")
    pred_dir = Path(config.get('output_dirs', {}).get('processed', 'data/processed'))
    
    df_pred_train = pd.read_csv(pred_dir / 'predictions_train.csv')
    df_pred_val = pd.read_csv(pred_dir / 'predictions_cv_validation.csv')
    df_pred_task = pd.read_csv(pred_dir / 'predictions_task.csv')
    
    y_train = df_pred_train['true_region'].values
    y_train_pred = df_pred_train['predicted_region'].values
    y_val = df_pred_val['true_region'].values
    y_val_pred = df_pred_val['predicted_region'].values
    y_task = df_pred_task['true_region'].values
    y_task_pred = df_pred_task['predicted_region'].values
    
    print(" All predictions loaded successfully")
    
    # Output
    figures_dir = Path('reports/figures/region_level_analysis')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print_section("Generating Confusion Matrices")
    
    print("\n Training Dataset...")
    plot_confusion_matrix(
        y_train, y_train_pred, region_list, "Training (Rest fMRI)",
        figures_dir / 'confusion_training.png'
    )
    
    print("\n Validation Dataset...")
    plot_confusion_matrix(
        y_val, y_val_pred, region_list, "Validation (Cross-Validation)",
        figures_dir / 'confusion_validation.png'
    )
    
    print("\n Task Dataset...")
    plot_confusion_matrix(
        y_task, y_task_pred, region_list, "Task (Gender Stroop)",
        figures_dir / 'confusion_task.png'
    )
    
    print_section("Generating Difference Maps")
    
    print("\n Training → Task Generalization...")
    plot_difference(
        y_train, y_train_pred, y_task, y_task_pred,
        region_list, "Training", "Task",
        figures_dir / 'difference_training_vs_task.png'
    )
    
    print("\n Validation → Task Generalization...")
    plot_difference(
        y_val, y_val_pred, y_task, y_task_pred,
        region_list, "Validation", "Task",
        figures_dir / 'difference_validation_vs_task.png'
    )
    
    print_section("ANALYSIS COMPLETE!")
    print(f"""
    Output Directory: {figures_dir}

Generated Visualizations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 confusion_training.png
 confusion_validation.png
 confusion_task.png
 difference_training_vs_task.png
 difference_validation_vs_task.png
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())