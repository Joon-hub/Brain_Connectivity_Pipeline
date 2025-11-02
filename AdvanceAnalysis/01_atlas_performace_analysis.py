#!/usr/bin/env python3
"""
Atlas Performance Analysis - Core Confusion Matrices & Error Maps
==================================================================
Generates comprehensive performance metrics across:
- Schaefer N17 (17 networks)
- Schaefer N7 (7 networks)  
- Tian Scale II (16 subcortical regions)
- Tian Scale I (8 subcortical regions)

Usage:
    python 01_atlas_performance_analysis.py --config config.yaml

Outputs:
    - Confusion matrices (CSV + PNG) for all atlas levels
    - Per-region and per-network error rates
    - Rest vs Task comparison metrics
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Local imports
from data import load_connectivity_data, extract_connection_columns
from features import create_classification_dataset
from model import predict
from utils import load_config, set_random_seeds, print_section


def map_regions_to_networks(region_list, network_type='N7'):
    """Map individual regions to Schaefer networks (cortical) or subcortical names."""
    network_mapping = {}
    for region in region_list:
        name = region.lower()

        # Subcortical regions
        if not region.startswith(('LH_', 'RH_')):
            if 'hip' in name:
                network = 'Hippocampus'
            elif 'amy' in name or 'amg' in name:
                network = 'Amygdala'
            elif 'tha' in name or '_th' in name or name.startswith('th'):
                network = 'Thalamus'
            elif 'nac' in name or 'accumb' in name:
                network = 'Accumbens'
            elif 'put' in name:
                network = 'Putamen'
            elif 'gp' in name or 'pallid' in name:
                network = 'Pallidum'
            elif 'cau' in name:
                network = 'Caudate'
            else:
                network = 'SubcorticalOther'
        # Cortical regions
        else:
            if network_type == 'N7':
                if 'vis' in name:
                    network = 'Visual'
                elif 'sommot' in name or 'senmot' in name or 'motor' in name:
                    network = 'Somatomotor'
                elif ('dorsattn' in name) or ('dorsal' in name and 'attn' in name):
                    network = 'DorsalAttention'
                elif ('salventattn' in name) or (('ventral' in name or 'salience' in name) and 'attn' in name):
                    network = 'VentralAttention'
                elif 'limbic' in name or 'limb' in name:
                    network = 'Limbic'
                elif 'cont' in name or 'frontoparietal' in name or 'control' in name:
                    network = 'FrontoParietal'
                elif 'default' in name or 'dmn' in name:
                    network = 'DefaultMode'
                else:
                    network = 'CorticalOther'
            elif network_type == 'N17':
                # Fine-grained mapping
                if 'viscent' in name or 'vis_cent' in name:
                    network = 'VisCent'
                elif 'visperi' in name or 'vis_peri' in name:
                    network = 'VisPeri'
                elif 'sommota' in name or 'senmota' in name:
                    network = 'SomMotA'
                elif 'sommotb' in name or 'senmotb' in name:
                    network = 'SomMotB'
                elif 'dorsattna' in name or 'dorsattn_a' in name:
                    network = 'DorsAttnA'
                elif 'dorsattnb' in name or 'dorsattn_b' in name:
                    network = 'DorsAttnB'
                elif 'salventattna' in name or 'salventattn_a' in name:
                    network = 'SalVentAttnA'
                elif 'salventattnb' in name or 'salventattn_b' in name:
                    network = 'SalVentAttnB'
                elif 'limbica' in name or 'limbic_a' in name:
                    network = 'LimbicA'
                elif 'limbicb' in name or 'limbic_b' in name:
                    network = 'LimbicB'
                elif 'conta' in name or 'cont_a' in name:
                    network = 'ContA'
                elif 'contb' in name or 'cont_b' in name:
                    network = 'ContB'
                elif 'contc' in name or 'cont_c' in name:
                    network = 'ContC'
                elif 'defaulta' in name or 'default_a' in name:
                    network = 'DefaultA'
                elif 'defaultb' in name or 'default_b' in name:
                    network = 'DefaultB'
                elif 'defaultc' in name or 'default_c' in name:
                    network = 'DefaultC'
                elif 'temppar' in name:
                    network = 'TempPar'
                else:
                    network = 'CorticalOther'

        network_mapping[region] = network
    return network_mapping


def map_regions_to_tian_scale(region_list, scale='I'):
    """Map subcortical regions to Tian Scale I or II.
    
    Args:
        region_list: List of region names
        scale: 'I' for 8 regions, 'II' for 16 subdivisions
        
    Returns:
        Dictionary mapping region names to Tian labels
        
    Scale I (8 regions):
        - Hippocampus
        - Amygdala
        - Thalamus_post (DP + VP)
        - Thalamus_ant (DA + VA)
        - Accumbens
        - Pallidum
        - Putamen
        - Caudate
        
    Scale II (16 regions):
        - Hippocampus_ant, Hippocampus_post
        - Amygdala_lat, Amygdala_med
        - Thalamus_DP, Thalamus_VP, Thalamus_VA, Thalamus_DA
        - Accumbens_shell, Accumbens_core
        - Pallidum_post, Pallidum_ant
        - Putamen_ant, Putamen_post
        - Caudate_ant, Caudate_post
    """
    tian_mapping = {}
    
    for region in region_list:
        name = region.lower()
        
        # Cortical regions -> mark as Cortical
        if region.startswith(('LH_', 'RH_')):
            tian_mapping[region] = 'Cortical'
            continue
        
        # Scale II (16 subdivisions)
        if scale == 'II':
            # Hippocampus
            if 'ahip' in name:
                label = 'Hippocampus_ant'
            elif 'phip' in name:
                label = 'Hippocampus_post'
            # Amygdala
            elif 'lamy' in name:
                label = 'Amygdala_lat'
            elif 'mamy' in name:
                label = 'Amygdala_med'
            # Thalamus
            elif 'tha-dp' in name or 'tha_dp' in name:
                label = 'Thalamus_DP'
            elif 'tha-vp' in name or 'tha_vp' in name:
                label = 'Thalamus_VP'
            elif 'tha-va' in name or 'tha_va' in name:
                label = 'Thalamus_VA'
            elif 'tha-da' in name or 'tha_da' in name:
                label = 'Thalamus_DA'
            # Accumbens
            elif 'nac-shell' in name or 'nac_shell' in name:
                label = 'Accumbens_shell'
            elif 'nac-core' in name or 'nac_core' in name:
                label = 'Accumbens_core'
            # Pallidum (GP = Globus Pallidus)
            elif 'pgp' in name:
                label = 'Pallidum_post'
            elif 'agp' in name:
                label = 'Pallidum_ant'
            # Putamen
            elif 'aput' in name:
                label = 'Putamen_ant'
            elif 'pput' in name:
                label = 'Putamen_post'
            # Caudate
            elif 'acau' in name:
                label = 'Caudate_ant'
            elif 'pcau' in name:
                label = 'Caudate_post'
            else:
                label = 'SubcorticalOther'
        
        # Scale I (8 base regions)
        else:
            if 'hip' in name:
                label = 'Hippocampus'
            elif 'amy' in name:
                label = 'Amygdala'
            # Thalamus posterior (DP and VP)
            elif 'tha-dp' in name or 'tha-vp' in name or 'tha_dp' in name or 'tha_vp' in name:
                label = 'Thalamus_post'
            # Thalamus anterior (DA and VA)
            elif 'tha-da' in name or 'tha-va' in name or 'tha_da' in name or 'tha_va' in name:
                label = 'Thalamus_ant'
            elif 'nac' in name:
                label = 'Accumbens'
            elif 'put' in name:
                label = 'Putamen'
            elif 'gp' in name:
                label = 'Pallidum'
            elif 'cau' in name:
                label = 'Caudate'
            else:
                label = 'SubcorticalOther'
        
        tian_mapping[region] = label
    
    return tian_mapping


def aggregate_predictions_to_network(y_true, y_pred, region_list, network_mapping):
    """
    Aggregate region-level predictions to network-level.
    
    Args:
        y_true: True region indices
        y_pred: Predicted region indices
        region_list: List of region names
        network_mapping: Dict mapping region_name -> network_label
        
    Returns:
        y_true_network: Network labels (true)
        y_pred_network: Network labels (predicted)
        network_labels: Ordered list of unique network names
    """
    # Map region indices to network labels
    y_true_network = [network_mapping[region_list[idx]] for idx in y_true]
    y_pred_network = [network_mapping[region_list[idx]] for idx in y_pred]
    
    # Get unique network labels (sorted)
    network_labels = sorted(set(network_mapping.values()))
    
    return np.array(y_true_network), np.array(y_pred_network), network_labels


def filter_cortical_only(y_true_network, y_pred_network, network_labels):
    """
    Filter to keep only cortical networks (exclude subcortical).
    
    Returns:
        Filtered arrays with only cortical networks
    """
    subcortical = {'Hippocampus', 'Amygdala', 'Thalamus', 'Accumbens', 
                   'Putamen', 'Pallidum', 'Caudate', 'SubcorticalOther'}
    
    cortical_labels = [net for net in network_labels if net not in subcortical]
    
    # Create mask for cortical samples
    mask = np.isin(y_true_network, cortical_labels)
    
    y_true_cortical = y_true_network[mask]
    y_pred_cortical = y_pred_network[mask]
    
    return y_true_cortical, y_pred_cortical, cortical_labels


def filter_subcortical_only(y_true_network, y_pred_network, network_labels):
    """
    Filter to keep only subcortical regions (Tian atlas).
    
    Returns:
        Filtered arrays with only subcortical regions
    """
    # For Tian Scale I and II, we need to include the subdivided labels
    subcortical_base = {'Hippocampus', 'Amygdala', 'Thalamus', 'Accumbens',
                        'Putamen', 'Pallidum', 'Caudate', 'SubcorticalOther'}
    
    # Check if we have Tian Scale I labels (with underscores)
    subcortical_scale_i = {'Hippocampus', 'Amygdala', 'Thalamus_post', 'Thalamus_ant',
                           'Accumbens', 'Putamen', 'Pallidum', 'Caudate', 'SubcorticalOther'}
    
    # Check if we have Tian Scale II labels (with underscores and subdivisions)
    subcortical_scale_ii_patterns = ['_ant', '_post', '_lat', '_med', '_shell', '_core', 
                                      '_DP', '_VP', '_VA', '_DA']
    
    # Collect all subcortical labels from the actual network_labels
    subcortical_labels = []
    for net in network_labels:
        # Base subcortical (for old N7 mapping)
        if net in subcortical_base:
            subcortical_labels.append(net)
        # Scale I subdivisions
        elif net in subcortical_scale_i:
            subcortical_labels.append(net)
        # Scale II subdivisions
        elif any(pattern in net for pattern in subcortical_scale_ii_patterns):
            subcortical_labels.append(net)
    
    # Remove duplicates and sort
    subcortical_labels = sorted(set(subcortical_labels))
    
    # Create mask for subcortical samples
    mask = np.isin(y_true_network, subcortical_labels)
    
    y_true_sub = y_true_network[mask]
    y_pred_sub = y_pred_network[mask]
    
    return y_true_sub, y_pred_sub, subcortical_labels


def plot_confusion_matrix(cm, labels, title, output_path):
    """
    Plot and save a confusion matrix using Matplotlib with integer counts.

    Args:
        cm: Confusion matrix (2D numpy array)
        labels: List of class labels
        title: Plot title
        output_path: File path to save the figure
    """
    # Figure size adjusts with number of classes
    fig_size = max(8, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Plot the confusion matrix as an image
    im = ax.imshow(cm, interpolation='nearest', cmap='YlGn', aspect='auto')

    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=270, labelpad=15, fontsize=12, fontweight='bold')

    # Titles and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True label', fontsize=14, fontweight='bold')

    # Tick labels
    tick_fontsize = 10 if len(labels) > 10 else 12
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)

    # Annotate every cell with integer count
    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            # Use white text for dark cells, black for light ones
            color = "white" if value > max_val / 2 else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=color, fontsize=10, fontweight='bold')

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_network_error_rates(y_true, y_pred, region_list, network_mapping):
    """
    Calculate error rates per network.
    
    Returns:
        DataFrame with network-level metrics
    """
    y_true_network, y_pred_network, network_labels = aggregate_predictions_to_network(
        y_true, y_pred, region_list, network_mapping
    )
    
    results = []
    for network in network_labels:
        mask = (y_true_network == network)
        if mask.sum() == 0:
            continue
        
        acc = accuracy_score(y_true_network[mask], y_pred_network[mask])
        error = 1.0 - acc
        n_samples = mask.sum()
        
        results.append({
            'network': network,
            'accuracy': acc,
            'error_rate': error,
            'n_samples': n_samples
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('error_rate', ascending=False).reset_index(drop=True)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Atlas Performance Analysis')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    args = parser.parse_args()
    
    # Setup
    print_section("ATLAS PERFORMANCE ANALYSIS")
    config = load_config(args.config)
    set_random_seeds(config.get('random_seed', 42))
    
    # Load data
    print_section("Step 1: Load Data")
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']
    
    if args.sample:
        piop2_file = "data/sample/sample_piop2_small.csv"
        piop1_file = "data/sample/sample_piop1_small.csv"
    
    df_piop2 = load_connectivity_data(piop2_file)
    df_piop1 = load_connectivity_data(piop1_file)
    
    connection_columns = extract_connection_columns(df_piop2)
    
    # Create datasets
    print_section("Step 2: Create Datasets")
    X_train, y_train, subjects_train, region_list = create_classification_dataset(
        df_piop2, connection_columns, diagonal_strategy=config.get('diagonal_strategy', 'network_mean')
    )
    
    X_test, y_test, subjects_test, _ = create_classification_dataset(
        df_piop1, connection_columns, diagonal_strategy=config.get('diagonal_strategy', 'network_mean')
    )
    
    print(f"\nRegions: {len(region_list)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train or load model
    print_section("Step 3: Train/Load Model")
    model_path = Path(config['output_dirs']['models']) / 'trained_model.pkl'
    
    if model_path.exists():
        from model import load_model
        model, scaler = load_model(str(model_path))
        print("✓ Loaded existing model")
    else:
        print("✗ Model file not found. Please train the model first.")
        sys.exit("Terminating: trained model.pkl not found")
        
        
    
    # Get predictions
    print_section("Step 4: Generate Predictions")
    y_train_pred, _ = predict(model, scaler, X_train)
    y_test_pred, _ = predict(model, scaler, X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Create output directories
    tables_dir = Path('reports/tables/atlas_analysis')
    figures_dir = Path('reports/figures/atlas_analysis')
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Create region list as csv file
    # =========================================================================
    
    region_list_path = tables_dir / 'region_list.csv'

    # Convert list to DataFrame if it's a list
    if isinstance(region_list, list):
        region_list_df = pd.DataFrame(region_list, columns=['region'])
    else:
        region_list_df = region_list.copy()

    region_list_df.to_csv(region_list_path, index=False)
    print(f"✓ Region list saved: {region_list_path}")
    
    if isinstance(region_list, pd.DataFrame):
        region_list = region_list['region'].tolist()
    
    # =========================================================================
    # ANALYSIS 1: Schaefer N7 (7 Cortical Networks Only)
    # =========================================================================
    print_section("Analysis 1: Schaefer N7 (7 Cortical Networks)")
    
    network_mapping_n7 = map_regions_to_networks(region_list, network_type='N7')
    
    # Diagnostic: Check which regions map to CorticalOther
    cortical_other_regions = [r for r, n in network_mapping_n7.items() 
                              if n == 'CorticalOther' and r.startswith(('LH_', 'RH_'))]
    if cortical_other_regions:
        print("\n⚠ Warning: Found regions mapped to 'CorticalOther':")
        for r in cortical_other_regions[:5]:  # Show first 5
            print(f"  - {r}")
        if len(cortical_other_regions) > 5:
            print(f"  ... and {len(cortical_other_regions) - 5} more")
        print("\nThese regions don't match standard Schaefer network patterns.")
        print("They will be excluded from the 7-network cortical analysis.\n")
    
    # Rest (training) - Get full network predictions first
    y_train_net7, y_train_pred_net7, net7_labels = aggregate_predictions_to_network(
        y_train, y_train_pred, region_list, network_mapping_n7
    )
    
    # Filter to cortical only (7 networks, excluding CorticalOther)
    y_train_net7_cort, y_train_pred_net7_cort, net7_cortical_labels = filter_cortical_only(
        y_train_net7, y_train_pred_net7, net7_labels
    )
    
    # Further filter to exclude CorticalOther
    if 'CorticalOther' in net7_cortical_labels:
        net7_cortical_labels = [n for n in net7_cortical_labels if n != 'CorticalOther']
        mask = ~np.isin(y_train_net7_cort, ['CorticalOther'])
        y_train_net7_cort = y_train_net7_cort[mask]
        y_train_pred_net7_cort = y_train_pred_net7_cort[mask]
    
    cm_train_n7 = confusion_matrix(y_train_net7_cort, y_train_pred_net7_cort, 
                                   labels=net7_cortical_labels)
    
    # Task (test) - cortical only, excluding CorticalOther
    y_test_net7, y_test_pred_net7, _ = aggregate_predictions_to_network(
        y_test, y_test_pred, region_list, network_mapping_n7
    )
    y_test_net7_cort, y_test_pred_net7_cort, _ = filter_cortical_only(
        y_test_net7, y_test_pred_net7, net7_labels
    )
    
    if 'CorticalOther' in net7_cortical_labels:
        mask = ~np.isin(y_test_net7_cort, ['CorticalOther'])
        y_test_net7_cort = y_test_net7_cort[mask]
        y_test_pred_net7_cort = y_test_pred_net7_cort[mask]
    
    cm_test_n7 = confusion_matrix(y_test_net7_cort, y_test_pred_net7_cort, 
                                  labels=net7_cortical_labels)
    
    print(f"\nN7 Cortical Networks: {', '.join(net7_cortical_labels)}")
    print(f"Number of networks: {len(net7_cortical_labels)}")
    
    # Save confusion matrices
    pd.DataFrame(cm_train_n7, index=net7_cortical_labels, 
                columns=net7_cortical_labels).to_csv(
        tables_dir / 'confusion_matrix_N7_cortical_rest.csv'
    )
    pd.DataFrame(cm_test_n7, index=net7_cortical_labels,
                columns=net7_cortical_labels).to_csv(
        tables_dir / 'confusion_matrix_N7_cortical_task.csv'
    )
    
    # Plot
    plot_confusion_matrix(cm_train_n7, net7_cortical_labels, 
                         'Confusion Matrix: Schaefer N7 Cortical (Rest)',
                         figures_dir / 'confusion_N7_cortical_rest.png')
    plot_confusion_matrix(cm_test_n7, net7_cortical_labels,
                         'Confusion Matrix: Schaefer N7 Cortical (Task)',
                         figures_dir / 'confusion_N7_cortical_task.png')
    
    # Error rates (for all networks including subcortical)
    error_n7_rest = calculate_network_error_rates(y_train, y_train_pred, region_list, network_mapping_n7)
    error_n7_task = calculate_network_error_rates(y_test, y_test_pred, region_list, network_mapping_n7)
    
    error_n7_rest.to_csv(tables_dir / 'error_rates_N7_rest.csv', index=False)
    error_n7_task.to_csv(tables_dir / 'error_rates_N7_task.csv', index=False)
    
    print("\nN7 Error Rates (Rest - All Networks):")
    print(error_n7_rest.to_string(index=False))
    
    # =========================================================================
    # ANALYSIS 2: Schaefer N17 (17 Cortical Networks Only)
    # =========================================================================
    print_section("Analysis 2: Schaefer N17 (17 Cortical Networks)")
    
    network_mapping_n17 = map_regions_to_networks(region_list, network_type='N17')
    
    # Rest - Get full network predictions first
    y_train_net17, y_train_pred_net17, net17_labels = aggregate_predictions_to_network(
        y_train, y_train_pred, region_list, network_mapping_n17
    )
    
    # Filter to cortical only (17 networks)
    y_train_net17_cort, y_train_pred_net17_cort, net17_cortical_labels = filter_cortical_only(
        y_train_net17, y_train_pred_net17, net17_labels
    )
    cm_train_n17 = confusion_matrix(y_train_net17_cort, y_train_pred_net17_cort,
                                    labels=net17_cortical_labels)
    
    # Task - cortical only
    y_test_net17, y_test_pred_net17, _ = aggregate_predictions_to_network(
        y_test, y_test_pred, region_list, network_mapping_n17
    )
    y_test_net17_cort, y_test_pred_net17_cort, _ = filter_cortical_only(
        y_test_net17, y_test_pred_net17, net17_labels
    )
    cm_test_n17 = confusion_matrix(y_test_net17_cort, y_test_pred_net17_cort,
                                   labels=net17_cortical_labels)
    
    print(f"\nN17 Cortical Networks: {', '.join(net17_cortical_labels)}")
    print(f"Number of networks: {len(net17_cortical_labels)}")
    
    # Save
    pd.DataFrame(cm_train_n17, index=net17_cortical_labels,
                columns=net17_cortical_labels).to_csv(
        tables_dir / 'confusion_matrix_N17_cortical_rest.csv'
    )
    pd.DataFrame(cm_test_n17, index=net17_cortical_labels,
                columns=net17_cortical_labels).to_csv(
        tables_dir / 'confusion_matrix_N17_cortical_task.csv'
    )
    
    # Plot
    plot_confusion_matrix(cm_train_n17, net17_cortical_labels,
                         'Confusion Matrix: Schaefer N17 Cortical (Rest)',
                         figures_dir / 'confusion_N17_cortical_rest.png')
    plot_confusion_matrix(cm_test_n17, net17_cortical_labels,
                         'Confusion Matrix: Schaefer N17 Cortical (Task)',
                         figures_dir / 'confusion_N17_cortical_task.png')
    
    # Error rates (for all networks including subcortical)
    error_n17_rest = calculate_network_error_rates(y_train, y_train_pred, region_list, network_mapping_n17)
    error_n17_task = calculate_network_error_rates(y_test, y_test_pred, region_list, network_mapping_n17)
    
    error_n17_rest.to_csv(tables_dir / 'error_rates_N17_rest.csv', index=False)
    error_n17_task.to_csv(tables_dir / 'error_rates_N17_task.csv', index=False)
    
    # =========================================================================
    # ANALYSIS 3: Tian Scale I (8 Subcortical Regions Only)
    # =========================================================================
    print_section("Analysis 3: Tian Scale I (8 Subcortical Regions)")
    
    tian_mapping_i = map_regions_to_tian_scale(region_list, scale='I')
    
    # Rest - Get full predictions first
    y_train_tian1, y_train_pred_tian1, tian1_labels = aggregate_predictions_to_network(
        y_train, y_train_pred, region_list, tian_mapping_i
    )
    
    # Filter to subcortical only
    y_train_tian1_sub, y_train_pred_tian1_sub, tian1_sub_labels = filter_subcortical_only(
        y_train_tian1, y_train_pred_tian1, tian1_labels
    )
    cm_train_tian1 = confusion_matrix(y_train_tian1_sub, y_train_pred_tian1_sub,
                                      labels=tian1_sub_labels)
    
    # Task - subcortical only
    y_test_tian1, y_test_pred_tian1, _ = aggregate_predictions_to_network(
        y_test, y_test_pred, region_list, tian_mapping_i
    )
    y_test_tian1_sub, y_test_pred_tian1_sub, _ = filter_subcortical_only(
        y_test_tian1, y_test_pred_tian1, tian1_labels
    )
    cm_test_tian1 = confusion_matrix(y_test_tian1_sub, y_test_pred_tian1_sub,
                                     labels=tian1_sub_labels)
    
    print(f"\nTian Scale I Subcortical Regions: {', '.join(tian1_sub_labels)}")
    print(f"Number of regions: {len(tian1_sub_labels)}")
    
    # Save
    pd.DataFrame(cm_train_tian1, index=tian1_sub_labels,
                columns=tian1_sub_labels).to_csv(
        tables_dir / 'confusion_matrix_TianI_subcortical_rest.csv'
    )
    pd.DataFrame(cm_test_tian1, index=tian1_sub_labels,
                columns=tian1_sub_labels).to_csv(
        tables_dir / 'confusion_matrix_TianI_subcortical_task.csv'
    )
    
    # Plot
    plot_confusion_matrix(cm_train_tian1, tian1_sub_labels,
                         'Confusion Matrix: Tian Scale I Subcortical (Rest)',
                         figures_dir / 'confusion_TianI_subcortical_rest.png')
    plot_confusion_matrix(cm_test_tian1, tian1_sub_labels,
                         'Confusion Matrix: Tian Scale I Subcortical (Task)',
                         figures_dir / 'confusion_TianI_subcortical_task.png')
    
    # Error rates
    error_tian1_rest = calculate_network_error_rates(y_train, y_train_pred, region_list, tian_mapping_i)
    error_tian1_task = calculate_network_error_rates(y_test, y_test_pred, region_list, tian_mapping_i)
    
    error_tian1_rest.to_csv(tables_dir / 'error_rates_TianI_rest.csv', index=False)
    error_tian1_task.to_csv(tables_dir / 'error_rates_TianI_task.csv', index=False)
    
    # =========================================================================
    # ANALYSIS 3B: Combined N7 Cortical + Tian Scale I Subcortical
    # =========================================================================
    print_section("Analysis 3B: Combined N7 (7 Cortical) + Tian I (8 Subcortical)")
    
    # Combine cortical (N7) and subcortical (Tian I) predictions
    # For rest
    combined_labels_rest = list(net7_cortical_labels) + list(tian1_sub_labels)
    y_train_combined = np.concatenate([y_train_net7_cort, y_train_tian1_sub])
    y_train_pred_combined = np.concatenate([y_train_pred_net7_cort, y_train_pred_tian1_sub])
    
    cm_train_combined = confusion_matrix(y_train_combined, y_train_pred_combined,
                                         labels=combined_labels_rest)
    
    # For task
    y_test_combined = np.concatenate([y_test_net7_cort, y_test_tian1_sub])
    y_test_pred_combined = np.concatenate([y_test_pred_net7_cort, y_test_pred_tian1_sub])
    
    cm_test_combined = confusion_matrix(y_test_combined, y_test_pred_combined,
                                        labels=combined_labels_rest)
    
    print(f"\nCombined Networks: {', '.join(combined_labels_rest)}")
    print(f"Total: {len(combined_labels_rest)} networks (7 cortical + 8 subcortical)")
    
    # Save
    pd.DataFrame(cm_train_combined, index=combined_labels_rest,
                columns=combined_labels_rest).to_csv(
        tables_dir / 'confusion_matrix_N7_TianI_combined_rest.csv'
    )
    pd.DataFrame(cm_test_combined, index=combined_labels_rest,
                columns=combined_labels_rest).to_csv(
        tables_dir / 'confusion_matrix_N7_TianI_combined_task.csv'
    )
    
    # Plot
    plot_confusion_matrix(cm_train_combined, combined_labels_rest,
                         'Confusion Matrix: N7 Cortical + Tian I Subcortical (Rest)',
                         figures_dir / 'confusion_N7_TianI_combined_rest.png')
    plot_confusion_matrix(cm_test_combined, combined_labels_rest,
                         'Confusion Matrix: N7 Cortical + Tian I Subcortical (Task)',
                         figures_dir / 'confusion_N7_TianI_combined_task.png')
    
    # =========================================================================
    # ANALYSIS 4: Tian Scale II (16 Subcortical Regions Only)
    # =========================================================================
    print_section("Analysis 4: Tian Scale II (16 Subcortical Regions)")
    
    tian_mapping_ii = map_regions_to_tian_scale(region_list, scale='II')
    
    # Rest - Get full predictions first
    y_train_tian2, y_train_pred_tian2, tian2_labels = aggregate_predictions_to_network(
        y_train, y_train_pred, region_list, tian_mapping_ii
    )
    
    # Filter to subcortical only
    y_train_tian2_sub, y_train_pred_tian2_sub, tian2_sub_labels = filter_subcortical_only(
        y_train_tian2, y_train_pred_tian2, tian2_labels
    )
    
    # Check if we have any subcortical labels
    if len(tian2_sub_labels) == 0:
        print("\n⚠ Warning: No subdivided subcortical regions found.")
        print("This likely means your data uses Tian Scale I only (no fine subdivisions).")
        print("Skipping Tian Scale II analysis.")
        print("\nActual subcortical regions found:")
        # Print actual subcortical region names
        subcortical_regions = [r for r in region_list if not r.startswith(('LH_', 'RH_'))]
        for r in subcortical_regions[:10]:  # Print first 10
            print(f"  - {r}")
        if len(subcortical_regions) > 10:
            print(f"  ... and {len(subcortical_regions) - 10} more")
    else:
        cm_train_tian2 = confusion_matrix(y_train_tian2_sub, y_train_pred_tian2_sub,
                                          labels=tian2_sub_labels)
        
        # Task - subcortical only
        y_test_tian2, y_test_pred_tian2, _ = aggregate_predictions_to_network(
            y_test, y_test_pred, region_list, tian_mapping_ii
        )
        y_test_tian2_sub, y_test_pred_tian2_sub, _ = filter_subcortical_only(
            y_test_tian2, y_test_pred_tian2, tian2_labels
        )
        cm_test_tian2 = confusion_matrix(y_test_tian2_sub, y_test_pred_tian2_sub,
                                         labels=tian2_sub_labels)
        
        print(f"\nTian Scale II Subcortical Regions: {', '.join(tian2_sub_labels)}")
        print(f"Number of regions: {len(tian2_sub_labels)}")
        
        # Save
        pd.DataFrame(cm_train_tian2, index=tian2_sub_labels,
                    columns=tian2_sub_labels).to_csv(
            tables_dir / 'confusion_matrix_TianII_subcortical_rest.csv'
        )
        pd.DataFrame(cm_test_tian2, index=tian2_sub_labels,
                    columns=tian2_sub_labels).to_csv(
            tables_dir / 'confusion_matrix_TianII_subcortical_task.csv'
        )
        
        # Plot
        plot_confusion_matrix(cm_train_tian2, tian2_sub_labels,
                             'Confusion Matrix: Tian Scale II Subcortical (Rest)',
                             figures_dir / 'confusion_TianII_subcortical_rest.png')
        plot_confusion_matrix(cm_test_tian2, tian2_sub_labels,
                             'Confusion Matrix: Tian Scale II Subcortical (Task)',
                             figures_dir / 'confusion_TianII_subcortical_task.png')
        
        # Error rates
        error_tian2_rest = calculate_network_error_rates(y_train, y_train_pred, region_list, tian_mapping_ii)
        error_tian2_task = calculate_network_error_rates(y_test, y_test_pred, region_list, tian_mapping_ii)
        
        error_tian2_rest.to_csv(tables_dir / 'error_rates_TianII_rest.csv', index=False)
        error_tian2_task.to_csv(tables_dir / 'error_rates_TianII_task.csv', index=False)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("ANALYSIS COMPLETE!")
    
    print(f"""
Generated Files:
================

Confusion Matrices (CSV):
  CORTICAL ONLY:
    - {tables_dir}/confusion_matrix_N7_cortical_rest.csv (7×7)
    - {tables_dir}/confusion_matrix_N7_cortical_task.csv (7×7)
    - {tables_dir}/confusion_matrix_N17_cortical_rest.csv (17×17)
    - {tables_dir}/confusion_matrix_N17_cortical_task.csv (17×17)
  
  SUBCORTICAL ONLY:
    - {tables_dir}/confusion_matrix_TianI_subcortical_rest.csv (8×8)
    - {tables_dir}/confusion_matrix_TianI_subcortical_task.csv (8×8)
    - {tables_dir}/confusion_matrix_TianII_subcortical_rest.csv (16×16)
    - {tables_dir}/confusion_matrix_TianII_subcortical_task.csv (16×16)
  
  COMBINED (N7 + Tian I):
    - {tables_dir}/confusion_matrix_N7_TianI_combined_rest.csv (15×15)
    - {tables_dir}/confusion_matrix_N7_TianI_combined_task.csv (15×15)

Error Rates (CSV) - includes both cortical and subcortical:
  - {tables_dir}/error_rates_N7_rest.csv
  - {tables_dir}/error_rates_N7_task.csv
  - {tables_dir}/error_rates_N17_rest.csv
  - {tables_dir}/error_rates_N17_task.csv
  - {tables_dir}/error_rates_TianI_rest.csv
  - {tables_dir}/error_rates_TianI_task.csv
  - {tables_dir}/error_rates_TianII_rest.csv
  - {tables_dir}/error_rates_TianII_task.csv

Visualizations (PNG) - with numbers in heatmaps:
  CORTICAL:
    - {figures_dir}/confusion_N7_cortical_rest.png
    - {figures_dir}/confusion_N7_cortical_task.png
    - {figures_dir}/confusion_N17_cortical_rest.png
    - {figures_dir}/confusion_N17_cortical_task.png
  
  SUBCORTICAL:
    - {figures_dir}/confusion_TianI_subcortical_rest.png
    - {figures_dir}/confusion_TianI_subcortical_task.png
    - {figures_dir}/confusion_TianII_subcortical_rest.png
    - {figures_dir}/confusion_TianII_subcortical_task.png
  
  COMBINED:
    - {figures_dir}/confusion_N7_TianI_combined_rest.png
    - {figures_dir}/confusion_N7_TianI_combined_task.png
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())