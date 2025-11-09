#!/usr/bin/env python3
"""
Atlas Performance Analysis - Simplified with 2x2 Plots
Shows Rest vs Task with Raw and Normalized confusion matrices.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import load_connectivity_data, extract_connection_columns
from features import create_classification_dataset
from model import predict, load_model
from utils import load_config, set_random_seeds, print_section


# =============================================================================
# NETWORK MAPPING
# =============================================================================

def map_schaefer(region_list, n_networks=7):
    """Map regions to Schaefer N7 or N17 networks."""
    mapping = {}
    
    for region in region_list:
        name = region.lower()
        
        if not region.startswith(('LH_', 'RH_')):
            mapping[region] = 'Subcortical'
            continue
        
        if n_networks == 7:
            if 'vis' in name:
                net = 'Visual'
            elif 'sommot' in name or 'senmot' in name:
                net = 'Somatomotor'
            elif 'dorsattn' in name:
                net = 'DorsalAttention'
            elif 'salventattn' in name:
                net = 'VentralAttention'
            elif 'limbic' in name:
                net = 'Limbic'
            elif 'cont' in name:
                net = 'FrontoParietal'
            elif 'default' in name:
                net = 'DefaultMode'
            else:
                net = 'CorticalOther'
        
        else:  # n_networks == 17
            if 'viscent' in name:
                net = 'VisCent'
            elif 'visperi' in name:
                net = 'VisPeri'
            elif 'sommota' in name:
                net = 'SomMotA'
            elif 'sommotb' in name:
                net = 'SomMotB'
            elif 'dorsattna' in name:
                net = 'DorsAttnA'
            elif 'dorsattnb' in name:
                net = 'DorsAttnB'
            elif 'salventattna' in name:
                net = 'SalVentAttnA'
            elif 'salventattnb' in name:
                net = 'SalVentAttnB'
            elif 'limbica' in name:
                net = 'LimbicA'
            elif 'limbicb' in name:
                net = 'LimbicB'
            elif 'conta' in name:
                net = 'ContA'
            elif 'contb' in name:
                net = 'ContB'
            elif 'contc' in name:
                net = 'ContC'
            elif 'defaulta' in name:
                net = 'DefaultA'
            elif 'defaultb' in name:
                net = 'DefaultB'
            elif 'defaultc' in name:
                net = 'DefaultC'
            elif 'temppar' in name:
                net = 'TempPar'
            else:
                net = 'CorticalOther'
        
        mapping[region] = net
    
    return mapping


def map_tian(region_list, scale='I'):
    """Map regions to Tian Scale I (8 regions) or II (16 regions)."""
    mapping = {}
    
    for region in region_list:
        name = region.lower()
        
        if region.startswith(('LH_', 'RH_')):
            mapping[region] = 'Cortical'
            continue
        
        if scale == 'II':
            if 'ahip' in name:
                net = 'Hip_ant'
            elif 'phip' in name:
                net = 'Hip_post'
            elif 'lamy' in name:
                net = 'Amyg_lat'
            elif 'mamy' in name:
                net = 'Amyg_med'
            elif 'tha-dp' in name or 'tha_dp' in name:
                net = 'Thal_DP'
            elif 'tha-vp' in name or 'tha_vp' in name:
                net = 'Thal_VP'
            elif 'tha-va' in name or 'tha_va' in name:
                net = 'Thal_VA'
            elif 'tha-da' in name or 'tha_da' in name:
                net = 'Thal_DA'
            elif 'nac-shell' in name or 'nac_shell' in name:
                net = 'NAc_shell'
            elif 'nac-core' in name or 'nac_core' in name:
                net = 'NAc_core'
            elif 'pgp' in name:
                net = 'GP_post'
            elif 'agp' in name:
                net = 'GP_ant'
            elif 'aput' in name:
                net = 'Put_ant'
            elif 'pput' in name:
                net = 'Put_post'
            elif 'acau' in name:
                net = 'Caud_ant'
            elif 'pcau' in name:
                net = 'Caud_post'
            else:
                net = 'SubcortOther'
        
        else:  # Scale I
            if 'hip' in name:
                net = 'Hippocampus'
            elif 'amy' in name:
                net = 'Amygdala'
            elif 'tha-dp' in name or 'tha-vp' in name:
                net = 'Thal_post'
            elif 'tha-da' in name or 'tha-va' in name:
                net = 'Thal_ant'
            elif 'nac' in name:
                net = 'Accumbens'
            elif 'put' in name:
                net = 'Putamen'
            elif 'gp' in name:
                net = 'Pallidum'
            elif 'cau' in name:
                net = 'Caudate'
            else:
                net = 'SubcortOther'
        
        mapping[region] = net
    
    return mapping


# =============================================================================
# PROCESSING
# =============================================================================

def aggregate_networks(y_true, y_pred, region_list, mapping):
    """Convert region predictions to network predictions."""
    y_true_net = np.array([mapping[region_list[i]] for i in y_true])
    y_pred_net = np.array([mapping[region_list[i]] for i in y_pred])
    labels = sorted(set(mapping.values()))
    return y_true_net, y_pred_net, labels


def filter_networks(y_true, y_pred, labels, network_type='all', exclude=None):
    """Filter to cortical or subcortical networks only."""
    subcort_patterns = ['Hippocampus', 'Amygdala', 'Thal', 'Accumbens', 
                        'Putamen', 'Pallidum', 'Caudate', 'SubcortOther',
                        'Hip_', 'Amyg_', 'NAc_', 'GP_', 'Put_', 'Caud_']
    
    if network_type == 'cortical':
        keep = [l for l in labels 
                if not any(p in l for p in subcort_patterns) and l != 'Cortical']
    elif network_type == 'subcortical':
        keep = [l for l in labels 
                if any(p in l for p in subcort_patterns) and l != 'Cortical']
    else:
        keep = labels.copy()
    
    if exclude:
        keep = [l for l in keep if l not in exclude]
    
    keep = sorted(keep)
    mask = np.isin(y_true, keep)
    
    return y_true[mask], y_pred[mask], keep


def calculate_errors(y_true, y_pred, labels):
    """Calculate per-network error rates."""
    results = []
    for net in labels:
        mask = (y_true == net)
        n = mask.sum()
        if n == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        results.append({
            'network': net,
            'accuracy': acc,
            'error_rate': 1.0 - acc,
            'n_samples': n
        })
    return pd.DataFrame(results).sort_values('error_rate', ascending=False)


# =============================================================================
# PLOTTING - 2x2 GRID
# =============================================================================

def plot_2x2_confusion(cm_rest, cm_task, labels, title, output_path):
    """
    Create 2x2 grid: Rest/Task × Raw/Normalized confusion matrices.
    
    Layout:
        [Rest Raw]      [Task Raw]
        [Rest Norm]     [Task Norm]
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Normalize confusion matrices (row-wise: each row sums to 100%)
    cm_rest_norm = cm_rest.astype('float') / cm_rest.sum(axis=1, keepdims=True) * 100
    cm_task_norm = cm_task.astype('float') / cm_task.sum(axis=1, keepdims=True) * 100
    
    # Determine font size based on number of labels
    fontsize = 9 if len(labels) > 10 else 11
    
    # Panel 1: Rest Raw
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(cm_rest, cmap='YlGn', aspect='auto')
    ax1.set_title('REST - Raw Counts', fontweight='bold', fontsize=13)
    ax1.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize)
    ax1.set_yticklabels(labels, fontsize=fontsize)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(cm_rest[i, j])
            color = 'white' if val > cm_rest.max() / 2 else 'black'
            ax1.text(j, i, str(val), ha='center', va='center', 
                    color=color, fontsize=9, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Count', fontweight='bold')
    
    # Panel 2: Task Raw
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(cm_task, cmap='YlGn', aspect='auto')
    ax2.set_title('TASK - Raw Counts', fontweight='bold', fontsize=13)
    ax2.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize)
    ax2.set_yticklabels(labels, fontsize=fontsize)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(cm_task[i, j])
            color = 'white' if val > cm_task.max() / 2 else 'black'
            ax2.text(j, i, str(val), ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Count', fontweight='bold')
    
    # Panel 3: Rest Normalized
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(cm_rest_norm, cmap='YlGn', aspect='auto', vmin=0, vmax=100)
    ax3.set_title('REST - Normalized (%)', fontweight='bold', fontsize=13)
    ax3.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax3.set_xticks(np.arange(len(labels)))
    ax3.set_yticks(np.arange(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize)
    ax3.set_yticklabels(labels, fontsize=fontsize)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_rest_norm[i, j]
            color = 'white' if val > 50 else 'black'
            ax3.text(j, i, f'{val:.1f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')
    
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Percentage (%)', fontweight='bold')
    
    # Panel 4: Task Normalized
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(cm_task_norm, cmap='YlGn', aspect='auto', vmin=0, vmax=100)
    ax4.set_title('TASK - Normalized (%)', fontweight='bold', fontsize=13)
    ax4.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax4.set_xticks(np.arange(len(labels)))
    ax4.set_yticks(np.arange(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize)
    ax4.set_yticklabels(labels, fontsize=fontsize)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_task_norm[i, j]
            color = 'white' if val > 50 else 'black'
            ax4.text(j, i, f'{val:.1f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')
    
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('Percentage (%)', fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path.name}")


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

def analyze_atlas(y_rest, y_rest_pred, y_task, y_task_pred, 
                  region_list, mapping, filter_type, exclude,
                  name, tables_dir, figures_dir):
    """Run complete analysis for one atlas configuration."""
    
    # Rest data
    y_rest_net, y_rest_pred_net, all_labels = aggregate_networks(
        y_rest, y_rest_pred, region_list, mapping
    )
    y_rest_filt, y_rest_pred_filt, labels = filter_networks(
        y_rest_net, y_rest_pred_net, all_labels, filter_type, exclude
    )
    
    # Task data
    y_task_net, y_task_pred_net, _ = aggregate_networks(
        y_task, y_task_pred, region_list, mapping
    )
    y_task_filt, y_task_pred_filt, _ = filter_networks(
        y_task_net, y_task_pred_net, all_labels, filter_type, exclude
    )
    
    if len(labels) == 0:
        print(f"⚠ No data for {name}")
        return None
    
    # Calculate confusion matrices
    cm_rest = confusion_matrix(y_rest_filt, y_rest_pred_filt, labels=labels)
    cm_task = confusion_matrix(y_task_filt, y_task_pred_filt, labels=labels)
    
    # Calculate error rates
    err_rest = calculate_errors(y_rest_filt, y_rest_pred_filt, labels)
    err_task = calculate_errors(y_task_filt, y_task_pred_filt, labels)
    
    # Accuracy
    acc_rest = accuracy_score(y_rest_filt, y_rest_pred_filt)
    acc_task = accuracy_score(y_task_filt, y_task_pred_filt)
    
    print(f"\n{name}:")
    print(f"  Networks: {len(labels)}")
    print(f"  Rest:  {len(y_rest_filt):4d} samples, {acc_rest:.4f} accuracy")
    print(f"  Task:  {len(y_task_filt):4d} samples, {acc_task:.4f} accuracy")
    
    # Save CSV files
    pd.DataFrame(cm_rest, index=labels, columns=labels).to_csv(
        tables_dir / f'confusion_matrix_{name}_rest.csv'
    )
    pd.DataFrame(cm_task, index=labels, columns=labels).to_csv(
        tables_dir / f'confusion_matrix_{name}_task.csv'
    )
    err_rest.to_csv(tables_dir / f'error_rates_{name}_rest.csv', index=False)
    err_task.to_csv(tables_dir / f'error_rates_{name}_task.csv', index=False)
    
    # Create 2x2 plot
    plot_2x2_confusion(
        cm_rest, cm_task, labels,
        f'{name.replace("_", " ").title()}',
        figures_dir / f'confusion_{name}_2x2.png'
    )
    
    return {
        'cm_rest': cm_rest,
        'cm_task': cm_task,
        'err_rest': err_rest,
        'err_task': err_task,
        'labels': labels,
        'acc_rest': acc_rest,
        'acc_task': acc_task
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    
    print_section("ATLAS PERFORMANCE ANALYSIS - SIMPLIFIED")
    
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
    
    # Create datasets
    print_section("Creating Datasets")
    X_rest, y_rest, _, region_list = create_classification_dataset(
        df_rest, conn_cols,
        diagonal_strategy=config.get('diagonal_strategy', 'network_mean')
    )
    
    X_task, y_task, _, _ = create_classification_dataset(
        df_task, conn_cols,
        diagonal_strategy=config.get('diagonal_strategy', 'network_mean')
    )
    
    print(f"Regions: {len(region_list)}")
    print(f"Rest samples: {len(X_rest)}")
    print(f"Task samples: {len(X_task)}")
    
    # Load model
    print_section("Loading Model")
    model_path = Path(config['output_dirs']['models']) / 'trained_model.pkl'
    
    if not model_path.exists():
        print("✗ Model not found. Train model first.")
        return 1
    
    model, scaler = load_model(str(model_path))
    print("✓ Model loaded")
    
    # Predictions
    print_section("Generating Predictions")
    y_rest_pred, _ = predict(model, scaler, X_rest)
    y_task_pred, _ = predict(model, scaler, X_task)
    
    print(f"Rest accuracy: {accuracy_score(y_rest, y_rest_pred):.4f}")
    print(f"Task accuracy: {accuracy_score(y_task, y_task_pred):.4f}")
    
    # Output directories
    tables_dir = Path('reports/tables/atlas_analysis')
    figures_dir = Path('reports/figures/atlas_analysis')
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save region list
    pd.DataFrame({'region': region_list}).to_csv(
        tables_dir / 'region_list.csv', index=False
    )
    
    # =============================================================================
    # RUN ANALYSES
    # =============================================================================
    print_section("Running Atlas Analyses")
    
    results = {}
    
    # 1. N7 Cortical
    print("\n" + "="*70)
    print("N7 CORTICAL (7 Networks)")
    print("="*70)
    mapping_n7 = map_schaefer(region_list, n_networks=7)
    results['N7_cortical'] = analyze_atlas(
        y_rest, y_rest_pred, y_task, y_task_pred,
        region_list, mapping_n7, 'cortical', ['CorticalOther'],
        'N7_cortical', tables_dir, figures_dir
    )
    
    # 2. N17 Cortical
    print("\n" + "="*70)
    print("N17 CORTICAL (17 Networks)")
    print("="*70)
    mapping_n17 = map_schaefer(region_list, n_networks=17)
    results['N17_cortical'] = analyze_atlas(
        y_rest, y_rest_pred, y_task, y_task_pred,
        region_list, mapping_n17, 'cortical', ['CorticalOther'],
        'N17_cortical', tables_dir, figures_dir
    )
    
    # 3. Tian I Subcortical
    print("\n" + "="*70)
    print("TIAN SCALE I (8 Subcortical Regions)")
    print("="*70)
    mapping_tian1 = map_tian(region_list, scale='I')
    results['TianI_subcortical'] = analyze_atlas(
        y_rest, y_rest_pred, y_task, y_task_pred,
        region_list, mapping_tian1, 'subcortical', ['SubcortOther'],
        'TianI_subcortical', tables_dir, figures_dir
    )
    
    # 4. Tian II Subcortical
    print("\n" + "="*70)
    print("TIAN SCALE II (16 Subcortical Regions)")
    print("="*70)
    mapping_tian2 = map_tian(region_list, scale='II')
    results['TianII_subcortical'] = analyze_atlas(
        y_rest, y_rest_pred, y_task, y_task_pred,
        region_list, mapping_tian2, 'subcortical', ['SubcortOther'],
        'TianII_subcortical', tables_dir, figures_dir
    )
    
    # 5. Combined N7 + Tian I
    print("\n" + "="*70)
    print("COMBINED N7 + TIAN I")
    print("="*70)
    
    # Get N7 cortical filtered data
    y_rest_n7, y_rest_pred_n7, _ = aggregate_networks(
        y_rest, y_rest_pred, region_list, mapping_n7
    )
    y_rest_n7_c, y_rest_pred_n7_c, n7_labels = filter_networks(
        y_rest_n7, y_rest_pred_n7, sorted(set(mapping_n7.values())),
        'cortical', ['CorticalOther']
    )
    
    # Get Tian I subcortical filtered data
    y_rest_t1, y_rest_pred_t1, _ = aggregate_networks(
        y_rest, y_rest_pred, region_list, mapping_tian1
    )
    y_rest_t1_s, y_rest_pred_t1_s, t1_labels = filter_networks(
        y_rest_t1, y_rest_pred_t1, sorted(set(mapping_tian1.values())),
        'subcortical', ['SubcortOther']
    )
    
    # Combine for rest
    combined_labels = sorted(n7_labels + t1_labels)
    y_rest_comb = np.concatenate([y_rest_n7_c, y_rest_t1_s])
    y_rest_pred_comb = np.concatenate([y_rest_pred_n7_c, y_rest_pred_t1_s])
    
    # Same for task
    y_task_n7, y_task_pred_n7, _ = aggregate_networks(
        y_task, y_task_pred, region_list, mapping_n7
    )
    y_task_n7_c, y_task_pred_n7_c, _ = filter_networks(
        y_task_n7, y_task_pred_n7, sorted(set(mapping_n7.values())),
        'cortical', ['CorticalOther']
    )
    
    y_task_t1, y_task_pred_t1, _ = aggregate_networks(
        y_task, y_task_pred, region_list, mapping_tian1
    )
    y_task_t1_s, y_task_pred_t1_s, _ = filter_networks(
        y_task_t1, y_task_pred_t1, sorted(set(mapping_tian1.values())),
        'subcortical', ['SubcortOther']
    )
    
    y_task_comb = np.concatenate([y_task_n7_c, y_task_t1_s])
    y_task_pred_comb = np.concatenate([y_task_pred_n7_c, y_task_pred_t1_s])
    
    # Calculate metrics
    cm_rest_comb = confusion_matrix(y_rest_comb, y_rest_pred_comb, labels=combined_labels)
    cm_task_comb = confusion_matrix(y_task_comb, y_task_pred_comb, labels=combined_labels)
    err_rest_comb = calculate_errors(y_rest_comb, y_rest_pred_comb, combined_labels)
    err_task_comb = calculate_errors(y_task_comb, y_task_pred_comb, combined_labels)
    acc_rest_comb = accuracy_score(y_rest_comb, y_rest_pred_comb)
    acc_task_comb = accuracy_score(y_task_comb, y_task_pred_comb)
    
    print(f"\nN7 + Tian I Combined:")
    print(f"  Networks: {len(combined_labels)} ({len(n7_labels)} cortical + {len(t1_labels)} subcortical)")
    print(f"  Rest:  {len(y_rest_comb):4d} samples, {acc_rest_comb:.4f} accuracy")
    print(f"  Task:  {len(y_task_comb):4d} samples, {acc_task_comb:.4f} accuracy")
    
    # Save combined
    pd.DataFrame(cm_rest_comb, index=combined_labels, columns=combined_labels).to_csv(
        tables_dir / 'confusion_matrix_N7_TianI_combined_rest.csv'
    )
    pd.DataFrame(cm_task_comb, index=combined_labels, columns=combined_labels).to_csv(
        tables_dir / 'confusion_matrix_N7_TianI_combined_task.csv'
    )
    err_rest_comb.to_csv(tables_dir / 'error_rates_N7_TianI_combined_rest.csv', index=False)
    err_task_comb.to_csv(tables_dir / 'error_rates_N7_TianI_combined_task.csv', index=False)
    
    plot_2x2_confusion(
        cm_rest_comb, cm_task_comb, combined_labels,
        'N7 + Tian I Combined',
        figures_dir / 'confusion_N7_TianI_combined_2x2.png'
    )
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    print_section("ANALYSIS COMPLETE!")
    
    print("\nSummary:")
    print("="*70)
    
    for name, res in results.items():
        if res:
            print(f"\n{name.replace('_', ' ').title()}:")
            print(f"  Rest: {res['acc_rest']:.4f} | Task: {res['acc_task']:.4f}")
    
    print(f"\nN7 + Tian I Combined:")
    print(f"  Rest: {acc_rest_comb:.4f} | Task: {acc_task_comb:.4f}")
    
    print(f"""
Generated Files:
===============
Tables:  {tables_dir}
Figures: {figures_dir}

✓ All analyses use 2×2 plots (Rest/Task × Raw/Normalized)
✓ Normalized confusion matrices show row-wise percentages
✓ Easy comparison of rest vs task performance
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())