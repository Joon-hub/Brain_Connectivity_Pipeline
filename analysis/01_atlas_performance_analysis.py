#!/usr/bin/env python3
"""
Atlas Performance Analysis - 2x2 Layout (IMPROVED)
Shows normalized Rest, normalized Task, Difference, and Summary Statistics.
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
from features import extract_regions
from utils import load_config, set_random_seeds, print_section


# NETWORK MAPPING

def map_schaefer(region_list, n_networks=7):
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
                net = 'Dorsal Attention'
            elif 'salventattn' in name:
                net = 'Ventral Attention'
            elif 'limbic' in name:
                net = 'Limbic'
            elif 'cont' in name:
                net = 'Fronto-Parietal'
            elif 'default' in name:
                net = 'Default Mode'
            else:
                net = 'Cortical Other'
        else:
            if 'viscent' in name:
                net = 'Vis Central'
            elif 'visperi' in name:
                net = 'Vis Peripheral'
            elif 'sommota' in name:
                net = 'SomMot A'
            elif 'sommotb' in name:
                net = 'SomMot B'
            elif 'dorsattna' in name:
                net = 'DorsAttn A'
            elif 'dorsattnb' in name:
                net = 'DorsAttn B'
            elif 'salventattna' in name:
                net = 'SalVentAttn A'
            elif 'salventattnb' in name:
                net = 'SalVentAttn B'
            elif 'limbica' in name:
                net = 'Limbic A'
            elif 'limbicb' in name:
                net = 'Limbic B'
            elif 'conta' in name:
                net = 'Control A'
            elif 'contb' in name:
                net = 'Control B'
            elif 'contc' in name:
                net = 'Control C'
            elif 'defaulta' in name:
                net = 'Default A'
            elif 'defaultb' in name:
                net = 'Default B'
            elif 'defaultc' in name:
                net = 'Default C'
            elif 'temppar' in name:
                net = 'Temporal-Parietal'
            else:
                net = 'Cortical Other'
        mapping[region] = net
    return mapping


def map_tian(region_list, scale='I'):
    mapping = {}
    for region in region_list:
        name = region.lower()
        if region.startswith(('LH_', 'RH_')):
            mapping[region] = 'Cortical'
            continue
        if scale == 'II':
            if 'ahip' in name:
                net = 'Hippocampus Anterior'
            elif 'phip' in name:
                net = 'Hippocampus Posterior'
            elif 'lamy' in name:
                net = 'Amygdala Lateral'
            elif 'mamy' in name:
                net = 'Amygdala Medial'
            elif 'tha-dp' in name or 'tha_dp' in name:
                net = 'Thalamus DP'
            elif 'tha-vp' in name or 'tha_vp' in name:
                net = 'Thalamus VP'
            elif 'tha-va' in name or 'tha_va' in name:
                net = 'Thalamus VA'
            elif 'tha-da' in name or 'tha_da' in name:
                net = 'Thalamus DA'
            elif 'nac-shell' in name or 'nac_shell' in name:
                net = 'NAc Shell'
            elif 'nac-core' in name or 'nac_core' in name:
                net = 'NAc Core'
            elif 'pgp' in name:
                net = 'Pallidum Posterior'
            elif 'agp' in name:
                net = 'Pallidum Anterior'
            elif 'aput' in name:
                net = 'Putamen Anterior'
            elif 'pput' in name:
                net = 'Putamen Posterior'
            elif 'acau' in name:
                net = 'Caudate Anterior'
            elif 'pcau' in name:
                net = 'Caudate Posterior'
            else:
                net = 'Subcortical Other'
        else:
            if 'hip' in name:
                net = 'Hippocampus'
            elif 'amy' in name:
                net = 'Amygdala'
            elif 'tha-dp' in name or 'tha_dp' in name or 'tha-vp' in name or 'tha_vp' in name:
                net = 'Thalamus Posterior'
            elif 'tha-da' in name or 'tha_da' in name or 'tha-va' in name or 'tha_va' in name:
                net = 'Thalamus Anterior'
            elif 'nac' in name:
                net = 'Accumbens'
            elif 'put' in name:
                net = 'Putamen'
            elif 'gp' in name:
                net = 'Pallidum'
            elif 'cau' in name:
                net = 'Caudate'
            else:
                net = 'Subcortical Other'
        mapping[region] = net
    return mapping


# PROCESSING

def aggregate_networks(y_true, y_pred, region_list, mapping):
    y_true_net = np.array([mapping[region_list[i]] for i in y_true])
    y_pred_net = np.array([mapping[region_list[i]] for i in y_pred])
    labels = sorted(set(mapping.values()))
    return y_true_net, y_pred_net, labels


def filter_networks(y_true, y_pred, labels, network_type='all', exclude=None):
    subcort_patterns = [
        'Hippocampus', 'Amygdala', 'Thal', 'Accumbens',
        'Putamen', 'Pallidum', 'Caudate', 'Subcortical Other',
        'Hip_', 'Amyg_', 'NAc', 'GP_', 'Put_', 'Caud_',
    ]
    if network_type == 'cortical':
        keep = [l for l in labels if not any(p in l for p in subcort_patterns) and l != 'Cortical']
    elif network_type == 'subcortical':
        keep = [l for l in labels if any(p in l for p in subcort_patterns) and l != 'Cortical']
    else:
        keep = labels.copy()
    if exclude:
        keep = [l for l in keep if l not in exclude]
    keep = sorted(keep)
    mask = np.isin(y_true, keep)
    return y_true[mask], y_pred[mask], keep


def calculate_errors(y_true, y_pred, labels):
    rows = []
    for net in labels:
        mask = (y_true == net)
        n = mask.sum()
        if n == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        rows.append(
            {'network': net, 'accuracy': acc, 'error_rate': 1.0 - acc, 'n_samples': n}
        )
    return pd.DataFrame(rows).sort_values('error_rate', ascending=False)


# PLOTTING (unchanged)

def plot_2x2_confusion(cm_rest, cm_task, labels, title, output_path,
                       acc_rest=None, acc_task=None):
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35,
                          left=0.08, right=0.95, top=0.93, bottom=0.05)

    cm_rest_norm = cm_rest.astype('float') / cm_rest.sum(axis=1, keepdims=True) * 100
    cm_task_norm = cm_task.astype('float') / cm_task.sum(axis=1, keepdims=True) * 100
    cm_diff = cm_rest_norm - cm_task_norm

    fontsize = 8 if len(labels) > 10 else 10
    label_fontsize = 9 if len(labels) > 10 else 11

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(cm_rest_norm, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    title_rest = '(A) Rest - Normalized'
    if acc_rest is not None:
        title_rest += f'\nAccuracy: {acc_rest:.2%}'
    ax1.set_title(title_rest, fontweight='bold', fontsize=18, pad=10)
    ax1.set_ylabel('True Network', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Predicted Network', fontweight='bold', fontsize=14)
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=label_fontsize)
    ax1.set_yticklabels(labels, fontsize=label_fontsize)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_rest_norm[i, j]
            if i == j or val > 10.0:
                color = 'white' if val > 50 else 'black'
                ax1.text(j, i, f'{val:.1f}', ha='center', va='center',
                         color=color, fontsize=fontsize, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Percentage [0-100]', fontweight='bold', fontsize=12)
    cbar1.ax.tick_params(labelsize=11)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(cm_task_norm, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    title_task = '(B) Task - Normalized'
    if acc_task is not None:
        title_task += f'\nAccuracy: {acc_task:.2%}'
    ax2.set_title(title_task, fontweight='bold', fontsize=18, pad=10)
    ax2.set_ylabel('True Network', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Predicted Network', fontweight='bold', fontsize=14)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=label_fontsize)
    ax2.set_yticklabels(labels, fontsize=label_fontsize)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_task_norm[i, j]
            if i == j or val > 10.0:
                color = 'white' if val > 50 else 'black'
                ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                         color=color, fontsize=fontsize, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Percentage [0-100]', fontweight='bold', fontsize=12)
    cbar2.ax.tick_params(labelsize=11)

    ax3 = fig.add_subplot(gs[1, 0])
    vmax_diff = max(abs(cm_diff.min()), abs(cm_diff.max()))
    im3 = ax3.imshow(cm_diff, cmap='RdBu_r', aspect='auto',
                     vmin=-vmax_diff, vmax=vmax_diff)
    acc_diff = acc_rest - acc_task if (acc_rest and acc_task) else None
    title_diff = '(C) Difference (Rest - Task)'
    if acc_diff is not None:
        sign = '+' if acc_diff >= 0 else ''
        title_diff += f'\nΔ Accuracy: {sign}{acc_diff:.2%}'
    ax3.set_title(title_diff, fontweight='bold', fontsize=18, pad=10)
    ax3.set_ylabel('True Network', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Predicted Network', fontweight='bold', fontsize=14)
    ax3.set_xticks(np.arange(len(labels)))
    ax3.set_yticks(np.arange(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=label_fontsize)
    ax3.set_yticklabels(labels, fontsize=label_fontsize)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_diff[i, j]
            if abs(val) > 5.0 or i == j:
                color = 'white' if abs(val) > vmax_diff * 0.5 else 'black'
                ax3.text(j, i, f'{val:+.1f}', ha='center', va='center',
                         color=color, fontsize=fontsize, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Difference (Percentage Points)', fontweight='bold', fontsize=12)
    cbar3.ax.tick_params(labelsize=11)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    diag_rest = np.diag(cm_rest_norm)
    diag_task = np.diag(cm_task_norm)
    diag_diff = diag_rest - diag_task
    n_rest = cm_rest.sum()
    n_task = cm_task.sum()

    summary_text = "(D) SUMMARY STATISTICS\n" + "="*50 + "\n\n"
    if acc_rest is not None and acc_task is not None:
        acc_diff = acc_rest - acc_task
        summary_text += "Overall Accuracy:\n"
        summary_text += f"  Rest: {acc_rest:.2%} (n={n_rest:.0f})\n"
        summary_text += f"  Task: {acc_task:.2%} (n={n_task:.0f})\n"
        sign = '+' if acc_diff >= 0 else ''
        summary_text += f"  Difference: {sign}{acc_diff:.2%}\n\n"
    summary_text += "Diagonal (Correct Classifications):\n"
    summary_text += f"  Rest Mean: {diag_rest.mean():.1f}%\n"
    summary_text += f"  Task Mean: {diag_task.mean():.1f}%\n"
    summary_text += f"  Difference: {diag_diff.mean():+.1f}%\n\n"
    summary_text += "Per-Network Performance:\n"
    summary_text += f"{'Network':<22} {'Rest':<8} {'Task':<8} {'Diff':<8}\n"
    summary_text += "-"*50 + "\n"
    for i, label in enumerate(labels):
        summary_text += (
            f"{label:<22} {diag_rest[i]:>6.1f}% {diag_task[i]:>6.1f}% {diag_diff[i]:>+6.1f}%\n"
        )
    top = np.argsort(diag_diff)[-3:][::-1]
    summary_text += "\nTop 3 Most Improved (Rest > Task):\n"
    for idx in top:
        if diag_diff[idx] > 0:
            summary_text += f"  {labels[idx]}: {diag_diff[idx]:+.1f}%\n"
    top_declined = np.argsort(diag_diff)[:3]
    summary_text += "\nTop 3 Most Declined (Task > Rest):\n"
    for idx in top_declined:
        if diag_diff[idx] < 0:
            summary_text += f"  {labels[idx]}: {diag_diff[idx]:+.1f}%\n"

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontfamily='monospace',
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    )

    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.97)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# ANALYSIS PIPELINE

def analyze_atlas(y_rest, y_rest_pred, y_task, y_task_pred,
                  region_list, mapping, filter_type, exclude,
                  name, tables_dir, figures_dir):
    y_rest_net, y_rest_pred_net, all_labels = aggregate_networks(
        y_rest, y_rest_pred, region_list, mapping
    )
    y_rest_filt, y_rest_pred_filt, labels = filter_networks(
        y_rest_net, y_rest_pred_net, all_labels, filter_type, exclude
    )
    y_task_net, y_task_pred_net, _ = aggregate_networks(
        y_task, y_task_pred, region_list, mapping
    )
    y_task_filt, y_task_pred_filt, _ = filter_networks(
        y_task_net, y_task_pred_net, all_labels, filter_type, exclude
    )
    if len(labels) == 0:
        print(f"No data for {name}")
        return None

    cm_rest = confusion_matrix(y_rest_filt, y_rest_pred_filt, labels=labels)
    cm_task = confusion_matrix(y_task_filt, y_task_pred_filt, labels=labels)

    err_rest = calculate_errors(y_rest_filt, y_rest_pred_filt, labels)
    err_task = calculate_errors(y_task_filt, y_task_pred_filt, labels)

    acc_rest = accuracy_score(y_rest_filt, y_rest_pred_filt)
    acc_task = accuracy_score(y_task_filt, y_task_pred_filt)

    print(f"\n{name}:")
    print(f"  Networks: {len(labels)}")
    print(f"  Rest:  {len(y_rest_filt):4d} samples, {acc_rest:.4f} accuracy")
    print(f"  Task:  {len(y_task_filt):4d} samples, {acc_task:.4f} accuracy")

    pd.DataFrame(cm_rest, index=labels, columns=labels).to_csv(
        tables_dir / f'confusion_matrix_{name}_rest.csv'
    )
    pd.DataFrame(cm_task, index=labels, columns=labels).to_csv(
        tables_dir / f'confusion_matrix_{name}_task.csv'
    )
    err_rest.to_csv(tables_dir / f'error_rates_{name}_rest.csv', index=False)
    err_task.to_csv(tables_dir / f'error_rates_{name}_task.csv', index=False)

    plot_2x2_confusion(
        cm_rest, cm_task, labels,
        f'{name.replace("_", " ").title()}',
        figures_dir / f'confusion_{name}.png',
        acc_rest=acc_rest,
        acc_task=acc_task,
    )

    return {
        'cm_rest': cm_rest,
        'cm_task': cm_task,
        'err_rest': err_rest,
        'err_task': err_task,
        'labels': labels,
        'acc_rest': acc_rest,
        'acc_task': acc_task,
    }


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()

    print_section("ATLAS PERFORMANCE ANALYSIS - IMPROVED 2x2 LAYOUT")

    config = load_config(args.config)
    set_random_seeds(config.get('random_seed', 42))

    print_section("Loading Data")
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']

    if args.sample:
        piop2_file = "data/sample/sample_piop2_small.csv"
        piop1_file = "data/sample/sample_piop1_small.csv"

    df_rest = load_connectivity_data(piop2_file)
    df_task = load_connectivity_data(piop1_file)
    conn_cols = extract_connection_columns(df_rest)

    print_section("Extracting Region Information")
    region_list, region_to_idx, n_regions = extract_regions(conn_cols)
    print(f"Regions: {n_regions}")
    print(f"Rest samples in data: {len(df_rest)}")
    print(f"Task samples in data: {len(df_task)}")

    print_section("Loading Existing Predictions")
    predictions_dir = Path(config.get('output_dirs', {}).get('processed', 'data/processed'))
    pred_rest_path = predictions_dir / 'predictions_train.csv'
    pred_task_path = predictions_dir / 'predictions_task.csv'

    if not pred_rest_path.exists():
        print(f"Rest predictions not found: {pred_rest_path}")
        print("Run the main pipeline first to generate predictions.")
        return 1
    if not pred_task_path.exists():
        print(f"Task predictions not found: {pred_task_path}")
        print("Run the main pipeline first to generate predictions.")
        return 1

    df_pred_rest = pd.read_csv(pred_rest_path)
    df_pred_task = pd.read_csv(pred_task_path)
    print(f"Loaded rest predictions: {len(df_pred_rest)} samples")
    print(f"Loaded task predictions: {len(df_pred_task)} samples")

    y_rest = df_pred_rest['true_region'].values
    y_rest_pred = df_pred_rest['predicted_region'].values
    y_task = df_pred_task['true_region'].values
    y_task_pred = df_pred_task['predicted_region'].values

    print(f"Rest accuracy: {accuracy_score(y_rest, y_rest_pred):.4f}")
    print(f"Task accuracy: {accuracy_score(y_task, y_task_pred):.4f}")

    tables_dir = Path('reports/tables/atlas_analysis')
    figures_dir = Path('reports/figures/atlas_analysis')
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({'region': region_list}).to_csv(
        tables_dir / 'region_list.csv', index=False
    )

    print_section("Running Atlas Analyses")

    results = {}

    print("\n" + "="*70)
    print("N7 CORTICAL (7 Networks)")
    print("="*70)
    mapping_n7 = map_schaefer(region_list, n_networks=7)
    results['N7_cortical'] = analyze_atlas(
        y_rest, y_rest_pred, y_task, y_task_pred,
        region_list, mapping_n7, 'cortical', ['Cortical Other'],
        'N7_cortical', tables_dir, figures_dir,
    )

    print("\n" + "="*70)
    print("N17 CORTICAL (17 Networks)")
    print("="*70)
    mapping_n17 = map_schaefer(region_list, n_networks=17)
    results['N17_cortical'] = analyze_atlas(
        y_rest, y_rest_pred, y_task, y_task_pred,
        region_list, mapping_n17, 'cortical', ['Cortical Other'],
        'N17_cortical', tables_dir, figures_dir,
    )

    print("\n" + "="*70)
    print("TIAN SCALE I (8 Subcortical Regions)")
    print("="*70)
    mapping_tian1 = map_tian(region_list, scale='I')
    results['TianI_subcortical'] = analyze_atlas(
        y_rest, y_rest_pred, y_task, y_task_pred,
        region_list, mapping_tian1, 'subcortical', ['Subcortical Other'],
        'TianI_subcortical', tables_dir, figures_dir,
    )

    print("\n" + "="*70)
    print("TIAN SCALE II (16 Subcortical Regions)")
    print("="*70)
    mapping_tian2 = map_tian(region_list, scale='II')
    results['TianII_subcortical'] = analyze_atlas(
        y_rest, y_rest_pred, y_task, y_task_pred,
        region_list, mapping_tian2, 'subcortical', ['Subcortical Other'],
        'TianII_subcortical', tables_dir, figures_dir,
    )

    print("\n" + "="*70)
    print("COMBINED N7 + TIAN I")
    print("="*70)

    y_rest_n7, y_rest_pred_n7, _ = aggregate_networks(
        y_rest, y_rest_pred, region_list, mapping_n7
    )
    y_rest_n7_c, y_rest_pred_n7_c, n7_labels = filter_networks(
        y_rest_n7, y_rest_pred_n7, sorted(set(mapping_n7.values())),
        'cortical', ['Cortical Other'],
    )
    y_rest_t1, y_rest_pred_t1, _ = aggregate_networks(
        y_rest, y_rest_pred, region_list, mapping_tian1
    )
    y_rest_t1_s, y_rest_pred_t1_s, t1_labels = filter_networks(
        y_rest_t1, y_rest_pred_t1, sorted(set(mapping_tian1.values())),
        'subcortical', ['Subcortical Other'],
    )
    combined_labels = sorted(n7_labels + t1_labels)
    y_rest_comb = np.concatenate([y_rest_n7_c, y_rest_t1_s])
    y_rest_pred_comb = np.concatenate([y_rest_pred_n7_c, y_rest_pred_t1_s])

    y_task_n7, y_task_pred_n7, _ = aggregate_networks(
        y_task, y_task_pred, region_list, mapping_n7
    )
    y_task_n7_c, y_task_pred_n7_c, _ = filter_networks(
        y_task_n7, y_task_pred_n7, sorted(set(mapping_n7.values())),
        'cortical', ['Cortical Other'],
    )
    y_task_t1, y_task_pred_t1, _ = aggregate_networks(
        y_task, y_task_pred, region_list, mapping_tian1
    )
    y_task_t1_s, y_task_pred_t1_s, _ = filter_networks(
        y_task_t1, y_task_pred_t1, sorted(set(mapping_tian1.values())),
        'subcortical', ['Subcortical Other'],
    )
    y_task_comb = np.concatenate([y_task_n7_c, y_task_t1_s])
    y_task_pred_comb = np.concatenate([y_task_pred_n7_c, y_task_pred_t1_s])

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
        figures_dir / 'confusion_N7_TianI_combined.png',
        acc_rest=acc_rest_comb,
        acc_task=acc_task_comb,
    )

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
Generated Files (IMPROVED VERSION):
====================================
Tables:  {tables_dir}
Figures: {figures_dir}
""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
