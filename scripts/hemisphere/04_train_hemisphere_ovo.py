"""
04_train_hemisphere_ovo.py

Train One-vs-One (OvO) binary classifiers for hemisphere-specific brain region classification.
Trains pairwise binary classifiers between selected region pairs to identify confusability patterns.

This reveals which specific region pairs are hard to distinguish from each other.

Usage:
    python scripts/hemisphere/04_train_hemisphere_ovo.py --hemisphere left
    python scripts/hemisphere/04_train_hemisphere_ovo.py --hemisphere right
    python scripts/hemisphere/04_train_hemisphere_ovo.py --hemisphere both --strategy error_driven

Author: Joon
Date: 2024
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import project modules
from src.hemisphere_data.hemisphere_utils import load_hemisphere_data, prepare_classification_data
from src.preprocessing.connectivity_preprocessor import ConnectivityPreprocessor


def setup_logging(output_dir: Path, hemisphere: str) -> logging.Logger:
    """Set up logging configuration."""
    log_file = output_dir / f"training_ovo_{hemisphere}_hemisphere.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train One-vs-One classifiers for selected region pairs'
    )
    
    parser.add_argument(
        '--hemisphere',
        type=str,
        required=True,
        choices=['left', 'right', 'both'],
        help='Which hemisphere to train on (left, right, or both)'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='error_driven',
        choices=['error_driven', 'network_focused', 'spatial', 'all_pairs', 'custom'],
        help='Strategy for selecting region pairs to analyze'
    )
    
    parser.add_argument(
        '--max_pairs',
        type=int,
        default=200,
        help='Maximum number of pairs to analyze (default: 200)'
    )
    
    parser.add_argument(
        '--confusion_threshold',
        type=float,
        default=0.05,
        help='Minimum confusion rate to consider a pair (for error_driven)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=project_root / 'data' / 'processed' / 'hemispheres',
        help='Directory containing hemisphere-specific data'
    )
    
    parser.add_argument(
        '--multinomial_results_dir',
        type=Path,
        default=None,
        help='Directory containing multinomial results (for error_driven strategy)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=project_root / 'data' / 'results' / 'hemisphere_analysis',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    parser.add_argument(
        '--regularization_C',
        type=float,
        default=1.0,
        help='Regularization parameter C for logistic regression'
    )
    
    parser.add_argument(
        '--diagonal_strategy',
        type=str,
        default='region_mean',
        choices=['zero', 'region_mean', 'network_mean', 'global_mean'],
        help='Strategy for handling diagonal values'
    )
    
    parser.add_argument(
        '--max_iter',
        type=int,
        default=1000,
        help='Maximum iterations for logistic regression'
    )
    
    parser.add_argument(
        '--save_models',
        action='store_true',
        help='Save trained models for each pair'
    )
    
    parser.add_argument(
        '--custom_pairs_file',
        type=Path,
        default=None,
        help='CSV file with custom pairs (region_id_1, region_id_2)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    return parser.parse_args()


def identify_error_driven_pairs(
    multinomial_results_dir: Path,
    confusion_threshold: float,
    max_pairs: int,
    logger: logging.Logger
) -> List[Tuple[int, int]]:
    """
    Identify region pairs based on multinomial confusion matrix errors.
    
    Parameters
    ----------
    multinomial_results_dir : Path
        Directory containing multinomial results
    confusion_threshold : float
        Minimum confusion rate to consider
    max_pairs : int
        Maximum number of pairs to return
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    pairs : List[Tuple[int, int]]
        List of (region_i, region_j) tuples
    """
    
    logger.info("Identifying error-driven region pairs...")
    
    # Load confusion matrix
    confusion_matrix_file = multinomial_results_dir / 'confusion_matrix.npy'
    
    if not confusion_matrix_file.exists():
        raise FileNotFoundError(
            f"Confusion matrix not found: {confusion_matrix_file}\n"
            f"Please run multinomial training first."
        )
    
    cm = np.load(confusion_matrix_file)
    n_regions = cm.shape[0]
    
    logger.info(f"  Loaded confusion matrix: {cm.shape}")
    
    # Normalize by row (true class)
    cm_normalized = cm.astype('float')
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm_normalized / row_sums
    
    # Extract off-diagonal confusions
    confusion_pairs = []
    
    for i in range(n_regions):
        for j in range(i + 1, n_regions):  # Only upper triangle
            # Average confusion in both directions
            confusion_rate = (cm_normalized[i, j] + cm_normalized[j, i]) / 2
            
            if confusion_rate >= confusion_threshold:
                # Also include raw counts
                raw_count = cm[i, j] + cm[j, i]
                confusion_pairs.append({
                    'region_i': i,
                    'region_j': j,
                    'confusion_rate': confusion_rate,
                    'raw_count': int(raw_count)
                })
    
    # Sort by confusion rate (descending)
    confusion_pairs.sort(key=lambda x: x['confusion_rate'], reverse=True)
    
    logger.info(f"  Found {len(confusion_pairs)} pairs with confusion >= {confusion_threshold}")
    
    # Limit to max_pairs
    if len(confusion_pairs) > max_pairs:
        logger.info(f"  Limiting to top {max_pairs} most confused pairs")
        confusion_pairs = confusion_pairs[:max_pairs]
    
    # Extract pairs
    pairs = [(p['region_i'], p['region_j']) for p in confusion_pairs]
    
    logger.info(f"  Selected {len(pairs)} pairs for OvO analysis")
    logger.info(f"  Top confusion rate: {confusion_pairs[0]['confusion_rate']:.4f}")
    logger.info(f"  Lowest confusion rate: {confusion_pairs[-1]['confusion_rate']:.4f}")
    
    return pairs, confusion_pairs


def identify_network_focused_pairs(
    region_info: pd.DataFrame,
    max_pairs: int,
    logger: logging.Logger
) -> List[Tuple[int, int]]:
    """
    Identify region pairs within the same functional network.
    
    Parameters
    ----------
    region_info : pd.DataFrame
        Region information with network assignments
    max_pairs : int
        Maximum number of pairs to return
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    pairs : List[Tuple[int, int]]
        List of (region_i, region_j) tuples
    """
    
    logger.info("Identifying network-focused region pairs...")
    
    if 'network' not in region_info.columns:
        logger.warning("No 'network' column in region_info. Cannot use network_focused strategy.")
        return []
    
    # Group regions by network
    networks = region_info.groupby('network')['region_id'].apply(list).to_dict()
    
    pairs = []
    pair_info = []
    
    for network, region_ids in networks.items():
        # All pairs within this network
        for i, region_i in enumerate(region_ids):
            for region_j in region_ids[i + 1:]:
                pairs.append((region_i, region_j))
                pair_info.append({
                    'region_i': region_i,
                    'region_j': region_j,
                    'network': network
                })
    
    logger.info(f"  Found {len(pairs)} within-network pairs")
    
    # Limit to max_pairs (randomly sample if too many)
    if len(pairs) > max_pairs:
        logger.info(f"  Randomly sampling {max_pairs} pairs")
        np.random.seed(42)
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]
        pair_info = [pair_info[i] for i in indices]
    
    logger.info(f"  Selected {len(pairs)} pairs for OvO analysis")
    
    return pairs, pair_info


def identify_all_pairs(
    n_regions: int,
    max_pairs: int,
    logger: logging.Logger
) -> List[Tuple[int, int]]:
    """
    Generate all possible region pairs (exhaustive OvO).
    
    Parameters
    ----------
    n_regions : int
        Number of regions
    max_pairs : int
        Maximum number of pairs (will sample if too many)
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    pairs : List[Tuple[int, int]]
        List of (region_i, region_j) tuples
    """
    
    logger.info("Generating all possible region pairs...")
    
    # All combinations
    total_pairs = n_regions * (n_regions - 1) // 2
    logger.info(f"  Total possible pairs: {total_pairs}")
    
    pairs = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            pairs.append((i, j))
    
    # Sample if too many
    if len(pairs) > max_pairs:
        logger.warning(
            f"  {total_pairs} pairs exceed max_pairs={max_pairs}. "
            f"Randomly sampling {max_pairs} pairs."
        )
        np.random.seed(42)
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]
    
    logger.info(f"  Selected {len(pairs)} pairs for OvO analysis")
    
    pair_info = [{'region_i': i, 'region_j': j} for i, j in pairs]
    
    return pairs, pair_info


def load_custom_pairs(
    custom_pairs_file: Path,
    logger: logging.Logger
) -> List[Tuple[int, int]]:
    """
    Load custom pairs from CSV file.
    
    Parameters
    ----------
    custom_pairs_file : Path
        Path to CSV with columns: region_id_1, region_id_2
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    pairs : List[Tuple[int, int]]
        List of (region_i, region_j) tuples
    """
    
    logger.info(f"Loading custom pairs from: {custom_pairs_file}")
    
    df = pd.read_csv(custom_pairs_file)
    
    required_cols = ['region_id_1', 'region_id_2']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Custom pairs file must have columns: {required_cols}")
    
    pairs = list(zip(df['region_id_1'].values, df['region_id_2'].values))
    
    logger.info(f"  Loaded {len(pairs)} custom pairs")
    
    pair_info = [{'region_i': i, 'region_j': j} for i, j in pairs]
    
    return pairs, pair_info


def train_pairwise_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    region_i: int,
    region_j: int,
    C: float,
    max_iter: int,
    random_state: int
) -> Dict:
    """
    Train binary classifier for a specific region pair.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    region_i, region_j : int
        Region IDs for the pair
    C : float
        Regularization parameter
    max_iter : int
        Maximum iterations
    random_state : int
        Random state
    
    Returns
    -------
    results : dict
        Dictionary with pair metrics
    """
    
    # Filter data to only include these two regions
    train_mask = np.isin(y_train, [region_i, region_j])
    test_mask = np.isin(y_test, [region_i, region_j])
    
    X_train_pair = X_train[train_mask]
    y_train_pair = y_train[train_mask]
    X_test_pair = X_test[test_mask]
    y_test_pair = y_test[test_mask]
    
    # Check if both classes are present
    if len(np.unique(y_train_pair)) < 2 or len(np.unique(y_test_pair)) < 2:
        return {
            'region_i': region_i,
            'region_j': region_j,
            'accuracy': np.nan,
            'n_train': len(y_train_pair),
            'n_test': len(y_test_pair),
            'error': 'insufficient_classes'
        }
    
    # Train binary classifier
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs'
    )
    
    try:
        classifier.fit(X_train_pair, y_train_pair)
        
        # Predict
        y_pred = classifier.predict(X_test_pair)
        y_proba = classifier.predict_proba(X_test_pair)
        
        # Compute metrics
        accuracy = accuracy_score(y_test_pair, y_pred)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test_pair, y_proba[:, 1])
        except:
            roc_auc = np.nan
        
        # Confusion for this pair
        i_as_j = np.sum((y_test_pair == region_i) & (y_pred == region_j))
        j_as_i = np.sum((y_test_pair == region_j) & (y_pred == region_i))
        
        results = {
            'region_i': region_i,
            'region_j': region_j,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'n_train': len(y_train_pair),
            'n_test': len(y_test_pair),
            'i_as_j': int(i_as_j),
            'j_as_i': int(j_as_i),
            'total_confusion': int(i_as_j + j_as_i),
            'classifier': classifier
        }
        
    except Exception as e:
        results = {
            'region_i': region_i,
            'region_j': region_j,
            'accuracy': np.nan,
            'n_train': len(y_train_pair),
            'n_test': len(y_test_pair),
            'error': str(e)
        }
    
    return results


def train_ovo_single_hemisphere(
    hemisphere: str,
    args: argparse.Namespace,
    logger: logging.Logger
) -> dict:
    """
    Train OvO classifiers for a single hemisphere.
    
    Parameters
    ----------
    hemisphere : str
        'left' or 'right'
    args : argparse.Namespace
        Command line arguments
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    results : dict
        Dictionary containing all results
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING ONE-VS-ONE CLASSIFIERS - {hemisphere.upper()} HEMISPHERE")
    logger.info(f"{'='*80}\n")
    
    # Create output directory
    output_dir = args.output_dir / f"{hemisphere}_hemisphere" / "ovo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading hemisphere-specific data...")
    data = load_hemisphere_data(
        data_dir=args.data_dir,
        hemisphere=hemisphere,
        dataset='rest'
    )
    
    connectivity = data['connectivity']
    region_info = data['region_info']
    subject_ids = data['subject_ids']
    
    n_subjects = data['n_subjects']
    n_regions = data['n_regions']
    
    logger.info(f"Data loaded:")
    logger.info(f"  Subjects: {n_subjects}")
    logger.info(f"  Regions: {n_regions}")
    
    # Prepare classification data
    logger.info("\nPreparing classification data...")
    X, y, groups = prepare_classification_data(
        connectivity=connectivity,
        region_info=region_info,
        subject_ids=subject_ids
    )
    
    # Identify pairs to analyze
    logger.info(f"\nIdentifying region pairs using '{args.strategy}' strategy...")
    
    if args.strategy == 'error_driven':
        # Need multinomial results
        if args.multinomial_results_dir is None:
            multinomial_dir = args.output_dir / f"{hemisphere}_hemisphere" / "multinomial"
        else:
            multinomial_dir = args.multinomial_results_dir
        
        pairs, pair_info = identify_error_driven_pairs(
            multinomial_dir,
            args.confusion_threshold,
            args.max_pairs,
            logger
        )
    
    elif args.strategy == 'network_focused':
        pairs, pair_info = identify_network_focused_pairs(
            region_info,
            args.max_pairs,
            logger
        )
    
    elif args.strategy == 'all_pairs':
        pairs, pair_info = identify_all_pairs(
            n_regions,
            args.max_pairs,
            logger
        )
    
    elif args.strategy == 'custom':
        if args.custom_pairs_file is None:
            raise ValueError("--custom_pairs_file required for custom strategy")
        pairs, pair_info = load_custom_pairs(args.custom_pairs_file, logger)
    
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    if len(pairs) == 0:
        logger.error("No pairs selected. Exiting.")
        sys.exit(1)
    
    # Save selected pairs
    pairs_df = pd.DataFrame(pair_info)
    pairs_df.to_csv(output_dir / 'selected_pairs.csv', index=False)
    
    # Set up cross-validation
    logger.info(f"\nSetting up {args.n_folds}-fold GroupKFold cross-validation...")
    gkf = GroupKFold(n_splits=args.n_folds)
    
    # Storage for results
    all_pair_results = []
    
    # Cross-validation loop
    logger.info(f"\nStarting cross-validation with OvO training for {len(pairs)} pairs...\n")
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        fold_start = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"{'='*60}")
        
        # Split data
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Preprocess
        logger.info("  Preprocessing data...")
        preprocessor = ConnectivityPreprocessor(
            diagonal_strategy=args.diagonal_strategy,
            apply_fisher_z=True,
            standardize=True,
            region_info=region_info
        )
        
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train pairwise classifiers
        logger.info(f"  Training {len(pairs)} pairwise classifiers...")
        
        for pair_idx, (region_i, region_j) in enumerate(pairs):
            if pair_idx % 50 == 0 and pair_idx > 0:
                elapsed = time.time() - fold_start
                estimated_total = elapsed * len(pairs) / pair_idx
                remaining = estimated_total - elapsed
                logger.info(f"    Progress: {pair_idx}/{len(pairs)} pairs "
                          f"(~{remaining:.1f}s remaining)")
            
            # Train pairwise classifier
            pair_results = train_pairwise_classifier(
                X_train_processed, y_train,
                X_test_processed, y_test,
                region_i, region_j,
                args.regularization_C,
                args.max_iter,
                args.random_state
            )
            
            pair_results['fold'] = fold_idx + 1
            
            # Don't save classifier in results list (too large)
            if 'classifier' in pair_results:
                del pair_results['classifier']
            
            all_pair_results.append(pair_results)
        
        fold_time = time.time() - fold_start
        logger.info(f"  Fold completed in {fold_time:.2f}s\n")
    
    total_time = time.time() - start_time
    logger.info(f"OvO cross-validation completed in {total_time:.2f}s\n")
    
    # Aggregate results
    logger.info("Aggregating pairwise results across folds...")
    results_df = pd.DataFrame(all_pair_results)
    
    # Compute mean metrics per pair
    pair_summary = results_df.groupby(['region_i', 'region_j']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'roc_auc': 'mean',
        'total_confusion': 'sum',
        'n_test': 'sum'
    }).reset_index()
    
    # Flatten column names
    pair_summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in pair_summary.columns.values]
    
    # Rename for clarity
    pair_summary = pair_summary.rename(columns={
        'accuracy_mean': 'mean_accuracy',
        'accuracy_std': 'std_accuracy',
        'accuracy_count': 'n_folds',
        'roc_auc_mean': 'mean_roc_auc',
        'total_confusion_sum': 'total_confusion',
        'n_test_sum': 'total_samples'
    })
    
    # Add region names if available
    if 'region_name' in region_info.columns:
        region_names = dict(zip(region_info['region_id'], region_info['region_name']))
        pair_summary['region_i_name'] = pair_summary['region_i'].map(region_names)
        pair_summary['region_j_name'] = pair_summary['region_j'].map(region_names)
    
    # Add network info if available
    if 'network' in region_info.columns:
        region_networks = dict(zip(region_info['region_id'], region_info['network']))
        pair_summary['region_i_network'] = pair_summary['region_i'].map(region_networks)
        pair_summary['region_j_network'] = pair_summary['region_j'].map(region_networks)
        pair_summary['same_network'] = (
            pair_summary['region_i_network'] == pair_summary['region_j_network']
        )
    
    # Sort by accuracy (ascending = hardest pairs first)
    pair_summary = pair_summary.sort_values('mean_accuracy')
    
    # Summary statistics
    logger.info(f"\nOvO PAIRWISE RESULTS ({hemisphere.upper()} HEMISPHERE):")
    logger.info(f"  Pairs analyzed: {len(pair_summary)}")
    logger.info(f"  Mean pairwise accuracy: {pair_summary['mean_accuracy'].mean():.4f}")
    logger.info(f"  Std pairwise accuracy: {pair_summary['mean_accuracy'].std():.4f}")
    logger.info(f"  Easiest pair accuracy: {pair_summary['mean_accuracy'].max():.4f}")
    logger.info(f"  Hardest pair accuracy: {pair_summary['mean_accuracy'].min():.4f}")
    
    # Most confused pairs
    logger.info(f"\nMOST CONFUSABLE PAIRS (Lowest Accuracy - Top 10):")
    for idx, row in pair_summary.head(10).iterrows():
        region_i_name = row.get('region_i_name', f"Region {row['region_i']}")
        region_j_name = row.get('region_j_name', f"Region {row['region_j']}")
        logger.info(f"  {region_i_name} <-> {region_j_name}: {row['mean_accuracy']:.4f}")
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save all fold results
    results_df.to_csv(output_dir / 'ovo_all_fold_results.csv', index=False)
    
    # Save pair summary
    pair_summary.to_csv(output_dir / 'ovo_pair_summary.csv', index=False)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # 1. Pairwise accuracy distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pair_summary['mean_accuracy'], bins=30, color='steelblue', 
            alpha=0.7, edgecolor='black')
    ax.axvline(pair_summary['mean_accuracy'].mean(), color='red', 
               linestyle='--', linewidth=2, 
               label=f"Mean: {pair_summary['mean_accuracy'].mean():.3f}")
    ax.set_xlabel('Pairwise Classification Accuracy', fontweight='bold')
    ax.set_ylabel('Number of Pairs', fontweight='bold')
    ax.set_title(f'{hemisphere.capitalize()} Hemisphere - OvO Pairwise Accuracy Distribution',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'ovo_accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Within vs between network comparison (if available)
    if 'same_network' in pair_summary.columns:
        within_network = pair_summary[pair_summary['same_network'] == True]['mean_accuracy']
        between_network = pair_summary[pair_summary['same_network'] == False]['mean_accuracy']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = [within_network.values, between_network.values]
        labels = [f'Within Network\n(n={len(within_network)})', 
                 f'Between Network\n(n={len(between_network)})']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True)
        
        # Color boxes
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Pairwise Classification Accuracy', fontweight='bold')
        ax.set_title(f'{hemisphere.capitalize()} Hemisphere - Within vs Between Network Confusability',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(within_network, between_network)
        
        ax.text(0.02, 0.98, 
                f'Mann-Whitney U test:\np = {p_value:.4f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ovo_within_vs_between_network.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\nWithin vs Between Network:")
        logger.info(f"  Within network mean accuracy: {within_network.mean():.4f}")
        logger.info(f"  Between network mean accuracy: {between_network.mean():.4f}")
        logger.info(f"  Mann-Whitney U p-value: {p_value:.4f}")
    
    # 3. Confusion network graph
    logger.info("\nCreating confusion network graph...")
    create_confusion_network(
        pair_summary,
        output_dir / 'ovo_confusion_network.png',
        accuracy_threshold=0.75,  # Show pairs with accuracy < 0.75
        hemisphere=hemisphere,
        region_info=region_info
    )
    
    logger.info(f"All results saved to: {output_dir}")
    
    # Prepare return dictionary
    results = {
        'hemisphere': hemisphere,
        'n_pairs': len(pairs),
        'pair_summary': pair_summary,
        'all_results': results_df,
        'output_dir': output_dir
    }
    
    return results


def create_confusion_network(
    pair_summary: pd.DataFrame,
    save_path: Path,
    accuracy_threshold: float,
    hemisphere: str,
    region_info: pd.DataFrame
):
    """
    Create network graph showing confused region pairs.
    
    Parameters
    ----------
    pair_summary : pd.DataFrame
        Summary of pairwise results
    save_path : Path
        Path to save figure
    accuracy_threshold : float
        Only show pairs below this accuracy
    hemisphere : str
        Hemisphere name
    region_info : pd.DataFrame
        Region information
    """
    
    # Filter for confused pairs
    confused_pairs = pair_summary[pair_summary['mean_accuracy'] < accuracy_threshold].copy()
    
    if len(confused_pairs) == 0:
        print(f"  No pairs below accuracy threshold {accuracy_threshold}")
        return
    
    print(f"  Creating network with {len(confused_pairs)} confused pairs")
    
    # Create graph
    G = nx.Graph()
    
    # Add edges (confused pairs)
    for _, row in confused_pairs.iterrows():
        weight = 1.0 - row['mean_accuracy']  # Higher weight = more confusion
        G.add_edge(row['region_i'], row['region_j'], weight=weight)
    
    # Node colors by network
    if 'network' in region_info.columns:
        networks = region_info['network'].unique()
        network_colors = dict(zip(networks, plt.cm.tab10(np.linspace(0, 1, len(networks)))))
        
        region_to_network = dict(zip(region_info['region_id'], region_info['network']))
        node_colors = [network_colors[region_to_network.get(node, 'Unknown')] 
                      for node in G.nodes()]
    else:
        node_colors = 'lightblue'
    
    # Draw graph
    fig, ax = plt.subplots(figsize=(14, 14))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges with width proportional to confusion
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos,
        width=[w * 5 for w in weights],
        alpha=0.5,
        edge_color='gray'
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=500,
        alpha=0.8,
        edgecolors='black',
        linewidths=1.5
    )
    
    # Draw labels (only for high-degree nodes to avoid clutter)
    degrees = dict(G.degree())
    high_degree_nodes = {node for node, deg in degrees.items() if deg >= 3}
    
    if 'region_name' in region_info.columns:
        region_names = dict(zip(region_info['region_id'], region_info['region_name']))
        labels = {node: region_names.get(node, str(node)) 
                 for node in high_degree_nodes}
    else:
        labels = {node: str(node) for node in high_degree_nodes}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    ax.set_title(
        f'{hemisphere.capitalize()} Hemisphere - Confusion Network\n'
        f'(Pairs with accuracy < {accuracy_threshold})',
        fontweight='bold',
        fontsize=14,
        pad=20
    )
    ax.axis('off')
    
    # Add legend for networks if available
    if 'network' in region_info.columns:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10, label=network)
            for network, color in network_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 frameon=True, title='Networks')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved confusion network: {save_path}")


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output_dir, args.hemisphere)
    
    logger.info("="*80)
    logger.info("ONE-VS-ONE HEMISPHERE-SPECIFIC CLASSIFICATION")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Hemisphere: {args.hemisphere}")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Max pairs: {args.max_pairs}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    try:
        # Train based on hemisphere argument
        if args.hemisphere == 'both':
            # Train both hemispheres
            left_results = train_ovo_single_hemisphere('left', args, logger)
            right_results = train_ovo_single_hemisphere('right', args, logger)
        else:
            # Train single hemisphere
            results = train_ovo_single_hemisphere(args.hemisphere, args, logger)
        
        logger.info("\n" + "="*80)
        logger.info("OVO TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()