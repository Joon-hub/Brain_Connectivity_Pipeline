"""
03_train_hemisphere_ovr.py

Train One-vs-Rest (OvR) binary classifiers for hemisphere-specific brain region classification.
Each region gets its own binary classifier that distinguishes it from all other regions.

This reveals which regions have uniquely discriminative connectivity patterns.

Usage:
    python scripts/hemisphere/03_train_hemisphere_ovr.py --hemisphere left
    python scripts/hemisphere/03_train_hemisphere_ovr.py --hemisphere right
    python scripts/hemisphere/03_train_hemisphere_ovr.py --hemisphere both

Author: Joon
Date: 2024
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import project modules
from src.hemisphere_data.hemisphere_utils import load_hemisphere_data, prepare_classification_data
from src.preprocessing.connectivity_preprocessor import ConnectivityPreprocessor
from src.evaluation.hemisphere_metrics import compute_classification_metrics
from src.visualization.hemisphere_viz import plot_per_region_accuracy


def setup_logging(output_dir: Path, hemisphere: str) -> logging.Logger:
    """Set up logging configuration."""
    log_file = output_dir / f"training_ovr_{hemisphere}_hemisphere.log"
    
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
        description='Train One-vs-Rest classifiers for hemisphere-specific classification'
    )
    
    parser.add_argument(
        '--hemisphere',
        type=str,
        required=True,
        choices=['left', 'right', 'both'],
        help='Which hemisphere to train on (left, right, or both)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=project_root / 'data' / 'processed' / 'hemispheres',
        help='Directory containing hemisphere-specific data'
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
        '--class_weight',
        type=str,
        default='balanced',
        choices=['balanced', 'none'],
        help='Class weighting strategy for handling imbalance'
    )
    
    parser.add_argument(
        '--max_iter',
        type=int,
        default=1000,
        help='Maximum iterations for logistic regression'
    )
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 uses all cores)'
    )
    
    parser.add_argument(
        '--save_models',
        action='store_true',
        help='Save trained models from each fold'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    return parser.parse_args()


class OneVsRestClassifier:
    """
    One-vs-Rest classifier for brain region classification.
    
    Trains separate binary classifiers for each region, where each classifier
    learns to distinguish one region from all others.
    """
    
    def __init__(
        self,
        n_regions: int,
        C: float = 1.0,
        class_weight: str = 'balanced',
        max_iter: int = 1000,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_regions = n_regions
        self.C = C
        self.class_weight = class_weight if class_weight != 'none' else None
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Storage for binary classifiers
        self.classifiers = {}
        
        # Storage for per-region metrics
        self.region_metrics = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, region_id: int):
        """
        Fit binary classifier for a specific region.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Original labels
        region_id : int
            ID of the region to train classifier for
        
        Returns
        -------
        classifier : LogisticRegression
            Fitted binary classifier
        """
        
        # Create binary labels (1 if region_id, 0 otherwise)
        y_binary = (y == region_id).astype(int)
        
        # Train binary classifier
        classifier = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=1,  # Parallel over regions, not within classifier
            solver='lbfgs'
        )
        
        classifier.fit(X, y_binary)
        
        return classifier
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for all regions using all binary classifiers.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        
        Returns
        -------
        probabilities : np.ndarray
            Probability matrix, shape (n_samples, n_regions)
        """
        
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_regions))
        
        for region_id, classifier in self.classifiers.items():
            # Get probability of being this region (positive class)
            probabilities[:, region_id] = classifier.predict_proba(X)[:, 1]
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict region labels using maximum probability from binary classifiers.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        
        Returns
        -------
        predictions : np.ndarray
            Predicted region labels
        """
        
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions


def train_ovr_single_hemisphere(
    hemisphere: str,
    args: argparse.Namespace,
    logger: logging.Logger
) -> dict:
    """
    Train OvR classifiers for a single hemisphere.
    
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
        Dictionary containing all results and metrics
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING ONE-VS-REST CLASSIFIERS - {hemisphere.upper()} HEMISPHERE")
    logger.info(f"{'='*80}\n")
    
    # Create output directory
    output_dir = args.output_dir / f"{hemisphere}_hemisphere" / "ovr"
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
    logger.info(f"  Connectivity shape: {connectivity.shape}")
    
    # Prepare classification data
    logger.info("\nPreparing classification data...")
    X, y, groups = prepare_classification_data(
        connectivity=connectivity,
        region_info=region_info,
        subject_ids=subject_ids
    )
    
    logger.info(f"Classification data:")
    logger.info(f"  Features (X): {X.shape}")
    logger.info(f"  Labels (y): {y.shape}")
    logger.info(f"  Groups: {groups.shape}")
    logger.info(f"  Unique labels: {len(np.unique(y))}")
    logger.info(f"  Unique subjects: {len(np.unique(groups))}")
    
    # Set up cross-validation
    logger.info(f"\nSetting up {args.n_folds}-fold GroupKFold cross-validation...")
    gkf = GroupKFold(n_splits=args.n_folds)
    
    # Initialize storage for results
    all_fold_results = []
    per_region_fold_metrics = {region_id: [] for region_id in range(n_regions)}
    
    fold_ovr_classifiers = [] if args.save_models else None
    
    # Storage for aggregated predictions
    all_predictions = np.zeros(len(y), dtype=int)
    all_probabilities = np.zeros((len(y), n_regions))
    all_true_labels = np.zeros(len(y), dtype=int)
    sample_to_fold = np.zeros(len(y), dtype=int)
    
    # Cross-validation loop
    logger.info("\nStarting cross-validation with OvR training...\n")
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        fold_start = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"{'='*60}")
        logger.info(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Split data
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Verify no subject leakage
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        assert len(train_subjects.intersection(test_subjects)) == 0, "Subject leakage detected!"
        
        logger.info(f"  Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
        
        # Preprocess data (within fold to prevent leakage)
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
        
        logger.info(f"    Processed train shape: {X_train_processed.shape}")
        logger.info(f"    Processed test shape: {X_test_processed.shape}")
        
        # Initialize OvR classifier
        ovr_classifier = OneVsRestClassifier(
            n_regions=n_regions,
            C=args.regularization_C,
            class_weight=args.class_weight,
            max_iter=args.max_iter,
            random_state=args.random_state,
            n_jobs=args.n_jobs
        )
        
        # Train binary classifier for each region
        logger.info(f"  Training {n_regions} binary classifiers...")
        training_start = time.time()
        
        for region_id in range(n_regions):
            if region_id % 20 == 0 and region_id > 0:
                elapsed = time.time() - training_start
                estimated_total = elapsed * n_regions / region_id
                remaining = estimated_total - elapsed
                logger.info(f"    Progress: {region_id}/{n_regions} classifiers "
                          f"(~{remaining:.1f}s remaining)")
            
            # Train binary classifier
            classifier = ovr_classifier.fit(X_train_processed, y_train, region_id)
            ovr_classifier.classifiers[region_id] = classifier
            
            # Compute binary metrics on training set (for monitoring)
            y_binary_train = (y_train == region_id).astype(int)
            y_binary_pred_train = classifier.predict(X_train_processed)
            train_acc = accuracy_score(y_binary_train, y_binary_pred_train)
            
            # Store for debugging if very low
            if train_acc < 0.6:
                logger.warning(f"      Region {region_id}: Low training accuracy {train_acc:.3f}")
        
        training_time = time.time() - training_start
        logger.info(f"  Binary classifiers trained in {training_time:.2f}s")
        
        # Predict on test set using OvR ensemble
        logger.info("  Making predictions on test set...")
        y_pred = ovr_classifier.predict(X_test_processed)
        y_proba = ovr_classifier.predict_proba(X_test_processed)
        
        # Compute overall fold metrics
        fold_acc = accuracy_score(y_test, y_pred)
        logger.info(f"  Fold accuracy: {fold_acc:.4f}")
        
        # Compute per-region binary metrics
        logger.info("  Computing per-region binary metrics...")
        for region_id in range(n_regions):
            # Binary labels
            y_binary_test = (y_test == region_id).astype(int)
            y_binary_pred = (y_pred == region_id).astype(int)
            
            # Get decision scores
            y_scores = y_proba[:, region_id]
            
            # Compute metrics
            binary_metrics = {
                'fold': fold_idx + 1,
                'region_id': region_id,
                'accuracy': accuracy_score(y_binary_test, y_binary_pred),
                'precision': precision_score(y_binary_test, y_binary_pred, zero_division=0),
                'recall': recall_score(y_binary_test, y_binary_pred, zero_division=0),
                'f1_score': f1_score(y_binary_test, y_binary_pred, zero_division=0),
                'n_positive': int(np.sum(y_binary_test)),
                'n_negative': int(len(y_binary_test) - np.sum(y_binary_test))
            }
            
            # ROC AUC (if both classes present)
            if len(np.unique(y_binary_test)) > 1:
                try:
                    binary_metrics['roc_auc'] = roc_auc_score(y_binary_test, y_scores)
                    binary_metrics['avg_precision'] = average_precision_score(y_binary_test, y_scores)
                except Exception as e:
                    logger.warning(f"    Could not compute AUC for region {region_id}: {e}")
                    binary_metrics['roc_auc'] = np.nan
                    binary_metrics['avg_precision'] = np.nan
            else:
                binary_metrics['roc_auc'] = np.nan
                binary_metrics['avg_precision'] = np.nan
            
            per_region_fold_metrics[region_id].append(binary_metrics)
        
        # Store fold results
        fold_results = {
            'fold': fold_idx + 1,
            'accuracy': fold_acc,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'training_time': training_time
        }
        all_fold_results.append(fold_results)
        
        # Store predictions for aggregation
        all_predictions[test_idx] = y_pred
        all_probabilities[test_idx] = y_proba
        all_true_labels[test_idx] = y_test
        sample_to_fold[test_idx] = fold_idx + 1
        
        # Save fold models if requested
        if args.save_models:
            fold_ovr_classifiers.append({
                'fold': fold_idx + 1,
                'classifiers': ovr_classifier.classifiers.copy(),
                'train_idx': train_idx,
                'test_idx': test_idx
            })
        
        fold_time = time.time() - fold_start
        logger.info(f"  Fold completed in {fold_time:.2f}s\n")
    
    total_time = time.time() - start_time
    logger.info(f"Cross-validation completed in {total_time:.2f}s\n")
    
    # Aggregate per-region metrics across folds
    logger.info("Aggregating per-region metrics across folds...")
    per_region_summary = []
    
    for region_id in range(n_regions):
        fold_metrics = per_region_fold_metrics[region_id]
        
        region_summary = {
            'region_id': region_id,
            'mean_accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
            'mean_precision': np.mean([m['precision'] for m in fold_metrics]),
            'mean_recall': np.mean([m['recall'] for m in fold_metrics]),
            'mean_f1_score': np.mean([m['f1_score'] for m in fold_metrics]),
            'mean_roc_auc': np.nanmean([m['roc_auc'] for m in fold_metrics]),
            'total_positive': sum([m['n_positive'] for m in fold_metrics]),
            'total_negative': sum([m['n_negative'] for m in fold_metrics])
        }
        
        per_region_summary.append(region_summary)
    
    per_region_df = pd.DataFrame(per_region_summary)
    
    # Add region names and network info
    if 'region_name' in region_info.columns:
        per_region_df = per_region_df.merge(
            region_info[['region_id', 'region_name', 'network']],
            on='region_id',
            how='left'
        )
    
    # Sort by mean accuracy
    per_region_df = per_region_df.sort_values('mean_accuracy', ascending=False)
    
    # Compute overall metrics using aggregated predictions
    logger.info("Computing overall OvR metrics...")
    overall_metrics = compute_classification_metrics(
        y_true=all_true_labels,
        y_pred=all_predictions,
        y_proba=all_probabilities
    )
    
    logger.info(f"\nOVERALL OvR RESULTS ({hemisphere.upper()} HEMISPHERE):")
    logger.info(f"  Mean CV Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"  Mean CV Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Top-5 Accuracy: {overall_metrics.get('top_5_accuracy', 'N/A')}")
    
    logger.info(f"\nPER-REGION BINARY METRICS:")
    logger.info(f"  Mean per-region accuracy: {per_region_df['mean_accuracy'].mean():.4f}")
    logger.info(f"  Std per-region accuracy: {per_region_df['mean_accuracy'].std():.4f}")
    logger.info(f"  Best region accuracy: {per_region_df['mean_accuracy'].max():.4f}")
    logger.info(f"  Worst region accuracy: {per_region_df['mean_accuracy'].min():.4f}")
    
    # Identify most and least discriminable regions
    logger.info(f"\nMOST DISCRIMINABLE REGIONS (Top 5):")
    for idx, row in per_region_df.head(5).iterrows():
        region_name = row.get('region_name', f"Region {row['region_id']}")
        logger.info(f"  {region_name}: {row['mean_accuracy']:.4f}")
    
    logger.info(f"\nLEAST DISCRIMINABLE REGIONS (Bottom 5):")
    for idx, row in per_region_df.tail(5).iterrows():
        region_name = row.get('region_name', f"Region {row['region_id']}")
        logger.info(f"  {region_name}: {row['mean_accuracy']:.4f}")
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save predictions
    np.save(output_dir / 'ovr_predictions.npy', all_predictions)
    np.save(output_dir / 'ovr_probabilities.npy', all_probabilities)
    np.save(output_dir / 'ovr_true_labels.npy', all_true_labels)
    np.save(output_dir / 'ovr_fold_indices.npy', sample_to_fold)
    
    # Save overall metrics
    with open(output_dir / 'ovr_overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    # Save fold metrics
    with open(output_dir / 'ovr_fold_metrics.json', 'w') as f:
        json.dump(all_fold_results, f, indent=2)
    
    # Save per-region metrics
    per_region_df.to_csv(output_dir / 'ovr_per_region_metrics.csv', index=False)
    
    # Save detailed per-region per-fold metrics
    detailed_metrics = []
    for region_id in range(n_regions):
        for fold_metric in per_region_fold_metrics[region_id]:
            detailed_metrics.append(fold_metric)
    
    detailed_df = pd.DataFrame(detailed_metrics)
    detailed_df.to_csv(output_dir / 'ovr_per_region_per_fold_metrics.csv', index=False)
    
    # Save models if requested
    if args.save_models and fold_ovr_classifiers is not None:
        import pickle
        with open(output_dir / 'ovr_fold_classifiers.pkl', 'wb') as f:
            pickle.dump(fold_ovr_classifiers, f)
        logger.info("OvR classifiers saved")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # Plot per-region discriminability
    plot_per_region_accuracy(
        per_region_metrics=per_region_df.rename(columns={'mean_accuracy': 'accuracy'}),
        save_path=output_dir / 'ovr_discriminability_map.png',
        title=f'{hemisphere.capitalize()} Hemisphere - OvR Binary Discriminability',
        color_by_network=True
    )
    
    # Create comparison plot: OvR accuracy distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(per_region_df['mean_accuracy'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(per_region_df['mean_accuracy'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {per_region_df['mean_accuracy'].mean():.3f}")
    ax.set_xlabel('Binary Classification Accuracy', fontweight='bold')
    ax.set_ylabel('Number of Regions', fontweight='bold')
    ax.set_title(f'{hemisphere.capitalize()} Hemisphere - OvR Discriminability Distribution', 
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'ovr_discriminability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Network-level discriminability
    if 'network' in per_region_df.columns:
        network_discriminability = per_region_df.groupby('network')['mean_accuracy'].agg(['mean', 'std', 'count'])
        network_discriminability = network_discriminability.sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(network_discriminability)))
        ax.bar(range(len(network_discriminability)), network_discriminability['mean'],
               yerr=network_discriminability['std'], color=colors, alpha=0.8, 
               edgecolor='black', linewidth=1.5, capsize=5)
        ax.set_xticks(range(len(network_discriminability)))
        ax.set_xticklabels(network_discriminability.index, rotation=45, ha='right')
        ax.set_ylabel('Mean Binary Accuracy', fontweight='bold')
        ax.set_xlabel('Functional Network', fontweight='bold')
        ax.set_title(f'{hemisphere.capitalize()} Hemisphere - OvR Discriminability by Network',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_dir / 'ovr_network_discriminability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save network summary
        network_discriminability.to_csv(output_dir / 'ovr_network_discriminability.csv')
    
    logger.info(f"All results saved to: {output_dir}")
    
    # Prepare return dictionary
    results = {
        'hemisphere': hemisphere,
        'n_subjects': n_subjects,
        'n_regions': n_regions,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'true_labels': all_true_labels,
        'overall_metrics': overall_metrics,
        'fold_metrics': all_fold_results,
        'per_region_metrics': per_region_df,
        'output_dir': output_dir
    }
    
    return results


def compare_ovr_to_multinomial(
    ovr_results: dict,
    multinomial_dir: Path,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Compare OvR results to multinomial baseline.
    
    Parameters
    ----------
    ovr_results : dict
        Results from OvR training
    multinomial_dir : Path
        Directory containing multinomial results
    output_dir : Path
        Output directory for comparison
    logger : logging.Logger
        Logger instance
    """
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPARING OvR TO MULTINOMIAL BASELINE")
    logger.info(f"{'='*80}\n")
    
    # Load multinomial results
    try:
        with open(multinomial_dir / 'overall_metrics.json', 'r') as f:
            multinomial_metrics = json.load(f)
        
        multinomial_per_region = pd.read_csv(multinomial_dir / 'per_region_metrics.csv')
    except FileNotFoundError:
        logger.warning("Multinomial results not found. Skipping comparison.")
        return
    
    # Compare overall accuracy
    ovr_acc = ovr_results['overall_metrics']['accuracy']
    multi_acc = multinomial_metrics['accuracy']
    
    logger.info(f"Overall Accuracy Comparison:")
    logger.info(f"  Multinomial: {multi_acc:.4f}")
    logger.info(f"  OvR: {ovr_acc:.4f}")
    logger.info(f"  Difference: {ovr_acc - multi_acc:+.4f}")
    
    # Per-region comparison
    ovr_per_region = ovr_results['per_region_metrics']
    
    # Merge on region_id
    comparison = pd.merge(
        multinomial_per_region[['region_id', 'accuracy']],
        ovr_per_region[['region_id', 'mean_accuracy']],
        on='region_id',
        suffixes=('_multinomial', '_ovr')
    )
    
    comparison['difference'] = comparison['mean_accuracy'] - comparison['accuracy_multinomial']
    
    # Correlation
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(comparison['accuracy_multinomial'], comparison['mean_accuracy'])
    
    logger.info(f"\nPer-Region Correlation:")
    logger.info(f"  Pearson r: {corr:.4f}")
    logger.info(f"  p-value: {p_value:.4f}")
    
    # Save comparison
    comparison.to_csv(output_dir / 'ovr_vs_multinomial_comparison.csv', index=False)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(comparison['accuracy_multinomial'], comparison['mean_accuracy'], 
               alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Agreement')
    ax.set_xlabel('Multinomial Accuracy', fontweight='bold')
    ax.set_ylabel('OvR Binary Accuracy', fontweight='bold')
    ax.set_title('OvR vs Multinomial Per-Region Accuracy', fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend()
    
    # Add correlation text
    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ovr_vs_multinomial_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison saved to: {output_dir}")


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output_dir, args.hemisphere)
    
    logger.info("="*80)
    logger.info("ONE-VS-REST HEMISPHERE-SPECIFIC CLASSIFICATION")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Hemisphere: {args.hemisphere}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Number of folds: {args.n_folds}")
    logger.info(f"  Random state: {args.random_state}")
    logger.info(f"  Regularization C: {args.regularization_C}")
    logger.info(f"  Diagonal strategy: {args.diagonal_strategy}")
    logger.info(f"  Class weight: {args.class_weight}")
    logger.info(f"  Max iterations: {args.max_iter}")
    logger.info(f"  Save models: {args.save_models}")
    
    try:
        # Train based on hemisphere argument
        if args.hemisphere == 'both':
            # Train both hemispheres
            left_results = train_ovr_single_hemisphere('left', args, logger)
            right_results = train_ovr_single_hemisphere('right', args, logger)
            
            # Compare OvR to multinomial for both
            for results, hemi in [(left_results, 'left'), (right_results, 'right')]:
                multinomial_dir = args.output_dir / f"{hemi}_hemisphere" / "multinomial"
                if multinomial_dir.exists():
                    compare_ovr_to_multinomial(
                        results,
                        multinomial_dir,
                        results['output_dir'],
                        logger
                    )
            
        else:
            # Train single hemisphere
            results = train_ovr_single_hemisphere(args.hemisphere, args, logger)
            
            # Compare to multinomial
            multinomial_dir = args.output_dir / f"{args.hemisphere}_hemisphere" / "multinomial"
            if multinomial_dir.exists():
                compare_ovr_to_multinomial(
                    results,
                    multinomial_dir,
                    results['output_dir'],
                    logger
                )
        
        logger.info("\n" + "="*80)
        logger.info("OVR TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()