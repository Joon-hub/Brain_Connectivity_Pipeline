"""
02_train_hemisphere_multinomial.py

Train multinomial logistic regression separately for left and right hemispheres.
This establishes the baseline performance for hemisphere-specific classification.

Usage:
    python scripts/hemisphere/02_train_hemisphere_multinomial.py --hemisphere left
    python scripts/hemisphere/02_train_hemisphere_multinomial.py --hemisphere right
    python scripts/hemisphere/02_train_hemisphere_multinomial.py --hemisphere both
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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import your existing modules
from src.hemisphere_data.hemisphere_utils import (
    load_region_info,
    get_hemisphere_indices,
    load_hemisphere_data_from_csv
)
from src.preprocessing.connectivity_preprocessor import ConnectivityPreprocessor
from src.evaluation.hemisphere_metrics import (
    compute_classification_metrics,
    compute_per_region_metrics,
    compute_network_level_metrics,
    create_confusion_matrix
)
from src.visualization.hemisphere_viz import (
    plot_confusion_matrix,
    plot_per_region_accuracy,
    plot_network_accuracy
)


def setup_logging(output_dir: Path, hemisphere: str) -> logging.Logger:
    """Set up logging configuration."""
    log_file = output_dir / f"training_{hemisphere}_hemisphere.log"
    
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
        description='Train hemisphere-specific multinomial logistic regression'
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
        default=project_root / 'data' / 'processed' / 'hemisphere',
        help='Directory containing hemisphere-specific data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=project_root / 'data' / 'results' / 'hemisphere_analysis',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--config_file',
        type=Path,
        default=project_root / 'configs' / 'hemisphere_config.yaml',
        help='Path to configuration file'
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
        default=None,
        help='Regularization parameter C. If None, uses value from whole-brain model'
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


def load_config(config_file: Path) -> dict:
    """Load configuration from YAML file."""
    import yaml
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return default configuration
        return {
            'preprocessing': {
                'apply_fisher_z': True,
                'standardize': True,
                'diagonal_strategy': 'region_mean'
            },
            'model': {
                'solver': 'lbfgs',
                'max_iter': 1000,
                'multi_class': 'multinomial'
            }
        }


def preprocess_fold_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    diagonal_strategy: str,
    region_info: pd.DataFrame,
    hemisphere: str,
    logger: logging.Logger
) -> tuple:
    """
    Preprocess data within a single fold (leak-free).
    
    Parameters
    ----------
    X_train : np.ndarray
        Training connectivity matrices (n_subjects, n_regions, n_regions)
    X_test : np.ndarray
        Test connectivity matrices (n_subjects, n_regions, n_regions)
    diagonal_strategy : str
        Strategy for diagonal imputation
    region_info : pd.DataFrame
        Region information with network assignments
    hemisphere : str
        'left' or 'right'
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    X_train_processed : np.ndarray
        Processed training features (n_train, n_features)
    X_test_processed : np.ndarray
        Processed test features (n_test, n_features)
    """
    
    logger.info(f"  Preprocessing with diagonal_strategy='{diagonal_strategy}'")
    
    # Initialize preprocessor
    preprocessor = ConnectivityPreprocessor(
        diagonal_strategy=diagonal_strategy,
        apply_fisher_z=True,
        standardize=True
    )
    
    # Fit on training data only
    preprocessor.fit(X_train, region_info=region_info)
    
    # Transform both train and test
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Validate no NaN/Inf
    if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
        raise ValueError("NaN or Inf detected in training data after preprocessing")
    if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
        raise ValueError("NaN or Inf detected in test data after preprocessing")
    
    logger.info(f"  Training features shape: {X_train_processed.shape}")
    logger.info(f"  Test features shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed


def train_single_hemisphere(
    hemisphere: str,
    args: argparse.Namespace,
    logger: logging.Logger
) -> dict:
    """
    Train multinomial model for a single hemisphere.
    
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
    logger.info(f"TRAINING {hemisphere.upper()} HEMISPHERE")
    logger.info(f"{'='*80}\n")
    
    # Create output directory
    output_dir = args.output_dir / f"{hemisphere}_hemisphere" / "multinomial"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading hemisphere-specific data...")
    data = load_hemisphere_data_from_csv(
        data_dir=args.data_dir,
        hemisphere=hemisphere,
        dataset='rest'  # or 'task' depending on your needs
    )
    
    connectivity = data['connectivity']  # (n_subjects, n_regions, n_regions)
    labels = data['labels']  # (n_subjects,)
    subject_ids = data['subject_ids']  # (n_subjects,)
    region_info = data['region_info']  # DataFrame with region metadata
    
    n_subjects, n_regions, _ = connectivity.shape
    n_classes = len(np.unique(labels))
    
    logger.info(f"Data loaded:")
    logger.info(f"  Subjects: {n_subjects}")
    logger.info(f"  Regions: {n_regions}")
    logger.info(f"  Classes: {n_classes}")
    logger.info(f"  Connectivity shape: {connectivity.shape}")
    logger.info(f"  Labels shape: {labels.shape}")
    
    # Validate data
    assert connectivity.shape[1] == connectivity.shape[2], "Connectivity must be square"
    assert connectivity.shape[0] == len(labels), "Mismatch between connectivity and labels"
    assert np.all(labels >= 0) and np.all(labels < n_classes), "Invalid label values"
    
    # Set up cross-validation
    logger.info(f"\nSetting up {args.n_folds}-fold GroupKFold cross-validation...")
    gkf = GroupKFold(n_splits=args.n_folds)
    
    # Initialize storage for results
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_fold_indices = []
    fold_models = [] if args.save_models else None
    fold_metrics = []
    
    # Cross-validation loop
    logger.info("\nStarting cross-validation...\n")
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(connectivity, labels, groups=subject_ids)):
        fold_start = time.time()
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"  Train subjects: {len(train_idx)}, Test subjects: {len(test_idx)}")
        
        # Split data - CRITICAL: subject-wise split
        X_train = connectivity[train_idx]
        X_test = connectivity[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        # Verify no subject leakage
        train_subjects = set(subject_ids[train_idx])
        test_subjects = set(subject_ids[test_idx])
        assert len(train_subjects.intersection(test_subjects)) == 0, "Subject leakage detected!"
        
        logger.info(f"  Train labels distribution: {np.bincount(y_train).tolist()[:10]}...")
        logger.info(f"  Test labels distribution: {np.bincount(y_test).tolist()[:10]}...")
        
        # Preprocess within fold (LEAK-FREE)
        X_train_processed, X_test_processed = preprocess_fold_data(
            X_train=X_train,
            X_test=X_test,
            diagonal_strategy=args.diagonal_strategy,
            region_info=region_info,
            hemisphere=hemisphere,
            logger=logger
        )
        
        # Train model
        logger.info("  Training multinomial logistic regression...")
        
        # Determine regularization parameter
        if args.regularization_C is not None:
            C = args.regularization_C
        else:
            # Use a reasonable default (you can load from config)
            C = 1.0
        
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=C,
            max_iter=args.max_iter,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=1 if args.verbose else 0
        )
        
        model.fit(X_train_processed, y_train)
        
        # Predict on test set
        logger.info("  Predicting on test set...")
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)
        
        # Compute fold metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        fold_acc = accuracy_score(y_test, y_pred)
        fold_bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        fold_metrics.append({
            'fold': fold_idx + 1,
            'accuracy': fold_acc,
            'balanced_accuracy': fold_bal_acc,
            'n_train': len(y_train),
            'n_test': len(y_test)
        })
        
        fold_time = time.time() - fold_start
        logger.info(f"  Fold accuracy: {fold_acc:.4f}")
        logger.info(f"  Fold balanced accuracy: {fold_bal_acc:.4f}")
        logger.info(f"  Fold time: {fold_time:.2f}s\n")
        
        # Store results
        all_predictions.extend(y_pred)
        all_probabilities.append(y_proba)
        all_true_labels.extend(y_test)
        all_fold_indices.extend([fold_idx + 1] * len(y_test))
        
        if args.save_models:
            fold_models.append({
                'fold': fold_idx + 1,
                'model': model,
                'train_idx': train_idx,
                'test_idx': test_idx
            })
    
    total_time = time.time() - start_time
    logger.info(f"Cross-validation completed in {total_time:.2f}s\n")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    all_true_labels = np.array(all_true_labels)
    all_fold_indices = np.array(all_fold_indices)
    
    # Compute overall metrics
    logger.info("Computing overall metrics...")
    overall_metrics = compute_classification_metrics(
        y_true=all_true_labels,
        y_pred=all_predictions,
        y_proba=all_probabilities
    )
    
    logger.info(f"\nOVERALL RESULTS ({hemisphere.upper()} HEMISPHERE):")
    logger.info(f"  Mean CV Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"  Mean CV Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Top-5 Accuracy: {overall_metrics.get('top_5_accuracy', 'N/A')}")
    
    # Compute per-region metrics
    logger.info("\nComputing per-region metrics...")
    per_region_metrics = compute_per_region_metrics(
        y_true=all_true_labels,
        y_pred=all_predictions,
        region_info=region_info
    )
    
    # Compute network-level metrics
    logger.info("Computing network-level metrics...")
    network_metrics = compute_network_level_metrics(
        y_true=all_true_labels,
        y_pred=all_predictions,
        region_info=region_info
    )
    
    # Create confusion matrix
    logger.info("Creating confusion matrix...")
    confusion_mat = create_confusion_matrix(
        y_true=all_true_labels,
        y_pred=all_predictions,
        n_classes=n_classes
    )
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save predictions
    np.save(output_dir / 'cv_predictions.npy', all_predictions)
    np.save(output_dir / 'cv_probabilities.npy', all_probabilities)
    np.save(output_dir / 'cv_true_labels.npy', all_true_labels)
    np.save(output_dir / 'cv_fold_indices.npy', all_fold_indices)
    
    # Save confusion matrix
    np.save(output_dir / 'confusion_matrix.npy', confusion_mat)
    
    # Save metrics
    with open(output_dir / 'overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    with open(output_dir / 'fold_metrics.json', 'w') as f:
        json.dump(fold_metrics, f, indent=2)
    
    # Save per-region metrics
    per_region_metrics.to_csv(output_dir / 'per_region_metrics.csv', index=False)
    
    # Save network metrics
    network_metrics.to_csv(output_dir / 'network_metrics.csv', index=False)
    
    # Save models if requested
    if args.save_models and fold_models is not None:
        import pickle
        with open(output_dir / 'fold_models.pkl', 'wb') as f:
            pickle.dump(fold_models, f)
        logger.info("Fold models saved")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # Confusion matrix plot
    plot_confusion_matrix(
        confusion_mat=confusion_mat,
        region_info=region_info,
        save_path=output_dir / 'confusion_matrix.png',
        title=f'{hemisphere.capitalize()} Hemisphere - Confusion Matrix'
    )
    
    # Per-region accuracy plot
    plot_per_region_accuracy(
        per_region_metrics=per_region_metrics,
        save_path=output_dir / 'per_region_accuracy.png',
        title=f'{hemisphere.capitalize()} Hemisphere - Per-Region Accuracy'
    )
    
    # Network-level accuracy plot
    plot_network_accuracy(
        network_metrics=network_metrics,
        save_path=output_dir / 'network_accuracy.png',
        title=f'{hemisphere.capitalize()} Hemisphere - Network-Level Accuracy'
    )
    
    logger.info(f"All results saved to: {output_dir}")
    
    # Prepare return dictionary
    results = {
        'hemisphere': hemisphere,
        'n_subjects': n_subjects,
        'n_regions': n_regions,
        'n_classes': n_classes,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'true_labels': all_true_labels,
        'fold_indices': all_fold_indices,
        'confusion_matrix': confusion_mat,
        'overall_metrics': overall_metrics,
        'fold_metrics': fold_metrics,
        'per_region_metrics': per_region_metrics,
        'network_metrics': network_metrics,
        'output_dir': output_dir
    }
    
    return results


def compare_hemispheres(
    left_results: dict,
    right_results: dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Compare results between left and right hemispheres.
    
    Parameters
    ----------
    left_results : dict
        Results from left hemisphere
    right_results : dict
        Results from right hemisphere
    output_dir : Path
        Directory to save comparison results
    logger : logging.Logger
        Logger instance
    """
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPARING LEFT AND RIGHT HEMISPHERES")
    logger.info(f"{'='*80}\n")
    
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    left_acc = left_results['overall_metrics']['accuracy']
    right_acc = right_results['overall_metrics']['accuracy']
    
    left_bal_acc = left_results['overall_metrics']['balanced_accuracy']
    right_bal_acc = right_results['overall_metrics']['balanced_accuracy']
    
    logger.info(f"Left Hemisphere Accuracy: {left_acc:.4f}")
    logger.info(f"Right Hemisphere Accuracy: {right_acc:.4f}")
    logger.info(f"Difference: {abs(left_acc - right_acc):.4f}")
    
    # Statistical test
    from scipy.stats import ttest_rel
    
    left_fold_accs = [m['accuracy'] for m in left_results['fold_metrics']]
    right_fold_accs = [m['accuracy'] for m in right_results['fold_metrics']]
    
    t_stat, p_value = ttest_rel(left_fold_accs, right_fold_accs)
    logger.info(f"\nPaired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        logger.info("Significant difference detected between hemispheres (p < 0.05)")
    else:
        logger.info("No significant difference between hemispheres (p >= 0.05)")
    
    # Per-region correlation
    left_per_region = left_results['per_region_metrics']
    right_per_region = right_results['per_region_metrics']
    
    # Align by region (they should have same regions)
    merged = pd.merge(
        left_per_region[['region_id', 'accuracy']],
        right_per_region[['region_id', 'accuracy']],
        on='region_id',
        suffixes=('_left', '_right')
    )
    
    from scipy.stats import pearsonr
    corr, corr_p = pearsonr(merged['accuracy_left'], merged['accuracy_right'])
    logger.info(f"\nPer-region accuracy correlation: r={corr:.4f}, p={corr_p:.4f}")
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    axes[0].bar(['Left', 'Right'], [left_acc, right_acc], color=['steelblue', 'coral'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Overall Accuracy Comparison')
    axes[0].set_ylim([0.8, 1.0])
    
    # Per-region scatter
    axes[1].scatter(merged['accuracy_left'], merged['accuracy_right'], alpha=0.6)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel('Left Hemisphere Accuracy')
    axes[1].set_ylabel('Right Hemisphere Accuracy')
    axes[1].set_title(f'Per-Region Accuracy Correlation (r={corr:.3f})')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'hemisphere_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison summary
    comparison_summary = {
        'left_accuracy': left_acc,
        'right_accuracy': right_acc,
        'accuracy_difference': abs(left_acc - right_acc),
        'paired_ttest': {
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        },
        'per_region_correlation': {
            'correlation': float(corr),
            'p_value': float(corr_p)
        }
    }
    
    with open(comparison_dir / 'hemisphere_comparison_summary.json', 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    logger.info(f"\nComparison results saved to: {comparison_dir}")


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output_dir, args.hemisphere)
    
    logger.info("="*80)
    logger.info("HEMISPHERE-SPECIFIC MULTINOMIAL LOGISTIC REGRESSION")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Hemisphere: {args.hemisphere}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Number of folds: {args.n_folds}")
    logger.info(f"  Random state: {args.random_state}")
    logger.info(f"  Regularization C: {args.regularization_C}")
    logger.info(f"  Diagonal strategy: {args.diagonal_strategy}")
    logger.info(f"  Max iterations: {args.max_iter}")
    logger.info(f"  Save models: {args.save_models}")
    
    try:
        # Train based on hemisphere argument
        if args.hemisphere == 'both':
            # Train both hemispheres
            left_results = train_single_hemisphere('left', args, logger)
            right_results = train_single_hemisphere('right', args, logger)
            
            # Compare results
            compare_hemispheres(left_results, right_results, args.output_dir, logger)
            
        else:
            # Train single hemisphere
            results = train_single_hemisphere(args.hemisphere, args, logger)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()