"""
03_train_hemisphere_xgboost.py

XGBoost Native Multiclass Classification for Hemisphere Analysis

Key features:
1. Native multiclass using 'multi:softprob' objective
2. Random seed changes per INNER FOLD only, not per trial
3. Per-fold hyperparameter tuning with Optuna
4. Feature importance tracking
5. Same preprocessing as OvR version (diagonal imputation, Fisher-Z)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import custom modules
from src.hemisphere.hemisphere_utils import (
    load_hemisphere_data,
    prepare_classification_data
)
from src.hemisphere.hemisphere_metrics import (
    compute_classification_metrics,
    compute_per_region_metrics,
    compute_network_level_metrics,
    create_confusion_matrix
)

# ==============================================================================
# SETUP & CONFIGURATION
# ==============================================================================

def setup_logging(output_dir: Path, hemisphere: str) -> logging.Logger:
    """Set up logging configuration."""
    log_file = output_dir / f"training_{hemisphere}_hemisphere_xgboost.log"
    
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
        description='Train hemisphere-specific XGBoost multiclass classifier'
    )
    
    # Core settings
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
    
    # Model settings
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
        '--diagonal_strategy',
        type=str,
        default='random',
        help='Diagonal imputation strategy'
    )
    
    # XGBoost default hyperparameters (used when --tune_hyperparams is not enabled)
    parser.add_argument(
        '--max_depth',
        type=int,
        default=6,
        help='Maximum tree depth'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Learning rate (eta)'
    )
    
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of boosting rounds'
    )
    
    parser.add_argument(
        '--min_child_weight',
        type=int,
        default=1,
        help='Minimum sum of instance weight needed in a child'
    )
    
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.8,
        help='Subsample ratio of training instances'
    )
    
    parser.add_argument(
        '--colsample_bytree',
        type=float,
        default=0.8,
        help='Subsample ratio of columns when constructing each tree'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0,
        help='Minimum loss reduction required to make a split'
    )
    
    parser.add_argument(
        '--reg_alpha',
        type=float,
        default=0,
        help='L1 regularization term on weights'
    )
    
    parser.add_argument(
        '--reg_lambda',
        type=float,
        default=1,
        help='L2 regularization term on weights'
    )
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 uses all cores)'
    )
    
    # Execution flags
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
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Number of subjects to sample for testing (takes first N subjects). If None, uses all subjects.'
    )
    
    # Optimization
    parser.add_argument(
        '--tune_hyperparams',
        action='store_true',
        help='Enable per-fold hyperparameter tuning using Optuna'
    )
    
    parser.add_argument(
        '--optuna_trials',
        type=int,
        default=50,
        help='Number of Optuna trials for hyperparameter optimization'
    )
    
    parser.add_argument(
        '--test_on_task',
        action='store_true',
        help='After CV on rest data, train final model and test on task data'
    )
    
    parser.add_argument(
        '--optuna_n_jobs',
        type=int,
        default=None,
        help='Number of parallel jobs for Optuna optimization. If None, uses min(n_jobs, 32)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate input arguments."""
    if args.sample is not None and args.sample <= 0:
        raise ValueError(f"Sample size must be positive, got {args.sample}")
    if args.n_folds < 2:
        raise ValueError(f"Number of folds must be at least 2, got {args.n_folds}")


def get_optuna_n_jobs(args):
    """Calculate safe number of jobs for Optuna."""
    if args.optuna_n_jobs is not None:
        return args.optuna_n_jobs
    if args.n_jobs == -1:
        return min(os.cpu_count() or 1, 32)
    elif args.n_jobs > 0:
        return min(args.n_jobs, 32)
    return 1


# ==============================================================================
# DATA PREPROCESSING & HELPERS
# ==============================================================================

def sample_first_n_subjects(
    data: dict,
    n_sample: int,
    logger: logging.Logger
) -> dict:
    """Sample first n subjects for testing (deterministic selection)."""
    
    total_subjects = data['n_subjects']
    
    if n_sample > total_subjects:
        logger.warning(
            f"Requested sample size ({n_sample}) exceeds available subjects ({total_subjects}). "
            f"Using all {total_subjects} subjects."
        )
        return data
    
    if n_sample <= 0:
        raise ValueError(f"Sample size must be positive, got {n_sample}")
    
    logger.info(f"SAMPLING: Using first {n_sample}/{total_subjects} subjects")
    
    connectivity_sampled = data['connectivity'][:n_sample]
    subject_ids_sampled = data['subject_ids'][:n_sample]
    
    sampled_data = {
        'connectivity': connectivity_sampled,
        'subject_ids': subject_ids_sampled,
        'region_info': data['region_info'],
        'hemisphere': data['hemisphere'],
        'n_subjects': n_sample,
        'n_regions': data['n_regions']
    }
    
    return sampled_data


def apply_diagonal_imputation(
    connectivity: np.ndarray,
    region_info: pd.DataFrame,
    strategy: str,
    logger: logging.Logger,
    seed: int | None = None,
    verbose: bool = True
) -> np.ndarray:
    """Apply diagonal imputation to connectivity matrices."""
    
    n_subjects, n_regions, _ = connectivity.shape
    imp = connectivity.copy()
    
    rng = np.random.default_rng(seed if seed is not None else 42)
    if strategy == 'random':
        for i in range(n_subjects):
            off_diag_mask = ~np.eye(n_regions, dtype=bool)
            off_diag_values = imp[i][off_diag_mask]
            min_val = off_diag_values.min()
            max_val = off_diag_values.max()
            
            for j in range(n_regions):
                imp[i, j, j] = rng.uniform(min_val, max_val)
        if verbose:
            logger.info(f"  Random diagonal imputation with seed={seed}")
    
    else:
        raise ValueError(f"Unknown diagonal strategy: {strategy}")
    
    return imp


def apply_fisher_z_transformation(
    connectivity: np.ndarray,
    logger: logging.Logger,
    verbose: bool = True
) -> np.ndarray:
    """Apply Fisher Z-transformation to connectivity matrices."""
    
    if verbose:
        logger.info("  Applying Fisher Z-transformation...")
    
    connectivity_clipped = np.clip(connectivity, -0.999, 0.999)
    connectivity_transformed = np.arctanh(connectivity_clipped)
    
    if np.any(np.isnan(connectivity_transformed)):
        raise ValueError("NaN detected after Fisher Z transformation")
    if np.any(np.isinf(connectivity_transformed)):
        raise ValueError("Inf detected after Fisher Z transformation")
    
    if verbose:
        logger.info(f"  Value range after Fisher Z: [{connectivity_transformed.min():.4f}, {connectivity_transformed.max():.4f}]")
    
    return connectivity_transformed


def preprocess_fold_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    logger: logging.Logger,
    verbose: bool = False
) -> tuple:
    """Standardize data within fold (leak-free)."""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
        raise ValueError("NaN/Inf in scaled training data")
    if np.any(np.isnan(X_test_scaled)) or np.any(np.isinf(X_test_scaled)):
        raise ValueError("NaN/Inf in scaled test data")
    
    return X_train_scaled, X_test_scaled, scaler


# ==============================================================================
# XGBOOST TRAINING
# ==============================================================================

def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    min_child_weight: int,
    subsample: float,
    colsample_bytree: float,
    gamma: float,
    reg_alpha: float,
    reg_lambda: float,
    random_state: int,
    n_jobs: int,
    logger: logging.Logger,
    verbose: bool = False
) -> xgb.XGBClassifier:
    """Train XGBoost multiclass classifier."""
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method='hist',  # CPU-optimized
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    return model


def extract_feature_importance(
    model: xgb.XGBClassifier,
    n_regions: int,
    region_info: pd.DataFrame
) -> pd.DataFrame:
    """Extract feature importance from XGBoost model."""
    
    # Get feature importance (gain-based)
    importance_gain = model.feature_importances_
    
    # Create feature names mapping
    feature_names = []
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                region_i = region_info.iloc[i]['region_name']
                region_j = region_info.iloc[j]['region_name']
                feature_names.append(f"{region_i}_{region_j}")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': importance_gain
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance_gain', ascending=False)
    
    return importance_df


# ==============================================================================
# OPTUNA HYPERPARAMETER TUNING
# ==============================================================================

def optimize_hyperparameters_xgboost_fold(
    X_train_unscaled: np.ndarray | None,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_classes: int,
    n_trials: int,
    random_state: int,
    optuna_n_jobs: int,
    logger: logging.Logger,
    verbose: bool = False,
    original_connectivity: np.ndarray | None = None,
    subject_ids: np.ndarray | None = None,
    region_info: pd.DataFrame | None = None,
    is_random: bool = False,
    fold_idx: int = 0,
    n_jobs: int = -1
) -> dict:
    """
    Run Optuna hyperparameter optimization for XGBoost.
    
    MODIFIED: Seed changes per inner fold, NOT per trial.
    Data preprocessing done ONCE before all trials.
    """
    
    logger.info(f"\n  Optimizing hyperparameters (Fold {fold_idx + 1}): {n_trials} trials")
    
    # Fixed seed for this fold's inner validation split
    inner_fold_seed = random_state + fold_idx * 100000
    
    best_score_tracker = {'score': 0.0, 'trial': 0}
    
    # Create inner split ONCE before all trials
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=inner_fold_seed)
    train_inner_idx, val_inner_idx = next(gss.split(range(len(groups_train)), groups=groups_train))
    
    # For random diagonal: preprocess data ONCE before all trials
    if is_random:
        train_subjects = np.unique(groups_train[train_inner_idx])
        val_subjects = np.unique(groups_train[val_inner_idx])
        
        train_subject_idx = np.isin(subject_ids, train_subjects)
        val_subject_idx = np.isin(subject_ids, val_subjects)
        
        train_conn = original_connectivity[train_subject_idx].copy()
        val_conn = original_connectivity[val_subject_idx].copy()
        
        # Apply diagonal imputation ONCE with fixed seeds
        train_conn = apply_diagonal_imputation(
            train_conn, region_info, 'random', logger, 
            seed=inner_fold_seed, verbose=False
        )
        val_conn = apply_diagonal_imputation(
            val_conn, region_info, 'random', logger, 
            seed=inner_fold_seed + 1000, verbose=False
        )
        
        train_conn = apply_fisher_z_transformation(train_conn, logger, verbose=False)
        val_conn = apply_fisher_z_transformation(val_conn, logger, verbose=False)
        
        X_train_inner_unscaled, y_inner_train, _ = prepare_classification_data(
            train_conn, region_info, subject_ids[train_subject_idx]
        )
        X_val_inner_unscaled, y_inner_val, _ = prepare_classification_data(
            val_conn, region_info, subject_ids[val_subject_idx]
        )
    else:
        X_train_inner_unscaled = X_train_unscaled[train_inner_idx]
        X_val_inner_unscaled = X_train_unscaled[val_inner_idx]
        y_inner_train = y_train[train_inner_idx]
        y_inner_val = y_train[val_inner_idx]
    
    def objective(trial):
        """Optuna objective function for XGBoost."""
        
        # Hyperparameter search space
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        gamma = trial.suggest_float('gamma', 0, 5)
        reg_alpha = trial.suggest_float('reg_alpha', 0, 10)
        reg_lambda = trial.suggest_float('reg_lambda', 0, 10)
        
        # Scale data
        scaler_inner = StandardScaler()
        X_train_inner = scaler_inner.fit_transform(X_train_inner_unscaled)
        X_val_inner = scaler_inner.transform(X_val_inner_unscaled)
        
        # Train model
        model = train_xgboost_classifier(
            X_train_inner, y_inner_train, n_classes,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=1,  # Use 1 job per trial to allow Optuna parallelization
            logger=logger,
            verbose=False
        )
        
        # Predict and evaluate
        predictions = model.predict(X_val_inner)
        score = accuracy_score(y_inner_val, predictions)
        
        is_best = score > best_score_tracker['score']
        if is_best:
            best_score_tracker.update({'score': score, 'trial': trial.number})
        
        if verbose:
            logger.info(f"    Trial {trial.number+1}: Acc={score:.4f}" + 
                       (" ★" if is_best else ""))
        
        return score
    
    optuna_start = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_seed = random_state + fold_idx * 100000
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=study_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=optuna_n_jobs)
    optuna_time = time.time() - optuna_start
    
    best_params = study.best_params
    best_params['_optuna_best_score'] = float(study.best_value)
    best_params['_optuna_time_seconds'] = float(optuna_time)
    
    logger.info(f"  Best: depth={best_params['max_depth']}, "
               f"lr={best_params['learning_rate']:.4f}, "
               f"n_est={best_params['n_estimators']}, "
               f"Acc={study.best_value:.4f} ({optuna_time:.1f}s)")
    
    return best_params


# ==============================================================================
# METRICS & REPORTING
# ==============================================================================

def compute_classification_metrics_enhanced(y_true, y_pred, y_proba=None):
    """Compute enhanced classification metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'n_samples': int(len(y_true)),
        'n_classes': int(len(np.unique(y_true)))
    }
    
    if y_proba is not None:
        for k in [3, 5, 10]:
            if k <= y_proba.shape[1]:
                top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
                top_k_correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
                metrics[f'top_{k}_accuracy'] = float(np.mean(top_k_correct))
    
    return metrics


def compute_per_region_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    region_info: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """Compute per-region binary classification metrics."""
    n_regions = probabilities.shape[1]
    per_region = []
    
    for region_idx in range(n_regions):
        y_true_binary = (y_true == region_idx).astype(int)
        y_pred_binary = (probabilities[:, region_idx] >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        
        per_region.append({
            'region_idx': region_idx,
            'region_name': region_info.iloc[region_idx]['region_name'],
            'network': region_info.iloc[region_idx]['network'],
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_positive_samples': int(y_true_binary.sum()),
            'n_total_samples': int(len(y_true_binary))
        })
    
    return pd.DataFrame(per_region)


def compute_network_level_metrics(per_region_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute network-level aggregated metrics."""
    network_metrics = per_region_metrics.groupby('network').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'n_positive_samples': 'sum',
        'n_total_samples': 'first'
    }).reset_index()
    
    network_metrics = network_metrics.sort_values('f1_score', ascending=False)
    return network_metrics


# ==============================================================================
# TASK TESTING
# ==============================================================================

def test_on_task_data(
    hemisphere: str,
    best_params: dict,
    random_state: int,
    n_jobs: int,
    diagonal_strategy: str,
    data_dir: Path,
    output_dir: Path,
    sample: int,
    region_info: pd.DataFrame,
    n_regions: int,
    original_connectivity_rest: np.ndarray,
    subject_ids_rest: np.ndarray,
    logger: logging.Logger
) -> dict:
    """Test trained XGBoost model on task data."""
    
    # Extract hyperparameters
    max_depth = best_params['max_depth']
    learning_rate = best_params['learning_rate']
    n_estimators = best_params['n_estimators']
    min_child_weight = best_params['min_child_weight']
    subsample = best_params['subsample']
    colsample_bytree = best_params['colsample_bytree']
    gamma = best_params['gamma']
    reg_alpha = best_params['reg_alpha']
    reg_lambda = best_params['reg_lambda']
    
    is_random = diagonal_strategy == 'random'
    
    logger.info("\n" + "="*80)
    logger.info("TASK TESTING")
    logger.info("="*80)
    logger.info(f"Training on rest, testing on task (Gender Stroop)")
    logger.info(f"Hyperparams: depth={max_depth}, lr={learning_rate:.4f}, n_est={n_estimators}")
    
    # Load task data
    task_data = load_hemisphere_data(
        data_dir=data_dir,
        hemisphere=hemisphere,
        dataset='task',
        return_matrix=True,
        validate=True
    )
    
    if sample:
        task_data = sample_first_n_subjects(task_data, sample, logger)
    
    logger.info(f"Task: {len(task_data['subject_ids'])} subjects")
    
    task_original = task_data['connectivity'].copy()
    
    logger.info("Clipping task off-diagonal to [-0.999, 0.999]")
    task_n = task_original.shape[0]
    for i in range(task_n):
        off_mask = ~np.eye(n_regions, dtype=bool)
        task_original[i][off_mask] = np.clip(task_original[i][off_mask], -0.999, 0.999)
    
    # Process REST
    rest_conn = original_connectivity_rest.copy()
    
    if is_random:
        rest_conn = apply_diagonal_imputation(
            rest_conn, region_info, 'random', logger, seed=random_state
        )
    else:
        rest_conn = apply_diagonal_imputation(
            rest_conn, region_info, diagonal_strategy, logger
        )
    
    rest_conn = apply_fisher_z_transformation(rest_conn, logger)
    X_rest, y_rest, groups_rest = prepare_classification_data(
        rest_conn, region_info, subject_ids_rest
    )
    
    # Process TASK
    task_conn = task_original.copy()
    
    if is_random:
        task_conn = apply_diagonal_imputation(
            task_conn, region_info, 'random', logger, seed=random_state + 9999
        )
    else:
        task_conn = apply_diagonal_imputation(
            task_conn, region_info, diagonal_strategy, logger
        )
    
    task_conn = apply_fisher_z_transformation(task_conn, logger)
    X_task, y_task, groups_task = prepare_classification_data(
        task_conn, region_info, task_data['subject_ids']
    )
    
    # Scale
    scaler_final = StandardScaler()
    X_rest_scaled = scaler_final.fit_transform(X_rest)
    X_task_scaled = scaler_final.transform(X_task)
    
    # Train on rest
    logger.info(f"Training XGBoost on rest data...")
    train_start = time.time()
    final_model = train_xgboost_classifier(
        X_rest_scaled, y_rest, n_regions,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        logger=logger,
        verbose=False
    )
    train_time = time.time() - train_start
    logger.info(f"Training complete ({train_time:.2f}s)")
    
    # Extract feature importance
    feature_importance = extract_feature_importance(final_model, n_regions, region_info)
    
    # Evaluate on rest
    rest_probabilities = final_model.predict_proba(X_rest_scaled)
    rest_predictions = final_model.predict(X_rest_scaled)
    rest_metrics = compute_classification_metrics_enhanced(y_rest, rest_predictions, rest_probabilities)
    
    logger.info(f"Rest Acc: {rest_metrics['accuracy']:.4f}")
    
    # Evaluate on task
    task_probabilities = final_model.predict_proba(X_task_scaled)
    task_predictions = final_model.predict(X_task_scaled)
    
    task_metrics = compute_classification_metrics_enhanced(y_task, task_predictions, task_probabilities)
    task_per_region = compute_per_region_binary_metrics(y_task, task_probabilities, region_info)
    task_network = compute_network_level_metrics(task_per_region)
    task_confusion = create_confusion_matrix(y_task, task_predictions, n_regions)
    rest_confusion = create_confusion_matrix(y_rest, rest_predictions, n_regions)
    
    logger.info(f"Task Acc: {task_metrics['accuracy']:.4f}")
    logger.info(f"Accuracy drop: {rest_metrics['accuracy'] - task_metrics['accuracy']:.4f}")
    
    # Save results
    task_output_dir = output_dir / f"{hemisphere}_hemisphere" / "task_testing_xgboost"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(task_output_dir / 'rest_predictions.npy', rest_predictions)
    np.save(task_output_dir / 'rest_probabilities.npy', rest_probabilities)
    np.save(task_output_dir / 'rest_true_labels.npy', y_rest)
    np.save(task_output_dir / 'rest_confusion_matrix.npy', rest_confusion)
    
    np.save(task_output_dir / 'task_predictions.npy', task_predictions)
    np.save(task_output_dir / 'task_probabilities.npy', task_probabilities)
    np.save(task_output_dir / 'task_true_labels.npy', y_task)
    np.save(task_output_dir / 'task_confusion_matrix.npy', task_confusion)
    
    # Save feature importance
    feature_importance.to_csv(task_output_dir / 'feature_importance.csv', index=False)
    
    task_summary = {
        'diagonal_strategy': diagonal_strategy,
        'rest_train_accuracy': float(rest_metrics['accuracy']),
        'task_test_accuracy': float(task_metrics['accuracy']),
        'accuracy_drop': float(rest_metrics['accuracy'] - task_metrics['accuracy']),
        'hyperparameters': best_params,
        'n_rest_subjects': int(len(np.unique(groups_rest))),
        'n_task_subjects': int(len(np.unique(groups_task))),
        'n_regions': int(n_regions),
        'training_time_seconds': float(train_time)
    }
    
    for metric_name in ['top_3_accuracy', 'top_5_accuracy', 'top_10_accuracy']:
        if metric_name in rest_metrics:
            task_summary[f'rest_{metric_name}'] = float(rest_metrics[metric_name])
        if metric_name in task_metrics:
            task_summary[f'task_{metric_name}'] = float(task_metrics[metric_name])
    
    with open(task_output_dir / 'task_testing_summary.json', 'w') as f:
        json.dump(task_summary, f, indent=2)
    
    task_per_region.to_csv(task_output_dir / 'task_per_region_binary_metrics.csv', index=False)
    task_network.to_csv(task_output_dir / 'task_network_metrics.csv', index=False)
    
    import pickle
    with open(task_output_dir / 'final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open(task_output_dir / 'final_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_final, f)
    
    logger.info(f"Saved to: {task_output_dir}")
    
    return {
        'task_metrics': task_metrics,
        'task_per_region': task_per_region,
        'task_network': task_network,
        'task_summary': task_summary,
        'rest_metrics': rest_metrics,
        'feature_importance': feature_importance
    }


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_single_hemisphere(
    hemisphere: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    optuna_n_jobs: int
) -> dict:
    """Train XGBoost multiclass model for a single hemisphere."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING {hemisphere.upper()} HEMISPHERE")
    logger.info(f"{'='*80}")
    
    # Create output directory
    output_dir = args.output_dir / f"{hemisphere}_hemisphere" / "xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data = load_hemisphere_data(
        data_dir=args.data_dir,
        hemisphere=hemisphere,
        dataset='rest',
        return_matrix=True,
        validate=True
    )
    
    if args.sample is not None:
        data = sample_first_n_subjects(data, args.sample, logger)
    
    connectivity = data['connectivity']
    original_connectivity = connectivity.copy()
    subject_ids = data['subject_ids']
    region_info = data['region_info']
    
    n_subjects, n_regions, _ = connectivity.shape
    
    logger.info(f"Subjects: {n_subjects}, Regions: {n_regions}")
    logger.info(f"Diagonal strategy: {args.diagonal_strategy}")
    
    logger.info("Clipping off-diagonal to [-0.999, 0.999]")
    for i in range(n_subjects):
        off_mask = ~np.eye(n_regions, dtype=bool)
        original_connectivity[i][off_mask] = np.clip(
            original_connectivity[i][off_mask], -0.999, 0.999
        )
    
    is_random = args.diagonal_strategy == 'random'
    
    _, y, groups = prepare_classification_data(original_connectivity, region_info, subject_ids)
    
    if not is_random:
        connectivity = apply_diagonal_imputation(
            original_connectivity, region_info, args.diagonal_strategy, logger
        )
        connectivity = apply_fisher_z_transformation(connectivity, logger)
        X, _, _ = prepare_classification_data(connectivity, region_info, subject_ids)
    else:
        X = None
    
    # CV loop
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION ({args.n_folds} folds)")
    logger.info(f"{'='*80}")
    
    gkf = GroupKFold(n_splits=args.n_folds)
    all_predictions, all_probabilities, all_true_labels = [], [], []
    fold_metrics = []
    fold_feature_importances = []
    best_fold_params = None
    best_fold_val_acc = 0.0
    best_fold_idx = -1
    
    cv_start_time = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(
        X if X is not None else range(len(y)), y, groups=groups
    )):
        logger.info(f"\nFold {fold_idx + 1}/{args.n_folds}")
        
        n_train = len(np.unique(groups[train_idx]))
        n_test = len(np.unique(groups[test_idx]))
        logger.info(f"  Train: {n_train} subjects, Test: {n_test} subjects")
        
        # Prepare fold data
        if not is_random:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
        else:
            train_subjects = np.unique(groups[train_idx])
            test_subjects = np.unique(groups[test_idx])
            
            train_mask = np.isin(subject_ids, train_subjects)
            test_mask = np.isin(subject_ids, test_subjects)
            
            train_conn = original_connectivity[train_mask].copy()
            test_conn = original_connectivity[test_mask].copy()
            
            fold_seed = args.random_state + (fold_idx + 1) * 1000
            
            train_conn = apply_diagonal_imputation(
                train_conn, region_info, 'random', logger, seed=fold_seed
            )
            test_conn = apply_diagonal_imputation(
                test_conn, region_info, 'random', logger, seed=fold_seed + 1000
            )
            
            train_conn = apply_fisher_z_transformation(train_conn, logger)
            test_conn = apply_fisher_z_transformation(test_conn, logger)
            
            X_train, y_train, groups_train = prepare_classification_data(
                train_conn, region_info, subject_ids[train_mask]
            )
            X_test, y_test, _ = prepare_classification_data(
                test_conn, region_info, subject_ids[test_mask]
            )
        
        # Hyperparameter tuning (per-fold)
        if args.tune_hyperparams:
            fold_best_params = optimize_hyperparameters_xgboost_fold(
                X_train_unscaled=None if is_random else X_train,
                y_train=y_train,
                groups_train=groups_train,
                n_classes=n_regions,
                n_trials=args.optuna_trials,
                random_state=args.random_state,
                optuna_n_jobs=optuna_n_jobs,
                logger=logger,
                verbose=args.verbose,
                original_connectivity=original_connectivity if is_random else None,
                subject_ids=subject_ids if is_random else None,
                region_info=region_info if is_random else None,
                is_random=is_random,
                fold_idx=fold_idx,
                n_jobs=args.n_jobs
            )
            max_depth = fold_best_params['max_depth']
            learning_rate = fold_best_params['learning_rate']
            n_estimators = fold_best_params['n_estimators']
            min_child_weight = fold_best_params['min_child_weight']
            subsample = fold_best_params['subsample']
            colsample_bytree = fold_best_params['colsample_bytree']
            gamma = fold_best_params['gamma']
            reg_alpha = fold_best_params['reg_alpha']
            reg_lambda = fold_best_params['reg_lambda']
        else:
            max_depth = args.max_depth
            learning_rate = args.learning_rate
            n_estimators = args.n_estimators
            min_child_weight = args.min_child_weight
            subsample = args.subsample
            colsample_bytree = args.colsample_bytree
            gamma = args.gamma
            reg_alpha = args.reg_alpha
            reg_lambda = args.reg_lambda
            fold_best_params = {
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'gamma': gamma,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda
            }
        
        # Scale and train
        X_train_scaled, X_test_scaled, _ = preprocess_fold_data(
            X_train, X_test, logger, args.verbose
        )
        
        fold_start = time.time()
        fold_model = train_xgboost_classifier(
            X_train_scaled, y_train, n_regions,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            logger=logger,
            verbose=False
        )
        fold_train_time = time.time() - fold_start
        
        # Extract feature importance
        fold_importance = extract_feature_importance(fold_model, n_regions, region_info)
        fold_importance['fold'] = fold_idx + 1
        fold_feature_importances.append(fold_importance)
        
        # Predictions
        train_probabilities = fold_model.predict_proba(X_train_scaled)
        train_predictions = fold_model.predict(X_train_scaled)
        fold_train_acc = accuracy_score(y_train, train_predictions)
        
        probabilities = fold_model.predict_proba(X_test_scaled)
        predictions = fold_model.predict(X_test_scaled)
        fold_val_acc = accuracy_score(y_test, predictions)
        
        # Track best fold
        if fold_val_acc > best_fold_val_acc:
            best_fold_val_acc = fold_val_acc
            best_fold_params = fold_best_params
            best_fold_idx = fold_idx
        
        fold_metrics.append({
            'fold': fold_idx + 1,
            'train_accuracy': float(fold_train_acc),
            'val_accuracy': float(fold_val_acc),
            'train_time': float(fold_train_time),
            'hyperparameters': fold_best_params
        })
        
        logger.info(f"  Train: {fold_train_acc:.4f}, Val: {fold_val_acc:.4f} ({fold_train_time:.1f}s)")
        
        all_predictions.extend(predictions)
        all_probabilities.append(probabilities)
        all_true_labels.extend(y_test)
        
        # Save fold model if requested
        if args.save_models:
            fold_model_dir = output_dir / 'fold_models'
            fold_model_dir.mkdir(parents=True, exist_ok=True)
            import pickle
            with open(fold_model_dir / f'model_fold_{fold_idx+1}.pkl', 'wb') as f:
                pickle.dump(fold_model, f)
    
    cv_total_time = time.time() - cv_start_time
    logger.info(f"\nCV complete ({cv_total_time:.1f}s)")
    logger.info(f"Best fold: {best_fold_idx + 1}, Val Acc: {best_fold_val_acc:.4f}")
    
    # Aggregate CV results
    train_accs = [f['train_accuracy'] for f in fold_metrics]
    val_accs = [f['val_accuracy'] for f in fold_metrics]
    mean_train_acc = np.mean(train_accs)
    mean_val_acc = np.mean(val_accs)
    
    logger.info(f"Mean Train: {mean_train_acc:.4f}, Mean Val: {mean_val_acc:.4f}")
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    all_true_labels = np.array(all_true_labels)
    
    overall_metrics = compute_classification_metrics_enhanced(
        all_true_labels, all_predictions, all_probabilities
    )
    per_region_metrics = compute_per_region_binary_metrics(
        all_true_labels, all_probabilities, region_info
    )
    network_metrics = compute_network_level_metrics(per_region_metrics)
    confusion_mat = create_confusion_matrix(all_true_labels, all_predictions, n_regions)
    
    logger.info(f"\nOverall Accuracy: {overall_metrics['accuracy']:.4f}")
    
    # Aggregate feature importance across folds
    all_importance = pd.concat(fold_feature_importances, ignore_index=True)
    avg_importance = all_importance.groupby('feature')['importance_gain'].mean().reset_index()
    avg_importance = avg_importance.sort_values('importance_gain', ascending=False)
    avg_importance.columns = ['feature', 'avg_importance_gain']
    
    # Save CV results
    np.save(output_dir / 'cv_predictions.npy', all_predictions)
    np.save(output_dir / 'cv_probabilities.npy', all_probabilities)
    np.save(output_dir / 'cv_true_labels.npy', all_true_labels)
    np.save(output_dir / 'confusion_matrix.npy', confusion_mat)
    
    # Save feature importance
    avg_importance.to_csv(output_dir / 'feature_importance_avg.csv', index=False)
    all_importance.to_csv(output_dir / 'feature_importance_all_folds.csv', index=False)
    
    cv_summary = {
        'model_type': 'xgboost_multiclass',
        'diagonal_strategy': args.diagonal_strategy,
        'best_fold_hyperparameters': best_fold_params,
        'best_fold_idx': int(best_fold_idx + 1),
        'best_fold_val_accuracy': float(best_fold_val_acc),
        'overall_metrics': overall_metrics,
        'fold_metrics': fold_metrics,
        'cv_time_seconds': float(cv_total_time),
        'mean_train_accuracy': float(mean_train_acc),
        'mean_val_accuracy': float(mean_val_acc),
        'generalization_gap': float(mean_train_acc - mean_val_acc)
    }
    
    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    with open(output_dir / 'overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    per_region_metrics.to_csv(output_dir / 'per_region_binary_metrics.csv', index=False)
    network_metrics.to_csv(output_dir / 'network_metrics.csv', index=False)
    
    logger.info(f"Saved to: {output_dir}")
    
    # Task testing
    task_results = None
    if args.test_on_task:
        try:
            task_results = test_on_task_data(
                hemisphere=hemisphere,
                best_params=best_fold_params,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                diagonal_strategy=args.diagonal_strategy,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                sample=args.sample,
                region_info=region_info,
                n_regions=n_regions,
                original_connectivity_rest=original_connectivity,
                subject_ids_rest=subject_ids,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Task testing failed: {str(e)}", exc_info=True)
    
    return {
        'hemisphere': hemisphere,
        'n_subjects': n_subjects,
        'n_regions': n_regions,
        'overall_metrics': overall_metrics,
        'per_region_metrics': per_region_metrics,
        'network_metrics': network_metrics,
        'task_results': task_results,
        'output_dir': output_dir,
        'cv_summary': cv_summary,
        'feature_importance': avg_importance
    }


def compare_hemispheres(left_results, right_results, output_dir, logger):
    """Compare left and right hemisphere results."""
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPARING HEMISPHERES")
    logger.info(f"{'='*80}")
    
    comparison_dir = output_dir / "comparison_xgboost"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    left_acc = left_results['overall_metrics']['accuracy']
    right_acc = right_results['overall_metrics']['accuracy']
    
    logger.info(f"Left: {left_acc:.4f}, Right: {right_acc:.4f}, Diff: {abs(left_acc - right_acc):.4f}")
    
    # Statistical test
    from scipy.stats import ttest_rel
    left_accs = [m['val_accuracy'] for m in left_results['cv_summary']['fold_metrics']]
    right_accs = [m['val_accuracy'] for m in right_results['cv_summary']['fold_metrics']]
    t_stat, p_value = ttest_rel(left_accs, right_accs)
    
    logger.info(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    summary = {
        'model_type': 'xgboost_multiclass',
        'left_accuracy': left_acc,
        'right_accuracy': right_acc,
        'difference': abs(left_acc - right_acc),
        'ttest_p_value': float(p_value),
        'left_best_fold': left_results['cv_summary']['best_fold_idx'],
        'right_best_fold': right_results['cv_summary']['best_fold_idx']
    }
    
    if left_results.get('task_results') and right_results.get('task_results'):
        left_task = left_results['task_results']['task_summary']['task_test_accuracy']
        right_task = right_results['task_results']['task_summary']['task_test_accuracy']
        
        summary['left_task_accuracy'] = left_task
        summary['right_task_accuracy'] = right_task
        summary['left_accuracy_drop'] = left_results['task_results']['task_summary']['accuracy_drop']
        summary['right_accuracy_drop'] = right_results['task_results']['task_summary']['accuracy_drop']
        
        logger.info(f"Task: Left={left_task:.4f}, Right={right_task:.4f}")
    
    with open(comparison_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Compare feature importance
    left_importance = left_results['feature_importance'].copy()
    right_importance = right_results['feature_importance'].copy()
    
    left_importance.columns = ['feature', 'left_importance']
    right_importance.columns = ['feature', 'right_importance']
    
    importance_comparison = pd.merge(left_importance, right_importance, on='feature', how='outer')
    importance_comparison = importance_comparison.fillna(0)
    importance_comparison['importance_diff'] = abs(
        importance_comparison['left_importance'] - importance_comparison['right_importance']
    )
    importance_comparison = importance_comparison.sort_values('importance_diff', ascending=False)
    
    importance_comparison.to_csv(comparison_dir / 'feature_importance_comparison.csv', index=False)
    
    logger.info(f"Saved to: {comparison_dir}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main function."""
    
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, args.hemisphere)
    optuna_n_jobs = get_optuna_n_jobs(args)
    
    logger.info("="*80)
    logger.info("HEMISPHERE CLASSIFICATION (XGBOOST MULTICLASS)")
    logger.info("="*80)
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Diagonal: {args.diagonal_strategy}")
    logger.info(f"Folds: {args.n_folds}")
    logger.info(f"Tune: {args.tune_hyperparams}")
    logger.info(f"Task test: {args.test_on_task}")
    logger.info("="*80)
    
    if args.sample:
        logger.warning(f"⚠️  TESTING MODE: {args.sample} subjects")
    
    try:
        if args.hemisphere == 'both':
            left_results = train_single_hemisphere('left', args, logger, optuna_n_jobs)
            right_results = train_single_hemisphere('right', args, logger, optuna_n_jobs)
            compare_hemispheres(left_results, right_results, args.output_dir, logger)
        else:
            results = train_single_hemisphere(args.hemisphere, args, logger, optuna_n_jobs)
        
        logger.info("\n" + "="*80)
        logger.info("✓ TRAINING COMPLETED")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()