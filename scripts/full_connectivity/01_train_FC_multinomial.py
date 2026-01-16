"""
01_train_fC_multinomial.py (CORRECTED VERSION: Fixed data leakage and other issues)

Train multinomial logistic regression on FULL brain connectivity (232 regions).
This establishes baseline performance for full-brain classification.

PREPROCESSING FLOW:
1. Diagonal imputation + Fisher Z BEFORE everything
2. StandardScaler WITHIN each CV fold (leak-free)
3. Optuna runs WITHIN each fold on fold's training data (with inner scaling)
4. [NEW] Final Optuna on ALL rest data before testing on task (most principled)

CRITICAL FIXES:
- Fixed data leakage: Optuna now receives unscaled data and scales within each trial
- Fixed network_mean imputation edge case
- Configurable n_jobs for Optuna
- Input validation for hyperparameters

Usage:
    python scripts/full_connectivity/01_train_fC_multinomial.py
    python scripts/full_connectivity/01_train_fC_multinomial.py --tune_hyperparams
    python scripts/full_connectivity/01_train_fC_multinomial.py --test_on_task
    python scripts/full_connectivity/01_train_fC_multinomial.py --tune_hyperparams --test_on_task
    
    # With sampling for testing:
    python scripts/full_connectivity/01_train_fC_multinomial.py --sample 30
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import your existing modules
from src.core.data import load_connectivity_data, extract_connection_columns
from src.core.features import extract_regions, reconstruct_matrices_from_dataframe, parse_networks


def setup_logging(output_dir: Path) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "training_full_connectivity_multinomial.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train full connectivity multinomial logistic regression')
    parser.add_argument('--rest_data', type=Path, default=project_root / 'data' / 'raw' / 'PIOP2_restingstate.csv')
    parser.add_argument('--task_data', type=Path, default=project_root / 'data' / 'raw' / 'PIOP1_gstroop.csv')
    parser.add_argument('--output_dir', type=Path, default=project_root / 'data' / 'results' / 'full_connectivity_analysis')
    parser.add_argument('--config_file', type=Path, default=project_root / 'configs' / 'FC_config.yaml')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--regularization_C', type=float, default=1.0)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--diagonal_strategy', type=str, default='region_mean', 
                       choices=['zero', 'region_mean', 'network_mean', 'global_mean'])
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--save_models', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--tune_hyperparams', action='store_true')
    parser.add_argument('--optuna_trials', type=int, default=50)
    parser.add_argument('--test_on_task', action='store_true')
    parser.add_argument('--final_optuna_trials', type=int, default=50)
    parser.add_argument('--optuna_n_jobs', type=int, default=None,
                       help='Number of parallel jobs for Optuna (default: same as n_jobs, max 32)')
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    if args.sample is not None and args.sample <= 0:
        raise ValueError(f"Sample size must be positive, got {args.sample}")
    
    if args.n_folds < 2:
        raise ValueError(f"Number of folds must be at least 2, got {args.n_folds}")
    
    if args.optuna_trials < 1:
        raise ValueError(f"Number of Optuna trials must be at least 1, got {args.optuna_trials}")
    
    if args.test_on_task and args.final_optuna_trials < 1:
        raise ValueError(f"Number of final Optuna trials must be at least 1, got {args.final_optuna_trials}")
    
    if args.regularization_C <= 0:
        raise ValueError(f"Regularization C must be positive, got {args.regularization_C}")
    
    if not args.rest_data.exists():
        raise FileNotFoundError(f"Rest data file not found: {args.rest_data}")
    
    if args.test_on_task and not args.task_data.exists():
        raise FileNotFoundError(f"Task data file not found: {args.task_data}")


def get_optuna_n_jobs(args):
    """Determine number of parallel jobs for Optuna."""
    if args.optuna_n_jobs is not None:
        return args.optuna_n_jobs
    
    # Use args.n_jobs but cap at 32
    if args.n_jobs == -1:
        return min(os.cpu_count() or 1, 32)
    elif args.n_jobs > 0:
        return min(args.n_jobs, 32)
    else:
        return 1


def create_region_info(region_list):
    """Create DataFrame with region information."""
    network_map = parse_networks(region_list)
    df = pd.DataFrame({
        'region_idx': range(len(region_list)),
        'region_name': region_list,
        'network': [network_map.get(r, 'Unknown') for r in region_list]
    })
    df['hemisphere'] = [
        'left' if r.startswith('LH_') or r.endswith('-lh') else 
        'right' if r.startswith('RH_') or r.endswith('-rh') else 
        'unknown' 
        for r in region_list
    ]
    return df


def sample_first_n_subjects(df, n_sample, logger):
    """Sample first n subjects from dataframe."""
    if n_sample > len(df):
        return df
    if n_sample <= 0:
        raise ValueError(f"Sample size must be positive, got {n_sample}")
    logger.info(f"Sampling first {n_sample} subjects.")
    return df.head(n_sample).copy()


def apply_diagonal_imputation(connectivity, region_info, strategy, logger):
    """
    Apply diagonal imputation strategy to connectivity matrices.
    
    FIXED: network_mean now properly handles edge cases without producing NaN.
    """
    logger.info(f"Applying diagonal imputation: {strategy}")
    n_sub, n_reg, _ = connectivity.shape
    imp = connectivity.copy()
    
    if strategy == 'zero':
        for i in range(n_sub):
            np.fill_diagonal(imp[i], 0.0)
    
    elif strategy == 'region_mean':
        for i in range(n_sub):
            for j in range(n_reg):
                row = imp[i, j, :]
                imp[i, j, j] = row[np.arange(n_reg) != j].mean()
    
    elif strategy == 'network_mean':
        if 'network' not in region_info.columns:
            logger.warning("Network column not found, falling back to region_mean")
            return apply_diagonal_imputation(connectivity, region_info, 'region_mean', logger)
        
        for i in range(n_sub):
            for j in range(n_reg):
                # Get indices of regions in same network, excluding diagonal
                network_mask = (region_info['network'] == region_info.iloc[j]['network']).values
                network_indices = np.where(network_mask)[0]
                network_indices = network_indices[network_indices != j]  # Exclude diagonal
                
                # FIXED: Properly handle edge cases
                if len(network_indices) > 0:
                    imp[i, j, j] = imp[i, j, network_indices].mean()
                else:
                    # Single region in network - use region mean as fallback
                    row = imp[i, j, :]
                    imp[i, j, j] = row[np.arange(n_reg) != j].mean()
    
    elif strategy == 'global_mean':
        for i in range(n_sub):
            np.fill_diagonal(imp[i], imp[i][~np.eye(n_reg, dtype=bool)].mean())
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Verify no NaN or Inf after imputation
    if np.any(np.isnan(imp)) or np.any(np.isinf(imp)):
        raise ValueError(f"NaN or Inf values after diagonal imputation with strategy '{strategy}'")
    
    return imp


def apply_fisher_z_transformation(connectivity, logger):
    """Apply Fisher Z-transformation to connectivity matrices."""
    logger.info("Applying Fisher Z-transformation")
    c_trans = np.arctanh(np.clip(connectivity, -0.999, 0.999))
    if np.any(np.isnan(c_trans)) or np.any(np.isinf(c_trans)):
        raise ValueError("NaN/Inf in Fisher Z transformation")
    return c_trans


def prepare_classification_data(connectivity, region_info, subject_ids):
    """Prepare data for classification by flattening connectivity matrices."""
    n_sub, n_reg, _ = connectivity.shape
    return (
        connectivity.reshape(n_sub * n_reg, n_reg),
        np.tile(np.arange(n_reg), n_sub),
        np.repeat(subject_ids, n_reg)
    )


def preprocess_fold_data(X_train, X_test, logger, verbose=False):
    """Scale data within a fold (leak-free)."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    
    if np.isnan(X_tr_s).any() or np.isnan(X_te_s).any():
        raise ValueError("NaN after scaling")
    
    if verbose:
        logger.info(f"  Scaled data - Train mean: {X_tr_s.mean():.4f}, Test mean: {X_te_s.mean():.4f}")
    
    return X_tr_s, X_te_s, scaler


def optimize_hyperparameters_optuna_fold(
    X_train_unscaled: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    fold_idx: int,
    n_trials: int,
    random_state: int,
    optuna_n_jobs: int,
    logger: logging.Logger,
    verbose: bool = False
) -> dict:
    """
    Optimize hyperparameters using Optuna within a single CV fold.
    """
    
    if verbose:
        logger.info(f"\n  {'─'*60}")
        logger.info(f"  FOLD {fold_idx} - HYPERPARAMETER OPTIMIZATION")
        logger.info(f"  {'─'*60}")
        logger.info(f"  Trials: {n_trials}")
        logger.info(f"  Training samples: {len(X_train_unscaled)}")
        logger.info(f"  Training subjects: {len(np.unique(groups_train))}")
        logger.info(f"  Features: {X_train_unscaled.shape[1]}")
        logger.info(f"  Classes: {len(np.unique(y_train))}")
    
    # Track best score for progress reporting
    best_score_tracker = {'score': 0.0, 'trial': 0}
    
    def objective(trial):
        """Optuna objective function."""
        
        # Suggest hyperparameters
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 500, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs','newton-cg'])
        
        # Use different split for each trial
        trial_seed = random_state + fold_idx * 1000 + trial.number
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=trial_seed)
        train_inner_idx, val_inner_idx = next(gss.split(X_train_unscaled, y_train, groups=groups_train))
        
        # Split UNSCALED data first
        X_train_inner_unscaled = X_train_unscaled[train_inner_idx]
        X_val_inner_unscaled = X_train_unscaled[val_inner_idx]
        y_train_inner = y_train[train_inner_idx]
        y_val_inner = y_train[val_inner_idx]
        
        # Scale AFTER splitting (leak-free)
        scaler_inner = StandardScaler()
        X_train_inner = scaler_inner.fit_transform(X_train_inner_unscaled)
        X_val_inner = scaler_inner.transform(X_val_inner_unscaled)
        
        # Verify no subject overlap
        train_subjects = set(groups_train[train_inner_idx])
        val_subjects = set(groups_train[val_inner_idx])
        assert len(train_subjects.intersection(val_subjects)) == 0, "Subject leakage in Optuna!"
        
        # Train model
        model = LogisticRegression(
            C=C,
            penalty='l2',
            multi_class='multinomial',
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            n_jobs=1,
            verbose=0
        )
        
        model.fit(X_train_inner, y_train_inner)
        y_pred = model.predict(X_val_inner)
        score = accuracy_score(y_val_inner, y_pred)
        
        # Report progress every trial if verbose
        if verbose:
            improvement = ""
            if score > best_score_tracker['score']:
                best_score_tracker['score'] = score
                best_score_tracker['trial'] = trial.number
                improvement = " ✓ NEW BEST"
            
            logger.info(f"    Trial {trial.number+1:3d}/{n_trials}: "
                       f"C={C:8.5f} {solver:13s} max_iter={max_iter:4d} → "
                       f"Val_Acc={score:.4f}{improvement}")
        
        return score
    
    # Create Optuna study
    optuna_start = time.time()
    
    # Set verbosity based on verbose flag
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state + fold_idx)
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,  # We'll show custom progress
        n_jobs=optuna_n_jobs
    )
    
    optuna_time = time.time() - optuna_start
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    if verbose:
        logger.info(f"\n  Optuna completed in {optuna_time:.2f}s")
        logger.info(f"  Best params: C={best_params['C']:.6f}, "
                   f"solver={best_params['solver']}, "
                   f"max_iter={best_params['max_iter']}")
        logger.info(f"  Best validation score: {best_score:.4f}")
        logger.info(f"  {'─'*60}\n")
    else:
        logger.info(f"  Optuna: C={best_params['C']:.4f}, {best_params['solver']}, "
                   f"max_iter={best_params['max_iter']} (val_acc={best_score:.4f}, {optuna_time:.1f}s)")
    
    # Add metadata
    best_params['_optuna_best_score'] = float(best_score)
    best_params['_optuna_time_seconds'] = float(optuna_time)
    
    return best_params


def select_best_fold_hyperparameters(
    fold_best_params: list,
    fold_metrics: list,
    logger: logging.Logger
) -> dict:
    """
    Select hyperparameters from the best performing fold.
    
    This ensures we use a hyperparameter combination that was actually tested
    and performed well, rather than creating an untested combination.
    """
    fold_accuracies = [m['accuracy'] for m in fold_metrics]
    best_fold_idx = np.argmax(fold_accuracies)
    
    best_params = fold_best_params[best_fold_idx].copy()
    best_params['_selection_method'] = 'best_fold'
    best_params['_best_fold_idx'] = int(best_fold_idx + 1)
    best_params['_best_fold_accuracy'] = float(fold_accuracies[best_fold_idx])
    
    # Add statistics about all folds
    best_params['_all_fold_accuracies'] = [float(acc) for acc in fold_accuracies]
    best_params['_accuracy_mean'] = float(np.mean(fold_accuracies))
    best_params['_accuracy_std'] = float(np.std(fold_accuracies))
    
    logger.info("\n" + "="*80)
    logger.info("BEST FOLD HYPERPARAMETER SELECTION")
    logger.info("="*80)
    logger.info(f"Selected hyperparameters from Fold {best_fold_idx + 1} (best performing)")
    logger.info(f"  C: {best_params['C']:.6f}")
    logger.info(f"  solver: {best_params['solver']}")
    logger.info(f"  max_iter: {best_params['max_iter']}")
    logger.info(f"  Fold accuracy: {best_params['_best_fold_accuracy']:.4f}")
    logger.info(f"\nAll fold accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    logger.info(f"Mean ± Std: {best_params['_accuracy_mean']:.4f} ± {best_params['_accuracy_std']:.4f}")
    logger.info("="*80 + "\n")
    
    return best_params


def optimize_on_full_rest_data(
    X_rest_unscaled: np.ndarray,
    y_rest: np.ndarray,
    groups_rest: np.ndarray,
    n_trials: int,
    random_state: int,
    optuna_n_jobs: int,
    logger: logging.Logger
) -> dict:
    """
    Run final hyperparameter optimization on ALL rest data before testing on task.
    """
    
    logger.info("\n" + "="*80)
    logger.info("FINAL HYPERPARAMETER OPTIMIZATION ON FULL REST DATA")
    logger.info("="*80)
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Samples: {len(X_rest_unscaled)}")
    logger.info(f"Subjects: {len(np.unique(groups_rest))}")
    logger.info(f"Features: {X_rest_unscaled.shape[1]}")
    logger.info(f"Classes: {len(np.unique(y_rest))}")
    logger.info("\nThis optimization uses ALL rest data to find the best")
    logger.info("hyperparameters for training the final model for task testing.")
    logger.info("="*80 + "\n")
    
    # Track progress
    best_score_tracker = {'score': 0.0, 'trial': 0}
    trial_counter = {'current': 0}
    
    def objective(trial):
        """Optuna objective function for full rest data."""
        
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 500, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg'])
        
        # Use different split for each trial
        trial_seed = random_state + 9999 + trial.number
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=trial_seed)
        train_idx, val_idx = next(gss.split(X_rest_unscaled, y_rest, groups=groups_rest))
        
        # Split UNSCALED data first
        X_train_inner_unscaled = X_rest_unscaled[train_idx]
        X_val_inner_unscaled = X_rest_unscaled[val_idx]
        y_train_inner = y_rest[train_idx]
        y_val_inner = y_rest[val_idx]
        
        # Scale AFTER splitting (leak-free)
        scaler_inner = StandardScaler()
        X_train_inner = scaler_inner.fit_transform(X_train_inner_unscaled)
        X_val_inner = scaler_inner.transform(X_val_inner_unscaled)
        
        # Verify no subject overlap
        train_subjects = set(groups_rest[train_idx])
        val_subjects = set(groups_rest[val_idx])
        assert len(train_subjects.intersection(val_subjects)) == 0, "Subject leakage!"
        
        model = LogisticRegression(
            C=C,
            penalty='l2',
            max_iter=max_iter,
            multi_class='multinomial',
            solver=solver,
            random_state=random_state,
            n_jobs=1,
            verbose=0
        )
        
        model.fit(X_train_inner, y_train_inner)
        y_pred = model.predict(X_val_inner)
        score = accuracy_score(y_val_inner, y_pred)
        
        # Report progress
        trial_counter['current'] += 1
        improvement = ""
        if score > best_score_tracker['score']:
            best_score_tracker['score'] = score
            best_score_tracker['trial'] = trial.number
            improvement = " ✓ NEW BEST"
        
        logger.info(f"  Trial {trial_counter['current']:3d}/{n_trials}: "
                   f"C={C:8.5f} {solver:13s} max_iter={max_iter:4d} → "
                   f"Val_Acc={score:.4f}{improvement}")
        
        return score
    
    # Create Optuna study
    optuna_start = time.time()
    
    # Enable INFO level to see progress
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state + 9999)
    )
    
    logger.info("Starting optimization...\n")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,  # We show custom progress
        n_jobs=optuna_n_jobs
    )
    
    optuna_time = time.time() - optuna_start
    
    best_params = study.best_params
    best_params['_optuna_best_score'] = float(study.best_value)
    best_params['_optuna_time_seconds'] = float(optuna_time)
    best_params['_selection_method'] = 'final_optuna_on_full_data'
    best_params['_n_trials'] = int(n_trials)
    
    logger.info(f"\n{'='*80}")
    logger.info("FINAL OPTIMIZATION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Best hyperparameters found:")
    logger.info(f"  C: {best_params['C']:.6f}")
    logger.info(f"  solver: {best_params['solver']}")
    logger.info(f"  max_iter: {best_params['max_iter']}")
    logger.info(f"  Validation accuracy: {study.best_value:.4f}")
    logger.info(f"  Found in trial: {best_score_tracker['trial'] + 1}")
    logger.info(f"  Optimization time: {optuna_time:.2f}s")
    logger.info(f"{'='*80}\n")
    
    return best_params

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> dict:
    """Compute comprehensive classification metrics."""
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'n_samples': int(len(y_true)),
        'n_classes': int(len(np.unique(y_true)))
    }
    
    # Top-k accuracy
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        for k in [3, 5, 10]:
            if k <= n_classes:
                top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
                top_k_correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
                metrics[f'top_{k}_accuracy'] = float(top_k_correct.mean())
    
    return metrics


def compute_per_region_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_info: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-region classification metrics."""
    
    n_regions = len(region_info)
    per_region = []
    
    for region_idx in range(n_regions):
        mask = (y_true == region_idx)
        if mask.sum() > 0:
            region_correct = (y_pred[mask] == region_idx).sum()
            region_total = mask.sum()
            region_acc = region_correct / region_total
        else:
            region_acc = 0.0
            region_total = 0
        
        per_region.append({
            'region_idx': region_idx,
            'region_name': region_info.iloc[region_idx]['region_name'],
            'network': region_info.iloc[region_idx]['network'],
            'hemisphere': region_info.iloc[region_idx].get('hemisphere', 'unknown'),
            'accuracy': region_acc,
            'n_samples': region_total
        })
    
    return pd.DataFrame(per_region)


def compute_network_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_info: pd.DataFrame
) -> pd.DataFrame:
    """Compute network-level classification metrics."""
    
    networks = region_info['network'].unique()
    network_metrics = []
    
    for network in networks:
        network_regions = region_info[region_info['network'] == network]['region_idx'].values
        
        mask = np.isin(y_true, network_regions)
        if mask.sum() > 0:
            network_correct = (y_pred[mask] == y_true[mask]).sum()
            network_total = mask.sum()
            network_acc = network_correct / network_total
        else:
            network_acc = 0.0
            network_total = 0
        
        network_metrics.append({
            'network': network,
            'accuracy': network_acc,
            'n_regions': len(network_regions),
            'n_samples': network_total
        })
    
    return pd.DataFrame(network_metrics).sort_values('accuracy', ascending=False)


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int
) -> np.ndarray:
    """Create confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def test_on_task_data(
    X_rest: np.ndarray,
    y_rest: np.ndarray,
    groups_rest: np.ndarray,
    region_info: pd.DataFrame,
    n_regions: int,
    best_params: dict,
    random_state: int,
    n_jobs: int,
    diagonal_strategy: str,
    task_data_path: Path,
    output_dir: Path,
    sample: int,
    logger: logging.Logger
) -> dict:
    """
    Train final model on ALL rest data using optimized params, then test on task data.
    """
    
    C = best_params['C']
    max_iter = best_params['max_iter']
    solver = best_params['solver']
    
    logger.info("\n" + "="*80)
    logger.info("TESTING ON TASK DATA (GENDER STROOP)")
    logger.info("="*80)
    logger.info(f"Training final model on ALL resting-state data")
    logger.info(f"Using optimized hyperparameters: C={C:.6f}, solver={solver}, max_iter={max_iter}")
    logger.info(f"Then testing on task data to measure generalization\n")
    
    # =========================================================================
    # STEP 1: Load and preprocess TASK data
    # =========================================================================
    
    logger.info("Step 1: Loading task data (TESTING)...")
    df_task = load_connectivity_data(str(task_data_path))
    
    if sample is not None:
        df_task = sample_first_n_subjects(df_task, sample, logger)
    
    task_connection_columns = extract_connection_columns(df_task)
    subject_ids_task = df_task.iloc[:, 0].values
    
    # Get region mapping from task data
    _, region_to_idx_task, _ = extract_regions(task_connection_columns)
    
    logger.info(f"  Task data: {len(df_task)} subjects")
    
    # Reconstruct connectivity matrices
    task_connectivity = reconstruct_matrices_from_dataframe(
        df_task, task_connection_columns, region_to_idx_task, n_regions
    )
    
    # Preprocess task data (SAME pipeline as rest)
    task_connectivity = apply_diagonal_imputation(
        connectivity=task_connectivity,
        region_info=region_info,
        strategy=diagonal_strategy,
        logger=logger
    )
    
    task_connectivity = apply_fisher_z_transformation(
        connectivity=task_connectivity,
        logger=logger
    )
    
    X_task, y_task, groups_task = prepare_classification_data(
        connectivity=task_connectivity,
        region_info=region_info,
        subject_ids=subject_ids_task
    )
    
    logger.info(f"  Task features: {X_task.shape}")
    
    # =========================================================================
    # STEP 2: Scale rest and task data together
    # =========================================================================
    
    logger.info("\nStep 2: Scaling data...")
    
    scaler_final = StandardScaler()
    X_rest_scaled = scaler_final.fit_transform(X_rest)
    X_task_scaled = scaler_final.transform(X_task)
    
    logger.info(f"  Rest data (scaled): mean={X_rest_scaled.mean():.4f}, std={X_rest_scaled.std():.4f}")
    logger.info(f"  Task data (scaled): mean={X_task_scaled.mean():.4f}, std={X_task_scaled.std():.4f}")
    
    # =========================================================================
    # STEP 3: Train final model on ALL rest data
    # =========================================================================
    
    logger.info("\nStep 3: Training final model on ALL resting-state data...")
    
    final_model = LogisticRegression(
        C=C,
        penalty='l2',
        multi_class='multinomial',
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0
    )
    
    train_start = time.time()
    final_model.fit(X_rest_scaled, y_rest)
    train_time = time.time() - train_start
    
    logger.info(f"  Model trained in {train_time:.2f}s")
    
    # Evaluate on rest data (sanity check)
    y_rest_pred = final_model.predict(X_rest_scaled)
    rest_accuracy = accuracy_score(y_rest, y_rest_pred)
    logger.info(f"  Rest data accuracy (training set): {rest_accuracy:.4f}")
    
    # =========================================================================
    # STEP 4: Test on task data
    # =========================================================================
    
    logger.info("\nStep 4: Testing on task data...")
    
    y_task_pred = final_model.predict(X_task_scaled)
    y_task_proba = final_model.predict_proba(X_task_scaled)
    
    # =========================================================================
    # STEP 5: Compute metrics
    # =========================================================================
    
    logger.info("\nStep 5: Computing task metrics...")
    
    task_metrics = compute_classification_metrics(
        y_true=y_task,
        y_pred=y_task_pred,
        y_proba=y_task_proba
    )
    
    task_per_region = compute_per_region_metrics(
        y_true=y_task,
        y_pred=y_task_pred,
        region_info=region_info
    )
    
    task_network = compute_network_level_metrics(
        y_true=y_task,
        y_pred=y_task_pred,
        region_info=region_info
    )
    
    task_confusion = create_confusion_matrix(
        y_true=y_task,
        y_pred=y_task_pred,
        n_classes=n_regions
    )
    
    # =========================================================================
    # STEP 6: Save results
    # =========================================================================
    
    task_output_dir = output_dir / "task_testing"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving task testing results to: {task_output_dir}")
    
    # Save predictions
    np.save(task_output_dir / 'task_predictions.npy', y_task_pred)
    np.save(task_output_dir / 'task_probabilities.npy', y_task_proba)
    np.save(task_output_dir / 'task_true_labels.npy', y_task)
    np.save(task_output_dir / 'task_confusion_matrix.npy', task_confusion)
    
    # Save metrics
    task_summary = {
        'rest_train_accuracy': float(rest_accuracy),
        'task_test_accuracy': float(task_metrics['accuracy']),
        'task_top_5_accuracy': float(task_metrics.get('top_5_accuracy', 0)),
        'accuracy_drop': float(rest_accuracy - task_metrics['accuracy']),
        'hyperparameters': best_params,
        'n_rest_subjects': int(len(np.unique(groups_rest))),
        'n_task_subjects': int(len(np.unique(groups_task))),
        'n_rest_samples': int(len(X_rest)),
        'n_task_samples': int(len(X_task)),
        'n_regions': int(n_regions)
    }
    
    with open(task_output_dir / 'task_testing_summary.json', 'w') as f:
        json.dump(task_summary, f, indent=2)
    
    task_per_region.to_csv(task_output_dir / 'task_per_region_metrics.csv', index=False)
    task_network.to_csv(task_output_dir / 'task_network_metrics.csv', index=False)
    
    # Save final model
    import pickle
    with open(task_output_dir / 'final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open(task_output_dir / 'final_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_final, f)
    
    # =========================================================================
    # STEP 7: Report results
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("TASK TESTING RESULTS")
    logger.info("="*80)
    logger.info(f"Rest (training) accuracy: {rest_accuracy:.4f}")
    logger.info(f"Task (testing) accuracy:  {task_metrics['accuracy']:.4f}")
    logger.info(f"Accuracy drop:            {task_summary['accuracy_drop']:.4f} "
                f"({task_summary['accuracy_drop']/rest_accuracy*100:.1f}%)")
    logger.info(f"Task top-5 accuracy:      {task_metrics.get('top_5_accuracy', 0):.4f}")
    logger.info(f"\nBest task network: {task_network.iloc[task_network['accuracy'].idxmax()]['network']} "
                f"({task_network['accuracy'].max():.4f})")
    logger.info(f"Worst task network: {task_network.iloc[task_network['accuracy'].idxmin()]['network']} "
                f"({task_network['accuracy'].min():.4f})")
    logger.info("="*80 + "\n")
    
    return {
        'task_metrics': task_metrics,
        'task_per_region': task_per_region,
        'task_network': task_network,
        'task_summary': task_summary,
        'final_model': final_model,
        'final_scaler': scaler_final
    }


def train_full_connectivity(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """
    Train multinomial model for full brain connectivity (232 regions).
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING FULL BRAIN CONNECTIVITY (232 REGIONS)")
    logger.info(f"{'='*80}\n")
    
    # Get Optuna n_jobs
    optuna_n_jobs = get_optuna_n_jobs(args)
    logger.info(f"Optuna parallel jobs: {optuna_n_jobs}")
    
    # Create output directory
    output_dir = args.output_dir / "multinomial"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    
    logger.info("Loading full connectivity data...")
    df_train = load_connectivity_data(str(args.rest_data))
    
    # Sample subjects if specified
    if args.sample is not None:
        df_train = sample_first_n_subjects(df_train, args.sample, logger)
        logger.warning(f"⚠️  TESTING MODE: Using first {args.sample} subjects only")
    
    # Extract connection info
    connection_columns = extract_connection_columns(df_train)
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    
    # Get subject IDs
    subject_ids = df_train.iloc[:, 0].values
    n_subjects = len(subject_ids)
    
    logger.info(f"Data loaded:")
    logger.info(f"  Subjects: {n_subjects}")
    logger.info(f"  Regions: {n_regions}")
    logger.info(f"  Connections: {len(connection_columns)}")
    
    # Reconstruct full connectivity matrices
    logger.info("\nReconstructing connectivity matrices...")
    connectivity = reconstruct_matrices_from_dataframe(
        df_train, connection_columns, region_to_idx, n_regions
    )
    logger.info(f"  Connectivity shape: {connectivity.shape}")
    
    # Create region info
    region_info = create_region_info(region_list)
    logger.info(f"  Networks found: {region_info['network'].unique().tolist()}")
    
    # ==========================================================================
    # PREPROCESSING (BEFORE CV)
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING (BEFORE CROSS-VALIDATION)")
    logger.info("="*80)
    
    # Step 1: Diagonal imputation
    connectivity = apply_diagonal_imputation(
        connectivity=connectivity,
        region_info=region_info,
        strategy=args.diagonal_strategy,
        logger=logger
    )
    
    # Step 2: Fisher Z transformation
    connectivity = apply_fisher_z_transformation(
        connectivity=connectivity,
        logger=logger
    )
    
    # Step 3: Prepare classification data
    logger.info("\nPreparing classification data (flattening to features)...")
    X, y, groups = prepare_classification_data(
        connectivity=connectivity,
        region_info=region_info,
        subject_ids=subject_ids
    )
    
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    
    logger.info(f"Classification data prepared:")
    logger.info(f"  Samples (X): {X.shape}")
    logger.info(f"  Labels (y): {y.shape}")
    logger.info(f"  Classes: {n_classes}")
    logger.info(f"  Samples per subject: {n_samples // n_subjects}")
    
    # Validate
    assert X.shape[0] == len(y), "Mismatch between X and y"
    assert X.shape[0] == len(groups), "Mismatch between X and groups"
    assert n_classes == n_regions, f"Class count ({n_classes}) != region count ({n_regions})"
    
    logger.info("\n✓ Preprocessing completed")
    logger.info("="*80)
    
    # ==========================================================================
    # CROSS-VALIDATION (WITH FOLD-WISE OPTUNA)
    # ==========================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION WITH FOLD-WISE HYPERPARAMETER TUNING")
    logger.info(f"{'='*80}")
    
    if args.tune_hyperparams:
        logger.info(f"Optuna will run INSIDE each fold ({args.optuna_trials} trials per fold)")
        logger.info(f"This prevents information leakage and provides fold-specific tuning")
        logger.info(f"CRITICAL: Optuna receives UNSCALED data and scales within each trial")
    else:
        logger.info(f"Using default hyperparameters (no tuning)")
        default_C = args.regularization_C if args.regularization_C is not None else 1.0
        logger.info(f"  C={default_C}, solver=lbfgs, max_iter={args.max_iter}")
    
    logger.info(f"\nRunning {args.n_folds}-fold GroupKFold cross-validation...\n")
    
    # Set up cross-validation
    gkf = GroupKFold(n_splits=args.n_folds)
    
    # Storage
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_fold_indices = []
    fold_models = [] if args.save_models else None
    fold_metrics = []
    fold_best_params = []
    
    # CV loop
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        fold_start = time.time()
        logger.info(f"{'='*80}")
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"{'='*80}")
        logger.info(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Split data (using UNSCALED X from preprocessing)
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
        
        # Verify no subject leakage
        train_subjects = set(groups_train)
        test_subjects = set(groups_test)
        assert len(train_subjects.intersection(test_subjects)) == 0, "Subject leakage!"
        
        logger.info(f"  Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
        
        # ======================================================================
        # HYPERPARAMETER TUNING (IF ENABLED) - RUNS ON UNSCALED FOLD DATA
        # ======================================================================
        
        if args.tune_hyperparams:
            # CRITICAL FIX: Pass UNSCALED data to Optuna
            best_params = optimize_hyperparameters_optuna_fold(
                X_train_unscaled=X_train,  # UNSCALED!
                y_train=y_train,
                groups_train=groups_train,
                fold_idx=fold_idx + 1,
                n_trials=args.optuna_trials,
                random_state=args.random_state,
                optuna_n_jobs=optuna_n_jobs,
                logger=logger,
                verbose=args.verbose
            )
            
            C = best_params['C']
            max_iter = best_params['max_iter']
            solver = best_params['solver']
            
            fold_best_params.append(best_params)
            
        else:
            # Use default parameters
            C = args.regularization_C if args.regularization_C is not None else 1.0
            max_iter = args.max_iter
            solver = 'lbfgs'
            
            best_params = {
                'C': C,
                'max_iter': max_iter,
                'solver': solver
            }
            fold_best_params.append(best_params)
        
        # ======================================================================
        # SCALE DATA FOR FINAL TRAINING (LEAK-FREE)
        # ======================================================================
        
        logger.info(f"  Scaling data within fold...")
        X_train_scaled, X_test_scaled, fold_scaler = preprocess_fold_data(
            X_train=X_train,
            X_test=X_test,
            logger=logger,
            verbose=args.verbose
        )
        
        # ======================================================================
        # TRAIN MODEL WITH BEST/DEFAULT PARAMS
        # ======================================================================
        
        logger.info(f"  Training with C={C:.6f}, solver={solver}, max_iter={max_iter}...")
        
        model = LogisticRegression(
            C=C,
            penalty='l2',
            multi_class='multinomial',
            max_iter=max_iter,
            solver=solver,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        logger.info("  Predicting on test set...")
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Metrics
        fold_acc = accuracy_score(y_test, y_pred)
        
        fold_metric_dict = {
            'fold': fold_idx + 1,
            'accuracy': fold_acc,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_train_subjects': len(train_subjects),
            'n_test_subjects': len(test_subjects),
            'hyperparameters': {
                'C': float(C),
                'max_iter': int(max_iter),
                'solver': solver
            }
        }
        
        if args.tune_hyperparams:
            fold_metric_dict['optuna_validation_score'] = best_params['_optuna_best_score']
            fold_metric_dict['optuna_time'] = best_params['_optuna_time_seconds']
        
        fold_metrics.append(fold_metric_dict)
        
        fold_time = time.time() - fold_start
        logger.info(f"\n  ✓ Fold {fold_idx + 1} completed:")
        logger.info(f"    Accuracy: {fold_acc:.4f}")
        logger.info(f"    Time: {fold_time:.2f}s\n")
        
        # Store results
        all_predictions.extend(y_pred)
        all_probabilities.append(y_proba)
        all_true_labels.extend(y_test)
        all_fold_indices.extend([fold_idx + 1] * len(y_test))
        
        if args.save_models:
            fold_models.append({
                'fold': fold_idx + 1,
                'model': model,
                'scaler': fold_scaler,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'hyperparameters': best_params
            })
    
    total_time = time.time() - start_time
    logger.info(f"{'='*80}")
    logger.info(f"Cross-validation completed in {total_time:.2f}s")
    logger.info(f"{'='*80}\n")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    all_true_labels = np.array(all_true_labels)
    all_fold_indices = np.array(all_fold_indices)
    
    # ==========================================================================
    # SELECT BEST HYPERPARAMETERS (for reference only if task testing disabled)
    # ==========================================================================
    
    if args.tune_hyperparams and fold_best_params:
        # Select best fold params (for logging/reference)
        reference_params = select_best_fold_hyperparameters(
            fold_best_params, fold_metrics, logger
        )
    else:
        reference_params = {
            'C': args.regularization_C if args.regularization_C is not None else 1.0,
            'max_iter': args.max_iter,
            'solver': 'lbfgs'
        }
    
    # ==========================================================================
    # COMPUTE OVERALL METRICS
    # ==========================================================================
    
    logger.info("Computing overall metrics...")
    overall_metrics = compute_classification_metrics(
        y_true=all_true_labels,
        y_pred=all_predictions,
        y_proba=all_probabilities
    )
    
    overall_metrics['best_hyperparameters'] = reference_params
    overall_metrics['fold_hyperparameters'] = fold_best_params if args.tune_hyperparams else None
    overall_metrics['preprocessing'] = {
        'diagonal_strategy': args.diagonal_strategy,
        'fisher_z_applied': True,
        'standardize_per_fold': True,
        'optuna_per_fold': args.tune_hyperparams,
        'optuna_receives_unscaled_data': True  # NEW: Document the fix
    }
    
    logger.info(f"\n{'='*80}")
    logger.info(f"OVERALL CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Mean CV Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy: {overall_metrics.get('top_5_accuracy', 'N/A')}")
    
    if args.tune_hyperparams:
        logger.info(f"\nHyperparameter tuning: Enabled (Optuna per fold, leak-free)")
        logger.info(f"  Trials per fold: {args.optuna_trials}")
        logger.info(f"  Best fold params (reference): C={reference_params['C']:.6f}, "
                   f"solver={reference_params['solver']}, max_iter={reference_params['max_iter']}")
    else:
        logger.info(f"\nHyperparameter tuning: Disabled")
        logger.info(f"  Fixed params: C={reference_params['C']:.6f}, "
                   f"solver={reference_params['solver']}, max_iter={reference_params['max_iter']}")
    
    logger.info(f"{'='*80}\n")
    
    # Compute per-region metrics
    logger.info("Computing per-region metrics...")
    per_region_metrics = compute_per_region_metrics(
        y_true=all_true_labels,
        y_pred=all_predictions,
        region_info=region_info
    )
    
    # Compute network metrics
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
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    
    logger.info("\nSaving results...")
    
    np.save(output_dir / 'cv_predictions.npy', all_predictions)
    np.save(output_dir / 'cv_probabilities.npy', all_probabilities)
    np.save(output_dir / 'cv_true_labels.npy', all_true_labels)
    np.save(output_dir / 'cv_fold_indices.npy', all_fold_indices)
    np.save(output_dir / 'confusion_matrix.npy', confusion_mat)
    
    with open(output_dir / 'overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    with open(output_dir / 'fold_metrics.json', 'w') as f:
        json.dump(fold_metrics, f, indent=2)
    
    per_region_metrics.to_csv(output_dir / 'per_region_metrics.csv', index=False)
    network_metrics.to_csv(output_dir / 'network_metrics.csv', index=False)
    region_info.to_csv(output_dir / 'region_info.csv', index=False)
    
    if args.save_models and fold_models is not None:
        import pickle
        with open(output_dir / 'fold_models.pkl', 'wb') as f:
            pickle.dump(fold_models, f)
    
    logger.info(f"Results saved to: {output_dir}")
    
    # ==========================================================================
    # TASK TESTING (IF ENABLED) - WITH FINAL OPTUNA ON FULL REST DATA
    # ==========================================================================
    task_results = None
    if args.test_on_task:
        logger.info("\n" + "="*80)
        logger.info("PREPARING FOR TASK TESTING")
        logger.info("="*80)
        
        # FIXED: Move conditional logic BEFORE misleading messages
        if args.tune_hyperparams:
            # Only show these messages when actually optimizing
            logger.info("Running FINAL hyperparameter optimization on ALL rest data")
            logger.info("This finds optimal parameters specifically for the full training set")
            logger.info("CRITICAL: Optuna receives UNSCALED data and scales within each trial")
            logger.info("="*80 + "\n")
            
            # Run final Optuna on UNSCALED full rest data
            final_best_params = optimize_on_full_rest_data(
                X_rest_unscaled=X,  # UNSCALED!
                y_rest=y,
                groups_rest=groups,
                n_trials=args.final_optuna_trials,
                random_state=args.random_state,
                optuna_n_jobs=optuna_n_jobs,
                logger=logger
            )
        else:
            # Use default parameters without optimization
            logger.info("Using default hyperparameters for task testing (no optimization)")
            logger.info("="*80 + "\n")
            
            final_best_params = reference_params.copy()
            final_best_params['_selection_method'] = 'default_no_tuning'
            
            logger.info(f"Task testing parameters: C={final_best_params['C']:.6f}, "
                    f"solver={final_best_params['solver']}, "
                    f"max_iter={final_best_params['max_iter']}")
        
        # Now test on task using these params
        task_results = test_on_task_data(
            X_rest=X,  # Pass UNSCALED rest data
            y_rest=y,
            groups_rest=groups,
            region_info=region_info,
            n_regions=n_regions,
            best_params=final_best_params,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            diagonal_strategy=args.diagonal_strategy,
            task_data_path=args.task_data,
            output_dir=args.output_dir,
            sample=args.sample,
            logger=logger
        )

    # Return results
    results = {
        'n_subjects': n_subjects,
        'n_regions': n_regions,
        'n_classes': n_classes,
        'n_samples': n_samples,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'true_labels': all_true_labels,
        'fold_indices': all_fold_indices,
        'confusion_matrix': confusion_mat,
        'overall_metrics': overall_metrics,
        'fold_metrics': fold_metrics,
        'per_region_metrics': per_region_metrics,
        'network_metrics': network_metrics,
        'region_info': region_info,
        'output_dir': output_dir,
        'fold_best_params': fold_best_params if args.tune_hyperparams else None,
        'reference_params': reference_params,
        'task_results': task_results
    }

    return results

def main():
    """Main function."""
    
    args = parse_arguments()
    
    # Validate arguments
    try:
        validate_arguments(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output_dir)
    
    logger.info("="*80)
    logger.info("FULL BRAIN CONNECTIVITY CLASSIFICATION (CORRECTED VERSION)")
    logger.info("232 Regions | 224 Subjects (PIOP2 Rest)")
    logger.info("="*80)
    logger.info("="*80)
    logger.info("Preprocessing: Diagonal imputation + Fisher Z → StandardScaler per fold")
    if args.tune_hyperparams:
        logger.info(f"Hyperparameter Tuning: Enabled (Optuna INSIDE each fold, leak-free)")
        logger.info(f"  • Trials per fold: {args.optuna_trials}")
    else:
        logger.info("Hyperparameter Tuning: Disabled (using defaults)")
    if args.test_on_task:
        logger.info("Task Testing: Enabled (Final Optuna on full rest data, leak-free)")
        logger.info(f"  • Final optimization trials: {args.final_optuna_trials}")
        logger.info(f"  • Most principled approach for task testing")
    logger.info("="*80)
    
    if args.sample:
        logger.warning(f"\n⚠️  TESTING MODE: {args.sample} subjects only\n")
    
    try:
        results = train_full_connectivity(args, logger)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"\nFinal Results:")
        logger.info(f"  • CV Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        logger.info(f"  • Top-5 Accuracy: {results['overall_metrics'].get('top_5_accuracy', 'N/A')}")
        
        if args.tune_hyperparams:
            logger.info(f"\nFold-specific hyperparameters:")
            for i, params in enumerate(results['fold_best_params'], 1):
                logger.info(f"  Fold {i}: C={params['C']:.4f}, {params['solver']}, max_iter={params['max_iter']}")
            
            ref = results['reference_params']
            logger.info(f"\nBest fold params (reference): C={ref['C']:.4f}, "
                       f"solver={ref['solver']}, max_iter={ref['max_iter']}")
        
        if results.get('task_results'):
            task_acc = results['task_results']['task_summary']['task_test_accuracy']
            rest_acc = results['task_results']['task_summary']['rest_train_accuracy']
            drop = results['task_results']['task_summary']['accuracy_drop']
            final_params = results['task_results']['task_summary']['hyperparameters']
            
            logger.info(f"\nTask Testing (with final Optuna on full rest data, leak-free):")
            logger.info(f"  • Final optimized params: C={final_params['C']:.6f}, "
                       f"solver={final_params['solver']}, max_iter={final_params['max_iter']}")
            logger.info(f"  • Rest (train) accuracy: {rest_acc:.4f}")
            logger.info(f"  • Task (test) accuracy: {task_acc:.4f}")
            logger.info(f"  • Accuracy drop: {drop:.4f} ({drop/rest_acc*100:.1f}%)")
        
        logger.info(f"\nResults saved to: {results['output_dir']}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()