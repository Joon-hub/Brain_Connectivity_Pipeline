"""
02_train_fC_one_vs_rest.py

Train One-vs-Rest logistic regression on FULL brain connectivity (232 regions).
Each region gets a binary classifier (Region X vs All Others), then predictions are aggregated.

Workflow: Load Data → Diagonal Imputation + Fisher Z → Per-Fold Optuna → CV with OvR Training → Task Testing

Usage:
    python scripts/full_connectivity/02_train_fC_one_vs_rest.py --tune_hyperparams --test_on_task
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
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning
import optuna
from optuna.samplers import TPESampler

# Suppress convergence warnings (we're handling this via hyperparameter tuning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import custom modules
from src.core.data import load_connectivity_data, extract_connection_columns
from src.core.features import extract_regions, reconstruct_matrices_from_dataframe, parse_networks

# ==============================================================================
# SETUP & CONFIGURATION
# ==============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging."""
    log_file = output_dir / "training_full_connectivity_one_vs_rest.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train full connectivity One-vs-Rest logistic regression')
    
    # Core settings
    parser.add_argument('--rest_data', type=Path, default=project_root / 'data' / 'raw' / 'PIOP2_restingstate.csv')
    parser.add_argument('--task_data', type=Path, default=project_root / 'data' / 'raw' / 'PIOP1_gstroop.csv')
    parser.add_argument('--output_dir', type=Path, default=project_root / 'data' / 'results' / 'full_connectivity_analysis')
    
    # Model settings
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--regularization_C', type=float, default=1.0)
    parser.add_argument('--diagonal_strategy', type=str, default='random', 
                       choices=['zero', 'region_mean', 'network_mean', 'global_mean', 'random'])
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    
    # Execution flags
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--sample', type=int, default=None)
    
    # Optimization
    parser.add_argument('--tune_hyperparams', action='store_true')
    parser.add_argument('--optuna_trials', type=int, default=50)
    parser.add_argument('--test_on_task', action='store_true')
    parser.add_argument('--optuna_n_jobs', type=int, default=None)
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate input arguments."""
    if args.sample is not None and args.sample <= 0:
        raise ValueError(f"Sample size must be positive, got {args.sample}")
    if args.n_folds < 2:
        raise ValueError(f"Number of folds must be at least 2, got {args.n_folds}")
    if args.test_on_task and not args.task_data.exists():
        raise FileNotFoundError(f"Task data file not found: {args.task_data}")

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

def create_region_info(region_list):
    """Creates a metadata DataFrame mapping Region Indices to Names and Networks."""
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

def sample_first_n_subjects(df: pd.DataFrame, n_sample: int, logger: logging.Logger) -> pd.DataFrame:
    """Sample first n subjects for testing."""
    total_subjects = len(df)
    if n_sample > total_subjects:
        logger.warning(f"Requested {n_sample} exceeds {total_subjects}. Using all subjects.")
        return df
    if n_sample <= 0:
        raise ValueError(f"Sample size must be positive, got {n_sample}")
    
    subject_ids = df.iloc[:, 0].values
    logger.info(f"\nSAMPLING MODE: First {n_sample}/{total_subjects} subjects")
    logger.info(f"Selected: {', '.join(map(str, subject_ids[:min(10, n_sample)]))}" + 
                (f"... (+{n_sample-10} more)" if n_sample > 10 else ""))
    
    return df.head(n_sample).copy()

def apply_diagonal_imputation(connectivity: np.ndarray, region_info: pd.DataFrame, 
                             strategy: str, logger: logging.Logger, seed: int | None = None) -> np.ndarray:
    """Apply diagonal imputation to connectivity matrices."""
    n_subjects, n_regions, _ = connectivity.shape
    imp = connectivity.copy()
    
    rng = np.random.default_rng(seed if seed is not None else 42)
    
    if strategy == 'zero':
        for i in range(n_subjects):
            np.fill_diagonal(imp[i], 0.0)
    
    elif strategy == 'random':
        for i in range(n_subjects):
            off_diag_mask = ~np.eye(n_regions, dtype=bool)
            off_diag_values = imp[i][off_diag_mask]
            min_val = off_diag_values.min()
            max_val = off_diag_values.max()
            
            for j in range(n_regions):
                imp[i, j, j] = rng.uniform(min_val, max_val)
            
    elif strategy == 'region_mean':
        for i in range(n_subjects):
            for j in range(n_regions):
                mask = np.ones(n_regions, dtype=bool)
                mask[j] = False
                imp[i, j, j] = imp[i, j, mask].mean()
                
    elif strategy == 'network_mean':
        if 'network' not in region_info.columns:
            logger.warning("  'network' column not found, falling back to region_mean")
            return apply_diagonal_imputation(connectivity, region_info, 'region_mean', logger, seed)
        for i in range(n_subjects):
            for j in range(n_regions):
                network_mask = (region_info['network'] == region_info.iloc[j]['network']).values
                network_indices = np.where(network_mask)[0]
                network_indices = network_indices[network_indices != j]
                if len(network_indices) > 0:
                    imp[i, j, j] = imp[i, j, network_indices].mean()
                else:
                    mask = np.ones(n_regions, dtype=bool)
                    mask[j] = False
                    imp[i, j, j] = imp[i, j, mask].mean()
                    
    elif strategy == 'global_mean':
        for i in range(n_subjects):
            mask = ~np.eye(n_regions, dtype=bool)
            np.fill_diagonal(imp[i], imp[i][mask].mean())
    else:
        raise ValueError(f"Unknown diagonal strategy: {strategy}")
    
    return imp

def apply_fisher_z_transformation(connectivity: np.ndarray) -> np.ndarray:
    """Apply Fisher Z-transformation to connectivity matrices."""
    c_trans = np.arctanh(np.clip(connectivity, -0.999, 0.999))
    
    if np.any(np.isnan(c_trans)) or np.any(np.isinf(c_trans)):
        raise ValueError("NaN/Inf detected after Fisher Z transformation")
    
    return c_trans

def prepare_classification_data(connectivity: np.ndarray, region_info: pd.DataFrame, 
                               subject_ids: np.ndarray) -> tuple:
    """Prepare data for classification."""
    n_subjects, n_regions, _ = connectivity.shape
    X = connectivity.reshape(n_subjects * n_regions, n_regions)
    y = np.tile(np.arange(n_regions), n_subjects)
    groups = np.repeat(subject_ids, n_regions)
    return X, y, groups

def preprocess_fold_data(X_train: np.ndarray, X_test: np.ndarray, 
                        logger: logging.Logger, verbose: bool = False) -> tuple:
    """Standardize data within fold (leak-free)."""
    if verbose:
        logger.info(f"  Scaling: Train {X_train.shape}, Test {X_test.shape}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
        raise ValueError("NaN/Inf in scaled training data")
    if np.any(np.isnan(X_test_scaled)) or np.any(np.isinf(X_test_scaled)):
        raise ValueError("NaN/Inf in scaled test data")
    
    if verbose:
        logger.info(f"  Train mean={X_train_scaled.mean():.4f}, Test mean={X_test_scaled.mean():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler

# ==============================================================================
# ONE-VS-REST TRAINING
# ==============================================================================

def train_ovr_classifiers(X_train: np.ndarray, y_train: np.ndarray, n_regions: int,
                         C: float, max_iter: int, solver: str, 
                         penalty: str, tol: float,
                         random_state: int, n_jobs: int, 
                         logger: logging.Logger, verbose: bool = False) -> list:
    """Train n_regions binary One-vs-Rest classifiers."""
    if verbose:
        logger.info(f"  Training {n_regions} OvR classifiers...")
        logger.info(f"    C={C:.6f}, {solver}, penalty={penalty}, tol={tol:.6f}")
    
    models = []
    for region_idx in range(n_regions):
        y_binary = (y_train == region_idx).astype(int)
        
        base_model = LogisticRegression(
            C=C, 
            penalty=penalty, 
            max_iter=max_iter, 
            solver=solver,
            tol=tol,
            random_state=random_state, 
            n_jobs=1, 
            verbose=0
        )
        model = OneVsRestClassifier(base_model, n_jobs=1)
        model.fit(X_train, y_binary)
        models.append(model)
    
    return models

def predict_ovr_probabilities(models: list, X_test: np.ndarray, n_regions: int) -> np.ndarray:
    n_samples = X_test.shape[0]
    probabilities = np.zeros((n_samples, n_regions))
    
    for region_idx, model in enumerate(models):
        probabilities[:, region_idx] = model.predict_proba(X_test)[:, 1]
    
    return probabilities

def aggregate_ovr_predictions(probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities, axis=1)

# ==============================================================================
# OPTUNA HYPERPARAMETER TUNING - PER-FOLD
# ==============================================================================

def optimize_hyperparameters_ovr_fold(
    X_train_unscaled: np.ndarray | None, y_train: np.ndarray, groups_train: np.ndarray,
    n_regions: int, n_trials: int, random_state: int, optuna_n_jobs: int,
    logger: logging.Logger, verbose: bool = False,
    original_connectivity: np.ndarray | None = None,
    subject_ids: np.ndarray | None = None,
    region_info: pd.DataFrame | None = None,
    is_random: bool = False,
    fold_idx: int = 0
) -> dict:
    """Run Optuna hyperparameter optimization for a single CV fold."""
    logger.info(f"\n  Hyperparameter optimization for Fold {fold_idx + 1}")
    logger.info(f"    {n_trials} trials, each training {n_regions} classifiers")
    
    best_score_tracker = {'score': 0.0, 'trial': 0}
    
    def objective(trial):
        C = trial.suggest_float('C', 0.001, 0.05, log=True)
        solver = trial.suggest_categorical('solver', ['sag', 'saga'])

        # Set max_iter based on solver
        max_iter = trial.suggest_int('max_iter', 100, 2000)

        penalty = 'l2'
        tol = trial.suggest_float('tol', 1e-4, 1e-1, log=True)
        
        # Use larger offsets to minimise seed collision risk
        trial_seed = random_state + fold_idx * 100000 + trial.number * 10000
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=trial_seed)
        train_inner_idx, val_inner_idx = next(gss.split(range(len(groups_train)), groups=groups_train))
        
        if not is_random:
            scaler_inner = StandardScaler()
            X_train_inner = scaler_inner.fit_transform(X_train_unscaled[train_inner_idx])
            X_val_inner = scaler_inner.transform(X_train_unscaled[val_inner_idx])
            y_inner_train = y_train[train_inner_idx]
            y_inner_val = y_train[val_inner_idx]
        else:
            train_subjects = np.unique(groups_train[train_inner_idx])
            val_subjects = np.unique(groups_train[val_inner_idx])
            
            train_subject_idx = np.isin(subject_ids, train_subjects)
            val_subject_idx = np.isin(subject_ids, val_subjects)
            
            train_conn = original_connectivity[train_subject_idx].copy()
            val_conn = original_connectivity[val_subject_idx].copy()
            
            train_conn = apply_diagonal_imputation(train_conn, region_info, 'random', logger, seed=trial_seed)
            val_conn = apply_diagonal_imputation(val_conn, region_info, 'random', logger, seed=trial_seed + 100000)
            
            train_conn = apply_fisher_z_transformation(train_conn)
            val_conn = apply_fisher_z_transformation(val_conn)
            
            X_train_inner, y_inner_train, _ = prepare_classification_data(train_conn, region_info, subject_ids[train_subject_idx])
            X_val_inner, y_inner_val, _ = prepare_classification_data(val_conn, region_info, subject_ids[val_subject_idx])
            
            scaler_inner = StandardScaler()
            X_train_inner = scaler_inner.fit_transform(X_train_inner)
            X_val_inner = scaler_inner.transform(X_val_inner)
        
        models = train_ovr_classifiers(
            X_train_inner, y_inner_train, n_regions,
            C=C, max_iter=max_iter, solver=solver,
            penalty=penalty, tol=tol,
            random_state=random_state, n_jobs=1, logger=logger, verbose=False
        )
        
        probabilities = predict_ovr_probabilities(models, X_val_inner, n_regions)
        predictions = aggregate_ovr_predictions(probabilities)
        score = accuracy_score(y_inner_val, predictions)
        
        is_best = score > best_score_tracker['score']
        if is_best:
            best_score_tracker.update({'score': score, 'trial': trial.number})
        
        if verbose:
            logger.info(f"      Trial {trial.number+1}/{n_trials}: "
                       f"C={C:.6f}, {solver} → {score:.4f}" + (" ⭐ New Best!" if is_best else ""))
        
        return score
    
    optuna_start = time.time()
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study_seed = random_state + fold_idx * 100000
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=study_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=optuna_n_jobs)
    optuna_time = time.time() - optuna_start
    
    best_params = study.best_params
    best_params['_optuna_best_score'] = float(study.best_value)
    best_params['_optuna_time_seconds'] = float(optuna_time)
    
    logger.info(f"\n    ⭐ BEST HYPERPARAMETERS (Fold {fold_idx + 1}):")
    logger.info(f"       C={best_params['C']:.6f}, solver={best_params['solver']}, "
               f"tol={best_params['tol']:.6f}, max_iter={best_params['max_iter']}")
    logger.info(f"       Validation Accuracy: {study.best_value:.4f}")
    logger.info(f"       Optimization Time: {optuna_time:.1f}s")
    
    return best_params

# ==============================================================================
# METRICS & REPORTING
# ==============================================================================

def compute_classification_metrics_enhanced(y_true, y_pred, y_proba=None):
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

def compute_per_region_binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, 
                                      region_info: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
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

def create_confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ==============================================================================
# TASK TESTING
# ==============================================================================

def test_on_task_data(best_params: dict, random_state: int, n_jobs: int,
                     diagonal_strategy: str, task_data_path: Path, output_dir: Path,
                     sample: int, region_info: pd.DataFrame, n_regions: int,
                     original_connectivity_rest: np.ndarray,
                     subject_ids_rest: np.ndarray,
                     logger: logging.Logger) -> dict:
    """Test trained OvR models on task data using hyperparameters from the best CV fold (or defaults)."""
    C = best_params['C']
    max_iter = best_params['max_iter']
    solver = best_params['solver']
    penalty = best_params.get('penalty', 'l2')
    tol = best_params['tol']
    
    is_random = diagonal_strategy == 'random'
    
    logger.info("\n" + "="*80)
    logger.info("TESTING ON TASK DATA (GENDER STROOP)")
    logger.info("="*80)
    logger.info(f"Training {n_regions} OvR classifiers on ALL rest data")
    logger.info("Hyperparameters (from best CV fold if --tune_hyperparams enabled, otherwise defaults):")
    logger.info(f"  C={C:.6f}, solver={solver}, penalty={penalty}, tol={tol:.6f}, max_iter={max_iter}")
    logger.info(f"Diagonal strategy: {diagonal_strategy}\n")
    
    # Load & preprocess TASK data
    logger.info("Loading task data...")
    df_task = load_connectivity_data(str(task_data_path))
    if sample:
        df_task = sample_first_n_subjects(df_task, sample, logger)
    logger.info(f"  Task: {len(df_task)} subjects")
    
    task_conn_cols = extract_connection_columns(df_task)
    _, region_to_idx_task, _ = extract_regions(task_conn_cols)
    task_original = reconstruct_matrices_from_dataframe(df_task, task_conn_cols, region_to_idx_task, n_regions)
    
    logger.info("\nClipping task data off-diagonal values to [-0.999, 0.999]")
    task_n = task_original.shape[0]
    for i in range(task_n):
        off_mask = ~np.eye(n_regions, dtype=bool)
        task_original[i][off_mask] = np.clip(task_original[i][off_mask], -0.999, 0.999)
    
    # Process REST data for training
    logger.info("\n" + "="*80)
    logger.info("PROCESSING REST DATA (TRAINING)")
    logger.info("="*80)
    
    rest_conn = original_connectivity_rest.copy()
    
    if is_random:
        logger.info("Using RANDOM diagonal imputation with fixed seed for rest data")
        rest_conn = apply_diagonal_imputation(rest_conn, region_info, 'random', logger, seed=random_state)
    else:
        logger.info(f"Using {diagonal_strategy.upper()} diagonal imputation for rest data")
        rest_conn = apply_diagonal_imputation(rest_conn, region_info, diagonal_strategy, logger)
    
    rest_conn = apply_fisher_z_transformation(rest_conn)
    X_rest, y_rest, groups_rest = prepare_classification_data(rest_conn, region_info, subject_ids_rest)
    
    # Process TASK data for testing
    logger.info("\n" + "="*80)
    logger.info("PROCESSING TASK DATA (TESTING)")
    logger.info("="*80)
    
    task_conn = task_original.copy()
    
    if is_random:
        logger.info("Using RANDOM diagonal imputation with different seed for task data")
        task_conn = apply_diagonal_imputation(task_conn, region_info, 'random', logger, seed=random_state + 9999)
    else:
        logger.info(f"Using {diagonal_strategy.upper()} diagonal imputation for task data")
        task_conn = apply_diagonal_imputation(task_conn, region_info, diagonal_strategy, logger)
    
    task_conn = apply_fisher_z_transformation(task_conn)
    X_task, y_task, groups_task = prepare_classification_data(task_conn, region_info, df_task.iloc[:, 0].values)
    
    # Scale data
    logger.info("\n" + "="*80)
    logger.info("SCALING DATA")
    logger.info("="*80)
    scaler_final = StandardScaler()
    X_rest_scaled = scaler_final.fit_transform(X_rest)
    X_task_scaled = scaler_final.transform(X_task)
    
    # Training on rest
    logger.info("\n" + "="*80)
    logger.info("TRAINING FINAL OVR MODELS ON REST DATA")
    logger.info("="*80)
    train_start = time.time()
    final_models = train_ovr_classifiers(
        X_rest_scaled, y_rest, n_regions, 
        C=C, max_iter=max_iter, solver=solver,
        penalty=penalty, tol=tol,
        random_state=random_state, n_jobs=n_jobs, 
        logger=logger, verbose=True
    )
    train_time = time.time() - train_start
    logger.info(f"  ✓ Trained {n_regions} classifiers in {train_time:.2f}s")
    
    # Evaluation on rest (sanity check)
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON REST DATA (TRAINING SET)")
    logger.info("="*80)
    rest_probabilities = predict_ovr_probabilities(final_models, X_rest_scaled, n_regions)
    rest_predictions = aggregate_ovr_predictions(rest_probabilities)
    rest_metrics = compute_classification_metrics_enhanced(y_rest, rest_predictions, rest_probabilities)
    
    logger.info(f"  Rest Accuracy: {rest_metrics['accuracy']:.4f}")
    if 'top_3_accuracy' in rest_metrics:
        logger.info(f"  Rest Top-3: {rest_metrics['top_3_accuracy']:.4f}")
        logger.info(f"  Rest Top-5: {rest_metrics['top_5_accuracy']:.4f}")
        logger.info(f"  Rest Top-10: {rest_metrics['top_10_accuracy']:.4f}")
    
    # Evaluation on task
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON TASK DATA (TEST SET)")
    logger.info("="*80)
    task_probabilities = predict_ovr_probabilities(final_models, X_task_scaled, n_regions)
    task_predictions = aggregate_ovr_predictions(task_probabilities)
    
    task_metrics = compute_classification_metrics_enhanced(y_task, task_predictions, task_probabilities)
    task_per_region = compute_per_region_binary_metrics(y_task, task_probabilities, region_info)
    task_network = compute_network_level_metrics(task_per_region)
    task_confusion = create_confusion_matrix(y_task, task_predictions, n_regions)
    
    logger.info(f"  Task Accuracy: {task_metrics['accuracy']:.4f}")
    if 'top_3_accuracy' in task_metrics:
        logger.info(f"  Task Top-3: {task_metrics['top_3_accuracy']:.4f}")
        logger.info(f"  Task Top-5: {task_metrics['top_5_accuracy']:.4f}")
        logger.info(f"  Task Top-10: {task_metrics['top_10_accuracy']:.4f}")
    
    # Save results
    task_output_dir = output_dir / "task_testing_one_vs_rest"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(task_output_dir / 'task_predictions.npy', task_predictions)
    np.save(task_output_dir / 'task_probabilities.npy', task_probabilities)
    np.save(task_output_dir / 'task_true_labels.npy', y_task)
    np.save(task_output_dir / 'task_confusion_matrix.npy', task_confusion)
    
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
    
    logger.info(f"  Saved to: {task_output_dir}")
    logger.info("\n" + "="*80)
    logger.info("TASK TESTING SUMMARY")
    logger.info("="*80)
    logger.info(f"  Rest Training Accuracy: {task_summary['rest_train_accuracy']:.4f}")
    logger.info(f"  Task Testing Accuracy:  {task_summary['task_test_accuracy']:.4f}")
    logger.info(f"  Accuracy Drop:          {task_summary['accuracy_drop']:.4f}")
    logger.info("="*80 + "\n")
    
    return {
        'task_metrics': task_metrics,
        'task_per_region': task_per_region,
        'task_network': task_network,
        'task_summary': task_summary,
        'rest_metrics': rest_metrics
    }

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_full_connectivity_ovr(args: argparse.Namespace, logger: logging.Logger, 
                                optuna_n_jobs: int) -> dict:
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING FULL CONNECTIVITY - ONE-VS-REST")
    logger.info(f"{'='*80}\n")
    
    output_dir = args.output_dir / "one_vs_rest"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading resting-state data...")
    df_train = load_connectivity_data(str(args.rest_data))
    if args.sample:
        df_train = sample_first_n_subjects(df_train, args.sample, logger)
    
    conn_cols = extract_connection_columns(df_train)
    region_list, region_map, n_regions = extract_regions(conn_cols)
    connectivity = reconstruct_matrices_from_dataframe(df_train, conn_cols, region_map, n_regions)
    original_connectivity = connectivity.copy()
    
    logger.info("\nClipping off-diagonal values to [-0.999, 0.999]")
    n_subjects = connectivity.shape[0]
    for i in range(n_subjects):
        off_mask = ~np.eye(n_regions, dtype=bool)
        original_connectivity[i][off_mask] = np.clip(original_connectivity[i][off_mask], -0.999, 0.999)
    
    region_info = create_region_info(region_list)
    subject_ids = df_train.iloc[:, 0].values
    
    logger.info(f"Data: {n_subjects} subjects, {n_regions} regions")
    logger.info(f"Diagonal strategy: {args.diagonal_strategy}")
    
    is_random = args.diagonal_strategy == 'random'
    
    _, y, groups = prepare_classification_data(original_connectivity, region_info, subject_ids)
    
    if not is_random:
        connectivity = apply_diagonal_imputation(original_connectivity, region_info, args.diagonal_strategy, logger)
        connectivity = apply_fisher_z_transformation(connectivity)
        X, _, _ = prepare_classification_data(connectivity, region_info, subject_ids)
    else:
        X = None
    
    # CV loop
    logger.info("\n" + "="*80)
    logger.info(f"CROSS-VALIDATION ({args.n_folds} folds)")
    logger.info("="*80)
    
    gkf = GroupKFold(n_splits=args.n_folds)
    all_predictions, all_probabilities, all_true_labels = [], [], []
    fold_metrics = []
    best_fold_params = None
    best_fold_val_acc = 0.0
    best_fold_idx = -1
    
    cv_start_time = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X if X is not None else range(len(y)), y, groups=groups)):
        logger.info(f"\n{'='*80}")
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"{'='*80}")
        
        n_train_subjects = len(np.unique(groups[train_idx]))
        n_test_subjects = len(np.unique(groups[test_idx]))
        logger.info(f"  Train: {n_train_subjects} subjects")
        logger.info(f"  Test:  {n_test_subjects} subjects")
        
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
            logger.info(f"  Random imputation seeds: train={fold_seed}, test={fold_seed + 1000}")
            
            train_conn = apply_diagonal_imputation(train_conn, region_info, 'random', logger, seed=fold_seed)
            test_conn = apply_diagonal_imputation(test_conn, region_info, 'random', logger, seed=fold_seed + 1000)
            
            train_conn = apply_fisher_z_transformation(train_conn)
            test_conn = apply_fisher_z_transformation(test_conn)
            
            X_train, y_train, groups_train = prepare_classification_data(train_conn, region_info, subject_ids[train_mask])
            X_test, y_test, _ = prepare_classification_data(test_conn, region_info, subject_ids[test_mask])
        
        # Hyperparameter tuning (per-fold)
        if args.tune_hyperparams:
            fold_best_params = optimize_hyperparameters_ovr_fold(
                X_train_unscaled=None if is_random else X_train,
                y_train=y_train,
                groups_train=groups_train,
                n_regions=n_regions,
                n_trials=args.optuna_trials,
                random_state=args.random_state,
                optuna_n_jobs=optuna_n_jobs,
                logger=logger,
                verbose=args.verbose,
                original_connectivity=original_connectivity if is_random else None,
                subject_ids=subject_ids if is_random else None,
                region_info=region_info if is_random else None,
                is_random=is_random,
                fold_idx=fold_idx
            )
            C = fold_best_params['C']
            max_iter = fold_best_params['max_iter']
            solver = fold_best_params['solver']
            penalty = fold_best_params.get('penalty', 'l2')
            tol = fold_best_params['tol']
        else:
            C = args.regularization_C
            max_iter = args.max_iter
            solver = 'lbfgs'
            penalty = 'l2'
            tol = 1e-4
            fold_best_params = {'C': C, 'max_iter': max_iter, 'solver': solver, 'penalty': penalty, 'tol': tol}
        
        logger.info(f"\n  Using hyperparameters for this fold:")
        logger.info(f"    C={C:.6f}, solver={solver}, penalty={penalty}, tol={tol:.6f}, max_iter={max_iter}")
        
        # Scale and train
        X_train_scaled, X_test_scaled, _ = preprocess_fold_data(X_train, X_test, logger, args.verbose)
        
        fold_start = time.time()
        fold_models = train_ovr_classifiers(
            X_train_scaled, y_train, n_regions, 
            C=C, max_iter=max_iter, solver=solver,
            penalty=penalty, tol=tol,
            random_state=args.random_state, n_jobs=args.n_jobs, 
            logger=logger, verbose=args.verbose
        )
        fold_train_time = time.time() - fold_start
        
        # Predictions
        train_probabilities = predict_ovr_probabilities(fold_models, X_train_scaled, n_regions)
        train_predictions = aggregate_ovr_predictions(train_probabilities)
        fold_train_acc = accuracy_score(y_train, train_predictions)
        
        probabilities = predict_ovr_probabilities(fold_models, X_test_scaled, n_regions)
        predictions = aggregate_ovr_predictions(probabilities)
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
        
        logger.info(f"  ✓ Fold {fold_idx + 1}: Train Acc={fold_train_acc:.4f}, Val Acc={fold_val_acc:.4f}, Time={fold_train_time:.2f}s")
        
        all_predictions.extend(predictions)
        all_probabilities.append(probabilities)
        all_true_labels.extend(y_test)
    
    cv_total_time = time.time() - cv_start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"Cross-validation completed in {cv_total_time:.2f}s")
    logger.info(f"Best Fold: {best_fold_idx + 1} with Val Accuracy: {best_fold_val_acc:.4f}")
    
    # Aggregate CV results
    train_accs = [f['train_accuracy'] for f in fold_metrics]
    val_accs = [f['val_accuracy'] for f in fold_metrics]
    mean_train_acc = np.mean(train_accs)
    mean_val_acc = np.mean(val_accs)
    generalization_gap = mean_train_acc - mean_val_acc
    
    logger.info(f"\nMean Train Accuracy: {mean_train_acc:.4f}")
    logger.info(f"Mean Validation Accuracy: {mean_val_acc:.4f}")
    logger.info(f"Generalization Gap: {generalization_gap:.4f}")
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    all_true_labels = np.array(all_true_labels)
    
    overall_metrics = compute_classification_metrics_enhanced(all_true_labels, all_predictions, all_probabilities)
    per_region_metrics = compute_per_region_binary_metrics(all_true_labels, all_probabilities, region_info)
    network_metrics = compute_network_level_metrics(per_region_metrics)
    confusion_mat = create_confusion_matrix(all_true_labels, all_predictions, n_regions)
    
    logger.info("\n" + "="*80)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*80)
    logger.info(f"  Overall Accuracy: {overall_metrics['accuracy']:.4f}")
    if 'top_3_accuracy' in overall_metrics:
        logger.info(f"  Top-3 Accuracy:   {overall_metrics['top_3_accuracy']:.4f}")
        logger.info(f"  Top-5 Accuracy:   {overall_metrics['top_5_accuracy']:.4f}")
        logger.info(f"  Top-10 Accuracy:  {overall_metrics['top_10_accuracy']:.4f}")
    
    # Save CV results
    np.save(output_dir / 'cv_predictions.npy', all_predictions)
    np.save(output_dir / 'cv_probabilities.npy', all_probabilities)
    np.save(output_dir / 'cv_true_labels.npy', all_true_labels)
    np.save(output_dir / 'confusion_matrix.npy', confusion_mat)
    
    cv_summary = {
        'diagonal_strategy': args.diagonal_strategy,
        'best_fold_hyperparameters': best_fold_params,
        'best_fold_idx': int(best_fold_idx + 1),
        'best_fold_val_accuracy': float(best_fold_val_acc),
        'overall_metrics': overall_metrics,
        'fold_metrics': fold_metrics,
        'cv_time_seconds': float(cv_total_time),
        'mean_train_accuracy': float(mean_train_acc),
        'mean_val_accuracy': float(mean_val_acc),
        'generalization_gap': float(generalization_gap)
    }
    
    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    with open(output_dir / 'overall_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    per_region_metrics.to_csv(output_dir / 'per_region_binary_metrics.csv', index=False)
    network_metrics.to_csv(output_dir / 'network_metrics.csv', index=False)
    
    logger.info(f"  Saved to: {output_dir}")
    
    # Task testing
    task_results = None
    if args.test_on_task:
        try:
            task_results = test_on_task_data(
                best_fold_params, args.random_state, args.n_jobs,
                args.diagonal_strategy, args.task_data, args.output_dir, args.sample,
                region_info, n_regions,
                original_connectivity_rest=original_connectivity,
                subject_ids_rest=subject_ids,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Task testing failed: {str(e)}", exc_info=True)
    
    return {
        'overall_metrics': overall_metrics,
        'per_region_metrics': per_region_metrics,
        'network_metrics': network_metrics,
        'task_results': task_results,
        'output_dir': output_dir,
        'cv_summary': cv_summary
    }

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    optuna_n_jobs = get_optuna_n_jobs(args)
    
    logger.info("="*80)
    logger.info("CONFIGURATION")
    logger.info("="*80)
    logger.info(f"  Rest data: {args.rest_data}")
    logger.info(f"  Task data: {args.task_data}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Diagonal strategy: {args.diagonal_strategy}")
    logger.info(f"  N folds: {args.n_folds}")
    logger.info(f"  Tune hyperparams: {args.tune_hyperparams}")
    logger.info(f"  Test on task: {args.test_on_task}")
    logger.info(f"  N jobs: {args.n_jobs}")
    logger.info(f"  Optuna N jobs: {optuna_n_jobs}")
    logger.info("="*80 + "\n")
    
    try:
        results = train_full_connectivity_ovr(args, logger, optuna_n_jobs)
        
        logger.info("\n" + "="*80)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"  CV Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        if results.get('task_results'):
            logger.info(f"  Task Accuracy: {results['task_results']['task_metrics']['accuracy']:.4f}")
            logger.info(f"  Accuracy Drop: {results['task_results']['task_summary']['accuracy_drop']:.4f}")
        logger.info(f"  Results saved to: {results['output_dir']}")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()