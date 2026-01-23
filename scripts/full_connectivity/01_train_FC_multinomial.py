"""
01_train_fC_multinomial.py

Train multinomial logistic regression on FULL brain connectivity (232 regions).

Workflow: Load Data → Diagonal Imputation + Fisher Z → CV with fold-wise Optuna → Task Testing

Usage:
    python scripts/full_connectivity/01_train_fC_multinomial.py --tune_hyperparams --test_on_task
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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler

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
    log_file = output_dir / "training_full_connectivity_multinomial.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train full connectivity multinomial logistic regression')
    
    # Core settings
    parser.add_argument('--rest_data', type=Path, default=project_root / 'data' / 'raw' / 'PIOP2_restingstate.csv')
    parser.add_argument('--task_data', type=Path, default=project_root / 'data' / 'raw' / 'PIOP1_gstroop.csv')
    parser.add_argument('--output_dir', type=Path, default=project_root / 'data' / 'results' / 'full_connectivity_analysis')
    parser.add_argument('--config_file', type=Path, default=project_root / 'configs' / 'FC_config.yaml')
    
    # Model settings
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--regularization_C', type=float, default=1.0)
    parser.add_argument('--diagonal_strategy', type=str, default='region_mean', 
                       choices=['zero', 'region_mean', 'network_mean', 'global_mean'])
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    
    # Execution flags
    parser.add_argument('--save_models', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--sample', type=int, default=None)
    
    # Optimization
    parser.add_argument('--tune_hyperparams', action='store_true')
    parser.add_argument('--optuna_trials', type=int, default=50)
    parser.add_argument('--test_on_task', action='store_true')
    parser.add_argument('--final_optuna_trials', type=int, default=50)
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
    # Identify hemisphere based on naming convention
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
                             strategy: str, logger: logging.Logger) -> np.ndarray:
    """Apply diagonal imputation to connectivity matrices."""
    logger.info(f"\nApplying diagonal imputation: {strategy}")
    n_subjects, n_regions, _ = connectivity.shape
    imp = connectivity.copy()
    
    # Log original
    orig_diag = np.array([connectivity[i].diagonal() for i in range(n_subjects)])
    logger.info(f"  Original diagonal: [{orig_diag.min():.4f}, {orig_diag.max():.4f}], mean={orig_diag.mean():.4f}")
    
    if strategy == 'zero':
        for i in range(n_subjects):
            np.fill_diagonal(imp[i], 0.0)
            
    elif strategy == 'region_mean':
        for i in range(n_subjects):
            for j in range(n_regions):
                mask = np.ones(n_regions, dtype=bool)
                mask[j] = False
                imp[i, j, j] = imp[i, j, mask].mean()
                
    elif strategy == 'network_mean':
        if 'network' not in region_info.columns:
            logger.warning("  'network' column not found, falling back to region_mean")
            return apply_diagonal_imputation(connectivity, region_info, 'region_mean', logger)
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
    
    # Log new
    new_diag = np.array([imp[i].diagonal() for i in range(n_subjects)])
    logger.info(f"  New diagonal: [{new_diag.min():.4f}, {new_diag.max():.4f}], mean={new_diag.mean():.4f}")
    return imp

def apply_fisher_z_transformation(connectivity: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Apply Fisher Z-transformation to connectivity matrices."""
    logger.info("\nApplying Fisher Z-transformation...")
    c_trans = np.arctanh(np.clip(connectivity, -0.999, 0.999))
    
    if np.any(np.isnan(c_trans)) or np.any(np.isinf(c_trans)):
        raise ValueError("NaN/Inf detected after Fisher Z transformation")
    
    logger.info(f"  Range: [{c_trans.min():.4f}, {c_trans.max():.4f}], mean={c_trans.mean():.4f}, std={c_trans.std():.4f}")
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
# OPTUNA HYPERPARAMETER TUNING
# ==============================================================================

def optimize_hyperparameters_optuna_fold(X_train_unscaled: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
                                         fold_idx: int, n_trials: int, random_state: int, optuna_n_jobs: int,
                                         logger: logging.Logger, verbose: bool = False) -> dict:
    """Run Optuna hyperparameter optimization within a single CV fold."""
    if verbose:
        logger.info(f"\n  FOLD {fold_idx} - Hyperparameter Optimization ({n_trials} trials)")
        logger.info(f"  Samples: {len(X_train_unscaled)}, Features: {X_train_unscaled.shape[1]}, Classes: {len(np.unique(y_train))}")

    best_score_tracker = {'score': 0.0, 'trial': 0}
    
    def objective(trial):
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 500, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg'])
        
        # Inner split using GroupShuffleSplit
        trial_seed = random_state + fold_idx * 1000 + trial.number
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=trial_seed)
        train_inner_idx, val_inner_idx = next(gss.split(X_train_unscaled, y_train, groups=groups_train))
        
        # Scale
        scaler_inner = StandardScaler()
        X_train_inner = scaler_inner.fit_transform(X_train_unscaled[train_inner_idx])
        X_val_inner = scaler_inner.transform(X_train_unscaled[val_inner_idx])
        
        # Train & evaluate
        model = LogisticRegression(C=C, penalty='l2', multi_class='multinomial', max_iter=max_iter,
                                   solver=solver, random_state=random_state, n_jobs=1, verbose=0)
        model.fit(X_train_inner, y_train[train_inner_idx])
        score = accuracy_score(y_train[val_inner_idx], model.predict(X_val_inner))
        
        if verbose and score > best_score_tracker['score']:
            best_score_tracker.update({'score': score, 'trial': trial.number})
            logger.info(f"    Trial {trial.number+1}: C={C:.4f} {solver} -> {score:.4f} (New Best)")
        return score
    
    # Run study
    optuna_start = time.time()
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state + fold_idx))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=optuna_n_jobs)
    optuna_time = time.time() - optuna_start
    
    best_params = study.best_params
    best_params['_optuna_best_score'] = float(study.best_value)
    best_params['_optuna_time_seconds'] = float(optuna_time)
    
    if not verbose:
        logger.info(f"  Optuna: C={best_params['C']:.4f}, {best_params['solver']}, "
                   f"max_iter={best_params['max_iter']} (val_acc={study.best_value:.4f}, {optuna_time:.1f}s)")
    return best_params

def optimize_on_full_rest_data(X_rest_unscaled: np.ndarray, y_rest: np.ndarray, groups_rest: np.ndarray,
                               n_trials: int, random_state: int, optuna_n_jobs: int, logger: logging.Logger) -> dict:
    """Run final hyperparameter optimization on entire resting-state dataset."""
    logger.info("\n" + "="*80)
    logger.info("FINAL OPTUNA OPTIMIZATION ON FULL REST DATA")
    logger.info("="*80)
    logger.info(f"Running {n_trials} trials for task testing parameters...")
    
    def objective(trial):
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 500, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg'])
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state + 9999 + trial.number)
        train_idx, val_idx = next(gss.split(X_rest_unscaled, y_rest, groups=groups_rest))
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_rest_unscaled[train_idx])
        X_val = scaler.transform(X_rest_unscaled[val_idx])
        
        model = LogisticRegression(C=C, penalty='l2', multi_class='multinomial', max_iter=max_iter,
                                   solver=solver, random_state=random_state, n_jobs=1, verbose=0)
        model.fit(X_tr, y_rest[train_idx])
        return accuracy_score(y_rest[val_idx], model.predict(X_val))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state + 9999))
    
    optuna_start = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=optuna_n_jobs)
    optuna_time = time.time() - optuna_start
    
    best_params = study.best_params
    best_params['_selection_method'] = 'final_optuna_on_full_data'
    best_params['_optuna_best_score'] = float(study.best_value)
    best_params['_optuna_time_seconds'] = float(optuna_time)
    
    logger.info(f"\nBest: C={best_params['C']:.6f}, {best_params['solver']}, max_iter={best_params['max_iter']}")
    logger.info(f"Validation accuracy: {study.best_value:.4f}, Time: {optuna_time:.2f}s")
    logger.info("="*80 + "\n")
    return best_params

def aggregate_fold_hyperparameters(fold_best_params: list) -> dict:
    """Aggregate hyperparameters across folds (median C, mode solver)."""
    C_values = [p['C'] for p in fold_best_params]
    max_iter_values = [p['max_iter'] for p in fold_best_params]
    solvers = [p['solver'] for p in fold_best_params]
    
    from collections import Counter
    best_solver = Counter(solvers).most_common(1)[0][0]
    best_max_iter = Counter(max_iter_values).most_common(1)[0][0]
    best_C = float(np.median(C_values))
    
    return {
        'C': best_C, 'max_iter': best_max_iter, 'solver': best_solver,
        '_aggregation_method': 'median_C_mode_solver',
        '_C_range': [float(min(C_values)), float(max(C_values))],
        '_C_mean': float(np.mean(C_values)), '_C_std': float(np.std(C_values))
    }

def select_best_fold_hyperparameters(fold_best_params: list, fold_metrics: list, logger: logging.Logger) -> dict:
    """Select best params from best-performing fold."""
    accs = [m['accuracy'] for m in fold_metrics]
    best_idx = np.argmax(accs)
    best_params = fold_best_params[best_idx].copy()
    best_params['_best_fold_accuracy'] = float(accs[best_idx])
    best_params['_selection_method'] = 'best_fold'
    return best_params

# ==============================================================================
# METRICS & REPORTING
# ==============================================================================

def compute_classification_metrics_enhanced(y_true, y_pred, y_proba=None):
    """Calculate accuracy and Top-K accuracy (k=3, 5, 10)."""
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

def compute_per_region_metrics(y_true, y_pred, region_info):
    """Calculates accuracy for every single brain region."""
    n_regions = len(region_info)
    per_region = []
    for r in range(n_regions):
        mask = (y_true == r)
        acc = (y_pred[mask] == r).sum() / mask.sum() if mask.sum() > 0 else 0.0
        per_region.append({
            'region_idx': r,
            'region_name': region_info.iloc[r]['region_name'],
            'network': region_info.iloc[r]['network'],
            'accuracy': acc,
            'n_samples': int(mask.sum())
        })
    return pd.DataFrame(per_region)

def compute_network_level_metrics(y_true, y_pred, region_info):
    """Aggregates accuracy by Brain Network."""
    networks = region_info['network'].unique()
    metrics = []
    for net in networks:
        net_regions = region_info[region_info['network'] == net]['region_idx'].values
        mask = np.isin(y_true, net_regions)
        acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum() if mask.sum() > 0 else 0.0
        metrics.append({'network': net, 'accuracy': float(acc), 'n_samples': int(mask.sum())})
    return pd.DataFrame(metrics).sort_values('accuracy', ascending=False)

def create_confusion_matrix(y_true, y_pred, n_classes):
    """Create confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ==============================================================================
# TASK TESTING LOGIC
# ==============================================================================

def test_on_task_data(best_params: dict, random_state: int, n_jobs: int,
                     diagonal_strategy: str, task_data_path: Path, output_dir: Path,
                     sample: int, region_info: pd.DataFrame, n_regions: int,
                     X_rest: np.ndarray, y_rest: np.ndarray, groups_rest: np.ndarray,
                     logger: logging.Logger) -> dict:
    """Train final model on ALL rest data, then test on task data."""
    C, max_iter, solver = best_params['C'], best_params['max_iter'], best_params['solver']
    
    logger.info("\n" + "="*80)
    logger.info("TESTING ON TASK DATA (GENDER STROOP)")
    logger.info("="*80)
    logger.info(f"Training on ALL rest data: C={C:.4f}, {solver}, max_iter={max_iter}\n")
    
    # Load & preprocess TASK data
    logger.info("Loading task data...")
    df_task = load_connectivity_data(str(task_data_path))
    if sample:
        df_task = sample_first_n_subjects(df_task, sample, logger)
    logger.info(f"  Task: {len(df_task)} subjects")
    
    # Reconstruct Task Matrices
    task_conn_cols = extract_connection_columns(df_task)
    _, region_to_idx_task, _ = extract_regions(task_conn_cols)
    task_connectivity = reconstruct_matrices_from_dataframe(df_task, task_conn_cols, region_to_idx_task, n_regions)
    
    task_connectivity = apply_diagonal_imputation(task_connectivity, region_info, diagonal_strategy, logger)
    task_connectivity = apply_fisher_z_transformation(task_connectivity, logger)
    X_task, y_task, groups_task = prepare_classification_data(task_connectivity, region_info, df_task.iloc[:, 0].values)
    
    # Scale & train
    logger.info("\nTraining final model...")
    scaler_final = StandardScaler()
    X_rest_scaled = scaler_final.fit_transform(X_rest)
    X_task_scaled = scaler_final.transform(X_task)
    
    final_model = LogisticRegression(C=C, penalty='l2', multi_class='multinomial', max_iter=max_iter,
                                     solver=solver, random_state=random_state, n_jobs=n_jobs, verbose=0)
    train_start = time.time()
    final_model.fit(X_rest_scaled, y_rest)
    logger.info(f"  Trained in {time.time() - train_start:.2f}s")
    
    # Evaluate on rest (sanity check)
    y_rest_pred = final_model.predict(X_rest_scaled)
    y_rest_proba = final_model.predict_proba(X_rest_scaled)
    rest_metrics = compute_classification_metrics_enhanced(y_rest, y_rest_pred, y_rest_proba)
    logger.info(f"  Rest accuracy: {rest_metrics['accuracy']:.4f}")
    
    # Test on task
    logger.info("\nTesting on task data...")
    y_task_pred = final_model.predict(X_task_scaled)
    y_task_proba = final_model.predict_proba(X_task_scaled)
    
    # Compute metrics
    task_metrics = compute_classification_metrics_enhanced(y_task, y_task_pred, y_task_proba)
    task_per_region = compute_per_region_metrics(y_task, y_task_pred, region_info)
    task_network = compute_network_level_metrics(y_task, y_task_pred, region_info)
    task_confusion = create_confusion_matrix(y_task, y_task_pred, len(np.unique(y_task)))
    
    # Save results
    task_output_dir = output_dir / "task_testing_multinomial"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(task_output_dir / 'task_predictions.npy', y_task_pred)
    np.save(task_output_dir / 'task_probabilities.npy', y_task_proba)
    np.save(task_output_dir / 'task_true_labels.npy', y_task)
    np.save(task_output_dir / 'task_confusion_matrix.npy', task_confusion)
    
    task_summary = {
        'rest_train_accuracy': float(rest_metrics['accuracy']),
        'rest_train_top_5_accuracy': float(rest_metrics.get('top_5_accuracy', 0)),
        'task_test_accuracy': float(task_metrics['accuracy']),
        'task_top_3_accuracy': float(task_metrics.get('top_3_accuracy', 0)),
        'task_top_5_accuracy': float(task_metrics.get('top_5_accuracy', 0)),
        'task_top_10_accuracy': float(task_metrics.get('top_10_accuracy', 0)),
        'accuracy_drop': float(rest_metrics['accuracy'] - task_metrics['accuracy']),
        'hyperparameters': best_params,
        'n_rest_subjects': int(len(np.unique(groups_rest))),
        'n_task_subjects': int(len(np.unique(groups_task))),
        'n_rest_samples': int(len(X_rest)),
        'n_task_samples': int(len(X_task))
    }
    
    with open(task_output_dir / 'task_testing_summary.json', 'w') as f:
        json.dump(task_summary, f, indent=2)
    task_per_region.to_csv(task_output_dir / 'task_per_region_metrics.csv', index=False)
    task_network.to_csv(task_output_dir / 'task_network_metrics.csv', index=False)
    
    import pickle
    with open(task_output_dir / 'final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open(task_output_dir / 'final_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_final, f)
    
    # Report
    logger.info("\n" + "="*80)
    logger.info("TASK TESTING RESULTS")
    logger.info("="*80)
    logger.info(f"Rest accuracy: {rest_metrics['accuracy']:.4f}")
    logger.info(f"Task accuracy: {task_metrics['accuracy']:.4f}")
    logger.info(f"Accuracy drop: {task_summary['accuracy_drop']:.4f} ({task_summary['accuracy_drop']/rest_metrics['accuracy']*100:.1f}%)")
    logger.info(f"Task top-3/5/10: {task_metrics.get('top_3_accuracy', 0):.4f} / "
                f"{task_metrics.get('top_5_accuracy', 0):.4f} / {task_metrics.get('top_10_accuracy', 0):.4f}")
    logger.info(f"Best network: {task_network.iloc[task_network['accuracy'].idxmax()]['network']} ({task_network['accuracy'].max():.4f})")
    logger.info("="*80 + "\n")
    
    return {
        'task_metrics': task_metrics,
        'task_per_region': task_per_region,
        'task_network': task_network,
        'task_summary': task_summary,
        'final_model': final_model,
        'final_scaler': scaler_final
    }

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_full_connectivity(args: argparse.Namespace, logger: logging.Logger, 
                           optuna_n_jobs: int) -> dict:
    """Train multinomial model for full connectivity."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING FULL CONNECTIVITY")
    logger.info(f"{'='*80}\n")
    
    output_dir = args.output_dir / "multinomial"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    df_train = load_connectivity_data(str(args.rest_data))
    if args.sample:
        df_train = sample_first_n_subjects(df_train, args.sample, logger)
        logger.warning(f"⚠️  TESTING MODE: {args.sample} subjects only")
    
    # Reconstruct Matrices
    conn_cols = extract_connection_columns(df_train)
    region_list, region_map, n_regions = extract_regions(conn_cols)
    connectivity = reconstruct_matrices_from_dataframe(df_train, conn_cols, region_map, n_regions)
    region_info = create_region_info(region_list)
    subject_ids = df_train.iloc[:, 0].values
    
    n_subjects = connectivity.shape[0]
    logger.info(f"Data: {n_subjects} subjects, {n_regions} regions, shape={connectivity.shape}")
    
    # Global preprocessing
    logger.info("\n" + "="*80)
    logger.info("GLOBAL PREPROCESSING")
    logger.info("="*80)
    connectivity = apply_diagonal_imputation(connectivity, region_info, args.diagonal_strategy, logger)
    connectivity = apply_fisher_z_transformation(connectivity, logger)
    
    logger.info("\nPreparing classification data...")
    X, y, groups = prepare_classification_data(connectivity, region_info, subject_ids)
    n_samples, n_classes = X.shape[0], len(np.unique(y))
    logger.info(f"Samples: {X.shape}, Labels: {y.shape}, Classes: {n_classes}")
    assert X.shape[0] == len(y) == len(groups), "Data mismatch!"
    logger.info("✓ Preprocessing complete")
    logger.info("="*80)
    
    # Cross-validation
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION ({args.n_folds} folds)")
    logger.info(f"{'='*80}")
    if args.tune_hyperparams:
        logger.info(f"Optuna: {args.optuna_trials} trials per fold (leak-free)")
    else:
        logger.info(f"Fixed params: C={args.regularization_C}, solver=lbfgs, max_iter={args.max_iter}")
    
    gkf = GroupKFold(n_splits=args.n_folds)
    all_predictions, all_probabilities, all_true_labels, all_fold_indices = [], [], [], []
    fold_models = [] if args.save_models else None
    fold_metrics, fold_best_params = [], []
    
    start_time = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        fold_start = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"{'='*80}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]
        
        train_subjects, test_subjects = set(groups_train), set(groups_test)
        assert len(train_subjects.intersection(test_subjects)) == 0, "Subject leakage!"
        logger.info(f"  Train: {len(train_idx)} samples ({len(train_subjects)} subjects)")
        logger.info(f"  Test: {len(test_idx)} samples ({len(test_subjects)} subjects)")
        
        # Hyperparameter tuning
        if args.tune_hyperparams:
            best_params = optimize_hyperparameters_optuna_fold(
                X_train, y_train, groups_train, fold_idx + 1, args.optuna_trials,
                args.random_state, optuna_n_jobs, logger, args.verbose
            )
            C, max_iter, solver = best_params['C'], best_params['max_iter'], best_params['solver']
            fold_best_params.append(best_params)
        else:
            C, max_iter, solver = args.regularization_C, args.max_iter, 'lbfgs'
            best_params = {'C': C, 'max_iter': max_iter, 'solver': solver}
        
        # Scale & train
        logger.info(f"  Scaling & training (C={C:.6f}, {solver}, max_iter={max_iter})...")
        X_train_scaled, X_test_scaled, fold_scaler = preprocess_fold_data(X_train, X_test, logger, args.verbose)
        
        model = LogisticRegression(C=C, penalty='l2', multi_class='multinomial', max_iter=max_iter,
                                   solver=solver, random_state=args.random_state, n_jobs=args.n_jobs,
                                   verbose=1 if args.verbose else 0)
        model.fit(X_train_scaled, y_train)
        
        # Predict & evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        fold_acc = accuracy_score(y_test, y_pred)
        
        fold_metric_dict = {
            'fold': fold_idx + 1, 'accuracy': fold_acc,
            'n_train': len(y_train), 'n_test': len(y_test),
            'n_train_subjects': len(train_subjects), 'n_test_subjects': len(test_subjects),
            'hyperparameters': {'C': float(C), 'max_iter': int(max_iter), 'solver': solver}
        }
        if args.tune_hyperparams:
            fold_metric_dict['optuna_validation_score'] = best_params['_optuna_best_score']
            fold_metric_dict['optuna_time'] = best_params['_optuna_time_seconds']
        fold_metrics.append(fold_metric_dict)
        
        logger.info(f"  ✓ Fold {fold_idx + 1}: Acc={fold_acc:.4f}, Time={time.time() - fold_start:.2f}s\n")
        
        # Store results
        all_predictions.extend(y_pred)
        all_probabilities.append(y_proba)
        all_true_labels.extend(y_test)
        all_fold_indices.extend([fold_idx + 1] * len(y_test))
        
        if args.save_models:
            fold_models.append({
                'fold': fold_idx + 1, 'model': model, 'scaler': fold_scaler,
                'train_idx': train_idx, 'test_idx': test_idx, 'hyperparameters': best_params
            })
    
    logger.info(f"{'='*80}")
    logger.info(f"CV completed in {time.time() - start_time:.2f}s")
    logger.info(f"{'='*80}\n")
    
    # Convert arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    all_true_labels = np.array(all_true_labels)
    all_fold_indices = np.array(all_fold_indices)
    
    # Aggregate results
    logger.info("Computing metrics...")
    overall_metrics = compute_classification_metrics_enhanced(all_true_labels, all_predictions, all_probabilities)
    
    # Hyperparameter aggregation
    if args.tune_hyperparams and fold_best_params:
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER SUMMARY")
        logger.info("="*80)
        for i, p in enumerate(fold_best_params, 1):
            logger.info(f"  Fold {i}: C={p['C']:.6f}, {p['solver']}, max_iter={p['max_iter']}, val_acc={p['_optuna_best_score']:.4f}")
        
        reference_params = aggregate_fold_hyperparameters(fold_best_params)
        logger.info(f"\nAggregated: C={reference_params['C']:.6f} (median), {reference_params['solver']} (mode)")
        logger.info(f"  C range: [{reference_params['_C_range'][0]:.6f}, {reference_params['_C_range'][1]:.6f}]")
        logger.info(f"  C mean±std: {reference_params['_C_mean']:.6f} ± {reference_params['_C_std']:.6f}")
        logger.info("="*80 + "\n")
    else:
        reference_params = {'C': args.regularization_C, 'max_iter': args.max_iter, 'solver': 'lbfgs'}
    
    overall_metrics['reference_hyperparameters'] = reference_params
    overall_metrics['fold_hyperparameters'] = fold_best_params if args.tune_hyperparams else None
    overall_metrics['preprocessing'] = {
        'diagonal_strategy': args.diagonal_strategy, 'fisher_z_applied': True,
        'standardize_per_fold': True, 'optuna_per_fold': args.tune_hyperparams
    }
    
    # Report
    logger.info(f"\n{'='*80}")
    logger.info(f"OVERALL CV RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"Top-3/5/10: {overall_metrics.get('top_3_accuracy', 'N/A')} / "
                f"{overall_metrics.get('top_5_accuracy', 'N/A')} / {overall_metrics.get('top_10_accuracy', 'N/A')}")
    if args.tune_hyperparams:
        logger.info(f"\nHyperparams: C={reference_params['C']:.6f}, {reference_params['solver']}, max_iter={reference_params['max_iter']}")
    logger.info(f"{'='*80}\n")
    
    # Compute additional metrics
    logger.info("Computing per-region & network metrics...")
    per_region_metrics = compute_per_region_metrics(all_true_labels, all_predictions, region_info)
    network_metrics = compute_network_level_metrics(all_true_labels, all_predictions, region_info)
    confusion_mat = create_confusion_matrix(all_true_labels, all_predictions, n_classes)
    
    # Save results
    logger.info("Saving results...")
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
    
    if args.save_models and fold_models:
        import pickle
        with open(output_dir / 'fold_models.pkl', 'wb') as f:
            pickle.dump(fold_models, f)
    
    logger.info(f"Results saved to: {output_dir}")
    
    # Task testing
    task_results = None
    if args.test_on_task:
        if args.tune_hyperparams:
            best_params_for_task = optimize_on_full_rest_data(
                X, y, groups, args.final_optuna_trials, args.random_state, optuna_n_jobs, logger
            )
        else:
            best_params_for_task = reference_params
        
        task_results = test_on_task_data(
            best_params_for_task, args.random_state, args.n_jobs,
            args.diagonal_strategy, args.task_data, args.output_dir, args.sample,
            region_info, n_regions, X, y, groups, logger
        )
    
    return {
        'n_subjects': n_subjects, 'n_regions': n_regions,
        'n_classes': n_classes, 'n_samples': n_samples,
        'predictions': all_predictions, 'probabilities': all_probabilities,
        'true_labels': all_true_labels, 'fold_indices': all_fold_indices,
        'confusion_matrix': confusion_mat, 'overall_metrics': overall_metrics,
        'fold_metrics': fold_metrics, 'per_region_metrics': per_region_metrics,
        'network_metrics': network_metrics, 'output_dir': output_dir,
        'fold_best_params': fold_best_params if args.tune_hyperparams else None,
        'reference_params': reference_params, 'task_results': task_results
    }

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except Exception as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    optuna_n_jobs = get_optuna_n_jobs(args)
    
    logger.info("="*80)
    logger.info("FULL CONNECTIVITY MULTINOMIAL CLASSIFICATION")
    logger.info("="*80)
    if args.tune_hyperparams:
        logger.info(f"Optuna: {args.optuna_trials} trials/fold" + 
                   (f", {args.final_optuna_trials} for task" if args.test_on_task else ""))
    if args.sample:
        logger.warning(f"⚠️  TEST MODE: {args.sample} subjects only")
    logger.info("="*80)
    
    try:
        results = train_full_connectivity(args, logger, optuna_n_jobs)
        
        logger.info("\n" + "="*80)
        logger.info("COMPLETED")
        logger.info("="*80)
        logger.info(f"CV Acc: {results['overall_metrics']['accuracy']:.4f}")
        logger.info(f"Top-3/5/10: {results['overall_metrics'].get('top_3_accuracy', 'N/A')} / "
                   f"{results['overall_metrics'].get('top_5_accuracy', 'N/A')} / "
                   f"{results['overall_metrics'].get('top_10_accuracy', 'N/A')}")
        
        if args.tune_hyperparams:
            ref = results['reference_params']
            logger.info(f"Ref params: C={ref['C']:.4f}, {ref['solver']}, max_iter={ref['max_iter']}")
        
        if results.get('task_results'):
            task = results['task_results']['task_summary']
            logger.info(f"\nTask: {task['task_test_accuracy']:.4f} (drop={task['accuracy_drop']:.4f})")
            logger.info(f"Top-5: {task['task_top_5_accuracy']:.4f}")
        
        logger.info(f"\nSaved: {results['output_dir']}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()