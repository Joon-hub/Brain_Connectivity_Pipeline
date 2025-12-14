"""
04_train_fC_ovo.py

Train One-vs-One (OvO) logistic regression on FULL brain connectivity (232 regions).
This establishes pairwise classification performance to identify confusability patterns
between specific region pairs.

PREPROCESSING FLOW:
1. Diagonal imputation + Fisher Z BEFORE everything
2. StandardScaler on FULL dataset for Optuna
3. StandardScaler WITHIN each CV fold (leak-free)

NEW FEATURE: After CV, train final model on ALL rest data and test on task data

NOTE: OvO trains N*(N-1)/2 binary classifiers = 232*231/2 = 26,796 classifiers!
This is computationally expensive but provides detailed pairwise confusion information.

Usage:
    python scripts/full_connectivity/04_train_fC_ovo.py
    python scripts/full_connectivity/04_train_fC_ovo.py --tune_hyperparams
    python scripts/full_connectivity/04_train_fC_ovo.py --test_on_task
    python scripts/full_connectivity/04_train_fC_ovo.py --tune_hyperparams --test_on_task
    
    # With sampling for testing:
    python scripts/full_connectivity/04_train_fC_ovo.py --sample 30
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
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
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
    """Set up logging configuration."""
    log_file = output_dir / "training_full_connectivity_ovo.log"
    
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
        description='Train full connectivity One-vs-One logistic regression (232 regions)'
    )
    
    parser.add_argument(
        '--rest_data',
        type=Path,
        default=project_root / 'data' / 'raw' / 'PIOP2_restingstate.csv',
        help='Path to resting-state data (PIOP2, 224 subjects)'
    )
    
    parser.add_argument(
        '--task_data',
        type=Path,
        default=project_root / 'data' / 'raw' / 'PIOP1_gstroop.csv',
        help='Path to task data (PIOP1 Gender Stroop, 200 subjects)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=project_root / 'data' / 'results' / 'full_connectivity_analysis',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--config_file',
        type=Path,
        default=project_root / 'configs' / 'FC_config.yaml',
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
        help='Regularization parameter C. If None and not tuning, uses default 1.0'
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
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Number of subjects to sample for testing (takes first N subjects). If None, uses all subjects.'
    )
    
    parser.add_argument(
        '--tune_hyperparams',
        action='store_true',
        help='Enable hyperparameter tuning using Optuna'
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
    
    return parser.parse_args()


def load_config(config_file: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        logging.warning("PyYAML not installed. Using default configuration.")
        return get_default_config()
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        return get_default_config()


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        'preprocessing': {
            'apply_fisher_z': True,
            'standardize': True,
            'diagonal_strategy': 'region_mean'
        },
        'model': {
            'solver': 'lbfgs',
            'max_iter': 1000,
            'multi_class': 'ovo'
        }
    }


def create_region_info(region_list: list) -> pd.DataFrame:
    """Create region_info DataFrame from region list."""
    network_map = parse_networks(region_list)
    
    region_info = pd.DataFrame({
        'region_idx': range(len(region_list)),
        'region_name': region_list,
        'network': [network_map.get(r, 'Unknown') for r in region_list]
    })
    
    hemispheres = []
    for region in region_list:
        if region.startswith('LH_') or region.endswith('-lh'):
            hemispheres.append('left')
        elif region.startswith('RH_') or region.endswith('-rh'):
            hemispheres.append('right')
        else:
            hemispheres.append('unknown')
    
    region_info['hemisphere'] = hemispheres
    
    return region_info


def sample_first_n_subjects(df: pd.DataFrame, n_sample: int, logger: logging.Logger) -> pd.DataFrame:
    """Sample first n subjects for testing (deterministic selection)."""
    
    total_subjects = len(df)
    
    if n_sample > total_subjects:
        logger.warning(f"Requested sample size ({n_sample}) exceeds available subjects ({total_subjects}). Using all.")
        return df
    
    if n_sample <= 0:
        raise ValueError(f"Sample size must be positive, got {n_sample}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SAMPLING MODE ACTIVATED")
    logger.info(f"{'='*60}")
    logger.info(f"Selecting first {n_sample} subjects out of {total_subjects} available")
    
    df_sampled = df.head(n_sample).copy()
    logger.info(f"Sampled DataFrame shape: {df_sampled.shape}")
    logger.info(f"{'='*60}\n")
    
    return df_sampled


def apply_diagonal_imputation(connectivity: np.ndarray, region_info: pd.DataFrame, strategy: str, logger: logging.Logger) -> np.ndarray:
    """Apply diagonal imputation to connectivity matrices."""
    
    logger.info(f"\nApplying diagonal imputation (strategy: {strategy})...")
    
    n_subjects, n_regions, _ = connectivity.shape
    connectivity_imputed = connectivity.copy()
    
    orig_diag = np.array([connectivity[i].diagonal() for i in range(n_subjects)])
    logger.info(f"  Original diagonal range: [{orig_diag.min():.4f}, {orig_diag.max():.4f}]")
    
    if strategy == 'zero':
        for i in range(n_subjects):
            np.fill_diagonal(connectivity_imputed[i], 0.0)
        logger.info(f"  Set diagonal to 0.0")
        
    elif strategy == 'region_mean':
        for i in range(n_subjects):
            for j in range(n_regions):
                row_vals = connectivity_imputed[i, j, :]
                mask = np.ones(n_regions, dtype=bool)
                mask[j] = False
                row_mean = row_vals[mask].mean()
                connectivity_imputed[i, j, j] = row_mean
        logger.info(f"  Replaced diagonal with row means")
        
    elif strategy == 'network_mean':
        if 'network' not in region_info.columns:
            logger.warning("  'network' column not found, falling back to region_mean")
            return apply_diagonal_imputation(connectivity, region_info, 'region_mean', logger)
        
        for i in range(n_subjects):
            for j in range(n_regions):
                network = region_info.iloc[j]['network']
                network_mask = (region_info['network'] == network).values
                network_vals = connectivity_imputed[i, j, network_mask]
                network_vals = network_vals[network_vals != connectivity_imputed[i, j, j]]
                network_mean = network_vals.mean() if len(network_vals) > 0 else 0.0
                connectivity_imputed[i, j, j] = network_mean
        logger.info(f"  Replaced diagonal with network means")
        
    elif strategy == 'global_mean':
        for i in range(n_subjects):
            mask = ~np.eye(n_regions, dtype=bool)
            global_mean = connectivity_imputed[i][mask].mean()
            np.fill_diagonal(connectivity_imputed[i], global_mean)
        logger.info(f"  Replaced diagonal with global mean")
    else:
        raise ValueError(f"Unknown diagonal strategy: {strategy}")
    
    new_diag = np.array([connectivity_imputed[i].diagonal() for i in range(n_subjects)])
    logger.info(f"  New diagonal range: [{new_diag.min():.4f}, {new_diag.max():.4f}]")
    
    return connectivity_imputed


def apply_fisher_z_transformation(connectivity: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Apply Fisher Z-transformation to connectivity matrices."""
    
    logger.info("\nApplying Fisher Z-transformation...")
    
    connectivity_clipped = np.clip(connectivity, -0.999, 0.999)
    connectivity_transformed = np.arctanh(connectivity_clipped)
    
    if np.any(np.isnan(connectivity_transformed)):
        raise ValueError("NaN detected after Fisher Z transformation")
    if np.any(np.isinf(connectivity_transformed)):
        raise ValueError("Inf detected after Fisher Z transformation")
    
    logger.info(f"  Value range after Fisher Z: [{connectivity_transformed.min():.4f}, {connectivity_transformed.max():.4f}]")
    
    return connectivity_transformed


def prepare_classification_data(connectivity: np.ndarray, region_info: pd.DataFrame, subject_ids: np.ndarray) -> tuple:
    """Prepare data for classification."""
    n_subjects, n_regions, _ = connectivity.shape
    X = connectivity.reshape(n_subjects * n_regions, n_regions)
    y = np.tile(np.arange(n_regions), n_subjects)
    groups = np.repeat(subject_ids, n_regions)
    return X, y, groups


def preprocess_fold_data(X_train: np.ndarray, X_test: np.ndarray, logger: logging.Logger) -> tuple:
    """Preprocess data within a single fold (StandardScaler only)."""
    
    logger.info(f"  Preprocessing fold data (StandardScaler only)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
        raise ValueError("NaN or Inf detected in training data after scaling")
    if np.any(np.isnan(X_test_scaled)) or np.any(np.isinf(X_test_scaled)):
        raise ValueError("Inf detected in test data after scaling")
    
    return X_train_scaled, X_test_scaled, scaler


def optimize_hyperparameters_optuna(X: np.ndarray, y: np.ndarray, n_trials: int, random_state: int, n_jobs: int, logger: logging.Logger) -> dict:
    """Optimize hyperparameters using Optuna."""
    
    n_classes = len(np.unique(y))
    n_pairwise = n_classes * (n_classes - 1) // 2
    
    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION WITH OPTUNA (ONE-VS-ONE)")
    logger.info("="*80)
    logger.info(f"  Classes: {n_classes}, Pairwise classifiers: {n_pairwise}")
    
    def objective(trial):
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        max_iter = trial.suggest_categorical('max_iter', [200, 500, 1000])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
        
        model = OneVsOneClassifier(
            LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=random_state, n_jobs=1, verbose=0),
            n_jobs=n_jobs
        )
        
        model.fit(X_train, y_train)
        return accuracy_score(y_val, model.predict(X_val))
    
    optuna_start = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)
    optuna_time = time.time() - optuna_start
    
    best_params = study.best_params
    best_params['_optuna_best_score'] = float(study.best_value)
    best_params['_optuna_n_trials'] = n_trials
    best_params['_optuna_time_seconds'] = float(optuna_time)
    
    logger.info(f"Best params: C={best_params['C']:.6f}, solver={best_params['solver']}, score={study.best_value:.4f}")
    logger.info("="*80 + "\n")
    
    return best_params


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_decision: np.ndarray = None) -> dict:
    """Compute classification metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'n_samples': int(len(y_true)),
        'n_classes': int(len(np.unique(y_true)))
    }
    
    if y_decision is not None and len(y_decision.shape) == 2:
        n_classes = y_decision.shape[1]
        for k in [3, 5, 10]:
            if k <= n_classes:
                top_k_preds = np.argsort(y_decision, axis=1)[:, -k:]
                top_k_correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
                metrics[f'top_{k}_accuracy'] = float(top_k_correct.mean())
    
    return metrics


def compute_per_region_metrics(y_true: np.ndarray, y_pred: np.ndarray, region_info: pd.DataFrame) -> pd.DataFrame:
    """Compute per-region classification metrics."""
    n_regions = len(region_info)
    per_region = []
    
    for region_idx in range(n_regions):
        mask = (y_true == region_idx)
        if mask.sum() > 0:
            region_acc = (y_pred[mask] == region_idx).sum() / mask.sum()
        else:
            region_acc = 0.0
        
        per_region.append({
            'region_idx': region_idx,
            'region_name': region_info.iloc[region_idx]['region_name'],
            'network': region_info.iloc[region_idx]['network'],
            'hemisphere': region_info.iloc[region_idx].get('hemisphere', 'unknown'),
            'accuracy': region_acc,
            'n_samples': int(mask.sum())
        })
    
    return pd.DataFrame(per_region)


def compute_network_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, region_info: pd.DataFrame) -> pd.DataFrame:
    """Compute network-level classification metrics."""
    networks = region_info['network'].unique()
    network_metrics = []
    
    for network in networks:
        network_regions = region_info[region_info['network'] == network]['region_idx'].values
        mask = np.isin(y_true, network_regions)
        if mask.sum() > 0:
            network_acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum()
        else:
            network_acc = 0.0
        
        network_metrics.append({
            'network': network,
            'accuracy': network_acc,
            'n_regions': len(network_regions),
            'n_samples': int(mask.sum())
        })
    
    return pd.DataFrame(network_metrics).sort_values('accuracy', ascending=False)


def create_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Create confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def plot_confusion_matrix(confusion_mat: np.ndarray, region_info: pd.DataFrame, save_path: Path, title: str = 'Confusion Matrix'):
    """Plot and save confusion matrix."""
    n_regions = len(confusion_mat)
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = confusion_mat / row_sums
    
    fig_size = max(12, n_regions * 0.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Region', fontsize=12)
    ax.set_ylabel('True Region', fontsize=12)
    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_region_accuracy(per_region_metrics: pd.DataFrame, save_path: Path, title: str = 'Per-Region Accuracy'):
    """Plot per-region accuracy."""
    sorted_metrics = per_region_metrics.sort_values('accuracy', ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_metrics) * 0.15)))
    colors = plt.cm.RdYlGn(sorted_metrics['accuracy'])
    ax.barh(range(len(sorted_metrics)), sorted_metrics['accuracy'], color=colors)
    ax.set_yticks(range(len(sorted_metrics)))
    ax.set_yticklabels(sorted_metrics['region_name'], fontsize=6)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=sorted_metrics['accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {sorted_metrics["accuracy"].mean():.3f}')
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_network_accuracy(network_metrics: pd.DataFrame, save_path: Path, title: str = 'Network-Level Accuracy'):
    """Plot network-level accuracy."""
    sorted_metrics = network_metrics.sort_values('accuracy', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(sorted_metrics['accuracy'])
    bars = ax.barh(range(len(sorted_metrics)), sorted_metrics['accuracy'], color=colors)
    ax.set_yticks(range(len(sorted_metrics)))
    ax.set_yticklabels(sorted_metrics['network'], fontsize=10)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=sorted_metrics['accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {sorted_metrics["accuracy"].mean():.3f}')
    ax.legend()
    ax.set_xlim(0, 1)
    for bar, acc in zip(bars, sorted_metrics['accuracy']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_on_task_data(C: float, max_iter: int, solver: str, random_state: int, n_jobs: int, diagonal_strategy: str, rest_data_path: Path, task_data_path: Path, output_dir: Path, sample: int, logger: logging.Logger) -> dict:
    """Train final OvO model on ALL rest data, then test on task data."""
    
    logger.info("\n" + "="*80)
    logger.info("TESTING ON TASK DATA (GENDER STROOP) - ONE-VS-ONE")
    logger.info("="*80)
    
    # Load rest data
    df_rest = load_connectivity_data(str(rest_data_path))
    if sample is not None:
        df_rest = sample_first_n_subjects(df_rest, sample, logger)
    
    connection_columns = extract_connection_columns(df_rest)
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    subject_ids_rest = df_rest.iloc[:, 0].values
    n_pairwise = n_regions * (n_regions - 1) // 2
    
    logger.info(f"  Rest: {len(df_rest)} subjects, {n_regions} regions, {n_pairwise} pairwise classifiers")
    
    rest_connectivity = reconstruct_matrices_from_dataframe(df_rest, connection_columns, region_to_idx, n_regions)
    region_info = create_region_info(region_list)
    rest_connectivity = apply_diagonal_imputation(rest_connectivity, region_info, diagonal_strategy, logger)
    rest_connectivity = apply_fisher_z_transformation(rest_connectivity, logger)
    X_rest, y_rest, _ = prepare_classification_data(rest_connectivity, region_info, subject_ids_rest)
    
    # Load task data
    df_task = load_connectivity_data(str(task_data_path))
    if sample is not None:
        df_task = sample_first_n_subjects(df_task, sample, logger)
    
    task_connection_columns = extract_connection_columns(df_task)
    subject_ids_task = df_task.iloc[:, 0].values
    
    task_connectivity = reconstruct_matrices_from_dataframe(df_task, task_connection_columns, region_to_idx, n_regions)
    task_connectivity = apply_diagonal_imputation(task_connectivity, region_info, diagonal_strategy, logger)
    task_connectivity = apply_fisher_z_transformation(task_connectivity, logger)
    X_task, y_task, _ = prepare_classification_data(task_connectivity, region_info, subject_ids_task)
    
    # Train final model
    scaler_final = StandardScaler()
    X_rest_scaled = scaler_final.fit_transform(X_rest)
    
    final_model = OneVsOneClassifier(
        LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=random_state, n_jobs=1, verbose=0),
        n_jobs=n_jobs
    )
    
    train_start = time.time()
    final_model.fit(X_rest_scaled, y_rest)
    logger.info(f"  OvO model trained in {time.time() - train_start:.2f}s ({len(final_model.estimators_)} classifiers)")
    
    y_rest_pred = final_model.predict(X_rest_scaled)
    rest_accuracy = accuracy_score(y_rest, y_rest_pred)
    
    # Test on task
    X_task_scaled = scaler_final.transform(X_task)
    y_task_pred = final_model.predict(X_task_scaled)
    
    try:
        y_task_decision = final_model.decision_function(X_task_scaled)
    except:
        y_task_decision = None
    
    task_metrics = compute_classification_metrics(y_task, y_task_pred, y_task_decision)
    task_per_region = compute_per_region_metrics(y_task, y_task_pred, region_info)
    task_network = compute_network_level_metrics(y_task, y_task_pred, region_info)
    task_confusion = create_confusion_matrix(y_task, y_task_pred, n_regions)
    
    # Save results
    task_output_dir = output_dir / "task_testing_ovo"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(task_output_dir / 'task_predictions.npy', y_task_pred)
    if y_task_decision is not None:
        np.save(task_output_dir / 'task_decision_function.npy', y_task_decision)
    np.save(task_output_dir / 'task_true_labels.npy', y_task)
    np.save(task_output_dir / 'task_confusion_matrix.npy', task_confusion)
    
    task_summary = {
        'rest_train_accuracy': float(rest_accuracy),
        'task_test_accuracy': float(task_metrics['accuracy']),
        'task_balanced_accuracy': float(task_metrics['balanced_accuracy']),
        'accuracy_drop': float(rest_accuracy - task_metrics['accuracy']),
        'hyperparameters': {'C': float(C), 'max_iter': int(max_iter), 'solver': solver, 'multi_class': 'ovo'},
        'n_rest_subjects': int(len(df_rest)),
        'n_task_subjects': int(len(df_task)),
        'n_regions': int(n_regions),
        'n_pairwise_classifiers': int(len(final_model.estimators_))
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
    
    plot_confusion_matrix(task_confusion, region_info, task_output_dir / 'task_confusion_matrix.png', 'Full Connectivity - Task Testing (OvO)')
    plot_per_region_accuracy(task_per_region, task_output_dir / 'task_per_region_accuracy.png', 'Task Per-Region Accuracy (OvO)')
    plot_network_accuracy(task_network, task_output_dir / 'task_network_accuracy.png', 'Task Network Accuracy (OvO)')
    
    logger.info(f"\nRest accuracy: {rest_accuracy:.4f}, Task accuracy: {task_metrics['accuracy']:.4f}, Drop: {task_summary['accuracy_drop']:.4f}")
    
    return {'task_metrics': task_metrics, 'task_per_region': task_per_region, 'task_network': task_network, 'task_summary': task_summary}


def train_full_connectivity_ovo(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """Train One-vs-One model for full brain connectivity (232 regions)."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING FULL BRAIN CONNECTIVITY (232 REGIONS) - ONE-VS-ONE")
    logger.info(f"{'='*80}\n")
    
    output_dir = args.output_dir / "ovo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_train = load_connectivity_data(str(args.rest_data))
    if args.sample is not None:
        df_train = sample_first_n_subjects(df_train, args.sample, logger)
        logger.warning(f"⚠️  TESTING MODE: Using first {args.sample} subjects only")
    
    connection_columns = extract_connection_columns(df_train)
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    subject_ids = df_train.iloc[:, 0].values
    n_subjects = len(subject_ids)
    n_pairwise = n_regions * (n_regions - 1) // 2
    
    logger.info(f"Data: {n_subjects} subjects, {n_regions} regions, {n_pairwise} pairwise classifiers")
    
    connectivity = reconstruct_matrices_from_dataframe(df_train, connection_columns, region_to_idx, n_regions)
    region_info = create_region_info(region_list)
    
    # Preprocessing
    connectivity = apply_diagonal_imputation(connectivity, region_info, args.diagonal_strategy, logger)
    connectivity = apply_fisher_z_transformation(connectivity, logger)
    X, y, groups = prepare_classification_data(connectivity, region_info, subject_ids)
    
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    
    # Hyperparameter optimization
    if args.tune_hyperparams:
        scaler_optuna = StandardScaler()
        X_scaled_optuna = scaler_optuna.fit_transform(X)
        best_params = optimize_hyperparameters_optuna(X_scaled_optuna, y, args.optuna_trials, args.random_state, args.n_jobs, logger)
        C, max_iter, solver = best_params['C'], best_params['max_iter'], best_params['solver']
    else:
        C = args.regularization_C if args.regularization_C is not None else 1.0
        max_iter, solver = args.max_iter, 'lbfgs'
        best_params = {'C': C, 'max_iter': max_iter, 'solver': solver, '_optuna_best_score': None, '_optuna_n_trials': 0}
        logger.info(f"Using fixed hyperparameters: C={C}, solver={solver}")
    
    # Cross-validation
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION (ONE-VS-ONE) - {n_pairwise} pairwise classifiers per fold")
    logger.info(f"{'='*80}\n")
    
    gkf = GroupKFold(n_splits=args.n_folds)
    all_predictions, all_decision_functions, all_true_labels, all_fold_indices = [], [], [], []
    fold_models = [] if args.save_models else None
    fold_metrics = []
    
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        fold_start = time.time()
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]
        
        assert len(set(groups_train) & set(groups_test)) == 0, "Subject leakage!"
        
        X_train_scaled, X_test_scaled, _ = preprocess_fold_data(X_train, X_test, logger)
        
        model = OneVsOneClassifier(
            LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=args.random_state, n_jobs=1, verbose=0),
            n_jobs=args.n_jobs
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        try:
            y_decision = model.decision_function(X_test_scaled)
        except:
            y_decision = None
        
        fold_acc = accuracy_score(y_test, y_pred)
        fold_bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        fold_metrics.append({
            'fold': fold_idx + 1, 'accuracy': fold_acc, 'balanced_accuracy': fold_bal_acc,
            'n_train': len(y_train), 'n_test': len(y_test), 'n_pairwise_classifiers': len(model.estimators_),
            'hyperparameters': {'C': float(C), 'max_iter': int(max_iter), 'solver': solver, 'multi_class': 'ovo'}
        })
        
        logger.info(f"  Accuracy: {fold_acc:.4f}, Time: {time.time() - fold_start:.2f}s\n")
        
        all_predictions.extend(y_pred)
        if y_decision is not None:
            all_decision_functions.append(y_decision)
        all_true_labels.extend(y_test)
        all_fold_indices.extend([fold_idx + 1] * len(y_test))
        
        if args.save_models:
            fold_models.append({'fold': fold_idx + 1, 'model': model, 'train_idx': train_idx, 'test_idx': test_idx})
    
    logger.info(f"CV completed in {time.time() - start_time:.2f}s\n")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_decision_functions = np.vstack(all_decision_functions) if all_decision_functions else None
    all_true_labels = np.array(all_true_labels)
    all_fold_indices = np.array(all_fold_indices)
    
    # Compute metrics
    overall_metrics = compute_classification_metrics(all_true_labels, all_predictions, all_decision_functions)
    overall_metrics['best_hyperparameters'] = best_params
    overall_metrics['model_type'] = 'one_vs_one'
    overall_metrics['n_pairwise_classifiers'] = n_pairwise
    
    per_region_metrics = compute_per_region_metrics(all_true_labels, all_predictions, region_info)
    network_metrics = compute_network_level_metrics(all_true_labels, all_predictions, region_info)
    confusion_mat = create_confusion_matrix(all_true_labels, all_predictions, n_classes)
    
    logger.info(f"OVERALL: Accuracy={overall_metrics['accuracy']:.4f}, Balanced={overall_metrics['balanced_accuracy']:.4f}")
    
    # Save results
    np.save(output_dir / 'cv_predictions.npy', all_predictions)
    if all_decision_functions is not None:
        np.save(output_dir / 'cv_decision_functions.npy', all_decision_functions)
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
    
    if args.save_models and fold_models:
        import pickle
        with open(output_dir / 'fold_models.pkl', 'wb') as f:
            pickle.dump(fold_models, f)
    
    # Visualizations
    plot_confusion_matrix(confusion_mat, region_info, output_dir / 'confusion_matrix.png', 'Full Connectivity (232 Regions) - OvO')
    plot_per_region_accuracy(per_region_metrics, output_dir / 'per_region_accuracy.png', 'Per-Region Accuracy (OvO)')
    plot_network_accuracy(network_metrics, output_dir / 'network_accuracy.png', 'Network Accuracy (OvO)')
    
    # Task testing
    task_results = None
    if args.test_on_task:
        task_results = test_on_task_data(C, max_iter, solver, args.random_state, args.n_jobs, args.diagonal_strategy, args.rest_data, args.task_data, args.output_dir, args.sample, logger)
    
    return {
        'n_subjects': n_subjects, 'n_regions': n_regions, 'n_classes': n_classes, 'n_samples': n_samples,
        'n_pairwise_classifiers': n_pairwise, 'predictions': all_predictions, 'decision_functions': all_decision_functions,
        'true_labels': all_true_labels, 'fold_indices': all_fold_indices, 'confusion_matrix': confusion_mat,
        'overall_metrics': overall_metrics, 'fold_metrics': fold_metrics, 'per_region_metrics': per_region_metrics,
        'network_metrics': network_metrics, 'region_info': region_info, 'output_dir': output_dir, 'task_results': task_results
    }


def main():
    """Main function."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("="*80)
    logger.info("FULL BRAIN CONNECTIVITY CLASSIFICATION (ONE-VS-ONE)")
    logger.info("232 Regions | 224 Subjects | 26,796 Pairwise Classifiers")
    logger.info("="*80)
    
    if args.sample:
        logger.warning(f"⚠️  TESTING MODE: {args.sample} subjects only")
    
    try:
        results = train_full_connectivity_ovo(args, logger)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED (ONE-VS-ONE)")
        logger.info("="*80)
        logger.info(f"CV Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        logger.info(f"Pairwise Classifiers: {results['n_pairwise_classifiers']}")
        
        if results.get('task_results'):
            logger.info(f"Task Accuracy: {results['task_results']['task_summary']['task_test_accuracy']:.4f}")
        
        logger.info(f"Results saved to: {results['output_dir']}")
        
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()