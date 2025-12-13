"""
02_train_hemisphere_multinomial.py

Train multinomial logistic regression separately for left and right hemispheres.
This establishes the baseline performance for hemisphere-specific classification.

FINAL PREPROCESSING FLOW:
1. Diagonal imputation + Fisher Z BEFORE everything
2. StandardScaler on FULL dataset for Optuna
3. StandardScaler WITHIN each CV fold (leak-free)

NEW FEATURE: After CV, train final model on ALL rest data and test on task data

Usage:
    python scripts/hemisphere/02_train_hemisphere_multinomial.py --hemisphere left
    python scripts/hemisphere/02_train_hemisphere_multinomial.py --hemisphere right
    python scripts/hemisphere/02_train_hemisphere_multinomial.py --hemisphere both
    
    # With task testing:
    python scripts/hemisphere/02_train_hemisphere_multinomial.py --hemisphere left --test_on_task
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
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler


# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import your existing modules
from src.hemisphere.hemisphere_utils import (
    load_hemisphere_data,
    prepare_classification_data
)
from src.core.preprocessing import ConnectivityPreprocessor
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
            'multi_class': 'multinomial'
        }
    }


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
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SAMPLING MODE ACTIVATED")
    logger.info(f"{'='*60}")
    logger.info(f"Selecting first {n_sample} subjects out of {total_subjects} available")
    
    connectivity_sampled = data['connectivity'][:n_sample]
    subject_ids_sampled = data['subject_ids'][:n_sample]
    
    logger.info(f"Selected subjects: {', '.join(map(str, subject_ids_sampled[:10]))}" + 
                (f"... (+{n_sample-10} more)" if n_sample > 10 else ""))
    
    sampled_data = {
        'connectivity': connectivity_sampled,
        'subject_ids': subject_ids_sampled,
        'region_info': data['region_info'],
        'hemisphere': data['hemisphere'],
        'n_subjects': n_sample,
        'n_regions': data['n_regions']
    }
    
    logger.info(f"Sampled connectivity shape: {connectivity_sampled.shape}")
    logger.info(f"Sampled subjects: {len(subject_ids_sampled)}")
    logger.info(f"{'='*60}\n")
    
    return sampled_data


def apply_diagonal_imputation(
    connectivity: np.ndarray,
    region_info: pd.DataFrame,
    strategy: str,
    logger: logging.Logger
) -> np.ndarray:
    """Apply diagonal imputation to connectivity matrices BEFORE creating classification data."""
    
    logger.info(f"\nApplying diagonal imputation (strategy: {strategy})...")
    
    n_subjects, n_regions, _ = connectivity.shape
    connectivity_imputed = connectivity.copy()
    
    # Log original diagonal values
    orig_diag = np.array([connectivity[i].diagonal() for i in range(n_subjects)])
    logger.info(f"  Original diagonal range: [{orig_diag.min():.4f}, {orig_diag.max():.4f}]")
    logger.info(f"  Original diagonal mean: {orig_diag.mean():.4f}")
    
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
                if len(network_vals) > 0:
                    network_mean = network_vals.mean()
                else:
                    network_mean = 0.0
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
    
    # Log new diagonal values
    new_diag = np.array([connectivity_imputed[i].diagonal() for i in range(n_subjects)])
    logger.info(f"  New diagonal range: [{new_diag.min():.4f}, {new_diag.max():.4f}]")
    logger.info(f"  New diagonal mean: {new_diag.mean():.4f}")
    
    return connectivity_imputed


def apply_fisher_z_transformation(
    connectivity: np.ndarray,
    logger: logging.Logger
) -> np.ndarray:
    """Apply Fisher Z-transformation to connectivity matrices."""
    
    logger.info("\nApplying Fisher Z-transformation...")
    
    connectivity_clipped = np.clip(connectivity, -0.999, 0.999)
    connectivity_transformed = np.arctanh(connectivity_clipped)
    
    if np.any(np.isnan(connectivity_transformed)):
        raise ValueError("NaN detected after Fisher Z transformation")
    if np.any(np.isinf(connectivity_transformed)):
        raise ValueError("Inf detected after Fisher Z transformation")
    
    logger.info(f"  Value range after Fisher Z: [{connectivity_transformed.min():.4f}, {connectivity_transformed.max():.4f}]")
    logger.info(f"  Mean: {connectivity_transformed.mean():.4f}, Std: {connectivity_transformed.std():.4f}")
    
    return connectivity_transformed


def preprocess_fold_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    logger: logging.Logger
) -> tuple:
    """
    Preprocess data within a single fold.
    
    ONLY StandardScaler (diagonal imputation and Fisher Z already done).
    This ensures no data leakage - scaler is fit on training data only.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (already Fisher Z transformed)
    X_test : np.ndarray
        Test features (already Fisher Z transformed)
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    X_train_scaled, X_test_scaled : tuple
        Scaled features
    """
    
    logger.info(f"  Preprocessing fold data (StandardScaler only)...")
    logger.info(f"  Input shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Standardization - fit on training data only (LEAK-FREE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Validate
    if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
        raise ValueError("NaN or Inf detected in training data after scaling")
    if np.any(np.isnan(X_test_scaled)) or np.any(np.isinf(X_test_scaled)):
        raise ValueError("Inf detected in test data after scaling")
    
    logger.info(f"  Scaled shapes - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def optimize_hyperparameters_optuna(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    random_state: int,
    logger: logging.Logger
) -> dict:
    """
    Optimize hyperparameters using Optuna on SCALED full dataset.
    
    NOTE: X should already be scaled with StandardScaler fit on full dataset.
    This is for Optuna exploration only - final CV will re-scale properly within folds.
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix (already scaled)
    y : np.ndarray
        Full labels
    n_trials : int
        Number of Optuna trials
    random_state : int
        Random state
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    best_params : dict
        Best hyperparameters
    """
    
    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    logger.info("="*80)
    logger.info(f"Running Optuna on SCALED full dataset")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Samples: {len(X)}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Classes: {len(np.unique(y))}")
    logger.info(f"  NOTE: Data already scaled, using simple train/val split")
    
    def objective(trial):
        """Optuna objective function."""
        
        # Suggest hyperparameters
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        max_iter = trial.suggest_categorical('max_iter', [200, 500, 1000])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
        
        # Simple train/test split (data already scaled)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # Train model
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            n_jobs=1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        return score
    
    # Create Optuna study
    logger.info("\nStarting Optuna optimization...")
    optuna_start = time.time()
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1
    )
    
    optuna_time = time.time() - optuna_start
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"\nOptuna optimization completed in {optuna_time:.2f}s")
    logger.info(f"\n✓ Best hyperparameters found:")
    logger.info(f"  • C: {best_params['C']:.6f}")
    logger.info(f"  • max_iter: {best_params['max_iter']}")
    logger.info(f"  • solver: {best_params['solver']}")
    logger.info(f"  • Best validation score: {best_score:.4f}")
    
    # Show top trials
    logger.info(f"\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False).head(5)
    
    for idx, (_, row) in enumerate(trials_df.iterrows(), 1):
        logger.info(
            f"  {idx}. Trial {int(row['number'])}: "
            f"score={row['value']:.4f}, "
            f"C={row['params_C']:.4f}, "
            f"solver={row['params_solver']}, "
            f"max_iter={int(row['params_max_iter'])}"
        )
    
    logger.info("="*80 + "\n")
    
    # Add metadata
    best_params['_optuna_best_score'] = float(best_score)
    best_params['_optuna_n_trials'] = n_trials
    best_params['_optuna_time_seconds'] = float(optuna_time)
    
    return best_params


def test_on_task_data(
    hemisphere: str,
    C: float,
    max_iter: int,
    solver: str,
    random_state: int,
    n_jobs: int,
    diagonal_strategy: str,
    data_dir: Path,
    output_dir: Path,
    sample: int,
    logger: logging.Logger
) -> dict:
    """
    Train final model on ALL rest data, then test on task data.
    
    This reveals how well the model generalizes from resting-state to task-based connectivity.
    
    Parameters
    ----------
    hemisphere : str
        'left' or 'right'
    C, max_iter, solver : model hyperparameters
    random_state, n_jobs : training parameters
    diagonal_strategy : preprocessing parameter
    data_dir, output_dir : paths
    sample : number of subjects (or None for all)
    logger : logging instance
    
    Returns
    -------
    task_results : dict
        Results from task testing
    """
    
    logger.info("\n" + "="*80)
    logger.info("TESTING ON TASK DATA (GENDER STROOP)")
    logger.info("="*80)
    logger.info(f"Training final model on ALL resting-state data")
    logger.info(f"Then testing on task data to measure generalization\n")
    
    # =========================================================================
    # STEP 1: Load and preprocess RESTING-STATE data (training)
    # =========================================================================
    
    logger.info("Step 1: Loading resting-state data (TRAINING)...")
    rest_data = load_hemisphere_data(
        data_dir=data_dir,
        hemisphere=hemisphere,
        dataset='rest',
        return_matrix=True,
        validate=True
    )
    
    if sample is not None:
        rest_data = sample_first_n_subjects(rest_data, sample, logger)
    
    logger.info(f"  Rest data: {rest_data['n_subjects']} subjects, {rest_data['n_regions']} regions")
    
    # Preprocess rest data
    rest_connectivity = apply_diagonal_imputation(
        connectivity=rest_data['connectivity'],
        region_info=rest_data['region_info'],
        strategy=diagonal_strategy,
        logger=logger
    )
    
    rest_connectivity = apply_fisher_z_transformation(
        connectivity=rest_connectivity,
        logger=logger
    )
    
    X_rest, y_rest, groups_rest = prepare_classification_data(
        connectivity=rest_connectivity,
        region_info=rest_data['region_info'],
        subject_ids=rest_data['subject_ids']
    )
    
    logger.info(f"  Rest features: {X_rest.shape}")
    
    # =========================================================================
    # STEP 2: Load and preprocess TASK data (testing)
    # =========================================================================
    
    logger.info("\nStep 2: Loading task data (TESTING)...")
    task_data = load_hemisphere_data(
        data_dir=data_dir,
        hemisphere=hemisphere,
        dataset='task',  # This loads PIOP1 Gender Stroop data
        return_matrix=True,
        validate=True
    )
    
    if sample is not None:
        task_data = sample_first_n_subjects(task_data, sample, logger)
    
    logger.info(f"  Task data: {task_data['n_subjects']} subjects, {task_data['n_regions']} regions")
    
    # Preprocess task data (SAME pipeline as rest)
    task_connectivity = apply_diagonal_imputation(
        connectivity=task_data['connectivity'],
        region_info=task_data['region_info'],
        strategy=diagonal_strategy,
        logger=logger
    )
    
    task_connectivity = apply_fisher_z_transformation(
        connectivity=task_connectivity,
        logger=logger
    )
    
    X_task, y_task, groups_task = prepare_classification_data(
        connectivity=task_connectivity,
        region_info=task_data['region_info'],
        subject_ids=task_data['subject_ids']
    )
    
    logger.info(f"  Task features: {X_task.shape}")
    
    # =========================================================================
    # STEP 3: Train final model on ALL rest data
    # =========================================================================
    
    logger.info("\nStep 3: Training final model on ALL resting-state data...")
    
    # Scale rest data
    scaler_final = StandardScaler()
    X_rest_scaled = scaler_final.fit_transform(X_rest)
    
    logger.info(f"  Scaled rest data: mean={X_rest_scaled.mean():.4f}, std={X_rest_scaled.std():.4f}")
    
    # Train final model
    final_model = LogisticRegression(
        C=C,
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
    
    # Scale task data using REST scaler (important!)
    X_task_scaled = scaler_final.transform(X_task)
    
    logger.info(f"  Scaled task data: mean={X_task_scaled.mean():.4f}, std={X_task_scaled.std():.4f}")
    
    # Predict on task data
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
        region_info=task_data['region_info']
    )
    
    task_network = compute_network_level_metrics(
        y_true=y_task,
        y_pred=y_task_pred,
        region_info=task_data['region_info']
    )
    
    task_confusion = create_confusion_matrix(
        y_true=y_task,
        y_pred=y_task_pred,
        n_classes=len(np.unique(y_task))
    )
    
    # =========================================================================
    # STEP 6: Save results
    # =========================================================================
    
    task_output_dir = output_dir / f"{hemisphere}_hemisphere" / "task_testing"
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
        'task_balanced_accuracy': float(task_metrics['balanced_accuracy']),
        'task_top_5_accuracy': float(task_metrics.get('top_5_accuracy', 0)),
        'accuracy_drop': float(rest_accuracy - task_metrics['accuracy']),
        'hyperparameters': {
            'C': float(C),
            'max_iter': int(max_iter),
            'solver': solver
        },
        'n_rest_subjects': int(rest_data['n_subjects']),
        'n_task_subjects': int(task_data['n_subjects']),
        'n_rest_samples': int(len(X_rest)),
        'n_task_samples': int(len(X_task))
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
    
    # Generate visualizations
    logger.info("\nGenerating task testing visualizations...")
    
    plot_confusion_matrix(
        confusion_mat=task_confusion,
        region_info=task_data['region_info'],
        save_path=task_output_dir / 'task_confusion_matrix.png',
        title=f'{hemisphere.capitalize()} - Task Testing Confusion Matrix'
    )
    
    plot_per_region_accuracy(
        per_region_metrics=task_per_region,
        save_path=task_output_dir / 'task_per_region_accuracy.png',
        title=f'{hemisphere.capitalize()} - Task Testing Per-Region Accuracy'
    )
    
    plot_network_accuracy(
        network_metrics=task_network,
        save_path=task_output_dir / 'task_network_accuracy.png',
        title=f'{hemisphere.capitalize()} - Task Testing Network Accuracy'
    )
    
    # =========================================================================
    # STEP 7: Report results
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("TASK TESTING RESULTS")
    logger.info("="*80)
    logger.info(f"Rest (training) accuracy: {rest_accuracy:.4f}")
    logger.info(f"Task (testing) accuracy:  {task_metrics['accuracy']:.4f}")
    logger.info(f"Accuracy drop:            {task_summary['accuracy_drop']:.4f} ({task_summary['accuracy_drop']/rest_accuracy*100:.1f}%)")
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


def train_single_hemisphere(
    hemisphere: str,
    args: argparse.Namespace,
    logger: logging.Logger
) -> dict:
    """
    Train multinomial model for a single hemisphere.
    
    PREPROCESSING FLOW:
    1. Load connectivity matrices
    2. Diagonal imputation (ONCE)
    3. Fisher Z transformation (ONCE)
    4. Flatten to features
    5. [If tuning] Scale with StandardScaler → Run Optuna
    6. Cross-validation (scale within each fold properly)
    7. [If test_on_task] Train final model on all rest data → Test on task data
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING {hemisphere.upper()} HEMISPHERE")
    logger.info(f"{'='*80}\n")
    
    # Create output directory
    output_dir = args.output_dir / f"{hemisphere}_hemisphere" / "multinomial"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading hemisphere-specific data...")
    data = load_hemisphere_data(
        data_dir=args.data_dir,
        hemisphere=hemisphere,
        dataset='rest',
        return_matrix=True,
        validate=True
    )
    
    # Sample subjects if specified
    if args.sample is not None:
        data = sample_first_n_subjects(data, args.sample, logger)
        logger.warning(f"⚠️  TESTING MODE: Using first {args.sample} subjects only")
    
    connectivity = data['connectivity']
    subject_ids = data['subject_ids']
    region_info = data['region_info']
    
    n_subjects, n_regions, _ = connectivity.shape
    
    logger.info(f"Data loaded:")
    logger.info(f"  Subjects: {n_subjects}")
    logger.info(f"  Regions: {n_regions}")
    logger.info(f"  Connectivity shape: {connectivity.shape}")
    
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
    
    # Validate
    assert X.shape[0] == len(y), "Mismatch between X and y"
    assert X.shape[0] == len(groups), "Mismatch between X and groups"
    
    logger.info("\n✓ Preprocessing completed")
    logger.info("="*80)
    
    # ==========================================================================
    # HYPERPARAMETER OPTIMIZATION (IF ENABLED)
    # ==========================================================================
    
    if args.tune_hyperparams:
        # Scale full dataset for Optuna
        logger.info("\nScaling full dataset for Optuna optimization...")
        scaler_optuna = StandardScaler()
        X_scaled_optuna = scaler_optuna.fit_transform(X)
        logger.info(f"  Scaled data - mean: {X_scaled_optuna.mean():.4f}, std: {X_scaled_optuna.std():.4f}")
        
        # Run Optuna on scaled data
        best_params = optimize_hyperparameters_optuna(
            X=X_scaled_optuna,
            y=y,
            n_trials=args.optuna_trials,
            random_state=args.random_state,
            logger=logger
        )
        
        C = best_params['C']
        max_iter = best_params['max_iter']
        solver = best_params['solver']
        
    else:
        # Use default parameters
        C = args.regularization_C if args.regularization_C is not None else 1.0
        max_iter = args.max_iter
        solver = 'lbfgs'
        
        best_params = {
            'C': C,
            'max_iter': max_iter,
            'solver': solver,
            '_optuna_best_score': None,
            '_optuna_n_trials': 0
        }
        
        logger.info(f"\nUsing fixed hyperparameters:")
        logger.info(f"  C: {C}, max_iter: {max_iter}, solver: {solver}\n")
    
    # ==========================================================================
    # CROSS-VALIDATION (WITH PROPER FOLD-WISE SCALING)
    # ==========================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION WITH FOLD-WISE SCALING")
    logger.info(f"{'='*80}")
    logger.info(f"Using hyperparameters: C={C:.6f}, solver={solver}, max_iter={max_iter}")
    logger.info(f"Running {args.n_folds}-fold GroupKFold cross-validation...")
    logger.info(f"Note: StandardScaler fit independently on each fold's training data\n")
    
    # Set up cross-validation
    gkf = GroupKFold(n_splits=args.n_folds)
    
    # Storage
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_fold_indices = []
    fold_models = [] if args.save_models else None
    fold_metrics = []
    
    # CV loop
    start_time = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        fold_start = time.time()
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
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
        
        # Scale within fold (LEAK-FREE)
        X_train_scaled, X_test_scaled, _ = preprocess_fold_data(
            X_train=X_train,
            X_test=X_test,
            logger=logger
        )
        
        # Train model
        logger.info(f"  Training with C={C:.6f}, solver={solver}...")
        
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=1 if args.verbose else 0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        logger.info("  Predicting on test set...")
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        fold_acc = accuracy_score(y_test, y_pred)
        fold_bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        fold_metric_dict = {
            'fold': fold_idx + 1,
            'accuracy': fold_acc,
            'balanced_accuracy': fold_bal_acc,
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
        
        fold_metrics.append(fold_metric_dict)
        
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
    
    overall_metrics['best_hyperparameters'] = best_params
    overall_metrics['preprocessing'] = {
        'diagonal_strategy': args.diagonal_strategy,
        'fisher_z_applied': True,
        'standardize_per_fold': True
    }
    
    logger.info(f"\nOVERALL RESULTS ({hemisphere.upper()} HEMISPHERE):")
    logger.info(f"  Mean CV Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"  Mean CV Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Top-5 Accuracy: {overall_metrics.get('top_5_accuracy', 'N/A')}")
    
    if args.tune_hyperparams:
        logger.info(f"\n  Hyperparameters (from Optuna):")
        logger.info(f"    C: {C:.6f}, solver: {solver}, max_iter: {max_iter}")
        logger.info(f"    Optuna validation score: {best_params['_optuna_best_score']:.4f}")
    
    # Compute per-region metrics
    logger.info("\nComputing per-region metrics...")
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
    
    # Save results
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
    
    if args.save_models and fold_models is not None:
        import pickle
        with open(output_dir / 'fold_models.pkl', 'wb') as f:
            pickle.dump(fold_models, f)
    
    # Visualizations
    logger.info("\nGenerating visualizations...")
    
    plot_confusion_matrix(
        confusion_mat=confusion_mat,
        region_info=region_info,
        save_path=output_dir / 'confusion_matrix.png',
        title=f'{hemisphere.capitalize()} Hemisphere - Confusion Matrix'
    )
    
    plot_per_region_accuracy(
        per_region_metrics=per_region_metrics,
        save_path=output_dir / 'per_region_accuracy.png',
        title=f'{hemisphere.capitalize()} Hemisphere - Per-Region Accuracy'
    )
    
    plot_network_accuracy(
        network_metrics=network_metrics,
        save_path=output_dir / 'network_accuracy.png',
        title=f'{hemisphere.capitalize()} Hemisphere - Network-Level Accuracy'
    )
    
    logger.info(f"All results saved to: {output_dir}")
    
    # ==========================================================================
    # TASK TESTING (IF ENABLED)
    # ==========================================================================
    
    task_results = None
    if args.test_on_task:
        task_results = test_on_task_data(
            hemisphere=hemisphere,
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            diagonal_strategy=args.diagonal_strategy,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            sample=args.sample,
            logger=logger
        )
    
    # Return results
    results = {
        'hemisphere': hemisphere,
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
        'output_dir': output_dir,
        'task_results': task_results
    }
    
    return results


def compare_hemispheres(left_results, right_results, output_dir, logger):
    """Compare left and right hemisphere results."""
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPARING HEMISPHERES")
    logger.info(f"{'='*80}\n")
    
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    left_acc = left_results['overall_metrics']['accuracy']
    right_acc = right_results['overall_metrics']['accuracy']
    
    logger.info(f"Left: {left_acc:.4f}, Right: {right_acc:.4f}, Diff: {abs(left_acc - right_acc):.4f}")
    
    # Statistical test
    from scipy.stats import ttest_rel
    left_accs = [m['accuracy'] for m in left_results['fold_metrics']]
    right_accs = [m['accuracy'] for m in right_results['fold_metrics']]
    t_stat, p_value = ttest_rel(left_accs, right_accs)
    
    logger.info(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    # Save summary
    summary = {
        'left_accuracy': left_acc,
        'right_accuracy': right_acc,
        'difference': abs(left_acc - right_acc),
        'ttest_p_value': float(p_value)
    }
    
    # Add task comparison if available
    if left_results.get('task_results') and right_results.get('task_results'):
        left_task_acc = left_results['task_results']['task_summary']['task_test_accuracy']
        right_task_acc = right_results['task_results']['task_summary']['task_test_accuracy']
        
        summary['left_task_accuracy'] = left_task_acc
        summary['right_task_accuracy'] = right_task_acc
        summary['left_accuracy_drop'] = left_results['task_results']['task_summary']['accuracy_drop']
        summary['right_accuracy_drop'] = right_results['task_results']['task_summary']['accuracy_drop']
        
        logger.info(f"\nTask Testing:")
        logger.info(f"  Left task accuracy: {left_task_acc:.4f}")
        logger.info(f"  Right task accuracy: {right_task_acc:.4f}")
        logger.info(f"  Left accuracy drop: {summary['left_accuracy_drop']:.4f}")
        logger.info(f"  Right accuracy drop: {summary['right_accuracy_drop']:.4f}")
    
    with open(comparison_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Comparison saved to: {comparison_dir}")


def main():
    """Main function."""
    
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output_dir, args.hemisphere)
    
    logger.info("="*80)
    logger.info("HEMISPHERE-SPECIFIC CLASSIFICATION")
    logger.info("Preprocessing: Diagonal imputation + Fisher Z → StandardScaler")
    logger.info("Optuna: Runs on scaled full dataset")
    logger.info("CV: Independent scaling per fold")
    if args.test_on_task:
        logger.info("Task Testing: Enabled (train on rest, test on task)")
    logger.info("="*80)
    
    if args.sample:
        logger.warning(f"\n⚠️  TESTING MODE: {args.sample} subjects only\n")
    
    try:
        if args.hemisphere == 'both':
            left_results = train_single_hemisphere('left', args, logger)
            right_results = train_single_hemisphere('right', args, logger)
            compare_hemispheres(left_results, right_results, args.output_dir, logger)
        else:
            results = train_single_hemisphere(args.hemisphere, args, logger)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()