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
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
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
        help='Enable hyperparameter tuning using settings from config file'
    )
    
    parser.add_argument(
        '--tune_method',
        type=str,
        default='grid',
        choices=['grid', 'random', 'optuna'],
        help='Hyperparameter tuning method: grid (GridSearchCV), random (RandomizedSearchCV), or optuna (Optuna TPE)'
    )
    
    parser.add_argument(
        '--optuna_trials',
        type=int,
        default=50,
        help='Number of Optuna trials (only used if tune_method=optuna)'
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
        # Return default configuration
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
    """
    Sample first n subjects for testing (deterministic selection).
    
    Parameters
    ----------
    data : dict
        Dictionary containing connectivity, subject_ids, region_info, etc.
    n_sample : int
        Number of subjects to sample
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    sampled_data : dict
        Dictionary with sampled data (first n subjects)
    """
    
    total_subjects = data['n_subjects']
    
    # Validate sample size
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
    
    # Sample connectivity matrices (first n subjects)
    connectivity_sampled = data['connectivity'][:n_sample]
    subject_ids_sampled = data['subject_ids'][:n_sample]
    
    # Log which subjects were selected
    logger.info(f"Selected subjects: {', '.join(map(str, subject_ids_sampled[:10]))}" + 
                (f"... (+{n_sample-10} more)" if n_sample > 10 else ""))
    
    # Create sampled data dictionary
    sampled_data = {
        'connectivity': connectivity_sampled,
        'subject_ids': subject_ids_sampled,
        'region_info': data['region_info'],  # Same for all subjects
        'hemisphere': data['hemisphere'],
        'n_subjects': n_sample,  # Update subject count
        'n_regions': data['n_regions']
    }
    
    logger.info(f"Sampled connectivity shape: {connectivity_sampled.shape}")
    logger.info(f"Sampled subjects: {len(subject_ids_sampled)}")
    logger.info(f"{'='*60}\n")
    
    return sampled_data


def determine_best_hyperparameters(fold_metrics: list, logger: logging.Logger) -> dict:
    """
    Determine the best overall hyperparameters from cross-validation folds.
    
    Selects parameters from the fold with highest test accuracy.
    Also provides statistics on parameter consistency across folds.
    
    Parameters
    ----------
    fold_metrics : list
        List of dictionaries containing metrics for each fold
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    best_params_summary : dict
        Dictionary with best parameters and selection statistics
    """
    
    # Check if hyperparameter tuning was used
    if 'best_params' not in fold_metrics[0]:
        logger.info("\nNo hyperparameter tuning was performed (using fixed parameters)")
        return None
    
    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER TUNING SUMMARY")
    logger.info("="*80)
    
    # Check tuning method
    tuning_method = fold_metrics[0].get('tuning_method', 'unknown')
    logger.info(f"\nTuning method used: {tuning_method.upper()}")
    
    # Extract parameters from each fold
    fold_params = []
    fold_accs = []
    
    for fold_metric in fold_metrics:
        fold_params.append(fold_metric['best_params'])
        fold_accs.append(fold_metric['accuracy'])
    
    # Find fold with best test accuracy
    best_fold_idx = np.argmax(fold_accs)
    best_params = fold_params[best_fold_idx]
    best_acc = fold_accs[best_fold_idx]
    
    logger.info(f"\n✓ Best parameters selected from Fold {best_fold_idx + 1} (accuracy: {best_acc:.4f}):")
    for param, value in best_params.items():
        logger.info(f"  • {param}: {value}")
    
    # Analyze parameter consistency across folds
    logger.info("\nParameter consistency across folds:")
    
    # Count occurrences of each parameter value
    from collections import Counter
    param_names = list(best_params.keys())
    
    for param_name in param_names:
        values = [fold_param[param_name] for fold_param in fold_params]
        value_counts = Counter(values)
        
        logger.info(f"\n  {param_name}:")
        for value, count in value_counts.most_common():
            percentage = (count / len(fold_params)) * 100
            logger.info(f"    {value}: {count}/{len(fold_params)} folds ({percentage:.1f}%)")
    
    # Create summary dictionary
    best_params_summary = {
        'best_parameters': best_params,
        'selected_from_fold': int(best_fold_idx + 1),
        'fold_accuracy': float(best_acc),
        'parameter_frequencies': {}
    }
    
    # Add frequency information for each parameter
    for param_name in param_names:
        values = [fold_param[param_name] for fold_param in fold_params]
        value_counts = Counter(values)
        best_params_summary['parameter_frequencies'][param_name] = {
            str(value): count for value, count in value_counts.items()
        }
    
    # Show fold-by-fold comparison
    logger.info("\n" + "-"*80)
    logger.info("Fold-by-fold hyperparameter comparison:")
    logger.info("-"*80)
    
    # Create header
    header = "Fold | " + " | ".join(f"{p:>10}" for p in param_names) + " | Accuracy"
    logger.info(header)
    logger.info("-" * len(header))
    
    for i, (params, acc) in enumerate(zip(fold_params, fold_accs)):
        param_str = " | ".join(f"{str(params[p]):>10}" for p in param_names)
        marker = " ← BEST" if i == best_fold_idx else ""
        logger.info(f"{i+1:4d} | {param_str} | {acc:.4f}{marker}")
    
    logger.info("="*80 + "\n")
    
    return best_params_summary


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
    
    NOTE: This function now expects 2D feature matrices (n_samples, n_features)
    not 3D connectivity matrices. The diagonal imputation should have been
    done before reshaping if needed, or we skip it here.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (n_train_samples, n_features)
    X_test : np.ndarray
        Test features (n_test_samples, n_features)
    diagonal_strategy : str
        Strategy for diagonal imputation (NOTE: may not be applicable here)
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
    
    logger.info(f"  Preprocessing fold data...")
    logger.info(f"  Input shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Apply Fisher Z-transformation if not already done
    # (typically should be done on full matrices before reshaping)
    X_train_processed = np.arctanh(np.clip(X_train, -0.999, 0.999))
    X_test_processed = np.arctanh(np.clip(X_test, -0.999, 0.999))
    
    # Standardization - fit on training data only
    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train_processed)
    X_test_processed = scaler.transform(X_test_processed)
    
    # Validate no NaN/Inf
    if np.any(np.isnan(X_train_processed)) or np.any(np.isinf(X_train_processed)):
        raise ValueError("NaN or Inf detected in training data after preprocessing")
    if np.any(np.isnan(X_test_processed)) or np.any(np.isinf(X_test_processed)):
        raise ValueError("NaN or Inf detected in test data after preprocessing")
    
    logger.info(f"  Processed shapes - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed


def create_optuna_objective(X_train, y_train, groups_train, inner_cv, random_state, logger):
    """
    Create Optuna objective function for hyperparameter optimization.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    groups_train : np.ndarray
        Training groups (subject IDs)
    inner_cv : GroupKFold
        Cross-validation splitter
    random_state : int
        Random state for reproducibility
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    objective : callable
        Optuna objective function
    """
    
    def objective(trial):
        """Optuna objective function."""
        
        # Suggest hyperparameters
        C = trial.suggest_float('C', 0.001, 10.0, log=True)
        max_iter = trial.suggest_categorical('max_iter', [200, 300, 500, 1000])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
        
        # Create model with suggested parameters
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            n_jobs=1,
            verbose=0
        )
        
        # Cross-validation on inner folds
        from sklearn.metrics import accuracy_score
        scores = []
        
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train, groups=groups_train):
            X_inner_train = X_train[inner_train_idx]
            X_inner_val = X_train[inner_val_idx]
            y_inner_train = y_train[inner_train_idx]
            y_inner_val = y_train[inner_val_idx]
            
            model.fit(X_inner_train, y_inner_train)
            y_inner_pred = model.predict(X_inner_val)
            score = accuracy_score(y_inner_val, y_inner_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    return objective


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
    
    # Load data - FIXED: Use load_hemisphere_data instead of load_hemisphere_data_from_csv
    logger.info("Loading hemisphere-specific data...")
    data = load_hemisphere_data(
        data_dir=args.data_dir,
        hemisphere=hemisphere,
        dataset='rest',  # or 'task' depending on your needs
        return_matrix=True,
        validate=True
    )
    
    # SAMPLE FIRST N SUBJECTS IF SPECIFIED (TESTING MODE)
    if args.sample is not None:
        data = sample_first_n_subjects(
            data=data,
            n_sample=args.sample,
            logger=logger
        )
        logger.warning(f"⚠️  TESTING MODE: Using first {args.sample} subjects only")
        logger.warning(f"⚠️  Results are for TESTING purposes - not production-ready!")
    
    connectivity = data['connectivity']  # (n_subjects, n_regions, n_regions)
    subject_ids = data['subject_ids']  # (n_subjects,)
    region_info = data['region_info']  # DataFrame with region metadata
    
    n_subjects, n_regions, _ = connectivity.shape
    
    logger.info(f"Data loaded:")
    logger.info(f"  Subjects: {n_subjects}")
    logger.info(f"  Regions: {n_regions}")
    logger.info(f"  Connectivity shape: {connectivity.shape}")
    
    # FIXED: Prepare classification data - creates samples and labels
    logger.info("\nPreparing classification data...")
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
    logger.info(f"  Groups: {groups.shape}")
    logger.info(f"  Classes: {n_classes}")
    logger.info(f"  Unique subjects in groups: {len(np.unique(groups))}")
    
    # Validate data - FIXED: Check correct shapes
    assert X.shape[0] == len(y), "Mismatch between X and y"
    assert X.shape[0] == len(groups), "Mismatch between X and groups"
    assert np.all(y >= 0) and np.all(y < n_classes), "Invalid label values"
    assert len(np.unique(groups)) == n_subjects, "Groups should have one entry per subject"
    
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
    
    # FIXED: Split using prepared data (X, y, groups)
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        fold_start = time.time()
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # Split data - FIXED: Now splitting sample-level data
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
        
        # Verify no subject leakage - FIXED: Check groups (subject IDs)
        train_subjects = set(groups_train)
        test_subjects = set(groups_test)
        assert len(train_subjects.intersection(test_subjects)) == 0, "Subject leakage detected!"
        
        logger.info(f"  Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
        logger.info(f"  Train labels distribution (first 10): {np.bincount(y_train)[:10].tolist()}")
        logger.info(f"  Test labels distribution (first 10): {np.bincount(y_test)[:10].tolist()}")
        
        # Preprocess within fold (LEAK-FREE) - FIXED: Now receives 2D features
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
        
        # Determine tuning method
        config = load_config(args.config_file)
        tune_hyperparams = args.tune_hyperparams and config.get('hyperparameter_optimization', {}).get('enabled', False)
        
        if tune_hyperparams:
            # Hyperparameter tuning enabled
            tune_method = args.tune_method
            logger.info(f"  Hyperparameter tuning ENABLED (method: {tune_method})")
            
            # Inner CV for hyperparameter tuning
            hyperparam_config = config.get('hyperparameter_optimization', {})
            inner_cv = GroupKFold(n_splits=hyperparam_config.get('cv_folds', 3))
            inner_groups = groups_train
            
            if tune_method == 'optuna':
                # OPTUNA OPTIMIZATION
                logger.info(f"  Running Optuna optimization ({args.optuna_trials} trials)...")
                
                # Create Optuna study
                study = optuna.create_study(
                    direction='maximize',
                    sampler=TPESampler(seed=args.random_state)
                )
                
                # Create objective function
                objective = create_optuna_objective(
                    X_train_processed, y_train, inner_groups,
                    inner_cv, args.random_state, logger
                )
                
                # Optimize
                study.optimize(
                    objective,
                    n_trials=args.optuna_trials,
                    show_progress_bar=False,
                    n_jobs=1  # Sequential for GroupKFold
                )
                
                # Get best parameters
                best_params = study.best_params
                best_score = study.best_value
                
                logger.info(f"  Best parameters: {best_params}")
                logger.info(f"  Best inner CV score: {best_score:.4f}")
                
                # Train final model with best parameters
                model = LogisticRegression(
                    C=best_params['C'],
                    max_iter=best_params['max_iter'],
                    solver=best_params['solver'],
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    verbose=0
                )
                model.fit(X_train_processed, y_train)
                
                # Store for metrics
                search_best_params = best_params
                search_best_score = best_score
                
            else:
                # GRID/RANDOM SEARCH
                hyperparam_config = config['hyperparameter_optimization']
                logger.info(f"  Method: {hyperparam_config['method']}")
                logger.info(f"  Param grid: {hyperparam_config['param_grid']}")
                
                base_model = LogisticRegression(
                    solver='lbfgs',
                    random_state=args.random_state,
                    n_jobs=1,
                    verbose=0
                )
                
                if hyperparam_config['method'] == 'GridSearchCV':
                    search = GridSearchCV(
                        estimator=base_model,
                        param_grid=hyperparam_config['param_grid'],
                        cv=inner_cv,
                        scoring='accuracy',
                        n_jobs=hyperparam_config.get('n_jobs', -1),
                        verbose=hyperparam_config.get('verbose', 1),
                        refit=True
                    )
                else:  # RandomizedSearchCV
                    search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=hyperparam_config['param_grid'],
                        n_iter=hyperparam_config.get('n_iter', 20),
                        cv=inner_cv,
                        scoring='accuracy',
                        n_jobs=hyperparam_config.get('n_jobs', -1),
                        verbose=hyperparam_config.get('verbose', 1),
                        random_state=args.random_state,
                        refit=True
                    )
                
                logger.info(f"  Running {hyperparam_config['method']}...")
                search.fit(X_train_processed, y_train, groups=inner_groups)
                
                model = search.best_estimator_
                search_best_params = search.best_params_
                search_best_score = search.best_score_
                
                logger.info(f"  Best parameters: {search_best_params}")
                logger.info(f"  Best inner CV score: {search_best_score:.4f}")
        
        else:
            # No hyperparameter tuning
            C = args.regularization_C if args.regularization_C is not None else 1.0
            logger.info(f"  Using fixed regularization C={C}")
            
            model = LogisticRegression(
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
        
        fold_metric_dict = {
            'fold': fold_idx + 1,
            'accuracy': fold_acc,
            'balanced_accuracy': fold_bal_acc,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_train_subjects': len(train_subjects),
            'n_test_subjects': len(test_subjects)
        }
        
        # Add hyperparameter info if tuning was performed
        if tune_hyperparams:
            fold_metric_dict['best_params'] = search_best_params
            fold_metric_dict['best_inner_cv_score'] = float(search_best_score)
            fold_metric_dict['tuning_method'] = args.tune_method
        
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
                'test_idx': test_idx,
                'train_subjects': list(train_subjects),
                'test_subjects': list(test_subjects)
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
    
    # Determine best hyperparameters if tuning was performed
    best_params_summary = determine_best_hyperparameters(fold_metrics, logger)
    
    # Add best params to overall metrics if available
    if best_params_summary is not None:
        overall_metrics['best_hyperparameters'] = best_params_summary
    
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


def determine_best_hyperparameters(fold_metrics: list, logger: logging.Logger) -> dict:
    """
    Determine the best overall hyperparameters from cross-validation folds.
    
    Selects parameters from the fold with highest test accuracy.
    Also provides statistics on parameter consistency across folds.
    
    Parameters
    ----------
    fold_metrics : list
        List of dictionaries containing metrics for each fold
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    best_params_summary : dict
        Dictionary with best parameters and selection statistics
    """
    
    # Check if hyperparameter tuning was used
    if 'best_params' not in fold_metrics[0]:
        logger.info("\nNo hyperparameter tuning was performed")
        return None
    
    logger.info("\n" + "="*60)
    logger.info("HYPERPARAMETER TUNING SUMMARY")
    logger.info("="*60)
    
    # Extract parameters from each fold
    fold_params = []
    fold_accs = []
    
    for fold_metric in fold_metrics:
        fold_params.append(fold_metric['best_params'])
        fold_accs.append(fold_metric['accuracy'])
    
    # Find fold with best test accuracy
    best_fold_idx = np.argmax(fold_accs)
    best_params = fold_params[best_fold_idx]
    best_acc = fold_accs[best_fold_idx]
    
    logger.info(f"\nBest parameters selected from Fold {best_fold_idx + 1} (accuracy: {best_acc:.4f}):")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    
    # Analyze parameter consistency across folds
    logger.info("\nParameter consistency across folds:")
    
    # Count occurrences of each parameter value
    from collections import Counter
    param_names = list(best_params.keys())
    
    for param_name in param_names:
        values = [fold_param[param_name] for fold_param in fold_params]
        value_counts = Counter(values)
        
        logger.info(f"\n  {param_name}:")
        for value, count in value_counts.most_common():
            logger.info(f"    {value}: {count}/{len(fold_params)} folds")
    
    # Create summary dictionary
    best_params_summary = {
        'best_parameters': best_params,
        'selected_from_fold': int(best_fold_idx + 1),
        'fold_accuracy': float(best_acc),
        'parameter_frequencies': {}
    }
    
    # Add frequency information for each parameter
    for param_name in param_names:
        values = [fold_param[param_name] for fold_param in fold_params]
        value_counts = Counter(values)
        best_params_summary['parameter_frequencies'][param_name] = {
            str(value): count for value, count in value_counts.items()
        }
    
    # Show fold-by-fold comparison
    logger.info("\nFold-by-fold hyperparameter comparison:")
    logger.info("Fold | " + " | ".join(param_names) + " | Accuracy")
    logger.info("-" * 80)
    
    for i, (params, acc) in enumerate(zip(fold_params, fold_accs)):
        param_str = " | ".join(str(params[p]) for p in param_names)
        logger.info(f"{i+1:4d} | {param_str} | {acc:.4f}")
    
    logger.info("="*60 + "\n")
    
    return best_params_summary


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
    logger.info(f"  Sample size: {args.sample if args.sample else 'All subjects (full dataset)'}")
    logger.info(f"  Hyperparameter tuning: {args.tune_hyperparams}")
    
    if args.tune_hyperparams:
        logger.info(f"  Tuning method: {args.tune_method}")
        if args.tune_method == 'optuna':
            logger.info(f"  Optuna trials: {args.optuna_trials}")
        
        config = load_config(args.config_file)
        if config.get('hyperparameter_optimization', {}).get('enabled', False):
            hyperparam_config = config['hyperparameter_optimization']
            logger.info(f"  Tuning method: {hyperparam_config.get('method', 'GridSearchCV')}")
            logger.info(f"  Inner CV folds: {hyperparam_config.get('cv_folds', 3)}")
            logger.info(f"  Param grid: {hyperparam_config.get('param_grid', {})}")
        else:
            logger.warning("  Hyperparameter tuning requested but not enabled in config!")
    
    if args.sample is not None:
        logger.warning(f"\n⚠️  TESTING MODE ENABLED: Using only first {args.sample} subjects")
        logger.warning(f"⚠️  This is NOT a full production run!\n")
    
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
        if args.sample is not None:
            logger.info(f"(TESTING MODE: Used {args.sample} subjects only)")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()