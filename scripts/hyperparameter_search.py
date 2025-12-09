#!/usr/bin/env python3
"""
Nested Cross-Validation for Hyperparameter Optimization
========================================================

Usage:
    # Test locally with sample data
    python src/hyperparameter_search.py --sample --n-jobs 4
    
    # Full search locally
    python src/hyperparameter_search.py --n-jobs 4
    
    # Single iteration for HTCondor
    python src/hyperparameter_search.py --iteration 1 --n-jobs 4
"""

import sys
import numpy as np
import yaml
import json
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_score
from scipy.stats import loguniform, uniform
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import BrainConnectivityPreprocessor
from src.data import load_connectivity_data, extract_connection_columns


def load_search_space(config_path):
    """
    Load hyperparameter search space from YAML configuration.
    
    Args:
        config_path: Path to search space YAML file
        
    Returns:
        param_dist: Dictionary of parameter distributions for sklearn
        config: Full configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    param_dist = {}
    
    for param, spec in config['param_distributions'].items():
        # CRITICAL: Keep the parameter name exactly as specified in YAML
        # Don't strip or modify parameter names - they should already have 'classifier__' prefix
        
        if spec['type'] == 'loguniform':
            # Use scipy's loguniform for log-scale sampling
            param_dist[param] = loguniform(spec['low'], spec['high'])
        elif spec['type'] == 'uniform':
            # Uniform distribution
            param_dist[param] = uniform(spec['low'], spec['high'] - spec['low'])
        elif spec['type'] == 'choice':
            # Discrete choices - keep as list
            param_dist[param] = spec['values']
        else:
            raise ValueError(f"Unknown distribution type: {spec['type']}")
    
    return param_dist, config


def prepare_data(
    df_train: pd.DataFrame,
    preprocessor: BrainConnectivityPreprocessor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for nested cross-validation.
    
    Args:
        df_train: Training DataFrame
        preprocessor: Fitted BrainConnectivityPreprocessor
        
    Returns:
        X: Feature matrix
        y: Labels
        subject_ids_expanded: Expanded subject IDs matching samples
    """
    # Step 1: Fit preprocessor and transform data
    print("Fitting preprocessor...")
    preprocessor.fit(df_train)
    X = preprocessor.transform(df_train)
    y = preprocessor.get_labels()
    
    # Step 2: Get SAMPLE-LEVEL subject IDs (not subject-level!)
    # This is the key fix - get subjects array that matches X and y
    subjects_per_sample = preprocessor.get_subjects()
    
    # Step 3: Map sample indices back to actual subject IDs
    subject_ids_original = df_train.iloc[:, 0].values  # Original subject IDs from DataFrame
    subject_ids_expanded = subject_ids_original[subjects_per_sample]
    
    print(f"\nData shapes:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  subject_ids_expanded: {subject_ids_expanded.shape}")
    print(f"  Unique subjects: {len(np.unique(subject_ids_expanded))}")
    
    # Verify shapes match
    assert X.shape[0] == y.shape[0] == subject_ids_expanded.shape[0], \
        f"Shape mismatch: X={X.shape[0]}, y={y.shape[0]}, subjects={subject_ids_expanded.shape[0]}"
    
    return X, y, subject_ids_expanded


def run_nested_cv(
    df_train: pd.DataFrame,
    preprocessor: BrainConnectivityPreprocessor,
    param_distributions: Dict,
    n_iter: int = 20,
    outer_cv: int = 5,
    inner_cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict:
    """
    Run nested cross-validation for hyperparameter search and unbiased performance evaluation.
    
    Nested CV Structure:
    - Outer loop (5 folds): Provides unbiased performance estimates
    - Inner loop (3 folds): Performs hyperparameter tuning via RandomizedSearchCV
    
    Args:
        df_train: Training DataFrame
        preprocessor: Fitted BrainConnectivityPreprocessor
        param_distributions: Parameter distributions for RandomizedSearchCV
        n_iter: Number of random samples for hyperparameter search
        outer_cv: Number of outer CV folds (default: 5)
        inner_cv: Number of inner CV folds (default: 3)
        random_state: Random seed
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary containing:
            - outer_scores: Test scores from outer CV loop
            - best_params_per_fold: Best hyperparameters found in each outer fold
            - mean_score: Mean outer CV score
            - std_score: Std of outer CV score
    """
    
    # Prepare data
    X, y, subject_ids_expanded = prepare_data(df_train, preprocessor)
    
    # Create outer CV splitter (subject-wise)
    outer_cv_splitter = GroupKFold(n_splits=outer_cv)
    
    # Store results
    outer_scores = []
    best_params_per_fold = []
    all_cv_results = []
    
    print(f"\n{'='*70}")
    print("STARTING NESTED CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Outer CV folds: {outer_cv}")
    print(f"Inner CV folds: {inner_cv}")
    print(f"Hyperparameter search iterations: {n_iter}")
    print(f"{'='*70}\n")
    
    # Outer CV loop
    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv_splitter.split(X, y, groups=subject_ids_expanded), 1
    ):
        print(f"\n{'='*70}")
        print(f"OUTER FOLD {fold_idx}/{outer_cv}")
        print(f"{'='*70}")
        
        # Split data for this outer fold
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        groups_train_outer = subject_ids_expanded[train_idx]
        
        print(f"Train samples: {len(X_train_outer)}, Test samples: {len(X_test_outer)}")
        print(f"Train subjects: {len(np.unique(groups_train_outer))}, "
              f"Test subjects: {len(np.unique(subject_ids_expanded[test_idx]))}")
        
        # Create pipeline for this fold
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            ))
        ])
        
        # Inner CV: Hyperparameter search with GroupKFold
        inner_cv_splitter = GroupKFold(n_splits=inner_cv)
        
        print(f"\nStarting inner CV (hyperparameter search)...")
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_cv_splitter,
            scoring='accuracy',
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=2,
            return_train_score=True
        )
        
        # Fit RandomizedSearchCV on outer training set
        random_search.fit(X_train_outer, y_train_outer, groups=groups_train_outer)
        
        # Get best model from inner CV
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        print(f"\nBest parameters from inner CV:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best inner CV score: {random_search.best_score_:.4f}")
        
        # Evaluate best model on outer test fold
        test_score = best_model.score(X_test_outer, y_test_outer)
        print(f"\n>>> OUTER FOLD {fold_idx} TEST SCORE: {test_score:.4f}")
        
        # Store results
        outer_scores.append(test_score)
        best_params_per_fold.append({
            'fold': fold_idx,
            'params': best_params,
            'inner_cv_score': random_search.best_score_,
            'outer_test_score': test_score
        })
        
        # Store detailed CV results for this fold
        cv_results_df = pd.DataFrame(random_search.cv_results_)
        cv_results_df['outer_fold'] = fold_idx
        all_cv_results.append(cv_results_df)
    
    # Compute final statistics
    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    
    print(f"\n{'='*70}")
    print("NESTED CV COMPLETE")
    print(f"{'='*70}")
    print(f"\nOuter CV Scores per fold:")
    for fold_idx, score in enumerate(outer_scores, 1):
        print(f"  Fold {fold_idx}: {score:.4f}")
    print(f"\nMean Outer CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"{'='*70}\n")
    
    # Combine all CV results
    combined_cv_results = pd.concat(all_cv_results, ignore_index=True)
    
    return {
        'outer_scores': outer_scores,
        'best_params_per_fold': best_params_per_fold,
        'mean_score': mean_score,
        'std_score': std_score,
        'cv_results': combined_cv_results
    }


def train_final_model(
    df_train: pd.DataFrame,
    preprocessor: BrainConnectivityPreprocessor,
    param_distributions: Dict,
    n_iter: int = 20,
    inner_cv: int = 3,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[Pipeline, Dict]:
    """
    After nested CV, train final model on ALL data using hyperparameter search.
    
    Args:
        df_train: Full training DataFrame
        preprocessor: Fitted BrainConnectivityPreprocessor
        param_distributions: Parameter distributions
        n_iter: Number of hyperparameter search iterations
        inner_cv: Number of CV folds for hyperparameter search
        random_state: Random seed
        n_jobs: Number of parallel jobs
        
    Returns:
        best_model: Trained final model
        search_results: Dictionary with search results
    """
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*70}\n")
    
    # Prepare data
    X, y, subject_ids_expanded = prepare_data(df_train, preprocessor)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ))
    ])
    
    # Inner CV for hyperparameter search on ALL data
    inner_cv_splitter = GroupKFold(n_splits=inner_cv)
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=inner_cv_splitter,
        scoring='accuracy',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=2,
        return_train_score=True
    )
    
    # Fit on all data
    random_search.fit(X, y, groups=subject_ids_expanded)
    
    print(f"\nFinal model best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Final model CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': pd.DataFrame(random_search.cv_results_)
    }


def save_results(nested_cv_results, final_model_results, output_dir, model_name, diagonal_strategy):
    """
    Save nested CV results and final model to disk.
    
    Args:
        nested_cv_results: Results from nested CV
        final_model_results: Results from final model training
        output_dir: Directory to save results
        model_name: Name of the model
        diagonal_strategy: Diagonal imputation strategy used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save nested CV summary
    nested_summary = {
        'outer_cv_scores': nested_cv_results['outer_scores'],
        'mean_outer_cv_score': nested_cv_results['mean_score'],
        'std_outer_cv_score': nested_cv_results['std_score'],
        'best_params_per_fold': nested_cv_results['best_params_per_fold'],
        'model': model_name,
        'diagonal_strategy': diagonal_strategy,
        'timestamp': timestamp
    }
    
    summary_path = output_dir / f'nested_cv_summary_{model_name}_{diagonal_strategy}_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(nested_summary, f, indent=2, default=str)
    print(f"\n✓ Saved nested CV summary: {summary_path}")
    
    # Save detailed CV results from all folds
    cv_results_path = output_dir / f'nested_cv_results_{model_name}_{diagonal_strategy}_{timestamp}.csv'
    nested_cv_results['cv_results'].to_csv(cv_results_path, index=False)
    print(f"✓ Saved detailed CV results: {cv_results_path}")
    
    # Save final model results
    final_model_path = output_dir / f'final_model_results_{model_name}_{diagonal_strategy}_{timestamp}.json'
    final_summary = {
        'best_params': final_model_results['best_params'],
        'best_cv_score': final_model_results['best_score'],
        'model': model_name,
        'diagonal_strategy': diagonal_strategy,
        'timestamp': timestamp
    }
    with open(final_model_path, 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    print(f"✓ Saved final model results: {final_model_path}")
    
    # Save final model CV results
    final_cv_path = output_dir / f'final_model_cv_results_{model_name}_{diagonal_strategy}_{timestamp}.csv'
    final_model_results['cv_results'].to_csv(final_cv_path, index=False)
    print(f"✓ Saved final model CV results: {final_cv_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nNested CV Performance:")
    print(f"  Mean Outer CV Score: {nested_cv_results['mean_score']:.4f}")
    print(f"  Std Outer CV Score: {nested_cv_results['std_score']:.4f}")
    print(f"\nFinal Model (trained on all data):")
    print(f"  Best CV Score: {final_model_results['best_score']:.4f}")
    print(f"  Best Parameters:")
    for param, value in final_model_results['best_params'].items():
        print(f"    {param}: {value}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Nested Cross-Validation for Hyperparameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with sample data
  python src/hyperparameter_search.py --sample 
        """
    )
    
    parser.add_argument('--data', type=str, 
                       default='data/raw/PIOP2_restingstate.csv',
                       help='Path to training data CSV')
    parser.add_argument('--model', type=str, 
                       default='logistic_regression',
                       help='Model name (for output naming)')
    parser.add_argument('--search-config', type=str, 
                       default='configs/hyperparameters/logistic_regression_search.yaml',
                       help='Path to search space YAML')
    parser.add_argument('--output', type=str, 
                       default='results/hyperparameter_search',
                       help='Output directory for results')
    parser.add_argument('--diagonal', type=str, 
                       default='zero',
                       choices=['zero', 'random', 'region_mean', 'network_mean', 'sample_matrix','sample_row'],
                       help='Diagonal imputation strategy')
    parser.add_argument('--n-iter', type=int,
                       help='Number of random samples to try (overrides config)')
    parser.add_argument('--outer-cv', type=int, default=5,
                       help='Number of outer CV folds (default: 5)')
    parser.add_argument('--inner-cv', type=int, default=3,
                       help='Number of inner CV folds (default: 3)')
    parser.add_argument('--n-jobs', type=int, 
                       default=-1,
                       help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--seed', type=int,
                       help='Random seed (overrides config)')
    parser.add_argument('--sample', action='store_true',
                       help='Run on sample data (30 subjects)')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Specific iteration for parallel execution on HTCondor')
    
    # Parse args once to get config file path
    args_preliminary = parser.parse_args()
    
    # Load config and set as defaults
    param_distributions, search_config = load_search_space(args_preliminary.search_config)
    
    # Map config keys to argparse argument names
    config_defaults = {
        'n_iter': search_config.get('n_iter'),
        'seed': search_config.get('random_seed')
    }
    
    # Set config values as defaults (only for non-None values)
    parser.set_defaults(**{k: v for k, v in config_defaults.items() if v is not None})
    
    # Command-line args will override config values
    args = parser.parse_args()
    
    # Adjust output directory if iteration specified
    if args.iteration is not None:
        args.output = f"{args.output}/iteration_{args.iteration:03d}"
    
    print(f"{'='*70}")
    print("NESTED CROSS-VALIDATION - HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Diagonal: {args.diagonal}")
    print(f"Outer CV: {args.outer_cv}")
    print(f"Inner CV: {args.inner_cv}")
    print(f"Iterations per search: {args.n_iter}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"Loading data from: {args.data}")
    df = load_connectivity_data(args.data)
    
    if args.sample:
        print("Using SAMPLE mode (30 subjects)")
        df = df.head(30)
    
    print(f"Loaded {len(df)} subjects\n")
    
    # Load search space
    print(f"Loading search space from: {args.search_config}")
    param_distributions, search_config = load_search_space(args.search_config)
    
    print("Parameter distributions:")
    for param, dist in param_distributions.items():
        print(f"  {param}: {dist}")
    print()
    
    # Extract connection columns
    connection_columns = extract_connection_columns(df)
    print(f"Connection columns: {len(connection_columns)}\n")
    
    # Create preprocessor
    preprocessor = BrainConnectivityPreprocessor(
        connection_columns=connection_columns,
        diagonal_strategy=args.diagonal,
        include_diagonal=True,
        apply_fisher_z=True,
        random_state=args.seed
    )
    
    try:
        # Run nested CV
        nested_cv_results = run_nested_cv(
            df_train=df,
            preprocessor=preprocessor,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            outer_cv=args.outer_cv,
            inner_cv=args.inner_cv,
            random_state=args.seed,
            n_jobs=args.n_jobs
        )
        
        # Train final model on all data
        final_model, final_model_results = train_final_model(
            df_train=df,
            preprocessor=preprocessor,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            inner_cv=args.inner_cv,
            random_state=args.seed,
            n_jobs=args.n_jobs
        )
        
        # Save all results
        save_results(
            nested_cv_results=nested_cv_results,
            final_model_results=final_model_results,
            output_dir=args.output,
            model_name=args.model,
            diagonal_strategy=args.diagonal
        )
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {e}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())