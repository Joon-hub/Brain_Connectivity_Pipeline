#!/usr/bin/env python3
"""
Random Search for Hyperparameter Optimization using Sklearn RandomizedSearchCV
==============================================================================

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
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import loguniform, uniform
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Dict

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


def run_random_search(
    df_train: pd.DataFrame,
    preprocessor: BrainConnectivityPreprocessor,
    param_distributions: Dict,
    n_iter: int = 20,
    n_splits: int = 3,
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomizedSearchCV:
    """
    Run randomized hyperparameter search with proper subject-wise cross-validation.
    
    CRITICAL FIX: Expand subject_ids to match per-region samples.
    
    Args:
        df_train: Training DataFrame
        preprocessor: Fitted BrainConnectivityPreprocessor
        param_distributions: Parameter distributions for RandomizedSearchCV
        n_iter: Number of random samples
        n_splits: Number of CV folds
        random_state: Random seed
        n_jobs: Number of parallel jobs
        
    Returns:
        Fitted RandomizedSearchCV object
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
    
    print(f"\nData shapes (after fix):")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  subject_ids_expanded: {subject_ids_expanded.shape}")
    print(f"  Unique subjects: {len(np.unique(subject_ids_expanded))}")
    
    # Verify shapes match
    assert X.shape[0] == y.shape[0] == subject_ids_expanded.shape[0], \
        f"Shape mismatch: X={X.shape[0]}, y={y.shape[0]}, subjects={subject_ids_expanded.shape[0]}"
    
    # Step 4: Create pipeline with scaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ))
    ])
    
    # Step 5: Setup GroupKFold with EXPANDED subject IDs
    cv = GroupKFold(n_splits=n_splits)
    
    # Step 6: Run RandomizedSearchCV
    print(f"\nStarting RandomizedSearchCV:")
    print(f"  Iterations: {n_iter}")
    print(f"  CV folds: {n_splits}")
    print(f"  Parallel jobs: {n_jobs}")
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=2,
        return_train_score=True
    )
    
    # Step 7: Fit with EXPANDED subject IDs for grouping
    print("\nFitting RandomizedSearchCV...")
    random_search.fit(X, y, groups=subject_ids_expanded)
    
    return random_search


def save_results(random_search, output_dir, model_name, diagonal_strategy):
    """
    Save hyperparameter search results to disk.
    
    Args:
        random_search: Fitted RandomizedSearchCV object
        output_dir: Directory to save results
        model_name: Name of the model
        diagonal_strategy: Diagonal imputation strategy used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results DataFrame
    results_df = pd.DataFrame(random_search.cv_results_)
    results_path = output_dir / f'cv_results_{model_name}_{diagonal_strategy}_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved CV results: {results_path}")
    
    # Save best parameters
    best_params = {
        'best_params': random_search.best_params_,
        'best_score': float(random_search.best_score_),
        'best_index': int(random_search.best_index_),
        'model': model_name,
        'diagonal_strategy': diagonal_strategy,
        'timestamp': timestamp
    }
    best_params_path = output_dir / f'best_params_{model_name}_{diagonal_strategy}_{timestamp}.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"✓ Saved best params: {best_params_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV Score: {random_search.best_score_:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Random Search Hyperparameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with sample data
  python src/hyperparameter_search.py --sample --n-jobs 4
  
  # Full search locally
  python src/hyperparameter_search.py --n-jobs 4
  
  # Single iteration for HTCondor
  python src/hyperparameter_search.py --iteration 1 --n-jobs 4
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
    parser.add_argument('--n-splits', type=int,
                       help='Number of CV folds (overrides config)')
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
        'n_splits': search_config.get('cv_folds'),
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
    print("HYPERPARAMETER SEARCH - RANDOM SEARCH")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Diagonal: {args.diagonal}")
    print(f"Iterations: {args.n_iter}")
    print(f"CV Folds: {args.n_splits}")
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
    
    # Print loaded parameters
    print(f"Loaded parameters: {list(param_distributions.keys())}")
    
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
    
    # Run search
    try:
        random_search = run_random_search(
            df_train=df,
            preprocessor=preprocessor,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            n_splits=args.n_splits,
            random_state=args.seed,
            n_jobs=args.n_jobs
        )
        
        # Save results
        save_results(random_search, args.output, args.model, args.diagonal)
        
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