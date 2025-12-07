#!/usr/bin/env python3
"""
Tune regularization parameter alpha for precision diagonal imputation strategy
"""

import sys
import argparse
import numpy as np 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_connectivity_data, extract_connection_columns
from src.features import BrainConnectivityPreprocessor, extract_regions
from src.utils import load_config, set_random_seeds

def tune_precision_alpha(alpha_value):
    """Test a single alpha value using direct preprocessing."""
    # Load config
    config = load_config('configs/config.yaml')
    set_random_seeds(config.get('random_seed', 42))

    # Load data
    df_train = load_connectivity_data(config['data']['piop2_file'])
    connection_columns = extract_connection_columns(df_train)
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)

    print(f"\n{'='*60}")
    print(f"Testing alpha={alpha_value}")
    print(f"{'='*60}")
    print(f"Subjects: {len(df_train)}")
    print(f"Regions: {n_regions}")
    print(f"{'='*60}\n")
    
    # Load model with best hyperparameters
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    model = LogisticRegression(
        C=0.0343304473310619,
        max_iter=1000,
        solver='saga',
        multi_class='multinomial'
    )

    # Create preprocessor with custom alpha
    preprocessor = BrainConnectivityPreprocessor(
        connection_columns=connection_columns,
        diagonal_strategy='diagonal_precision',
        region_list=region_list,
        include_diagonal=True,
        apply_fisher_z=True,
        random_state=42,
        precision_alpha=alpha_value  # Set custom alpha
    )

    # Manual cross-validation with leak-free preprocessing
    n_splits = 3
    subject_ids = df_train.iloc[:, 0].values
    
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    
    print(f"Starting {n_splits}-fold cross-validation...\n")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=subject_ids), 1):
        print(f"Fold {fold}/{n_splits}:")
        
        # Split data
        df_train_fold = df_train.iloc[train_idx].copy()
        df_val_fold = df_train.iloc[val_idx].copy()
        
        # Fit preprocessor on training fold only
        preprocessor_fold = BrainConnectivityPreprocessor(
            connection_columns=connection_columns,
            diagonal_strategy='diagonal_precision',
            region_list=region_list,
            include_diagonal=True,
            apply_fisher_z=True,
            random_state=42,
            precision_alpha=alpha_value  # Use same alpha
        )
        
        preprocessor_fold.fit(df_train_fold)
        
        # Transform both sets
        X_train = preprocessor_fold.transform(df_train_fold)
        y_train = preprocessor_fold.get_labels()
        
        X_val = preprocessor_fold.transform(df_val_fold)
        y_val = preprocessor_fold.get_labels()
        
        # Create pipeline with scaler and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=0.0343304473310619,
                max_iter=1000,
                solver='saga',
                multi_class='multinomial'
            ))
        ])
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        fold_results.append({
            'fold': fold,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy:   {val_acc:.4f}\n")
    
    # Aggregate results
    train_accs = [r['train_acc'] for r in fold_results]
    val_accs = [r['val_acc'] for r in fold_results]
    
    result = {
        'alpha': alpha_value,
        'val_mean': float(np.mean(val_accs)),
        'val_std': float(np.std(val_accs)),
        'train_mean': float(np.mean(train_accs)),
        'train_std': float(np.std(train_accs)),
        'fold_results': fold_results
    }
    
    print(f"{'='*60}")
    print(f"RESULTS FOR ALPHA = {alpha_value}")
    print(f"{'='*60}")
    print(f"Validation: {result['val_mean']:.4f} ± {result['val_std']:.4f}")
    print(f"Training:   {result['train_mean']:.4f} ± {result['train_std']:.4f}")
    print(f"{'='*60}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Tune alpha for precision diagonal imputation'
    )
    parser.add_argument('--alpha', type=float, required=True,
                       help='Alpha value to test (e.g., 0.1)')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number (for parallel execution tracking)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"PRECISION ALPHA TUNING - Job #{args.fold}")
    print(f"{'='*70}")
    print(f"Alpha value: {args.alpha}")
    print(f"Timestamp: {__import__('datetime').datetime.now()}")
    print(f"{'='*70}\n")
    
    # Test the single alpha value
    result = tune_precision_alpha(args.alpha)
    
    # Save result to file
    output_dir = Path(f'results/alpha_tuning/alpha_{str(args.alpha).replace(".", "_")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    result_file = output_dir / 'result.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved results to: {result_file}")
    
    # Also save as CSV for easy aggregation
    import pandas as pd
    summary_df = pd.DataFrame([{
        'alpha': result['alpha'],
        'val_mean': result['val_mean'],
        'val_std': result['val_std'],
        'train_mean': result['train_mean'],
        'train_std': result['train_std']
    }])
    csv_file = output_dir / 'summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"✓ Saved summary to: {csv_file}")
    
    print(f"\n{'='*70}")
    print("JOB COMPLETE")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())