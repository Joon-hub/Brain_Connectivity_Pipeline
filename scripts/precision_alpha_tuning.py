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
from src.brain_region_classifier import BrainRegionClassifier
from src.utils import load_config, set_random_seeds

# 
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--fold', type=int, required=True)
args = parser.parse_args()

def tune_precision_alpha():
    # Load config
    config = load_config('configs/config.yaml')
    set_random_seeds(config.get('random_seed', 42))

    # Load data
    df_train = load_connectivity_data(config['data']['piop2_file'])
    connection_columns = extract_connection_columns(df_train)
    region_list, _, _ = extract_regions(connection_columns)

    # alpha values to test
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]

    results = []

    for alpha in alphas:
        print(f"Testing alpha={alpha}")
        
        # Load model -> apply best hyperparameters of logistic regression model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C = 0.0343304473310619,
            max_iter= 1000,
            solver = 'saga',
            multi_class = 'multinomial'
        )

        # Modify BrainConnectivityPreprocessor to accept alpha
        preprocess_params = {
            'connection_columns': connection_columns,
            'diagonal_strategy': 'daigonal_precision',
            'region_list': region_list,
            'include_diagonal': True,
            'apply_fisher_z': True,
            'random_state': 42,
            'precision_alpha': alpha           # new parameter
        }


        # Test on small dataset
        # sample_df = df_train.head(30)

        # Quick Validation
        classifier = BrainRegionClassifier(
            preprocessor_class=BrainConnectivityPreprocessor,
            model_instance=model,
            diagonal_strategy="daigonal_precision",
            connection_columns=connection_columns,
            n_splits=3,
            random_state=42
        )

        classifier.fit(df_train, verbose=True)
        cv_results = classifier.get_cv_results()

        results.append({
            'alpha': alpha,
            'val_mean': cv_results['val_mean'],
            'val_std': cv_results['val_std'],
            'train_mean': cv_results['train_mean'],
            'train_std': cv_results['train_std'],
        })
        print(f"Validation mean: {cv_results['val_mean']:.4f} ± {cv_results['val_std']:.4f}")
        print(f"Training mean: {cv_results['train_mean']:.4f} ± {cv_results['train_std']:.4f}")

    # print summary 
    print("\nSummary Alpha Tuning Results:")
    for r in results:
        print(f"Alpha: {r['alpha']:.4f} | Validation Mean: {r['val_mean']:.4f} ± {r['val_std']:.4f} | Training Mean: {r['train_mean']:.4f} ± {r['train_std']:.4f}")
    
    # Best alpha
    best = max(results, key=lambda x: x['val_mean'])
    print(f"\nBest Alpha: {best['alpha']:.4f} | Validation Mean: {best['val_mean']:.4f} ± {best['val_std']:.4f} | Training Mean: {best['train_mean']:.4f} ± {best['train_std']:.4f}")

if __name__ == "__main__":
    tune_precision_alpha()