#!/usr/bin/env python3
"""
Brain Connectivity Classification Pipeline (LEAK-FREE VERSION)
===============================================================
This version eliminates all data leakage by:
1. Fitting preprocessing INSIDE each CV fold
2. Computing all statistics only on training data
3. Proper subject-level splitting before any preprocessing

Usage:
    python run.py --config config.yaml
    python run.py --diagonal region_mean --C 0.01
    python run.py --sample  # Quick test
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from src.features import BrainConnectivityPreprocessor
from src.model import BrainRegionClassifier
from src.evaluate import (
    calculate_error_map, save_results_csv, 
    save_confusion_matrix, compare_error_maps
)
from src.visualize import plot_error_map, plot_rest_vs_task_comparison


# ============================================================================
# CONFIGURATION & UTILITIES
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_connectivity_data(filepath: str) -> pd.DataFrame:
    """Load connectivity CSV file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} subjects from {filepath.name}")
    return df


def extract_connection_columns(df: pd.DataFrame) -> list:
    """Extract connection column names (containing '~')."""
    connection_cols = [col for col in df.columns if '~' in str(col)]
    return connection_cols


def create_sample_dataset(input_path: str, output_path: str, n_subjects: int = 10):
    """Create small sample dataset for testing."""
    df = pd.read_csv(input_path)
    df_sample = df.head(n_subjects)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(output_path, index=False)
    print(f"✓ Created sample dataset: {output_path} ({n_subjects} subjects)")


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def format_time(seconds: float) -> str:
    """Format elapsed time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> int:
    """Run complete leak-free pipeline."""
    start_time = time.time()
    
    # ========================================================================
    # ARGUMENT PARSING
    # ========================================================================
    parser = argparse.ArgumentParser(
        description='Leak-Free Brain Connectivity Classification Pipeline'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--sample', action='store_true',
                        help='Run on sample data for quick testing')
    parser.add_argument('--diagonal', type=str,
                        choices=['zero', 'random', 'region_mean', 'network_mean', 'sample_from_matrix'],
                        help='Override diagonal imputation strategy')
    parser.add_argument('--C', type=float,
                        help='Override logistic regression C parameter')
    parser.add_argument('--n_splits', type=int,
                        help='Override number of CV folds')
    parser.add_argument('--seed', type=int,
                        help='Override random seed')
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    print_section("LEAK-FREE BRAIN CONNECTIVITY CLASSIFICATION")
    print(f"Mode: {'SAMPLE DATA' if args.sample else 'FULL DATA'}")
    print(f"Config: {args.config}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract settings with overrides
    random_seed = args.seed or config.get('model', {}).get('random_seed', 42)
    set_random_seeds(random_seed)
    
    diagonal_strategy = args.diagonal or config.get('diagonal_strategy', 'zero')
    C = args.C or config.get('model', {}).get('C', 0.01)
    n_splits = args.n_splits or config.get('model', {}).get('n_splits', 5)
    max_iter = config.get('model', {}).get('max_iter', 1000)
    
    # Data paths
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']
    
    # Output directories
    output_dirs = {k: Path(v) for k, v in config['output_dirs'].items()}
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Results dictionary
    results = {
        'diagonal_strategy': diagonal_strategy,
        'random_seed': random_seed,
        'n_splits': n_splits,
        'C': C,
        'max_iter': max_iter
    }
    
    print(f"Configuration:")
    print(f"  Diagonal strategy: {diagonal_strategy}")
    print(f"  C (regularization): {C}")
    print(f"  CV folds: {n_splits}")
    print(f"  Random seed: {random_seed}")
    
    # ========================================================================
    # STEP 1: LOAD TRAINING DATA (PIOP-2 Resting State)
    # ========================================================================
    print_section("STEP 1: Load Training Data (PIOP-2 Resting State)")
    
    if args.sample:
        sample_path = "data/sample/sample_piop2.csv"
        if not Path(sample_path).exists():
            create_sample_dataset(piop2_file, sample_path, n_subjects=30)
        piop2_file = sample_path
    
    df_train = load_connectivity_data(piop2_file)
    connection_columns = extract_connection_columns(df_train)
    
    print(f"  Subjects: {len(df_train)}")
    print(f"  Connections: {len(connection_columns)}")
    
    # ========================================================================
    # STEP 2: TRAIN CLASSIFIER WITH LEAK-FREE CV
    # ========================================================================
    print_section("STEP 2: Train Classifier (Leak-Free Cross-Validation)")
    
    print("⚠️  IMPORTANT: Preprocessing happens INSIDE each CV fold")
    print("   → Statistics computed only on training fold")
    print("   → No information leakage to validation fold\n")
    
    classifier = BrainRegionClassifier(
        preprocessor_class=BrainConnectivityPreprocessor,
        diagonal_strategy=diagonal_strategy,
        connection_columns=connection_columns,
        include_diagonal=True,  # CRITICAL: Exclude diagonal from features!
        C=C,
        max_iter=max_iter,
        n_splits=n_splits,
        random_state=random_seed
    )
    
    # Fit with leak-free CV
    classifier.fit(df_train, verbose=True)
    
    # Get CV results
    cv_results = classifier.get_cv_results()
    n_regions = classifier.n_regions_
    region_list = classifier.region_list_

    # save region list as csv file by converting it df first into data/processed directory
    region_list_df = pd.DataFrame(region_list)
    region_list_df.to_csv(output_dirs['processed'] / 'region_list.csv', index=False)
    print(f"Region list saved to {output_dirs['processed'] / 'region_list.csv'}")
    
    results.update({
        'n_regions': n_regions,
        'n_train_subjects': len(df_train),
        'cv_val_mean': cv_results['val_mean'],
        'cv_val_std': cv_results['val_std'],
        'cv_train_mean': cv_results['train_mean'],
        'cv_train_std': cv_results['train_std']
    })
    
    # Save model
    classifier.save(str(output_dirs['models']))
    
    # ========================================================================
    # STEP 3: EVALUATE ON TRAINING DATA
    # ========================================================================
    print_section("STEP 3: Evaluate on Training Data")
    
    y_train_pred, y_train_true, subjects_train = classifier.predict(df_train)
    train_acc = accuracy_score(y_train_true, y_train_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Total samples: {len(y_train_true)}")
    
    # Calculate error map
    error_map_train = calculate_error_map(y_train_true, y_train_pred, region_list)
    
    # Save predictions
    train_pred_df = pd.DataFrame({
        'subject_id': subjects_train,
        'true_region': y_train_true,
        'predicted_region': y_train_pred,
        'correct': y_train_true == y_train_pred
    })
    train_pred_df.to_csv(
        output_dirs['processed'] / f'predictions_train.csv',
        index=False
    )
    
    # Save error map
    save_results_csv(
        error_map_train,
        output_dirs['tables'] / f'error_map_rest.csv'
    )
    
    # Save confusion matrix
    save_confusion_matrix(
        y_train_true, y_train_pred, region_list,
        dataset_name=f"rest_{diagonal_strategy}"
    )
    
    results['train_accuracy'] = train_acc
    
    # ========================================================================
    # STEP 4: APPLY TO TASK DATA (PIOP-1)
    # ========================================================================
    print_section("STEP 4: Apply to Task Data (PIOP-1 Gender Stroop)")
    
    task_available = False
    
    try:
        if args.sample:
            sample_task_path = "data/sample/sample_piop1.csv"
            if not Path(sample_task_path).exists():
                create_sample_dataset(piop1_file, sample_task_path, n_subjects=20)
            piop1_file = sample_task_path
        
        df_test = load_connectivity_data(piop1_file)
        
        # Verify schema match
        if not all(col in df_test.columns for col in connection_columns):
            raise ValueError("Column mismatch between PIOP-1 and PIOP-2!")
        
        # Predict on task data
        y_test_pred, y_test_true, subjects_test = classifier.predict(df_test)
        test_acc = accuracy_score(y_test_true, y_test_pred)
        
        print(f"Task prediction accuracy: {test_acc:.4f}")
        print(f"Test subjects: {len(df_test)}")
        print(f"Test samples: {len(y_test_true)}")
        
        # Calculate error map
        error_map_test = calculate_error_map(y_test_true, y_test_pred, region_list)
        
        # Save predictions
        test_pred_df = pd.DataFrame({
            'subject_id': subjects_test,
            'true_region': y_test_true,
            'predicted_region': y_test_pred,
            'correct': y_test_true == y_test_pred
        })
        test_pred_df.to_csv(
            output_dirs['processed'] / f'predictions_task.csv',
            index=False
        )
        
        # Save error map
        save_results_csv(
            error_map_test,
            output_dirs['tables'] / f'error_map_task.csv'
        )
        
        # Save confusion matrix
        save_confusion_matrix(
            y_test_true, y_test_pred, region_list,
            dataset_name=f"task_{diagonal_strategy}"
        )
        
        # Compare rest vs task
        comparison = compare_error_maps(error_map_train, error_map_test)
        save_results_csv(
            comparison,
            output_dirs['tables'] / f'comparison_rest_vs_task.csv'
        )
        
        results['test_accuracy'] = test_acc
        results['n_test_subjects'] = len(df_test)
        
        task_available = True
        
    except FileNotFoundError as e:
        print(f"⚠️  Task data not found: {e}")
        print("   Skipping task analysis...")
    except Exception as e:
        print(f"⚠️  Error processing task data: {e}")
        print("   Continuing with training data only...")
    
    # ========================================================================
    # STEP 5: GENERATE VISUALIZATIONS
    # ========================================================================
    print_section("STEP 5: Generate Visualizations")
    
    figures_dir = output_dirs['figures']
    figure_count = 0
    
    # Plot 1: Training error map
    plot_error_map(
        error_map_train,
        title=f'Resting-State Error Map ({diagonal_strategy})',
        output_path=str(figures_dir / f'error_map_rest.png')
    )
    figure_count += 1
    print(f"✓ Generated: error_map_rest.png")
    
    if task_available:
        # Plot 2: Task error map
        plot_error_map(
            error_map_test,
            title=f'Task Error Map ({diagonal_strategy})',
            output_path=str(figures_dir / f'error_map_task.png')
        )
        figure_count += 1
        print(f"✓ Generated: error_map_task.png")
        
        # Plot 3: Rest vs Task comparison
        plot_rest_vs_task_comparison(
            error_map_train, error_map_test, comparison,
            output_path=str(figures_dir / f'comparison_rest_vs_task.png')
        )
        figure_count += 1
        print(f"✓ Generated: comparison_rest_vs_task.png")
    
    print(f"\nTotal figures: {figure_count}")
    
    # ========================================================================
    # STEP 6: SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    chance_level = 1.0 / n_regions
    improvement = results['cv_val_mean'] / chance_level
    
    print_section("PIPELINE COMPLETE!")
    
    summary = f"""
RESULTS SUMMARY
{'='*70}

Data:
  Training subjects: {results['n_train_subjects']}
  Brain regions: {n_regions}
  Chance level: {chance_level:.4f}

Model Configuration:
  Diagonal strategy: {diagonal_strategy}
  Regularization (C): {C}
  CV folds: {n_splits}
  max_iter: {max_iter}
  

Cross-Validation Results (LEAK-FREE):
  Validation accuracy: {results['cv_val_mean']:.4f} ± {results['cv_val_std']:.4f}
  Training accuracy: {results['cv_train_mean']:.4f} ± {results['cv_train_std']:.4f}
  Improvement over chance: {improvement:.1f}x

Final Training Accuracy: {results['train_accuracy']:.4f}
"""
    
    if task_available:
        summary += f"""
Task Data (Gender Stroop):
  Test subjects: {results['n_test_subjects']}
  Test accuracy: {results['test_accuracy']:.4f}
"""
    
    summary += f"""
{'='*70}

Output Locations:
  Models: {output_dirs['models']}
  Tables: {output_dirs['tables']}
  Figures: {output_dirs['figures']}
  Predictions: {output_dirs['processed']}

Execution Time: {format_time(elapsed)}
Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
"""
    
    print(summary)
    
    # Save summary
    with open(output_dirs['tables'] / f'summary_{diagonal_strategy}.txt', 'w') as f:
        f.write(summary)
    
    print("✅ Pipeline completed successfully!\n")
    
    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    sys.exit(main())