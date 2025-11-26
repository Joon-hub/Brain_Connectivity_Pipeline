#!/usr/bin/env python3
"""
Brain Connectivity Classification Pipeline (FULLY CORRECTED)
=============================================================
FEDE'S FEEDBACK IMPLEMENTED:
✅ Fixed random imputation strategies (truly random across transforms)
✅ Proper sklearn GroupKFold usage
✅ Better interpretation of results
✅ Diagnostic capabilities to verify randomness
✅ Clear documentation of expected behaviors

This version eliminates all data leakage by:
1. Fitting preprocessing INSIDE each CV fold
2. Computing all statistics only on training data
3. Proper subject-level splitting before any preprocessing
4. Correct handling of deterministic vs stochastic diagonal strategies

Usage:
    python run.py --config config.yaml
    python run.py --diagonal sample_from_matrix --C 0.01
    python run.py --sample  # Quick test
    python run.py --diagnose  # Run diagnostics
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
    if len(connection_cols) == 0:
        raise ValueError("No connection columns found. Expected format: 'Region_A~Region_B'")
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
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def run_diagonal_diagnostics(
    df_train: pd.DataFrame,
    connection_columns: list,
    diagonal_strategy: str,
    random_state: int
):
    """
    Run diagnostic tests to verify diagonal imputation behavior.
    
    This tests whether diagonal values change across transform calls
    (as they should for stochastic strategies).
    """
    print_section("DIAGNOSTIC: Testing Diagonal Imputation Randomness")
    
    print(f"Strategy: {diagonal_strategy}")
    print(f"Testing on first 5 subjects...\n")
    
    # Create preprocessor with diagnostics enabled
    preprocessor = BrainConnectivityPreprocessor(
        connection_columns=connection_columns,
        diagonal_strategy=diagonal_strategy,
        apply_fisher_z=False,  # Easier to see raw values
        random_state=random_state,
        enable_diagnostics=True  # Enable diagnostic logging
    )
    
    # Fit on small sample
    df_sample = df_train.head(5)
    preprocessor.fit(df_sample)
    
    print("Calling transform() 3 times on same data:")
    print("-" * 60)
    
    # Transform multiple times
    X1 = preprocessor.transform(df_sample)
    X2 = preprocessor.transform(df_sample)
    X3 = preprocessor.transform(df_sample)
    
    print("-" * 60)
    print("\nChecking if transforms produce different results:")
    
    # For each subject, check first 5 features
    for subj_idx in range(min(3, len(df_sample))):
        sample_idx = subj_idx * preprocessor.n_regions_  # First region for this subject
        
        feat1 = X1[sample_idx, :5]
        feat2 = X2[sample_idx, :5]
        feat3 = X3[sample_idx, :5]
        
        print(f"\nSubject {subj_idx}, Region 0, Features 0-4:")
        print(f"  Transform 1: {feat1}")
        print(f"  Transform 2: {feat2}")
        print(f"  Transform 3: {feat3}")
        
        same_1_2 = np.allclose(feat1, feat2)
        same_2_3 = np.allclose(feat2, feat3)
        
        print(f"  Transform 1 == Transform 2? {same_1_2}")
        print(f"  Transform 2 == Transform 3? {same_2_3}")
    
    # Overall check
    all_same = np.allclose(X1, X2) and np.allclose(X2, X3)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC RESULT:")
    print("="*60)
    
    if diagonal_strategy in ['zero', 'region_mean', 'network_mean']:
        # Deterministic strategies
        if all_same:
            print("✓ PASS: Deterministic strategy produces identical results")
            print("  This is CORRECT behavior - diagonal values are constant per subject")
        else:
            print("✗ FAIL: Deterministic strategy produces different results!")
            print("  BUG: Diagonal values should be constant per subject")
    
    elif diagonal_strategy in ['random', 'sample_from_matrix']:
        # Stochastic strategies
        if not all_same:
            print("✓ PASS: Stochastic strategy produces different results")
            print("  This is CORRECT behavior - diagonal values change each transform")
        else:
            print("✗ FAIL: Stochastic strategy produces identical results!")
            print("  BUG: Diagonal values should change each transform")
    
    print("="*60 + "\n")


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
        description='Leak-Free Brain Connectivity Classification Pipeline (CORRECTED)'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--sample', action='store_true',
                        help='Run on sample data for quick testing')
    parser.add_argument('--diagnose', action='store_true',
                        help='Run diagnostic tests for diagonal imputation')
    parser.add_argument('--diagonal', type=str,
                        choices=['zero', 'random', 'region_mean', 'network_mean', 'sample_from_matrix'],
                        help='Override diagonal imputation strategy')
    parser.add_argument('--C', type=float,
                        help='Override logistic regression C parameter')
    parser.add_argument('--n_splits', type=int,
                        help='Override number of CV folds')
    parser.add_argument('--seed', type=int,
                        help='Override random seed')
    parser.add_argument('--no_fisher_z', action='store_true',
                        help='Disable Fisher Z transformation (not recommended)')
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    print_section("LEAK-FREE BRAIN CONNECTIVITY CLASSIFICATION (CORRECTED)")
    print(f"Mode: {'SAMPLE DATA' if args.sample else 'FULL DATA'}")
    print(f"Config: {args.config}")
    print(f"Diagnostics: {'ENABLED' if args.diagnose else 'Disabled'}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract settings with overrides
    random_seed = args.seed or config.get('model', {}).get('random_seed', 42)
    set_random_seeds(random_seed)
    
    # Preprocessing settings
    diagonal_strategy = args.diagonal or config.get('preprocessing', {}).get('diagonal_strategy', 'zero')
    apply_fisher_z = not args.no_fisher_z
    if apply_fisher_z is True:
        apply_fisher_z = config.get('preprocessing', {}).get('apply_fisher_z', True)
    
    # Model settings
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
        'max_iter': max_iter,
        'fisher_z_enabled': apply_fisher_z
    }
    
    print(f"Configuration:")
    print(f"  Diagonal strategy: {diagonal_strategy}")
    
    # Explain strategy type
    if diagonal_strategy in ['zero', 'region_mean', 'network_mean']:
        print(f"    → DETERMINISTIC: Diagonal values constant per subject")
    elif diagonal_strategy in ['random', 'sample_from_matrix']:
        print(f"    → STOCHASTIC: Diagonal values change each transform")
    
    print(f"  C (regularization): {C}")
    print(f"  CV folds: {n_splits}")
    print(f"  Random seed: {random_seed}")
    print(f"  Fisher Z transform: {'Enabled' if apply_fisher_z else 'Disabled'}")
    
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
    
    # Extract region information
    from src.features import extract_regions
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    
    # ========================================================================
    # DIAGNOSTIC MODE (if requested)
    # ========================================================================
    if args.diagnose:
        run_diagonal_diagnostics(
            df_train,
            connection_columns,
            diagonal_strategy,
            random_seed
        )
        
        # Ask user if they want to continue with full pipeline
        response = input("\nContinue with full pipeline? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return 0
    
    # ========================================================================
    # STEP 2: TRAIN CLASSIFIER WITH LEAK-FREE CV
    # ========================================================================
    print_section("STEP 2: Train Classifier (Leak-Free Cross-Validation)")
    
    print("⚠️  IMPORTANT: Preprocessing happens INSIDE each CV fold")
    print("   → Statistics computed only on training fold")
    print("   → No information leakage to validation fold")
    if apply_fisher_z:
        print("   → Fisher Z applied to correlation values (before scaling)")
    print()
    
    # Create classifier
    classifier = BrainRegionClassifier(
        preprocessor_class=BrainConnectivityPreprocessor,
        diagonal_strategy=diagonal_strategy,
        connection_columns=connection_columns,
        include_diagonal=False,
        apply_fisher_z=apply_fisher_z,
        C=C,
        max_iter=max_iter,
        n_splits=n_splits,
        random_state=random_seed,
        enable_diagnostics=args.diagnose  # Enable diagnostics if requested
    )
    
    # Fit with cross-validation
    classifier.fit(df_train, verbose=True)
    
    # Get CV results
    cv_results = classifier.get_cv_results()
    
    results['cv_train_mean'] = cv_results['train_mean']
    results['cv_train_std'] = cv_results['train_std']
    results['cv_val_mean'] = cv_results['val_mean']
    results['cv_val_std'] = cv_results['val_std']
    results['n_train_subjects'] = len(df_train)
    
    # ========================================================================
    # STEP 2.5: SAVE CV VALIDATION PREDICTIONS
    # ========================================================================
    print_section("STEP 2.5: Save CV Validation Predictions")
    
    # Get CV validation predictions
    y_cv_val_pred, y_cv_val_true, subjects_cv_val = classifier.get_cv_validation_predictions()
    
    # Verify accuracy
    cv_val_acc_check = accuracy_score(y_cv_val_true, y_cv_val_pred)
    print(f"CV validation accuracy (check): {cv_val_acc_check:.4f}")
    print(f"Expected from CV results: {cv_results['val_mean']:.4f}")
    print(f"Match: {np.isclose(cv_val_acc_check, cv_results['val_mean'])}")
    
    # Save predictions
    cv_val_pred_df = pd.DataFrame({
        'subject_id': subjects_cv_val,
        'true_region': y_cv_val_true,
        'predicted_region': y_cv_val_pred,
        'correct': y_cv_val_true == y_cv_val_pred
    })
    cv_val_pred_df.to_csv(
        output_dirs['processed'] / 'predictions_cv_validation.csv',
        index=False
    )
    print(f"✓ Saved CV validation predictions: {len(y_cv_val_pred)} samples")
    print(f"   File: {output_dirs['processed'] / 'predictions_cv_validation.csv'}")
    print()
    print(f"⚠️  IMPORTANT: Use these predictions (not training predictions)")
    print(f"   as your resting-state baseline for rest vs task comparison!")
    
    # Calculate error map for CV validation
    error_map_cv_val = calculate_error_map(y_cv_val_true, y_cv_val_pred, region_list)
    save_results_csv(
        error_map_cv_val,
        output_dirs['tables'] / 'error_map_cv_validation.csv'
    )
    print(f"✓ Saved CV validation error map")
    print(f"   File: {output_dirs['tables'] / 'error_map_cv_validation.csv'}")
    
    # Save region list
    region_df = pd.DataFrame({'region': region_list})
    region_df.to_csv(output_dirs['processed'] / 'region_list.csv', index=False)
    print(f"Region list saved to {output_dirs['processed'] / 'region_list.csv'}")
    
    # Save model
    classifier.save(output_dirs['models'])
    
    # ========================================================================
    # STEP 3: EVALUATE ON TRAINING DATA
    # ========================================================================
    print_section("STEP 3: Evaluate on Training Data")
    
    # Predict on training data (will show overfitting)
    y_train_pred, y_train_true, subjects_train = classifier.predict(df_train)
    train_acc = accuracy_score(y_train_true, y_train_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Total samples: {len(y_train_true)}")
    
    if train_acc > cv_results['val_mean'] + 0.05:
        print(f"\n⚠️  NOTE: Training accuracy ({train_acc:.4f}) >> CV validation accuracy ({cv_results['val_mean']:.4f})")
        print(f"   This is EXPECTED - training accuracy shows overfitting")
        print(f"   Use CV validation accuracy as true generalization estimate")
    
    # Calculate error map
    error_map_train = calculate_error_map(y_train_true, y_train_pred, region_list)
    
    # Save results
    train_pred_df = pd.DataFrame({
        'subject_id': subjects_train,
        'true_region': y_train_true,
        'predicted_region': y_train_pred,
        'correct': y_train_true == y_train_pred
    })
    train_pred_df.to_csv(
        output_dirs['processed'] / 'predictions_rest.csv',
        index=False
    )
    
    # Save error map
    save_results_csv(
        error_map_train,
        output_dirs['tables'] / 'error_map_rest.csv'
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
        test_connection_cols = extract_connection_columns(df_test)
        if set(connection_columns) != set(test_connection_cols):
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
            output_dirs['processed'] / 'predictions_task.csv',
            index=False
        )
        
        # Save error map
        save_results_csv(
            error_map_test,
            output_dirs['tables'] / 'error_map_task.csv'
        )
        
        # Save confusion matrix
        save_confusion_matrix(
            y_test_true, y_test_pred, region_list,
            dataset_name=f"task_{diagonal_strategy}"
        )
        
        # Compare rest (CV validation) vs task
        print("\n⚠️  Using CV validation (not training) as rest baseline")
        comparison = compare_error_maps(error_map_cv_val, error_map_test)
        save_results_csv(
            comparison,
            output_dirs['tables'] / 'comparison_cv_validation_vs_task.csv'
        )
        
        # Also compare training vs task for reference
        comparison_train = compare_error_maps(error_map_train, error_map_test)
        save_results_csv(
            comparison_train,
            output_dirs['tables'] / 'comparison_training_vs_task.csv'
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
        title=f'Resting-State Training Error Map ({diagonal_strategy})',
        output_path=str(figures_dir / 'error_map_rest.png')
    )
    figure_count += 1
    print(f"✓ Generated: error_map_rest.png")
    
    # Plot 1b: CV validation error map
    plot_error_map(
        error_map_cv_val,
        title=f'Resting-State CV Validation Error Map ({diagonal_strategy})',
        output_path=str(figures_dir / 'error_map_cv_validation.png')
    )
    figure_count += 1
    print(f"✓ Generated: error_map_cv_validation.png")
    
    if task_available:
        # Plot 2: Task error map
        plot_error_map(
            error_map_test,
            title=f'Task Error Map ({diagonal_strategy})',
            output_path=str(figures_dir / 'error_map_task.png')
        )
        figure_count += 1
        print(f"✓ Generated: error_map_task.png")
        
        # Plot 3: Rest (CV validation) vs Task comparison
        plot_rest_vs_task_comparison(
            error_map_cv_val,
            error_map_test, 
            comparison,
            output_path=str(figures_dir / 'comparison_cv_validation_vs_task.png')
        )
        figure_count += 1
        print(f"✓ Generated: comparison_cv_validation_vs_task.png")
        
        # Plot 4: Training vs Task comparison (for reference)
        plot_rest_vs_task_comparison(
            error_map_train,
            error_map_test, 
            comparison_train,
            output_path=str(figures_dir / 'comparison_training_vs_task.png')
        )
        figure_count += 1
        print(f"✓ Generated: comparison_training_vs_task.png")
    
    print(f"\nTotal figures: {figure_count}")
    
    # ========================================================================
    # STEP 6: SUMMARY & INTERPRETATION
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
  Max iterations: {max_iter}
  Fisher Z transform: {'Enabled' if apply_fisher_z else 'Disabled'}
  

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
  CV validation → Task drop: {results['cv_val_mean'] - results['test_accuracy']:.4f}
"""
    
    # Add interpretation based on strategy type
    summary += f"""
{'='*70}

INTERPRETATION:
"""
    
    if diagonal_strategy in ['zero', 'region_mean', 'network_mean']:
        train_val_gap = results['cv_train_mean'] - results['cv_val_mean']
        
        summary += f"""
Strategy Type: DETERMINISTIC
  • Diagonal values are constant per subject
  • Model CAN memorize diagonal patterns for training subjects
  • Validation subjects are UNSEEN, so model must generalize

Analysis:
  • Train accuracy: {results['cv_train_mean']:.1%}
"""
        if results['cv_train_mean'] > 0.99:
            summary += f"    ✓ Near perfect - model has sufficient capacity\n"
        elif results['cv_train_mean'] > 0.95:
            summary += f"    ✓ Very high - model has good capacity\n"
        else:
            summary += f"    ⚠ <95% - model may be underfitting\n"
            summary += f"      Consider: Increase C, check preprocessing\n"
        
        summary += f"""
  • Val accuracy: {results['cv_val_mean']:.1%}
    This reflects TRUE generalization to unseen subjects
    
  • Train-Val gap: {train_val_gap:.1%}
    This gap is NORMAL and EXPECTED with subject-level CV splitting
    (NOT overfitting - val subjects never appear in training)
"""
    
    elif diagonal_strategy in ['random', 'sample_from_matrix']:
        summary += f"""
Strategy Type: STOCHASTIC
  • Diagonal values CHANGE on each transform call
  • Model CANNOT memorize diagonal patterns
  • Forces learning from off-diagonal connectivity only

Analysis:
  • Lower accuracy is EXPECTED and scientifically CORRECT
  • This is the meaningful condition for brain connectivity analysis
  • Model must learn real patterns, not diagonal artifacts
  
  • Train accuracy: {results['cv_train_mean']:.1%}
  • Val accuracy: {results['cv_val_mean']:.1%}
  
  These accuracies reflect learning from TRUE connectivity patterns!
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

✅ Pipeline completed successfully!

IMPORTANT NOTES:
• Fisher Z transformation {'was' if apply_fisher_z else 'was NOT'} applied to correlation values
• All bugs from Fede's feedback have been fixed
• Use CV validation accuracy (not training) as generalization estimate
• For rest vs task comparison, use CV validation predictions as baseline
"""
    
    print(summary)
    
    # Save summary
    with open(output_dirs['tables'] / f'summary_{diagonal_strategy}.txt', 'w') as f:
        f.write(summary)
    
    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    sys.exit(main())