#!/usr/bin/env python3
"""
Brain Connectivity Classification Pipeline (Phase 2 - Integrated)
==================================================================
Pipeline to classify brain regions from functional connectivity data
with strict leak-free cross-validation and evaluation on task data.

Steps:
1. Load PIOP-2 resting-state data for training
2. Train classifier with leak-free cross-validation (preprocessing inside each fold)
3. Evaluate on training data and save results
4. Apply to PIOP-1 task data and compare performance

Usage:
    python run.py --config configs/config.yaml
    python run.py --config configs/config.yaml --model xgboost
    python run.py --config configs/config.yaml --sample
    python run.py --config configs/config.yaml --diagonal random --n_splits 5
"""

import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from src.models import load_model_from_config
from src.features import BrainConnectivityPreprocessor
from src.brain_region_classifier import BrainRegionClassifier
from src.evaluate import calculate_error_map, save_results_csv, save_confusion_matrix, compare_error_maps
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
# EXPERIMENT TRACKING
# ============================================================================

class ExperimentLogger:
    """Simple experiment tracking without external dependencies."""
    
    def __init__(self, base_dir: str = "results/experiments"):
        self.base_dir = Path(base_dir)
        self.experiment_dir = None
        self.metadata = {}
    
    def create_experiment(self, name: str, config: dict) -> Path:
        """Create directory structure for new experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{name}_{timestamp}"
        self.experiment_dir = self.base_dir / exp_name
        
        # Create subdirectories
        subdirs = ['config', 'models', 'predictions', 'metrics', 'figures', 'logs']
        for subdir in subdirs:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.experiment_dir / 'config' / 'run_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✓ Created experiment: {exp_name}")
        print(f"  Location: {self.experiment_dir}")
        
        return self.experiment_dir
    
    def log_metadata(self, key: str, value):
        """Log metadata key-value pair."""
        self.metadata[key] = value
    
    def save_metadata(self):
        """Save all logged metadata to JSON."""
        if self.experiment_dir is None:
            raise RuntimeError("Must create experiment first")
        
        metadata_path = self.experiment_dir / 'logs' / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✓ Saved metadata to: {metadata_path.name}")
    
    def get_paths(self) -> dict:
        """Get paths to all experiment subdirectories."""
        if self.experiment_dir is None:
            raise RuntimeError("Must create experiment first")
        
        return {
            'models': self.experiment_dir / 'models',
            'predictions': self.experiment_dir / 'predictions',
            'metrics': self.experiment_dir / 'metrics',
            'figures': self.experiment_dir / 'figures',
            'logs': self.experiment_dir / 'logs'
        }


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
    
    elif diagonal_strategy in ['random', 'sample_from_row', 'sample_from_matrix']:
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
    """Run complete leak-free pipeline with experiment tracking."""
    start_time = time.time()
    
    # ========================================================================
    # ARGUMENT PARSING
    # ========================================================================
    parser = argparse.ArgumentParser(
        description='Brain Connectivity Classification Pipeline (Phase 2 - Integrated)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with logistic regression (default)
  python run.py --config configs/config.yaml
  
  # Use XGBoost model
  python run.py --config configs/config.yaml --model xgboost
  
  # Quick test with sample data
  python run.py --config configs/config.yaml --sample
  
  # Override preprocessing strategy
  python run.py --config configs/config.yaml --diagonal random
  
  # Run diagnostics
  python run.py --config configs/config.yaml --diagnose
        """
    )
    
    # Required arguments
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    
    # Model selection (NEW in Phase 2!)
    parser.add_argument('--model', type=str, default='logistic_regression',
                        help='Model to use (corresponds to configs/models/{model}.yaml)')
    
    # Model parameter overrides
    parser.add_argument('--model-params', type=str, nargs='*',
                        help='Override model parameters (e.g., --model-params C=10.0 max_iter=2000)')
    
    # Data and execution modes
    parser.add_argument('--sample', action='store_true',
                        help='Run on sample data for quick testing')
    parser.add_argument('--diagnose', action='store_true',
                        help='Run diagnostic tests for diagonal imputation')
    
    # Preprocessing overrides
    parser.add_argument('--diagonal', type=str,
                        choices=['zero', 'random', 'region_mean', 'network_mean', 'sample_from_matrix'],
                        help='Override diagonal imputation strategy')
    parser.add_argument('--no-fisher-z', action='store_true',
                        help='Disable Fisher Z transformation (not recommended)')
    
    # Training overrides
    parser.add_argument('--n-splits', type=int,
                        help='Override number of CV folds')
    parser.add_argument('--seed', type=int,
                        help='Override random seed')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str,
                        help='Custom experiment name (default: auto-generated)')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable experiment tracking (use old output structure)')
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    print_section("BRAIN CONNECTIVITY CLASSIFICATION - PHASE 2")
    print(f"Mode: {'SAMPLE DATA' if args.sample else 'FULL DATA'}")
    print(f"Config: {args.config}")
    print(f"Model: {args.model}")
    print(f"Diagnostics: {'ENABLED' if args.diagnose else 'Disabled'}")
    print(f"Experiment Tracking: {'DISABLED' if args.no_tracking else 'ENABLED'}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract settings with overrides
    random_seed = args.seed if args.seed is not None else config.get('model', {}).get('random_seed', 42)
    set_random_seeds(random_seed)
    
    # Preprocessing settings
    diagonal_strategy = args.diagonal or config.get('preprocessing', {}).get('diagonal_strategy', 'zero')
    apply_fisher_z = not args.no_fisher_z and config.get('preprocessing', {}).get('apply_fisher_z', True)
    
    # Training settings
    n_splits = args.n_splits if args.n_splits is not None else config.get('model', {}).get('n_splits', 5)
    
    # Data paths
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']
    
    # Parse model parameter overrides
    model_param_overrides = {}
    if args.model_params:
        for param in args.model_params:
            try:
                key, value = param.split('=')
                # Try to convert to appropriate type
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                model_param_overrides[key] = value
            except ValueError:
                print(f"⚠️  Warning: Invalid parameter format '{param}', skipping")
    
    # Setup experiment tracking or use old structure
    if not args.no_tracking:
        # NEW: Experiment tracking
        exp_name = args.experiment_name or f"{args.model}_{diagonal_strategy}"
        logger = ExperimentLogger()
        
        # Create experiment structure
        experiment_config = {
            'model': args.model,
            'model_params': model_param_overrides,
            'diagonal_strategy': diagonal_strategy,
            'apply_fisher_z': apply_fisher_z,
            'n_splits': n_splits,
            'random_seed': random_seed,
            'sample_mode': args.sample,
            'data': {
                'train': piop2_file,
                'test': piop1_file
            }
        }
        logger.create_experiment(exp_name, experiment_config)
        output_dirs = logger.get_paths()
        
        # Log basic metadata
        logger.log_metadata('start_time', datetime.now().isoformat())
        logger.log_metadata('model', args.model)
        logger.log_metadata('diagonal_strategy', diagonal_strategy)
    else:
        # OLD: Use config-specified directories
        output_dirs = {k: Path(v) for k, v in config['output_dirs'].items()}
        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        logger = None
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Diagonal strategy: {diagonal_strategy}")
    
    # Explain strategy type
    if diagonal_strategy in ['zero', 'region_mean', 'network_mean']:
        print(f"    → DETERMINISTIC: Diagonal values constant per subject")
    elif diagonal_strategy in ['random', 'sample_from_matrix']:
        print(f"    → STOCHASTIC: Diagonal values change each transform")
    
    print(f"  CV folds: {n_splits}")
    print(f"  Random seed: {random_seed}")
    print(f"  Fisher Z transform: {'Enabled' if apply_fisher_z else 'Disabled'}")
    if model_param_overrides:
        print(f"  Model parameter overrides: {model_param_overrides}")
    
    # ========================================================================
    # STEP 1: LOAD TRAINING DATA (PIOP-2 Resting State)
    # ========================================================================
    print_section("STEP 1: Load Training Data (PIOP-2 Resting State)")
    
    if args.sample:
        sample_path = "data/sample/sample_piop2.csv"
        if not Path(sample_path).exists():
            n_sample = config.get('sample', {}).get('n_subjects', 30)
            create_sample_dataset(piop2_file, sample_path, n_subjects=n_sample)
        piop2_file = sample_path
    
    df_train = load_connectivity_data(piop2_file)
    connection_columns = extract_connection_columns(df_train)
    
    print(f"  Subjects: {len(df_train)}")
    print(f"  Connections: {len(connection_columns)}")
    
    # Extract region information
    from src.features import extract_regions
    region_list, region_to_idx, n_regions = extract_regions(connection_columns)
    
    if logger:
        logger.log_metadata('n_train_subjects', len(df_train))
        logger.log_metadata('n_regions', n_regions)
    
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
    # STEP 2: LOAD MODEL & CREATE CLASSIFIER
    # ========================================================================
    print_section("STEP 2: Load Model & Create Classifier")
    
    # Load model from config (NEW in Phase 2!)
    try:
        model_instance = load_model_from_config(
            args.model,
            config_dir="configs/models",
            **model_param_overrides
        )
        print(f"✓ Loaded model: {type(model_instance).__name__}")
    except Exception as e:
        print(f"✗ Error loading model '{args.model}': {e}")
        print(f"\nAvailable models in configs/models/:")
        from src.models import list_available_models
        available = list_available_models("configs/models")
        for m in available:
            print(f"  - {m}")
        return 1
    
    # Create classifier (NEW API in Phase 2!)
    classifier = BrainRegionClassifier(
        preprocessor_class=BrainConnectivityPreprocessor,
        model_instance=model_instance,
        model_name=args.model,
        diagonal_strategy=diagonal_strategy,
        connection_columns=connection_columns,
        include_diagonal=False,
        apply_fisher_z=apply_fisher_z,
        n_splits=n_splits,
        random_state=random_seed,
        enable_diagnostics=args.diagnose
    )
    
    print(f"✓ Created classifier")
    print(f"  Preprocessor: BrainConnectivityPreprocessor")
    print(f"  Model: {type(model_instance).__name__}")
    print(f"  Cross-validation: {n_splits}-fold GroupKFold")
    
    # ========================================================================
    # STEP 3: TRAIN WITH LEAK-FREE CROSS-VALIDATION
    # ========================================================================
    print_section("STEP 3: Train with Leak-Free Cross-Validation")
    
    print("⚠️  IMPORTANT: Preprocessing happens INSIDE each CV fold")
    print("   → Statistics computed only on training fold")
    print("   → No information leakage to validation fold")
    if apply_fisher_z:
        print("   → Fisher Z applied to correlation values (before scaling)")
    print()
    
    # Train classifier
    classifier.fit(df_train, verbose=True)
    
    # Get CV results
    cv_results = classifier.get_cv_results()
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary:")
    print(f"  Training Accuracy:   {cv_results['train_mean']:.4f} ± {cv_results['train_std']:.4f}")
    print(f"  Validation Accuracy: {cv_results['val_mean']:.4f} ± {cv_results['val_std']:.4f}")
    print(f"{'='*60}\n")
    
    # Get CV validation predictions for error map
    y_pred_cv_val, y_true_cv_val, subjects_cv_val = classifier.get_cv_validation_predictions()
    
    # Calculate error maps
    error_map_cv_val = calculate_error_map(y_true_cv_val, y_pred_cv_val, n_regions)
    
    # Save CV results
    cv_results_df = pd.DataFrame(cv_results['fold_results'])
    save_results_csv(
        cv_results_df,
        output_dirs['metrics'] / 'cv_fold_results.csv'
    )
    
    # Save CV validation predictions
    cv_preds_df = pd.DataFrame({
        'subject_id': subjects_cv_val,
        'true_label': y_true_cv_val,
        'predicted_label': y_pred_cv_val
    })
    cv_preds_df.to_csv(output_dirs['predictions'] / 'cv_validation_predictions.csv', index=False)
    
    # Save error map
    error_map_df = pd.DataFrame({
        'region': region_list,
        'error_rate': error_map_cv_val
    })
    save_results_csv(
        error_map_df,
        output_dirs['metrics'] / 'error_map_cv_validation.csv'
    )
    
    # Save confusion matrix
    save_confusion_matrix(
        y_true_cv_val,
        y_pred_cv_val,
        region_list,
        output_dirs['metrics'] / 'confusion_matrix_cv_validation.csv'
    )
    
    # Log to experiment tracker
    if logger:
        logger.log_metadata('cv_train_mean', float(cv_results['train_mean']))
        logger.log_metadata('cv_train_std', float(cv_results['train_std']))
        logger.log_metadata('cv_val_mean', float(cv_results['val_mean']))
        logger.log_metadata('cv_val_std', float(cv_results['val_std']))
    
    # ========================================================================
    # STEP 3.5: EVALUATE ON TRAINING DATA (FOR REFERENCE)
    # ========================================================================
    print_section("STEP 3.5: Evaluate on Training Data (for reference)")
    
    y_pred_train, y_true_train, subjects_train = classifier.predict(df_train)
    train_acc = accuracy_score(y_true_train, y_pred_train)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"NOTE: This is overfitted - use CV validation for unbiased estimate")
    
    # Calculate error map
    error_map_train = calculate_error_map(y_true_train, y_pred_train, n_regions)
    
    # Save predictions
    train_preds_df = pd.DataFrame({
        'subject_id': subjects_train,
        'true_label': y_true_train,
        'predicted_label': y_pred_train
    })
    train_preds_df.to_csv(output_dirs['predictions'] / 'train_predictions.csv', index=False)
    
    # Save error map
    train_error_df = pd.DataFrame({
        'region': region_list,
        'error_rate': error_map_train
    })
    save_results_csv(
        train_error_df,
        output_dirs['metrics'] / 'error_map_train.csv'
    )
    
    # Save confusion matrix
    save_confusion_matrix(
        y_true_train,
        y_pred_train,
        region_list,
        output_dirs['metrics'] / 'confusion_matrix_train.csv'
    )
    
    if logger:
        logger.log_metadata('train_accuracy', float(train_acc))
    
    # ========================================================================
    # STEP 4: SAVE MODEL
    # ========================================================================
    print_section("STEP 4: Save Trained Model")
    
    classifier.save(str(output_dirs['models']))
    print(f"✓ Model saved to: {output_dirs['models']}")
    
    # ========================================================================
    # STEP 5: EVALUATE ON TASK DATA (PIOP-1)
    # ========================================================================
    print_section("STEP 5: Evaluate on Task Data (PIOP-1 Gender Stroop)")
    
    task_available = False
    
    try:
        if args.sample:
            sample_path = "data/sample/sample_piop1.csv"
            if not Path(sample_path).exists():
                n_sample = config.get('sample', {}).get('n_subjects', 30)
                create_sample_dataset(piop1_file, sample_path, n_subjects=n_sample)
            piop1_file = sample_path
        
        df_test = load_connectivity_data(piop1_file)
        
        print(f"  Test subjects: {len(df_test)}")
        
        # Predict on task data
        y_pred_test, y_true_test, subjects_test = classifier.predict(df_test)
        test_acc = accuracy_score(y_true_test, y_pred_test)
        
        print(f"\nTest Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  CV Validation → Test drop: {cv_results['val_mean'] - test_acc:.4f}")
        
        # Calculate error map for task data
        error_map_test = calculate_error_map(y_true_test, y_pred_test, n_regions)
        
        # Save test predictions
        test_preds_df = pd.DataFrame({
            'subject_id': subjects_test,
            'true_label': y_true_test,
            'predicted_label': y_pred_test
        })
        test_preds_df.to_csv(output_dirs['predictions'] / 'test_predictions.csv', index=False)
        
        # Save test error map
        test_error_df = pd.DataFrame({
            'region': region_list,
            'error_rate': error_map_test
        })
        save_results_csv(
            test_error_df,
            output_dirs['metrics'] / 'error_map_test.csv'
        )
        
        # Save test confusion matrix
        save_confusion_matrix(
            y_true_test,
            y_pred_test,
            region_list,
            output_dirs['metrics'] / 'confusion_matrix_test.csv'
        )
        
        # Compare CV validation vs Task
        comparison = compare_error_maps(error_map_cv_val, error_map_test)
        save_results_csv(
            comparison,
            output_dirs['metrics'] / 'comparison_cv_validation_vs_task.csv'
        )
        
        # Compare training vs Task
        comparison_train = compare_error_maps(error_map_train, error_map_test)
        save_results_csv(
            comparison_train,
            output_dirs['metrics'] / 'comparison_train_vs_task.csv'
        )
        
        # Log to experiment tracker
        if logger:
            logger.log_metadata('n_test_subjects', len(df_test))
            logger.log_metadata('test_accuracy', float(test_acc))
        
        task_available = True
        
    except FileNotFoundError as e:
        print(f"⚠️  Task data not found: {e}")
        print("   Skipping task analysis...")
    except Exception as e:
        print(f"⚠️  Error processing task data: {e}")
        print("   Continuing with training data only...")
    
    # ========================================================================
    # STEP 6: GENERATE VISUALIZATIONS
    # ========================================================================
    print_section("STEP 6: Generate Visualizations")
    
    figures_dir = output_dirs['figures']
    figure_count = 0
    
    # Plot 1: CV Validation error map
    plot_error_map(
        error_map_cv_val,
        title=f'CV Validation Error Map ({args.model}, {diagonal_strategy})',
        output_path=str(figures_dir / 'error_map_cv_validation.png'),
        region_list=region_list
    )
    figure_count += 1
    print(f"✓ Generated: error_map_cv_validation.png")
    
    # Plot 2: Training error map
    plot_error_map(
        error_map_train,
        title=f'Training Error Map ({args.model}, {diagonal_strategy})',
        output_path=str(figures_dir / 'error_map_train.png'),
        region_list=region_list
    )
    figure_count += 1
    print(f"✓ Generated: error_map_train.png")
    
    if task_available:
        # Plot 3: Task error map
        plot_error_map(
            error_map_test,
            title=f'Task Error Map ({args.model}, {diagonal_strategy})',
            output_path=str(figures_dir / 'error_map_task.png'),
            region_list=region_list
        )
        figure_count += 1
        print(f"✓ Generated: error_map_task.png")
        
        # Plot 4: CV Validation vs Task comparison
        plot_rest_vs_task_comparison(
            error_map_cv_val,
            error_map_test,
            comparison,
            output_path=str(figures_dir / 'comparison_cv_validation_vs_task.png'),
            region_list=region_list
        )
        figure_count += 1
        print(f"✓ Generated: comparison_cv_validation_vs_task.png")
        
        # Plot 5: Training vs Task comparison
        plot_rest_vs_task_comparison(
            error_map_train,
            error_map_test,
            comparison_train,
            output_path=str(figures_dir / 'comparison_train_vs_task.png'),
            region_list=region_list
        )
        figure_count += 1
        print(f"✓ Generated: comparison_train_vs_task.png")
    
    print(f"\nTotal figures: {figure_count}")
    
    # ========================================================================
    # STEP 7: SAVE EXPERIMENT METADATA & SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    
    if logger:
        logger.log_metadata('end_time', datetime.now().isoformat())
        logger.log_metadata('duration_seconds', elapsed)
        logger.log_metadata('n_figures_generated', figure_count)
        logger.save_metadata()
    
    # ========================================================================
    # STEP 8: GENERATE SUMMARY
    # ========================================================================
    chance_level = 1.0 / n_regions
    improvement = cv_results['val_mean'] / chance_level
    
    print_section("PIPELINE COMPLETE!")
    
    summary = f"""
RESULTS SUMMARY
{'='*70}

Experiment: {args.model}_{diagonal_strategy}
Model: {type(model_instance).__name__}

Data:
  Training subjects: {len(df_train)}
  Brain regions: {n_regions}
  Chance level: {chance_level:.4f}

Configuration:
  Diagonal strategy: {diagonal_strategy}
  CV folds: {n_splits}
  Fisher Z transform: {'Enabled' if apply_fisher_z else 'Disabled'}
  Random seed: {random_seed}

Cross-Validation Results (LEAK-FREE):
  Validation accuracy: {cv_results['val_mean']:.4f} ± {cv_results['val_std']:.4f}  
  Training accuracy: {cv_results['train_mean']:.4f} ± {cv_results['train_std']:.4f}  
  Full Training Accuracy: {train_acc:.4f} (overfitted)
  Improvement over chance: {improvement:.1f}x
"""
    
    if task_available:
        summary += f"""
Task Data (Gender Stroop):
  Test subjects: {len(df_test)}
  Test accuracy: {test_acc:.4f}
  CV validation → Task drop: {cv_results['val_mean'] - test_acc:.4f}
"""
    
    summary += f"""
{'='*70}

Output Locations:
"""
    
    if logger:
        summary += f"  Experiment: {logger.experiment_dir}\n"
    
    summary += f"""  Models: {output_dirs['models']}
  Metrics: {output_dirs['metrics']}
  Figures: {output_dirs['figures']}
  Predictions: {output_dirs['predictions']}

Execution Time: {format_time(elapsed)}
Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}

✅ Pipeline completed successfully!
"""
    
    print(summary)
    
    # Save summary
    summary_path = output_dirs['logs'] / f'summary_{args.model}_{diagonal_strategy}.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    sys.exit(main())