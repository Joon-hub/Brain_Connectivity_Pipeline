#!/usr/bin/env python3
"""
Brain Connectivity Classification Pipeline

Usage:
    python run.py --config configs/config.yaml
    python run.py --config configs/config.yaml --model logistic_regression
    python run.py --config configs/config.yaml --diagonal random --n-splits 5
"""

import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import load_model_from_config
from src.features import BrainConnectivityPreprocessor, extract_regions
from src.brain_region_classifier import BrainRegionClassifier
from src.evaluate import (
    calculate_error_map,
    save_results_csv,
    save_confusion_matrix,
    compare_error_maps,
)
from src.visualize import plot_error_map, plot_rest_vs_task_comparison


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_random_seeds(seed: int):
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_connectivity_data(filepath: str) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} subjects from {filepath.name}")
    return df


def extract_connection_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if '~' in str(c)]
    if not cols:
        raise ValueError("No connection columns found. Expected 'Region_A~Region_B'")
    return cols


def create_sample_dataset(input_path: str, output_path: str, n_subjects: int = 10):
    df = pd.read_csv(input_path).head(n_subjects)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset: {output_path} ({n_subjects} subjects)")


def print_section(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class ExperimentLogger:
    def __init__(self, base_dir: str = "results/experiments"):
        self.base_dir = Path(base_dir)
        self.experiment_dir = None
        self.metadata = {}

    def create_experiment(self, name: str, config: dict) -> Path:
        self.experiment_dir = self.base_dir / name
        if self.experiment_dir.exists():
            print(f"Experiment '{name}' already exists. Overwriting...")
            import shutil
            shutil.rmtree(self.experiment_dir)
        
        for sub in ['config', 'models', 'predictions', 'metrics', 'figures', 'logs']:
            (self.experiment_dir / sub).mkdir(parents=True, exist_ok=True)
        
        cfg_path = self.experiment_dir / 'config' / 'run_config.yaml'
        with open(cfg_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Created experiment: {name}")
        print(f"  Location: {self.experiment_dir}")
        return self.experiment_dir

    def log_metadata(self, key: str, value):
        self.metadata[key] = value

    def save_metadata(self):
        if self.experiment_dir is None:
            raise RuntimeError("Must create experiment first")
        meta_path = self.experiment_dir / 'logs' / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata to: {meta_path.name}")

    def get_paths(self) -> dict:
        if self.experiment_dir is None:
            raise RuntimeError("Must create experiment first")
        base = self.experiment_dir
        return {
            'models': base / 'models',
            'predictions': base / 'predictions',
            'metrics': base / 'metrics',
            'figures': base / 'figures',
            'logs': base / 'logs',
        }


def main() -> int:
    start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Brain Connectivity Classification Pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default='logistic_regression',
                        help='Model to use (configs/models/{model}.yaml)')
    parser.add_argument('--model-params', type=str, nargs='*',
                        help='Override model parameters, e.g. C=10.0 max_iter=2000')
    parser.add_argument('--sample', action='store_true',
                        help='Run on sample data')
    parser.add_argument('--diagonal', type=str,
                        choices=['zero', 'region_mean', 'network_mean','random', 'sample_matrix', 'sample_row'],
                        help='Override diagonal imputation strategy')
    parser.add_argument('--no-fisher-z', action='store_true',
                        help='Disable Fisher Z transformation')
    parser.add_argument('--n-splits', type=int,
                        help='Override number of CV folds')
    parser.add_argument('--seed', type=int,
                        help='Override random seed')
    parser.add_argument('--experiment-name', type=str,
                        help='Custom experiment name')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable experiment tracking')
    args = parser.parse_args()

    # Print header
    print_section("BRAIN CONNECTIVITY CLASSIFICATION")
    print(f"Mode: {'SAMPLE DATA' if args.sample else 'FULL DATA'}")
    print(f"Config: {args.config}")
    print(f"Model: {args.model}")
    print(f"Experiment Tracking: {'DISABLED' if args.no_tracking else 'ENABLED'}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load config
    config = load_config(args.config)
    random_seed = args.seed if args.seed is not None else config.get('model', {}).get('random_seed', 42)
    set_random_seeds(random_seed)

    # Get parameters
    diagonal_strategy = args.diagonal or config.get('preprocessing', {}).get('diagonal_strategy', 'zero')
    apply_fisher_z = not args.no_fisher_z and config.get('preprocessing', {}).get('apply_fisher_z', True)
    include_diagonal = config.get('preprocessing', {}).get('include_diagonal', True)
    n_splits = args.n_splits if args.n_splits is not None else config.get('model', {}).get('n_splits', 5)

    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']

    # Parse model parameter overrides
    model_param_overrides = {}
    if args.model_params:
        for param in args.model_params:
            try:
                key, value = param.split('=')
                try:
                    value_f = float(value)
                    value = int(value_f) if value_f.is_integer() else value_f
                except ValueError:
                    pass
                model_param_overrides[key] = value
            except ValueError:
                print(f"Warning: Invalid parameter format '{param}', skipping")

    # Setup experiment tracking
    if not args.no_tracking:
        exp_name = args.experiment_name or f"{args.model}_{diagonal_strategy}"
        logger = ExperimentLogger()
        experiment_config = {
            'model': args.model,
            'model_params': model_param_overrides,
            'diagonal_strategy': diagonal_strategy,
            'apply_fisher_z': apply_fisher_z,
            'n_splits': n_splits,
            'random_seed': random_seed,
            'sample_mode': args.sample,
            'data': {'train': piop2_file, 'test': piop1_file},
        }
        logger.create_experiment(exp_name, experiment_config)
        output_dirs = logger.get_paths()
        logger.log_metadata('start_time', datetime.now().isoformat())
        logger.log_metadata('model', args.model)
        logger.log_metadata('diagonal_strategy', diagonal_strategy)
    else:
        output_dirs = {k: Path(v) for k, v in config['output_dirs'].items()}
        for d in output_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        logger = None

    # Print configuration
    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Diagonal strategy: {diagonal_strategy}")
    print(f"  CV folds: {n_splits}")
    print(f"  Random seed: {random_seed}")
    print(f"  Fisher Z transform: {'Enabled' if apply_fisher_z else 'Disabled'}")
    if model_param_overrides:
        print(f"  Model parameter overrides: {model_param_overrides}")

    # Load training data
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

    region_list, region_to_idx, n_regions = extract_regions(connection_columns)

    if logger:
        logger.log_metadata('n_train_subjects', len(df_train))
        logger.log_metadata('n_regions', n_regions)

    # Load model
    print_section("STEP 2: Load Model & Create Classifier")
    try:
        model_instance = load_model_from_config(
            args.model,
            config_dir="configs/models",
            **model_param_overrides,
        )
        print(f"Loaded model: {type(model_instance).__name__}")
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}")
        from src.models import list_available_models
        print("\nAvailable models in configs/models/:")
        for m in list_available_models("configs/models"):
            print(f"  - {m}")
        return 1

    # Create classifier
    classifier = BrainRegionClassifier(
        preprocessor_class=BrainConnectivityPreprocessor,
        model_instance=model_instance,
        model_name=args.model,
        diagonal_strategy=diagonal_strategy,
        connection_columns=connection_columns,
        include_diagonal=include_diagonal,
        apply_fisher_z=apply_fisher_z,
        n_splits=n_splits,
        random_state=random_seed,
    )

    print("Created classifier")
    print(f"  Model: {type(model_instance).__name__}")
    print(f"  Cross-validation: {n_splits}-fold GroupKFold")

    # Train with cross-validation
    print_section("STEP 3: Train with Cross-Validation")
    classifier.fit(df_train, verbose=True)
    cv_results = classifier.get_cv_results()

    # Print CV results
    if cv_results['n_splits'] > 1:
        print("\n" + "="*70)
        print("CROSS-VALIDATION RESULTS")
        print("="*70)
        print(f"  Training Accuracy:   {cv_results['train_mean']:.4f} ± {cv_results['train_std']:.4f}")
        print(f"  Validation Accuracy: {cv_results['val_mean']:.4f} ± {cv_results['val_std']:.4f}")
        print(f"  Number of Folds:     {cv_results['n_splits']}")
        
        if 'fold_results' in cv_results and cv_results['fold_results']:
            print("\n  Individual Fold Results:")
            for fold_result in cv_results['fold_results']:
                print(f"    Fold {fold_result['fold']}: "
                      f"Train={fold_result['train_accuracy']:.4f}, "
                      f"Val={fold_result['val_accuracy']:.4f}")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("NO CROSS-VALIDATION PERFORMED")
        print("="*70)
        print("  Model trained directly on all data (n_splits=1)")
        print("="*70 + "\n")

    # Save CV results
    y_pred_cv_val, y_true_cv_val, subjects_cv_val = classifier.get_cv_validation_predictions()
    error_map_cv_val = calculate_error_map(y_true_cv_val, y_pred_cv_val, n_regions)

    if cv_results['fold_results']:
        cv_results_df = pd.DataFrame(cv_results['fold_results'])
        save_results_csv(cv_results_df, output_dirs['metrics'] / 'cv_fold_results.csv')

    if len(y_pred_cv_val) > 0:
        cv_preds_df = pd.DataFrame({
            'subject_id': subjects_cv_val,
            'true_region': y_true_cv_val,
            'predicted_region': y_pred_cv_val,
        })
        cv_preds_df.to_csv(output_dirs['predictions'] / 'cv_validation_predictions.csv', index=False)

        error_map_df = pd.DataFrame({'region': region_list, 'error_rate': error_map_cv_val})
        save_results_csv(error_map_df, output_dirs['metrics'] / 'error_map_cv_validation.csv')

        save_confusion_matrix(
            y_true_cv_val,
            y_pred_cv_val,
            region_list,
            output_dirs['metrics'] / 'confusion_matrix_cv_validation.csv',
        )

    if logger and cv_results['n_splits'] > 1:
        logger.log_metadata('cv_train_mean', float(cv_results['train_mean']))
        logger.log_metadata('cv_train_std', float(cv_results['train_std']))
        logger.log_metadata('cv_val_mean', float(cv_results['val_mean']))
        logger.log_metadata('cv_val_std', float(cv_results['val_std']))

    # Evaluate on training data
    print_section("STEP 4: Evaluate on Training Data")
    y_pred_train, y_true_train, subjects_train = classifier.predict(df_train)
    train_acc = accuracy_score(y_true_train, y_pred_train)
    print(f"Training accuracy: {train_acc:.4f}")
    print("Note: Overfitted; use CV validation for unbiased estimate")

    error_map_train = calculate_error_map(y_true_train, y_pred_train, n_regions)

    train_preds_df = pd.DataFrame({
        'subject_id': subjects_train,
        'true_region': y_true_train,
        'predicted_region': y_pred_train,
    })
    train_preds_df.to_csv(output_dirs['predictions'] / 'train_predictions.csv', index=False)

    train_error_df = pd.DataFrame({'region': region_list, 'error_rate': error_map_train})
    save_results_csv(train_error_df, output_dirs['metrics'] / 'error_map_train.csv')

    save_confusion_matrix(
        y_true_train,
        y_pred_train,
        region_list,
        output_dirs['metrics'] / 'confusion_matrix_train.csv',
    )

    if logger:
        logger.log_metadata('train_accuracy', float(train_acc))

    # Save model
    print_section("STEP 5: Save Model & Region List")
    classifier.save(str(output_dirs['models']))
    print(f"Model saved to: {output_dirs['models']}")

    region_list_df = pd.DataFrame({'region': region_list})
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    region_list_processed_path = processed_dir / 'region_list.csv'
    region_list_df.to_csv(region_list_processed_path, index=False)
    print(f"Region list saved: {region_list_processed_path}")

    # Evaluate on task data
    print_section("STEP 6: Evaluate on Task Data (PIOP-1 Gender Stroop)")
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

        y_pred_test, y_true_test, subjects_test = classifier.predict(df_test)
        test_acc = accuracy_score(y_true_test, y_pred_test)

        print("\nTest Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        if cv_results['n_splits'] > 1:
            print(f"  CV Validation → Test drop: {cv_results['val_mean'] - test_acc:.4f}")

        error_map_test = calculate_error_map(y_true_test, y_pred_test, n_regions)

        test_preds_df = pd.DataFrame({
            'subject_id': subjects_test,
            'true_region': y_true_test,
            'predicted_region': y_pred_test,
        })
        test_preds_df.to_csv(output_dirs['predictions'] / 'test_predictions.csv', index=False)

        test_error_df = pd.DataFrame({'region': region_list, 'error_rate': error_map_test})
        save_results_csv(test_error_df, output_dirs['metrics'] / 'error_map_test.csv')

        save_confusion_matrix(
            y_true_test,
            y_pred_test,
            region_list,
            output_dirs['metrics'] / 'confusion_matrix_test.csv',
        )

        if len(y_pred_cv_val) > 0:
            comparison = compare_error_maps(error_map_cv_val, error_map_test)
            save_results_csv(
                comparison,
                output_dirs['metrics'] / 'comparison_cv_validation_vs_task.csv',
            )

        comparison_train = compare_error_maps(error_map_train, error_map_test)
        save_results_csv(
            comparison_train,
            output_dirs['metrics'] / 'comparison_train_vs_task.csv',
        )

        if logger:
            logger.log_metadata('n_test_subjects', len(df_test))
            logger.log_metadata('test_accuracy', float(test_acc))

        task_available = True

    except FileNotFoundError as e:
        print(f"Task data not found: {e}")
        print("Skipping task analysis...")
    except Exception as e:
        print(f"Error processing task data: {e}")
        print("Continuing with training data only...")

    # Generate visualizations
    print_section("STEP 7: Generate Visualizations")
    figures_dir = output_dirs['figures']
    figure_count = 0

    if len(y_pred_cv_val) > 0:
        plot_error_map(
            error_map_cv_val,
            title=f'CV Validation Error Map ({args.model}, {diagonal_strategy})',
            output_path=str(figures_dir / 'error_map_cv_validation.png'),
            region_list=region_list,
        )
        figure_count += 1

    plot_error_map(
        error_map_train,
        title=f'Training Error Map ({args.model}, {diagonal_strategy})',
        output_path=str(figures_dir / 'error_map_train.png'),
        region_list=region_list,
    )
    figure_count += 1

    if task_available:
        plot_error_map(
            error_map_test,
            title=f'Task Error Map ({args.model}, {diagonal_strategy})',
            output_path=str(figures_dir / 'error_map_task.png'),
            region_list=region_list,
        )
        figure_count += 1

        if len(y_pred_cv_val) > 0:
            plot_rest_vs_task_comparison(
                error_map_cv_val,
                error_map_test,
                comparison,
                output_path=str(figures_dir / 'comparison_cv_validation_vs_task.png'),
                region_list=region_list,
            )
            figure_count += 1

        plot_rest_vs_task_comparison(
            error_map_train,
            error_map_test,
            comparison_train,
            output_path=str(figures_dir / 'comparison_train_vs_task.png'),
            region_list=region_list,
        )
        figure_count += 1

    print(f"\nTotal figures: {figure_count}")

    # Save metadata and print summary
    elapsed = time.time() - start_time
    if logger:
        logger.log_metadata('end_time', datetime.now().isoformat())
        logger.log_metadata('duration_seconds', elapsed)
        logger.log_metadata('n_figures_generated', figure_count)
        logger.save_metadata()

    chance_level = 1.0 / n_regions
    if cv_results['n_splits'] > 1:
        improvement = cv_results['val_mean'] / chance_level
    else:
        improvement = train_acc / chance_level

    # Print summary
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

"""
    
    if cv_results['n_splits'] > 1:
        summary += f"""Cross-Validation Results:
  Validation accuracy: {cv_results['val_mean']:.4f} ± {cv_results['val_std']:.4f}
  Training accuracy: {cv_results['train_mean']:.4f} ± {cv_results['train_std']:.4f}
  Full Training Accuracy: {train_acc:.4f} (overfitted)
  Improvement over chance: {improvement:.1f}x
"""
    else:
        summary += f"""Training Results:
  Training accuracy: {train_acc:.4f}
  Improvement over chance: {improvement:.1f}x
"""
    
    if task_available:
        summary += f"""
Task Data (Gender Stroop):
  Test subjects: {len(df_test)}
  Test accuracy: {test_acc:.4f}
"""
        if cv_results['n_splits'] > 1:
            summary += f"  CV validation → Task drop: {cv_results['val_mean'] - test_acc:.4f}\n"
    
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
"""

    print(summary)
    summary_path = output_dirs['logs'] / f'summary_{args.model}_{diagonal_strategy}.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())