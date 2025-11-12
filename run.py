#!/usr/bin/env python3
"""
Brain Connectivity Classification Pipeline
===========================================
Main entry point for the complete analysis pipeline.

Usage:
    python run.py --config config.yaml                         # Full pipeline
    python run.py --sample                                     # Quick test with sample data
    python run.py --config config.yaml --diagonal zero --C 1.0 # Override params
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# ======================================================================
# Project Modules
# ======================================================================
from data import (
    load_connectivity_data, extract_connection_columns, extract_subjects,
    create_sample_dataset, validate_schema
)
from src.features import (
    extract_regions, parse_networks, impute_diagonal,
    train_region_models, save_region_models, load_region_models,
    BrainConnectivityPreprocessor
)
from model import BrainRegionClassifierPipeline
from evaluate import (
    calculate_error_map, calculate_global_metrics,
    save_results_csv, save_confusion_matrix, compare_error_maps
)
from visualize import (
    plot_error_map, plot_rest_vs_task_comparison, plot_network_analysis
)
from utils import (
    setup_logging, set_random_seeds, load_config, log_provenance,
    print_section, format_time, save_config_copy
)

# ======================================================================
# Helper: Sample Data Handler
# ======================================================================
def get_dataset_path(original_path: str, sample: bool, sample_path: str):
    """Create sample if needed and return correct path."""
    if sample:
        Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(sample_path).exists():
            print(f"Creating sample dataset: {sample_path}")
            create_sample_dataset(original_path, sample_path, n_subjects=10)
        return sample_path
    if not Path(original_path).exists():
        raise FileNotFoundError(f"Data file not found: {original_path}")
    return original_path


# ======================================================================
# Config helpers (support both flat and nested 'model' keys)
# ======================================================================
def _get_cfg(cfg: dict, key: str, default=None):
    """Get a value from config; prefer nested cfg['model'][key] when present."""
    if isinstance(cfg.get('model'), dict) and key in cfg['model']:
        return cfg['model'].get(key, default)
    return cfg.get(key, default)


# ======================================================================
# Main Pipeline
# ======================================================================
def main() -> int:
    """Run the complete brain connectivity classification pipeline."""
    start_time = time.time()

    # ---------------------------------------------------------------
    # Argument parsing
    # ---------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Brain Connectivity Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--sample', action='store_true',
                        help='Run on sample data (10 subjects) for quick testing')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to log file (optional)')
    parser.add_argument(
        '--diagonal',
        type=str,
        choices=[
            'zero', 'mean', 'region_mean',
            'knn', 'random', 'network_mean', 'regression_predictive'
        ],
        help='Override diagonal imputation strategy'
    )
    parser.add_argument('--C', type=float, help='Logistic regression C parameter')
    parser.add_argument('--seed', type=int, help='Random seed override')
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    setup_logging(args.log)
    print_section("BRAIN CONNECTIVITY CLASSIFICATION PIPELINE")
    print(f"Mode: {'SAMPLE DATA (quick test)' if args.sample else 'FULL DATA'}")
    print(f"Config: {args.config}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load config and seed
    config = load_config(args.config)

    # Prefer nested model keys when present
    random_seed = args.seed or _get_cfg(config, 'random_seed', 42)
    set_random_seeds(random_seed)

    # Override config from CLI
    if args.diagonal:
        # store under nested model if present, else flat for backward-compat
        if isinstance(config.get('model'), dict):
            config['model']['diagonal_strategy'] = args.diagonal
        else:
            config['diagonal_strategy'] = args.diagonal
    if args.C is not None:
        if isinstance(config.get('model'), dict):
            config['model']['C'] = args.C
        else:
            config['C'] = args.C

    # Extract settings (support both config styles)
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']
    diagonal_strategy = _get_cfg(config, 'diagonal_strategy', 'region_mean')
    n_splits = _get_cfg(config, 'n_splits', 5)
    C = _get_cfg(config, 'C', 0.01)
    max_iter = _get_cfg(config, 'max_iter', 1000)

    # Output directories
    output_dirs = {k: Path(v) for k, v in config['output_dirs'].items()}

    # Create all output directories upfront
    for dir_path in output_dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Save config copy for reproducibility
    # save_config_copy(args.config, Path(output_dirs['tables']) / 'used_config.yaml')

    # Base results (we’ll add to this as we go)
    results = {
        'diagonal_strategy': diagonal_strategy,
        'random_seed': random_seed,
        'n_splits': n_splits,
        'C': C,
        'max_iter': max_iter,
        'used_region_models': False,
        'region_models_path': None,
    }

    # =========================================================================
    # STEP 1: Load Resting-State Data (PIOP-2)
    # =========================================================================
    print_section("STEP 1/6: Load Resting-State Data (PIOP-2)")
    piop2_path = get_dataset_path(
        piop2_file,
        args.sample,
        "data/sample/sample_piop2_small.csv"
    )
    df_piop2 = load_connectivity_data(piop2_path)
    connection_columns = extract_connection_columns(df_piop2)
    print(f"Loaded {len(df_piop2)} samples, {len(connection_columns)} connections")

    # =========================================================================
    # STEP 2: Preprocessing (conditionally train/load region models)
    # =========================================================================
    print_section("STEP 2/6: Extract Features and Preprocess Data")
    region_models = None
    region_models_pkl = Path(output_dirs['models']) / 'region_regression_models.pkl'

    if diagonal_strategy == 'regression_predictive':
        if region_models_pkl.exists():
            print(f"Loading existing region models from {region_models_pkl}")
            region_models, _ = load_region_models(region_models_pkl)
            results['used_region_models'] = True
            results['region_models_path'] = str(region_models_pkl)
        else:
            print("Training region-specific regression models for diagonal imputation...")
            region_models, _ = train_region_models(df_piop2, connection_columns)
            save_region_models(region_models, None, region_models_pkl)  # region_list not required here
            print(f"Region models saved to {region_models_pkl}")
            results['used_region_models'] = True
            results['region_models_path'] = str(region_models_pkl)
    else:
        print(f"Skipping regression model training/loading (strategy='{diagonal_strategy}').")

    # Initialize and fit preprocessor
    preprocessor = BrainConnectivityPreprocessor(
        connection_columns=connection_columns,
        diagonal_strategy=diagonal_strategy,
        region_models=region_models
    )
    print("Fitting preprocessor on resting-state data...")
    preprocessor.fit(df_piop2)

    # Canonical region metadata (single source of truth)
    region_list = preprocessor.region_list_
    n_regions = preprocessor.n_regions_

    # Transform to features/labels/subjects
    X_train = preprocessor.transform(df_piop2)
    y_train = preprocessor.get_labels()
    subjects_train_ids = preprocessor.get_subjects()

    # Save preprocessor
    preprocessor_path = Path(output_dirs['models']) / 'preprocessor.pkl'
    import joblib
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")

    # Update results
    results.update({
        'n_regions': n_regions,
        'n_train_samples': len(X_train),
        'n_train_subjects': len(np.unique(subjects_train_ids)),
    })

    # =========================================================================
    # STEP 3: Train Classifier with GroupCV
    # =========================================================================
    print_section("STEP 3/6: Train Brain Region Classifier (GroupKFold)")
    classifier = BrainRegionClassifierPipeline(
        C=C,
        max_iter=max_iter,
        n_splits=n_splits,
        random_state=random_seed
    )
    print("Training with subject-aware cross-validation...")
    classifier.fit(X_train, y_train, groups=subjects_train_ids, verbose=True)
    cv_results = classifier.get_cv_results()

    results.update({
        'cv_mean_accuracy': cv_results['mean_accuracy'],
        'cv_std_accuracy': cv_results['std_accuracy'],
        'train_accuracy': cv_results['train_accuracy'],
    })

    # Save classifier
    model_path = Path(output_dirs['models']) / f'brain_region_classifier_{diagonal_strategy}.pkl'
    classifier.save(str(model_path))
    print(f"Classifier saved to {model_path}")

    # Predictions on training data
    y_train_pred = classifier.predict(X_train)
    y_train_proba = classifier.predict_proba(X_train)
    error_map_train = calculate_error_map(y_train, y_train_pred, region_list)

    # =========================================================================
    # STEP 4: Apply to Task Data (PIOP-1)
    # =========================================================================
    print_section("STEP 4/6: Apply to Task Data (PIOP-1)")
    task_data_available = False
    y_test = y_test_pred = error_map_test = subjects_test_ids = None

    try:
        piop1_path = get_dataset_path(
            piop1_file,
            args.sample,
            "data/sample/sample_piop1_small.csv"
        )
        df_piop1 = load_connectivity_data(piop1_path)
        validate_schema(df_piop1)

        if not df_piop1.columns.equals(df_piop2.columns):
            raise ValueError("Column mismatch between PIOP-1 and PIOP-2!")

        print("Transforming task data using fitted preprocessor...")
        X_test = preprocessor.transform(df_piop1)
        y_test = preprocessor.get_labels()
        subjects_test = preprocessor.get_subjects()
        subject_ids_raw = df_piop1.iloc[:, 0].values
        subjects_test_ids = subject_ids_raw[subjects_test]

        y_test_pred = classifier.predict(X_test)
        y_test_proba = classifier.predict_proba(X_test)
        error_map_test = calculate_error_map(y_test, y_test_pred, region_list)

        results.update({
            'n_test_samples': len(X_test),
            'n_test_subjects': len(np.unique(subjects_test_ids)),
        })
        task_data_available = True
        print(f"Task prediction complete: {len(X_test)} samples, {len(np.unique(subjects_test_ids))} subjects")

    except FileNotFoundError as e:
        print(f"\nTask data not found: {e}")
        print("Skipping task-based analysis (Steps 4–6 partial).")
    except Exception as e:
        print(f"\nError processing task data: {e}")
        print("Continuing with rest-only analysis...")

    # =========================================================================
    # STEP 5: Save Predictions & Tables
    # =========================================================================
    print_section("STEP 5/6: Save Predictions and Results")
    processed_dir = Path(output_dirs['processed'])
    tables_dir = Path(output_dirs['tables'])

    # Training predictions
    train_df = pd.DataFrame({
        'subject_id': subjects_train_ids,
        'true_region': y_train,
        'predicted_region': y_train_pred,
        'correct': y_train == y_train_pred
    })
    train_df.to_csv(processed_dir / 'predictions_train.csv', index=False)

    if task_data_available:
        test_df = pd.DataFrame({
            'subject_id': subjects_test_ids,
            'true_region': y_test,
            'predicted_region': y_test_pred,
            'correct': y_test == y_test_pred
        })
        test_df.to_csv(processed_dir / 'predictions_task.csv', index=False)
        # Save probabilities
        proba_df = pd.DataFrame(y_test_proba, columns=region_list)
        proba_df.insert(0, 'subject_id', subjects_test_ids)
        proba_df.to_csv(processed_dir / 'prediction_probabilities_task.csv', index=False)

    # Error maps and confusion
    save_results_csv(error_map_train, tables_dir / 'error_map_rest.csv')
    save_confusion_matrix(y_train, y_train_pred, region_list, dataset_name="rest")

    if task_data_available:
        save_results_csv(error_map_test, tables_dir / 'error_map_task.csv')
        save_confusion_matrix(y_test, y_test_pred, region_list, dataset_name="task")
        comparison = compare_error_maps(error_map_train, error_map_test)
        save_results_csv(comparison, tables_dir / 'comparison_rest_vs_task.csv')

    # =========================================================================
    # STEP 6: Generate Figures
    # =========================================================================
    print_section("STEP 6/6: Generate Thesis Figures")
    figures_dir = Path(output_dirs['figures'])
    figure_count = 0

    plot_error_map(
        error_map_train,
        title='Resting-State Error Map (Training)',
        output_path=str(figures_dir / 'fig1_error_map_rest.png')
    )
    figure_count += 1

    if task_data_available:
        plot_error_map(
            error_map_test,
            title='Task Error Map (Gender Stroop)',
            output_path=str(figures_dir / 'fig3_error_map_task.png')
        )
        figure_count += 1

        plot_rest_vs_task_comparison(
            error_map_train, error_map_test, comparison,
            output_path=str(figures_dir / 'fig4_rest_vs_task_comparison.png')
        )
        figure_count += 1

    print(f"Generated {figure_count} figures in {figures_dir}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    chance = 1.0 / n_regions
    improvement = results['cv_mean_accuracy'] / chance

    print_section("PIPELINE COMPLETE!")
    summary = f"""
Summary of Results
{'='*50}
diagonal_strategy: {diagonal_strategy}

Training (Resting-State):
  Subjects: {results['n_train_subjects']:,}
  Samples : {results['n_train_samples']:,}
  Regions : {n_regions}

Model Performance (Strategy: {diagonal_strategy}):
  CV Accuracy : {results['cv_mean_accuracy']:.4f} ± {results['cv_std_accuracy']:.4f}
  Train Acc.  : {results['train_accuracy']:.4f}
  Chance Level: {chance:.4f} ({n_regions} classes)
  Improvement : {improvement:.1f}x above chance

Task Data (PIOP-1){' (SKIPPED)' if not task_data_available else ''}:
  Subjects: {results.get('n_test_subjects', 0):,}
  Samples : {results.get('n_test_samples', 0):,}
  Test Acc.  : {results.get('test_accuracy', 0.0):.4f}
{'='*50}

Output Directories:
  Models      → {output_dirs['models']}
  Tables      → {output_dirs['tables']}
  Figures     → {figures_dir}
  Processed   → {processed_dir}

Execution Time: {format_time(elapsed)}
Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    print(summary)

    # Final provenance
    log_provenance(tables_dir, config, results)

    print("Pipeline executed successfully!\n")
    return 0


# ======================================================================
# Entry Point
# ======================================================================
if __name__ == '__main__':
    sys.exit(main())
