#!/usr/bin/env python3
"""
Brain Connectivity Classification Pipeline
===========================================
Main entry point for the complete analysis pipeline.

Usage:
    python run.py --config config.yaml         # Full pipeline
    python run.py --sample                     # Quick test with sample data
    python run.py --config config.yaml --help  # Show options

Steps:
    1. Load resting-state data (PIOP-2)
    2. Extract features and train classifier
    3. Cross-validate and save model
    4. Load task data (PIOP-1) and apply model
    5. Create error maps, confusion matrices, and comparisons
    6. Generate visualizations and reports
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# ======================================================================
# Import project modules
# ======================================================================
from data import (
    load_connectivity_data, extract_connection_columns, extract_subjects,
    create_sample_dataset, validate_schema
)
from features import extract_regions, create_classification_dataset
from model import train_classifier, predict, save_model, load_model
from evaluate import (
    calculate_error_map,
    calculate_global_metrics,
    save_results_csv,
    save_confusion_matrix,
    compare_error_maps,
)
from visualize import (
    plot_error_map,
    plot_rest_vs_task_comparison,
    plot_network_analysis
)
from utils import (
    setup_logging, set_random_seeds, load_config, log_provenance,
    print_section, format_time
)

# ======================================================================
# Main Pipeline
# ======================================================================

def main():
    """Run the complete brain connectivity classification pipeline."""

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
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Setup and logging
    # ---------------------------------------------------------------
    start_time = time.time()
    setup_logging(args.log)

    print_section("BRAIN CONNECTIVITY CLASSIFICATION PIPELINE")
    print(f"\nMode: {'SAMPLE DATA (quick test)' if args.sample else 'FULL DATA'}")
    print(f"Config: {args.config}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load configuration
    config = load_config(args.config)
    set_random_seeds(config.get('random_seed', 42))

    # Extract settings
    piop2_file = config['data']['piop2_file']
    piop1_file = config['data']['piop1_file']
    diagonal_strategy = config.get('diagonal_strategy', 'network_mean')
    n_splits = config.get('n_splits', 5)
    C = config.get('C', 0.01)

    results = {}

    # =========================================================================
    # STEP 1: Load Resting-State Data (PIOP-2)
    # =========================================================================
    print_section("STEP 1/6: Load Resting-State Data")

    if args.sample:
        create_sample_dataset(piop2_file, "data/sample/sample_piop2_small.csv", n_subjects=10)
        df_piop2 = load_connectivity_data("data/sample/sample_piop2_small.csv")
    else:
        df_piop2 = load_connectivity_data(piop2_file)

    connection_columns = extract_connection_columns(df_piop2)

    # =========================================================================
    # STEP 2: Extract Features
    # =========================================================================
    print_section("STEP 2/6: Extract Features and Create Dataset")

    region_list, region_to_idx, n_regions = extract_regions(connection_columns)

    X_train, y_train, subjects_train, region_list = create_classification_dataset(
        df_piop2, connection_columns, diagonal_strategy=diagonal_strategy
    )

    results['n_regions'] = n_regions
    results['n_train_samples'] = len(X_train)
    results['n_train_subjects'] = len(np.unique(subjects_train))

    # =========================================================================
    # STEP 3: Train Classifier with Cross-Validation
    # =========================================================================
    print_section("STEP 3/6: Train Brain Region Classifier")

    model, scaler, cv_results = train_classifier(
        X_train, y_train, subjects_train, n_splits=n_splits, C=C
    )

    results['cv_mean_accuracy'] = cv_results['mean_accuracy']
    results['cv_std_accuracy'] = cv_results['std_accuracy']
    results['train_accuracy'] = cv_results['train_accuracy']

    # Save trained model
    model_path = Path(config['output_dirs']['models']) / 'trained_model.pkl'
    save_model(model, scaler, str(model_path))

    # Predict on training data
    y_train_pred, _ = predict(model, scaler, X_train)

    # Training error map
    error_map_train = calculate_error_map(y_train, y_train_pred, region_list)

    # =========================================================================
    # STEP 4: Apply to Task Data (PIOP-1)
    # =========================================================================
    print_section("STEP 4/6: Apply Classifier to Task Data")

    task_data_available = False
    y_test = y_test_pred = None
    error_map_test = None

    try:
        if args.sample:
            create_sample_dataset(piop1_file, "data/sample/sample_piop1_small.csv", n_subjects=10)
            df_piop1 = load_connectivity_data("data/sample/sample_piop1_small.csv")
        else:
            df_piop1 = load_connectivity_data(piop1_file)

        validate_schema(df_piop1)
        if not df_piop1.columns.equals(df_piop2.columns):
            raise ValueError("Task dataset schema does not match resting-state dataset.")

        X_test, y_test, subjects_test, region_list = create_classification_dataset(
            df_piop1, connection_columns, diagonal_strategy=diagonal_strategy
        )

        y_test_pred, _ = predict(model, scaler, X_test)

        error_map_test = calculate_error_map(y_test, y_test_pred, region_list)

        results['n_test_samples'] = len(X_test)
        results['n_test_subjects'] = len(np.unique(subjects_test))

        task_data_available = True

    except FileNotFoundError as e:
        print(f"\n⚠ Task data not found: {e}")
        print("  Skipping task analysis (Step 4–5 will be partial)")

    # =========================================================================
    # STEP 5: Create Error Maps, Confusion Matrices, and Comparisons
    # =========================================================================
    print_section("STEP 5/6: Generate Error Maps and Statistics")

    tables_dir = Path(config['output_dirs']['tables'])
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save training error map
    save_results_csv(error_map_train, tables_dir / 'error_map_rest.csv')

    # Save confusion matrices
    save_confusion_matrix(
    y_train, y_train_pred, region_list, dataset_name="train"
    )
    if task_data_available:
        save_confusion_matrix(
            y_test, y_test_pred, region_list, dataset_name="task"
        )

    # Skip network-level aggregation (advanced analysis handled separately)
    print("⚠ Skipping network-level aggregation (user-defined advanced analysis).")
    network_stats = None

    # Compute and save rest vs task comparison (if applicable)
    if task_data_available:
        comparison = compare_error_maps(error_map_train, error_map_test)
        save_results_csv(comparison, tables_dir / 'comparison_rest_vs_task.csv')

    print(f"\n✓ Generated CSV outputs in {tables_dir}")

    # =========================================================================
    # STEP 6: Create Visualizations
    # =========================================================================
    print_section("STEP 6/6: Generate Thesis Figures")

    figures_dir = Path(config['output_dirs']['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Training error map
    plot_error_map(
        error_map_train,
        title='Resting-State Error Map (Training)',
        output_path=str(figures_dir / 'fig1_error_map_rest.png')
    )

    # Figure 2: Skip network-level figure if not aggregated
    if network_stats is not None:
        plot_network_analysis(
            error_map_train,
            network_stats,
            output_path=str(figures_dir / 'fig2_network_analysis_rest.png')
        )
    else:
        print("⚠ Skipping network-level visualization (user-defined advanced analysis).")

    # Figure 3 & 4: Only if task data available
    if task_data_available:
        plot_error_map(
            error_map_test,
            title='Task Error Map (Gender Stroop)',
            output_path=str(figures_dir / 'fig3_error_map_task.png')
        )

        comparison = compare_error_maps(error_map_train, error_map_test)
        plot_rest_vs_task_comparison(
            error_map_train,
            error_map_test,
            comparison,
            output_path=str(figures_dir / 'fig4_rest_vs_task_comparison.png')
        )

        print(f"\n✓ Generated 4 thesis figures in {figures_dir}")
    else:
        print(f"\n✓ Generated 2 thesis figures in {figures_dir} (task analysis skipped)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed_time = time.time() - start_time

    print_section("PIPELINE COMPLETE!")

    print(f"""
Summary of Results:
==================

Training Data:
  Subjects: {results['n_train_subjects']}
  Regions: {results['n_regions']}
  Samples: {results['n_train_samples']}

Model Performance:
  CV Accuracy: {results['cv_mean_accuracy']:.4f} ± {results['cv_std_accuracy']:.4f}
  Train Accuracy: {results['train_accuracy']:.4f}
  Random Baseline: {1/results['n_regions']:.4f} ({results['n_regions']} classes)
  Improvement: {results['cv_mean_accuracy']/(1/results['n_regions']):.1f}x better than chance
""")

    if task_data_available:
        print(f"""Task Data:
  Subjects: {results['n_test_subjects']}
  Samples: {results['n_test_samples']}
""")

    print(f"""
Output Files:
  Model: {model_path}
  Tables: {tables_dir}/
  Figures: {figures_dir}/

Execution Time: {format_time(elapsed_time)}
Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}
""")

    # Log provenance for reproducibility
    log_provenance(
        config['output_dirs']['tables'],
        config,
        results
    )

    print("✅ All done! Results are ready for your thesis.\n")
    return 0


# ======================================================================
# Entry Point
# ======================================================================
if __name__ == '__main__':
    sys.exit(main())
