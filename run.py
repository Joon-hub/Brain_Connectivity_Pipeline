#!/usr/bin/env python3
"""
Brain Connectivity Classification Pipeline
===========================================
Main entry point for the complete analysis pipeline.
Usage:
    python run.py --config config.yaml           # Full pipeline
    python run.py --sample                        # Quick test with sample data
    python run.py --config config.yaml --help     # Show options
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
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
# CHANGED: Use new features.py instead of old_features
from features import BrainConnectivityPreprocessor
# CHANGED: Use new model.py instead of old_model
from model import BrainRegionClassifierPipeline
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

# Main Pipeline
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
    diagonal_strategy = config.get('diagonal_strategy', 'region_mean')  # CHANGED: default to region_mean
    n_splits = config.get('n_splits', 5)
    C = config.get('C', 0.01)
    max_iter = config.get('max_iter', 1000)  # NEW: max iterations for LogisticRegression
    random_state = config.get('random_seed', 42)

    results = {}

    # =========================================================================
    # STEP 1: Load Resting-State Data (PIOP-2)
    # =========================================================================
    print_section("STEP 1/6: Load Resting-State Data (PIOP-2)")
    if args.sample:
        sample_path = "data/sample/sample_piop2_small.csv"
        create_sample_dataset(piop2_file, sample_path, n_subjects=10)
        df_piop2 = load_connectivity_data(sample_path)
    else:
        df_piop2 = load_connectivity_data(piop2_file)

    connection_columns = extract_connection_columns(df_piop2)
    print(f"Loaded {len(df_piop2)} samples, {len(connection_columns)} connections")

    # =========================================================================
    # STEP 2: Extract Features & Create Dataset (NEW APPROACH)
    # =========================================================================
    print_section("STEP 2/6: Extract Features and Create Dataset")
    
    # CHANGED: Use new BrainConnectivityPreprocessor
    preprocessor = BrainConnectivityPreprocessor(
        connection_columns=connection_columns,
        diagonal_strategy=diagonal_strategy
    )
    
    # Fit and transform training data
    print("Fitting preprocessor on training data...")
    preprocessor.fit(df_piop2)
    X_train = preprocessor.transform(df_piop2)
    y_train = preprocessor.get_labels()
    subjects_train = preprocessor.get_subjects()
    
    # Get region information
    region_list = preprocessor.region_list_
    n_regions = preprocessor.n_regions_
    
    # Convert subject indices to actual subject IDs
    subject_ids = df_piop2.iloc[:, 0].values
    subjects_train_ids = np.array([subject_ids[idx] for idx in subjects_train])

    results['n_regions'] = n_regions
    results['n_train_samples'] = len(X_train)
    results['n_train_subjects'] = len(np.unique(subjects_train_ids))

    # Save region list
    region_list_path = Path(config['output_dirs']['processed']) / 'region_list.csv'
    region_list_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Region": region_list}).to_csv(region_list_path, index=False)
    print(f"Region list saved to {region_list_path}")

    # =========================================================================
    # STEP 3: Train Classifier with Cross-Validation (NEW APPROACH)
    # =========================================================================
    print_section("STEP 3/6: Train Brain Region Classifier")
    
    # CHANGED: Use new BrainRegionClassifierPipeline
    classifier = BrainRegionClassifierPipeline(
        C=C,
        max_iter=max_iter,
        n_splits=n_splits,
        random_state=random_state
    )
    
    print("Training classifier with cross-validation...")
    classifier.fit(X_train, y_train, subjects_train_ids, verbose=True)
    cv_results = classifier.get_cv_results()

    results.update({
        'cv_mean_accuracy': cv_results['mean_accuracy'],
        'cv_std_accuracy': cv_results['std_accuracy'],
        'train_accuracy': cv_results['train_accuracy'],
        'diagonal_strategy': diagonal_strategy
    })

    # Save model
    model_path = Path(config['output_dirs']['models']) / f'brain_region_classifier_{diagonal_strategy}.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(str(model_path))
    print(f"Classifier saved to {model_path}")

    # Predict on training data
    y_train_pred = classifier.predict(X_train)
    y_train_proba = classifier.predict_proba(X_train)
    error_map_train = calculate_error_map(y_train, y_train_pred, region_list)

    # =========================================================================
    # STEP 4: Apply to Task Data (PIOP-1)
    # =========================================================================
    print_section("STEP 4/6: Apply Classifier to Task Data (PIOP-1)")
    task_data_available = False
    y_test = y_test_pred = error_map_test = None

    try:
        if args.sample:
            sample_task_path = "data/sample/sample_piop1_small.csv"
            create_sample_dataset(piop1_file, sample_task_path, n_subjects=10)
            df_piop1 = load_connectivity_data(sample_task_path)
        else:
            df_piop1 = load_connectivity_data(piop1_file)

        validate_schema(df_piop1)
        if not df_piop1.columns.equals(df_piop2.columns):
            raise ValueError("Task and rest datasets have mismatched columns!")

        # CHANGED: Use fitted preprocessor to transform test data
        print("Transforming test data with fitted preprocessor...")
        X_test = preprocessor.transform(df_piop1)
        y_test = preprocessor.get_labels()
        subjects_test = preprocessor.get_subjects()
        
        # Convert subject indices to actual subject IDs
        subject_ids_test = df_piop1.iloc[:, 0].values
        subjects_test_ids = np.array([subject_ids_test[idx] for idx in subjects_test])

        # Predict on test data
        y_test_pred = classifier.predict(X_test)
        y_test_proba = classifier.predict_proba(X_test)
        error_map_test = calculate_error_map(y_test, y_test_pred, region_list)

        results['n_test_samples'] = len(X_test)
        results['n_test_subjects'] = len(np.unique(subjects_test_ids))
        task_data_available = True

        print(f"Task prediction complete: {len(X_test)} samples, {len(subjects_test_ids)} subjects")

    except FileNotFoundError as e:
        print(f"\nTask data not found: {e}")
        print("Skipping task-based analysis (Steps 4–6 partial).")

    # Save predictions
    processed_dir = Path(config['output_dirs']['processed'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(y_train_pred, columns=['predicted_region']).to_csv(processed_dir / 'y_train_pred.csv', index=False)
    pd.DataFrame(y_train, columns=['true_region']).to_csv(processed_dir / 'y_train.csv', index=False)
    if task_data_available:
        pd.DataFrame(y_test_pred, columns=['predicted_region']).to_csv(processed_dir / 'y_test_pred.csv', index=False)
        pd.DataFrame(y_test, columns=['true_region']).to_csv(processed_dir / 'y_test.csv', index=False)
    print(f"Predictions saved to {processed_dir}")

    # =========================================================================
    # STEP 5: Generate Error Maps, Confusion Matrices, and Comparisons
    # =========================================================================
    print_section("STEP 5/6: Generate Error Maps and Statistics")
    tables_dir = Path(config['output_dirs']['tables'])
    tables_dir.mkdir(parents=True, exist_ok=True)

    save_results_csv(error_map_train, tables_dir / 'error_map_rest.csv')
    save_confusion_matrix(y_train, y_train_pred, region_list, dataset_name="rest", output_dir=tables_dir)

    if task_data_available:
        save_results_csv(error_map_test, tables_dir / 'error_map_task.csv')
        save_confusion_matrix(y_test, y_test_pred, region_list, dataset_name="task", output_dir=tables_dir)
        comparison = compare_error_maps(error_map_train, error_map_test)
        save_results_csv(comparison, tables_dir / 'comparison_rest_vs_task.csv')

    print(f"Tables and matrices saved to {tables_dir}")

    # =========================================================================
    # STEP 6: Create Visualizations
    # =========================================================================
    print_section("STEP 6/6: Generate Thesis Figures")
    figures_dir = Path(config['output_dirs']['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_error_map(
        error_map_train,
        title='Resting-State Error Map (Training)',
        output_path=str(figures_dir / 'fig1_error_map_rest.png')
    )

    if task_data_available:
        plot_error_map(
            error_map_test,
            title='Task Error Map (Gender Stroop)',
            output_path=str(figures_dir / 'fig3_error_map_task.png')
        )
        comparison = compare_error_maps(error_map_train, error_map_test)
        plot_rest_vs_task_comparison(
            error_map_train, error_map_test, comparison,
            output_path=str(figures_dir / 'fig4_rest_vs_task_comparison.png')
        )
        print(f"Generated 4 figures in {figures_dir}")
    else:
        print(f"Generated 2 figures in {figures_dir} (task data skipped)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed_time = time.time() - start_time
    print_section("PIPELINE COMPLETE!")
    chance = 1 / results['n_regions']
    improvement = results['cv_mean_accuracy'] / chance

    summary = f"""
Summary of Results:
==================
Training Data:
  Subjects: {results['n_train_subjects']}
  Regions: {results['n_regions']}
  Samples: {results['n_train_samples']}

Model Performance (Diagonal Strategy: {diagonal_strategy}):
  CV Accuracy:  {results['cv_mean_accuracy']:.4f} ± {results['cv_std_accuracy']:.4f}
  Train Accuracy: {results['train_accuracy']:.4f}
  Chance Level:  {chance:.4f} ({results['n_regions']} classes)
  Improvement:   {improvement:.1f}x above chance

Output Files:
  Model:         {model_path}
  Tables:        {tables_dir}/
  Figures:       {figures_dir}/
  Processed:     {processed_dir}/

Execution Time: {format_time(elapsed_time)}
Completed:      {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    print(summary)

    if task_data_available:
        print(f"Task Data Applied:")
        print(f"  Subjects: {results['n_test_subjects']}")
        print(f"  Samples:  {results['n_test_samples']}")

    # Log provenance
    log_provenance(config['output_dirs']['tables'], config, results)

    print("Pipeline executed successfully!\n")
    return 0


# ======================================================================
# Entry Point
# ======================================================================
if __name__ == '__main__':
    sys.exit(main())