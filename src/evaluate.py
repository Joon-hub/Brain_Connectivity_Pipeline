"""
Model Evaluation and Metrics
=============================
Calculate error maps, confusion matrices, and save results to CSV.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
from typing import Dict, List


# ===============================================================
# Core Metrics
# ===============================================================

def calculate_error_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_list: List[str]
) -> pd.DataFrame:
    """
    Calculate per-region misclassification rates.

    Returns:
        DataFrame with columns: region_index, region_name,
        misclassification_rate, n_samples
    """
    n_regions = len(region_list)
    error_rates = np.zeros(n_regions)
    sample_counts = np.zeros(n_regions, dtype=int)

    for region_idx in range(n_regions):
        mask = (y_true == region_idx)
        if mask.any():
            region_true = y_true[mask]
            region_pred = y_pred[mask]
            acc = accuracy_score(region_true, region_pred)
            error_rates[region_idx] = 1.0 - acc
            sample_counts[region_idx] = mask.sum()

    df = pd.DataFrame({
        "region_index": range(n_regions),
        "region_name": region_list,
        "misclassification_rate": error_rates,
        "n_samples": sample_counts
    })

    df = df.sort_values("misclassification_rate", ascending=False).reset_index(drop=True)
    return df


def calculate_global_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "unknown"
) -> Dict:
    """Compute overall accuracy and improvement over random baseline."""
    acc = accuracy_score(y_true, y_pred)
    n_classes = len(np.unique(y_true))
    random_baseline = 1.0 / n_classes
    improvement = acc / random_baseline

    return {
        "dataset": dataset_name,
        "accuracy": acc,
        "n_samples": len(y_true),
        "n_classes": n_classes,
        "random_baseline": random_baseline,
        "improvement_over_random": improvement
    }


def save_results_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV, creating directories if needed."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")


# ===============================================================
# Confusion Matrices & Predictions
# ===============================================================

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_list: List[str],
    dataset_name: str
) -> None:
    """
    Save raw and normalized confusion matrices.

    Automatically ensures full class coverage.
    """
    base_dir = Path("reports/tables/confusion_matrix")
    base_dir.mkdir(parents=True, exist_ok=True)

    labels = np.arange(len(region_list))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Raw
    cm_df = pd.DataFrame(cm, index=region_list, columns=region_list)
    save_results_csv(cm_df.reset_index().rename(columns={"index": "True_Label"}),
                     base_dir / f"{dataset_name}_raw.csv")

    # Normalized
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    cm_norm_df = pd.DataFrame(cm_norm, index=region_list, columns=region_list)
    save_results_csv(cm_norm_df.reset_index().rename(columns={"index": "True_Label"}),
                     base_dir / f"{dataset_name}_normalized.csv")

    print(f"✓ Confusion matrices saved for {dataset_name} set (raw & normalized)")


def save_predictions_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
    region_list: List[str]
) -> None:
    """Save table of true vs predicted labels for inspection."""
    base_dir = Path("reports/tables/confusion_matrix")
    base_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "y_true_index": y_true,
        "y_true_label": [region_list[i] for i in y_true],
        "y_pred_index": y_pred,
        "y_pred_label": [region_list[i] for i in y_pred]
    })

    filepath = base_dir / f"{dataset_name}_predictions.csv"
    df.to_csv(filepath, index=False)
    print(f"✓ Saved predictions table: {filepath}")

def compare_error_maps(error_rest, error_task):
    """
    Compare misclassification rates between rest and task.
    Returns a DataFrame with error difference per region.
    """
    comparison = error_rest[['region_name', 'misclassification_rate']].copy()
    comparison = comparison.merge(
        error_task[['region_name', 'misclassification_rate']],
        on='region_name',
        suffixes=('_rest', '_task')
    )
    comparison['error_increase'] = comparison['misclassification_rate_task'] - comparison['misclassification_rate_rest']
    comparison = comparison.sort_values('error_increase', ascending=False).reset_index(drop=True)
    return comparison

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_list: List[str],
    dataset_name: str = "dataset"
) -> None:
    """ Save raw and normalized confusion matrices. Automatically ensures full class coverage. """ 
    base_dir = Path("reports/tables/confusion_matrix") 
    base_dir.mkdir(parents=True, exist_ok=True) 
    labels = np.arange(len(region_list)) 
    cm = confusion_matrix(y_true, y_pred, labels=labels) 
    
    # Raw 
    cm_df = pd.DataFrame(cm, index=region_list, columns=region_list) 
    save_results_csv(cm_df.reset_index().rename(columns={"index": "True_Label"}), base_dir / f"{dataset_name}_raw.csv") 
    
    # Normalized 
    with np.errstate(divide='ignore', invalid='ignore'): cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) 
    cm_norm = np.nan_to_num(cm_norm) 
    cm_norm_df = pd.DataFrame(cm_norm, index=region_list, columns=region_list) 
    save_results_csv(cm_norm_df.reset_index().rename(columns={"index": "True_Label"}), base_dir / f"{dataset_name}_normalized.csv") 
    print(f"✓ Confusion matrices saved for {dataset_name} set (raw & normalized)")


# ===============================================================
# Example Usage
# ===============================================================

if __name__ == "__main__" and False:
    np.random.seed(42)

    n_samples = 1000
    n_regions = 232
    region_list = [f"Region_{i}" for i in range(n_regions)]

    # --- Dummy training data ---
    y_true_train = np.random.randint(0, n_regions, n_samples)
    y_pred_train = y_true_train.copy()
    mask_train = np.random.rand(n_samples) < 0.2
    y_pred_train[mask_train] = np.random.randint(0, n_regions, mask_train.sum())

    # --- Dummy test data ---
    y_true_test = np.random.randint(0, n_regions, n_samples)
    y_pred_test = y_true_test.copy()
    mask_test = np.random.rand(n_samples) < 0.3
    y_pred_test[mask_test] = np.random.randint(0, n_regions, mask_test.sum())

    # --- Metrics ---
    train_metrics = calculate_global_metrics(y_true_train, y_pred_train, "train")
    test_metrics = calculate_global_metrics(y_true_test, y_pred_test, "test")
    print(f"\nTrain Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")

    # --- Confusion matrices ---
    save_confusion_matrix(y_true_train, y_pred_train, region_list, "train")
    save_confusion_matrix(y_true_test, y_pred_test, region_list, "test")
