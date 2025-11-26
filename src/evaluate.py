"""
Model Evaluation and Metrics
=============================
Calculate error maps, confusion matrices, and save results to CSV.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
from typing import Dict, List, Union


def calculate_error_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_regions: int
) -> np.ndarray:
    """
    Calculate per-region misclassification rates.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_regions: Number of regions
    
    Returns:
        Array of error rates (one per region)
    """
    error_rates = np.zeros(n_regions)
    
    for region_idx in range(n_regions):
        mask = (y_true == region_idx)
        if mask.any():
            region_true = y_true[mask]
            region_pred = y_pred[mask]
            acc = accuracy_score(region_true, region_pred)
            error_rates[region_idx] = 1.0 - acc
        else:
            # No samples for this region
            error_rates[region_idx] = 0.0
    
    return error_rates


def calculate_error_map_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_list: List[str]
) -> pd.DataFrame:
    """
    Calculate per-region misclassification rates with detailed info.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        region_list: List of region names
    
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


def save_results_csv(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """Save DataFrame to CSV, creating directories if needed."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath.name}")


# ===============================================================
# Confusion Matrices & Predictions
# ===============================================================

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_list: List[str],
    output_path: Union[str, Path]
) -> None:
    """
    Save raw and normalized confusion matrices.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        region_list: List of region names
        output_path: Path where to save the confusion matrix CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    labels = np.arange(len(region_list))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Save raw confusion matrix
    cm_df = pd.DataFrame(cm, index=region_list, columns=region_list)
    cm_df_with_index = cm_df.reset_index().rename(columns={"index": "True_Label"})
    cm_df_with_index.to_csv(output_path, index=False)
    
    print(f"✓ Saved confusion matrix: {output_path.name}")


def save_predictions_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_list: List[str],
    output_path: Union[str, Path]
) -> None:
    """Save table of true vs predicted labels for inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "y_true_index": y_true,
        "y_true_label": [region_list[i] for i in y_true],
        "y_pred_index": y_pred,
        "y_pred_label": [region_list[i] for i in y_pred]
    })

    df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions table: {output_path.name}")


def compare_error_maps(error_rest: np.ndarray, error_task: np.ndarray) -> pd.DataFrame:
    """
    Compare misclassification rates between rest and task.
    
    Args:
        error_rest: Error rates for rest condition (numpy array)
        error_task: Error rates for task condition (numpy array)
    
    Returns:
        DataFrame with error comparison per region
    """
    n_regions = len(error_rest)
    
    comparison = pd.DataFrame({
        'region_index': range(n_regions),
        'error_rate_rest': error_rest,
        'error_rate_task': error_task,
        'error_increase': error_task - error_rest
    })
    
    comparison = comparison.sort_values('error_increase', ascending=False).reset_index(drop=True)
    return comparison


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

    # --- Error maps ---
    error_map_train = calculate_error_map(y_true_train, y_pred_train, n_regions)
    print(f"\nAverage training error: {error_map_train.mean():.4f}")

    # --- Confusion matrices ---
    save_confusion_matrix(y_true_train, y_pred_train, region_list, "test_output/confusion_train.csv")
    save_confusion_matrix(y_true_test, y_pred_test, region_list, "test_output/confusion_test.csv")