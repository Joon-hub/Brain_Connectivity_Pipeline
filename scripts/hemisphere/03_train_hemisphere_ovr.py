"""
03_train_hemisphere_ovr.py

Train One-vs-Rest logistic regression on hemisphere-specific brain connectivity.
Each region gets a binary classifier (Region X vs All Others); predictions are
aggregated via argmax over OvR probability scores.

Design decisions  (mirroring 02_train_fC_one_vs_rest.py)
---------------------------------------------------------
1.  PAIRED T-TEST SUPPORT
    Task data is pre-processed once before the CV loop (clip + impute +
    Fisher Z) and scaled inside each fold using that fold's rest-fitted
    StandardScaler.  This produces one matched (rest_val, task) accuracy pair
    per fold -- the pairs required for a valid paired t-test.

2.  IDENTIFIABILITY METRIC: RECALL, NOT BINARY ACCURACY
    Per-region identifiability is measured as multiclass recall:
        recall(R) = correct_R / total_R
    Binary OvR accuracy is dominated by true negatives and must not be used
    for identifiability analyses.

3.  CONSISTENT OOF EVALUATION FOR BOTH STATES
    Rest OOF predictions accumulate held-out fold predictions in the usual
    way.  Task predictions are computed inside every fold (same fold scaler
    and models), then majority-voted across folds to produce one consensus
    prediction per task sample.

4.  CROSS-STATE REGIONAL ANALYSIS
    After the CV loop, compute_cross_state_region_analysis() reports:
      - Pearson r between rest and task per-region recall
      - % regions with task recall >= 0.75
      - % regions with rest->task drop > 0.10

5.  FINAL MODEL ARTEFACTS
    A single model trained on all rest data produces probability-based
    artefacts (per-region binary metrics, network metrics, confusion matrix).
    Its accuracy is a training-set sanity check only; the paired t-test uses
    the per-fold numbers in cv_summary.json.

Output files  (<hemisphere>_hemisphere/one_vs_rest/)
----------------------------------------------------
cv_predictions.npy
cv_probabilities.npy
cv_true_labels.npy
confusion_matrix.npy
rest_per_region_recall.csv
task_per_region_recall.csv
cross_state_region_stability.csv
cross_state_analysis.json
cv_summary.json                  includes fold_metrics[*].task_accuracy
overall_metrics.json

<hemisphere>_hemisphere/task_testing_one_vs_rest/
task_predictions.npy
task_probabilities.npy
task_true_labels.npy
task_confusion_matrix.npy
task_per_region_binary_metrics.csv
task_network_metrics.csv
task_testing_summary.json

Usage
-----
    python scripts/hemisphere/03_train_hemisphere_ovr.py \\
        --hemisphere both --tune_hyperparams --test_on_task
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mode as scipy_mode
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.hemisphere.hemisphere_utils import (
    load_hemisphere_data,
    prepare_classification_data,
)
from src.hemisphere.hemisphere_metrics import (
    compute_classification_metrics,
    create_confusion_matrix,
)


# ==============================================================================
# SETUP & CONFIGURATION
# ==============================================================================

def setup_logging(output_dir: Path, hemisphere: str) -> logging.Logger:
    log_file = output_dir / f"training_{hemisphere}_hemisphere_ovr.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hemisphere-specific One-vs-Rest logistic regression"
    )
    parser.add_argument(
        "--hemisphere", type=str, required=True,
        choices=["left", "right", "both"],
    )
    parser.add_argument(
        "--data_dir", type=Path,
        default=project_root / "data" / "processed" / "hemispheres",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=project_root / "data" / "results" / "hemisphere_analysis",
    )
    parser.add_argument("--n_folds",          type=int,   default=5)
    parser.add_argument("--random_state",     type=int,   default=42)
    parser.add_argument("--C",                type=float, default=1.0)
    parser.add_argument("--solver",           type=str,   default="liblinear")
    parser.add_argument("--tolerance",        type=float, default=1e-4)
    parser.add_argument("--diagonal_strategy", type=str,  default="random")
    parser.add_argument("--max_iter",         type=int,   default=1000)
    parser.add_argument("--n_jobs",           type=int,   default=-1)
    parser.add_argument("--save_models",      action="store_true")
    parser.add_argument("--verbose",          action="store_true")
    parser.add_argument("--sample",           type=int,   default=None)
    parser.add_argument("--tune_hyperparams", action="store_true")
    parser.add_argument("--optuna_trials",    type=int,   default=50)
    parser.add_argument("--test_on_task",     action="store_true")
    parser.add_argument("--optuna_n_jobs",    type=int,   default=None)
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.sample is not None and args.sample <= 0:
        raise ValueError(f"Sample size must be positive, got {args.sample}")
    if args.n_folds < 2:
        raise ValueError(f"Number of folds must be >= 2, got {args.n_folds}")


def get_optuna_n_jobs(args: argparse.Namespace) -> int:
    if args.optuna_n_jobs is not None:
        return args.optuna_n_jobs
    if args.n_jobs == -1:
        return min(os.cpu_count() or 1, 32)
    elif args.n_jobs > 0:
        return min(args.n_jobs, 32)
    return 1


# ==============================================================================
# DATA PREPROCESSING HELPERS
# ==============================================================================

def sample_first_n_subjects(
    data: dict, n_sample: int, logger: logging.Logger
) -> dict:
    total = data["n_subjects"]
    if n_sample > total:
        logger.warning(
            f"Requested {n_sample} exceeds {total}. Using all subjects."
        )
        return data
    if n_sample <= 0:
        raise ValueError(f"Sample size must be positive, got {n_sample}")
    logger.info(f"SAMPLING: Using first {n_sample}/{total} subjects")
    return {
        "connectivity": data["connectivity"][:n_sample],
        "subject_ids":  data["subject_ids"][:n_sample],
        "region_info":  data["region_info"],
        "hemisphere":   data["hemisphere"],
        "n_subjects":   n_sample,
        "n_regions":    data["n_regions"],
    }


def clip_off_diagonal(connectivity: np.ndarray, n_regions: int) -> np.ndarray:
    """Clip off-diagonal entries to [-0.999, 0.999] in-place and return."""
    off_mask = ~np.eye(n_regions, dtype=bool)
    for i in range(connectivity.shape[0]):
        connectivity[i][off_mask] = np.clip(connectivity[i][off_mask], -0.999, 0.999)
    return connectivity


def apply_diagonal_imputation(
    connectivity: np.ndarray,
    region_info: pd.DataFrame,
    strategy: str,
    logger: logging.Logger,
    seed: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    n_subjects, n_regions, _ = connectivity.shape
    imp = connectivity.copy()
    rng = np.random.default_rng(seed if seed is not None else 42)

    if strategy == "random":
        for i in range(n_subjects):
            off_mask = ~np.eye(n_regions, dtype=bool)
            off_vals = imp[i][off_mask]
            lo, hi   = off_vals.min(), off_vals.max()
            for j in range(n_regions):
                imp[i, j, j] = rng.uniform(lo, hi)
        if verbose:
            logger.info(f"  Random diagonal imputation (seed={seed})")
    else:
        raise ValueError(f"Unknown diagonal strategy: {strategy}")

    return imp


def apply_fisher_z_transformation(
    connectivity: np.ndarray,
    logger: logging.Logger,
    verbose: bool = True,
) -> np.ndarray:
    if verbose:
        logger.info("  Applying Fisher Z-transformation ...")
    c = np.arctanh(np.clip(connectivity, -0.999, 0.999))
    if np.any(np.isnan(c)):
        raise ValueError("NaN after Fisher Z transformation")
    if np.any(np.isinf(c)):
        raise ValueError("Inf after Fisher Z transformation")
    if verbose:
        logger.info(f"  Range after Fisher Z: [{c.min():.4f}, {c.max():.4f}]")
    return c


def preprocess_fold_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    logger: logging.Logger,
    verbose: bool = False,
) -> tuple:
    """
    Fit StandardScaler on fold training split only.

    Returns (X_train_scaled, X_test_scaled, fitted_scaler).
    The scaler is returned so it can be reused to scale task data with the
    same mean/variance -- this is what makes the paired t-test valid.
    """
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    for tag, arr in [("train", X_train_scaled), ("test", X_test_scaled)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"NaN/Inf in scaled {tag} data")
    return X_train_scaled, X_test_scaled, scaler


# ==============================================================================
# TASK DATA PRE-LOADING  (called once, before the CV loop)
# ==============================================================================

def preload_task_data(
    hemisphere: str,
    data_dir: Path,
    n_regions: int,
    region_info: pd.DataFrame,
    diagonal_strategy: str,
    random_state: int,
    sample: int | None,
    logger: logging.Logger,
) -> tuple:
    """
    Load, clip, impute and Fisher-Z transform task data exactly once.

    Returns
    -------
    X_task      : (n_rows, n_regions)  unscaled feature matrix
    y_task      : (n_rows,)            region label per row
    groups_task : (n_rows,)            subject ID per row

    Scaling is deferred to each CV fold so that the scaler fitted on that
    fold's rest-training split is applied to task data too.  This guarantees
    rest-val and task accuracies share the same normalisation within every
    fold -- the matched-pair assumption of the paired t-test.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"PRE-LOADING TASK DATA  [{hemisphere} hemisphere]"
                f"  (processed once, scaled per fold)")
    logger.info("=" * 80)

    task_data = load_hemisphere_data(
        data_dir=data_dir,
        hemisphere=hemisphere,
        dataset="task",
        return_matrix=True,
        validate=True,
    )
    if sample:
        task_data = sample_first_n_subjects(task_data, sample, logger)
    logger.info(f"  Task subjects: {task_data['n_subjects']}")

    task_conn = task_data["connectivity"].copy()
    task_conn = clip_off_diagonal(task_conn, n_regions)

    # Fixed seed distinct from every fold seed -- stable across reruns
    task_seed = random_state + 9_999
    logger.info(f"  Diagonal imputation: random  (seed={task_seed})")
    task_conn = apply_diagonal_imputation(
        task_conn, region_info, "random", logger,
        seed=task_seed, verbose=False,
    )
    task_conn = apply_fisher_z_transformation(task_conn, logger, verbose=False)

    X_task, y_task, groups_task = prepare_classification_data(
        task_conn, region_info, task_data["subject_ids"]
    )

    logger.info(f"  X_task : {X_task.shape}  (unscaled -- scaled per fold)")
    logger.info(f"  y_task : {y_task.shape}")
    return X_task, y_task, groups_task


# ==============================================================================
# ONE-VS-REST TRAINING
# ==============================================================================

def train_ovr_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_regions: int,
    C: float,
    max_iter: int,
    solver: str,
    penalty: str,
    tol: float,
    random_state: int,
    n_jobs: int,
    logger: logging.Logger,
    verbose: bool = False,
) -> list:
    models = []
    for region_idx in range(n_regions):
        y_bin = (y_train == region_idx).astype(int)
        base  = LogisticRegression(
            C=C, penalty=penalty, max_iter=max_iter, solver=solver,
            tol=tol, random_state=random_state, n_jobs=1, verbose=0,
        )
        clf = OneVsRestClassifier(base, n_jobs=1)
        clf.fit(X_train, y_bin)
        models.append(clf)
    return models


def predict_ovr_probabilities(
    models: list, X: np.ndarray, n_regions: int
) -> np.ndarray:
    probs = np.zeros((X.shape[0], n_regions))
    for idx, model in enumerate(models):
        probs[:, idx] = model.predict_proba(X)[:, 1]
    return probs


def aggregate_ovr_predictions(probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities, axis=1)


# ==============================================================================
# OPTUNA HYPERPARAMETER TUNING -- PER FOLD
# (seed changes per inner fold only, data preprocessed once before all trials)
# ==============================================================================

def optimize_hyperparameters_ovr_fold(
    X_train_unscaled,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_regions: int,
    n_trials: int,
    random_state: int,
    optuna_n_jobs: int,
    logger: logging.Logger,
    verbose: bool = False,
    original_connectivity=None,
    subject_ids=None,
    region_info=None,
    is_random: bool = False,
    fold_idx: int = 0,
) -> dict:
    """
    Per-fold Optuna optimisation.

    Key design: the inner split seed and data preprocessing (diagonal
    imputation + Fisher Z) are fixed once per fold, not per trial.  Only
    hyperparameters vary across trials.
    """
    logger.info(
        f"\n  Hyperparameter optimisation -- Fold {fold_idx + 1}"
        f"  ({n_trials} trials x {n_regions} classifiers)"
    )

    # Fixed seed for this fold's inner validation split
    inner_fold_seed = random_state + fold_idx * 100_000

    best_tracker = {"score": 0.0}

    # Create inner split ONCE before all trials
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                            random_state=inner_fold_seed)
    tr_idx, va_idx = next(gss.split(range(len(groups_train)),
                                    groups=groups_train))

    # For random diagonal: preprocess data ONCE before all trials
    if is_random:
        tr_subj = np.unique(groups_train[tr_idx])
        va_subj = np.unique(groups_train[va_idx])
        tr_mask = np.isin(subject_ids, tr_subj)
        va_mask = np.isin(subject_ids, va_subj)

        tr_conn = apply_diagonal_imputation(
            original_connectivity[tr_mask].copy(), region_info,
            "random", logger, seed=inner_fold_seed, verbose=False,
        )
        va_conn = apply_diagonal_imputation(
            original_connectivity[va_mask].copy(), region_info,
            "random", logger, seed=inner_fold_seed + 1_000, verbose=False,
        )
        tr_conn = apply_fisher_z_transformation(tr_conn, logger, verbose=False)
        va_conn = apply_fisher_z_transformation(va_conn, logger, verbose=False)

        X_tr_raw, y_tr, _ = prepare_classification_data(
            tr_conn, region_info, subject_ids[tr_mask]
        )
        X_va_raw, y_va, _ = prepare_classification_data(
            va_conn, region_info, subject_ids[va_mask]
        )
    else:
        X_tr_raw = X_train_unscaled[tr_idx]
        X_va_raw = X_train_unscaled[va_idx]
        y_tr     = y_train[tr_idx]
        y_va     = y_train[va_idx]

    def objective(trial: optuna.Trial) -> float:
        C        = trial.suggest_float("C", 0.001, 0.05, log=True)
        solver   = trial.suggest_categorical("solver", ["sag", "saga"])
        max_iter = trial.suggest_int("max_iter", 100, 2000)
        tol      = trial.suggest_float("tol", 1e-4, 1e-1, log=True)
        penalty  = "l2"

        # Scaling must happen per trial (hyperparams don't affect scaler,
        # but we keep it here to match production behaviour)
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr_raw)
        X_va = sc.transform(X_va_raw)

        mdls  = train_ovr_classifiers(
            X_tr, y_tr, n_regions, C=C, max_iter=max_iter,
            solver=solver, penalty=penalty, tol=tol,
            random_state=random_state, n_jobs=1,
            logger=logger, verbose=False,
        )
        probs = predict_ovr_probabilities(mdls, X_va, n_regions)
        score = accuracy_score(y_va, aggregate_ovr_predictions(probs))

        is_best = score > best_tracker["score"]
        if is_best:
            best_tracker["score"] = score
        if verbose:
            logger.info(
                f"    Trial {trial.number + 1}/{n_trials}: "
                f"C={C:.6f}, {solver} -> {score:.4f}"
                + ("  * best" if is_best else "")
            )
        return score

    optuna_start = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=inner_fold_seed),
    )
    study.optimize(
        objective, n_trials=n_trials,
        show_progress_bar=False, n_jobs=optuna_n_jobs,
    )
    elapsed = time.time() - optuna_start

    bp = study.best_params
    bp["_optuna_best_score"]   = float(study.best_value)
    bp["_optuna_time_seconds"] = float(elapsed)

    logger.info(
        f"  Best (Fold {fold_idx + 1}): "
        f"C={bp['C']:.6f}, solver={bp['solver']}, "
        f"tol={bp['tol']:.6f}, max_iter={bp['max_iter']}, "
        f"val_acc={study.best_value:.4f}  ({elapsed:.1f}s)"
    )
    return bp


# ==============================================================================
# METRICS
# ==============================================================================

def compute_classification_metrics_enhanced(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Overall accuracy + optional top-k accuracy."""
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "n_classes": int(len(np.unique(y_true))),
    }
    if y_proba is not None:
        for k in [3, 5, 10]:
            if k <= y_proba.shape[1]:
                top_k = np.argsort(y_proba, axis=1)[:, -k:]
                metrics[f"top_{k}_accuracy"] = float(
                    np.mean([y_true[i] in top_k[i] for i in range(len(y_true))])
                )
    return metrics


def compute_per_region_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-region recall -- the correct identifiability metric.

    For each region R:
        recall(R) = #{rows of class R predicted as R} / #{rows of class R}

    Why NOT binary OvR accuracy
    ---------------------------
    With 100 classes, a region with zero true positives still achieves ~99%
    binary OvR accuracy because the 99 true-negative rows dominate.  Binary
    accuracy is therefore useless for cross-region or cross-state comparisons.
    """
    records = []
    for region_idx in range(len(region_info)):
        mask      = (y_true == region_idx)
        n_samples = int(mask.sum())
        recall    = (
            float((y_pred[mask] == region_idx).mean())
            if n_samples > 0 else np.nan
        )
        records.append({
            "region_idx":  region_idx,
            "region_name": region_info.iloc[region_idx]["region_name"],
            "network":     region_info.iloc[region_idx]["network"],
            "hemisphere":  region_info.iloc[region_idx].get("hemisphere", "unknown"),
            "recall":      recall,
            "n_samples":   n_samples,
        })
    return pd.DataFrame(records)


def compute_per_region_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    region_info: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Full OvR binary metrics (precision, recall, F1, binary accuracy) per region.

    Used for final-model artefacts where probability scores are available.
    'binary_accuracy' here is dominated by true negatives -- use 'recall'
    for identifiability analyses.
    """
    per_region = []
    for region_idx in range(probabilities.shape[1]):
        y_tb = (y_true == region_idx).astype(int)
        y_pb = (probabilities[:, region_idx] >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_tb, y_pb, average="binary", zero_division=0
        )
        per_region.append({
            "region_idx":         region_idx,
            "region_name":        region_info.iloc[region_idx]["region_name"],
            "network":            region_info.iloc[region_idx]["network"],
            "hemisphere":         region_info.iloc[region_idx].get("hemisphere", "unknown"),
            "binary_accuracy":    float(accuracy_score(y_tb, y_pb)),
            "precision":          float(p),
            "recall":             float(r),
            "f1_score":           float(f1),
            "n_positive_samples": int(y_tb.sum()),
            "n_total_samples":    int(len(y_tb)),
        })
    return pd.DataFrame(per_region)


def compute_network_level_metrics(per_region_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-region metrics to network level.

    Works with both compute_per_region_recall() output (recall + n_samples)
    and compute_per_region_binary_metrics() output (adds precision, f1_score).
    """
    agg_dict: dict = {
        "mean_recall": ("recall", "mean"),
        "n_regions":   ("region_idx", "count"),
    }
    if "precision" in per_region_df.columns:
        agg_dict["mean_precision"] = ("precision", "mean")
    if "f1_score" in per_region_df.columns:
        agg_dict["mean_f1"] = ("f1_score", "mean")
    sample_col = (
        "n_samples" if "n_samples" in per_region_df.columns
        else "n_positive_samples"
    )
    agg_dict["total_samples"] = (sample_col, "sum")

    result = (
        per_region_df.groupby("network")
        .agg(**{k: pd.NamedAgg(column=c, aggfunc=fn)
                for k, (c, fn) in agg_dict.items()})
        .reset_index()
        .sort_values("mean_recall", ascending=False)
    )
    return result


# ==============================================================================
# CROSS-STATE REGIONAL ANALYSIS
# ==============================================================================

def compute_cross_state_region_analysis(
    rest_region_df: pd.DataFrame,
    task_region_df: pd.DataFrame,
    logger: logging.Logger,
    hemisphere: str = "",
) -> dict:
    """
    Compute cross-state stability statistics from per-region recall DataFrames.

    Both DataFrames must derive from OOF predictions accumulated across the
    same CV folds and the same fold scalers (guaranteed by the preload +
    per-fold scaling design).

    Returns
    -------
    dict with keys:
        pearson_r              Pearson r between rest and task recall
        pearson_p              Two-tailed p-value
        pct_regions_task_ge75  % regions with task recall >= 0.75
        pct_regions_drop_gt10  % regions where (rest - task) > 0.10
        mean_rest_recall
        mean_task_recall
        mean_drop
        n_regions_analysed
        region_detail          DataFrame: region_idx, region_name, network,
                               hemisphere, recall_rest, recall_task, drop
    """
    merged = (
        rest_region_df[
            ["region_idx", "region_name", "network", "hemisphere", "recall"]
        ]
        .merge(
            task_region_df[["region_idx", "recall"]],
            on="region_idx",
            suffixes=("_rest", "_task"),
        )
        .dropna(subset=["recall_rest", "recall_task"])
        .copy()
    )

    rest_recall    = merged["recall_rest"].values
    task_recall    = merged["recall_task"].values
    drop           = rest_recall - task_recall
    merged["drop"] = drop

    r, p_corr      = stats.pearsonr(rest_recall, task_recall)
    pct_ge75       = float((task_recall >= 0.75).mean() * 100)
    pct_drop_gt10  = float((drop > 0.10).mean() * 100)
    mean_rest      = float(rest_recall.mean())
    mean_task      = float(task_recall.mean())
    mean_drop_val  = float(drop.mean())

    tag = f"  [{hemisphere.upper()} hemisphere]" if hemisphere else ""
    logger.info("\n" + "=" * 80)
    logger.info(f"CROSS-STATE REGIONAL ANALYSIS{tag}")
    logger.info("=" * 80)
    logger.info(f"  Regions analysed                : {len(merged)}")
    logger.info(f"  Mean rest recall                : {mean_rest:.4f}")
    logger.info(f"  Mean task recall                : {mean_task:.4f}")
    logger.info(f"  Mean rest->task drop            : {mean_drop_val:.4f}")
    logger.info(f"  Rest-task Pearson r             : {r:.4f}  (p = {p_corr:.4f})")
    logger.info(f"  Regions with task recall >=75%  : {pct_ge75:.1f}%")
    logger.info(f"  Regions with drop >10%          : {pct_drop_gt10:.1f}%")
    logger.info("=" * 80)

    return {
        "pearson_r":             float(r),
        "pearson_p":             float(p_corr),
        "pct_regions_task_ge75": pct_ge75,
        "pct_regions_drop_gt10": pct_drop_gt10,
        "mean_rest_recall":      mean_rest,
        "mean_task_recall":      mean_task,
        "mean_drop":             mean_drop_val,
        "n_regions_analysed":    int(len(merged)),
        "region_detail":         merged,
    }


# ==============================================================================
# FINAL MODEL  (all rest -> full task artefacts)
# ==============================================================================

def train_final_model_and_evaluate_task(
    hemisphere: str,
    best_params: dict,
    random_state: int,
    n_jobs: int,
    diagonal_strategy: str,
    output_dir: Path,
    region_info: pd.DataFrame,
    n_regions: int,
    original_connectivity_rest: np.ndarray,
    subject_ids_rest: np.ndarray,
    X_task_unscaled: np.ndarray,
    y_task: np.ndarray,
    groups_task: np.ndarray,
    logger: logging.Logger,
    save_models: bool = False,
) -> dict:
    """
    Train one model on ALL rest data and evaluate on the full task set.

    Purpose
    -------
    Produces artefacts that require full-set probability scores: per-region
    binary metrics, network-level metrics, confusion matrix.

    Statistical note
    ----------------
    The per-fold task accuracies in cv_summary.json -> fold_metrics[*].task_accuracy
    are the correct inputs for the paired t-test, not the scalar here.
    """
    C         = best_params["C"]
    max_iter  = best_params["max_iter"]
    solver    = best_params["solver"]
    penalty   = best_params.get("penalty", "l2")
    tol       = best_params["tol"]

    logger.info("\n" + "=" * 80)
    logger.info(
        f"FINAL MODEL  [{hemisphere.upper()}]  --  ALL REST  ->  FULL TASK ARTEFACTS"
    )
    logger.info("=" * 80)
    logger.info(
        f"  C={C:.6f}  solver={solver}  penalty={penalty}  "
        f"tol={tol:.6f}  max_iter={max_iter}"
    )

    # ---- Prepare rest training data ------------------------------------
    rest_conn = original_connectivity_rest.copy()
    rest_conn = apply_diagonal_imputation(
        rest_conn, region_info, "random", logger, seed=random_state
    )
    rest_conn = apply_fisher_z_transformation(rest_conn, logger)
    X_rest, y_rest, groups_rest = prepare_classification_data(
        rest_conn, region_info, subject_ids_rest
    )

    # ---- Scale: fit on all rest, apply to task -------------------------
    scaler_final  = StandardScaler()
    X_rest_scaled = scaler_final.fit_transform(X_rest)
    X_task_scaled = scaler_final.transform(X_task_unscaled)

    # ---- Train ---------------------------------------------------------
    t0 = time.time()
    final_models = train_ovr_classifiers(
        X_rest_scaled, y_rest, n_regions,
        C=C, max_iter=max_iter, solver=solver, penalty=penalty, tol=tol,
        random_state=random_state, n_jobs=n_jobs,
        logger=logger, verbose=False,
    )
    train_time = time.time() - t0
    logger.info(f"  {n_regions} classifiers trained in {train_time:.2f}s")

    # ---- Evaluate on rest (sanity check -- not for statistics) ---------
    rest_probs   = predict_ovr_probabilities(final_models, X_rest_scaled, n_regions)
    rest_preds   = aggregate_ovr_predictions(rest_probs)
    rest_metrics = compute_classification_metrics_enhanced(y_rest, rest_preds, rest_probs)
    logger.info(
        f"\n  Rest training-set accuracy : {rest_metrics['accuracy']:.4f}"
        f"  [sanity check -- not used for t-test]"
    )

    # ---- Evaluate on task ----------------------------------------------
    task_probs   = predict_ovr_probabilities(final_models, X_task_scaled, n_regions)
    task_preds   = aggregate_ovr_predictions(task_probs)
    task_metrics = compute_classification_metrics_enhanced(y_task, task_preds, task_probs)

    task_per_region = compute_per_region_binary_metrics(
        y_true=y_task,
        probabilities=task_probs,
        region_info=region_info,
    )
    task_network   = compute_network_level_metrics(task_per_region)
    task_confusion = create_confusion_matrix(y_task, task_preds, n_regions)
    rest_confusion = create_confusion_matrix(y_rest, rest_preds, n_regions)

    logger.info(f"  Task test accuracy         : {task_metrics['accuracy']:.4f}")
    logger.info(
        f"  Accuracy drop              : "
        f"{rest_metrics['accuracy'] - task_metrics['accuracy']:.4f}"
    )
    for k in [3, 5, 10]:
        key = f"top_{k}_accuracy"
        if key in task_metrics:
            logger.info(f"  Task Top-{k}                : {task_metrics[key]:.4f}")

    # ---- Save ----------------------------------------------------------
    task_out = output_dir / f"{hemisphere}_hemisphere" / "task_testing_one_vs_rest"
    task_out.mkdir(parents=True, exist_ok=True)

    np.save(task_out / "rest_predictions.npy",     rest_preds)
    np.save(task_out / "rest_probabilities.npy",   rest_probs)
    np.save(task_out / "rest_true_labels.npy",     y_rest)
    np.save(task_out / "rest_confusion_matrix.npy", rest_confusion)
    np.save(task_out / "task_predictions.npy",     task_preds)
    np.save(task_out / "task_probabilities.npy",   task_probs)
    np.save(task_out / "task_true_labels.npy",     y_task)
    np.save(task_out / "task_confusion_matrix.npy", task_confusion)

    task_summary: dict = {
        "note": (
            "Single final model trained on all rest data. "
            "Per-fold task accuracies for the paired t-test are in "
            "cv_summary.json -> fold_metrics[*].task_accuracy."
        ),
        "hemisphere":            hemisphere,
        "diagonal_strategy":     diagonal_strategy,
        "rest_train_accuracy":   float(rest_metrics["accuracy"]),
        "task_test_accuracy":    float(task_metrics["accuracy"]),
        "accuracy_drop":         float(rest_metrics["accuracy"] - task_metrics["accuracy"]),
        "hyperparameters":       best_params,
        "n_rest_subjects":       int(len(np.unique(groups_rest))),
        "n_task_subjects":       int(len(np.unique(groups_task))),
        "n_regions":             int(n_regions),
        "training_time_seconds": float(train_time),
    }
    for k in ["top_3_accuracy", "top_5_accuracy", "top_10_accuracy"]:
        if k in rest_metrics:
            task_summary[f"rest_{k}"] = float(rest_metrics[k])
        if k in task_metrics:
            task_summary[f"task_{k}"] = float(task_metrics[k])

    with open(task_out / "task_testing_summary.json", "w") as f:
        json.dump(task_summary, f, indent=2)
    task_per_region.to_csv(task_out / "task_per_region_binary_metrics.csv", index=False)
    task_network.to_csv(task_out / "task_network_metrics.csv", index=False)

    if save_models:
        import pickle
        with open(task_out / "final_models.pkl", "wb") as f:
            pickle.dump(final_models, f)
        with open(task_out / "final_scaler.pkl", "wb") as f:
            pickle.dump(scaler_final, f)
        logger.info("  Saved final_models.pkl + final_scaler.pkl")

    logger.info(f"\n  Artefacts saved to: {task_out}")

    return {
        "task_metrics":    task_metrics,
        "task_per_region": task_per_region,
        "task_network":    task_network,
        "task_summary":    task_summary,
        "rest_metrics":    rest_metrics,
    }


# ==============================================================================
# MAIN TRAINING FUNCTION -- SINGLE HEMISPHERE
# ==============================================================================

def train_single_hemisphere(
    hemisphere: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    optuna_n_jobs: int,
) -> dict:
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING {hemisphere.upper()} HEMISPHERE")
    logger.info(f"{'=' * 80}")

    output_dir = args.output_dir / f"{hemisphere}_hemisphere" / "one_vs_rest"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Load and preprocess rest data
    # ------------------------------------------------------------------
    logger.info("Loading rest data ...")
    data = load_hemisphere_data(
        data_dir=args.data_dir,
        hemisphere=hemisphere,
        dataset="rest",
        return_matrix=True,
        validate=True,
    )
    if args.sample is not None:
        data = sample_first_n_subjects(data, args.sample, logger)

    original_connectivity = data["connectivity"].copy()
    subject_ids           = data["subject_ids"]
    region_info           = data["region_info"]
    n_subjects, n_regions, _ = original_connectivity.shape

    logger.info(f"Subjects: {n_subjects}, Regions: {n_regions}")
    logger.info(f"Diagonal strategy: {args.diagonal_strategy}")

    original_connectivity = clip_off_diagonal(original_connectivity, n_regions)

    is_random = args.diagonal_strategy == "random"

    # Flat label / group arrays used by GroupKFold
    _, y, groups = prepare_classification_data(
        original_connectivity, region_info, subject_ids
    )

    # For non-random strategies build the full feature matrix once
    if not is_random:
        connectivity = apply_diagonal_imputation(
            original_connectivity, region_info, args.diagonal_strategy, logger
        )
        connectivity = apply_fisher_z_transformation(connectivity, logger)
        X, _, _ = prepare_classification_data(connectivity, region_info, subject_ids)
    else:
        X = None

    # ------------------------------------------------------------------
    # 2.  Pre-load task data ONCE  (scaling deferred to each fold)
    # ------------------------------------------------------------------
    X_task_unscaled = y_task = groups_task = None
    if args.test_on_task:
        X_task_unscaled, y_task, groups_task = preload_task_data(
            hemisphere=hemisphere,
            data_dir=args.data_dir,
            n_regions=n_regions,
            region_info=region_info,
            diagonal_strategy=args.diagonal_strategy,
            random_state=args.random_state,
            sample=args.sample,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # 3.  Cross-validation loop
    # ------------------------------------------------------------------
    logger.info(f"\n{'=' * 80}")
    logger.info(f"CROSS-VALIDATION  ({args.n_folds} folds)")
    logger.info(f"{'=' * 80}")

    gkf = GroupKFold(n_splits=args.n_folds)

    # OOF accumulators -- rest
    oof_rest_preds : list = []
    oof_rest_probs : list = []
    oof_rest_true  : list = []

    # Task predictions per fold (same complete task set evaluated each fold)
    oof_task_preds_per_fold: list = []

    fold_metrics      : list       = []
    best_fold_params  : dict | None = None
    best_fold_val_acc : float       = 0.0
    best_fold_idx     : int         = -1

    cv_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X if X is not None else range(len(y)), y, groups=groups)
    ):
        logger.info(f"\nFold {fold_idx + 1}/{args.n_folds}")

        n_tr = len(np.unique(groups[train_idx]))
        n_va = len(np.unique(groups[test_idx]))
        logger.info(f"  Rest -- train: {n_tr} subjects,  val: {n_va} subjects")

        # ---- Fold-specific rest data -----------------------------------
        if not is_random:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train    = groups[train_idx]
        else:
            train_subjects = np.unique(groups[train_idx])
            test_subjects  = np.unique(groups[test_idx])
            tr_mask = np.isin(subject_ids, train_subjects)
            va_mask = np.isin(subject_ids, test_subjects)

            fold_seed = args.random_state + (fold_idx + 1) * 1_000
            logger.info(
                f"  Random imputation seeds: "
                f"rest-train={fold_seed},  rest-val={fold_seed + 1_000}"
            )
            tr_conn = apply_diagonal_imputation(
                original_connectivity[tr_mask].copy(), region_info,
                "random", logger, seed=fold_seed, verbose=False,
            )
            va_conn = apply_diagonal_imputation(
                original_connectivity[va_mask].copy(), region_info,
                "random", logger, seed=fold_seed + 1_000, verbose=False,
            )
            tr_conn = apply_fisher_z_transformation(tr_conn, logger, verbose=False)
            va_conn = apply_fisher_z_transformation(va_conn, logger, verbose=False)

            X_train, y_train, groups_train = prepare_classification_data(
                tr_conn, region_info, subject_ids[tr_mask]
            )
            X_test, y_test, _ = prepare_classification_data(
                va_conn, region_info, subject_ids[va_mask]
            )

        # ---- Hyperparameter tuning ------------------------------------
        if args.tune_hyperparams:
            fold_best_params = optimize_hyperparameters_ovr_fold(
                X_train_unscaled=None if is_random else X_train,
                y_train=y_train, groups_train=groups_train,
                n_regions=n_regions, n_trials=args.optuna_trials,
                random_state=args.random_state, optuna_n_jobs=optuna_n_jobs,
                logger=logger, verbose=args.verbose,
                original_connectivity=original_connectivity if is_random else None,
                subject_ids=subject_ids if is_random else None,
                region_info=region_info if is_random else None,
                is_random=is_random, fold_idx=fold_idx,
            )
        else:
            fold_best_params = {
                "C": args.C, "max_iter": args.max_iter,
                "solver": args.solver, "penalty": "l2", "tol": args.tolerance,
            }

        C       = fold_best_params["C"]
        max_it  = fold_best_params["max_iter"]
        solver  = fold_best_params["solver"]
        penalty = fold_best_params.get("penalty", "l2")
        tol     = fold_best_params["tol"]

        # ---- Scale: fit on rest-train, apply to rest-val AND task ------
        #
        # Core of the paired t-test design: rest-val and task are both
        # normalised with the scaler fitted only on this fold's rest-training
        # split, satisfying the matched-pair assumption.
        #
        X_train_sc, X_test_sc, fold_scaler = preprocess_fold_data(
            X_train, X_test, logger, args.verbose
        )
        X_task_fold_sc = (
            fold_scaler.transform(X_task_unscaled)
            if X_task_unscaled is not None else None
        )

        # ---- Train models ---------------------------------------------
        t0 = time.time()
        fold_models = train_ovr_classifiers(
            X_train_sc, y_train, n_regions,
            C=C, max_iter=max_it, solver=solver, penalty=penalty, tol=tol,
            random_state=args.random_state, n_jobs=args.n_jobs,
            logger=logger, verbose=False,
        )
        fold_train_time = time.time() - t0

        # ---- Evaluate: rest train (overfit check) ---------------------
        tr_probs       = predict_ovr_probabilities(fold_models, X_train_sc, n_regions)
        tr_preds       = aggregate_ovr_predictions(tr_probs)
        fold_train_acc = accuracy_score(y_train, tr_preds)

        # ---- Evaluate: rest val (OOF) ---------------------------------
        va_probs     = predict_ovr_probabilities(fold_models, X_test_sc, n_regions)
        va_preds     = aggregate_ovr_predictions(va_probs)
        fold_val_acc = accuracy_score(y_test, va_preds)

        oof_rest_preds.extend(va_preds.tolist())
        oof_rest_probs.append(va_probs)
        oof_rest_true.extend(y_test.tolist())

        # ---- Evaluate: task (THIS fold's scaler) ----------------------
        if X_task_fold_sc is not None:
            task_probs_fold = predict_ovr_probabilities(
                fold_models, X_task_fold_sc, n_regions
            )
            task_preds_fold = aggregate_ovr_predictions(task_probs_fold)
            fold_task_acc   = float(accuracy_score(y_task, task_preds_fold))
            oof_task_preds_per_fold.append(task_preds_fold)
        else:
            fold_task_acc = None

        # ---- Track best fold ------------------------------------------
        if fold_val_acc > best_fold_val_acc:
            best_fold_val_acc = fold_val_acc
            best_fold_params  = fold_best_params
            best_fold_idx     = fold_idx

        fold_metrics.append({
            "fold":           fold_idx + 1,
            "train_accuracy": float(fold_train_acc),
            "val_accuracy":   float(fold_val_acc),
            "task_accuracy":  fold_task_acc,   # matched pair for paired t-test
            "train_time":     float(fold_train_time),
            "hyperparameters": fold_best_params,
        })

        task_str = f"{fold_task_acc:.4f}" if fold_task_acc is not None else "N/A"
        logger.info(
            f"  Train: {fold_train_acc:.4f},  Val: {fold_val_acc:.4f},  "
            f"Task: {task_str}  <- paired t-test  ({fold_train_time:.1f}s)"
        )

    cv_total_time = time.time() - cv_start

    # ------------------------------------------------------------------
    # 4.  Aggregate fold metrics
    # ------------------------------------------------------------------
    val_accs   = [f["val_accuracy"]   for f in fold_metrics]
    train_accs = [f["train_accuracy"] for f in fold_metrics]
    task_accs  = [f["task_accuracy"]  for f in fold_metrics
                  if f["task_accuracy"] is not None]

    mean_val_acc   = float(np.mean(val_accs))
    mean_train_acc = float(np.mean(train_accs))
    gen_gap        = float(mean_train_acc - mean_val_acc)
    mean_task_acc  = float(np.mean(task_accs)) if task_accs else None
    mean_drop      = (
        float(mean_val_acc - mean_task_acc) if mean_task_acc is not None else None
    )

    logger.info(f"\nCV done in {cv_total_time:.1f}s  |  "
                f"best fold: {best_fold_idx + 1}  (val={best_fold_val_acc:.4f})")
    logger.info(f"  Mean train: {mean_train_acc:.4f}  "
                f"Mean val: {mean_val_acc:.4f}  "
                f"Gen gap: {gen_gap:.4f}")
    if mean_task_acc is not None:
        logger.info(f"  Mean task: {mean_task_acc:.4f}  "
                    f"Mean drop: {mean_drop:.4f}")
        logger.info(
            "  Per-fold pairs: "
            + "  ".join(
                f"[{v:.4f}|{t:.4f}]"
                for v, t in zip(val_accs, task_accs)
            )
        )

    # ------------------------------------------------------------------
    # 5.  OOF rest metrics and per-region recall
    # ------------------------------------------------------------------
    oof_rest_preds_arr = np.array(oof_rest_preds)
    oof_rest_probs_arr = np.vstack(oof_rest_probs)
    oof_rest_true_arr  = np.array(oof_rest_true)

    overall_metrics = compute_classification_metrics_enhanced(
        oof_rest_true_arr, oof_rest_preds_arr, oof_rest_probs_arr
    )
    confusion_mat = create_confusion_matrix(
        oof_rest_true_arr, oof_rest_preds_arr, n_regions
    )
    rest_region_recall_df = compute_per_region_recall(
        oof_rest_true_arr, oof_rest_preds_arr, region_info
    )

    logger.info(f"\nOOF accuracy: {overall_metrics['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # 6.  Cross-state regional analysis
    # ------------------------------------------------------------------
    cross_state_results   = None
    task_region_recall_df = None

    if oof_task_preds_per_fold:
        stacked          = np.stack(oof_task_preds_per_fold, axis=1)
        consensus_result = scipy_mode(stacked, axis=1, keepdims=False)
        consensus = np.asarray(
            consensus_result.mode
            if hasattr(consensus_result, "mode")
            else consensus_result[0]
        ).ravel()

        task_region_recall_df = compute_per_region_recall(
            y_task, consensus, region_info
        )
        cross_state_results = compute_cross_state_region_analysis(
            rest_region_recall_df, task_region_recall_df, logger,
            hemisphere=hemisphere,
        )

    # ------------------------------------------------------------------
    # 7.  Save CV outputs
    # ------------------------------------------------------------------
    np.save(output_dir / "cv_predictions.npy",   oof_rest_preds_arr)
    np.save(output_dir / "cv_probabilities.npy",  oof_rest_probs_arr)
    np.save(output_dir / "cv_true_labels.npy",   oof_rest_true_arr)
    np.save(output_dir / "confusion_matrix.npy",  confusion_mat)

    rest_region_recall_df.to_csv(
        output_dir / "rest_per_region_recall.csv", index=False
    )
    if task_region_recall_df is not None:
        task_region_recall_df.to_csv(
            output_dir / "task_per_region_recall.csv", index=False
        )

    cross_state_stats = None
    if cross_state_results is not None:
        cross_state_results["region_detail"].to_csv(
            output_dir / "cross_state_region_stability.csv", index=False
        )
        cross_state_stats = {
            k: v for k, v in cross_state_results.items() if k != "region_detail"
        }
        with open(output_dir / "cross_state_analysis.json", "w") as f:
            json.dump(cross_state_stats, f, indent=2)

    cv_summary: dict = {
        "hemisphere":                hemisphere,
        "diagonal_strategy":         args.diagonal_strategy,
        "best_fold_hyperparameters": best_fold_params,
        "best_fold_idx":             int(best_fold_idx + 1),
        "best_fold_val_accuracy":    float(best_fold_val_acc),
        "overall_metrics":           overall_metrics,
        "fold_metrics":              fold_metrics,
        "mean_train_accuracy":       mean_train_acc,
        "mean_val_accuracy":         mean_val_acc,
        "generalization_gap":        gen_gap,
        "mean_task_accuracy":        mean_task_acc,
        "mean_rest_task_drop":       mean_drop,
        "cv_time_seconds":           float(cv_total_time),
        "paired_ttest_note": (
            "fold_metrics[i].val_accuracy and fold_metrics[i].task_accuracy "
            "are matched pairs (same fold scaler). "
            "Use scipy.stats.ttest_rel(val_accs, task_accs) for the paired t-test."
        ),
        "output_files": {
            "rest_per_region_recall":    "rest_per_region_recall.csv",
            "task_per_region_recall":    "task_per_region_recall.csv",
            "cross_state_stability":     "cross_state_region_stability.csv",
            "cross_state_summary":       "cross_state_analysis.json",
        },
    }
    if cross_state_stats is not None:
        cv_summary["cross_state_analysis"] = cross_state_stats

    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)
    with open(output_dir / "overall_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    logger.info(f"Saved to: {output_dir}")

    # ------------------------------------------------------------------
    # 8.  Final model on all rest -> full task artefacts
    # ------------------------------------------------------------------
    task_results = None
    if args.test_on_task and X_task_unscaled is not None:
        try:
            task_results = train_final_model_and_evaluate_task(
                hemisphere=hemisphere,
                best_params=best_fold_params,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                diagonal_strategy=args.diagonal_strategy,
                output_dir=args.output_dir,
                region_info=region_info,
                n_regions=n_regions,
                original_connectivity_rest=original_connectivity,
                subject_ids_rest=subject_ids,
                X_task_unscaled=X_task_unscaled,
                y_task=y_task,
                groups_task=groups_task,
                logger=logger,
                save_models=args.save_models,
            )
        except Exception as exc:
            logger.error(f"Final model evaluation failed: {exc}", exc_info=True)

    return {
        "hemisphere":           hemisphere,
        "n_subjects":           n_subjects,
        "n_regions":            n_regions,
        "overall_metrics":      overall_metrics,
        "rest_region_recall":   rest_region_recall_df,
        "task_region_recall":   task_region_recall_df,
        "cross_state_results":  cross_state_results,
        "task_results":         task_results,
        "output_dir":           output_dir,
        "cv_summary":           cv_summary,
    }


# ==============================================================================
# HEMISPHERE COMPARISON
# ==============================================================================

def compare_hemispheres(
    left_results: dict,
    right_results: dict,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARING HEMISPHERES")
    logger.info(f"{'=' * 80}")

    comparison_dir = output_dir / "comparison_one_vs_rest"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    left_acc  = left_results["overall_metrics"]["accuracy"]
    right_acc = right_results["overall_metrics"]["accuracy"]

    logger.info(
        f"OOF accuracy -- Left: {left_acc:.4f},  Right: {right_acc:.4f},  "
        f"Diff: {abs(left_acc - right_acc):.4f}"
    )

    # Paired t-test on rest CV val accuracies
    left_val_accs  = [m["val_accuracy"] for m in left_results["cv_summary"]["fold_metrics"]]
    right_val_accs = [m["val_accuracy"] for m in right_results["cv_summary"]["fold_metrics"]]
    t_stat, p_val  = stats.ttest_rel(left_val_accs, right_val_accs)
    logger.info(f"Rest paired t-test: t={t_stat:.4f}, p={p_val:.4f}")

    summary: dict = {
        "model_type":       "one_vs_rest",
        "left_accuracy":    left_acc,
        "right_accuracy":   right_acc,
        "difference":       abs(left_acc - right_acc),
        "ttest_t_stat":     float(t_stat),
        "ttest_p_value":    float(p_val),
        "left_best_fold":   left_results["cv_summary"]["best_fold_idx"],
        "right_best_fold":  right_results["cv_summary"]["best_fold_idx"],
    }

    # Cross-state comparison
    for side, res in [("left", left_results), ("right", right_results)]:
        cs = res["cv_summary"].get("cross_state_analysis")
        if cs:
            summary[f"{side}_cross_state_r"]            = cs["pearson_r"]
            summary[f"{side}_cross_state_p"]            = cs["pearson_p"]
            summary[f"{side}_pct_task_ge75"]            = cs["pct_regions_task_ge75"]
            summary[f"{side}_pct_drop_gt10"]            = cs["pct_regions_drop_gt10"]
            summary[f"{side}_mean_rest_recall"]         = cs["mean_rest_recall"]
            summary[f"{side}_mean_task_recall"]         = cs["mean_task_recall"]
            summary[f"{side}_mean_drop"]                = cs["mean_drop"]

    # Task accuracy comparison
    left_task  = left_results.get("task_results")
    right_task = right_results.get("task_results")
    if left_task and right_task:
        l_ts = left_task["task_summary"]
        r_ts = right_task["task_summary"]
        summary["left_task_accuracy"]   = l_ts["task_test_accuracy"]
        summary["right_task_accuracy"]  = r_ts["task_test_accuracy"]
        summary["left_accuracy_drop"]   = l_ts["accuracy_drop"]
        summary["right_accuracy_drop"]  = r_ts["accuracy_drop"]
        logger.info(
            f"Task accuracy -- Left: {l_ts['task_test_accuracy']:.4f},  "
            f"Right: {r_ts['task_test_accuracy']:.4f}"
        )

        # Paired t-test on task fold accuracies
        left_task_accs  = [
            m["task_accuracy"] for m in left_results["cv_summary"]["fold_metrics"]
            if m["task_accuracy"] is not None
        ]
        right_task_accs = [
            m["task_accuracy"] for m in right_results["cv_summary"]["fold_metrics"]
            if m["task_accuracy"] is not None
        ]
        n_folds = len(left_val_accs)
        if len(left_task_accs) == len(right_task_accs) == n_folds:
            t_task, p_task = stats.ttest_rel(left_task_accs, right_task_accs)
            summary["task_ttest_t_stat"] = float(t_task)
            summary["task_ttest_p_value"] = float(p_task)
            logger.info(f"Task paired t-test: t={t_task:.4f}, p={p_task:.4f}")

    with open(comparison_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Comparison saved to: {comparison_dir}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main() -> None:
    args = parse_arguments()
    try:
        validate_arguments(args)
    except Exception as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger        = setup_logging(args.output_dir, args.hemisphere)
    optuna_n_jobs = get_optuna_n_jobs(args)

    logger.info("=" * 80)
    logger.info("HEMISPHERE CLASSIFICATION (ONE-VS-REST)")
    logger.info("=" * 80)
    for label, val in [
        ("Hemisphere",        args.hemisphere),
        ("Data dir",          args.data_dir),
        ("Output dir",        args.output_dir),
        ("Diagonal strategy", args.diagonal_strategy),
        ("N folds",           args.n_folds),
        ("Tune hyperparams",  args.tune_hyperparams),
        ("Test on task",      args.test_on_task),
        ("N jobs",            args.n_jobs),
        ("Optuna N jobs",     optuna_n_jobs),
    ]:
        logger.info(f"  {label:<20}: {val}")
    logger.info("=" * 80)

    if args.sample:
        logger.warning(f"TESTING MODE: {args.sample} subjects only")

    try:
        if args.hemisphere == "both":
            left_results  = train_single_hemisphere("left",  args, logger, optuna_n_jobs)
            right_results = train_single_hemisphere("right", args, logger, optuna_n_jobs)
            compare_hemispheres(left_results, right_results, args.output_dir, logger)
        else:
            results = train_single_hemisphere(
                args.hemisphere, args, logger, optuna_n_jobs
            )

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as exc:
        logger.error(f"\nFATAL ERROR: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()