"""
Brain Region Classifier (Flexible Preprocessing)
================================================
Supports both StandardScaler and Fisher Z-transform via parameter.
Uses sklearn Pipeline for reproducibility.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from pathlib import Path
from typing import Tuple, Dict, Union, Literal
import warnings

# Custom Fisher Z transformer (safe for pipelines)
class FisherZTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_value: float = 0.99999999999):
        self.clip_value = clip_value

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X_clipped = np.clip(X, -self.clip_value, self.clip_value)
        return 0.5 * np.log((1 + X_clipped) / (1 - X_clipped))

    def fit_transform(self, X, y=None):
        return self.transform(X)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    C: float = 0.01,
    max_iter: int = 1000,
    n_splits: int = 5,
    random_state: int = 42,
    scaler_type: Literal["standard", "fisher_z"] = "fisher_z"
) -> Tuple[Pipeline, Dict]:
    """
    Train brain region classifier with cross-validation and flexible scaling.

    Args:
        X: Raw correlation matrices (n_samples × n_features), values in [-1, 1]
        y: Labels (region indices)
        subjects: Subject IDs for GroupKFold
        C, max_iter, etc.
        scaler_type: 'standard' or 'fisher_z'

    Returns:
        pipeline: Fitted sklearn Pipeline (includes preprocessing + model)
        cv_results: Dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING BRAIN REGION CLASSIFIER ({scaler_type.upper()})")
    print(f"{'='*60}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Subjects: {len(np.unique(subjects))}")

    # Define preprocessing
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "fisher_z":
        scaler = FisherZTransformer()
    else:
        raise ValueError("scaler_type must be 'standard' or 'fisher_z'")

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('clf', LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',
            penalty='l2',
            n_jobs=-1
        ))
    ])

    # Cross-validation
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    print(f"\nRunning {n_splits}-fold GroupKFold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subjects), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone pipeline to avoid state leakage across folds
        fold_pipe = Pipeline(pipeline.steps)  # deep copy
        fold_pipe.fit(X_train, y_train)
        
        y_pred = fold_pipe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_scores.append(acc)
        print(f"   Fold {fold}: {acc:.4f}")

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    print(f"\nCV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    # Final training on full data
    print(f"\nTraining final model on full dataset...")
    pipeline.fit(X, y)

    train_acc = accuracy_score(y, pipeline.predict(X))
    print(f"Full training accuracy: {train_acc:.4f}")
    print(f"{'='*60}\n")

    cv_results = {
        'fold_scores': fold_scores,
        'mean_accuracy': cv_mean,
        'std_accuracy': cv_std,
        'train_accuracy': train_acc,
        'scaler_type': scaler_type
    }

    return pipeline, cv_results


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Save entire pipeline (preprocessing + model)"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model+preprocessing saved: {filepath}")


def load_model(filepath: str) -> Pipeline:
    """Load full pipeline"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"Model+preprocessing loaded: {filepath}")
    return pipeline


def predict(pipeline: Pipeline, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on new data using the saved pipeline.
    X must be raw correlation values in [-1, 1].
    """
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    return y_pred, y_proba


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.uniform(-0.8, 0.8, (1000, 232))  # Realistic correlations
    y = np.random.randint(0, 232, 1000)
    subjects = np.repeat(np.arange(20), 50)

    # Try both scalers
    for scaler in ["fisher_z", "standard"]:
        print(f"\n\n{'#'*80}")
        pipe, results = train_classifier(
            X=X,
            y=y,
            subjects=subjects,
            scaler_type=scaler,
            n_splits=5
        )
        print(f"{scaler.upper()} -> CV: {results['mean_accuracy']:.4f}")

        # Save best one
        if scaler == "fisher_z":  # suppose this is best
            save_model(pipe, "models/brain_region_classifier_fisherz.pkl")

    # Later: inference
    print("\n" + "="*60)
    print("INFERENCE ON NEW DATA")
    print("="*60)
    model = load_model("models/brain_region_classifier_fisherz.pkl")
    
    X_new = np.random.uniform(-0.8, 0.8, (5, 232))
    y_pred, y_proba = predict(model, X_new)
    print("Predictions:", y_pred)
    print("Top-3 probabilities:", y_proba.max(axis=1))