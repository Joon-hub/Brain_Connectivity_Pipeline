"""
Brain Region Classifier with sklearn Pipeline Integration
==========================================================
Supports Logistic Regression only.
Uses sklearn Pipeline for proper preprocessing integration.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional


# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================
def create_pipeline(
    C: float = 0.01,
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """
    Create sklearn Pipeline with StandardScaler and Logistic Regression.
    """
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        penalty='l2',
        n_jobs=-1,
        multi_class='multinomial'
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    return pipeline

def train_pipeline(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_splits: int = 5,
    verbose: bool = True
) -> Tuple[Pipeline, Dict]:
    """
    Train pipeline with subject-wise cross-validation.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    if verbose:
        print(f"\nRunning {n_splits}-fold GroupKFold cross-validation (by subject)...")

    from sklearn.base import clone
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subjects), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        y_pred = fold_pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_scores.append(acc)

        if verbose:
            print(f"   Fold {fold}: Train={accuracy_score(y_train, fold_pipeline.predict(X_train)):.4f}, Val={acc:.4f}")

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    if verbose:
        print(f"\nCV Results: {cv_mean:.4f} ± {cv_std:.4f}")

    print("Training final model on full dataset...")
    pipeline.fit(X, y)
    train_acc = accuracy_score(y, pipeline.predict(X))

    if verbose:
        print(f"Final training accuracy: {train_acc:.4f}")

    cv_results = {
        'fold_scores': fold_scores,
        'mean_accuracy': cv_mean,
        'std_accuracy': cv_std,
        'train_accuracy': train_acc
    }
    return pipeline, cv_results

def predict_pipeline(pipeline: Pipeline, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    return y_pred, y_proba


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved: {filepath}")

def load_pipeline(filepath: str) -> Pipeline:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pipeline not found: {filepath}")
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"Pipeline loaded: {filepath}")
    return pipeline


# ============================================================================
# WRAPPER CLASS FOR PIPELINE
# ============================================================================
class BrainRegionClassifierPipeline:
    """
    Wrapper for brain region classification using Logistic Regression Pipeline.
    """
    def __init__(self, C: float = 0.01, max_iter: int = 1000, n_splits: int = 5, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.n_splits = n_splits
        self.random_state = random_state
        self.pipeline = None
        self.cv_results = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, verbose: bool = True):
        """
        Fit the classifier with GroupKFold CV.
        Args:
            X: Features
            y: Labels
            groups: Subject IDs (required!)
            verbose: Print progress
        """
        self.pipeline = create_pipeline(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.pipeline, self.cv_results = train_pipeline(
            self.pipeline,
            X, y,
            subjects=groups,
            n_splits=self.n_splits,
            verbose=verbose
        )
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }

    def save(self, filepath: str):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        save_pipeline(self.pipeline, filepath)

    def load(self, filepath: str):
        self.pipeline = load_pipeline(filepath)
        self.is_fitted = True

    def get_cv_results(self) -> Optional[Dict]:
        return self.cv_results

if __name__ == "__main__":
    np.random.seed(42)
    n_subjects = 10
    n_regions = 50
    n_samples = n_subjects * n_regions
    n_features = n_regions - 1

    X = np.random.randn(n_samples, n_features)
    y = np.repeat(np.arange(n_regions), n_subjects)
    subjects = np.tile(np.arange(n_subjects), n_regions)

    lr_classifier = BrainRegionClassifierPipeline(C=0.01, n_splits=3)
    lr_classifier.fit(X, y, groups=subjects)  # ← NOW WORKS!
    y_pred = lr_classifier.predict(X[:100])
    print(f"Sample predictions: {y_pred[:10]}")

    results = lr_classifier.get_cv_results()
    print(f"CV Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")