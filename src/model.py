"""
Brain Region Classifier
=======================
Simple logistic regression classifier for brain regions.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
from typing import Tuple, Dict


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    C: float = 0.01,
    max_iter: int = 1000,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[LogisticRegression, StandardScaler, Dict]:
    """
    Train brain region classifier with cross-validation.
    
    Args:
        X: Features (n_samples × n_features)
        y: Labels (region indices)
        subjects: Subject IDs for group splitting
        C: Regularization strength
        max_iter: Maximum iterations
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        model: Trained classifier
        scaler: Fitted feature scaler
        cv_results: Cross-validation metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING BRAIN REGION CLASSIFIER")
    print(f"{'='*60}")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Subjects: {len(np.unique(subjects))}")
    
    # Cross-validation (subject-wise splitting)
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    
    print(f"\nRunning {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subjects), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train)
        X_val_scaled = scaler_fold.transform(X_val)
        
        # Train
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',
            penalty='l2',
            n_jobs=-1
        )
        clf.fit(X_train_scaled, y_train)
        
        # Validate
        y_pred = clf.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        fold_scores.append(acc)
        
        print(f"  Fold {fold}: {acc:.4f}")
    
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    print(f"\nCV Results: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Train final model on all data
    print(f"\nTraining final model on full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        penalty='l2',
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # Training accuracy
    y_pred_train = model.predict(X_scaled)
    train_acc = accuracy_score(y, y_pred_train)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"{'='*60}\n")
    
    cv_results = {
        'fold_scores': fold_scores,
        'mean_accuracy': cv_mean,
        'std_accuracy': cv_std,
        'train_accuracy': train_acc
    }
    
    return model, scaler, cv_results


def predict(model: LogisticRegression, scaler: StandardScaler, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on new data.
    
    Args:
        model: Trained classifier
        scaler: Fitted scaler
        X: Features
        
    Returns:
        y_pred: Predicted labels
        y_proba: Prediction probabilities
    """
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    return y_pred, y_proba


def save_model(model: LogisticRegression, scaler: StandardScaler, filepath: str) -> None:
    """
    Save trained model and scaler.
    
    Args:
        model: Trained classifier
        scaler: Fitted scaler
        filepath: Where to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print(f"✓ Model saved: {filepath}")


def load_model(filepath: str) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Load trained model and scaler.
    
    Args:
        filepath: Model file path
        
    Returns:
        model: Trained classifier
        scaler: Fitted scaler
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Model loaded: {filepath}")
    return data['model'], data['scaler']


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    
    X = np.random.randn(1000, 232)  # 1000 samples, 232 features
    y = np.random.randint(0, 232, 1000)  # 232 classes
    subjects = np.repeat(np.arange(10), 100)  # 10 subjects
    
    model, scaler, cv_results = train_classifier(X, y, subjects)
    
    print(f"CV accuracy: {cv_results['mean_accuracy']:.4f}")
    
    # Test prediction
    X_test = np.random.randn(10, 232)
    y_pred, y_proba = predict(model, scaler, X_test)
    print(f"Predictions: {y_pred[:5]}")
    
    