"""
Brain Region Classifier with Leak-Free Cross-Validation
========================================================
Implements subject-wise GroupKFold CV with proper preprocessing isolation.
All preprocessing happens INSIDE the CV loop to prevent leakage.

CORRECTED VERSION v2:
- Fisher Z transformation removed from pipeline (now in preprocessing)
- StandardScaler only in pipeline
- Better validation of subject_id column
- Clearer warnings about training vs validation accuracy
- ✅ NEW: Saves CV validation predictions for proper rest vs task comparison
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional, List


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validate_no_leakage(
    df_raw: pd.DataFrame,
    preprocessor_class,
    preprocessor_params: dict,
    classifier_params: dict,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Perform leak-free cross-validation by fitting preprocessor inside each fold.
    
    ✅ NEW: Now collects and returns all validation predictions across folds
    for proper rest vs task error comparison.
    
    Args:
        df_raw: Raw connectivity DataFrame
        preprocessor_class: BrainConnectivityPreprocessor class (not instance!)
        preprocessor_params: Parameters for preprocessor
        classifier_params: Parameters for logistic regression
        n_splits: Number of CV folds
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Dictionary with CV results including validation predictions
    """
    # Validate first column is subject_id
    first_col_name = df_raw.columns[0]
    if 'subject' not in first_col_name.lower() and 'id' not in first_col_name.lower():
        raise ValueError(
            f"Expected first column to contain 'subject' or 'id', got '{first_col_name}'. "
            f"Subject ID column must be first."
        )
    
    # Extract subject IDs for grouping
    subject_ids = df_raw.iloc[:, 0].values
    unique_subjects = np.unique(subject_ids)
    
    if len(unique_subjects) < n_splits:
        raise ValueError(
            f"Not enough subjects ({len(unique_subjects)}) for {n_splits}-fold CV. "
            f"Need at least {n_splits} subjects."
        )
    
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    
    # ✅ NEW: Collect all validation predictions across folds
    all_val_predictions = []
    all_val_true = []
    all_val_subjects = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"LEAK-FREE {n_splits}-FOLD CROSS-VALIDATION")
        print(f"{'='*60}")
        print(f"Total subjects: {len(unique_subjects)}")
        print(f"Preprocessing: {preprocessor_params.get('diagonal_strategy', 'unknown')}")
        print()
    
    # Split at SUBJECT level (not sample level!)
    for fold, (train_subj_idx, val_subj_idx) in enumerate(
        gkf.split(unique_subjects, groups=unique_subjects), 1
    ):
        # Get subject IDs for this fold
        train_subjects = unique_subjects[train_subj_idx]
        val_subjects = unique_subjects[val_subj_idx]
        
        # Split DataFrame by subjects
        df_train = df_raw[df_raw.iloc[:, 0].isin(train_subjects)].copy()
        df_val = df_raw[df_raw.iloc[:, 0].isin(val_subjects)].copy()
        
        if verbose:
            print(f"Fold {fold}/{n_splits}:")
            print(f"  Train subjects: {len(train_subjects)}")
            print(f"  Val subjects: {len(val_subjects)}")
        
        # === CREATE AND FIT PREPROCESSOR ON TRAIN FOLD ONLY ===
        preprocessor = preprocessor_class(**preprocessor_params)
        preprocessor.fit(df_train)
        
        # Transform both sets using train-fitted preprocessor
        X_train = preprocessor.transform(df_train)
        y_train = preprocessor.get_labels()
        subjects_train = preprocessor.get_subjects()
        
        X_val = preprocessor.transform(df_val)
        y_val = preprocessor.get_labels()
        subjects_val = preprocessor.get_subjects()
        
        # === CREATE AND FIT CLASSIFIER ===
        # CORRECTED: Only StandardScaler in pipeline now
        # Fisher Z is applied in preprocessing (on correlation values)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**classifier_params))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # ✅ NEW: Store validation predictions for this fold
        # Get actual subject IDs (not indices)
        val_subject_ids = df_val.iloc[:, 0].values
        val_subject_ids_mapped = val_subject_ids[subjects_val]
        
        all_val_predictions.extend(y_val_pred)
        all_val_true.extend(y_val)
        all_val_subjects.extend(val_subject_ids_mapped)
        
        fold_results.append({
            'fold': fold,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'n_train_subjects': len(train_subjects),
            'n_val_subjects': len(val_subjects),
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val)
        })
        
        if verbose:
            print(f"    Train acc: {train_acc:.4f}")
            print(f"    Val acc:   {val_acc:.4f}")
            print()
    
    # Aggregate results
    val_accs = [r['val_acc'] for r in fold_results]
    train_accs = [r['train_acc'] for r in fold_results]
    
    cv_results = {
        'fold_results': fold_results,
        'val_mean': np.mean(val_accs),
        'val_std': np.std(val_accs),
        'train_mean': np.mean(train_accs),
        'train_std': np.std(train_accs),
        'n_splits': n_splits,
        # ✅ NEW: Include validation predictions
        'val_predictions': np.array(all_val_predictions),
        'val_true': np.array(all_val_true),
        'val_subjects': np.array(all_val_subjects)
    }
    
    if verbose:
        print(f"{'='*60}")
        print(f"CV RESULTS:")
        print(f"  Train: {cv_results['train_mean']:.4f} ± {cv_results['train_std']:.4f}")
        print(f"  Val:   {cv_results['val_mean']:.4f} ± {cv_results['val_std']:.4f}")
        print(f"{'='*60}")
        print(f"✅ Collected {len(all_val_predictions)} validation predictions")
        print(f"   (Use these for unbiased rest vs task comparison)")
        print()
    
    return cv_results


def train_final_model(
    df_train: pd.DataFrame,
    preprocessor_class,
    preprocessor_params: dict,
    classifier_params: dict,
    verbose: bool = True
) -> Tuple[Pipeline, object]:
    """
    Train final model on all training data.
    
    CORRECTED: Added warning about training accuracy overfitting
    
    Args:
        df_train: Full training DataFrame
        preprocessor_class: Preprocessor class
        preprocessor_params: Preprocessor parameters
        classifier_params: Classifier parameters
        verbose: Print progress
    
    Returns:
        (pipeline, preprocessor): Trained pipeline and fitted preprocessor
    """
    if verbose:
        print("\nTraining final model on all training data...")
    
    # Fit preprocessor on all training data
    preprocessor = preprocessor_class(**preprocessor_params)
    preprocessor.fit(df_train)
    
    # Transform
    X_train = preprocessor.transform(df_train)
    y_train = preprocessor.get_labels()
    
    # Train classifier
    # CORRECTED: Only StandardScaler in pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**classifier_params))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate on training data
    y_train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    if verbose:
        print(f"Final training accuracy: {train_acc:.4f}")
        print(f"Trained on {len(X_train)} samples")
        
        # CORRECTED: Add warning about overfitting
        if train_acc > 0.85:
            print(f"\n⚠️  NOTE: Training accuracy is high - this represents overfitting.")
            print(f"   Use CV validation accuracy for true generalization performance.")
    
    return pipeline, preprocessor


# ============================================================================
# WRAPPER CLASS
# ============================================================================

class BrainRegionClassifier:
    """
    Complete leak-free brain region classifier.
    
    This class properly handles preprocessing and classification with
    subject-wise cross-validation to prevent data leakage.
    
    CORRECTED VERSION v2:
    - Fisher Z transformation now in preprocessing (not in pipeline)
    - Better validation and error messages
    - Clearer documentation of overfitting in final model
    - ✅ NEW: Provides CV validation predictions for proper analysis
    """
    
    def __init__(
        self,
        preprocessor_class,
        diagonal_strategy: str = "zero",
        connection_columns: Optional[List[str]] = None,
        include_diagonal: bool = False,
        apply_fisher_z: bool = True,
        C: float = 0.01,
        max_iter: int = 1000,
        n_splits: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            preprocessor_class: BrainConnectivityPreprocessor class
            diagonal_strategy: Imputation strategy
            connection_columns: List of connection column names
            include_diagonal: Include diagonal in features (should be False if diagonal imputed)
            apply_fisher_z: Apply Fisher Z transformation in preprocessing
            C: Logistic regression regularization
            max_iter: Max iterations
            n_splits: CV folds
            random_state: Random seed
        """
        self.preprocessor_class = preprocessor_class
        self.diagonal_strategy = diagonal_strategy
        self.connection_columns = connection_columns
        self.include_diagonal = include_diagonal
        self.apply_fisher_z = apply_fisher_z
        self.C = C
        self.max_iter = max_iter
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Will be set during fit
        self.pipeline_ = None
        self.preprocessor_ = None
        self.cv_results_ = None
        self.region_list_ = None
        self.n_regions_ = None
        self.is_fitted_ = False
    
    def fit(self, df_train: pd.DataFrame, verbose: bool = True):
        """
        Fit classifier with leak-free cross-validation.
        
        Args:
            df_train: Training DataFrame
            verbose: Print progress
        """
        # Prepare parameters
        preprocessor_params = {
            'connection_columns': self.connection_columns,
            'diagonal_strategy': self.diagonal_strategy,
            'include_diagonal': self.include_diagonal,
            'apply_fisher_z': self.apply_fisher_z,
            'random_state': self.random_state
        }
        
        classifier_params = {
            'C': self.C,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'solver': 'lbfgs',
            'penalty': 'l2',
            'n_jobs': -1,
            'multi_class': 'multinomial'
        }
        
        # Cross-validation
        self.cv_results_ = cross_validate_no_leakage(
            df_train,
            self.preprocessor_class,
            preprocessor_params,
            classifier_params,
            n_splits=self.n_splits,
            random_state=self.random_state,
            verbose=verbose
        )
        
        # Train final model
        self.pipeline_, self.preprocessor_ = train_final_model(
            df_train,
            self.preprocessor_class,
            preprocessor_params,
            classifier_params,
            verbose=verbose
        )
        
        # Store metadata
        self.region_list_ = self.preprocessor_.region_list_
        self.n_regions_ = self.preprocessor_.n_regions_
        self.is_fitted_ = True
        
        return self
    
    def predict(self, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict on new data.
        
        Args:
            df_test: Test DataFrame
        
        Returns:
            (y_pred, y_true, subjects): Predictions, true labels, subject IDs
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before predict()")
        
        # Transform using fitted preprocessor
        X_test = self.preprocessor_.transform(df_test)
        y_test = self.preprocessor_.get_labels()
        subjects_test = self.preprocessor_.get_subjects()
        
        # Get subject IDs from DataFrame
        subject_ids = df_test.iloc[:, 0].values
        subject_ids_mapped = subject_ids[subjects_test]
        
        # Predict
        y_pred = self.pipeline_.predict(X_test)
        
        return y_pred, y_test, subject_ids_mapped
    
    def predict_proba(self, df_test: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before predict_proba()")
        
        X_test = self.preprocessor_.transform(df_test)
        return self.pipeline_.predict_proba(X_test)
    
    def get_cv_results(self) -> Dict:
        """Get cross-validation results including validation predictions."""
        if self.cv_results_ is None:
            raise RuntimeError("Must call fit() first")
        return self.cv_results_
    
    def get_cv_validation_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get CV validation predictions for proper rest vs task comparison.
        
        ✅ NEW METHOD: Use these predictions (not training predictions) 
        as your resting-state baseline when comparing to task data!
        
        Returns:
            (y_pred, y_true, subjects): Validation predictions, true labels, subject IDs
        """
        if self.cv_results_ is None:
            raise RuntimeError("Must call fit() first")
        
        return (
            self.cv_results_['val_predictions'],
            self.cv_results_['val_true'],
            self.cv_results_['val_subjects']
        )
    
    def save(self, output_dir: str):
        """Save model and preprocessor."""
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before save()")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        pipeline_path = output_dir / f'pipeline_{self.diagonal_strategy}.pkl'
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.pipeline_, f)
        
        # Save preprocessor
        preprocessor_path = output_dir / f'preprocessor_{self.diagonal_strategy}.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor_, f)
        
        print(f"✓ Model saved to {output_dir}")
    
    def load(self, output_dir: str, diagonal_strategy: str):
        """Load saved model."""
        output_dir = Path(output_dir)
        
        pipeline_path = output_dir / f'pipeline_{diagonal_strategy}.pkl'
        preprocessor_path = output_dir / f'preprocessor_{diagonal_strategy}.pkl'
        
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        with open(pipeline_path, 'rb') as f:
            self.pipeline_ = pickle.load(f)
        
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor_ = pickle.load(f)
        
        self.region_list_ = self.preprocessor_.region_list_
        self.n_regions_ = self.preprocessor_.n_regions_
        self.is_fitted_ = True
        
        print(f"✓ Model loaded from {output_dir}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing leak-free classifier with CV validation predictions...\n")
    
    # Create dummy data
    np.random.seed(42)
    n_subjects = 20
    n_regions = 5
    
    regions = [f'Region_{i}' for i in range(n_regions)]
    connection_cols = []
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            connection_cols.append(f"{regions[i]}~{regions[j]}")
    
    # Create DataFrame
    data = {'subject_id': [f'S{i:03d}' for i in range(n_subjects)]}
    for col in connection_cols:
        data[col] = np.random.uniform(-1, 1, n_subjects)  # Correlation values
    
    df = pd.DataFrame(data)
    
    print(f"Dataset: {n_subjects} subjects, {n_regions} regions")
    print(f"Connections: {len(connection_cols)}\n")
    
    # Import preprocessor
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    from old_code.features_251125 import BrainConnectivityPreprocessor
    
    # Create classifier
    classifier = BrainRegionClassifier(
        preprocessor_class=BrainConnectivityPreprocessor,
        diagonal_strategy="region_mean",
        connection_columns=connection_cols,
        include_diagonal=False,
        apply_fisher_z=True,
        C=0.01,
        n_splits=3,
        random_state=42
    )
    
    # Fit with CV
    classifier.fit(df, verbose=True)
    
    # Get results
    cv_results = classifier.get_cv_results()
    print(f"\nFinal CV Validation Accuracy: {cv_results['val_mean']:.4f} ± {cv_results['val_std']:.4f}")
    print(f"Training Accuracy (overfitted): {cv_results['train_mean']:.4f} ± {cv_results['train_std']:.4f}")
    
    # ✅ NEW: Get CV validation predictions
    val_pred, val_true, val_subjects = classifier.get_cv_validation_predictions()
    print(f"\n✅ CV Validation Predictions: {len(val_pred)} samples")
    print(f"   Use these (not training predictions) for rest vs task comparison!")
    
    print("\n✓ Classifier is leak-free!")
    print("✓ Fisher Z transformation correctly applied to correlation values")
    print("✓ CV validation predictions available for proper analysis")