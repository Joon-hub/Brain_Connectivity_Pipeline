"""
Brain Region Classifier with Leak Free Cross-Validation
========================================================
Implements subject-wise GroupKFold cross-validation with proper preprocessing isolation. 
All preprocessing happens inside the cross-validation loop to prevent data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import clone
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.models import load_model_from_config

# ==============================================
# Cross-Validation with Preprocessing Isolation
# ==============================================

def cross_validate_no_leakage(
    df_raw: pd.DataFrame,
    preprocessor_class,
    preprocessor_params: Dict,
    classifier_params: Dict,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Perform leak-free cross-validation by fitting preprocessor inside each fold.

    Uses sklearn's GroupKFold for proper subject-level splitting.

    Args:
        df_raw : Raw connectivity DataFrame
        preprocessor_class : Preprocessor class to be instantiated
        preprocessor_params : Parameters for the preprocessor
        classifier_params : Parameters for the classifier
        n_splits : Number of cross-validation splits
        random_state : Random seed
        verbose : Print progress

    Returns:
        results : Dictionary with accuracy, classification report, confusion matrix
    """

    # Validate first column is subject_id
    first_col_name = df_raw.columns[0]
    if 'subject' not in first_col_name.lower() and 'id' not in first_col_name.lower():
        raise ValueError(
            F"Expected first column to contain 'subject' or 'id', got '{first_col_name}'. "
            f"Subject ID column must be the first column in the DataFrame."
        )

    # Extract subject IDs for grouping
    subject_ids = df_raw.iloc[:, 0].values
    unique_subjects = np.unique(subject_ids)

    if len(unique_subjects) < n_splits:
        raise ValueError(
            f"Number of unique subjects ({len(unique_subjects)}) is less than n_splits ({n_splits}). "
            "Reduce n_splits or provide more subjects."
        )

    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []

    # Collect all validation predictions across folds
    all_val_predictions = []
    all_val_true = []
    all_val_subjects = []

    if verbose:
        print(f"\n {'='*60}")
        print(f"Leak-Free Cross-Validation with {n_splits} Folds")
        print(f"{'='*60}\n")
        print(f"Total number of subjects: {len(unique_subjects)}")
        print(f"Total number of samples: {df.shape[0]}\n")
        print(f"Preprocessing: {preprocessor_params.get('diagonal_strategy', 'Unknown')}")

        # check if stratergy is deterministic or stochastic
        strategy = preprocessor_params.get('diagonal_strategy', 'unknown')

        # Deterministic strategies
        if strategy in ['zero', 'region_mean', 'network_mean']:
            print(f"Deterministic diagonal imputation strategy: {strategy}, diagonal values are constant per subject.")
        # Stochastic strategies
        elif strategy in ['random','sample_from_row', 'sample_from_matrix']:
            print(f"Stochastic diagonal imputation strategy: {strategy}, diagonal values vary per subject.")
        print()

    # Split at sample level with subject grouping

    for fold, (train_idx,val_idx) in enumerate(gkf.split(df_raw, groups=subject_ids), 1):
        
        # get dataframe for this fold
        df_train = df_raw.iloc[train_idx].copy()
        df_val = df_raw.iloc[val_idx].copy()

        # Extract subject IDs for train and validation sets
        train_subjects = df_train.iloc[:, 0].unique()
        val_subjects = df_val.iloc[:, 0].unique()

        if verbose:
            print(f"Fold {fold}/{n_splits}:") 
            print(f"  Training subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}")
        
        # Create and fit preprocessor on training fold only 
        preprocessor = preprocessor_class(**preprocessor_params)
        preprocessor.fit(df_train)

        # Transform both training and validation data using fitted preprocessor
        X_train = preprocessor.transform(df_train)
        y_train = preprocessor.get_labels()

        X_val = preprocessor.transform(df_val)
        y_val = preprocessor.get_labels()
        subjects_val = preprocessor.get_subjects()

        # Create and fit classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**classifier_params, random_state=random_state))
        ])

        pipeline.fit(X_train, y_train)

        # Evaluate 
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        # Store validation predictions for this fold
        val_subject_ids = df_val.iloc[:, 0].values
        val_subject_ids_mapped = val_subject_ids[subjects_val]

        all_val_predictions.extend(y_val_pred)
        all_val_true.extend(y_val)
        all_val_subjects.extend(val_subject_ids_mapped)

        fold_results.append({
            'fold': fold,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'n_train_subjects': len(train_subjects),
            'n_val_subjects': len(val_subjects),
            'n_train_samples': X_train.shape[0],
            'n_val_samples': X_val.shape[0],
        })

        if verbose:
            print(f"  Training Accuracy: {train_acc:.4f}")
            print(f"  Validation Accuracy: {val_acc:.4f}\n")
            print()

    # Aggregate overall validation results
    val_accs = [r['val_acc'] for r in fold_results]
    train_accs = [r['train_acc'] for r in fold_results]

    cv_results = {
        'fold_results': fold_results,
        'val_mean': np.mean(val_accs),
        'val_std': np.std(val_accs),
        'train_mean': np.mean(train_accs),
        'train_std': np.std(train_accs),
        'n_splits': n_splits,
        'val_predictions': np.array(all_val_predictions),
        'val_true': np.array(all_val_true),
        'val_subjects': np.array(all_val_subjects)
    }

    if verbose:
        print(f"{'='*60}")
        print(f"Cross-Validation Results:")
        print(f"  Mean Training Accuracy: {cv_results['train_mean']:.4f} ± {cv_results['train_std']:.4f}")
        print(f"  Mean Validation Accuracy: {cv_results['val_mean']:.4f} ± {cv_results['val_std']:.4f}")
        print(f"{'='*60}\n")

    return cv_results

def train_final_model(
        df_train: pd.DataFrame,
        preprocessor_class,
        preprocessor_params: Dict,
        classifier_params: Dict,
        verbose: bool = True
    ) -> Tuple[Pipeline, object]:
    """
    Train final model on all training data.
    
    
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

    # Create and fit classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**classifier_params, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on training data
    y_train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    if verbose:
        print(f"{'='*60}")
        print(f"Final Model Results:")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Trained on {X_train.shape[0]} samples.")
        print()
        print(f"{'='*60}\n")

    return pipeline, preprocessor

# ==============================
# Main Classifier Class
# ==============================

class BrainRegionClassifier:
    """
    Brain Connectivity Classifier with Leak-Free Cross-Validation.
    """

    def __init__(
        self,
        preprocessor_class,
        model_name: str = "logistic_regression"
        diagonal_strategy: str = "zero",
        connection_columns: Optional[List[str]] = None,
        include_diagonal: bool = False,
        apply_fisher_z: bool = True,
        n_splits: int = 5,
        random_state: int = 42,
        enable_diagnostics: bool = False ):

        self.preprocessor_class = preprocessor_class
        self.model_name = model_name
        self.diagonal_strategy = diagonal_strategy
        self.connection_columns = connection_columns
        self.include_diagonal = include_diagonal
        self.apply_fisher_z = apply_fisher_z
        self.n_splits = n_splits
        self.random_state = random_state
        self.enable_diagnostics = enable_diagnostics

        # Initialize preprocessor and classifier
        self.pipeline_ = None
        self.preprocessor_ = None
        self.cv_results_ = None
        self.region_list_ = None
        self.n_regions_ = None
        self.is_fitted_ = False

    def fit(self, df_train: pd.DataFrame, verbose: bool = True):
        """
        Fit the classifier with leak-free cross-validation.

        Args:
            df_train: Training DataFrame
            verbose: Print progress
        """

        # Prepare preprocessor parameters
        preprocessor_params = {
            'connection_columns': self.connection_columns,
            'diagonal_strategy': self.diagonal_strategy,
            'include_diagonal': self.include_diagonal,
            'apply_fisher_z': self.apply_fisher_z,
            'random_state': self.random_state,
            'enable_diagnostic': self.enable_diagnostic
        }

        # Prepare classifier parameters
        classifier_params = {
            'C': self.C,
            'max_iter': self.max_iter,
            'multi_class': 'multinomial',
            'random_state': self.random_state
        }

        # Perform cross-validation
        self.cv_results_ = cross_validate_no_leakage(
            df_train,
            self.preprocessor_class,
            preprocessor_params,
            classifier_params,
            n_splits=self.n_splits,
            random_state=self.random_state,
            verbose=verbose
        )

        # Train final model on all training data
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
            (y_pred, y_true, subjects): Predicted labels, true labels, subject IDs
        """

        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction.")

        # Transform test data
        X_test = self.preprocessor_.transform(df_test)
        y_test = self.preprocessor_.get_labels()
        subject_ids = self.preprocessor_.get_subjects()

        # Get subject IDs from DataFrame
        subject_ids = df_test.iloc[:, 0].values
        subject_ids_mapped = subject_ids[subject_ids]

        # Predict
        y_pred = self.pipeline_.predict(X_test)

        return y_pred, y_test, subject_ids_mapped
    
    def predict_proba(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities on new data.

        Args:
            df_test: Test DataFrame

        Returns:
            y_proba: Predicted class probabilities
        """

        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction.")

        # Transform test data
        X_test = self.preprocessor_.transform(df_test)

        # Predict probabilities
        y_proba = self.pipeline_.predict_proba(X_test)

        return y_proba
    
    def get_cv_results(self):
        """
        Get cross-validation results including validation predictions.
        """
        if self.cv_results_ is None:
            raise RuntimeError("Cross-validation results are not available. Must call fit() first.")
        return self.cv_results_
    
    def get_cv_validation_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cross-validation validation predictions, true labels, and subject IDs.
        """
        if self.cv_results_ is None:
            raise RuntimeError("Cross-validation results are not available. Must call fit() first.")
        return (
            self.cv_results_['val_predictions'],
            self.cv_results_['val_true'],
            self.cv_results_['val_subjects']
        )
    
    def save(self, output_dir: str):
        """Save the trained model and preprocessor to disk."""
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before saving.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pipeline
        pipeline_path = output_dir / f"pipeline_{}_{self.diagonal_strategy}.pkl"
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.pipeline_, f)

        # Save preprocessor
        preprocessor_path = output_dir / f"preprocessor_{self.diagonal_strategy}.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor_, f)

        print(f"Model and preprocessor saved to {output_dir}")
