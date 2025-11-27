"""
Brain Region Classifier with Leak-Free Cross-Validation
========================================================
Implements subject-wise GroupKFold cross-validation with proper preprocessing isolation. 
All preprocessing happens inside the cross-validation loop to prevent data leakage.

Key Features:
- Subject-wise cross-validation (GroupKFold) to prevent data leakage
- Preprocessing done inside each CV fold
- Support for multiple models via YAML configuration
- Comprehensive diagnostic capabilities
- Proper handling of symmetric connectivity matrices
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import clone
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


# ==============================================
# Cross-Validation with Preprocessing Isolation
# ==============================================

def cross_validate_no_leakage(
    df_raw: pd.DataFrame,
    preprocessor_class,
    preprocessor_params: Dict,
    model_instance,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Perform leak-free cross-validation by fitting preprocessor inside each fold.

    Uses sklearn's GroupKFold for proper subject-level splitting.

    Args:
        df_raw (pd.DataFrame): Raw connectivity DataFrame
        preprocessor_class: Preprocessor class to be instantiated
        preprocessor_params (dict): Parameters for the preprocessor
        model_instance: Instantiated classifier (e.g., LogisticRegression, XGBClassifier)
        n_splits (int): Number of cross-validation splits
        random_state (int): Random seed
        verbose (bool): Print progress

    Returns:
        results (dict): Dictionary with accuracy, fold results, and validation predictions
    """

    # Validate first column is subject_id
    first_col_name = df_raw.columns[0]
    if 'subject' not in first_col_name.lower() and 'id' not in first_col_name.lower():
        raise ValueError(
            f"Expected first column to contain 'subject' or 'id', got '{first_col_name}'. "
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
        print(f"\n{'='*60}")
        print(f"Leak-Free Cross-Validation with {n_splits} Folds")
        print(f"{'='*60}\n")
        print(f"Total number of subjects: {len(unique_subjects)}")
        print(f"Total number of samples: {df_raw.shape[0]}\n")
        print(f"Preprocessing: {preprocessor_params.get('diagonal_strategy', 'Unknown')}")

        # Check if strategy is deterministic or stochastic
        strategy = preprocessor_params.get('diagonal_strategy', 'unknown')

        # Deterministic strategies
        if strategy in ['zero', 'region_mean', 'network_mean']:
            print(f"Deterministic diagonal imputation strategy: {strategy}, diagonal values are constant per subject.")
        # Stochastic strategies
        elif strategy in ['random', 'sample_from_row', 'sample_from_matrix']:
            print(f"Stochastic diagonal imputation strategy: {strategy}, diagonal values vary per subject.")
        print()

    # Split at sample level with subject grouping
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_raw, groups=subject_ids), 1):
        
        # Get dataframe for this fold
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

        # Create pipeline with scaler and cloned model
        pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            ('classifier', clone(model_instance))
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

    # Aggregate overall validation results
    val_accs = [r['val_accuracy'] for r in fold_results]
    train_accs = [r['train_accuracy'] for r in fold_results]

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
    model_instance,
    verbose: bool = True
) -> Tuple[Pipeline, object]:
    """
    Train final model on all training data.
    
    Args:
        df_train (pd.DataFrame): Full training DataFrame
        preprocessor_class: Preprocessor class
        preprocessor_params (dict): Preprocessor parameters
        model_instance: Instantiated classifier
        verbose (bool): Print progress
    
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

    # Create and fit pipeline
    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('classifier', clone(model_instance))
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
    
    This class handles the complete workflow:
    1. Subject-wise cross-validation with GroupKFold
    2. Preprocessing isolation (no data leakage)
    3. Training final model on all data
    4. Prediction on new data
    
    Example:
        >>> from src.models import load_model_from_config
        >>> from src.features import BrainConnectivityPreprocessor
        >>> 
        >>> model = load_model_from_config("logistic_regression")
        >>> classifier = BrainRegionClassifier(
        ...     preprocessor_class=BrainConnectivityPreprocessor,
        ...     model_instance=model,
        ...     diagonal_strategy="zero"
        ... )
        >>> classifier.fit(df_train)
        >>> y_pred, y_true, subjects = classifier.predict(df_test)
    """

    def __init__(
        self,
        preprocessor_class,
        model_instance,
        model_name: str = "unknown_model",
        diagonal_strategy: str = "zero",
        connection_columns: Optional[List[str]] = None,
        include_diagonal: bool = True,
        apply_fisher_z: bool = True,
        n_splits: int = 5,
        random_state: int = 42,
        enable_diagnostics: bool = False
    ):
        """
        Initialize the Brain Region Classifier.
        
        Args:
            preprocessor_class: Class for preprocessing (e.g., BrainConnectivityPreprocessor)
            model_instance: Instantiated model (e.g., from load_model_from_config())
            model_name (str): Name of the model for logging/saving
            diagonal_strategy (str): Diagonal imputation strategy
            connection_columns (list): List of connection column names
            include_diagonal (bool): Include diagonal in features
            apply_fisher_z (bool): Apply Fisher Z transformation
            n_splits (int): Number of CV folds
            random_state (int): Random seed
            enable_diagnostics (bool): Enable diagnostic logging
        """
        self.preprocessor_class = preprocessor_class
        self.model_instance = model_instance
        self.model_name = model_name
        self.diagonal_strategy = diagonal_strategy
        self.connection_columns = connection_columns
        self.include_diagonal = include_diagonal
        self.apply_fisher_z = apply_fisher_z
        self.n_splits = n_splits
        self.random_state = random_state
        self.enable_diagnostics = enable_diagnostics

        # Will be set during fit
        self.pipeline_ = None
        self.preprocessor_ = None
        self.cv_results_ = None
        self.region_list_ = None
        self.n_regions_ = None
        self.is_fitted_ = False
        self.fit_timestamp_ = None

    def fit(self, df_train: pd.DataFrame, verbose: bool = True):
        """
        Fit the classifier with leak-free cross-validation.

        Args:
            df_train (pd.DataFrame): Training DataFrame
            verbose (bool): Print progress
            
        Returns:
            self: Fitted classifier
        """
        self.fit_timestamp_ = datetime.now()

        # Prepare preprocessor parameters
        preprocessor_params = {
            'connection_columns': self.connection_columns,
            'diagonal_strategy': self.diagonal_strategy,
            'include_diagonal': self.include_diagonal,
            'apply_fisher_z': self.apply_fisher_z,
            'random_state': self.random_state,
            'enable_diagnostics': self.enable_diagnostics
        }

        # Perform cross-validation
        self.cv_results_ = cross_validate_no_leakage(
            df_train,
            self.preprocessor_class,
            preprocessor_params,
            self.model_instance,
            n_splits=self.n_splits,
            random_state=self.random_state,
            verbose=verbose
        )

        # Train final model on all training data
        self.pipeline_, self.preprocessor_ = train_final_model(
            df_train,
            self.preprocessor_class,
            preprocessor_params,
            self.model_instance,
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
            df_test (pd.DataFrame): Test DataFrame

        Returns:
            (y_pred, y_true, subjects): Predicted labels, true labels, subject IDs
        """
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction.")

        # Transform test data
        X_test = self.preprocessor_.transform(df_test)
        y_test = self.preprocessor_.get_labels()
        subjects = self.preprocessor_.get_subjects()

        # Get subject IDs from DataFrame
        subject_ids = df_test.iloc[:, 0].values
        subject_ids_mapped = subject_ids[subjects]

        # Predict
        y_pred = self.pipeline_.predict(X_test)

        return y_pred, y_test, subject_ids_mapped
    
    def predict_proba(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities on new data.

        Args:
            df_test (pd.DataFrame): Test DataFrame

        Returns:
            y_proba (np.ndarray): Predicted class probabilities
        """
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction.")

        # Transform test data
        X_test = self.preprocessor_.transform(df_test)

        # Predict probabilities
        y_proba = self.pipeline_.predict_proba(X_test)

        return y_proba
    
    def get_cv_results(self) -> Dict:
        """
        Get cross-validation results including validation predictions.
        
        Returns:
            dict: CV results with fold statistics and predictions
        """
        if self.cv_results_ is None:
            raise RuntimeError("Cross-validation results are not available. Must call fit() first.")
        return self.cv_results_
    
    def get_cv_validation_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cross-validation validation predictions, true labels, and subject IDs.
        
        Returns:
            (val_predictions, val_true, val_subjects): Arrays of predictions, labels, and subjects
        """
        if self.cv_results_ is None:
            raise RuntimeError("Cross-validation results are not available. Must call fit() first.")
        return (
            self.cv_results_['val_predictions'],
            self.cv_results_['val_true'],
            self.cv_results_['val_subjects']
        )
    
    def get_metadata(self) -> Dict:
        """
        Get metadata about this classifier run.
        
        Returns:
            dict: Metadata including model config, preprocessing, and results
        """
        if not self.is_fitted_:
            raise RuntimeError("Metadata not available. Must call fit() first.")
        
        return {
            'model_name': self.model_name,
            'model_type': type(self.model_instance).__name__,
            'diagonal_strategy': self.diagonal_strategy,
            'apply_fisher_z': self.apply_fisher_z,
            'include_diagonal': self.include_diagonal,
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'n_regions': self.n_regions_,
            'fit_timestamp': self.fit_timestamp_.isoformat() if self.fit_timestamp_ else None,
            'cv_val_mean': self.cv_results_['val_mean'],
            'cv_val_std': self.cv_results_['val_std'],
            'cv_train_mean': self.cv_results_['train_mean'],
            'cv_train_std': self.cv_results_['train_std'],
        }
    
    def save(self, output_dir: str):
        """
        Save the trained model and preprocessor to disk.
        
        Args:
            output_dir (str): Directory to save models
        """
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before saving.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pipeline
        pipeline_path = output_dir / f"pipeline_{self.model_name}_{self.diagonal_strategy}.pkl"
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.pipeline_, f)

        # Save preprocessor
        preprocessor_path = output_dir / f"preprocessor_{self.diagonal_strategy}.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor_, f)

        # Save metadata
        metadata_path = output_dir / f"metadata_{self.model_name}_{self.diagonal_strategy}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.get_metadata(), f, indent=2)

        print(f"✓ Model saved to {output_dir}")
        print(f"  - Pipeline: {pipeline_path.name}")
        print(f"  - Preprocessor: {preprocessor_path.name}")
        print(f"  - Metadata: {metadata_path.name}")