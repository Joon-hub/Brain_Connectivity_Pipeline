"""
Smoke Tests for Brain Connectivity Pipeline
===========================================
Quick tests to verify installation and basic functionality.

Run: python tests/test_smoke.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd


def test_imports():
    """Test 1: Can we import all modules?"""
    print("\n[Test 1/7] Testing imports...")
    try:
        from data import load_connectivity_data, get_connection_columns
        from features import extract_regions, create_dataset
        from model import train_classifier, predict
        from evaluate import calculate_error_map
        from visualize import plot_error_map
        from utils import set_random_seeds, load_config
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_loading():
    """Test 2: Can we create dummy data?"""
    print("\n[Test 2/7] Testing data structures...")
    try:
        # Create dummy connectivity data
        n_subjects = 5
        n_regions = 10
        n_connections = (n_regions * (n_regions - 1)) // 2
        
        # Create region names
        regions = [f"Region_{i}" for i in range(n_regions)]
        
        # Create connection columns
        columns = ['subject_id']
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                columns.append(f"{regions[i]}~{regions[j]}")
        
        # Create data
        data = {
            'subject_id': [f'sub-{i:03d}' for i in range(n_subjects)]
        }
        for col in columns[1:]:
            data[col] = np.random.uniform(-1, 1, n_subjects)
        
        df = pd.DataFrame(data)
        
        assert df.shape == (n_subjects, n_connections + 1)
        assert df.columns[0] == 'subject_id'
        print(f"✓ Created dummy dataset: {df.shape}")
        return True, df, columns[1:]
    except Exception as e:
        print(f"✗ Data creation failed: {e}")
        return False, None, None


def test_region_extraction():
    """Test 3: Can we extract regions from connection columns?"""
    print("\n[Test 3/7] Testing region extraction...")
    try:
        from features import extract_regions
        
        # Dummy connection columns
        connections = ['RegionA~RegionB', 'RegionA~RegionC', 'RegionB~RegionC']
        
        region_list, region_to_idx, n_regions = extract_regions(connections)
        
        assert n_regions == 3
        assert len(region_list) == 3
        assert 'RegionA' in region_list
        print(f"✓ Extracted {n_regions} regions")
        return True
    except Exception as e:
        print(f"✗ Region extraction failed: {e}")
        return False


def test_feature_creation(df, connection_columns):
    """Test 4: Can we create features?"""
    print("\n[Test 4/7] Testing feature creation...")
    try:
        from features import extract_regions, create_dataset
        
        region_list, region_to_idx, n_regions = extract_regions(connection_columns)
        
        X, y, subjects = create_dataset(
            df, connection_columns, region_list, region_to_idx,
            diagonal_strategy='mean'  # Use simple strategy for test
        )
        
        n_samples = len(df) * n_regions
        assert X.shape[0] == n_samples
        assert X.shape[1] == n_regions
        assert len(y) == n_samples
        print(f"✓ Created features: X.shape={X.shape}, y.shape={y.shape}")
        return True, X, y, subjects
    except Exception as e:
        print(f"✗ Feature creation failed: {e}")
        return False, None, None, None


def test_model_training(X, y, subjects):
    """Test 5: Can we train a model?"""
    print("\n[Test 5/7] Testing model training...")
    try:
        from model import train_classifier
        
        model, scaler, cv_results = train_classifier(
            X, y, subjects, n_splits=2, C=0.01, random_state=42
        )
        
        assert model is not None
        assert scaler is not None
        assert 'mean_accuracy' in cv_results
        print(f"✓ Model trained: CV accuracy = {cv_results['mean_accuracy']:.4f}")
        return True, model, scaler
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False, None, None


def test_prediction(model, scaler, X):
    """Test 6: Can we make predictions?"""
    print("\n[Test 6/7] Testing predictions...")
    try:
        from model import predict
        
        y_pred, y_proba = predict(model, scaler, X)
        
        assert len(y_pred) == len(X)
        assert y_proba.shape[0] == len(X)
        print(f"✓ Predictions successful: {len(y_pred)} predictions")
        return True, y_pred
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False, None


def test_evaluation(y, y_pred):
    """Test 7: Can we calculate error map?"""
    print("\n[Test 7/7] Testing evaluation...")
    try:
        from features import extract_regions
        from evaluate import calculate_error_map
        
        # Create dummy region list
        n_regions = len(np.unique(y))
        region_list = [f"Region_{i}" for i in range(n_regions)]
        
        error_df = calculate_error_map(y, y_pred, region_list)
        
        assert len(error_df) == n_regions
        assert 'misclassification_rate' in error_df.columns
        print(f"✓ Error map created: {len(error_df)} regions")
        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("="*70)
    print("BRAIN CONNECTIVITY PIPELINE - SMOKE TESTS")
    print("="*70)
    
    # Test 1
    if not test_imports():
        print("\n❌ FAILED: Fix import errors first")
        return 1
    
    # Test 2
    success, df, columns = test_data_loading()
    if not success:
        print("\n❌ FAILED: Data structure issues")
        return 1
    
    # Test 3
    if not test_region_extraction():
        print("\n❌ FAILED: Region extraction issues")
        return 1
    
    # Test 4
    success, X, y, subjects = test_feature_creation(df, columns)
    if not success:
        print("\n❌ FAILED: Feature creation issues")
        return 1
    
    # Test 5
    success, model, scaler = test_model_training(X, y, subjects)
    if not success:
        print("\n❌ FAILED: Model training issues")
        return 1
    
    # Test 6
    success, y_pred = test_prediction(model, scaler, X)
    if not success:
        print("\n❌ FAILED: Prediction issues")
        return 1
    
    # Test 7
    if not test_evaluation(y, y_pred):
        print("\n❌ FAILED: Evaluation issues")
        return 1
    
    # Success!
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
    print("\nYour installation is working correctly!")
    print("Next step: python run.py --sample")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())