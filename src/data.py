"""
Data Loading and Validation
============================
Simple, testable functions for loading brain connectivity data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List


def load_connectivity_data(filepath: str, validate: bool = True) -> pd.DataFrame:
    """
    Load connectivity CSV with validation.
    
    Args:
        filepath: Path to CSV file
        validate: Whether to validate schema
        
    Returns:
        DataFrame with connectivity data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If schema validation fails
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    if validate:
        validate_schema(df)
    
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate connectivity data schema.
    
    Expected format:
    - Column 0: subject_id
    - Columns 1-N: Region_A~Region_B (pairwise connectivity)
    
    Raises:
        ValueError: If schema is invalid
    """
    # Check minimum columns
    if df.shape[1] < 100:
        raise ValueError(f"Too few columns: {df.shape[1]}. Expected >100 connection columns.")
    
    # Check connection column format
    connection_cols = df.columns[1:]
    sample_cols = connection_cols[:10]
    
    for col in sample_cols:
        if '~' not in col:
            raise ValueError(f"Invalid connection column format: {col}. Expected 'Region_A~Region_B'")
    
    # Check for NaN
    if df.isnull().any().any():
        n_missing = df.isnull().sum().sum()
        raise ValueError(f"Data contains {n_missing} missing values. Clean data required.")
    
    print(f"✓ Schema valid: {df.shape[0]} subjects, {df.shape[1]-1} connections")


def extract_connection_columns(df: pd.DataFrame) -> List[str]:
    """Get list of connection column names (excludes subject ID)."""
    return df.columns[1:].tolist()


def extract_subjects(df: pd.DataFrame) -> np.ndarray:
    """Get array of subject IDs."""
    return df.iloc[:, 0].to_numpy()


def create_sample_dataset(input_path: str, output_path: str, n_subjects: int = 10) -> None:
    """
    Create a small sample dataset for testing.
    
    Args:
        input_path: Full dataset path
        output_path: Where to save sample
        n_subjects: Number of subjects to include
    """
    df = pd.read_csv(input_path)
    sample = df.head(n_subjects)
    sample.to_csv(output_path, index=False)
    print(f"✓ Created sample with {n_subjects} subjects: {output_path}")


# Example usage for testing
if __name__ == "__main__":
    # Test on sample data
    df = load_connectivity_data("../data/sample/sample_piop2.csv")
    print(f"Loaded {df.shape[0]} subjects with {df.shape[1]-1} connections")
    
    connections = extract_connection_columns(df)
    print(f"First connection: {connections[0]}")
    
    subjects = extract_subjects(df)
    print(f"Subjects: {subjects[:5]}")