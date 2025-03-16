import pytest
import pandas as pd
import os

# Path to the dataset
DATASET_PATH = 'data/heart_failure_balanced.csv'

# Ensure the dataset file exists
assert os.path.exists(DATASET_PATH), f"Dataset file not found at {DATASET_PATH}"

def test_file_exists():
    """
    Test if the dataset file exists.
    """
    assert os.path.exists(DATASET_PATH), f"Dataset file not found at {DATASET_PATH}"
    print("✅ Dataset file exists test passed!")

def test_csv_format():
    """
    Test if the dataset is in CSV format.
    """
    assert DATASET_PATH.endswith('.csv'), "Dataset file is not in CSV format."
    print("✅ CSV format test passed!")

def test_missing_values():
    """
    Test if the dataset has missing values.
    """
    df = pd.read_csv(DATASET_PATH)
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values."
    print("✅ Missing values test passed!")

def test_data_types():
    """
    Test if the dataset columns have the correct data types.
    """
    df = pd.read_csv(DATASET_PATH)
    assert df['age'].dtype in [float, int], "Age column should be of type float or int."
    assert df['ejection_fraction'].dtype in [float, int], "Ejection fraction column should be of type float or int."
    assert df['serum_creatinine'].dtype in [float, int], "Serum creatinine column should be of type float or int."
    print("✅ Data types test passed!")

def test_column_names():
    """
    Test if the dataset contains the required columns.
    """
    df = pd.read_csv(DATASET_PATH)
    expected_columns = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
        'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
        'smoking', 'time', 'DEATH_EVENT'
    ]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    assert not missing_columns, f"Missing expected columns: {missing_columns}"
    print("✅ Column names test passed!")

def test_column_ranges():
    """
    Test if numeric columns have values within realistic ranges.
    """
    df = pd.read_csv(DATASET_PATH)
    assert df['age'].between(18, 100).all(), "Age column contains unrealistic values."
    assert df['ejection_fraction'].between(10, 80).all(), "Ejection fraction column contains unrealistic values."
    assert df['serum_creatinine'].between(0.1, 10).all(), "Serum creatinine column contains unrealistic values."
    assert df['serum_sodium'].between(110, 150).all(), "Serum sodium column contains unrealistic values."
    print("✅ Column ranges test passed!")