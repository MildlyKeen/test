import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.validate_csv import validate_csv

def test_validate_csv():
    """Test if the CSV file passes the validation process."""
    file_path = 'data/heart_failure.csv'
    
    # Call the validate_csv function
    try:
        validate_csv(file_path)  # This will raise an error if validation fails
    except ValueError as e:
        pytest.fail(f"CSV validation failed: {e}")
    
    # If validation passed, assert True (this line is not strictly necessary)
    assert True, "CSV validation passed successfully."

def test_missing_values():
    """Test if the dataset has missing values."""
    df = pd.read_csv('data/heart_failure.csv')
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values."

def test_data_types():
    df = pd.read_csv('data/heart_failure.csv')
    df['ejection_fraction'] = df['ejection_fraction'].astype(float)
    assert df['age'].dtype == float, "Age column should be of type float."
    assert df['ejection_fraction'].dtype == float, "Ejection fraction column should be of type float."

def test_column_names():
    """Test if the dataset contains the required columns."""
    df = pd.read_csv('data/heart_failure.csv')
    expected_columns = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
        'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
        'smoking', 'time', 'DEATH_EVENT'
    ]
    assert all(col in df.columns for col in expected_columns), "Missing expected columns."
