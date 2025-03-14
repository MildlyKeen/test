import pytest
import pandas as pd
from scripts.validate_csv import validate_data

def test_missing_values():
    df = pd.read_csv('data/heart_failure.csv')
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values."

def test_data_types():
    df = pd.read_csv('data/heart_failure.csv')
    assert df['age'].dtype == float, "Age column should be of type float."
    assert df['ejection_fraction'].dtype == float, "Ejection fraction column should be of type float."

def test_column_names():
    df = pd.read_csv('data/heart_failure.csv')
    expected_columns = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'DEATH_EVENT']
    assert all(col in df.columns for col in expected_columns), "Missing expected columns."
