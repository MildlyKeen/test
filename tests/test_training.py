import sys
import os
import joblib
import pytest
from sklearn.metrics import accuracy_score
from train import train_model

def test_model_training():
    # Test that model training completes successfully
    model = train_model('data/heart_failure_balanced.csv')
    assert model is not None, "Model training failed"
    assert os.path.exists('models/final_model.pkl'), "Model file not saved"

def test_model_accuracy():
    # Test that model accuracy meets expectations
    model = train_model('data/heart_failure_balanced.csv')
    assert model is not None, "Model training failed"
    
    # Load test data
    import pandas as pd
    df = pd.read_csv('data/heart_failure_balanced.csv')
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    
    # Evaluate model
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    assert accuracy > 0.7, f"Model accuracy {accuracy:.2f} is below expected threshold"
    print(f"Model accuracy: {accuracy:.2f}")
