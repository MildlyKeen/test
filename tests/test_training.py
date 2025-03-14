import pytest
import os
from train import train_model

def test_model_training():
    # Assuming `train_model` returns a trained model or saves the model to a file
    model = train_model('data/heart_failure_balanced.csv')
    assert model is not None, "Model training failed."
    assert os.path.exists('models/final_model.pkl'), "Model file not saved."

def test_model_accuracy():
    # Assuming your model has a method to evaluate its accuracy
    model = train_model('data/heart_failure_balanced.csv')
    accuracy = model.evaluate()  # Replace with actual evaluation code
    assert accuracy > 0.75, "Model accuracy is below the expected threshold."
