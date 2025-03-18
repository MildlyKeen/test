import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import shap
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the best model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')

# Ensure the model and scaler files exist
assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
assert os.path.exists(SCALER_PATH), f"Scaler file not found at {SCALER_PATH}"

def explain_model(sample_size=100):
    """
    Generate SHAP values for the best model using a synthetic dataset.
    """
    # Load the model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Create a synthetic dataset for testing
    X = pd.DataFrame({
        "age": np.random.randint(18, 100, size=1000),
        "anaemia": np.random.randint(0, 2, size=1000),
        "creatinine_phosphokinase": np.random.randint(10, 10000, size=1000),
        "diabetes": np.random.randint(0, 2, size=1000),
        "ejection_fraction": np.random.randint(10, 80, size=1000),
        "high_blood_pressure": np.random.randint(0, 2, size=1000),
        "platelets": np.random.uniform(50000, 500000, size=1000),
        "serum_creatinine": np.random.uniform(0.1, 10, size=1000),
        "serum_sodium": np.random.uniform(120, 150, size=1000),
        "sex": np.random.randint(0, 2, size=1000),
        "smoking": np.random.randint(0, 2, size=1000),
        "time": np.random.randint(0, 365, size=1000),
    })

    # Scale the dataset
    X_scaled = scaler.transform(X)

    # Select a sample
    X_sample = pd.DataFrame(X_scaled, columns=X.columns).sample(n=sample_size, random_state=42)

    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return shap_values, X_sample

def test_shap_values():
    """
    Test that SHAP values are generated correctly.
    """
    shap_values, X_sample = explain_model(sample_size=100)
    assert shap_values is not None, "SHAP values are None"
    assert X_sample is not None, "Sample data is None"
    assert len(shap_values) > 0, "SHAP values not generated"
    assert len(X_sample) == 100, "Incorrect sample size"
    print("✅ SHAP values test passed!")

def test_shap_plot():
    """
    Test that SHAP plotting works without errors.
    """
    shap_values, X_sample = explain_model(sample_size=50)
    try:
        # Save the SHAP plot to a file
        output_file = "test_shap_summary_plot.png"
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(output_file)
        assert os.path.exists(output_file), "SHAP plot file not created"
        os.remove(output_file)  # Clean up after test
        print("✅ SHAP plot test passed!")
    except Exception as e:
        pytest.fail(f"SHAP plotting failed with error: {str(e)}")

# Add a pytest hook to display a message after all tests pass
def pytest_sessionfinish(session, exitstatus):
    """
    Display a message after all tests pass.
    """
    if exitstatus == 0:
        print("\n🎉 All tests passed successfully!")
