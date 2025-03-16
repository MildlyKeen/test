import pytest
import os
import joblib

# Paths to the model and scaler
MODEL_PATH = os.path.join("models", "best_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

def test_model_file_exists():
    """
    Test if the model file exists.
    """
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    print("✅ Model file exists test passed!")

def test_scaler_file_exists():
    """
    Test if the scaler file exists.
    """
    assert os.path.exists(SCALER_PATH), f"Scaler file not found at {SCALER_PATH}"
    print("✅ Scaler file exists test passed!")

def test_model_loading():
    """
    Test loading the saved model.
    """
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Failed to load the model"
    print("✅ Model loading test passed!")

def test_scaler_loading():
    """
    Test loading the saved scaler.
    """
    scaler = joblib.load(SCALER_PATH)
    assert scaler is not None, "Failed to load the scaler"
    print("✅ Scaler loading test passed!")

# Add a pytest hook to display a message after all tests pass
def pytest_sessionfinish(session, exitstatus):
    """
    Display a message after all tests pass.
    """
    if exitstatus == 0:
        print("\n🎉 All tests passed successfully!")

if __name__ == "__main__":
    pytest.main()
