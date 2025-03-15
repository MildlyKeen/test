import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import shap
from explainability import explain_model

def test_shap_values():
    # Test that SHAP values are generated correctly
    shap_values, X_sample = explain_model(sample_size=100)
    assert shap_values is not None
    assert X_sample is not None
    assert len(shap_values) > 0, "SHAP values not generated"
    assert len(X_sample) == 100, "Incorrect sample size"

def test_shap_plot():
    # Test that SHAP plotting works without errors
    shap_values, X_sample = explain_model(sample_size=50)
    try:
        shap.summary_plot(shap_values, X_sample)
        assert True
    except Exception as e:
        pytest.fail(f"SHAP plotting failed with error: {str(e)}")
