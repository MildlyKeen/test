import pytest
import shap
from explainability import explain_model

def test_shap_values():
    model, X_train = explain_model('models/final_model.pkl', 'data/heart_failure_balanced.csv')
    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    assert len(shap_values) > 0, "SHAP values not generated."
    assert isinstance(shap_values, list), "SHAP values should be in a list format."

def test_shap_plot():
    # Check that SHAP plotting works correctly (e.g., for a summary plot)
    model, X_train = explain_model('models/final_model.pkl', 'data/heart_failure_balanced.csv')
    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    shap.summary_plot(shap_values, X_train)  # This will generate a plot, so a visual check may be required
    assert True  # If no errors are raised, the test is considered successful
