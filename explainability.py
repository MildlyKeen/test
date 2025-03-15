import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def explain_model(sample_size: int = 100) -> Tuple:
    """Explain model predictions using SHAP values
    
    Args:
        sample_size: Number of samples to explain
        
    Returns:
        Tuple containing (shap_values, X_sample)
    """
    try:
        # Load data and model
        df = pd.read_csv("data/heart_failure_balanced.csv")
        X = df.drop(columns=["DEATH_EVENT"])
        model = joblib.load("models/final_model.pkl")

        # Create SHAP explainer based on model type
        if str(type(model)).endswith("XGBClassifier'>"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X)

        # Explain a subset of the data
        sample_size = min(sample_size, len(X))
        X_sample = X.iloc[:sample_size]
        shap_values = explainer(X_sample)

        # Create and save SHAP summary plot
        shap.summary_plot(shap_values, X_sample)
        plt.savefig("models/shap_summary.png")
        plt.close()
        
        print("✅ SHAP analysis complete!")
        return shap_values, X_sample
        
    except Exception as e:
        print(f"❌ Error in explain_model: {str(e)}")
        raise

if __name__ == "__main__":
    explain_model()
