import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/heart_failure_balanced.csv")
X = df.drop(columns=["DEATH_EVENT"])

model = joblib.load("models/final_model.pkl")
explainer = shap.Explainer(model, X)
shap_values = explainer(X[:100])

shap.summary_plot(shap_values, X)
plt.savefig("models/shap_summary.png")
print("✅ SHAP analysis complete!")
