import pandas as pd
from imblearn.over_sampling import SMOTE

def balance_data(input_path="data/heart_failure.csv", output_path="data/heart_failure_balanced.csv"):
    df = pd.read_csv(input_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced["target"] = y_resampled
    df_balanced.to_csv(output_path, index=False)
    
    print("✅ Dataset balanced and saved!")

if __name__ == "__main__":
    balance_data()
