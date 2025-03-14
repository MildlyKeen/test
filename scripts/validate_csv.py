import pandas as pd

REQUIRED_COLUMNS = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", 
                    "ejection_fraction", "platelets", "serum_creatinine", 
                    "serum_sodium", "sex", "smoking", "time", "target"]

def validate_csv(file_path="data/heart_failure.csv"):
    df = pd.read_csv(file_path)
    
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values!")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    print("✅ CSV validation passed!")

if __name__ == "__main__":
    validate_csv()
