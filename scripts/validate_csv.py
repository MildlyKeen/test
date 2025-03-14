import pandas as pd

# Updated required columns based on the dataset structure
REQUIRED_COLUMNS = ["age", "anaemia", "creatinine_phosphokinase", "diabetes",
                    "ejection_fraction", "high_blood_pressure", "platelets",
                    "serum_creatinine", "serum_sodium", "sex", "smoking", "time", "DEATH_EVENT"]

def validate_csv(file_path="data/heart_failure_clinical_records_dataset.csv"):
    """
    Validates the dataset by checking:
    - If the file exists and can be loaded.
    - If all required columns are present.
    - If there are missing values in the dataset.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        print(f"✅ Loaded dataset: {file_path} with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Check for missing values
        if df.isnull().sum().sum() > 0:
            raise ValueError("❌ ERROR: Dataset contains missing values!")

        # Check if all required columns exist
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"❌ ERROR: Dataset is missing required columns: {missing_columns}")

        print("✅ CSV validation passed!")

    except FileNotFoundError:
        print(f"❌ ERROR: File '{file_path}' not found! Please check the file path.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    validate_csv()
