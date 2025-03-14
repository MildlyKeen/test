import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_model():
    # Load model and data
    model = joblib.load("models/final_model.pkl")
    df = pd.read_csv("data/heart_failure_balanced.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Final Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
