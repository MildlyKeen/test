import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(data_path="data/heart_failure_balanced.csv"):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    best_model = None
    best_score = 0
    model_scores = {}

    # Train each model and evaluate performance
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        
        # Save model scores
        model_scores[name] = score

        print(f"{name} Model Accuracy: {score:.4f}")
        
        # Track best model
        if score > best_score:
            best_score = score
            best_model = model

    # Save the best model
    joblib.dump(best_model, "models/final_model.pkl")
    print("✅ Best model trained and saved!")

    return best_model

if __name__ == "__main__":
    best_model = train_model()
    print("\nModel trained and saved!")
