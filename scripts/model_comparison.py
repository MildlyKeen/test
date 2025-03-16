import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier  # Import XGBoost
from lightgbm import LGBMClassifier  # Import LightGBM
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import numpy as np

# Suppress warnings globally
warnings.filterwarnings('ignore')

def main():
    # Load balanced dataset
    data = pd.read_csv('data/heart_failure_balanced.csv')
    X = data.drop('DEATH_EVENT', axis=1)
    y = data['DEATH_EVENT']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models to compare
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }

    # Compare models
    best_model = None
    best_score = 0
    model_scores = {}

    for name, model in models.items():
        # Perform cross-validation on the training set
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        mean_score = np.mean(scores)
        model_scores[name] = mean_score

        print(f"{name} - Cross-Validation Scores: {scores}")
        print(f"{name} - Mean Cross-Validation Accuracy: {mean_score:.3f}")

        # Track the best model
        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    # Train the best model on the training set
    best_model.fit(X_train_scaled, y_train)

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("\n✅ Best model trained and evaluated!")
    print(f"Best Model: {best_model.__class__.__name__}")
    print(f"Test Set Accuracy: {test_accuracy:.3f}")

    # Print feature importance for RandomForest
    if isinstance(best_model, RandomForestClassifier):
        feature_importances = best_model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print("\nFeature Importances:")
        print(importance_df)

    # Save the best model and scaler
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("\n✅ Best model and scaler saved!")

if __name__ == "__main__":
    main()