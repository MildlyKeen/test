from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import numpy as np
import os

# Load data
data = pd.read_csv('data/heart_failure.csv')
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Initialize models to compare
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Train and evaluate models
best_score = 0
best_model = None
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    mean_score = np.mean(scores)
    print(f"{name} - Mean Accuracy: {mean_score:.3f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_model_name = name

# Train best model on full dataset
best_model.fit(X_scaled, y)

# Save best model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print(f"\nBest model: {best_model_name} with accuracy: {best_score:.3f}")
