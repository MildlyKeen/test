from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Create a simple random forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
# Train on dummy data
X = np.random.rand(100, 12)
y = np.random.randint(0, 2, 100)
model.fit(X, y)

# Create a simple scaler
scaler = StandardScaler()
scaler.fit(X)

# Save the model and scaler
joblib.dump(model, 'models/final_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Test model and scaler created successfully")
