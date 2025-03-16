import logging
from flask import Flask, request, render_template_string, redirect, url_for, session
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Simulated user database
USERS = {"medecin@example.com": "password123"}

# Model features
FEATURES = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
            "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
            "sex", "smoking", "time"]

# Valid ranges for numeric inputs
VALID_RANGES = {
    "age": (18, 100),
    "anaemia": (0, 1),  # Binary
    "creatinine_phosphokinase": (10, 10000),  # U/L
    "diabetes": (0, 1),  # Binary
    "ejection_fraction": (10, 80),  # Percentage
    "high_blood_pressure": (0, 1),  # Binary
    "platelets": (50000, 500000),  # per µL
    "serum_creatinine": (0.1, 10),  # mg/dL
    "serum_sodium": (110, 150),  # mmol/L
    "sex": (0, 1),  # Binary
    "smoking": (0, 1),  # Binary
    "time": (0, 365)  # Days
}

# Units of measure for each feature
UNITS = {
    "age": "years",
    "anaemia": "",
    "creatinine_phosphokinase": "U/L",
    "diabetes": "",
    "ejection_fraction": "%",
    "high_blood_pressure": "",
    "platelets": "per µL",
    "serum_creatinine": "mg/dL",
    "serum_sodium": "mmol/L",
    "sex": "",
    "smoking": "",
    "time": "days"
}

# Model and scaler paths
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    print(f"Model files not found: {e}")
    model = None
    scaler = None

# Generate pie chart
def generate_pie_chart(probability):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Risque élevé', 'Risque faible']
        sizes = [probability * 100, (1 - probability) * 100]
        colors = ['#ff6b6b', '#4ecdc4']
        explode = (0.1, 0)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        ax.set_title('Probabilité de risque cardiaque', fontsize=16)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64
    finally:
        plt.close('all')
        buf.close()

@app.route('/', methods=['GET', 'POST'])
def login():
    try:
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            if not email or not password:
                return render_template_string(LOGIN_TEMPLATE, error="Email et mot de passe requis.")
            if email in USERS and USERS[email] == password:
                session['logged_in'] = True
                return redirect(url_for('predict'))
            return render_template_string(LOGIN_TEMPLATE, error="Identifiants incorrects.")
        return render_template_string(LOGIN_TEMPLATE)
    except Exception as e:
        return render_template_string(LOGIN_TEMPLATE, error="Une erreur s'est produite.")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    prediction = None
    probability = None
    pie_chart_image = None
    form_values = {feature: "" for feature in FEATURES}

    if request.method == 'POST':
        try:
            input_data = {}
            for feature in FEATURES:
                value = request.form.get(feature)
                if not value:
                    raise ValueError(f"Missing value for {feature}")
                if feature == 'sex':
                    input_data[feature] = 1 if value == '1' else 0
                elif feature in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking']:
                    input_data[feature] = int(value)
                else:
                    value = float(value)
                    logging.debug(f"Received {feature} value: {value}")
                    if feature in VALID_RANGES:
                        min_val, max_val = VALID_RANGES[feature]
                        if value < min_val or value > max_val:
                            raise ValueError(f"{feature} must be between {min_val} and {max_val}")
                    input_data[feature] = value

            form_values = input_data
            input_df = pd.DataFrame([input_data])

            if model is None or scaler is None:
                raise Exception("Model or scaler not loaded")

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0, 1]
            pie_chart_image = generate_pie_chart(probability)

        except Exception as e:
            return render_template_string(PREDICT_TEMPLATE, features=FEATURES,
                                          error=f"Erreur: {str(e)}",
                                          form_values=form_values,
                                          valid_ranges=VALID_RANGES,
                                          units=UNITS,
                                          prediction=prediction,
                                          probability=probability,
                                          pie_chart_image=pie_chart_image)

    return render_template_string(PREDICT_TEMPLATE, features=FEATURES,
                                  prediction=prediction, probability=probability,
                                  pie_chart_image=pie_chart_image,
                                  form_values=form_values,
                                  valid_ranges=VALID_RANGES,
                                  units=UNITS)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

LOGIN_TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Connexion Médecin</title>
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background: url('https://via.placeholder.com/1920x1080?text=Medicine+Background') no-repeat center center fixed;
      background-size: cover;
      margin: 0;
      padding: 0;
    }
    .login-container {
      max-width: 400px;
      margin: 100px auto;
      padding: 20px;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 10px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    h1 {
      text-align: center;
      color: #333;
      margin-bottom: 20px;
    }
    .form-group {
      margin-bottom: 15px;
    }
    label {
      display: block;
      margin-bottom: 5px;
      font-weight: bold;
    }
    input[type="email"], input[type="password"] {
      width: 100%;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 5px;
    }
    button {
      width: 100%;
      padding: 10px;
      background-color: #007bff;
      color: white;
      border: none;
      border-radius: 5px;
      cursor: pointer;
    }
    button:hover {
      background-color: #0056b3;
    }
    .error {
      color: red;
      margin-top: 10px;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="login-container">
    <h1>Connexion Médecin</h1>
    <form method="POST">
      <div class="form-group">
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
      </div>
      <div class="form-group">
        <label for="password">Mot de passe:</label>
        <input type="password" id="password" name="password" required>
      </div>
      <button type="submit">Se connecter</button>
      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}
    </form>
  </div>
</body>
</html>
"""

PREDICT_TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Prédiction de Risque Cardiaque</title>
  <style>
    body {
      font-family: 'Roboto', sans-serif;
      background: url('https://via.placeholder.com/1920x1080?text=Medicine+Background') no-repeat center center fixed;
      background-size: cover;
      margin: 0;
      padding: 0;
    }
    .container {
      max-width: 800px;
      margin: 50px auto;
      padding: 20px;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 10px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    h1 {
      text-align: center;
      color: #333;
      margin-bottom: 30px;
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
    }
    .form-group {
      margin-bottom: 15px;
    }
    label {
      display: block;
      margin-bottom: 5px;
      font-weight: bold;
    }
    input[type="number"], select {
      width: 100%;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 5px;
    }
    button {
      grid-column: span 2;
      padding: 12px;
      background-color: #007bff;
      color: white;
      border: none;
      border-radius: 5px;
      cursor: pointer;
    }
    button:hover {
      background-color: #0056b3;
    }
    .results {
      margin-top: 30px;
      padding: 20px;
      background-color: #f8f9fa;
      border-radius: 5px;
    }
    .error {
      color: red;
      margin-top: 10px;
    }
    .logout {
      text-align: right;
      margin-top: 20px;
    }
    .logout a {
      color: #007bff;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Prédiction de Risque Cardiaque</h1>
    <form method="POST">
      <div class="form-grid">
        {% for feature in features %}
        <div class="form-group">
          <label for="{{ feature }}">{{ feature.replace('_', ' ').title() }} ({{ valid_ranges[feature][0] }} - {{ valid_ranges[feature][1] }} {{ units.get(feature, '') }}):</label>
          {% if feature == 'sex' %}
            <select id="{{ feature }}" name="{{ feature }}" required>
              <option value="1" {% if form_values[feature] == 1 %}selected{% endif %}>Male</option>
              <option value="0" {% if form_values[feature] == 0 %}selected{% endif %}>Female</option>
            </select>
          {% elif feature in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking'] %}
            <select id="{{ feature }}" name="{{ feature }}" required>
              <option value="1" {% if form_values[feature] == 1 %}selected{% endif %}>Yes</option>
              <option value="0" {% if form_values[feature] == 0 %}selected{% endif %}>No</option>
            </select>
          {% else %}
            <input type="number" id="{{ feature }}" name="{{ feature }}" 
                   value="{{ form_values[feature] }}" 
                   min="{{ valid_ranges[feature][0] }}" 
                   max="{{ valid_ranges[feature][1] }}" 
                   step="any" required>
          {% endif %}
        </div>
        {% endfor %}
      </div>
      <button type="submit">Prédire le Risque</button>
    </form>
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
    {% if prediction is not none %}
      <div class="results">
        <h2>Résultat:</h2>
        <p>Prédiction: 
          {% if prediction == 1 %}
            <span style="color: red;">Risque Élevé</span>
          {% else %}
            <span style="color: green;">Risque Faible</span>
          {% endif %}
        </p>
        <p>Probabilité: {{ (probability * 100)|round(2) }}%</p>
        {% if pie_chart_image %}
          <img src="data:image/png;base64,{{ pie_chart_image }}" 
               alt="Graphique de Risque" style="max-width: 100%;">
        {% endif %}
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)