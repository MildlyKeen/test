import logging
from flask import Flask, request, render_template_string, redirect, url_for, session

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Clé secrète pour gérer les sessions

# Simuler une base de données d'utilisateurs (médecins)
USERS = {"medecin@example.com": "password123"}

# Définition des caractéristiques pour le modèle
FEATURES = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", 
           "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", 
           "sex", "smoking", "time"]

# Créer un modèle simple si aucun modèle n'existe
MODEL_PATH = "c:/Users/mahmo/Documents/GitHub/test/models/best_model.pkl"
SCALER_PATH = "c:/Users/mahmo/Documents/GitHub/test/models/scaler.pkl"

# Charger le modèle et le scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    print(f"Model files not found: {e}")
    model = None
    scaler = None

# Fonction pour générer un diagramme circulaire
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
        app.logger.debug('Login route accessed')
        if request.method == 'POST':
            app.logger.debug('POST request received')
            email = request.form.get('email')
            password = request.form.get('password')
            
            if not email or not password:
                app.logger.warning('Missing email or password')
                return render_template_string(LOGIN_TEMPLATE, error="Email et mot de passe requis.")
                
            if email in USERS and USERS[email] == password:
                app.logger.debug(f'Successful login for {email}')
                session['logged_in'] = True
                return redirect(url_for('predict'))
            
            app.logger.warning(f'Failed login attempt for {email}')
            return render_template_string(LOGIN_TEMPLATE, error="Identifiants incorrects.")
        
        app.logger.debug('Rendering login template')
        return render_template_string(LOGIN_TEMPLATE)
    except Exception as e:
        app.logger.error(f'Error in login route: {str(e)}')
        return render_template_string(LOGIN_TEMPLATE, error="Une erreur s'est produite. Veuillez réessayer.")

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

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
                value = request.form[feature]
                if feature == 'sex':
                    input_data[feature] = 'Male' if float(value) == 1 else 'Female'
                elif feature in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking']:
                    if float(value) not in [0, 1]:
                        raise ValueError(f"{feature} must be 0 or 1")
                    input_data[feature] = 'Yes' if float(value) == 1 else 'No'
                else:
                    input_data[feature] = float(value)
            form_values = input_data
            input_df = pd.DataFrame([input_data])
            
            if model is None or scaler is None:
                raise Exception("Model not loaded")
                
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0, 1]
            pie_chart_image = generate_pie_chart(probability)
            
        except Exception as e:
            for feature in FEATURES:
                if feature in request.form:
                    form_values[feature] = request.form[feature]
            
            return render_template_string(PREDICT_TEMPLATE, features=FEATURES, 
                                       error=f"Erreur: {str(e)}", 
                                       form_values=form_values)

    return render_template_string(PREDICT_TEMPLATE, features=FEATURES, 
                               prediction=prediction, probability=probability,
                               pie_chart_image=pie_chart_image,
                               form_values=form_values)

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
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px;
      background-color: #f5f5f5;
    }
    .login-container {
      max-width: 400px;
      margin: 100px auto;
      padding: 20px;
      background: white;
      border-radius: 5px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    h1 {
      text-align: center;
      color: #333;
    }
    .form-group {
      margin-bottom: 15px;
    }
    label {
      display: block;
      margin-bottom: 5px;
    }
    input[type="email"], input[type="password"] {
      width: 100%;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 3px;
    }
    button {
      width: 100%;
      padding: 10px;
      background-color: #007bff;
      color: white;
      border: none;
      border-radius: 3px;
      cursor: pointer;
    }
    .error {
      color: red;
      margin-top: 10px;
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
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px;
      background-color: #f5f5f5;
    }
    .container {
      max-width: 800px;
      margin: 50px auto;
      padding: 20px;
      background: white;
      border-radius: 5px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
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
      color: #555;
    }
    input[type="number"] {
      width: 100%;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 3px;
    }
    button {
      grid-column: span 2;
      padding: 12px;
      background-color: #007bff;
      color: white;
      border: none;
      border-radius: 3px;
      cursor: pointer;
      font-size: 16px;
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
          <label for="{{ feature }}">{{ feature.replace('_', ' ').title() }}:</label>
          {% if feature == 'sex' %}
            <select id="{{ feature }}" name="{{ feature }}" required>
              <option value="1" {% if form_values[feature] == 'Male' %}selected{% endif %}>Male</option>
              <option value="0" {% if form_values[feature] == 'Female' %}selected{% endif %}>Female</option>
            </select>
          {% elif feature in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking'] %}
            <input type="number" id="{{ feature }}" name="{{ feature }}" 
                   min="0" max="1" step="1" required>
          {% else %}
            <input type="number" step="any" id="{{ feature }}" name="{{ feature }}" 
                   value="{{ form_values[feature] }}" required>
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

    <div class="logout">
      <a href="{{ url_for('logout') }}">Déconnexion</a>
    </div>
  </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
