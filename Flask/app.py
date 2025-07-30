from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler from Flask folder
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    raise

categorical_mappings = {
    'Stages': {0: 'HYPERTENSION (Stage-1)', 1: 'HYPERTENSION (Stage-2)',
               2: 'HYPERTENSIVE CRISIS', 3: 'NORMAL'}
}


# Serve static files (e.g., favicon.ico)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


# Serve index page
@app.route('/')
def index():
    return render_template('index.html')


# Serve details page (form)
@app.route('/details')
def details():
    return render_template('details.html')


# Serve results page
@app.route('/results')
def results():
    return render_template('results.html')


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        feature_names = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication', 'Severity',
                         'BreathShortness', 'VisualChanges', 'NoseBleeding', 'Whendiagnoused',
                         'Systolic', 'Diastolic', 'ControlledDiet']

        # Validate all required features are present
        missing_features = [f for f in feature_names if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        # Validate data types and values
        for feature in feature_names:
            if not isinstance(data[feature], (int, float)):
                return jsonify({'error': f'Feature {feature} must be a number, got {type(data[feature])}'}), 400

        # Validate categorical features
        binary_features = ['Gender', 'History', 'Patient', 'TakeMedication', 'BreathShortness',
                           'VisualChanges', 'NoseBleeding', 'ControlledDiet']
        for feature in binary_features:
            if data[feature] not in [0, 1]:
                return jsonify({'error': f'Feature {feature} must be 0 or 1, got {data[feature]}'}), 400

        if data['Severity'] not in [0, 1, 2]:
            return jsonify({'error': f'Severity must be 0, 1, or 2, got {data["Severity"]}'}), 400

        if data['Whendiagnoused'] not in [0, 1, 2]:
            return jsonify({'error': f'Whendiagnoused must be 0, 1, or 2, got {data["Whendiagnoused"]}'}), 400

        # Create DataFrame
        input_df = pd.DataFrame([data], columns=feature_names)

        # Scale numerical features
        numerical_cols = ['Age', 'Systolic', 'Diastolic']
        input_df_scaled = input_df.copy()
        input_df_scaled[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Predict
        prediction = model.predict(input_df_scaled)[0]
        prediction_decoded = categorical_mappings['Stages'][prediction]

        # Get probabilities
        probabilities = model.predict_proba(input_df_scaled)[0]
        prob_dict = {categorical_mappings['Stages'][i]: float(prob) for i, prob in enumerate(probabilities)}

        return jsonify({
            'prediction': prediction_decoded,
            'probabilities': prob_dict
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(debug=True)