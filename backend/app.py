from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and scaler
# Assuming models are in ../output/ relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'output', 'best_model_smote.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'output', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Feature columns in the correct order
FEATURE_COLS = [
    'GENDER', 'WBC', 'NE#', 'LY#', 'MO#', 'EO#', 'BA#', 'RBC', 'HGB', 'HCT', 
    'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'MPV', 'PCT', 'PDW', 'SD', 'SDTSD', 
    'TSD', 'FERRITTE', 'FOLATE', 'B12'
]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        # Extract features in the correct order
        features = []
        for col in FEATURE_COLS:
            val = data.get(col)
            if val is None:
                return jsonify({"error": f"Missing feature: {col}"}), 400
            features.append(float(val))
            
        # Add missing features that were accidentally included in training (Data Leakage in original model)
        # 'Folate_anemia_class' and 'B12_Anemia_class' were not excluded due to case sensitivity in training script
        # We set them to 0 as we don't know the diagnosis yet
        features.append(0.0) # Folate_anemia_class
        features.append(0.0) # B12_Anemia_class
            
        # Create DataFrame for scaling (to match training structure if needed, or just numpy array)
        # The scaler expects a 2D array
        features_array = np.array([features])
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Map class to label (if available, otherwise just return class index)
        # Classes: 0: No anemia, 1: HGB-anemia, 2: Iron deficiency, 3: Folate deficiency, 4: B12 deficiency
        class_labels = {
            0: "No Anemia",
            1: "HGB Anemia",
            2: "Iron Deficiency Anemia",
            3: "Folate Deficiency Anemia",
            4: "B12 Deficiency Anemia"
        }
        
        result = {
            "class_id": int(prediction),
            "class_label": class_labels.get(int(prediction), "Unknown"),
            "probability": float(probabilities[int(prediction)]),
            "all_probabilities": {class_labels[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
