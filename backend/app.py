from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and scaler
# Assuming models are in ./model and ./scaler relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_model_smote.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler', 'scaler.pkl')

try:
    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Load model (saved as dict with model info)
    model_data = joblib.load(MODEL_PATH)
    if isinstance(model_data, dict):
        model = model_data['model']  # 提取模型对象
        print(f"Model type: {model_data.get('model_name', 'Unknown')}")
    else:
        model = model_data

    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Feature columns (15个简化特征)
FEATURE_COLS = [
    'HGB', 'HCT', 'RBC', 'RDW', 'MCH', 'MCHC',
    'MCV', 'SD', 'TSD', 'PLT', 'LY#', 'PCT',
    'PDW', 'FOLATE', 'NE#'
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

        # Create DataFrame for scaling
        features_array = np.array([features])
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Map class to label (Chinese + English format)
        # 类别: 0: 无贫血, 1: 血红蛋白贫血, 2: 缺铁性贫血, 3: 叶酸缺乏症, 4: 维生素B12缺乏症
        class_labels = {
            0: "无贫血(No Anemia)",
            1: "血红蛋白贫血(HGB Anemia)",
            2: "缺铁性贫血(Iron Deficiency Anemia)",
            3: "叶酸缺乏症(Folate Deficiency Anemia)",
            4: "维生素B12缺乏症(B12 Deficiency Anemia)"
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
