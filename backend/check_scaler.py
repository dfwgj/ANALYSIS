import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, '..', 'output', 'scaler.pkl')

try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded.")
    if hasattr(scaler, 'feature_names_in_'):
        print("Feature names:", list(scaler.feature_names_in_))
    else:
        print("Scaler has no feature_names_in_ attribute.")
        print("n_features_in_:", scaler.n_features_in_)
except Exception as e:
    print(f"Error: {e}")
