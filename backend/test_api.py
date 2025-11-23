import requests
import json

url = 'http://127.0.0.1:5000/predict'

# Sample data (Normal values)
data = {
    'GENDER': 1, 
    'WBC': 7.5, 
    'NE#': 4.5, 
    'LY#': 2.0, 
    'MO#': 0.5, 
    'EO#': 0.2, 
    'BA#': 0.05, 
    'RBC': 4.8, 
    'HGB': 140, 
    'HCT': 42, 
    'MCV': 90, 
    'MCH': 30, 
    'MCHC': 330, 
    'RDW': 13, 
    'PLT': 250, 
    'MPV': 10, 
    'PCT': 0.25, 
    'PDW': 12, 
    'SD': 100, 
    'SDTSD': 30, 
    'TSD': 300, 
    'FERRITTE': 100, 
    'FOLATE': 10, 
    'B12': 400
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
