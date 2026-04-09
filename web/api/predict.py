"""
Vercel Python serverless function for California housing price prediction.
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import joblib
import numpy as np

# Load model at cold start
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "gradient_boosting_california_housing_model.pkl")
model_data = joblib.load(MODEL_PATH)

# Handle both dict-wrapped and raw model formats
if isinstance(model_data, dict) and "model" in model_data:
    model = model_data["model"]
    feature_names = model_data.get("feature_names", [])
else:
    model = model_data
    feature_names = [
        "median_income", "housing_median_age", "ave_rooms",
        "ave_bedrooms", "population", "ave_occupancy",
        "latitude", "longitude",
    ]

INFLATION_MULTIPLIER = 4.5


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            # Build feature array in the correct order
            features = [float(body.get(f, 0)) for f in feature_names]
            X = np.array([features])

            prediction = float(model.predict(X)[0])
            adjusted = prediction * INFLATION_MULTIPLIER

            result = {
                "prediction_1990": round(prediction),
                "prediction_2024": round(adjusted),
                "model": "gradient_boosting",
                "features_used": feature_names,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
