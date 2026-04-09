"""
Vercel Python serverless function for housing price prediction.
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import pickle

import numpy as np
import pandas as pd

# Cache loaded models across warm invocations
_model_cache = {}

MODEL_FILES = {
    "gradient_boosting": "gradient_boosting_california_housing_model.pkl",
    "random_forest": "random_forest_california_housing_model.pkl",
    "ridge": "ridge_california_housing_model.pkl",
}

FEATURES = [
    "median_income",
    "housing_median_age",
    "ave_rooms",
    "ave_bedrooms",
    "population",
    "ave_occupancy",
    "latitude",
    "longitude",
]

TREE_MODELS = {"gradient_boosting", "random_forest"}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "trained_models")


def load_model(name: str):
    if name in _model_cache:
        return _model_cache[name]
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    with open(path, "rb") as f:
        model = pickle.load(f)
    _model_cache[name] = model
    return model


def validate_input(data: dict) -> str | None:
    for feat in FEATURES:
        if feat not in data:
            return f"Missing required field: {feat}"
    if not (-124.5 <= data["longitude"] <= -114.0):
        return "Longitude must be between -124.5 and -114.0"
    if not (32.5 <= data["latitude"] <= 42.0):
        return "Latitude must be between 32.5 and 42.0"
    if not (0.5 <= data["median_income"] <= 15.0):
        return "Median income must be between 0.5 and 15.0"
    return None


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))

            # Validate
            error = validate_input(body)
            if error:
                self._json_response(400, {"error": error})
                return

            model_name = body.get("model", "gradient_boosting")
            if model_name not in MODEL_FILES:
                self._json_response(400, {"error": f"Unknown model: {model_name}"})
                return

            # Load model and predict
            model = load_model(model_name)
            input_df = pd.DataFrame([{feat: float(body[feat]) for feat in FEATURES}])
            prediction = float(model.predict(input_df)[0])

            # Calculate derived metrics
            ave_rooms = float(body.get("ave_rooms", 5))
            median_income = float(body.get("median_income", 3))
            price_per_room = prediction / max(ave_rooms, 1)
            income_to_price_ratio = prediction / max(median_income * 10000, 1)

            result = {
                "prediction": round(prediction, 2),
                "model_used": model_name,
                "price_per_room": round(price_per_room, 2),
                "income_to_price_ratio": round(income_to_price_ratio, 2),
            }

            # Confidence interval for tree-based models
            if model_name in TREE_MODELS and hasattr(model, "estimators_"):
                tree_preds = np.array(
                    [t.predict(input_df)[0] for t in model.estimators_]
                )
                std = float(np.std(tree_preds))
                result["confidence_interval"] = {
                    "lower": round(prediction - 1.96 * std, 2),
                    "upper": round(prediction + 1.96 * std, 2),
                }

            self._json_response(200, result)

        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _json_response(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self._json_response(200, {})
