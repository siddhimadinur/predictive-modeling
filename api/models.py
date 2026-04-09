"""
Vercel Python serverless function to return model metadata.
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import csv


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "trained_models")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Read deployment summary
            summary_path = os.path.join(MODEL_DIR, "deployment_summary.json")
            with open(summary_path) as f:
                summary = json.load(f)

            # Read evaluation CSV
            eval_path = os.path.join(MODEL_DIR, "california_housing_model_evaluation.csv")
            models = []
            with open(eval_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    models.append({
                        "name": row.get("model_name", ""),
                        "val_rmse": float(row.get("val_rmse", 0)),
                        "val_r2": float(row.get("val_r2", 0)),
                        "val_mae": float(row.get("val_mae", 0)),
                        "status": row.get("status", ""),
                    })

            result = {
                "models": models,
                "champion": summary.get("champion_model", ""),
                "training_samples": summary.get("training_samples", 0),
                "feature_count": summary.get("feature_count", 8),
            }

            self._json_response(200, result)

        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _json_response(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
