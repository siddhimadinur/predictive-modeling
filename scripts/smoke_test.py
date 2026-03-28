#!/usr/bin/env python3
"""
Smoke test: verify trained models load and serve predictions.

Simulates what the Streamlit app does without requiring the Streamlit runtime.

Usage:
    PYTHONPATH=. python3 scripts/smoke_test.py
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"


def main():
    print("=" * 60)
    print("SMOKE TEST: Model Loading & Prediction")
    print("=" * 60)

    # 1. Verify deployment_summary.json
    summary_path = MODELS_DIR / "deployment_summary.json"
    assert summary_path.exists(), "deployment_summary.json missing"
    with open(summary_path) as f:
        deployment = json.load(f)
    print(f"\n[OK] deployment_summary.json loaded")
    print(f"     Champion: {deployment['champion_model']}")
    print(f"     R2: {deployment['champion_r2']:.4f}")
    print(f"     Models: {deployment['all_models_trained']}")

    # 2. Load all .pkl model files (same glob pattern as model_loader.py)
    pkl_files = sorted(MODELS_DIR.glob("*california_housing*.pkl"))
    # Exclude the training summary pkl
    model_files = [p for p in pkl_files if "training_summary" not in p.name]
    print(f"\n[OK] Found {len(model_files)} model .pkl files")

    models = {}
    feature_names = None
    for mf in model_files:
        data = joblib.load(mf)
        if isinstance(data, dict) and "model" in data:
            name = data.get("name", mf.stem.replace("_california_housing", ""))
            models[name] = data["model"]
            if data.get("feature_names") and feature_names is None:
                feature_names = data["feature_names"]
            print(f"     Loaded: {name} ({mf.stat().st_size / 1024:.1f} KB)")
        else:
            name = mf.stem.replace("_california_housing", "")
            models[name] = data
            print(f"     Loaded (raw): {name}")

    assert len(models) >= 5, f"Expected 5+ models, got {len(models)}"
    assert feature_names is not None, "No feature names found in any model"
    print(f"\n[OK] {len(models)} models loaded, {len(feature_names)} features expected")

    # 3. Build a sample SF Bay Area input and align to training features
    sample_input = {
        "longitude": -122.4,
        "latitude": 37.8,
        "housing_median_age": 30.0,
        "total_rooms": 3000.0,
        "total_bedrooms": 500.0,
        "population": 1500.0,
        "households": 600.0,
        "median_income": 8.0,
        "ocean_proximity": "NEAR BAY",
    }

    input_df = pd.DataFrame([{f: 0.0 for f in feature_names}])
    for col in sample_input:
        if col in input_df.columns:
            input_df[col] = sample_input[col]

    # 4. Run predictions with every model
    print(f"\nPredictions for SF Bay Area sample:")
    print(f"  {'Model':<30} {'Prediction':>15}")
    print(f"  {'-'*30} {'-'*15}")
    for name, model in models.items():
        pred = model.predict(input_df)[0]
        assert np.isfinite(pred), f"{name} returned non-finite prediction"
        print(f"  {name:<30} ${pred:>14,.0f}")

    # 5. Verify evaluation CSV
    eval_csv = MODELS_DIR / "california_housing_model_evaluation.csv"
    assert eval_csv.exists(), "Evaluation CSV missing"
    eval_df = pd.read_csv(eval_csv)
    print(f"\n[OK] Evaluation CSV has {len(eval_df)} rows")

    # 6. Verify training summary pkl
    ts_path = MODELS_DIR / "california_housing_training_summary.pkl"
    assert ts_path.exists(), "Training summary pkl missing"
    print("[OK] Training summary pkl exists")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
