#!/usr/bin/env python3
"""
End-to-end training script for California Housing price prediction models.

Usage:
    PYTHONPATH=. python scripts/train_models.py
"""
import sys
import json
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, RANDOM_STATE
from src.data_loader import load_california_housing_data
from src.data_pipeline import CaliforniaHousingPipeline
from src.model_training import CaliforniaHousingModelTrainer


def main():
    start_time = time.time()

    # ── Ticket 1: Load & validate the full California Housing dataset ────
    print("=" * 70)
    print("TICKET 1: LOAD & VALIDATE DATASET")
    print("=" * 70)

    train_data, test_data = load_california_housing_data()

    print(f"\nDataset loaded:")
    print(f"  Train shape: {train_data.shape}")
    print(f"  Test shape:  {test_data.shape}")
    print(f"  Columns:     {list(train_data.columns)}")

    assert train_data.shape[0] > 10000, f"Expected 10K+ training rows, got {train_data.shape[0]}"
    assert "median_house_value" in train_data.columns, "Target column 'median_house_value' missing"
    print("\n[OK] Dataset validation passed.")

    # ── Ticket 2: Run data processing & feature engineering pipeline ─────
    print("\n" + "=" * 70)
    print("TICKET 2: DATA PROCESSING & FEATURE ENGINEERING")
    print("=" * 70)

    pipeline = CaliforniaHousingPipeline(train_data, test_data, target_col="median_house_value")
    pipeline_results = pipeline.run_pipeline(save_processed=True)

    X_train = pipeline.X_train
    X_val = pipeline.X_val
    y_train = pipeline.y_train
    y_val = pipeline.y_val

    print(f"\nPipeline results:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  Features: {X_train.shape[1]}")

    assert X_train.shape[1] > 9, f"Expected engineered features (>9), got {X_train.shape[1]}"
    print("[OK] Feature engineering validation passed.")

    # ── Ticket 3: Train all registered models ────────────────────────────
    print("\n" + "=" * 70)
    print("TICKET 3: TRAIN ALL MODELS")
    print("=" * 70)

    trainer = CaliforniaHousingModelTrainer(X_train, y_train, X_val, y_val)
    trainer.register_california_housing_models()
    trained_models = trainer.train_all_models(tune_hyperparameters=False, cv_folds=5)

    assert len(trained_models) >= 5, f"Expected 5+ trained models, got {len(trained_models)}"
    print(f"\n[OK] {len(trained_models)} models trained successfully.")

    # ── Ticket 4: Evaluate models & select champion ──────────────────────
    print("\n" + "=" * 70)
    print("TICKET 4: EVALUATE MODELS & SELECT CHAMPION")
    print("=" * 70)

    evaluation_df = trainer.evaluate_models()
    print("\nModel Evaluation Results:")
    print(evaluation_df[["model", "val_rmse", "val_r2", "val_mae", "cv_r2"]].to_string(index=False))

    best_name, best_model = trainer.get_best_model()
    best_row = evaluation_df[evaluation_df["model"] == best_name].iloc[0]

    print(f"\nChampion model: {best_name}")
    print(f"  Val R2:   {best_row['val_r2']:.4f}")
    print(f"  Val RMSE: ${best_row['val_rmse']:,.0f}")
    print(f"  Val MAE:  ${best_row['val_mae']:,.0f}")

    # Save evaluation CSV
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(MODELS_DIR / "california_housing_model_evaluation.csv", index=False)
    print(f"\n[OK] Evaluation CSV saved.")

    # ── Ticket 5: Serialize & save all models ────────────────────────────
    print("\n" + "=" * 70)
    print("TICKET 5: SERIALIZE & SAVE ALL MODELS")
    print("=" * 70)

    trainer.save_models()

    # Update deployment_summary.json with real results
    deployment_summary = {
        "champion_model": f"{best_name}_california_housing.pkl",
        "champion_r2": float(best_row["val_r2"]),
        "champion_rmse": float(best_row["val_rmse"]),
        "champion_mae": float(best_row["val_mae"]),
        "dataset": "california_housing",
        "training_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "target_column": "median_house_value",
        "all_models_trained": list(trained_models.keys()),
    }

    with open(MODELS_DIR / "deployment_summary.json", "w") as f:
        json.dump(deployment_summary, f, indent=2)

    # Verify files were created
    pkl_files = list(MODELS_DIR.glob("*california_housing*.pkl"))
    print(f"\nSaved {len(pkl_files)} .pkl files:")
    for p in sorted(pkl_files):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name} ({size_kb:.1f} KB)")

    assert len(pkl_files) >= 5, f"Expected 5+ .pkl files, got {len(pkl_files)}"
    print("[OK] All models serialized.")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time:     {elapsed:.1f}s")
    print(f"  Models trained: {len(trained_models)}")
    print(f"  Champion:       {best_name} (R2={best_row['val_r2']:.4f})")
    print(f"  Output dir:     {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
