"""Integration tests: verify the saved artifacts from the training run."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from config.settings import MODELS_DIR


# ── Artifact Existence ───────────────────────────────────────────────────

class TestArtifactsExist:
    """Verify all expected files were produced by the training script."""

    def test_models_dir_exists(self, models_dir):
        assert models_dir.exists()

    def test_deployment_summary_exists(self, models_dir):
        assert (models_dir / "deployment_summary.json").exists()

    def test_evaluation_csv_exists(self, models_dir):
        assert (models_dir / "california_housing_model_evaluation.csv").exists()

    def test_training_summary_pkl_exists(self, models_dir):
        assert (models_dir / "california_housing_training_summary.pkl").exists()

    def test_at_least_five_model_pkl_files(self, models_dir):
        pkl_files = [
            p for p in models_dir.glob("*california_housing*.pkl")
            if "training_summary" not in p.name
        ]
        assert len(pkl_files) >= 5, f"Only found {len(pkl_files)} model .pkl files"

    def test_processed_data_exists(self, processed_data_dir):
        assert (processed_data_dir / "X_train.csv").exists()
        assert (processed_data_dir / "X_val.csv").exists()
        assert (processed_data_dir / "y_train.csv").exists()
        assert (processed_data_dir / "y_val.csv").exists()

    def test_pipeline_artifacts_exist(self, processed_data_dir):
        assert (processed_data_dir / "pipeline_artifacts.pkl").exists()


# ── Deployment Summary ───────────────────────────────────────────────────

class TestDeploymentSummary:

    @pytest.fixture(autouse=True)
    def _load(self, models_dir):
        with open(models_dir / "deployment_summary.json") as f:
            self.summary = json.load(f)

    def test_has_champion_model(self):
        assert "champion_model" in self.summary
        assert self.summary["champion_model"].endswith(".pkl")

    def test_champion_r2_above_threshold(self):
        assert self.summary["champion_r2"] > 0.80

    def test_training_samples_realistic(self):
        assert self.summary["training_samples"] > 5000

    def test_feature_count_reflects_engineering(self):
        assert self.summary["feature_count"] > 9

    def test_all_models_list_nonempty(self):
        assert len(self.summary["all_models_trained"]) >= 5


# ── Evaluation CSV ───────────────────────────────────────────────────────

class TestEvaluationCSV:

    @pytest.fixture(autouse=True)
    def _load(self, models_dir):
        self.df = pd.read_csv(models_dir / "california_housing_model_evaluation.csv")

    def test_has_expected_columns(self):
        required = {"model", "val_rmse", "val_r2", "val_mae"}
        assert required.issubset(set(self.df.columns))

    def test_no_nan_in_key_metrics(self):
        assert self.df["val_rmse"].notna().all()
        assert self.df["val_r2"].notna().all()

    def test_at_least_five_models_evaluated(self):
        assert len(self.df) >= 5

    def test_best_r2_exceeds_threshold(self):
        assert self.df["val_r2"].max() > 0.80


# ── Saved Model Files ───────────────────────────────────────────────────

class TestSavedModelFiles:
    """Load each .pkl and verify structure + basic predictions."""

    @pytest.fixture(autouse=True)
    def _load_all(self, models_dir):
        self.models = {}
        self.feature_names = None
        for mf in models_dir.glob("*california_housing*.pkl"):
            if "training_summary" in mf.name:
                continue
            data = joblib.load(mf)
            if isinstance(data, dict) and "model" in data:
                name = data.get("name", mf.stem)
                self.models[name] = data
                if data.get("feature_names") and self.feature_names is None:
                    self.feature_names = data["feature_names"]

    def test_all_models_have_correct_structure(self):
        for name, data in self.models.items():
            assert "model" in data, f"{name} missing 'model' key"
            assert "feature_names" in data, f"{name} missing 'feature_names'"
            assert "model_type" in data, f"{name} missing 'model_type'"
            assert data["model_type"] == "california_housing_predictor"

    def test_all_models_produce_predictions(self):
        assert self.feature_names is not None, "No feature names found"
        dummy_input = pd.DataFrame([{f: 0.0 for f in self.feature_names}])
        for name, data in self.models.items():
            pred = data["model"].predict(dummy_input)
            assert len(pred) == 1, f"{name} wrong prediction shape"
            assert np.isfinite(pred[0]), f"{name} returned non-finite prediction"

    def test_feature_names_consistent_across_models(self):
        """All models should have been trained on the same feature set."""
        names_list = [
            tuple(data["feature_names"])
            for data in self.models.values()
            if data.get("feature_names")
        ]
        unique = set(names_list)
        assert len(unique) == 1, f"Found {len(unique)} different feature name sets across models"

    def test_pkl_files_not_empty(self, models_dir):
        for mf in models_dir.glob("*california_housing*.pkl"):
            if "training_summary" in mf.name:
                continue
            assert mf.stat().st_size > 100, f"{mf.name} suspiciously small"
