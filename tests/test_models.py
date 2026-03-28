"""Tests for model training, evaluation, and serialization."""
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_california_housing_data
from src.data_pipeline import CaliforniaHousingPipeline
from src.model_training import CaliforniaHousingModelTrainer
from src.models.base_model import BaseModel
from src.models.linear_models import RidgeRegressionModel
from src.models.ensemble_models import RandomForestModel, GradientBoostingModel


# ── Shared fixture: trained pipeline + trainer ───────────────────────────

@pytest.fixture(scope="module")
def trained_env():
    """Run the pipeline and train models once for the whole module."""
    train, test = load_california_housing_data()
    pipeline = CaliforniaHousingPipeline(train, test, target_col="median_house_value")
    pipeline.run_pipeline(save_processed=False)

    trainer = CaliforniaHousingModelTrainer(
        pipeline.X_train, pipeline.y_train,
        pipeline.X_val, pipeline.y_val,
    )
    trainer.register_california_housing_models()
    trainer.train_all_models(tune_hyperparameters=False, cv_folds=3)

    return trainer, pipeline


# ── Model Training ───────────────────────────────────────────────────────

class TestModelTraining:

    def test_at_least_five_models_trained(self, trained_env):
        trainer, _ = trained_env
        assert len(trainer.trained_models) >= 5

    def test_all_models_are_fitted(self, trained_env):
        trainer, _ = trained_env
        for name, model in trainer.trained_models.items():
            assert model.is_fitted, f"{name} is not marked as fitted"

    def test_models_have_validation_metrics(self, trained_env):
        trainer, _ = trained_env
        for name, model in trainer.trained_models.items():
            assert model.validation_metrics is not None, f"{name} missing validation metrics"
            assert "r2" in model.validation_metrics, f"{name} missing r2 metric"
            assert "rmse" in model.validation_metrics, f"{name} missing rmse metric"

    def test_models_have_cv_scores(self, trained_env):
        trainer, _ = trained_env
        for name, model in trainer.trained_models.items():
            assert model.cv_scores is not None, f"{name} missing cv_scores"


# ── Model Evaluation ─────────────────────────────────────────────────────

class TestModelEvaluation:

    def test_evaluation_returns_dataframe(self, trained_env):
        trainer, _ = trained_env
        df = trainer.evaluate_models()
        assert isinstance(df, pd.DataFrame)

    def test_evaluation_has_all_models(self, trained_env):
        trainer, _ = trained_env
        df = trainer.evaluate_models()
        assert len(df) == len(trainer.trained_models)

    def test_evaluation_has_required_columns(self, trained_env):
        trainer, _ = trained_env
        df = trainer.evaluate_models()
        required = {"model", "val_rmse", "val_r2", "val_mae", "train_r2"}
        assert required.issubset(set(df.columns))

    def test_best_model_r2_above_threshold(self, trained_env):
        trainer, _ = trained_env
        best_name, best_model = trainer.get_best_model()
        r2 = best_model.validation_metrics["r2"]
        assert r2 > 0.80, f"Champion R² ({r2:.4f}) below 0.80 threshold"

    def test_r2_scores_are_valid(self, trained_env):
        trainer, _ = trained_env
        df = trainer.evaluate_models()
        # R² should be between -1 and 1 for reasonable models
        assert (df["val_r2"] > -1.0).all()
        assert (df["val_r2"] <= 1.0).all()

    def test_rmse_is_positive(self, trained_env):
        trainer, _ = trained_env
        df = trainer.evaluate_models()
        assert (df["val_rmse"] > 0).all()


# ── Model Predictions ────────────────────────────────────────────────────

class TestModelPredictions:

    def test_predictions_are_finite(self, trained_env):
        trainer, pipeline = trained_env
        X_sample = pipeline.X_val.head(10)
        for name, model in trainer.trained_models.items():
            preds = model.predict(X_sample)
            assert np.all(np.isfinite(preds)), f"{name} produced non-finite predictions"

    def test_predictions_shape_matches_input(self, trained_env):
        trainer, pipeline = trained_env
        X_sample = pipeline.X_val.head(20)
        for name, model in trainer.trained_models.items():
            preds = model.predict(X_sample)
            assert len(preds) == 20, f"{name} prediction count mismatch"

    def test_ensemble_predictions_in_realistic_range(self, trained_env):
        """Ensemble models should predict within the range of training targets."""
        trainer, pipeline = trained_env
        y_min = pipeline.y_train.min()
        y_max = pipeline.y_train.max()
        margin = (y_max - y_min) * 0.5  # 50% margin outside range

        ensemble_names = ["random_forest", "gradient_boosting", "extra_trees"]
        X_sample = pipeline.X_val.head(50)
        for name in ensemble_names:
            if name not in trainer.trained_models:
                continue
            preds = trainer.trained_models[name].predict(X_sample)
            assert np.all(preds > y_min - margin), f"{name} predicts unrealistically low"
            assert np.all(preds < y_max + margin), f"{name} predicts unrealistically high"


# ── Model Serialization ─────────────────────────────────────────────────

class TestModelSerialization:

    def test_save_and_load_roundtrip(self, trained_env):
        """Save a model, load it back, verify predictions are identical."""
        trainer, pipeline = trained_env
        best_name, best_model = trainer.get_best_model()
        X_sample = pipeline.X_val.head(5)
        original_preds = best_model.predict(X_sample)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            best_model.save_model(path)

            assert path.exists()
            assert path.stat().st_size > 0

            # Load and predict
            loaded_data = joblib.load(path)
            assert isinstance(loaded_data, dict)
            assert "model" in loaded_data
            assert "feature_names" in loaded_data
            assert loaded_data["model_type"] == "california_housing_predictor"

            loaded_preds = loaded_data["model"].predict(X_sample)
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_saved_model_contains_metadata(self, trained_env):
        trainer, _ = trained_env
        best_name, best_model = trainer.get_best_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            best_model.save_model(path)
            data = joblib.load(path)

            assert "name" in data
            assert "training_metrics" in data
            assert "validation_metrics" in data
            assert "cv_scores" in data
            assert len(data["feature_names"]) > 0
