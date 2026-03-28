"""Tests for data loading, pipeline, and feature engineering."""
import pandas as pd
import numpy as np
import pytest

from src.data_loader import load_california_housing_data
from src.data_pipeline import CaliforniaHousingPipeline


# ── Data Loading ─────────────────────────────────────────────────────────

class TestDataLoading:
    """Verify the California Housing dataset loads correctly."""

    @pytest.fixture(autouse=True, scope="class")
    def _load_data(self, request):
        train, test = load_california_housing_data()
        request.cls.train = train
        request.cls.test = test

    def test_train_has_enough_rows(self):
        assert len(self.train) > 10_000, "Train set too small"

    def test_test_has_enough_rows(self):
        assert len(self.test) > 2_000, "Test set too small"

    def test_target_column_exists(self):
        assert "median_house_value" in self.train.columns

    def test_required_features_present(self):
        expected = {
            "longitude", "latitude", "housing_median_age",
            "total_rooms", "total_bedrooms", "population",
            "households", "median_income", "ocean_proximity",
        }
        assert expected.issubset(set(self.train.columns))

    def test_no_nulls_in_core_features(self):
        core = ["longitude", "latitude", "median_income", "median_house_value"]
        assert self.train[core].isnull().sum().sum() == 0

    def test_target_values_are_positive(self):
        assert (self.train["median_house_value"] > 0).all()

    def test_longitude_latitude_ranges(self):
        assert self.train["longitude"].between(-125, -113).all()
        assert self.train["latitude"].between(32, 43).all()


# ── Data Pipeline ────────────────────────────────────────────────────────

class TestDataPipeline:
    """Verify the preprocessing pipeline produces valid output."""

    @pytest.fixture(autouse=True, scope="class")
    def _run_pipeline(self, request):
        train, test = load_california_housing_data()
        pipeline = CaliforniaHousingPipeline(train, test, target_col="median_house_value")
        request.cls.results = pipeline.run_pipeline(save_processed=False)
        request.cls.pipeline = pipeline

    def test_pipeline_returns_results_dict(self):
        assert isinstance(self.results, dict)
        assert "feature_count" in self.results

    def test_features_were_engineered(self):
        assert self.results["feature_count"] > 9, "Expected engineered features beyond the original 9"

    def test_train_val_split_exists(self):
        assert self.pipeline.X_train is not None
        assert self.pipeline.X_val is not None
        assert self.pipeline.y_train is not None
        assert self.pipeline.y_val is not None

    def test_train_larger_than_val(self):
        assert len(self.pipeline.X_train) > len(self.pipeline.X_val)

    def test_no_target_leakage_in_features(self):
        assert "median_house_value" not in self.pipeline.X_train.columns

    def test_no_nans_in_features(self):
        nan_count = self.pipeline.X_train.isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in training features"

    def test_no_infinite_values_in_features(self):
        numeric = self.pipeline.X_train.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric).sum().sum()
        assert inf_count == 0, f"Found {inf_count} infinite values in training features"

    def test_target_is_numeric(self):
        assert pd.api.types.is_numeric_dtype(self.pipeline.y_train)

    def test_feature_columns_match_between_train_and_val(self):
        assert list(self.pipeline.X_train.columns) == list(self.pipeline.X_val.columns)
