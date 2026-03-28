"""Shared fixtures for all test modules."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS_DIR


# ── Paths ────────────────────────────────────────────────────────────────
@pytest.fixture
def models_dir():
    return MODELS_DIR


@pytest.fixture
def processed_data_dir():
    return PROJECT_ROOT / "data" / "processed"


# ── Sample data ──────────────────────────────────────────────────────────
@pytest.fixture
def sample_california_row():
    """A single realistic SF Bay Area housing row."""
    return {
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


@pytest.fixture
def sample_inland_row():
    """A single realistic Central Valley housing row."""
    return {
        "longitude": -119.8,
        "latitude": 36.7,
        "housing_median_age": 15.0,
        "total_rooms": 2000.0,
        "total_bedrooms": 400.0,
        "population": 1200.0,
        "households": 400.0,
        "median_income": 3.5,
        "ocean_proximity": "INLAND",
    }
