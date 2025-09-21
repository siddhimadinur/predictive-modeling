"""
Configuration for the California Housing Streamlit application.
"""
import streamlit as st
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# App configuration
APP_CONFIG = {
    'page_title': 'California Housing Price Predictor',
    'page_icon': '🏠',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Model configuration
MODEL_CONFIG = {
    'model_dir': PROJECT_ROOT / 'models' / 'trained_models',
    'champion_model': 'champion_california_housing_model.pkl',
    'feature_importance_top_n': 20
}

# California Housing Feature configuration for input forms
FEATURE_CONFIG = {
    'numerical_features': {
        'median_income': {
            'label': 'Median Income (in $10,000s)',
            'min_value': 0.5,
            'max_value': 15.0,
            'value': 5.0,
            'step': 0.1,
            'help': 'Median household income for the area in tens of thousands of dollars'
        },
        'total_rooms': {
            'label': 'Total Rooms in Area',
            'min_value': 500,
            'max_value': 8000,
            'value': 2500,
            'step': 50,
            'help': 'Total number of rooms in the census block group'
        },
        'total_bedrooms': {
            'label': 'Total Bedrooms in Area',
            'min_value': 100,
            'max_value': 1500,
            'value': 500,
            'step': 10,
            'help': 'Total number of bedrooms in the census block group'
        },
        'housing_median_age': {
            'label': 'Housing Median Age (years)',
            'min_value': 1,
            'max_value': 52,
            'value': 25,
            'step': 1,
            'help': 'Median age of houses in the area'
        },
        'population': {
            'label': 'Population',
            'min_value': 300,
            'max_value': 5000,
            'value': 2000,
            'step': 50,
            'help': 'Total population in the census block group'
        },
        'households': {
            'label': 'Number of Households',
            'min_value': 100,
            'max_value': 1800,
            'value': 800,
            'step': 10,
            'help': 'Total number of households in the area'
        },
        'longitude': {
            'label': 'Longitude',
            'min_value': -124.5,
            'max_value': -114.0,
            'value': -118.0,
            'step': 0.1,
            'help': 'Longitude coordinate (West Coast California)'
        },
        'latitude': {
            'label': 'Latitude',
            'min_value': 32.5,
            'max_value': 42.0,
            'value': 37.0,
            'step': 0.1,
            'help': 'Latitude coordinate (California range)'
        }
    },
    'categorical_features': {
        'ocean_proximity': {
            'label': 'Ocean Proximity',
            'options': ['NEAR BAY', 'NEAR OCEAN', 'INLAND', '<1H OCEAN', 'ISLAND'],
            'value': 'INLAND',
            'help': 'Proximity to the Pacific Ocean'
        }
    }
}

# Quick preset configurations for different California areas
CALIFORNIA_PRESETS = {
    "San Francisco Bay Area": {
        'median_income': 12.0,
        'total_rooms': 4000,
        'total_bedrooms': 800,
        'housing_median_age': 35,
        'population': 3500,
        'households': 1200,
        'longitude': -122.4,
        'latitude': 37.8,
        'ocean_proximity': 'NEAR BAY'
    },
    "Los Angeles Metro": {
        'median_income': 8.5,
        'total_rooms': 3500,
        'total_bedrooms': 700,
        'housing_median_age': 28,
        'population': 4000,
        'households': 1100,
        'longitude': -118.2,
        'latitude': 34.1,
        'ocean_proximity': 'NEAR OCEAN'
    },
    "San Diego Area": {
        'median_income': 9.0,
        'total_rooms': 2800,
        'total_bedrooms': 600,
        'housing_median_age': 22,
        'population': 2800,
        'households': 950,
        'longitude': -117.2,
        'latitude': 32.7,
        'ocean_proximity': 'NEAR OCEAN'
    },
    "Sacramento Valley": {
        'median_income': 6.5,
        'total_rooms': 2200,
        'total_bedrooms': 450,
        'housing_median_age': 18,
        'population': 2200,
        'households': 700,
        'longitude': -121.5,
        'latitude': 38.6,
        'ocean_proximity': 'INLAND'
    },
    "Central Valley": {
        'median_income': 4.5,
        'total_rooms': 1800,
        'total_bedrooms': 350,
        'housing_median_age': 15,
        'population': 1800,
        'households': 550,
        'longitude': -120.0,
        'latitude': 36.0,
        'ocean_proximity': 'INLAND'
    }
}

# UI Theme for California vibe
UI_THEME = {
    'primary_color': '#FF6B35',  # California sunset orange
    'background_color': '#ffffff',
    'secondary_background_color': '#f0f8ff',
    'text_color': '#262730',
    'font': 'sans serif'
}

# Display options
DISPLAY_OPTIONS = {
    'prediction_precision': 0,
    'metric_precision': 4,
    'confidence_level': 0.95,
    'chart_height': 400,
    'max_features_plot': 20
}

# California region information for context
CALIFORNIA_REGIONS = {
    'Northern California': {
        'lat_range': (37.0, 42.0),
        'description': 'San Francisco Bay Area, Sacramento, and northern regions',
        'typical_prices': '$400K - $800K+'
    },
    'Central California': {
        'lat_range': (35.0, 37.0),
        'description': 'Central Valley, Monterey, and central coast',
        'typical_prices': '$300K - $600K'
    },
    'Southern California': {
        'lat_range': (32.5, 35.0),
        'description': 'Los Angeles, Orange County, San Diego areas',
        'typical_prices': '$500K - $1M+'
    }
}