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
# Based on the real sklearn California Housing dataset
FEATURE_CONFIG = {
    'numerical_features': {
        'median_income': {
            'label': 'Median Income (in $10,000s)',
            'min_value': 0.5,
            'max_value': 15.0,
            'value': 3.9,
            'step': 0.1,
            'help': 'Median household income for the block group in tens of thousands of dollars'
        },
        'housing_median_age': {
            'label': 'Housing Median Age (years)',
            'min_value': 1,
            'max_value': 52,
            'value': 29,
            'step': 1,
            'help': 'Median age of houses in the block group'
        },
        'ave_rooms': {
            'label': 'Avg Rooms per Household',
            'min_value': 1.0,
            'max_value': 15.0,
            'value': 5.4,
            'step': 0.1,
            'help': 'Average number of rooms per household in the block group'
        },
        'ave_bedrooms': {
            'label': 'Avg Bedrooms per Household',
            'min_value': 0.3,
            'max_value': 5.0,
            'value': 1.1,
            'step': 0.1,
            'help': 'Average number of bedrooms per household in the block group'
        },
        'population': {
            'label': 'Block Group Population',
            'min_value': 3,
            'max_value': 10000,
            'value': 1425,
            'step': 50,
            'help': 'Total population in the census block group'
        },
        'ave_occupancy': {
            'label': 'Avg Household Size',
            'min_value': 0.5,
            'max_value': 10.0,
            'value': 3.1,
            'step': 0.1,
            'help': 'Average number of people per household'
        },
        'longitude': {
            'label': 'Longitude',
            'min_value': -124.5,
            'max_value': -114.0,
            'value': -119.6,
            'step': 0.1,
            'help': 'Longitude coordinate (West Coast California)'
        },
        'latitude': {
            'label': 'Latitude',
            'min_value': 32.5,
            'max_value': 42.0,
            'value': 35.6,
            'step': 0.1,
            'help': 'Latitude coordinate (California range)'
        }
    },
    'categorical_features': {}
}

# Quick preset configurations for different California areas
# Using real feature names matching the sklearn California Housing dataset
CALIFORNIA_PRESETS = {
    "San Francisco Bay Area": {
        'median_income': 8.5,
        'housing_median_age': 35,
        'ave_rooms': 6.2,
        'ave_bedrooms': 1.05,
        'population': 1500,
        'ave_occupancy': 2.8,
        'longitude': -122.4,
        'latitude': 37.8,
    },
    "Los Angeles Metro": {
        'median_income': 5.0,
        'housing_median_age': 30,
        'ave_rooms': 5.0,
        'ave_bedrooms': 1.05,
        'population': 2000,
        'ave_occupancy': 3.2,
        'longitude': -118.2,
        'latitude': 34.1,
    },
    "San Diego Area": {
        'median_income': 5.5,
        'housing_median_age': 25,
        'ave_rooms': 5.5,
        'ave_bedrooms': 1.08,
        'population': 1800,
        'ave_occupancy': 3.0,
        'longitude': -117.2,
        'latitude': 32.7,
    },
    "Sacramento Valley": {
        'median_income': 4.2,
        'housing_median_age': 20,
        'ave_rooms': 5.8,
        'ave_bedrooms': 1.1,
        'population': 1200,
        'ave_occupancy': 2.9,
        'longitude': -121.5,
        'latitude': 38.6,
    },
    "Central Valley": {
        'median_income': 2.8,
        'housing_median_age': 18,
        'ave_rooms': 4.8,
        'ave_bedrooms': 1.1,
        'population': 1100,
        'ave_occupancy': 3.3,
        'longitude': -120.0,
        'latitude': 36.0,
    }
}

# UI Theme for California vibe
UI_THEME = {
    'primary_color': '#E85D26',  # California sunset orange (refined)
    'primary_gradient': 'linear-gradient(135deg, #E85D26 0%, #F09819 50%, #EDDE5D 100%)',
    'accent_color': '#1B6B93',  # Pacific blue
    'success_color': '#2E7D32',
    'background_color': '#FAFBFD',
    'card_background': 'rgba(255, 255, 255, 0.85)',
    'secondary_background_color': '#F0F4F8',
    'text_color': '#1A1A2E',
    'text_secondary': '#64748B',
    'font': 'sans serif',
    'border_radius': '16px',
    'shadow': '0 4px 24px rgba(0, 0, 0, 0.06)',
    'shadow_hover': '0 8px 32px rgba(0, 0, 0, 0.12)',
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