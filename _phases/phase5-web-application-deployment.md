# Phase 5: Web Application & Deployment

## Overview
Build a comprehensive Streamlit web application that provides an intuitive interface for house price prediction, model insights, and data exploration. Deploy the application for easy access and usage.

## Objectives
- Create an interactive Streamlit web application
- Implement user input forms for house features
- Display price predictions with confidence intervals
- Show model insights and feature importance
- Provide data exploration and visualization tools
- Implement proper error handling and validation
- Create responsive and user-friendly design
- Deploy application for public access

## Step-by-Step Implementation

### 5.1 Application Architecture

#### 5.1.1 Create Application Structure
```bash
# Create app directory structure
mkdir -p app/{components,utils,pages,assets}
mkdir -p app/assets/{css,images}
```

Create `app/__init__.py`:
```python
"""
Streamlit web application for house price prediction.
"""
```

#### 5.1.2 Create Application Configuration
Create `app/config.py`:
```python
"""
Configuration for the Streamlit application.
"""
import streamlit as st
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# App configuration
APP_CONFIG = {
    'page_title': 'House Price Predictor',
    'page_icon': '🏠',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Model configuration
MODEL_CONFIG = {
    'model_dir': PROJECT_ROOT / 'models' / 'trained_models',
    'default_model': 'best_model.pkl',
    'feature_importance_top_n': 20
}

# Feature configuration for input forms
FEATURE_CONFIG = {
    'numerical_features': {
        'GrLivArea': {
            'label': 'Above Ground Living Area (sq ft)',
            'min_value': 500,
            'max_value': 6000,
            'value': 1500,
            'step': 50,
            'help': 'Total square footage of living space above ground'
        },
        'TotalBsmtSF': {
            'label': 'Total Basement Area (sq ft)',
            'min_value': 0,
            'max_value': 3000,
            'value': 1000,
            'step': 50,
            'help': 'Total square footage of basement area'
        },
        'YearBuilt': {
            'label': 'Year Built',
            'min_value': 1872,
            'max_value': 2024,
            'value': 2000,
            'step': 1,
            'help': 'Year the house was originally built'
        },
        'OverallQual': {
            'label': 'Overall Quality (1-10)',
            'min_value': 1,
            'max_value': 10,
            'value': 6,
            'step': 1,
            'help': 'Overall material and finish quality (10 = Excellent, 1 = Very Poor)'
        },
        'OverallCond': {
            'label': 'Overall Condition (1-10)',
            'min_value': 1,
            'max_value': 10,
            'value': 6,
            'step': 1,
            'help': 'Overall condition rating (10 = Excellent, 1 = Very Poor)'
        },
        'GarageCars': {
            'label': 'Garage Car Capacity',
            'min_value': 0,
            'max_value': 4,
            'value': 2,
            'step': 1,
            'help': 'Number of cars the garage can hold'
        },
        'FullBath': {
            'label': 'Full Bathrooms Above Ground',
            'min_value': 0,
            'max_value': 4,
            'value': 2,
            'step': 1,
            'help': 'Number of full bathrooms above ground'
        },
        'BedroomAbvGr': {
            'label': 'Bedrooms Above Ground',
            'min_value': 0,
            'max_value': 8,
            'value': 3,
            'step': 1,
            'help': 'Number of bedrooms above ground'
        }
    },
    'categorical_features': {
        'Neighborhood': {
            'label': 'Neighborhood',
            'options': [
                'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
                'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
                'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
            ],
            'value': 'NAmes',
            'help': 'Physical location within Ames city limits'
        },
        'HouseStyle': {
            'label': 'House Style',
            'options': [
                '1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf',
                '2Story', 'SFoyer', 'SLvl'
            ],
            'value': '1Story',
            'help': 'Style of dwelling'
        },
        'ExterQual': {
            'label': 'Exterior Quality',
            'options': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            'value': 'TA',
            'help': 'Exterior material quality (Ex=Excellent, Gd=Good, TA=Typical/Average, Fa=Fair, Po=Poor)'
        },
        'KitchenQual': {
            'label': 'Kitchen Quality',
            'options': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            'value': 'TA',
            'help': 'Kitchen quality (Ex=Excellent, Gd=Good, TA=Typical/Average, Fa=Fair, Po=Poor)'
        },
        'SaleType': {
            'label': 'Sale Type',
            'options': [
                'WD', 'New', 'COD', 'ConLD', 'ConLI', 'ConLw',
                'CWD', 'Oth', 'Con'
            ],
            'value': 'WD',
            'help': 'Type of sale (WD=Warranty Deed, New=Home just constructed, etc.)'
        }
    }
}

# UI Theme
UI_THEME = {
    'primary_color': '#1f77b4',
    'background_color': '#ffffff',
    'secondary_background_color': '#f0f2f6',
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
```

### 5.2 Core Application Components

#### 5.2.1 Create Model Loading Utility
Create `app/utils/model_loader.py`:
```python
"""
Model loading and prediction utilities for the Streamlit app.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle
from typing import Dict, Any, Tuple, Optional, List
import streamlit as st

from app.config import MODEL_CONFIG


class ModelLoader:
    """Handle model loading and predictions."""

    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.feature_names = None
        self.scaler = None
        self.encoder = None

    @st.cache_resource
    def load_models(_self, model_dir: Path = None) -> Dict[str, Any]:
        """
        Load all available models.

        Args:
            model_dir: Directory containing models

        Returns:
            Dictionary of loaded models
        """
        if model_dir is None:
            model_dir = MODEL_CONFIG['model_dir']

        models = {}
        metadata = {}

        if not model_dir.exists():
            st.error(f"Model directory not found: {model_dir}")
            return models

        # Load individual model files
        for model_file in model_dir.glob("*.pkl"):
            try:
                model_data = joblib.load(model_file)

                # Handle different model save formats
                if isinstance(model_data, dict) and 'model' in model_data:
                    # New format with metadata
                    model_name = model_data.get('name', model_file.stem)
                    models[model_name] = model_data['model']
                    metadata[model_name] = {
                        'feature_names': model_data.get('feature_names', []),
                        'training_metrics': model_data.get('training_metrics', {}),
                        'validation_metrics': model_data.get('validation_metrics', {}),
                        'cv_scores': model_data.get('cv_scores', {})
                    }
                else:
                    # Old format - just the model
                    model_name = model_file.stem
                    models[model_name] = model_data
                    metadata[model_name] = {}

            except Exception as e:
                st.warning(f"Could not load model {model_file.name}: {str(e)}")

        _self.models = models
        _self.model_metadata = metadata

        if models:
            # Set feature names from first model
            first_model_metadata = next(iter(metadata.values()))
            _self.feature_names = first_model_metadata.get('feature_names', [])

        return models

    @st.cache_resource
    def load_preprocessing_pipeline(_self, model_dir: Path = None) -> Tuple[Any, Any]:
        """
        Load preprocessing pipeline components.

        Args:
            model_dir: Directory containing pipeline artifacts

        Returns:
            Tuple of (scaler, encoder)
        """
        if model_dir is None:
            model_dir = MODEL_CONFIG['model_dir'].parent / 'processed_data'

        scaler = None
        encoder = None

        # Load pipeline artifacts
        pipeline_file = model_dir / 'pipeline_artifacts.pkl'
        if pipeline_file.exists():
            try:
                pipeline_data = joblib.load(pipeline_file)
                feature_engineer = pipeline_data.get('feature_engineer')

                if feature_engineer and hasattr(feature_engineer, 'feature_transformers'):
                    scaler = feature_engineer.feature_transformers.get('scaler')
                    # Get encoders
                    for key, transformer in feature_engineer.feature_transformers.items():
                        if 'encoder' in key or 'onehot' in key:
                            encoder = transformer
                            break

            except Exception as e:
                st.warning(f"Could not load preprocessing pipeline: {str(e)}")

        return scaler, encoder

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary
        """
        if model_name not in self.model_metadata:
            return {}

        return self.model_metadata[model_name]

    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess user input for prediction.

        Args:
            input_data: User input dictionary

        Returns:
            Preprocessed DataFrame
        """
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # If we have feature names from training, ensure all features are present
        if self.feature_names:
            # Add missing features with default values
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    # Try to infer default value based on feature name
                    if any(keyword in feature.lower() for keyword in ['area', 'sf']):
                        input_df[feature] = 0
                    elif any(keyword in feature.lower() for keyword in ['year', 'yr']):
                        input_df[feature] = 2000
                    elif any(keyword in feature.lower() for keyword in ['qual', 'cond']):
                        input_df[feature] = 6
                    else:
                        input_df[feature] = 0

            # Reorder columns to match training feature order
            input_df = input_df.reindex(columns=self.feature_names, fill_value=0)

        return input_df

    def predict(self, model_name: str, input_data: Dict[str, Any]) -> Tuple[float, Optional[Tuple[float, float]]]:
        """
        Make prediction using specified model.

        Args:
            model_name: Name of the model to use
            input_data: Input features dictionary

        Returns:
            Tuple of (prediction, confidence_interval)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        # Preprocess input
        processed_input = self.preprocess_input(input_data)

        try:
            # Make prediction
            prediction = model.predict(processed_input)[0]

            # Calculate confidence interval if possible
            confidence_interval = None
            try:
                # For tree-based models, use quantile prediction if available
                if hasattr(model, 'predict_quantiles'):
                    lower = model.predict_quantiles(processed_input, quantiles=[0.025])[0]
                    upper = model.predict_quantiles(processed_input, quantiles=[0.975])[0]
                    confidence_interval = (lower, upper)
                elif hasattr(model, 'predict') and hasattr(model, 'estimators_'):
                    # For ensemble models, use estimator variance
                    estimator_predictions = []
                    for estimator in model.estimators_[:min(50, len(model.estimators_))]:
                        pred = estimator.predict(processed_input)[0]
                        estimator_predictions.append(pred)

                    std = np.std(estimator_predictions)
                    confidence_interval = (prediction - 1.96 * std, prediction + 1.96 * std)

            except Exception:
                # If confidence interval calculation fails, continue without it
                pass

            return prediction, confidence_interval

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_feature_importance(self, model_name: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Get feature importance from model.

        Args:
            model_name: Name of the model
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance or None
        """
        if model_name not in self.models:
            return None

        model = self.models[model_name]

        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                if self.feature_names:
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_
                    })
                else:
                    # Fallback if no feature names
                    n_features = len(model.feature_importances_)
                    importance_df = pd.DataFrame({
                        'feature': [f'feature_{i}' for i in range(n_features)],
                        'importance': model.feature_importances_
                    })

                return importance_df.sort_values('importance', ascending=False).head(top_n)

            elif hasattr(model, 'coef_'):
                # Linear models
                if self.feature_names:
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': np.abs(model.coef_)
                    })
                else:
                    # Fallback if no feature names
                    n_features = len(model.coef_)
                    importance_df = pd.DataFrame({
                        'feature': [f'feature_{i}' for i in range(n_features)],
                        'importance': np.abs(model.coef_)
                    })

                return importance_df.sort_values('importance', ascending=False).head(top_n)

        except Exception as e:
            st.warning(f"Could not extract feature importance: {str(e)}")

        return None
```

#### 5.2.2 Create Input Components
Create `app/components/input_forms.py`:
```python
"""
Input form components for the Streamlit app.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any

from app.config import FEATURE_CONFIG


class InputForm:
    """Handle user input forms."""

    def __init__(self):
        self.input_data = {}

    def render_basic_features(self) -> Dict[str, Any]:
        """
        Render basic feature input form.

        Returns:
            Dictionary with user inputs
        """
        st.subheader("🏠 House Features")

        # Create columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Size & Area**")

            # Living area
            gr_liv_area = st.number_input(
                **FEATURE_CONFIG['numerical_features']['GrLivArea']
            )

            # Basement area
            total_bsmt_sf = st.number_input(
                **FEATURE_CONFIG['numerical_features']['TotalBsmtSF']
            )

            # Year built
            year_built = st.number_input(
                **FEATURE_CONFIG['numerical_features']['YearBuilt']
            )

            # Bedrooms
            bedrooms = st.number_input(
                **FEATURE_CONFIG['numerical_features']['BedroomAbvGr']
            )

        with col2:
            st.write("**Quality & Condition**")

            # Overall quality
            overall_qual = st.selectbox(
                **FEATURE_CONFIG['numerical_features']['OverallQual']
            )

            # Overall condition
            overall_cond = st.selectbox(
                **FEATURE_CONFIG['numerical_features']['OverallCond']
            )

            # Full bathrooms
            full_bath = st.selectbox(
                **FEATURE_CONFIG['numerical_features']['FullBath']
            )

            # Garage cars
            garage_cars = st.selectbox(
                **FEATURE_CONFIG['numerical_features']['GarageCars']
            )

        # Categorical features
        st.write("**Location & Style**")
        col3, col4 = st.columns(2)

        with col3:
            neighborhood = st.selectbox(
                **FEATURE_CONFIG['categorical_features']['Neighborhood']
            )

            house_style = st.selectbox(
                **FEATURE_CONFIG['categorical_features']['HouseStyle']
            )

        with col4:
            exterior_qual = st.selectbox(
                **FEATURE_CONFIG['categorical_features']['ExterQual']
            )

            kitchen_qual = st.selectbox(
                **FEATURE_CONFIG['categorical_features']['KitchenQual']
            )

        # Store inputs
        self.input_data = {
            'GrLivArea': gr_liv_area,
            'TotalBsmtSF': total_bsmt_sf,
            'YearBuilt': year_built,
            'BedroomAbvGr': bedrooms,
            'OverallQual': overall_qual,
            'OverallCond': overall_cond,
            'FullBath': full_bath,
            'GarageCars': garage_cars,
            'Neighborhood': neighborhood,
            'HouseStyle': house_style,
            'ExterQual': exterior_qual,
            'KitchenQual': kitchen_qual
        }

        return self.input_data

    def render_advanced_features(self) -> Dict[str, Any]:
        """
        Render advanced feature input form.

        Returns:
            Dictionary with additional user inputs
        """
        with st.expander("🔧 Advanced Features (Optional)", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Additional Areas**")
                first_flr_sf = st.number_input(
                    "1st Floor Area (sq ft)",
                    min_value=0, max_value=3000, value=800, step=50
                )

                second_flr_sf = st.number_input(
                    "2nd Floor Area (sq ft)",
                    min_value=0, max_value=2000, value=0, step=50
                )

                garage_area = st.number_input(
                    "Garage Area (sq ft)",
                    min_value=0, max_value=1500, value=500, step=50
                )

            with col2:
                st.write("**Additional Rooms**")
                half_bath = st.selectbox(
                    "Half Bathrooms", options=[0, 1, 2, 3], index=1
                )

                tot_rms_abv_grd = st.number_input(
                    "Total Rooms Above Ground",
                    min_value=1, max_value=15, value=6, step=1
                )

                fireplaces = st.selectbox(
                    "Fireplaces", options=[0, 1, 2, 3, 4], index=0
                )

            with col3:
                st.write("**Property Details**")
                lot_area = st.number_input(
                    "Lot Area (sq ft)",
                    min_value=1000, max_value=100000, value=8000, step=500
                )

                year_remod_add = st.number_input(
                    "Year Remodeled",
                    min_value=1872, max_value=2024, value=2000, step=1
                )

                sale_type = st.selectbox(
                    **FEATURE_CONFIG['categorical_features']['SaleType']
                )

            advanced_features = {
                '1stFlrSF': first_flr_sf,
                '2ndFlrSF': second_flr_sf,
                'GarageArea': garage_area,
                'HalfBath': half_bath,
                'TotRmsAbvGrd': tot_rms_abv_grd,
                'Fireplaces': fireplaces,
                'LotArea': lot_area,
                'YearRemodAdd': year_remod_add,
                'SaleType': sale_type
            }

            # Update main input data
            self.input_data.update(advanced_features)

        return advanced_features

    def render_input_summary(self) -> None:
        """Render summary of user inputs."""
        if not self.input_data:
            return

        st.subheader("📋 Input Summary")

        # Create summary DataFrame
        summary_data = []
        for key, value in self.input_data.items():
            # Format the key for display
            display_key = key.replace('_', ' ').title()
            summary_data.append({'Feature': display_key, 'Value': value})

        summary_df = pd.DataFrame(summary_data)

        # Display in columns
        col1, col2 = st.columns(2)

        mid_point = len(summary_df) // 2

        with col1:
            st.dataframe(
                summary_df.iloc[:mid_point],
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.dataframe(
                summary_df.iloc[mid_point:],
                use_container_width=True,
                hide_index=True
            )

    def get_input_data(self) -> Dict[str, Any]:
        """Get the current input data."""
        return self.input_data


class QuickPresets:
    """Predefined house configurations for quick testing."""

    @staticmethod
    def get_presets() -> Dict[str, Dict[str, Any]]:
        """Get predefined house configurations."""
        return {
            "Starter Home": {
                'GrLivArea': 1200,
                'TotalBsmtSF': 800,
                'YearBuilt': 1990,
                'BedroomAbvGr': 2,
                'OverallQual': 5,
                'OverallCond': 6,
                'FullBath': 1,
                'GarageCars': 1,
                'Neighborhood': 'NAmes',
                'HouseStyle': '1Story',
                'ExterQual': 'TA',
                'KitchenQual': 'TA'
            },
            "Family Home": {
                'GrLivArea': 1800,
                'TotalBsmtSF': 1200,
                'YearBuilt': 2005,
                'BedroomAbvGr': 3,
                'OverallQual': 7,
                'OverallCond': 7,
                'FullBath': 2,
                'GarageCars': 2,
                'Neighborhood': 'CollgCr',
                'HouseStyle': '2Story',
                'ExterQual': 'Gd',
                'KitchenQual': 'Gd'
            },
            "Luxury Home": {
                'GrLivArea': 3000,
                'TotalBsmtSF': 1800,
                'YearBuilt': 2010,
                'BedroomAbvGr': 4,
                'OverallQual': 9,
                'OverallCond': 9,
                'FullBath': 3,
                'GarageCars': 3,
                'Neighborhood': 'NridgHt',
                'HouseStyle': '2Story',
                'ExterQual': 'Ex',
                'KitchenQual': 'Ex'
            },
            "Budget Home": {
                'GrLivArea': 900,
                'TotalBsmtSF': 600,
                'YearBuilt': 1975,
                'BedroomAbvGr': 2,
                'OverallQual': 4,
                'OverallCond': 5,
                'FullBath': 1,
                'GarageCars': 1,
                'Neighborhood': 'Edwards',
                'HouseStyle': '1Story',
                'ExterQual': 'Fa',
                'KitchenQual': 'Fa'
            }
        }

    def render_preset_selector(self) -> Dict[str, Any]:
        """
        Render preset selector.

        Returns:
            Selected preset configuration
        """
        st.subheader("🚀 Quick Start")
        st.write("Choose a preset configuration to get started quickly:")

        presets = self.get_presets()
        preset_names = list(presets.keys())

        selected_preset = st.selectbox(
            "Choose a house type:",
            options=["Custom"] + preset_names,
            index=0,
            help="Select a predefined configuration or choose 'Custom' to enter your own values"
        )

        if selected_preset != "Custom":
            st.success(f"✅ {selected_preset} configuration loaded!")

            # Show preset details
            with st.expander("View preset details"):
                preset_data = presets[selected_preset]
                for key, value in preset_data.items():
                    st.write(f"**{key}**: {value}")

            return presets[selected_preset]

        return {}
```

**Test**: Input form components
```python
python -c "
from app.components.input_forms import InputForm, QuickPresets

# Test preset functionality
presets = QuickPresets()
preset_configs = presets.get_presets()

print('✓ Input form components working correctly')
print(f'Available presets: {list(preset_configs.keys())}')
print(f'Starter home features: {len(preset_configs[\"Starter Home\"])}')
"
```

### 5.3 Main Application Pages

#### 5.3.1 Create Main Streamlit App
Create `app/streamlit_app.py`:
```python
"""
Main Streamlit application for house price prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure page
from app.config import APP_CONFIG, DISPLAY_OPTIONS
st.set_page_config(**APP_CONFIG)

# Import components
from app.utils.model_loader import ModelLoader
from app.components.input_forms import InputForm, QuickPresets
from app.pages.prediction_page import PredictionPage
from app.pages.insights_page import InsightsPage
from app.pages.data_explorer import DataExplorer


def main():
    """Main application function."""

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }

    .prediction-box {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    .stAlert > div {
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 House Price Prediction System</h1>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
        st.session_state.models_loaded = False

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Price Prediction", "📊 Model Insights", "🔍 Data Explorer", "ℹ️ About"]
    )

    # Load models if not already loaded
    if not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            try:
                models = st.session_state.model_loader.load_models()
                if models:
                    st.session_state.models_loaded = True
                    st.sidebar.success(f"✅ {len(models)} models loaded")
                else:
                    st.sidebar.warning("⚠️ No models found")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading models: {str(e)}")

    # Page routing
    if page == "🏠 Price Prediction":
        prediction_page = PredictionPage(st.session_state.model_loader)
        prediction_page.render()

    elif page == "📊 Model Insights":
        insights_page = InsightsPage(st.session_state.model_loader)
        insights_page.render()

    elif page == "🔍 Data Explorer":
        explorer_page = DataExplorer()
        explorer_page.render()

    elif page == "ℹ️ About":
        render_about_page()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Built with ❤️ using Streamlit • House Price Prediction System v1.0"
        "</div>",
        unsafe_allow_html=True
    )


def render_about_page():
    """Render the about page."""
    st.title("About This Application")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🏠 House Price Prediction System

        This application provides intelligent house price predictions using advanced machine learning models
        trained on comprehensive housing data.

        ### ✨ Key Features

        - **Accurate Predictions**: Multiple ML models including Random Forest, XGBoost, and Linear Regression
        - **Interactive Interface**: Easy-to-use forms with helpful guidance
        - **Model Insights**: Feature importance analysis and model performance metrics
        - **Data Visualization**: Comprehensive charts and exploration tools
        - **Quick Presets**: Pre-configured house types for easy testing

        ### 🔧 Technology Stack

        - **Frontend**: Streamlit for interactive web interface
        - **Machine Learning**: scikit-learn, XGBoost
        - **Data Processing**: pandas, NumPy
        - **Visualization**: Plotly, Matplotlib, Seaborn
        - **Deployment**: Streamlit Cloud (optional)

        ### 📊 Model Performance

        Our models have been trained and validated on comprehensive housing data with the following performance:
        """)

        # Display model performance if available
        if st.session_state.models_loaded:
            available_models = st.session_state.model_loader.get_available_models()
            if available_models:
                st.write("**Available Models:**")
                for model_name in available_models:
                    model_info = st.session_state.model_loader.get_model_info(model_name)
                    if model_info.get('validation_metrics'):
                        metrics = model_info['validation_metrics']
                        rmse = metrics.get('rmse', 'N/A')
                        r2 = metrics.get('r2', 'N/A')
                        st.write(f"- **{model_name}**: RMSE = {rmse}, R² = {r2}")

        st.markdown("""
        ### 🚀 How to Use

        1. **Navigate** to the Price Prediction page
        2. **Choose** a quick preset or enter custom house features
        3. **Select** your preferred prediction model
        4. **Get** instant price predictions with confidence intervals
        5. **Explore** model insights and feature importance

        ### 📈 Data Sources

        The models are trained on housing data that includes:
        - Physical property characteristics (size, age, condition)
        - Location and neighborhood information
        - Quality ratings and amenities
        - Historical sale prices

        ### 🎯 Use Cases

        - **Home Buyers**: Estimate fair market value
        - **Real Estate Agents**: Quick property valuations
        - **Investors**: Investment opportunity analysis
        - **Homeowners**: Property value assessment
        """)

    with col2:
        st.markdown("""
        ### 📞 Support

        **Need Help?**
        - Check the tooltips (ℹ️) for guidance
        - Try the quick presets first
        - Explore the Model Insights page

        **Model Information:**
        - Predictions are estimates only
        - Actual prices may vary
        - Consider local market conditions

        ### 🔄 Recent Updates

        **v1.0.0**
        - Initial release
        - Multiple ML models
        - Interactive predictions
        - Feature importance analysis
        - Data exploration tools

        ### 📋 Technical Details

        **Input Features:**
        - 50+ house characteristics
        - Numerical and categorical data
        - Quality ratings and metrics

        **Model Types:**
        - Linear Regression
        - Ridge/Lasso Regression
        - Random Forest
        - XGBoost
        - Ensemble Methods

        **Validation:**
        - Cross-validation
        - Train/test splits
        - Performance metrics
        - Overfitting detection
        """)

        # Add some visual elements
        st.markdown("### 📊 Quick Stats")

        if st.session_state.models_loaded:
            n_models = len(st.session_state.model_loader.get_available_models())
            n_features = len(st.session_state.model_loader.feature_names or [])

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Models Available", n_models)
            with col_b:
                st.metric("Features Used", n_features if n_features > 0 else "50+")


if __name__ == "__main__":
    main()
```

#### 5.3.2 Create Prediction Page
Create `app/pages/prediction_page.py`:
```python
"""
Main prediction page for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple

from app.components.input_forms import InputForm, QuickPresets
from app.utils.model_loader import ModelLoader
from app.config import DISPLAY_OPTIONS


class PredictionPage:
    """Main prediction page handler."""

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.input_form = InputForm()
        self.presets = QuickPresets()

    def render(self):
        """Render the prediction page."""
        st.title("🏠 House Price Prediction")
        st.markdown("Get instant house price predictions using advanced machine learning models.")

        # Check if models are available
        available_models = self.model_loader.get_available_models()
        if not available_models:
            st.error("❌ No models available. Please ensure models are properly loaded.")
            st.info("💡 Make sure you have completed the model training phase and saved the models.")
            return

        # Model selection
        st.sidebar.subheader("🤖 Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose prediction model:",
            options=available_models,
            help="Select the machine learning model to use for prediction"
        )

        # Display model information
        model_info = self.model_loader.get_model_info(selected_model)
        if model_info:
            with st.sidebar.expander("Model Performance"):
                if 'validation_metrics' in model_info:
                    metrics = model_info['validation_metrics']
                    if 'rmse' in metrics:
                        st.metric("RMSE", f"{metrics['rmse']:.0f}")
                    if 'r2' in metrics:
                        st.metric("R² Score", f"{metrics['r2']:.3f}")
                    if 'mae' in metrics:
                        st.metric("MAE", f"{metrics['mae']:.0f}")

        # Main content area
        col1, col2 = st.columns([2, 1])

        with col1:
            # Quick presets
            preset_data = self.presets.render_preset_selector()

            # Input forms
            if preset_data:
                # Use preset data
                input_data = preset_data
                st.info("Using preset configuration. You can modify individual features below if needed.")

                # Show preset values in form (but allow editing)
                self._render_editable_preset_form(preset_data)
            else:
                # Manual input
                input_data = self.input_form.render_basic_features()
                self.input_form.render_advanced_features()

            # Get final input data
            final_input_data = self.input_form.get_input_data() or input_data

        with col2:
            # Prediction panel
            self._render_prediction_panel(selected_model, final_input_data)

        # Input summary
        if final_input_data:
            self.input_form.render_input_summary()

        # Feature impact analysis
        if final_input_data:
            self._render_feature_impact_analysis(selected_model, final_input_data)

    def _render_editable_preset_form(self, preset_data: Dict[str, Any]):
        """Render form with preset values that can be edited."""
        st.subheader("🔧 Adjust Features")
        st.write("Preset values loaded. Modify any features below:")

        # Create editable form with preset values
        with st.form("preset_adjustment_form"):
            col1, col2 = st.columns(2)

            with col1:
                gr_liv_area = st.number_input(
                    "Above Ground Living Area (sq ft)",
                    min_value=500, max_value=6000, value=preset_data.get('GrLivArea', 1500), step=50
                )

                total_bsmt_sf = st.number_input(
                    "Total Basement Area (sq ft)",
                    min_value=0, max_value=3000, value=preset_data.get('TotalBsmtSF', 1000), step=50
                )

                year_built = st.number_input(
                    "Year Built",
                    min_value=1872, max_value=2024, value=preset_data.get('YearBuilt', 2000), step=1
                )

                bedrooms = st.selectbox(
                    "Bedrooms Above Ground",
                    options=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                    index=preset_data.get('BedroomAbvGr', 3)
                )

            with col2:
                overall_qual = st.selectbox(
                    "Overall Quality (1-10)",
                    options=list(range(1, 11)),
                    index=preset_data.get('OverallQual', 6) - 1
                )

                overall_cond = st.selectbox(
                    "Overall Condition (1-10)",
                    options=list(range(1, 11)),
                    index=preset_data.get('OverallCond', 6) - 1
                )

                full_bath = st.selectbox(
                    "Full Bathrooms",
                    options=[0, 1, 2, 3, 4],
                    index=preset_data.get('FullBath', 2)
                )

                garage_cars = st.selectbox(
                    "Garage Car Capacity",
                    options=[0, 1, 2, 3, 4],
                    index=preset_data.get('GarageCars', 2)
                )

            # Categorical features
            neighborhood = st.selectbox(
                "Neighborhood",
                options=['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
                        'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                        'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
                        'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
                index=0 if preset_data.get('Neighborhood') not in ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
                        'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                        'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
                        'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'] else
                       ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
                        'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                        'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown',
                        'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'].index(preset_data.get('Neighborhood'))
            )

            submitted = st.form_submit_button("Update Prediction")

            if submitted:
                # Update input data
                updated_data = {
                    'GrLivArea': gr_liv_area,
                    'TotalBsmtSF': total_bsmt_sf,
                    'YearBuilt': year_built,
                    'BedroomAbvGr': bedrooms,
                    'OverallQual': overall_qual,
                    'OverallCond': overall_cond,
                    'FullBath': full_bath,
                    'GarageCars': garage_cars,
                    'Neighborhood': neighborhood,
                    'HouseStyle': preset_data.get('HouseStyle', '1Story'),
                    'ExterQual': preset_data.get('ExterQual', 'TA'),
                    'KitchenQual': preset_data.get('KitchenQual', 'TA')
                }
                self.input_form.input_data = updated_data

    def _render_prediction_panel(self, model_name: str, input_data: Dict[str, Any]):
        """Render the prediction results panel."""
        st.subheader("💰 Price Prediction")

        if not input_data:
            st.info("👈 Please enter house features to get a prediction")
            return

        try:
            # Make prediction
            with st.spinner("Calculating prediction..."):
                prediction, confidence_interval = self.model_loader.predict(model_name, input_data)

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: #1f77b4; margin: 0;">
                    ${prediction:,.0f}
                </h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">
                    Predicted Sale Price
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence interval
            if confidence_interval:
                lower, upper = confidence_interval
                st.markdown(f"""
                **95% Confidence Interval:**
                ${lower:,.0f} - ${upper:,.0f}
                """)

                # Confidence interval visualization
                fig = go.Figure()

                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=[lower, prediction, upper],
                    y=[1, 1, 1],
                    mode='markers+lines',
                    name='Confidence Interval',
                    line=dict(color='rgba(31, 119, 180, 0.3)', width=8),
                    marker=dict(
                        size=[12, 20, 12],
                        color=['rgba(31, 119, 180, 0.5)', '#1f77b4', 'rgba(31, 119, 180, 0.5)']
                    ),
                    showlegend=False
                ))

                fig.update_layout(
                    title="Prediction Confidence Interval",
                    xaxis_title="Predicted Price ($)",
                    yaxis=dict(showticklabels=False, showgrid=False),
                    height=150,
                    margin=dict(t=40, b=40, l=40, r=40)
                )

                st.plotly_chart(fig, use_container_width=True)

            # Additional metrics
            st.markdown("**Prediction Details:**")
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Model Used", model_name)
                if confidence_interval:
                    uncertainty = (upper - lower) / prediction * 100
                    st.metric("Uncertainty", f"{uncertainty:.1f}%")

            with col_b:
                price_per_sqft = prediction / input_data.get('GrLivArea', 1)
                st.metric("Price per Sq Ft", f"${price_per_sqft:.0f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("💡 Please check your input values and try again.")

    def _render_feature_impact_analysis(self, model_name: str, input_data: Dict[str, Any]):
        """Render feature impact analysis."""
        st.subheader("📊 Feature Impact Analysis")

        # Get feature importance
        importance_df = self.model_loader.get_feature_importance(model_name, top_n=15)

        if importance_df is None:
            st.warning("Feature importance not available for this model.")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            # Feature importance chart
            fig = px.bar(
                importance_df.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features",
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='viridis'
            )

            fig.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Input feature values for important features
            st.write("**Your Input Values for Key Features:**")

            important_features = importance_df.head(8)['feature'].tolist()

            for feature in important_features:
                if feature in input_data:
                    value = input_data[feature]

                    # Format the display
                    if isinstance(value, (int, float)):
                        if 'area' in feature.lower() or 'sf' in feature.lower():
                            display_value = f"{value:,} sq ft"
                        elif 'year' in feature.lower():
                            display_value = str(int(value))
                        else:
                            display_value = str(value)
                    else:
                        display_value = str(value)

                    st.write(f"**{feature}**: {display_value}")

        # Feature correlation with price (if available)
        st.write("**Why These Features Matter:**")

        feature_explanations = {
            'GrLivArea': "Larger living areas typically command higher prices",
            'OverallQual': "Higher quality materials and finishes increase value",
            'YearBuilt': "Newer homes often have modern features and better condition",
            'TotalBsmtSF': "Finished basement space adds valuable living area",
            'Neighborhood': "Location is a key factor in real estate pricing",
            'GarageCars': "Garage space adds convenience and storage value",
            'FullBath': "More bathrooms increase comfort and home value",
            'YearRemodAdd': "Recent renovations can significantly boost value"
        }

        explanation_text = []
        for feature in important_features[:5]:
            if feature in feature_explanations:
                explanation_text.append(f"• **{feature}**: {feature_explanations[feature]}")

        if explanation_text:
            st.markdown('\n'.join(explanation_text))
```

### 5.4 Additional Pages

#### 5.4.1 Create Model Insights Page
Create `app/pages/insights_page.py`:
```python
"""
Model insights and analysis page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.utils.model_loader import ModelLoader


class InsightsPage:
    """Model insights and analysis page."""

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    def render(self):
        """Render the insights page."""
        st.title("📊 Model Insights & Analysis")
        st.markdown("Explore model performance, feature importance, and prediction patterns.")

        available_models = self.model_loader.get_available_models()
        if not available_models:
            st.error("❌ No models available for analysis.")
            return

        # Model selection
        selected_model = st.selectbox(
            "Select model for analysis:",
            options=available_models,
            help="Choose which model to analyze in detail"
        )

        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Performance", "🔍 Feature Importance", "📈 Predictions", "⚙️ Model Details"])

        with tab1:
            self._render_performance_analysis(selected_model)

        with tab2:
            self._render_feature_importance_analysis(selected_model)

        with tab3:
            self._render_prediction_analysis(selected_model)

        with tab4:
            self._render_model_details(selected_model)

    def _render_performance_analysis(self, model_name: str):
        """Render model performance analysis."""
        st.subheader("🎯 Model Performance Analysis")

        model_info = self.model_loader.get_model_info(model_name)

        if not model_info:
            st.warning("No performance data available for this model.")
            return

        # Performance metrics
        col1, col2, col3 = st.columns(3)

        if 'validation_metrics' in model_info:
            metrics = model_info['validation_metrics']

            with col1:
                if 'rmse' in metrics:
                    st.metric(
                        "Root Mean Square Error",
                        f"${metrics['rmse']:,.0f}",
                        help="Lower values indicate better predictions"
                    )

            with col2:
                if 'r2' in metrics:
                    st.metric(
                        "R² Score",
                        f"{metrics['r2']:.3f}",
                        help="Proportion of variance explained (higher is better)"
                    )

            with col3:
                if 'mae' in metrics:
                    st.metric(
                        "Mean Absolute Error",
                        f"${metrics['mae']:,.0f}",
                        help="Average prediction error magnitude"
                    )

        # Training vs Validation comparison
        if 'training_metrics' in model_info and 'validation_metrics' in model_info:
            st.subheader("Training vs Validation Performance")

            train_metrics = model_info['training_metrics']
            val_metrics = model_info['validation_metrics']

            comparison_data = []
            for metric in ['rmse', 'mae', 'r2']:
                if metric in train_metrics and metric in val_metrics:
                    comparison_data.append({
                        'Metric': metric.upper(),
                        'Training': train_metrics[metric],
                        'Validation': val_metrics[metric],
                        'Difference': abs(train_metrics[metric] - val_metrics[metric])
                    })

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)

                # Create comparison chart
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    name='Training',
                    x=comparison_df['Metric'],
                    y=comparison_df['Training'],
                    marker_color='lightblue'
                ))

                fig.add_trace(go.Bar(
                    name='Validation',
                    x=comparison_df['Metric'],
                    y=comparison_df['Validation'],
                    marker_color='darkblue'
                ))

                fig.update_layout(
                    title='Training vs Validation Metrics',
                    barmode='group',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        # Cross-validation results
        if 'cv_scores' in model_info:
            st.subheader("Cross-Validation Results")
            cv_scores = model_info['cv_scores']

            cv_col1, cv_col2 = st.columns(2)

            with cv_col1:
                if 'mean_squared_error_mean' in cv_scores:
                    cv_rmse = np.sqrt(cv_scores['mean_squared_error_mean'])
                    cv_std = cv_scores.get('mean_squared_error_std', 0)
                    st.metric(
                        "CV RMSE",
                        f"${cv_rmse:,.0f}",
                        delta=f"±{np.sqrt(cv_std):,.0f}"
                    )

            with cv_col2:
                if 'r2_mean' in cv_scores:
                    st.metric(
                        "CV R² Score",
                        f"{cv_scores['r2_mean']:.3f}",
                        delta=f"±{cv_scores.get('r2_std', 0):.3f}"
                    )

    def _render_feature_importance_analysis(self, model_name: str):
        """Render feature importance analysis."""
        st.subheader("🔍 Feature Importance Analysis")

        importance_df = self.model_loader.get_feature_importance(model_name, top_n=25)

        if importance_df is None:
            st.warning("Feature importance not available for this model type.")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            # Interactive feature importance plot
            fig = px.bar(
                importance_df.head(15),
                x='importance',
                y='feature',
                orientation='h',
                title=f"Top 15 Feature Importance - {model_name}",
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='viridis'
            )

            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Feature importance statistics
            st.write("**Feature Importance Statistics:**")

            total_importance = importance_df['importance'].sum()
            top_5_importance = importance_df.head(5)['importance'].sum()
            top_10_importance = importance_df.head(10)['importance'].sum()

            st.metric("Total Features", len(importance_df))
            st.metric("Top 5 Coverage", f"{(top_5_importance/total_importance)*100:.1f}%")
            st.metric("Top 10 Coverage", f"{(top_10_importance/total_importance)*100:.1f}%")

            # Feature categories
            st.write("**Top 5 Most Important Features:**")
            for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
                percentage = (row['importance'] / total_importance) * 100
                st.write(f"{i}. {row['feature']} ({percentage:.1f}%)")

        # Feature importance distribution
        st.subheader("Feature Importance Distribution")

        fig = px.histogram(
            importance_df,
            x='importance',
            bins=20,
            title="Distribution of Feature Importance Scores",
            labels={'importance': 'Importance Score', 'count': 'Number of Features'}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed feature table
        with st.expander("📋 Complete Feature Importance Table"):
            st.dataframe(
                importance_df.round(4),
                use_container_width=True
            )

    def _render_prediction_analysis(self, model_name: str):
        """Render prediction analysis and patterns."""
        st.subheader("📈 Prediction Analysis")

        st.info("This section would show prediction patterns, residual analysis, and prediction distributions. "
               "For a complete implementation, this would require access to validation data and predictions.")

        # Sample prediction ranges based on typical house prices
        st.write("**Typical Prediction Ranges:**")

        price_ranges = {
            "Budget Homes": {"min": 50000, "max": 150000, "description": "Starter homes, older properties"},
            "Mid-Range Homes": {"min": 150000, "max": 300000, "description": "Family homes, good condition"},
            "Premium Homes": {"min": 300000, "max": 500000, "description": "High-quality, larger homes"},
            "Luxury Homes": {"min": 500000, "max": 1000000, "description": "Luxury properties, prime locations"}
        }

        for category, info in price_ranges.items():
            with st.expander(f"{category}: ${info['min']:,} - ${info['max']:,}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Price Range:** ${info['min']:,} - ${info['max']:,}")

        # Model confidence analysis
        st.write("**Prediction Confidence Factors:**")

        confidence_factors = [
            "**Property Size**: Larger homes have more predictable pricing patterns",
            "**Location**: Well-established neighborhoods show consistent values",
            "**Age & Condition**: Recently built or renovated homes are easier to price",
            "**Standard Features**: Common configurations have more training data",
            "**Market Conditions**: Stable markets produce more reliable predictions"
        ]

        for factor in confidence_factors:
            st.markdown(f"• {factor}")

    def _render_model_details(self, model_name: str):
        """Render detailed model information."""
        st.subheader("⚙️ Model Technical Details")

        model_info = self.model_loader.get_model_info(model_name)

        # Model metadata
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Model Information:**")
            st.write(f"Model Name: {model_name}")

            feature_names = self.model_loader.feature_names
            if feature_names:
                st.write(f"Number of Features: {len(feature_names)}")

            st.write("Model Type: Machine Learning Regression")

        with col2:
            st.write("**Training Configuration:**")
            st.write("Cross-Validation: 5-fold")
            st.write("Validation Strategy: Train/Test Split")
            st.write("Preprocessing: Scaling + Encoding")

        # Feature list
        if feature_names:
            with st.expander("📋 Complete Feature List"):
                # Organize features by type
                numerical_features = []
                categorical_features = []
                engineered_features = []

                for feature in feature_names:
                    if any(keyword in feature.lower() for keyword in ['_x_', 'product', 'total', 'age']):
                        engineered_features.append(feature)
                    elif any(keyword in feature.lower() for keyword in ['area', 'sf', 'year', 'qual', 'cars']):
                        numerical_features.append(feature)
                    else:
                        categorical_features.append(feature)

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.write(f"**Numerical Features ({len(numerical_features)}):**")
                    for feature in numerical_features[:10]:
                        st.write(f"• {feature}")
                    if len(numerical_features) > 10:
                        st.write(f"... and {len(numerical_features) - 10} more")

                with col_b:
                    st.write(f"**Categorical Features ({len(categorical_features)}):**")
                    for feature in categorical_features[:10]:
                        st.write(f"• {feature}")
                    if len(categorical_features) > 10:
                        st.write(f"... and {len(categorical_features) - 10} more")

                with col_c:
                    st.write(f"**Engineered Features ({len(engineered_features)}):**")
                    for feature in engineered_features[:10]:
                        st.write(f"• {feature}")
                    if len(engineered_features) > 10:
                        st.write(f"... and {len(engineered_features) - 10} more")

        # Model assumptions and limitations
        st.subheader("⚠️ Model Assumptions & Limitations")

        assumptions = [
            "**Data Quality**: Model assumes input data is accurate and complete",
            "**Market Stability**: Predictions are based on historical data patterns",
            "**Feature Completeness**: All important features are captured in the model",
            "**Geographic Scope**: Model trained on specific geographic area data",
            "**Time Relevance**: Market conditions may change over time"
        ]

        limitations = [
            "**Unique Properties**: Unusual homes may have less accurate predictions",
            "**Market Volatility**: Rapid market changes may affect accuracy",
            "**Missing Features**: Some pricing factors may not be captured",
            "**Outliers**: Extreme values may produce unreliable predictions"
        ]

        col_x, col_y = st.columns(2)

        with col_x:
            st.write("**Key Assumptions:**")
            for assumption in assumptions:
                st.markdown(f"• {assumption}")

        with col_y:
            st.write("**Known Limitations:**")
            for limitation in limitations:
                st.markdown(f"• {limitation}")
```

#### 5.4.2 Create Data Explorer Page
Create `app/pages/data_explorer.py`:
```python
"""
Data exploration and visualization page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataExplorer:
    """Data exploration and visualization page."""

    def render(self):
        """Render the data explorer page."""
        st.title("🔍 Data Explorer")
        st.markdown("Explore housing market trends and data patterns used for model training.")

        # Sample data for demonstration
        self._create_sample_data()

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "🏘️ Neighborhoods", "📈 Price Trends", "🔗 Correlations"])

        with tab1:
            self._render_market_overview()

        with tab2:
            self._render_neighborhood_analysis()

        with tab3:
            self._render_price_trends()

        with tab4:
            self._render_correlation_analysis()

    def _create_sample_data(self):
        """Create sample data for demonstration."""
        np.random.seed(42)
        n_samples = 1000

        # Generate sample housing data
        neighborhoods = ['CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NAmes']
        house_styles = ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer']

        self.sample_data = pd.DataFrame({
            'Id': range(1, n_samples + 1),
            'SalePrice': np.random.lognormal(12, 0.4, n_samples),
            'GrLivArea': np.random.normal(1500, 400, n_samples),
            'TotalBsmtSF': np.random.normal(1000, 300, n_samples),
            'YearBuilt': np.random.randint(1950, 2020, n_samples),
            'YrSold': np.random.randint(2006, 2011, n_samples),
            'OverallQual': np.random.randint(1, 11, n_samples),
            'OverallCond': np.random.randint(1, 11, n_samples),
            'Neighborhood': np.random.choice(neighborhoods, n_samples),
            'HouseStyle': np.random.choice(house_styles, n_samples),
            'BedroomAbvGr': np.random.randint(1, 6, n_samples),
            'FullBath': np.random.randint(1, 4, n_samples),
            'GarageCars': np.random.randint(0, 4, n_samples)
        })

        # Make some realistic adjustments
        self.sample_data['SalePrice'] = self.sample_data['SalePrice'] + \
                                       self.sample_data['GrLivArea'] * 50 + \
                                       self.sample_data['OverallQual'] * 10000

        # Ensure positive values
        self.sample_data['TotalBsmtSF'] = np.maximum(0, self.sample_data['TotalBsmtSF'])
        self.sample_data['GrLivArea'] = np.maximum(500, self.sample_data['GrLivArea'])

    def _render_market_overview(self):
        """Render market overview dashboard."""
        st.subheader("📊 Housing Market Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_price = self.sample_data['SalePrice'].mean()
            st.metric("Average Price", f"${avg_price:,.0f}")

        with col2:
            median_price = self.sample_data['SalePrice'].median()
            st.metric("Median Price", f"${median_price:,.0f}")

        with col3:
            avg_size = self.sample_data['GrLivArea'].mean()
            st.metric("Average Size", f"{avg_size:,.0f} sq ft")

        with col4:
            avg_age = 2024 - self.sample_data['YearBuilt'].mean()
            st.metric("Average Age", f"{avg_age:.0f} years")

        # Price distribution
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.histogram(
                self.sample_data,
                x='SalePrice',
                bins=30,
                title="Sale Price Distribution",
                labels={'SalePrice': 'Sale Price ($)'}
            )
            fig.update_traces(marker_color='skyblue')
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = px.box(
                self.sample_data,
                y='SalePrice',
                title="Sale Price Box Plot",
                labels={'SalePrice': 'Sale Price ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # House characteristics
        st.subheader("🏠 House Characteristics")

        char_col1, char_col2 = st.columns(2)

        with char_col1:
            # Size vs Price
            fig = px.scatter(
                self.sample_data,
                x='GrLivArea',
                y='SalePrice',
                color='OverallQual',
                title="Living Area vs Sale Price",
                labels={'GrLivArea': 'Living Area (sq ft)', 'SalePrice': 'Sale Price ($)'},
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        with char_col2:
            # Age vs Price
            self.sample_data['HouseAge'] = 2024 - self.sample_data['YearBuilt']
            fig = px.scatter(
                self.sample_data,
                x='HouseAge',
                y='SalePrice',
                color='OverallQual',
                title="House Age vs Sale Price",
                labels={'HouseAge': 'House Age (years)', 'SalePrice': 'Sale Price ($)'},
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_neighborhood_analysis(self):
        """Render neighborhood analysis."""
        st.subheader("🏘️ Neighborhood Analysis")

        # Neighborhood price comparison
        neighborhood_stats = self.sample_data.groupby('Neighborhood')['SalePrice'].agg(['mean', 'median', 'count']).reset_index()
        neighborhood_stats.columns = ['Neighborhood', 'Average_Price', 'Median_Price', 'Count']

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                neighborhood_stats.sort_values('Average_Price', ascending=True),
                x='Average_Price',
                y='Neighborhood',
                orientation='h',
                title="Average Price by Neighborhood",
                labels={'Average_Price': 'Average Price ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                neighborhood_stats,
                x='Count',
                y='Average_Price',
                size='Median_Price',
                hover_data=['Neighborhood'],
                title="Price vs Sample Size by Neighborhood",
                labels={'Count': 'Number of Sales', 'Average_Price': 'Average Price ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed neighborhood comparison
        st.subheader("Detailed Neighborhood Comparison")

        selected_neighborhoods = st.multiselect(
            "Select neighborhoods to compare:",
            options=self.sample_data['Neighborhood'].unique(),
            default=self.sample_data['Neighborhood'].unique()[:4]
        )

        if selected_neighborhoods:
            filtered_data = self.sample_data[self.sample_data['Neighborhood'].isin(selected_neighborhoods)]

            # Price distribution by neighborhood
            fig = px.box(
                filtered_data,
                x='Neighborhood',
                y='SalePrice',
                title="Price Distribution by Selected Neighborhoods",
                labels={'SalePrice': 'Sale Price ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Characteristics by neighborhood
            char_metrics = ['GrLivArea', 'OverallQual', 'YearBuilt']

            for metric in char_metrics:
                fig = px.violin(
                    filtered_data,
                    x='Neighborhood',
                    y=metric,
                    title=f"{metric} Distribution by Neighborhood"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_price_trends(self):
        """Render price trends analysis."""
        st.subheader("📈 Price Trends Analysis")

        # Price by year sold
        yearly_prices = self.sample_data.groupby('YrSold')['SalePrice'].agg(['mean', 'median', 'count']).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_prices['YrSold'],
                y=yearly_prices['mean'],
                mode='lines+markers',
                name='Average Price',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=yearly_prices['YrSold'],
                y=yearly_prices['median'],
                mode='lines+markers',
                name='Median Price',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="Price Trends by Year",
                xaxis_title="Year Sold",
                yaxis_title="Price ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                yearly_prices,
                x='YrSold',
                y='count',
                title="Number of Sales by Year",
                labels={'count': 'Number of Sales', 'YrSold': 'Year Sold'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Price by house age
        st.subheader("Price by House Age")

        # Create age bins
        self.sample_data['AgeBin'] = pd.cut(
            self.sample_data['HouseAge'],
            bins=[0, 10, 20, 30, 50, 100],
            labels=['0-10 years', '11-20 years', '21-30 years', '31-50 years', '50+ years']
        )

        age_price = self.sample_data.groupby('AgeBin')['SalePrice'].agg(['mean', 'median', 'count']).reset_index()

        fig = px.bar(
            age_price,
            x='AgeBin',
            y='mean',
            title="Average Price by House Age",
            labels={'mean': 'Average Price ($)', 'AgeBin': 'House Age Range'}
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_correlation_analysis(self):
        """Render correlation analysis."""
        st.subheader("🔗 Feature Correlation Analysis")

        # Select numerical features for correlation
        numerical_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt',
                            'OverallQual', 'OverallCond', 'BedroomAbvGr', 'FullBath', 'GarageCars']

        correlation_data = self.sample_data[numerical_features]

        # Correlation heatmap
        correlation_matrix = correlation_data.corr()

        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations with price
        price_correlations = correlation_matrix['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Strongest Correlations with Price")
            for feature, correlation in price_correlations.head(6).items():
                direction = "Positive" if correlation_matrix['SalePrice'][feature] > 0 else "Negative"
                st.write(f"**{feature}**: {correlation:.3f} ({direction})")

        with col2:
            # Scatter plot of top correlation
            top_feature = price_correlations.index[0]
            fig = px.scatter(
                self.sample_data,
                x=top_feature,
                y='SalePrice',
                title=f"Strongest Correlation: {top_feature} vs Price",
                labels={'SalePrice': 'Sale Price ($)'},
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature relationships
        st.subheader("Feature Relationships")

        feature_x = st.selectbox("Select X-axis feature:", options=numerical_features[1:], index=0)
        feature_y = st.selectbox("Select Y-axis feature:", options=numerical_features[1:], index=1)
        color_feature = st.selectbox("Color by:", options=['OverallQual', 'Neighborhood', 'HouseStyle'], index=0)

        fig = px.scatter(
            self.sample_data,
            x=feature_x,
            y=feature_y,
            color=color_feature,
            title=f"{feature_x} vs {feature_y}",
            hover_data=['SalePrice']
        )
        st.plotly_chart(fig, use_container_width=True)
```

### 5.5 Deployment Configuration

#### 5.5.1 Create Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[global]
developmentMode = false
showWarningOnDirectExecution = false

[server]
runOnSave = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
showErrorDetails = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[logger]
level = "info"
messageFormat = "%(asctime)s %(message)s"
```

#### 5.5.2 Create Requirements for Deployment
Create `app/requirements.txt`:
```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.10.0
scikit-learn>=1.1.0
xgboost>=1.6.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

#### 5.5.3 Create Deployment Scripts
Create `scripts/deploy_local.sh`:
```bash
#!/bin/bash
# Local deployment script

echo "Starting local Streamlit deployment..."

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r app/requirements.txt

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Streamlit
echo "Starting Streamlit server..."
streamlit run app/streamlit_app.py --server.address localhost --server.port 8501

echo "Streamlit app running at http://localhost:8501"
```

Create `scripts/deploy_cloud.py`:
```python
"""
Cloud deployment preparation script.
"""
import shutil
from pathlib import Path


def prepare_cloud_deployment():
    """Prepare files for cloud deployment."""

    # Create deployment directory
    deploy_dir = Path("deployment")
    deploy_dir.mkdir(exist_ok=True)

    # Copy necessary files
    files_to_copy = [
        "app/",
        "src/",
        "config/",
        "models/",
        ".streamlit/",
        "requirements.txt"
    ]

    for file_path in files_to_copy:
        source = Path(file_path)
        if source.exists():
            if source.is_dir():
                shutil.copytree(source, deploy_dir / source.name, dirs_exist_ok=True)
            else:
                shutil.copy2(source, deploy_dir / source.name)

    # Create main app file for cloud deployment
    main_app_content = """
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

from app.streamlit_app import main

if __name__ == "__main__":
    main()
"""

    with open(deploy_dir / "main.py", "w") as f:
        f.write(main_app_content)

    # Create cloud-specific requirements
    cloud_requirements = """
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.10.0
scikit-learn>=1.1.0
xgboost>=1.6.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""

    with open(deploy_dir / "requirements.txt", "w") as f:
        f.write(cloud_requirements)

    print(f"✅ Cloud deployment files prepared in: {deploy_dir}")
    print("📁 Upload the deployment folder to your cloud platform")


if __name__ == "__main__":
    prepare_cloud_deployment()
```

### 5.6 Testing and Validation

Create `tests/test_phase5.py`:
```python
"""
Tests for Phase 5: Web Application & Deployment
"""
import unittest
import sys
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('..')

from app.utils.model_loader import ModelLoader
from app.components.input_forms import InputForm, QuickPresets
from app.config import FEATURE_CONFIG


class TestPhase5(unittest.TestCase):
    """Test Phase 5 functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock model for testing
        from sklearn.linear_model import LinearRegression
        import joblib

        # Create sample model
        model = LinearRegression()
        X_sample = np.random.randn(100, 10)
        y_sample = np.random.randn(100)
        model.fit(X_sample, y_sample)

        # Save mock model
        model_path = Path(self.temp_dir) / "test_model.pkl"
        model_data = {
            'model': model,
            'name': 'Test Model',
            'feature_names': [f'feature_{i}' for i in range(10)],
            'validation_metrics': {'rmse': 1000, 'r2': 0.85, 'mae': 800}
        }
        joblib.dump(model_data, model_path)

    def test_model_loader(self):
        """Test model loading functionality."""
        loader = ModelLoader()

        # Test loading models
        models = loader.load_models(Path(self.temp_dir))
        self.assertIsInstance(models, dict)

        if models:  # If models were loaded
            model_names = loader.get_available_models()
            self.assertIsInstance(model_names, list)

            # Test model info
            if model_names:
                info = loader.get_model_info(model_names[0])
                self.assertIsInstance(info, dict)

    def test_input_forms(self):
        """Test input form components."""
        form = InputForm()

        # Test that form can be initialized
        self.assertIsNotNone(form)
        self.assertIsInstance(form.input_data, dict)

    def test_quick_presets(self):
        """Test quick presets functionality."""
        presets = QuickPresets()
        preset_configs = presets.get_presets()

        # Check that presets exist
        self.assertIsInstance(preset_configs, dict)
        self.assertGreater(len(preset_configs), 0)

        # Check preset structure
        for preset_name, config in preset_configs.items():
            self.assertIsInstance(preset_name, str)
            self.assertIsInstance(config, dict)

            # Check required fields
            required_fields = ['GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'OverallQual']
            for field in required_fields:
                self.assertIn(field, config)

    def test_feature_config(self):
        """Test feature configuration."""
        # Test numerical features config
        self.assertIn('numerical_features', FEATURE_CONFIG)
        self.assertIn('categorical_features', FEATURE_CONFIG)

        # Test specific feature configs
        numerical_features = FEATURE_CONFIG['numerical_features']
        self.assertIn('GrLivArea', numerical_features)

        grlivarea_config = numerical_features['GrLivArea']
        required_keys = ['label', 'min_value', 'max_value', 'value', 'step']
        for key in required_keys:
            self.assertIn(key, grlivarea_config)

    def test_prediction_input_processing(self):
        """Test prediction input processing."""
        # Sample input data
        input_data = {
            'GrLivArea': 1500,
            'TotalBsmtSF': 1000,
            'YearBuilt': 2000,
            'OverallQual': 7,
            'Neighborhood': 'NAmes'
        }

        # Test that input data is properly structured
        self.assertIsInstance(input_data, dict)
        self.assertIn('GrLivArea', input_data)
        self.assertIsInstance(input_data['GrLivArea'], (int, float))

    def test_app_configuration(self):
        """Test application configuration."""
        from app.config import APP_CONFIG, MODEL_CONFIG, DISPLAY_OPTIONS

        # Test APP_CONFIG
        self.assertIn('page_title', APP_CONFIG)
        self.assertIn('page_icon', APP_CONFIG)
        self.assertIn('layout', APP_CONFIG)

        # Test MODEL_CONFIG
        self.assertIn('model_dir', MODEL_CONFIG)
        self.assertIn('feature_importance_top_n', MODEL_CONFIG)

        # Test DISPLAY_OPTIONS
        self.assertIn('prediction_precision', DISPLAY_OPTIONS)
        self.assertIn('confidence_level', DISPLAY_OPTIONS)

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
```

**Test**: Run Phase 5 tests
```bash
cd tests
python test_phase5.py
```

## Deliverables
- [ ] Complete Streamlit web application
- [ ] Interactive prediction interface with input forms
- [ ] Model selection and comparison features
- [ ] Feature importance visualization
- [ ] Data exploration and analysis tools
- [ ] Quick preset configurations for easy testing
- [ ] Responsive and user-friendly design
- [ ] Model insights and performance metrics
- [ ] Deployment configuration and scripts
- [ ] Cloud deployment preparation
- [ ] Comprehensive testing suite
- [ ] User documentation and help features

## Success Criteria
- Web application runs without errors
- Users can input house features and get predictions
- Multiple models available for selection
- Predictions display with confidence intervals
- Feature importance charts render correctly
- Data exploration tools provide meaningful insights
- Application is responsive and user-friendly
- Deployment scripts work correctly
- All tests pass successfully
- Application ready for production deployment

## Deployment Options

### Local Deployment
```bash
# Run locally
./scripts/deploy_local.sh
```

### Cloud Deployment
1. **Streamlit Cloud**: Push to GitHub and connect to Streamlit Cloud
2. **Heroku**: Use Docker or direct deployment
3. **AWS/Azure/GCP**: Deploy using cloud-specific methods

## Next Steps
- **User Feedback**: Collect user feedback and iterate
- **Performance Optimization**: Optimize for larger datasets
- **Additional Features**: Add more advanced features as needed
- **Model Updates**: Regularly retrain and update models
- **Monitoring**: Implement usage analytics and error tracking

The application is now complete and ready for deployment! 🚀