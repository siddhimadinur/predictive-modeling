"""
Model loading and prediction utilities for California Housing Streamlit app.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from typing import Dict, Any, Tuple, Optional, List
import streamlit as st

from app.config import MODEL_CONFIG


class CaliforniaHousingModelLoader:
    """Handle model loading and predictions for California housing."""

    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.feature_names = None
        self.deployment_info = None

    @st.cache
    def load_models(_self, model_dir: Path = None) -> Dict[str, Any]:
        """
        Load all available California housing models.

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
            st.info("💡 Please ensure Phase 4 (Model Training) has been completed")
            return models

        # Load deployment summary if available
        deployment_path = model_dir / 'deployment_summary.json'
        if deployment_path.exists():
            try:
                with open(deployment_path, 'r') as f:
                    _self.deployment_info = json.load(f)
            except Exception as e:
                st.warning(f"Could not load deployment summary: {e}")

        # Load individual model files
        for model_file in model_dir.glob("*california_housing*.pkl"):
            try:
                model_data = joblib.load(model_file)

                # Handle different model save formats
                if isinstance(model_data, dict) and 'model' in model_data:
                    # New format with metadata
                    model_name = model_data.get('name', model_file.stem.replace('_california_housing', ''))
                    models[model_name] = model_data['model']
                    metadata[model_name] = {
                        'feature_names': model_data.get('feature_names', []),
                        'training_metrics': model_data.get('training_metrics', {}),
                        'validation_metrics': model_data.get('validation_metrics', {}),
                        'cv_scores': model_data.get('cv_scores', {}),
                        'model_type': model_data.get('model_type', 'california_housing_predictor')
                    }
                else:
                    # Old format - just the model
                    model_name = model_file.stem.replace('_california_housing', '')
                    models[model_name] = model_data
                    metadata[model_name] = {}

                st.success(f"✅ Loaded model: {model_name}")

            except Exception as e:
                st.warning(f"Could not load model {model_file.name}: {str(e)}")

        _self.models = models
        _self.model_metadata = metadata

        if models:
            # Set feature names from first model with metadata
            for model_meta in metadata.values():
                if model_meta.get('feature_names'):
                    _self.feature_names = model_meta['feature_names']
                    break

        st.success(f"📊 Loaded {len(models)} California housing models")
        return models

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

        model_info = self.model_metadata[model_name].copy()

        # Add deployment info if available
        if self.deployment_info and model_name == self.deployment_info.get('champion_model', '').replace('_california_housing', ''):
            model_info['is_champion'] = True
            model_info['deployment_info'] = self.deployment_info

        return model_info

    def get_champion_model(self) -> Tuple[Optional[str], Optional[Any]]:
        """
        Get the champion model if available.

        Returns:
            Tuple of (model_name, model) or (None, None)
        """
        if self.deployment_info:
            champion_name = self.deployment_info.get('champion_model', '')
            champion_name = champion_name.replace('_california_housing_model.pkl', '')

            if champion_name in self.models:
                return champion_name, self.models[champion_name]

        # Fallback: return first model
        if self.models:
            first_model_name = list(self.models.keys())[0]
            return first_model_name, self.models[first_model_name]

        return None, None

    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess user input for California housing prediction.

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
                    # California housing specific defaults
                    if 'distance_to_' in feature:
                        # Calculate approximate distance (simplified)
                        if 'Los_Angeles' in feature:
                            lat_diff = abs(input_data.get('latitude', 34.0) - 34.05)
                            lon_diff = abs(input_data.get('longitude', -118.2) - (-118.24))
                            input_df[feature] = np.sqrt(lat_diff**2 + lon_diff**2)
                        elif 'San_Francisco' in feature:
                            lat_diff = abs(input_data.get('latitude', 37.8) - 37.77)
                            lon_diff = abs(input_data.get('longitude', -122.4) - (-122.42))
                            input_df[feature] = np.sqrt(lat_diff**2 + lon_diff**2)
                        else:
                            input_df[feature] = 1.0  # Default distance

                    elif 'per_household' in feature:
                        # Calculate ratios
                        if 'rooms_per_household' in feature:
                            rooms = input_data.get('total_rooms', 2500)
                            households = input_data.get('households', 800)
                            input_df[feature] = rooms / max(households, 1)
                        elif 'population_per_household' in feature:
                            population = input_data.get('population', 2000)
                            households = input_data.get('households', 800)
                            input_df[feature] = population / max(households, 1)

                    elif 'bedrooms_per_room' in feature:
                        bedrooms = input_data.get('total_bedrooms', 500)
                        rooms = input_data.get('total_rooms', 2500)
                        input_df[feature] = bedrooms / max(rooms, 1)

                    elif 'is_northern_ca' in feature:
                        latitude = input_data.get('latitude', 37.0)
                        input_df[feature] = 1 if latitude > 36.0 else 0

                    elif 'is_coastal' in feature:
                        longitude = input_data.get('longitude', -118.0)
                        input_df[feature] = 1 if longitude > -121.0 else 0

                    elif feature.startswith('ocean_proximity_'):
                        # One-hot encoded ocean proximity
                        user_proximity = input_data.get('ocean_proximity', 'INLAND')
                        feature_proximity = feature.replace('ocean_proximity_', '')
                        input_df[feature] = 1 if user_proximity == feature_proximity else 0

                    elif feature.startswith('income_category_'):
                        # Income category encoding
                        income = input_data.get('median_income', 5.0)
                        if income <= 2: category = 'Very_Low'
                        elif income <= 4: category = 'Low'
                        elif income <= 6: category = 'Medium'
                        elif income <= 8: category = 'High'
                        elif income <= 10: category = 'Very_High'
                        else: category = 'Ultra_High'

                        feature_category = feature.replace('income_category_', '')
                        input_df[feature] = 1 if category == feature_category else 0

                    elif feature.startswith('age_category_'):
                        # Age category encoding
                        age = input_data.get('housing_median_age', 25)
                        if age <= 10: category = 'New'
                        elif age <= 20: category = 'Modern'
                        elif age <= 30: category = 'Mature'
                        elif age <= 40: category = 'Older'
                        else: category = 'Vintage'

                        feature_category = feature.replace('age_category_', '')
                        input_df[feature] = 1 if category == feature_category else 0

                    elif 'log_median_income' in feature:
                        income = input_data.get('median_income', 5.0)
                        input_df[feature] = np.log1p(income)

                    elif 'squared' in feature:
                        # Polynomial features
                        base_feature = feature.replace('_squared', '')
                        if base_feature in input_data:
                            input_df[feature] = input_data[base_feature] ** 2

                    elif 'interaction' in feature:
                        # Interaction features
                        if 'income_rooms' in feature:
                            income = input_data.get('median_income', 5.0)
                            rooms = input_data.get('total_rooms', 2500)
                            input_df[feature] = income * rooms / 1000
                        elif 'income_age' in feature:
                            income = input_data.get('median_income', 5.0)
                            age = input_data.get('housing_median_age', 25)
                            input_df[feature] = income * (50 - age)

                    else:
                        # Default values for unknown features
                        input_df[feature] = 0

            # Reorder columns to match training feature order
            input_df = input_df.reindex(columns=self.feature_names, fill_value=0)

        return input_df

    def predict(self, model_name: str, input_data: Dict[str, Any]) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Make California housing price prediction using specified model.

        Args:
            model_name: Name of the model to use
            input_data: Input features dictionary

        Returns:
            Tuple of (prediction, additional_info)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        # Preprocess input
        processed_input = self.preprocess_input(input_data)

        try:
            # Make prediction
            prediction = model.predict(processed_input)[0]

            # Calculate additional info
            additional_info = {
                'prediction_per_sqft': prediction / max(input_data.get('total_rooms', 1), 1) * 500,  # Rough estimate
                'income_to_price_ratio': prediction / max(input_data.get('median_income', 1) * 10000, 1),
                'model_used': model_name
            }

            # Add confidence interval estimation if possible
            try:
                # For tree-based models, use estimator variance for confidence
                if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                    # Get predictions from individual trees
                    tree_predictions = []
                    for estimator in model.estimators_[:min(100, len(model.estimators_))]:
                        pred = estimator.predict(processed_input)[0]
                        tree_predictions.append(pred)

                    prediction_std = np.std(tree_predictions)
                    confidence_interval = {
                        'lower': prediction - 1.96 * prediction_std,
                        'upper': prediction + 1.96 * prediction_std,
                        'std': prediction_std
                    }
                    additional_info['confidence_interval'] = confidence_interval

            except Exception:
                # If confidence interval calculation fails, continue without it
                pass

            return prediction, additional_info

        except Exception as e:
            raise RuntimeError(f"Prediction failed for {model_name}: {str(e)}")

    def get_feature_importance(self, model_name: str, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Get feature importance from California housing model.

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

    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get deployment summary information."""
        return self.deployment_info or {}

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate user input for California housing.

        Args:
            input_data: User input dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        required_fields = ['median_income', 'total_rooms', 'housing_median_age', 'longitude', 'latitude']
        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate California-specific ranges
        if input_data.get('longitude'):
            if not (-124.5 <= input_data['longitude'] <= -114.0):
                errors.append("Longitude must be within California range (-124.5 to -114.0)")

        if input_data.get('latitude'):
            if not (32.5 <= input_data['latitude'] <= 42.0):
                errors.append("Latitude must be within California range (32.5 to 42.0)")

        if input_data.get('median_income'):
            if not (0.5 <= input_data['median_income'] <= 15.0):
                errors.append("Median income must be between $5,000 and $150,000")

        if input_data.get('housing_median_age'):
            if not (1 <= input_data['housing_median_age'] <= 52):
                errors.append("Housing age must be between 1 and 52 years")

        # Logical consistency checks
        if input_data.get('total_bedrooms') and input_data.get('total_rooms'):
            if input_data['total_bedrooms'] > input_data['total_rooms']:
                errors.append("Total bedrooms cannot exceed total rooms")

        if input_data.get('households') and input_data.get('population'):
            if input_data['households'] > input_data['population']:
                errors.append("Number of households cannot exceed population")

        return len(errors) == 0, errors

    def get_california_region(self, latitude: float) -> str:
        """
        Determine California region based on latitude.

        Args:
            latitude: Latitude coordinate

        Returns:
            Region name
        """
        if latitude >= 37.0:
            return "Northern California"
        elif latitude >= 35.0:
            return "Central California"
        else:
            return "Southern California"

    def estimate_property_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide context about the property based on input features.

        Args:
            input_data: User input data

        Returns:
            Dictionary with property context
        """
        context = {}

        # Geographic context
        latitude = input_data.get('latitude', 37.0)
        longitude = input_data.get('longitude', -118.0)
        context['region'] = self.get_california_region(latitude)

        # Income context
        income = input_data.get('median_income', 5.0)
        if income < 3:
            context['income_level'] = "Low income area"
        elif income < 6:
            context['income_level'] = "Moderate income area"
        elif income < 10:
            context['income_level'] = "High income area"
        else:
            context['income_level'] = "Very high income area"

        # Housing characteristics
        age = input_data.get('housing_median_age', 25)
        if age < 10:
            context['housing_character'] = "New development area"
        elif age < 25:
            context['housing_character'] = "Modern housing area"
        elif age < 40:
            context['housing_character'] = "Established neighborhood"
        else:
            context['housing_character'] = "Mature/vintage housing area"

        # Density analysis
        rooms = input_data.get('total_rooms', 2500)
        households = input_data.get('households', 800)
        if households > 0:
            rooms_per_household = rooms / households
            if rooms_per_household < 4:
                context['density'] = "High density area (small units)"
            elif rooms_per_household < 6:
                context['density'] = "Medium density area"
            else:
                context['density'] = "Low density area (large homes)"

        # Ocean proximity context
        proximity = input_data.get('ocean_proximity', 'INLAND')
        proximity_descriptions = {
            'NEAR BAY': "Close to San Francisco Bay - premium location",
            'NEAR OCEAN': "Close to Pacific Ocean - highly desirable",
            '<1H OCEAN': "Within 1 hour of ocean - good access",
            'INLAND': "Inland location - more affordable",
            'ISLAND': "Island location - unique premium"
        }
        context['location_appeal'] = proximity_descriptions.get(proximity, "Standard location")

        return context


# Test the model loader
if __name__ == "__main__":
    print("Testing California Housing Model Loader...")

    try:
        loader = CaliforniaHousingModelLoader()

        # Test input preprocessing
        sample_input = {
            'median_income': 8.5,
            'total_rooms': 3500,
            'total_bedrooms': 700,
            'housing_median_age': 28,
            'population': 4000,
            'households': 1100,
            'longitude': -118.2,
            'latitude': 34.1,
            'ocean_proximity': 'NEAR OCEAN'
        }

        # Test validation
        is_valid, errors = loader.validate_input(sample_input)
        print(f"✅ Input validation: {'Valid' if is_valid else 'Invalid'}")
        if errors:
            for error in errors:
                print(f"  • {error}")

        # Test context estimation
        context = loader.estimate_property_context(sample_input)
        print(f"\\n📍 Property Context:")
        for key, value in context.items():
            print(f"  • {key}: {value}")

        print(f"\\n✅ Model Loader working correctly")

    except Exception as e:
        print(f"❌ Error in Model Loader: {e}")
        import traceback
        traceback.print_exc()