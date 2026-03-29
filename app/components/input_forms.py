"""
Input form components for California Housing Streamlit app.
Uses the real sklearn California Housing dataset features.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any

from app.config import FEATURE_CONFIG, CALIFORNIA_PRESETS


class CaliforniaHousingInputForm:
    """Handle user input forms for California housing prediction."""

    def __init__(self):
        self.input_data = {}

    def render_location_features(self) -> Dict[str, Any]:
        """Render location-based input features."""
        st.subheader("Location")

        col1, col2 = st.columns(2)

        with col1:
            longitude = st.number_input(
                **FEATURE_CONFIG['numerical_features']['longitude']
            )

        with col2:
            latitude = st.number_input(
                **FEATURE_CONFIG['numerical_features']['latitude']
            )

            if latitude:
                if latitude >= 37.0:
                    region = "Northern California (Bay Area, Sacramento)"
                elif latitude >= 35.0:
                    region = "Central California (Central Valley, Monterey)"
                else:
                    region = "Southern California (LA, Orange County, San Diego)"
                st.info(f"Region: {region}")

        return {'longitude': longitude, 'latitude': latitude}

    def render_housing_features(self) -> Dict[str, Any]:
        """Render housing characteristics input."""
        st.subheader("Housing Characteristics")

        col1, col2 = st.columns(2)

        with col1:
            ave_rooms = st.number_input(
                **FEATURE_CONFIG['numerical_features']['ave_rooms']
            )
            ave_bedrooms = st.number_input(
                **FEATURE_CONFIG['numerical_features']['ave_bedrooms']
            )
            if ave_bedrooms > ave_rooms:
                st.warning("Bedrooms per household cannot exceed rooms per household")

        with col2:
            housing_median_age = st.number_input(
                **FEATURE_CONFIG['numerical_features']['housing_median_age']
            )

            if housing_median_age <= 10:
                age_desc = "New Construction"
            elif housing_median_age <= 20:
                age_desc = "Modern Housing"
            elif housing_median_age <= 30:
                age_desc = "Mature Housing"
            elif housing_median_age <= 40:
                age_desc = "Older Housing"
            else:
                age_desc = "Vintage Housing"
            st.info(f"Category: {age_desc}")

        return {
            'ave_rooms': ave_rooms,
            'ave_bedrooms': min(ave_bedrooms, ave_rooms),
            'housing_median_age': housing_median_age,
        }

    def render_demographic_features(self) -> Dict[str, Any]:
        """Render demographic input features."""
        st.subheader("Area Demographics")

        col1, col2 = st.columns(2)

        with col1:
            median_income = st.number_input(
                **FEATURE_CONFIG['numerical_features']['median_income']
            )
            population = st.number_input(
                **FEATURE_CONFIG['numerical_features']['population']
            )

        with col2:
            ave_occupancy = st.number_input(
                **FEATURE_CONFIG['numerical_features']['ave_occupancy']
            )

        # Income level context
        if median_income < 3:
            income_desc = "Low Income Area"
        elif median_income < 6:
            income_desc = "Moderate Income Area"
        elif median_income < 10:
            income_desc = "High Income Area"
        else:
            income_desc = "Very High Income Area"
        st.info(f"{income_desc} (${median_income*10:,.0f}K median)")

        return {
            'median_income': median_income,
            'population': population,
            'ave_occupancy': ave_occupancy,
        }

    def render_complete_form(self) -> Dict[str, Any]:
        """Render the complete input form."""
        location_data = self.render_location_features()
        housing_data = self.render_housing_features()
        demographic_data = self.render_demographic_features()

        self.input_data = {**location_data, **housing_data, **demographic_data}
        return self.input_data

    def render_input_summary(self) -> None:
        """Render summary of user inputs."""
        if not self.input_data:
            return

        st.subheader("Input Summary")

        categories = {
            "Location": {
                'Longitude': self.input_data.get('longitude', 0),
                'Latitude': self.input_data.get('latitude', 0),
            },
            "Housing": {
                'Avg Rooms/HH': f"{self.input_data.get('ave_rooms', 0):.1f}",
                'Avg Bedrooms/HH': f"{self.input_data.get('ave_bedrooms', 0):.1f}",
                'Housing Age': f"{self.input_data.get('housing_median_age', 0)} years",
            },
            "Demographics": {
                'Population': f"{self.input_data.get('population', 0):,}",
                'Avg HH Size': f"{self.input_data.get('ave_occupancy', 0):.1f}",
                'Median Income': f"${self.input_data.get('median_income', 0)*10:,.0f}K",
            }
        }

        cols = st.columns(len(categories))
        for i, (category, items) in enumerate(categories.items()):
            with cols[i]:
                st.write(f"**{category}**")
                for key, value in items.items():
                    st.write(f"- {key}: {value}")

    def get_input_data(self) -> Dict[str, Any]:
        """Get the current input data."""
        return self.input_data


class CaliforniaPresets:
    """Predefined California area configurations for quick testing."""

    @staticmethod
    def get_presets() -> Dict[str, Dict[str, Any]]:
        """Get predefined California area configurations."""
        return CALIFORNIA_PRESETS

    def render_preset_selector(self) -> Dict[str, Any]:
        """Render preset selector for California areas."""
        st.subheader("Quick Start - California Areas")
        st.write("Choose a predefined California area to get started quickly:")

        presets = self.get_presets()
        preset_names = list(presets.keys())

        selected_preset = st.selectbox(
            "Choose a California area:",
            options=["Custom Input"] + preset_names,
            index=0,
            help="Select a predefined area configuration or choose 'Custom Input' for manual entry"
        )

        if selected_preset != "Custom Input":
            preset_data = presets[selected_preset]
            st.success(f"{selected_preset} configuration loaded!")

            with st.expander("View area details"):
                st.markdown(
                    f"**Location:** {preset_data['longitude']}, {preset_data['latitude']}  \n"
                    f"**Housing:** {preset_data['ave_rooms']:.1f} rooms/hh, "
                    f"{preset_data['ave_bedrooms']:.1f} bedrooms/hh, "
                    f"{preset_data['housing_median_age']} yrs old  \n"
                    f"**Demographics:** Pop {preset_data['population']:,}, "
                    f"{preset_data['ave_occupancy']:.1f} people/hh, "
                    f"${preset_data['median_income']*10:,.0f}K income"
                )

            return preset_data

        return {}
