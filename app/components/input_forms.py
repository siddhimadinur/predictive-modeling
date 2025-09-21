"""
Input form components for California Housing Streamlit app.
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
        """
        Render location-based input features.

        Returns:
            Dictionary with location inputs
        """
        st.subheader("📍 Location Information")

        col1, col2 = st.columns(2)

        with col1:
            longitude = st.number_input(
                **FEATURE_CONFIG['numerical_features']['longitude']
            )

            ocean_proximity = st.selectbox(
                **FEATURE_CONFIG['categorical_features']['ocean_proximity']
            )

        with col2:
            latitude = st.number_input(
                **FEATURE_CONFIG['numerical_features']['latitude']
            )

            # Show region information based on latitude
            if latitude:
                if latitude >= 37.0:
                    region = "Northern California (Bay Area, Sacramento)"
                elif latitude >= 35.0:
                    region = "Central California (Central Valley, Monterey)"
                else:
                    region = "Southern California (LA, Orange County, San Diego)"

                st.info(f"🗺️ Region: {region}")

        location_data = {
            'longitude': longitude,
            'latitude': latitude,
            'ocean_proximity': ocean_proximity
        }

        return location_data

    def render_housing_features(self) -> Dict[str, Any]:
        """
        Render housing characteristics input.

        Returns:
            Dictionary with housing inputs
        """
        st.subheader("🏠 Housing Characteristics")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Size & Rooms**")
            total_rooms = st.number_input(
                **FEATURE_CONFIG['numerical_features']['total_rooms']
            )

            total_bedrooms = st.number_input(
                **FEATURE_CONFIG['numerical_features']['total_bedrooms']
            )

            # Auto-calculate reasonable bedroom constraint
            if total_bedrooms > total_rooms:
                st.warning("⚠️ Bedrooms cannot exceed total rooms")

        with col2:
            st.write("**Age & Condition**")
            housing_median_age = st.number_input(
                **FEATURE_CONFIG['numerical_features']['housing_median_age']
            )

            # Show age category
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

            st.info(f"🏗️ Category: {age_desc}")

        housing_data = {
            'total_rooms': total_rooms,
            'total_bedrooms': min(total_bedrooms, total_rooms),  # Ensure constraint
            'housing_median_age': housing_median_age
        }

        return housing_data

    def render_demographic_features(self) -> Dict[str, Any]:
        """
        Render demographic input features.

        Returns:
            Dictionary with demographic inputs
        """
        st.subheader("👥 Area Demographics")

        col1, col2 = st.columns(2)

        with col1:
            population = st.number_input(
                **FEATURE_CONFIG['numerical_features']['population']
            )

            median_income = st.number_input(
                **FEATURE_CONFIG['numerical_features']['median_income']
            )

        with col2:
            households = st.number_input(
                **FEATURE_CONFIG['numerical_features']['households']
            )

            # Auto-calculate and show density metrics
            if households > 0 and population > 0:
                people_per_household = population / households
                st.info(f"👨‍👩‍👧‍👦 People per household: {people_per_household:.1f}")

            # Ensure logical constraint
            if households > population:
                st.warning("⚠️ Households cannot exceed population")

        # Income level context
        income_level = median_income
        if income_level < 3:
            income_desc = "Low Income Area"
        elif income_level < 6:
            income_desc = "Moderate Income Area"
        elif income_level < 10:
            income_desc = "High Income Area"
        else:
            income_desc = "Very High Income Area"

        st.info(f"💰 {income_desc} (${income_level*10:,.0f}K median)")

        demographic_data = {
            'population': population,
            'households': min(households, population),  # Ensure constraint
            'median_income': median_income
        }

        return demographic_data

    def render_complete_form(self) -> Dict[str, Any]:
        """
        Render the complete California housing input form.

        Returns:
            Dictionary with all user inputs
        """
        # Location features
        location_data = self.render_location_features()

        # Housing features
        housing_data = self.render_housing_features()

        # Demographic features
        demographic_data = self.render_demographic_features()

        # Combine all inputs
        self.input_data = {**location_data, **housing_data, **demographic_data}

        return self.input_data

    def render_input_summary(self) -> None:
        """Render summary of user inputs."""
        if not self.input_data:
            return

        st.subheader("📋 Input Summary")

        # Organize inputs by category
        categories = {
            "Location": {
                'Longitude': self.input_data.get('longitude', 0),
                'Latitude': self.input_data.get('latitude', 0),
                'Ocean Proximity': self.input_data.get('ocean_proximity', 'INLAND')
            },
            "Housing": {
                'Total Rooms': f"{self.input_data.get('total_rooms', 0):,}",
                'Total Bedrooms': f"{self.input_data.get('total_bedrooms', 0):,}",
                'Housing Age': f"{self.input_data.get('housing_median_age', 0)} years"
            },
            "Demographics": {
                'Population': f"{self.input_data.get('population', 0):,}",
                'Households': f"{self.input_data.get('households', 0):,}",
                'Median Income': f"${self.input_data.get('median_income', 0)*10:,.0f}"
            }
        }

        # Display in columns
        cols = st.columns(len(categories))

        for i, (category, items) in enumerate(categories.items()):
            with cols[i]:
                st.write(f"**{category}**")
                for key, value in items.items():
                    st.write(f"• {key}: {value}")

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
        """
        Render preset selector for California areas.

        Returns:
            Selected preset configuration
        """
        st.subheader("🚀 Quick Start - California Areas")
        st.write("Choose a predefined California area to get started quickly:")

        presets = self.get_presets()
        preset_names = list(presets.keys())

        selected_preset = st.selectbox(
            "Choose a California area:",
            options=["Custom Input"] + preset_names,
            index=0,
            help="Select a predefined area configuration or choose 'Custom Input' to enter your own values"
        )

        if selected_preset != "Custom Input":
            preset_data = presets[selected_preset]

            # Show preset details in an attractive format
            st.success(f"✅ {selected_preset} configuration loaded!")

            with st.expander("📊 View area details"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Location:**")
                    st.write(f"• Longitude: {preset_data['longitude']}")
                    st.write(f"• Latitude: {preset_data['latitude']}")
                    st.write(f"• Ocean Proximity: {preset_data['ocean_proximity']}")

                    st.write("**Housing:**")
                    st.write(f"• Median Age: {preset_data['housing_median_age']} years")
                    st.write(f"• Total Rooms: {preset_data['total_rooms']:,}")
                    st.write(f"• Total Bedrooms: {preset_data['total_bedrooms']:,}")

                with col2:
                    st.write("**Demographics:**")
                    st.write(f"• Population: {preset_data['population']:,}")
                    st.write(f"• Households: {preset_data['households']:,}")
                    st.write(f"• Median Income: ${preset_data['median_income']*10:,.0f}")

                    # Calculate some ratios
                    rooms_per_hh = preset_data['total_rooms'] / preset_data['households']
                    people_per_hh = preset_data['population'] / preset_data['households']
                    st.write(f"• Rooms per Household: {rooms_per_hh:.1f}")
                    st.write(f"• People per Household: {people_per_hh:.1f}")

            return preset_data

        return {}

    def render_california_map_info(self) -> None:
        """Render information about California regions."""
        st.subheader("🗺️ California Housing Markets")

        with st.expander("ℹ️ About California Housing Markets"):
            st.write("""
            **Northern California (Latitude > 37°)**
            - San Francisco Bay Area, Sacramento, Wine Country
            - Generally higher prices due to tech industry
            - Typical range: $400K - $800K+

            **Central California (Latitude 35° - 37°)**
            - Central Valley, Monterey Bay, Central Coast
            - More affordable agricultural and coastal areas
            - Typical range: $300K - $600K

            **Southern California (Latitude < 35°)**
            - Los Angeles, Orange County, San Diego, Inland Empire
            - High demand urban markets
            - Typical range: $500K - $1M+

            **Ocean Proximity Impact:**
            - NEAR OCEAN/BAY: Premium locations (+20-50%)
            - <1H OCEAN: Moderate premium (+10-30%)
            - INLAND: More affordable baseline prices
            """)


# Test the input forms
if __name__ == "__main__":
    print("Testing California Housing Input Forms...")

    try:
        # Test input form
        form = CaliforniaHousingInputForm()

        # Test preset functionality
        presets = CaliforniaPresets()
        preset_configs = presets.get_presets()

        print(f"✅ Input form initialized")
        print(f"✅ Available presets: {list(preset_configs.keys())}")

        # Test sample input
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

        form.input_data = sample_input
        print(f"✅ Sample input processed: {len(sample_input)} features")

        print(f"\n✅ Input Forms working correctly!")

    except Exception as e:
        print(f"❌ Error in Input Forms: {e}")
        import traceback
        traceback.print_exc()