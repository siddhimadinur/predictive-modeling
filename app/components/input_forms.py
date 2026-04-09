"""
Simplified input form for California Housing Streamlit app.
Users pick a city and adjust 3 simple sliders — no census jargon.
"""
import streamlit as st
from typing import Dict, Any

from app.config import CALIFORNIA_CITIES


class CaliforniaHousingInputForm:
    """Simple input form: city dropdown + rooms/bedrooms/household size sliders."""

    def __init__(self):
        self.input_data = {}

    def render_complete_form(self) -> Dict[str, Any]:
        """Render the simplified input form."""

        # --- City selector ---
        st.subheader("Pick a Location")
        city_names = list(CALIFORNIA_CITIES.keys())
        selected_city = st.selectbox(
            "City / Neighborhood",
            options=city_names,
            index=city_names.index("Los Angeles"),
            help="We auto-fill location details based on your selection",
        )

        city_data = CALIFORNIA_CITIES[selected_city]

        # --- Property sliders (integers only) ---
        st.subheader("Describe Your Property")

        col1, col2, col3 = st.columns(3)

        with col1:
            rooms = st.slider(
                "Total Rooms",
                min_value=1,
                max_value=5,
                value=4,
                step=1,
                help="Total number of rooms in the house",
            )

        with col2:
            bedrooms = st.slider(
                "Bedrooms",
                min_value=1,
                max_value=min(rooms, 5),
                value=min(2, rooms),
                step=1,
                help="Number of bedrooms",
            )

        with col3:
            household_size = st.slider(
                "People in Household",
                min_value=1,
                max_value=8,
                value=3,
                step=1,
                help="Number of people living in the house",
            )

        # Build the full feature dict for the model
        self.input_data = {
            'latitude': city_data['latitude'],
            'longitude': city_data['longitude'],
            'median_income': city_data['median_income'],
            'population': city_data['population'],
            'housing_median_age': city_data['housing_median_age'],
            'ave_rooms': float(rooms),
            'ave_bedrooms': float(bedrooms),
            'ave_occupancy': float(household_size),
        }

        return self.input_data

    def render_input_summary(self) -> None:
        """Render a compact summary of what the model will use."""
        if not self.input_data:
            return

        st.subheader("Your Inputs")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Rooms:** {int(self.input_data['ave_rooms'])}")
            st.write(f"**Bedrooms:** {int(self.input_data['ave_bedrooms'])}")
            st.write(f"**Household Size:** {int(self.input_data['ave_occupancy'])}")

        with col2:
            st.write(f"**Area Median Income:** ${self.input_data['median_income'] * 10:,.0f}K")
            st.write(f"**Typical Housing Age:** {self.input_data['housing_median_age']} years")
            st.write(f"**Area Population:** {self.input_data['population']:,}")

    def get_input_data(self) -> Dict[str, Any]:
        """Get the current input data."""
        return self.input_data


# Keep for backward compat — but now just returns city list
class CaliforniaPresets:
    """No longer needed — city selection is built into the main form."""

    @staticmethod
    def get_presets() -> Dict[str, Dict[str, Any]]:
        return CALIFORNIA_CITIES

    def render_preset_selector(self) -> Dict[str, Any]:
        # Presets are now integrated into the main form
        return {}
