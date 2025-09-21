"""
Main Streamlit application for California Housing Price Prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configure page
from app.config import APP_CONFIG, UI_THEME
st.set_page_config(**APP_CONFIG)

# Import components
from app.utils.model_loader import CaliforniaHousingModelLoader
from app.components.input_forms import CaliforniaHousingInputForm, CaliforniaPresets


def main():
    """Main application function."""

    # Custom CSS for California Housing theme
    st.markdown(f"""
    <style>
    .main-header {{
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD23F 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}

    .main-header h1 {{
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 2.5rem;
    }}

    .main-header p {{
        color: white;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }}

    .prediction-box {{
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid {UI_THEME['primary_color']};
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}

    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }}

    .info-box {{
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }}

    .stAlert > div {{
        padding: 1rem;
        border-radius: 8px;
    }}

    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🏠 California Housing Price Predictor</h1>
        <p>AI-Powered Property Valuation for the Golden State</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = CaliforniaHousingModelLoader()
        st.session_state.models_loaded = False

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Price Prediction", "📊 Model Insights", "🗺️ California Explorer", "ℹ️ About"],
        help="Navigate between different sections of the app"
    )

    # Load models if not already loaded
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading California housing models..."):
            try:
                models = st.session_state.model_loader.load_models()
                if models:
                    st.session_state.models_loaded = True

                    # Show model loading success in sidebar
                    deployment_info = st.session_state.model_loader.get_deployment_summary()
                    if deployment_info:
                        st.sidebar.success(f"✅ {len(models)} models loaded")
                        if 'champion_model' in deployment_info:
                            champion = deployment_info['champion_model'].replace('_california_housing_model.pkl', '')
                            st.sidebar.info(f"🏆 Champion: {champion}")
                    else:
                        st.sidebar.success(f"✅ {len(models)} models loaded")
                else:
                    st.sidebar.warning("⚠️ No models found")
                    st.sidebar.info("💡 Complete Phase 4 (Model Training) first")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading models: {str(e)}")

    # Page routing
    if page == "🏠 Price Prediction":
        render_prediction_page()
    elif page == "📊 Model Insights":
        render_insights_page()
    elif page == "🗺️ California Explorer":
        render_california_explorer_page()
    elif page == "ℹ️ About":
        render_about_page()

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            Built with ❤️ using Streamlit & scikit-learn • California Housing Price Predictor v1.0<br>
            <small>Trained on {st.session_state.model_loader.deployment_info.get('training_samples', '20K+')} California properties</small>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_prediction_page():
    """Render the main prediction page."""
    st.title("🏠 California Housing Price Prediction")
    st.markdown("Get instant property valuations using advanced machine learning models trained on California housing data.")

    # Check if models are available
    available_models = st.session_state.model_loader.get_available_models()
    if not available_models:
        st.error("❌ No models available for prediction")
        st.info("💡 Please ensure Phase 4 (Model Training) has been completed and models are saved.")
        return

    # Model selection in sidebar
    st.sidebar.subheader("🤖 Model Selection")

    # Try to get champion model first
    champion_name, champion_model = st.session_state.model_loader.get_champion_model()
    default_model = champion_name if champion_name else available_models[0]

    selected_model = st.sidebar.selectbox(
        "Choose prediction model:",
        options=available_models,
        index=available_models.index(default_model) if default_model in available_models else 0,
        help="Select the machine learning model for prediction"
    )

    # Show model performance if available
    model_info = st.session_state.model_loader.get_model_info(selected_model)
    if model_info:
        with st.sidebar.expander("📈 Model Performance"):
            if 'validation_metrics' in model_info:
                metrics = model_info['validation_metrics']
                if 'rmse' in metrics:
                    st.metric("RMSE", f"${metrics['rmse']:,.0f}")
                if 'r2' in metrics:
                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                if 'mae' in metrics:
                    st.metric("MAE", f"${metrics['mae']:,.0f}")

            if model_info.get('is_champion'):
                st.success("🏆 Champion Model")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # California area presets
        presets = CaliforniaPresets()
        preset_data = presets.render_preset_selector()

        # Input forms
        input_form = CaliforniaHousingInputForm()

        if preset_data:
            # Use preset data
            input_data = preset_data
            st.info("🏠 Using preset configuration. Modify values below if needed.")

            # Show editable form with preset values
            input_data = render_editable_preset_form(preset_data, input_form)
        else:
            # Manual input
            input_data = input_form.render_complete_form()

        # Get final input data
        final_input_data = input_form.get_input_data() or input_data

    with col2:
        # Prediction panel
        render_prediction_panel(selected_model, final_input_data)

    # Input summary
    if final_input_data:
        input_form.input_data = final_input_data
        input_form.render_input_summary()

    # California region information
    if final_input_data:
        render_california_context(final_input_data)


def render_editable_preset_form(preset_data: Dict[str, Any], input_form: CaliforniaHousingInputForm) -> Dict[str, Any]:
    """Render form with preset values that can be edited."""
    st.subheader("🔧 Adjust California Area Features")

    with st.form("california_preset_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**📍 Location**")
            longitude = st.number_input(
                "Longitude",
                min_value=-124.5, max_value=-114.0,
                value=preset_data.get('longitude', -118.0),
                step=0.1
            )
            latitude = st.number_input(
                "Latitude",
                min_value=32.5, max_value=42.0,
                value=preset_data.get('latitude', 37.0),
                step=0.1
            )
            ocean_proximity = st.selectbox(
                "Ocean Proximity",
                options=['NEAR BAY', 'NEAR OCEAN', 'INLAND', '<1H OCEAN', 'ISLAND'],
                index=['NEAR BAY', 'NEAR OCEAN', 'INLAND', '<1H OCEAN', 'ISLAND'].index(preset_data.get('ocean_proximity', 'INLAND'))
            )

        with col2:
            st.write("**🏠 Housing**")
            total_rooms = st.number_input(
                "Total Rooms",
                min_value=500, max_value=8000,
                value=preset_data.get('total_rooms', 2500),
                step=50
            )
            total_bedrooms = st.number_input(
                "Total Bedrooms",
                min_value=100, max_value=1500,
                value=preset_data.get('total_bedrooms', 500),
                step=10
            )
            housing_median_age = st.number_input(
                "Housing Age (years)",
                min_value=1, max_value=52,
                value=preset_data.get('housing_median_age', 25),
                step=1
            )

        with col3:
            st.write("**👥 Demographics**")
            population = st.number_input(
                "Population",
                min_value=300, max_value=5000,
                value=preset_data.get('population', 2000),
                step=50
            )
            households = st.number_input(
                "Households",
                min_value=100, max_value=1800,
                value=preset_data.get('households', 800),
                step=10
            )
            median_income = st.number_input(
                "Median Income ($10K)",
                min_value=0.5, max_value=15.0,
                value=preset_data.get('median_income', 5.0),
                step=0.1
            )

        submitted = st.form_submit_button("🔄 Update Prediction", use_container_width=True)

        if submitted:
            updated_data = {
                'longitude': longitude,
                'latitude': latitude,
                'ocean_proximity': ocean_proximity,
                'total_rooms': total_rooms,
                'total_bedrooms': min(total_bedrooms, total_rooms),
                'housing_median_age': housing_median_age,
                'population': population,
                'households': min(households, population),
                'median_income': median_income
            }
            input_form.input_data = updated_data
            return updated_data

    return preset_data


def render_prediction_panel(model_name: str, input_data: Dict[str, Any]):
    """Render the prediction results panel."""
    st.subheader("💰 Property Value Prediction")

    if not input_data:
        st.info("👈 Please enter property details to get a prediction")
        return

    # Validate input
    is_valid, errors = st.session_state.model_loader.validate_input(input_data)

    if not is_valid:
        st.error("❌ Input validation failed:")
        for error in errors:
            st.write(f"• {error}")
        return

    try:
        # Make prediction
        with st.spinner("🔮 Calculating property value..."):
            prediction, additional_info = st.session_state.model_loader.predict(model_name, input_data)

        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: #FF6B35; margin: 0; font-size: 2.5rem;">
                ${prediction:,.0f}
            </h2>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 1.1rem;">
                Predicted California Property Value
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Additional prediction info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Model Used",
                model_name.title(),
                help="Machine learning model used for prediction"
            )

        with col2:
            if additional_info and 'income_to_price_ratio' in additional_info:
                ratio = additional_info['income_to_price_ratio']
                st.metric(
                    "Price-to-Income Ratio",
                    f"{ratio:.1f}x",
                    help="Property value relative to area median income"
                )

        with col3:
            if additional_info and 'prediction_per_sqft' in additional_info:
                per_sqft = additional_info['prediction_per_sqft']
                st.metric(
                    "Est. Price per Room",
                    f"${per_sqft:,.0f}",
                    help="Rough estimate of value per room"
                )

        # Confidence interval if available
        if additional_info and 'confidence_interval' in additional_info:
            ci = additional_info['confidence_interval']

            st.subheader("📊 Prediction Confidence")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Lower Bound", f"${ci['lower']:,.0f}")
            with col_b:
                st.metric("Upper Bound", f"${ci['upper']:,.0f}")

            # Confidence visualization
            fig = go.Figure()

            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=[ci['lower'], prediction, ci['upper']],
                y=[1, 1, 1],
                mode='markers+lines',
                name='95% Confidence Interval',
                line=dict(color='rgba(255, 107, 53, 0.3)', width=10),
                marker=dict(
                    size=[15, 25, 15],
                    color=['rgba(255, 107, 53, 0.6)', '#FF6B35', 'rgba(255, 107, 53, 0.6)']
                ),
                showlegend=False
            ))

            fig.update_layout(
                title="Prediction Confidence Range",
                xaxis_title="Predicted Value ($)",
                yaxis=dict(showticklabels=False, showgrid=False, range=[0.5, 1.5]),
                height=200,
                margin=dict(t=40, b=40, l=40, r=40)
            )

            st.plotly_chart(fig, use_container_width=True)

        # Property context
        context = st.session_state.model_loader.estimate_property_context(input_data)
        if context:
            st.subheader("🏠 Property Context")

            context_col1, context_col2 = st.columns(2)

            with context_col1:
                st.write(f"**📍 Location:** {context.get('region', 'Unknown')}")
                st.write(f"**💰 Income Level:** {context.get('income_level', 'Unknown')}")
                st.write(f"**🏗️ Housing Character:** {context.get('housing_character', 'Unknown')}")

            with context_col2:
                st.write(f"**🏘️ Area Density:** {context.get('density', 'Unknown')}")
                st.write(f"**🌊 Location Appeal:** {context.get('location_appeal', 'Unknown')}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("💡 Please check your input values and try again.")

        # Debug info for development
        if st.checkbox("Show debug info"):
            st.write("Debug information:")
            st.write(f"Model: {model_name}")
            st.write(f"Input data: {input_data}")
            st.write(f"Error: {e}")


def render_insights_page():
    """Render the model insights page."""
    st.title("📊 Model Insights & Performance")
    st.markdown("Explore model performance, feature importance, and prediction patterns for California housing.")

    available_models = st.session_state.model_loader.get_available_models()
    if not available_models:
        st.error("❌ No models available for analysis.")
        return

    # Model selection for analysis
    selected_model = st.selectbox(
        "Select model for detailed analysis:",
        options=available_models,
        help="Choose which model to analyze in detail"
    )

    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["🎯 Performance", "🔍 Feature Importance", "📈 California Insights"])

    with tab1:
        render_performance_analysis(selected_model)

    with tab2:
        render_feature_importance_analysis(selected_model)

    with tab3:
        render_california_specific_insights(selected_model)


def render_performance_analysis(model_name: str):
    """Render model performance analysis."""
    st.subheader("🎯 Model Performance Analysis")

    model_info = st.session_state.model_loader.get_model_info(model_name)

    if not model_info:
        st.warning("No performance data available for this model.")
        return

    # Performance metrics
    if 'validation_metrics' in model_info:
        metrics = model_info['validation_metrics']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'rmse' in metrics:
                st.metric(
                    "RMSE",
                    f"${metrics['rmse']:,.0f}",
                    help="Root Mean Square Error - lower is better"
                )

        with col2:
            if 'r2' in metrics:
                st.metric(
                    "R² Score",
                    f"{metrics['r2']:.3f}",
                    help="Coefficient of determination - higher is better (max 1.0)"
                )

        with col3:
            if 'mae' in metrics:
                st.metric(
                    "MAE",
                    f"${metrics['mae']:,.0f}",
                    help="Mean Absolute Error"
                )

        with col4:
            if 'mape' in metrics:
                st.metric(
                    "MAPE",
                    f"{metrics['mape']:.1f}%",
                    help="Mean Absolute Percentage Error"
                )

    # Performance interpretation
    if 'validation_metrics' in model_info:
        rmse = model_info['validation_metrics'].get('rmse', 0)
        r2 = model_info['validation_metrics'].get('r2', 0)

        st.subheader("📈 Performance Interpretation")

        # RMSE interpretation
        mean_house_value = 400000  # Approximate CA average
        rmse_pct = (rmse / mean_house_value) * 100

        if rmse_pct < 5:
            performance_level = "🌟 Excellent"
        elif rmse_pct < 10:
            performance_level = "✅ Very Good"
        elif rmse_pct < 15:
            performance_level = "👍 Good"
        elif rmse_pct < 25:
            performance_level = "⚠️ Fair"
        else:
            performance_level = "❌ Needs Improvement"

        st.write(f"**Model Performance:** {performance_level}")
        st.write(f"**Typical Prediction Error:** ±${rmse:,.0f} ({rmse_pct:.1f}% of average CA home value)")
        st.write(f"**Accuracy:** The model explains {r2*100:.1f}% of the variation in California housing prices")


def render_feature_importance_analysis(model_name: str):
    """Render feature importance analysis."""
    st.subheader("🔍 Feature Importance Analysis")

    importance_df = st.session_state.model_loader.get_feature_importance(model_name, top_n=25)

    if importance_df is None:
        st.warning("Feature importance not available for this model type.")
        return

    # Interactive feature importance plot
    fig = px.bar(
        importance_df.head(15),
        x='importance',
        y='feature',
        orientation='h',
        title=f"Top 15 Feature Importance - {model_name}",
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature categories analysis
    st.subheader("📊 Feature Categories")

    # Categorize features
    categories = {
        'Original Features': [],
        'Geographic Features': [],
        'Density Features': [],
        'Income Features': [],
        'Engineered Features': []
    }

    for _, row in importance_df.iterrows():
        feature = row['feature']
        if any(x in feature for x in ['distance_', 'northern_', 'coastal']):
            categories['Geographic Features'].append((feature, row['importance']))
        elif any(x in feature for x in ['per_household', 'per_room', 'density']):
            categories['Density Features'].append((feature, row['importance']))
        elif any(x in feature for x in ['income', 'Income']):
            categories['Income Features'].append((feature, row['importance']))
        elif any(x in feature for x in ['squared', 'interaction', 'category_']):
            categories['Engineered Features'].append((feature, row['importance']))
        else:
            categories['Original Features'].append((feature, row['importance']))

    # Display feature categories
    for category, features in categories.items():
        if features:
            with st.expander(f"{category} ({len(features)} features)"):
                for feature, importance in sorted(features, key=lambda x: x[1], reverse=True):
                    st.write(f"• **{feature}**: {importance:.4f}")


def render_california_specific_insights(model_name: str):
    """Render California-specific housing insights."""
    st.subheader("🏠 California Housing Market Insights")

    # California housing market context
    st.write("""
    ### 🌟 Key Insights from California Housing Model

    Based on the trained model, here are the key factors that drive housing prices in California:
    """)

    importance_df = st.session_state.model_loader.get_feature_importance(model_name, top_n=10)

    if importance_df is not None:
        st.write("**🔝 Top Price Drivers:**")

        insights = {
            'median_income': "💰 **Income** is the strongest predictor - affluent areas command premium prices",
            'ocean_proximity': "🌊 **Ocean proximity** significantly impacts value - coastal properties are premium",
            'total_rooms': "🏠 **Property size** (total rooms) directly correlates with value",
            'latitude': "📍 **North-South location** affects prices (SF Bay Area vs Central Valley)",
            'longitude': "📍 **East-West location** impacts value (coastal vs inland)",
            'housing_median_age': "📅 **Property age** - newer developments often command higher prices",
            'population': "👥 **Population density** affects demand and pricing",
            'distance_to_Los_Angeles': "🏙️ **Distance from LA** - proximity to major cities increases value",
            'distance_to_San_Francisco': "🏙️ **Distance from SF** - Bay Area proximity is major price factor"
        }

        for _, row in importance_df.head(8).iterrows():
            feature = row['feature']
            for key, insight in insights.items():
                if key in feature.lower():
                    st.markdown(f"• {insight}")
                    break

    # California market overview
    st.write("""
    ### 📊 California Housing Market Overview

    **🌎 Regional Patterns:**
    - **Bay Area (SF)**: Tech industry drives highest prices ($800K-$2M+)
    - **Los Angeles Metro**: Entertainment/aerospace industries ($600K-$1.5M)
    - **San Diego**: Military/biotech with ocean premium ($700K-$1.2M)
    - **Central Valley**: Agricultural areas with affordable housing ($300K-$600K)

    **🎯 Price Factors:**
    1. **Income** - Areas with higher median income have significantly higher home values
    2. **Location** - Coastal proximity and major city access drive premiums
    3. **Density** - Rooms per household indicates property size and desirability
    4. **Age** - Newer construction often commands premium prices
    5. **Demographics** - Population and household characteristics affect demand
    """)


def render_california_explorer_page():
    """Render California housing data explorer."""
    st.title("🗺️ California Housing Market Explorer")
    st.markdown("Explore California housing market trends and regional patterns.")

    # Sample data visualization (since we don't have the full dataset in the app)
    st.subheader("📊 California Housing Market Overview")

    # Create sample data for visualization
    np.random.seed(42)
    n_samples = 1000

    # Generate sample California data
    sample_data = pd.DataFrame({
        'longitude': np.random.uniform(-124, -114, n_samples),
        'latitude': np.random.uniform(32.5, 42, n_samples),
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'median_house_value': np.random.uniform(100000, 500000, n_samples),
        'ocean_proximity': np.random.choice(['NEAR BAY', 'NEAR OCEAN', 'INLAND', '<1H OCEAN'], n_samples)
    })

    # Add regional patterns
    sample_data.loc[sample_data['latitude'] > 37, 'median_house_value'] *= 1.3  # Northern CA premium
    sample_data.loc[sample_data['ocean_proximity'].isin(['NEAR BAY', 'NEAR OCEAN']), 'median_house_value'] *= 1.2

    # Interactive map
    st.subheader("🗺️ California Housing Value Map")

    fig = px.scatter_mapbox(
        sample_data,
        lat="latitude",
        lon="longitude",
        color="median_house_value",
        size="median_income",
        hover_data=["ocean_proximity"],
        color_continuous_scale="Viridis",
        mapbox_style="open-street-map",
        zoom=5,
        center={"lat": 37, "lon": -119},
        title="California Housing Values by Location"
    )

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Regional analysis
    col1, col2 = st.columns(2)

    with col1:
        # Price by region
        sample_data['region'] = sample_data['latitude'].apply(
            lambda x: 'Northern CA' if x > 37 else 'Central CA' if x > 35 else 'Southern CA'
        )

        region_stats = sample_data.groupby('region')['median_house_value'].agg(['mean', 'median', 'count']).reset_index()

        fig = px.bar(
            region_stats,
            x='region',
            y='mean',
            title="Average House Value by California Region",
            labels={'mean': 'Average Value ($)', 'region': 'Region'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Income vs house value
        fig = px.scatter(
            sample_data,
            x='median_income',
            y='median_house_value',
            color='ocean_proximity',
            title="Income vs House Value Relationship",
            labels={'median_income': 'Median Income ($10K)', 'median_house_value': 'House Value ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_about_page():
    """Render the about page."""
    st.title("ℹ️ About California Housing Price Predictor")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🏠 California Housing Price Prediction System

        This application provides intelligent California property price predictions using advanced
        machine learning models trained on comprehensive California housing data.

        ### ✨ Key Features

        - **🎯 Accurate Predictions**: Multiple ML models trained on California housing data
        - **🗺️ California-Specific**: Optimized for Golden State property characteristics
        - **📊 Rich Features**: 39 engineered features including geographic and demographic factors
        - **🏆 Champion Model**: Best-performing model automatically selected
        - **🔍 Deep Insights**: Feature importance and market analysis

        ### 🔧 Technology Stack

        - **Frontend**: Streamlit for interactive web interface
        - **Machine Learning**: scikit-learn, XGBoost for predictive models
        - **Data Processing**: pandas, NumPy for data manipulation
        - **Visualization**: Plotly for interactive charts and maps
        - **Deployment**: Streamlit Cloud ready

        ### 📊 Model Performance

        Our California housing models achieve excellent performance:
        """)

        # Display model performance if available
        if st.session_state.models_loaded:
            deployment_info = st.session_state.model_loader.get_deployment_summary()
            if deployment_info:
                st.write("**🏆 Champion Model Performance:**")
                if 'champion_r2' in deployment_info:
                    st.write(f"- **R² Score**: {deployment_info['champion_r2']:.3f}")
                if 'champion_rmse' in deployment_info:
                    st.write(f"- **RMSE**: ${deployment_info['champion_rmse']:,.0f}")
                if 'training_samples' in deployment_info:
                    st.write(f"- **Training Data**: {deployment_info['training_samples']:,} California properties")

        st.markdown("""
        ### 🎯 Use Cases

        - **🏡 Home Buyers**: Estimate fair market value for California properties
        - **🏢 Real Estate Professionals**: Quick property valuations
        - **💼 Investors**: Investment opportunity analysis
        - **🏠 Homeowners**: Property value assessment
        - **📊 Market Research**: California housing market analysis

        ### 📈 California Housing Features

        The model analyzes these key factors:
        - **Geographic**: Longitude, latitude, ocean proximity
        - **Property**: Rooms, bedrooms, age
        - **Economic**: Area median income levels
        - **Demographic**: Population, households
        - **Engineered**: Distance to major cities, density ratios, income categories
        """)

    with col2:
        st.markdown("""
        ### 📞 How to Use

        1. **🚀 Quick Start**: Choose a California area preset
        2. **📝 Custom Input**: Enter specific property details
        3. **🤖 Model Selection**: Pick your preferred ML model
        4. **💰 Get Prediction**: Instant price estimate with confidence
        5. **📊 Explore**: Analyze feature importance and insights

        ### 🔍 California Regions

        **Northern California**
        - Bay Area tech hub
        - Higher income, premium prices
        - Average: $600K - $1M+

        **Central California**
        - Agricultural Central Valley
        - Moderate prices
        - Average: $300K - $600K

        **Southern California**
        - LA/OC/SD metro areas
        - Entertainment/aerospace/biotech
        - Average: $500K - $1.2M

        ### 📊 Quick Stats
        """)

        # Quick stats
        if st.session_state.models_loaded:
            deployment_info = st.session_state.model_loader.get_deployment_summary()
            available_models = st.session_state.model_loader.get_available_models()

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Models Available", len(available_models))
                if deployment_info.get('feature_count'):
                    st.metric("Features Used", deployment_info['feature_count'])

            with col_b:
                if deployment_info.get('training_samples'):
                    st.metric("Training Properties", f"{deployment_info['training_samples']:,}")

        st.markdown("""
        ### ⚠️ Important Notes

        - Predictions are estimates based on historical data
        - Actual prices may vary due to market conditions
        - Consider local factors and recent sales
        - Use predictions as guidance, not definitive valuations
        """)


def render_california_context(input_data: Dict[str, Any]):
    """Render California-specific context information."""
    st.subheader("🌎 California Market Context")

    latitude = input_data.get('latitude', 37.0)
    longitude = input_data.get('longitude', -118.0)
    income = input_data.get('median_income', 5.0)
    proximity = input_data.get('ocean_proximity', 'INLAND')

    col1, col2, col3 = st.columns(3)

    with col1:
        # Regional context
        if latitude >= 37.0:
            region_info = {
                'name': 'Northern California',
                'characteristics': 'Tech hub, high incomes, premium prices',
                'typical_range': '$600K - $1M+'
            }
        elif latitude >= 35.0:
            region_info = {
                'name': 'Central California',
                'characteristics': 'Agricultural, moderate prices',
                'typical_range': '$300K - $600K'
            }
        else:
            region_info = {
                'name': 'Southern California',
                'characteristics': 'Urban metros, entertainment industry',
                'typical_range': '$500K - $1.2M'
            }

        st.info(f"""
        **📍 {region_info['name']}**

        {region_info['characteristics']}

        Typical Range: {region_info['typical_range']}
        """)

    with col2:
        # Income context
        income_annual = income * 10000
        if income < 3:
            income_context = "Below median CA income"
        elif income < 6:
            income_context = "Around CA median income"
        elif income < 10:
            income_context = "Above median CA income"
        else:
            income_context = "High income area"

        st.info(f"""
        **💰 Income Level**

        ${income_annual:,.0f} annually

        {income_context}
        """)

    with col3:
        # Location premium
        proximity_premiums = {
            'NEAR BAY': '+30-50% premium',
            'NEAR OCEAN': '+20-40% premium',
            '<1H OCEAN': '+10-20% premium',
            'INLAND': 'Baseline pricing',
            'ISLAND': '+50%+ premium'
        }

        st.info(f"""
        **🌊 Location Premium**

        {proximity}

        {proximity_premiums.get(proximity, 'Standard pricing')}
        """)


if __name__ == "__main__":
    main()