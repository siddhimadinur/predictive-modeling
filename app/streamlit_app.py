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
from app.components.input_forms import CaliforniaHousingInputForm


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 2rem;
}}

/* ---------- Hero header ---------- */
.hero {{
    background: {UI_THEME['primary_gradient']};
    padding: 2.5rem 2rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(232, 93, 38, 0.18);
}}
.hero::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.15) 0%, transparent 60%);
    pointer-events: none;
}}
.hero h1 {{
    color: #fff;
    text-align: center;
    margin: 0;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}
.hero p {{
    color: rgba(255,255,255,0.92);
    text-align: center;
    margin: 0.4rem 0 0;
    font-size: 1.15rem;
    font-weight: 400;
}}

/* ---------- Stats bar ---------- */
.stats-bar {{
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}}
.stat-pill {{
    background: rgba(255,255,255,0.22);
    backdrop-filter: blur(8px);
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    color: #fff;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}}

/* ---------- Glass cards ---------- */
.glass-card {{
    background: {UI_THEME['card_background']};
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: {UI_THEME['border_radius']};
    padding: 1.5rem;
    box-shadow: {UI_THEME['shadow']};
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem;
}}
.glass-card:hover {{
    transform: translateY(-2px);
    box-shadow: {UI_THEME['shadow_hover']};
}}

/* ---------- Prediction result ---------- */
.prediction-result {{
    background: linear-gradient(135deg, #f8faff 0%, #eef4ff 100%);
    border: 1px solid #d0e0ff;
    border-left: 6px solid {UI_THEME['primary_color']};
    border-radius: {UI_THEME['border_radius']};
    padding: 2rem;
    text-align: center;
    box-shadow: {UI_THEME['shadow']};
    margin: 1rem 0;
}}
.prediction-result .price {{
    font-size: 3rem;
    font-weight: 800;
    background: {UI_THEME['primary_gradient']};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}}
.prediction-result .label {{
    color: {UI_THEME['text_secondary']};
    font-size: 1rem;
    font-weight: 500;
    margin-top: 0.3rem;
}}

/* ---------- Metric tiles ---------- */
.metric-tile {{
    background: #fff;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid #E8ECF1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: transform 0.15s ease;
}}
.metric-tile:hover {{
    transform: translateY(-2px);
}}
.metric-tile .value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {UI_THEME['text_color']};
}}
.metric-tile .title {{
    font-size: 0.82rem;
    color: {UI_THEME['text_secondary']};
    font-weight: 500;
    margin-top: 0.2rem;
}}

/* ---------- Info accent box ---------- */
.accent-box {{
    background: #F0F7FF;
    border-left: 4px solid {UI_THEME['accent_color']};
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.95rem;
    color: #334155;
}}
.accent-box.success {{
    background: #F0FFF4;
    border-left-color: {UI_THEME['success_color']};
}}
.accent-box.warning {{
    background: #FFFDF0;
    border-left-color: #E6A817;
}}

/* ---------- Section headers ---------- */
.section-header {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.5rem 0 0.8rem;
}}
.section-header h3 {{
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: {UI_THEME['text_color']};
}}
.section-header .icon {{
    font-size: 1.3rem;
}}

/* ---------- Feature card grid (About) ---------- */
.feature-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}}
.feature-card {{
    background: #fff;
    border-radius: 14px;
    padding: 1.4rem;
    border: 1px solid #E8ECF1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.feature-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}}
.feature-card .icon {{
    font-size: 2rem;
    margin-bottom: 0.5rem;
}}
.feature-card h4 {{
    margin: 0 0 0.3rem;
    font-size: 1rem;
    font-weight: 600;
    color: {UI_THEME['text_color']};
}}
.feature-card p {{
    margin: 0;
    font-size: 0.88rem;
    color: {UI_THEME['text_secondary']};
    line-height: 1.45;
}}

/* ---------- Footer ---------- */
.app-footer {{
    text-align: center;
    color: {UI_THEME['text_secondary']};
    padding: 1.5rem 0;
    font-size: 0.85rem;
    border-top: 1px solid #E8ECF1;
    margin-top: 2rem;
}}
.app-footer a {{
    color: {UI_THEME['accent_color']};
    text-decoration: none;
    font-weight: 500;
}}

/* ---------- Sidebar polish ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FAFBFD 0%, #F0F4F8 100%);
}}
section[data-testid="stSidebar"] .stRadio > label {{
    font-weight: 600;
}}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0;
    font-weight: 600;
}}

/* ---------- Expander ---------- */
.streamlit-expanderHeader {{
    font-weight: 600;
    font-size: 0.95rem;
}}
</style>
"""

# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", color=UI_THEME['text_color']),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFBFD",
        colorway=[
            "#E85D26", "#1B6B93", "#2E7D32", "#EDDE5D",
            "#7C3AED", "#DB2777", "#0EA5E9", "#F97316",
        ],
        margin=dict(t=48, b=40, l=48, r=24),
        title=dict(font=dict(size=16, color=UI_THEME['text_color'])),
        xaxis=dict(gridcolor="#E8ECF1"),
        yaxis=dict(gridcolor="#E8ECF1"),
    )
)


def styled_plotly(fig, height=400):
    """Apply consistent styling to a plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Main application function."""

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = CaliforniaHousingModelLoader()
        st.session_state.models_loaded = False

    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading California housing models..."):
            try:
                models = st.session_state.model_loader.load_models()
                if models:
                    st.session_state.models_loaded = True
            except Exception:
                pass

    # ---- Hero header ----
    deployment_info = st.session_state.model_loader.get_deployment_summary() or {}
    n_models = len(st.session_state.model_loader.get_available_models())
    training_samples = deployment_info.get('training_samples', '20K+')
    champion_r2 = deployment_info.get('champion_r2')

    stats_html = '<div class="stats-bar">'
    stats_html += f'<span class="stat-pill">{n_models} Models</span>'
    stats_html += f'<span class="stat-pill">{training_samples:,} Properties</span>' if isinstance(training_samples, int) else f'<span class="stat-pill">{training_samples} Properties</span>'
    if champion_r2:
        stats_html += f'<span class="stat-pill">R\u00b2 {champion_r2:.3f}</span>'
    stats_html += '</div>'

    st.markdown(f"""
    <div class="hero">
        <h1>California Housing Price Predictor</h1>
        <p>AI-powered property valuation for the Golden State</p>
        {stats_html}
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar navigation ----
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Price Prediction", "Model Insights", "California Explorer", "About"],
    )

    # Sidebar model status
    if st.session_state.models_loaded:
        st.sidebar.markdown("---")
        st.sidebar.success(f"{n_models} models loaded")
        if 'champion_model' in deployment_info:
            champion = deployment_info['champion_model'].replace('_california_housing_model.pkl', '')
            st.sidebar.info(f"Champion: **{champion}**")
    else:
        st.sidebar.warning("No models found")
        st.sidebar.caption("Complete Phase 4 (Model Training) first")

    # ---- Page routing ----
    if page == "Price Prediction":
        render_prediction_page()
    elif page == "Model Insights":
        render_insights_page()
    elif page == "California Explorer":
        render_california_explorer_page()
    elif page == "About":
        render_about_page()

    # ---- Footer ----
    st.markdown(f"""
    <div class="app-footer">
        Built with Streamlit &amp; scikit-learn &middot; California Housing Price Predictor v1.0<br>
        <small>Trained on {training_samples if isinstance(training_samples, str) else f'{training_samples:,}'} California properties</small>
    </div>
    """, unsafe_allow_html=True)


# ===================================================================
# PREDICTION PAGE
# ===================================================================

def render_prediction_page():
    """Render the main prediction page."""

    st.markdown('<div class="section-header"><span class="icon">&#127968;</span><h3>Property Price Prediction</h3></div>', unsafe_allow_html=True)
    st.caption("Pick a city and describe your property to get an instant price estimate.")

    available_models = st.session_state.model_loader.get_available_models()
    if not available_models:
        st.error("No models available for prediction. Ensure Phase 4 has been completed.")
        return

    # ---- Sidebar model selection ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Model Selection")

    champion_name, _ = st.session_state.model_loader.get_champion_model()
    default_model = champion_name if champion_name else available_models[0]

    selected_model = st.sidebar.selectbox(
        "Choose prediction model:",
        options=available_models,
        index=available_models.index(default_model) if default_model in available_models else 0,
    )

    model_info = st.session_state.model_loader.get_model_info(selected_model)
    if model_info and 'validation_metrics' in model_info:
        metrics = model_info['validation_metrics']
        with st.sidebar.expander("Model Performance", expanded=False):
            if 'rmse' in metrics:
                st.metric("RMSE", f"${metrics['rmse']:,.0f}")
            if 'r2' in metrics:
                st.metric("R\u00b2 Score", f"{metrics['r2']:.3f}")
            if model_info.get('is_champion'):
                st.markdown("**Champion Model**")

    # ---- Simplified input: city + 3 sliders ----
    input_form = CaliforniaHousingInputForm()
    input_data = input_form.render_complete_form()

    # ---- Prediction result ----
    st.markdown("---")
    render_prediction_panel(selected_model, input_data)

    # Input summary
    if input_data:
        input_form.render_input_summary()
        render_california_context(input_data)


def render_prediction_panel(model_name: str, input_data: Dict[str, Any]):
    """Render the prediction results panel."""
    st.markdown('<div class="section-header"><span class="icon">&#128176;</span><h3>Prediction</h3></div>', unsafe_allow_html=True)

    if not input_data:
        st.info("Enter property details to get a prediction.")
        return

    is_valid, errors = st.session_state.model_loader.validate_input(input_data)
    if not is_valid:
        st.error("Input validation failed:")
        for error in errors:
            st.write(f"- {error}")
        return

    try:
        with st.spinner("Calculating property value..."):
            prediction, additional_info = st.session_state.model_loader.predict(model_name, input_data)

        # --- Inflation-adjusted estimate ---
        # Dataset is from 1990 Census; CA home prices are ~4.5x higher today
        INFLATION_MULTIPLIER = 4.5
        adjusted_prediction = prediction * INFLATION_MULTIPLIER

        # --- Large price display (both values) ---
        price_cols = st.columns(2)
        with price_cols[0]:
            st.markdown(f"""
            <div class="prediction-result">
                <div class="price">${adjusted_prediction:,.0f}</div>
                <div class="label">Estimated 2024 Value</div>
            </div>
            """, unsafe_allow_html=True)
        with price_cols[1]:
            st.markdown(f"""
            <div class="prediction-result" style="border-left-color: {UI_THEME['accent_color']};">
                <div class="price" style="background: linear-gradient(135deg, #1B6B93, #0EA5E9); -webkit-background-clip: text; background-clip: text;">${prediction:,.0f}</div>
                <div class="label">1990 Census Value (raw model output)</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("The model is trained on 1990 Census data. The 2024 estimate applies a 4.5x inflation adjustment for California housing.")

        # --- Gauge chart (adjusted value) ---
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=adjusted_prediction,
            number=dict(prefix="$", valueformat=",.0f", font=dict(size=20)),
            gauge=dict(
                axis=dict(range=[100000, 2500000], tickprefix="$", tickformat=",.0f"),
                bar=dict(color=UI_THEME['primary_color']),
                bgcolor="#F0F4F8",
                steps=[
                    dict(range=[100000, 800000], color="#E8ECF1"),
                    dict(range=[800000, 1500000], color="#D4E4F7"),
                    dict(range=[1500000, 2500000], color="#BDD5F0"),
                ],
                threshold=dict(line=dict(color="#1B6B93", width=3), thickness=0.8, value=adjusted_prediction),
            ),
        ))
        fig_gauge.update_layout(height=200, margin=dict(t=20, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- Key metrics ---
        m_cols = st.columns(2)
        with m_cols[0]:
            if additional_info and 'income_to_price_ratio' in additional_info:
                st.metric("Price-to-Income", f"{additional_info['income_to_price_ratio']:.1f}x")
        with m_cols[1]:
            if additional_info and 'price_per_room' in additional_info:
                st.metric("Est. $/Room", f"${additional_info['price_per_room']:,.0f}")

        # --- Confidence interval ---
        if additional_info and 'confidence_interval' in additional_info:
            ci = additional_info['confidence_interval']
            st.markdown("##### 95% Confidence Interval")

            ci_cols = st.columns(2)
            with ci_cols[0]:
                st.metric("Lower", f"${ci['lower']:,.0f}")
            with ci_cols[1]:
                st.metric("Upper", f"${ci['upper']:,.0f}")

            fig_ci = go.Figure()
            fig_ci.add_trace(go.Bar(
                x=[ci['upper'] - ci['lower']],
                y=["Interval"],
                base=[ci['lower']],
                orientation='h',
                marker=dict(color="rgba(232,93,38,0.15)", line=dict(color=UI_THEME['primary_color'], width=1.5)),
                showlegend=False,
                hoverinfo='skip',
            ))
            fig_ci.add_trace(go.Scatter(
                x=[prediction], y=["Interval"],
                mode='markers',
                marker=dict(size=14, color=UI_THEME['primary_color'], symbol='diamond'),
                name="Prediction",
                showlegend=False,
            ))
            fig_ci.update_layout(
                height=90, margin=dict(t=5, b=5, l=10, r=10),
                xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#E8ECF1"),
                yaxis=dict(showticklabels=False),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ci, use_container_width=True)

        # --- Property context ---
        context = st.session_state.model_loader.estimate_property_context(input_data)
        if context:
            st.markdown("##### Property Context")
            st.markdown(f"""
            <div class="glass-card" style="padding:1rem;">
                <b>Region:</b> {context.get('region', 'Unknown')}<br>
                <b>Income Level:</b> {context.get('income_level', 'Unknown')}<br>
                <b>Density:</b> {context.get('density', 'Unknown')}<br>
                <b>Location Appeal:</b> {context.get('location_appeal', 'Unknown')}
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.caption("Check your input values and try again.")
        if st.checkbox("Show debug info"):
            st.write({"model": model_name, "input": input_data, "error": str(e)})


# ===================================================================
# MODEL INSIGHTS PAGE
# ===================================================================

def render_insights_page():
    """Render the model insights page."""
    st.markdown('<div class="section-header"><span class="icon">&#128202;</span><h3>Model Insights &amp; Performance</h3></div>', unsafe_allow_html=True)
    st.caption("Explore model accuracy, feature importance, and California housing patterns.")

    available_models = st.session_state.model_loader.get_available_models()
    if not available_models:
        st.error("No models available for analysis.")
        return

    selected_model = st.selectbox("Select model for analysis:", options=available_models)

    tab1, tab2, tab3 = st.tabs(["Performance", "Feature Importance", "Market Insights"])

    with tab1:
        render_performance_analysis(selected_model)
    with tab2:
        render_feature_importance_analysis(selected_model)
    with tab3:
        render_california_specific_insights(selected_model)


def render_performance_analysis(model_name: str):
    """Render model performance analysis."""
    model_info = st.session_state.model_loader.get_model_info(model_name)
    if not model_info or 'validation_metrics' not in model_info:
        st.warning("No performance data available for this model.")
        return

    metrics = model_info['validation_metrics']

    # Metric tiles row
    tiles = []
    if 'rmse' in metrics:
        tiles.append(("RMSE", f"${metrics['rmse']:,.0f}", "Lower is better"))
    if 'r2' in metrics:
        tiles.append(("R\u00b2 Score", f"{metrics['r2']:.3f}", "Higher is better (max 1.0)"))
    if 'mae' in metrics:
        tiles.append(("MAE", f"${metrics['mae']:,.0f}", "Mean Absolute Error"))
    if 'mape' in metrics:
        tiles.append(("MAPE", f"{metrics['mape']:.1f}%", "Percentage Error"))

    cols = st.columns(len(tiles))
    for i, (title, value, tip) in enumerate(tiles):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="value">{value}</div>
                <div class="title">{title}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(tip)

    # Performance interpretation
    rmse = metrics.get('rmse', 0)
    r2 = metrics.get('r2', 0)
    mean_house_value = 400000
    rmse_pct = (rmse / mean_house_value) * 100

    if rmse_pct < 5:
        level, color = "Excellent", UI_THEME['success_color']
    elif rmse_pct < 10:
        level, color = "Very Good", "#2E7D32"
    elif rmse_pct < 15:
        level, color = "Good", UI_THEME['accent_color']
    elif rmse_pct < 25:
        level, color = "Fair", "#E6A817"
    else:
        level, color = "Needs Improvement", "#DC2626"

    st.markdown(f"""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin:0 0 0.5rem;">Performance Summary</h4>
        <p style="margin:0 0 0.3rem;"><b>Rating:</b> <span style="color:{color}; font-weight:700;">{level}</span></p>
        <p style="margin:0 0 0.3rem;"><b>Typical Error:</b> &plusmn;${rmse:,.0f} ({rmse_pct:.1f}% of avg CA home value)</p>
        <p style="margin:0;"><b>Variance Explained:</b> {r2*100:.1f}% of California housing price variation</p>
    </div>
    """, unsafe_allow_html=True)


def render_feature_importance_analysis(model_name: str):
    """Render feature importance analysis."""
    importance_df = st.session_state.model_loader.get_feature_importance(model_name, top_n=25)
    if importance_df is None:
        st.warning("Feature importance not available for this model type.")
        return

    top_15 = importance_df.head(15)

    fig = go.Figure(go.Bar(
        x=top_15['importance'].values[::-1],
        y=top_15['feature'].values[::-1],
        orientation='h',
        marker=dict(
            color=top_15['importance'].values[::-1],
            colorscale=[[0, "#D4E4F7"], [0.5, "#1B6B93"], [1, "#E85D26"]],
        ),
    ))
    fig.update_layout(
        title=f"Top 15 Features \u2014 {model_name}",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=550,
        margin=dict(l=180, t=48, b=40, r=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFBFD",
        xaxis=dict(gridcolor="#E8ECF1"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature categories
    st.markdown("##### Feature Categories")
    categories = {
        'Geographic': [], 'Density': [], 'Income': [], 'Engineered': [], 'Original': [],
    }
    for _, row in importance_df.iterrows():
        f = row['feature']
        if any(x in f for x in ['distance_', 'northern_', 'coastal']):
            categories['Geographic'].append((f, row['importance']))
        elif any(x in f for x in ['per_household', 'per_room', 'density']):
            categories['Density'].append((f, row['importance']))
        elif any(x in f for x in ['income', 'Income']):
            categories['Income'].append((f, row['importance']))
        elif any(x in f for x in ['squared', 'interaction', 'category_']):
            categories['Engineered'].append((f, row['importance']))
        else:
            categories['Original'].append((f, row['importance']))

    for cat, feats in categories.items():
        if feats:
            with st.expander(f"{cat} ({len(feats)} features)"):
                for feat, imp in sorted(feats, key=lambda x: x[1], reverse=True):
                    st.write(f"- **{feat}**: {imp:.4f}")


def render_california_specific_insights(model_name: str):
    """Render California-specific housing insights."""

    importance_df = st.session_state.model_loader.get_feature_importance(model_name, top_n=10)

    if importance_df is not None:
        st.markdown("##### Top Price Drivers")

        insights = {
            'median_income': ("Income", "Affluent areas command premium prices"),
            'ocean_proximity': ("Ocean Proximity", "Coastal properties carry a significant premium"),
            'total_rooms': ("Property Size", "Larger homes correlate with higher values"),
            'latitude': ("North-South Location", "Bay Area vs Central Valley pricing gap"),
            'longitude': ("East-West Location", "Coastal vs inland value differential"),
            'housing_median_age': ("Property Age", "Newer developments often list higher"),
            'population': ("Population Density", "Demand drives pricing in dense areas"),
            'distance_to_Los_Angeles': ("LA Proximity", "Distance from LA inversely affects value"),
            'distance_to_San_Francisco': ("SF Proximity", "Bay Area proximity is a major factor"),
        }

        for _, row in importance_df.head(8).iterrows():
            feature = row['feature']
            for key, (title, desc) in insights.items():
                if key in feature.lower():
                    st.markdown(f'<div class="accent-box"><b>{title}</b> &mdash; {desc}</div>', unsafe_allow_html=True)
                    break

    # Regional breakdown
    st.markdown("##### Regional Price Patterns")

    regions = [
        ("Bay Area (SF)", "$800K\u2013$2M+", "Tech industry", "#E85D26"),
        ("Los Angeles Metro", "$600K\u2013$1.5M", "Entertainment / aerospace", "#1B6B93"),
        ("San Diego", "$700K\u2013$1.2M", "Military / biotech", "#2E7D32"),
        ("Central Valley", "$300K\u2013$600K", "Agriculture", "#7C3AED"),
    ]

    region_cols = st.columns(len(regions))
    for i, (name, price, industry, color) in enumerate(regions):
        with region_cols[i]:
            st.markdown(f"""
            <div class="metric-tile" style="border-top: 3px solid {color};">
                <div class="title" style="font-weight:600; color:{color};">{name}</div>
                <div class="value" style="font-size:1.1rem;">{price}</div>
                <div class="title">{industry}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin:0 0 0.5rem;">Key Price Factors</h4>
        <ol style="margin:0; padding-left:1.2rem; color:#334155; line-height:1.8;">
            <li><b>Income</b> &mdash; strongest predictor of home values</li>
            <li><b>Location</b> &mdash; coastal proximity & major city access</li>
            <li><b>Density</b> &mdash; rooms per household indicates desirability</li>
            <li><b>Age</b> &mdash; newer construction often lists higher</li>
            <li><b>Demographics</b> &mdash; population characteristics affect demand</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ===================================================================
# CALIFORNIA EXPLORER PAGE
# ===================================================================

def render_california_explorer_page():
    """Render California housing data explorer."""
    st.markdown('<div class="section-header"><span class="icon">&#128506;</span><h3>California Housing Market Explorer</h3></div>', unsafe_allow_html=True)
    st.caption("Interactive visualizations of housing trends and regional patterns.")

    np.random.seed(42)
    n_samples = 1000

    sample_data = pd.DataFrame({
        'longitude': np.random.uniform(-124, -114, n_samples),
        'latitude': np.random.uniform(32.5, 42, n_samples),
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'median_house_value': np.random.uniform(100000, 500000, n_samples),
        'ocean_proximity': np.random.choice(['NEAR BAY', 'NEAR OCEAN', 'INLAND', '<1H OCEAN'], n_samples),
    })
    sample_data.loc[sample_data['latitude'] > 37, 'median_house_value'] *= 1.3
    sample_data.loc[sample_data['ocean_proximity'].isin(['NEAR BAY', 'NEAR OCEAN']), 'median_house_value'] *= 1.2

    # Map
    st.markdown("##### Housing Value Map")
    fig_map = px.scatter_mapbox(
        sample_data, lat="latitude", lon="longitude",
        color="median_house_value", size="median_income",
        hover_data=["ocean_proximity"],
        color_continuous_scale=[[0, "#D4E4F7"], [0.5, "#1B6B93"], [1, "#E85D26"]],
        mapbox_style="carto-positron",
        zoom=4.8, center={"lat": 37.2, "lon": -119.5},
    )
    fig_map.update_layout(
        height=550, margin=dict(t=10, b=10, l=0, r=0),
        coloraxis_colorbar=dict(title="Value ($)", tickprefix="$", tickformat=",.0f"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        sample_data['region'] = sample_data['latitude'].apply(
            lambda x: 'Northern CA' if x > 37 else 'Central CA' if x > 35 else 'Southern CA'
        )
        region_stats = sample_data.groupby('region')['median_house_value'].mean().reset_index()
        region_stats.columns = ['region', 'avg_value']

        fig_bar = go.Figure(go.Bar(
            x=region_stats['region'], y=region_stats['avg_value'],
            marker=dict(color=["#E85D26", "#1B6B93", "#2E7D32"], cornerradius=6),
            text=region_stats['avg_value'].apply(lambda v: f"${v:,.0f}"),
            textposition='outside',
        ))
        fig_bar.update_layout(
            title="Average Value by Region",
            yaxis_title="Value ($)", xaxis_title="",
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFD",
            yaxis=dict(gridcolor="#E8ECF1", tickprefix="$", tickformat=",.0f"),
            margin=dict(t=48, b=40, l=60, r=24),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(
            sample_data, x='median_income', y='median_house_value',
            color='ocean_proximity',
            color_discrete_sequence=["#E85D26", "#1B6B93", "#2E7D32", "#7C3AED"],
            labels={'median_income': 'Median Income ($10K)', 'median_house_value': 'House Value ($)'},
            opacity=0.6,
        )
        fig_scatter.update_layout(
            title="Income vs House Value",
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFD",
            yaxis=dict(gridcolor="#E8ECF1", tickprefix="$", tickformat=",.0f"),
            xaxis=dict(gridcolor="#E8ECF1"),
            margin=dict(t=48, b=40, l=60, r=24),
            legend=dict(title="", orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ===================================================================
# ABOUT PAGE
# ===================================================================

def render_about_page():
    """Render the about page."""
    st.markdown('<div class="section-header"><span class="icon">&#8505;&#65039;</span><h3>About This Project</h3></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="margin:0 0 0.5rem;">California Housing Price Prediction System</h3>
        <p style="margin:0; color:#64748B; line-height:1.6;">
            An end-to-end machine learning application that predicts California property values
            using multiple regression models trained on comprehensive housing data with 39+ engineered features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("##### Key Features")
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="icon">&#127919;</div>
            <h4>Accurate Predictions</h4>
            <p>Multiple ML models including Random Forest, XGBoost, and Ridge regression.</p>
        </div>
        <div class="feature-card">
            <div class="icon">&#128506;</div>
            <h4>California-Specific</h4>
            <p>Optimized for Golden State geography, demographics, and market patterns.</p>
        </div>
        <div class="feature-card">
            <div class="icon">&#128202;</div>
            <h4>39+ Features</h4>
            <p>Geographic distances, density ratios, income categories, and interaction terms.</p>
        </div>
        <div class="feature-card">
            <div class="icon">&#127942;</div>
            <h4>Champion Model</h4>
            <p>Automatic selection of the best-performing model from the training pipeline.</p>
        </div>
        <div class="feature-card">
            <div class="icon">&#128200;</div>
            <h4>Deep Insights</h4>
            <p>Feature importance analysis, confidence intervals, and market context.</p>
        </div>
        <div class="feature-card">
            <div class="icon">&#9881;&#65039;</div>
            <h4>Production Ready</h4>
            <p>Full pipeline from data ingestion to web deployment with Streamlit.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two columns: tech stack + how to use
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Technology Stack")
        st.markdown("""
        <div class="glass-card">
            <p style="margin:0; line-height:2;">
                <b>Frontend:</b> Streamlit<br>
                <b>ML Models:</b> scikit-learn, XGBoost<br>
                <b>Data:</b> pandas, NumPy<br>
                <b>Visualization:</b> Plotly<br>
                <b>Deployment:</b> Streamlit Cloud
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Model perf
        if st.session_state.models_loaded:
            deployment_info = st.session_state.model_loader.get_deployment_summary()
            if deployment_info:
                st.markdown("##### Champion Model")
                perf_lines = []
                if 'champion_r2' in deployment_info:
                    perf_lines.append(f"R\u00b2 Score: **{deployment_info['champion_r2']:.3f}**")
                if 'champion_rmse' in deployment_info:
                    perf_lines.append(f"RMSE: **${deployment_info['champion_rmse']:,.0f}**")
                if 'training_samples' in deployment_info:
                    perf_lines.append(f"Training Data: **{deployment_info['training_samples']:,} properties**")
                st.markdown("  \n".join(perf_lines))

    with col2:
        st.markdown("##### How to Use")
        st.markdown("""
        <div class="glass-card">
            <ol style="margin:0; padding-left:1.2rem; line-height:2; color:#334155;">
                <li>Choose a <b>California area preset</b> or enter custom values</li>
                <li>Select your preferred <b>ML model</b> in the sidebar</li>
                <li>Get an <b>instant price estimate</b> with confidence interval</li>
                <li>Explore <b>Model Insights</b> for feature analysis</li>
                <li>Use the <b>Explorer</b> for market visualization</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### California Regions")
        regions_data = [
            ("Northern CA", "Bay Area tech hub", "$600K\u2013$1M+"),
            ("Central CA", "Agricultural heartland", "$300K\u2013$600K"),
            ("Southern CA", "LA / SD metros", "$500K\u2013$1.2M"),
        ]
        for name, desc, price in regions_data:
            st.markdown(f'<div class="accent-box"><b>{name}</b> \u2014 {desc} \u2014 {price}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="accent-box warning" style="margin-top:1rem;">
        <b>Disclaimer:</b> Predictions are estimates based on historical data. Actual prices vary with market conditions,
        local factors, and recent comparable sales. Use as guidance, not definitive valuations.
    </div>
    """, unsafe_allow_html=True)


# ===================================================================
# CALIFORNIA CONTEXT
# ===================================================================

def render_california_context(input_data: Dict[str, Any]):
    """Render California-specific context information."""
    st.markdown('<div class="section-header"><span class="icon">&#127758;</span><h3>Market Context</h3></div>', unsafe_allow_html=True)

    latitude = input_data.get('latitude', 37.0)
    income = input_data.get('median_income', 5.0)
    proximity = input_data.get('ocean_proximity', 'INLAND')

    col1, col2, col3 = st.columns(3)

    with col1:
        if latitude >= 37.0:
            region_name, region_char, region_range = "Northern California", "Tech hub, high incomes", "$600K\u2013$1M+"
        elif latitude >= 35.0:
            region_name, region_char, region_range = "Central California", "Agricultural, moderate prices", "$300K\u2013$600K"
        else:
            region_name, region_char, region_range = "Southern California", "Urban metros, entertainment", "$500K\u2013$1.2M"

        st.markdown(f"""
        <div class="metric-tile" style="border-top:3px solid {UI_THEME['primary_color']}; text-align:left; padding:1rem;">
            <div class="title" style="font-weight:600;">{region_name}</div>
            <div style="font-size:0.88rem; color:#64748B; margin:0.3rem 0;">{region_char}</div>
            <div class="value" style="font-size:1rem;">{region_range}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        income_annual = income * 10000
        if income < 3:
            income_context = "Below median"
        elif income < 6:
            income_context = "Around median"
        elif income < 10:
            income_context = "Above median"
        else:
            income_context = "High income"

        st.markdown(f"""
        <div class="metric-tile" style="border-top:3px solid {UI_THEME['accent_color']}; text-align:left; padding:1rem;">
            <div class="title" style="font-weight:600;">Income Level</div>
            <div style="font-size:0.88rem; color:#64748B; margin:0.3rem 0;">{income_context}</div>
            <div class="value" style="font-size:1rem;">${income_annual:,.0f}/year</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        premiums = {
            'NEAR BAY': ('+30\u201350%', UI_THEME['success_color']),
            'NEAR OCEAN': ('+20\u201340%', "#2E7D32"),
            '<1H OCEAN': ('+10\u201320%', "#7C3AED"),
            'INLAND': ('Baseline', "#64748B"),
            'ISLAND': ('+50%+', "#E85D26"),
        }
        prem_text, prem_color = premiums.get(proximity, ('Standard', '#64748B'))

        st.markdown(f"""
        <div class="metric-tile" style="border-top:3px solid {prem_color}; text-align:left; padding:1rem;">
            <div class="title" style="font-weight:600;">Location Premium</div>
            <div style="font-size:0.88rem; color:#64748B; margin:0.3rem 0;">{proximity}</div>
            <div class="value" style="font-size:1rem; color:{prem_color};">{prem_text}</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
