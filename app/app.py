"""
Crop Yield Prediction App
=========================
Streamlit application for farmers to predict crop yield.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from model_utils import explain_prediction, get_recommendations, predict_yield

# Page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1565C0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .yield-value {
        font-size: 3rem;
        font-weight: bold;
    }
    .yield-unit {
        font-size: 1.2rem;
    }
    .success-tip {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        color: #1B5E20;
    }
    .warning-tip {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 10px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        color: #E65100;
    }
    .info-tip {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 10px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        color: #0D47A1;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    # Header
    st.header("🌾 Crop Yield Predictor")
    st.markdown(
        "Predict your maize yield based on farm conditions and management practices"
    )
    st.markdown("---")

    # Sidebar - About
    st.sidebar.title("About")
    st.sidebar.info(
        """
        **Crop Yield Predictor v1.0**

        This app predicts maize yield in bags per hectare based on:
        - Farm characteristics
        - Crop management practices
        - Environmental factors

        **Model Performance:**
        - R² Score: 0.75
        - Average Error: ±3.25 bags/ha
        """
    )

    # Main content - Input form
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<p class="sub-header">📋 Farm Information</p>', unsafe_allow_html=True
        )

        # Farm Characteristics
        with st.expander("🏡 Farm Characteristics", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                farm_size = st.number_input(
                    "Farm Size (hectares)",
                    min_value=0.1,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    help="Total area of your farm in hectares",
                )
            with col_b:
                experience = st.number_input(
                    "Farming Experience (years)",
                    min_value=0,
                    max_value=60,
                    value=10,
                    help="Years of farming experience",
                )

        # Soil & Location
        with st.expander("🌍 Soil & Location", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                soil_type = st.selectbox(
                    "Soil Type",
                    options=["sandy", "loamy", "clayey", "silty"],
                    help="Type of soil on your farm",
                )
            with col_b:
                water_retention = st.slider(
                    "Water Retention Score",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1=Low water retention, 5=High water retention",
                )

        # Crop Management
        with st.expander("🌱 Crop Management", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                seed_type = st.selectbox(
                    "Seed Type",
                    options=["improved", "local"],
                    help="Improved seeds give higher yields",
                )
            with col_b:
                maize_variety = st.selectbox(
                    "Maize Variety",
                    options=["white", "yellow"],
                    help="White or yellow maize",
                )

            planting_month = st.selectbox(
                "Planting Month",
                options=["April", "May", "June", "July", "August"],
                help="Month when you planted/plan to plant",
            )

        # Irrigation
        with st.expander("💧 Irrigation", expanded=False):
            is_irrigated = st.checkbox(
                "Use Irrigation", help="Do you have irrigation system?"
            )
            irrigation_vol = 0
            if is_irrigated:
                irrigation_vol = st.number_input(
                    "Irrigation Volume (liters)",
                    min_value=0,
                    max_value=10000,
                    value=3000,
                    step=100,
                    help="Amount of water used for irrigation",
                )

        # Fertilizer
        with st.expander("🧪 Fertilizer", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                inorganic_fert = st.checkbox("Use Inorganic Fertilizer")
            with col_b:
                organic_fert = st.checkbox("Use Organic Fertilizer")

            inorganic_qty = 0
            if inorganic_fert:
                col_c, col_d = st.columns(2)
                with col_c:
                    st.selectbox(
                        "Inorganic Fertilizer Type",
                        options=["npk", "urea"],
                        help="Common fertilizer types",
                    )
                with col_d:
                    inorganic_qty = st.number_input(
                        "Inorganic Fertilizer Quantity (kg)",
                        min_value=0,
                        max_value=5000,
                        value=500,
                        step=50,
                    )

            organic_qty = 0
            if organic_fert:
                organic_qty = st.number_input(
                    "Organic Fertilizer Quantity (kg)",
                    min_value=0,
                    max_value=5000,
                    value=300,
                    step=50,
                    help="Amount of organic fertilizer used",
                )

        # Pest Management
        with st.expander("🐛 Pest Management", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                pest_occurred = st.checkbox("Pest Attack Occurred")
            with col_b:
                pesticide_used = st.checkbox("Pesticide Used")

            weeks_to_attack = 0
            if pest_occurred:
                weeks_to_attack = st.slider(
                    "Weeks Before Pest Attack",
                    min_value=1,
                    max_value=12,
                    value=4,
                    help="How many weeks after planting did pest attack occur?",
                )

            weeding_freq = st.slider(
                "Weeding Frequency",
                min_value=1,
                max_value=4,
                value=3,
                help="Number of times weeding was done",
            )

        # Environmental (Estimates)
        with st.expander("🌦️ Environmental Factors (Estimates)", expanded=False):
            rainfall = st.number_input(
                "Expected Total Rainfall (mm)",
                min_value=0,
                max_value=3000,
                value=1500,
                step=50,
                help="Historical/average rainfall for the season",
            )

            flooding_risk = st.selectbox(
                "Flooding Risk",
                options=["Low", "Medium", "High"],
                help="Likelihood of flooding in your area",
            )

    # Prediction Button
    with col2:
        st.markdown("### 🎯 Predict Yield")
        predict_btn = st.button(
            "Predict My Yield", type="primary", use_container_width=True
        )

        if predict_btn:
            # Prepare input data
            input_data = {
                "is_irrigated": 1 if is_irrigated else 0,
                "flooding_occurrence": 1
                if flooding_risk == "High"
                else (0 if flooding_risk == "Low" else 0),
                "total_rainfall_mm": rainfall,
                "water_retention_score": water_retention,
                "inorganic_fert_used": 1 if inorganic_fert else 0,
                "inorganic_fert_qty_kg": inorganic_qty if inorganic_fert else 0,
                "organic_fert_used": 1 if organic_fert else 0,
                "organic_fert_qty_kg": organic_qty if organic_fert else 0,
                "pest_attack_occurred": 1 if pest_occurred else 0,
                "weeks_before_attack": weeks_to_attack if pest_occurred else 0,
                "farm_size_ha": farm_size,
                "experience_years": experience,
                "weeding_frequency": weeding_freq,
                "pesticide_used": 1 if pesticide_used else 0,
                "irrigation_vol_ltrs": irrigation_vol,
                "seed_type": seed_type,
                "maize_variety": maize_variety,
                "soil_type": soil_type,
                "planting_month": planting_month,
                "flooding_risk": flooding_risk,
            }

            # Make prediction
            try:
                result = predict_yield(input_data)

                # Display prediction
                st.markdown(
                    f"""
                <div class="prediction-box">
                    <div class="yield-unit">Predicted Yield</div>
                    <div class="yield-value">{result["yield"]}</div>
                    <div class="yield-unit">bags per hectare</div>
                    <br>
                    <small>Range: {result["confidence_low"]} - {result["confidence_high"]} bags/ha</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Show feature contributions
                explanation = explain_prediction(input_data)
                st.markdown("### 🔍 What's Driving This Prediction?")
                contrib_df = pd.DataFrame(explanation["contributions"])
                # Reverse so largest impact is at top
                contrib_df = contrib_df.iloc[::-1]
                colors = [
                    "#4CAF50" if v > 0 else "#F44336"
                    for v in contrib_df["shap_value"]
                ]
                fig = go.Figure(
                    go.Bar(
                        x=contrib_df["shap_value"],
                        y=contrib_df["feature"],
                        orientation="h",
                        marker_color=colors,
                    )
                )
                fig.update_layout(
                    xaxis_title="Impact on yield (bags/ha)",
                    yaxis_title="",
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"Baseline yield: {explanation['base_value']:.2f} bags/ha. "
                    "Green bars increase yield, red bars decrease it."
                )

                # Get recommendations
                recommendations = get_recommendations(input_data, result["yield"])

                st.markdown("### 💡 Recommendations")
                for rec in recommendations:
                    if rec["type"] == "success":
                        st.markdown(
                            f"""
                        <div class="success-tip">
                            <strong>✓ {rec["title"]}</strong><br>
                            {rec["message"]}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif rec["type"] == "warning":
                        st.markdown(
                            f"""
                        <div class="warning-tip">
                            <strong>⚠ {rec["title"]}</strong><br>
                            {rec["message"]}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                        <div class="info-tip">
                            <strong>💡 {rec["title"]}</strong><br>
                            {rec["message"]}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check your inputs and try again.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🌾 Crop Yield Prediction Model | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
