import streamlit as st
import pandas as pd
import joblib
import os
from groq import Groq

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Mountain Mining Approval System",
    page_icon="⛏️",
    layout="centered"
)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# =====================================================
# LOAD GROQ CLIENT
# =====================================================
@st.cache_resource
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return Groq(api_key=api_key)
    return None

groq_client = load_groq_client()

# =====================================================
# ENCODING MAPS
# =====================================================
SEISMIC_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
RISK_MAP = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}

# =====================================================
# ALL TRAINING FEATURES (CRITICAL)
# =====================================================
MODEL_FEATURES = [
    "Region_ID", "Wildlife_Sanctuary", "Elevation_m", "Slope_deg", "NDVI",
    "Forest_Cover_Percent", "Distance_to_River_km", "Protected_Area",
    "Existing_Mining", "Deforestation_Risk", "Water_Pollution_Risk",
    "Air_Pollution_Risk", "Annual_Rainfall_mm",
    "Seasonal_Rainfall_Variability", "Avg_Temperature_C",
    "Max_Temperature_C", "Min_Temperature_C", "Seismic_Zone",
    "Population_Density_per_km2", "Mining_Employment_Dependency_%",
    "Past_Mining_Accidents", "Previous_Hazard_Report",
    "Distance_to_Road_km", "Distance_to_Rail_km",
    "Distance_to_Town_km", "Road_Connectivity", "Rail_Connectivity"
]

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⛏️ Mining System")
page = st.sidebar.radio("Navigation", ["Input Details", "Result"])
use_ai = st.sidebar.checkbox("🤖 Enable AI Explanation", value=True)

# =====================================================
# INPUT PAGE
# =====================================================
if page == "Input Details":
    st.title("📝 Mining Site Details")

    with st.form("input_form"):
        elevation = st.number_input("Elevation (m)", 100, 5000, 1200)
        slope = st.slider("Slope (°)", 0, 60, 20)
        forest = st.slider("Forest Cover (%)", 0, 100, 40)
        rainfall = st.number_input("Annual Rainfall (mm)", 500, 5000, 1800)
        population = st.number_input("Population Density", 1, 5000, 120)
        river_dist = st.number_input("Distance to River (km)", 0.0, 20.0, 3.0)
        road_dist = st.number_input("Distance to Road (km)", 0.0, 20.0, 2.5)
        ndvi = st.slider("NDVI", 0.0, 1.0, 0.45)

        seismic = st.selectbox("Seismic Zone", list(SEISMIC_MAP.keys()))
        def_risk = st.selectbox("Deforestation Risk", list(RISK_MAP.keys()))
        water_risk = st.selectbox("Water Pollution Risk", list(RISK_MAP.keys()))
        air_risk = st.selectbox("Air Pollution Risk", list(RISK_MAP.keys()))

        submit = st.form_submit_button("Save")

    if submit:
        # ---------------- FULL FEATURE ROW ----------------
        row = {
            "Region_ID": 1,
            "Wildlife_Sanctuary": 0,
            "Elevation_m": elevation,
            "Slope_deg": slope,
            "NDVI": ndvi,
            "Forest_Cover_Percent": forest,
            "Distance_to_River_km": river_dist,
            "Protected_Area": 0,
            "Existing_Mining": 0,
            "Deforestation_Risk": RISK_MAP[def_risk],
            "Water_Pollution_Risk": RISK_MAP[water_risk],
            "Air_Pollution_Risk": RISK_MAP[air_risk],
            "Annual_Rainfall_mm": rainfall,
            "Seasonal_Rainfall_Variability": 0.3,
            "Avg_Temperature_C": 25,
            "Max_Temperature_C": 35,
            "Min_Temperature_C": 15,
            "Seismic_Zone": SEISMIC_MAP[seismic],
            "Population_Density_per_km2": population,
            "Mining_Employment_Dependency_%": 20,
            "Past_Mining_Accidents": 0,
            "Previous_Hazard_Report": 0,
            "Distance_to_Road_km": road_dist,
            "Distance_to_Rail_km": 8,
            "Distance_to_Town_km": 12,
            "Road_Connectivity": 1,
            "Rail_Connectivity": 0
        }

        st.session_state.input_df = pd.DataFrame([row])[MODEL_FEATURES]
        st.success("Input saved successfully!")

# =====================================================
# RESULT PAGE
# =====================================================
elif page == "Result":
    st.title("📊 Mining Approval Result")

    if "input_df" not in st.session_state:
        st.warning("Please enter input details first.")
    else:
        df = st.session_state.input_df
        st.dataframe(df)

        if st.button("Run Prediction"):
            prediction = model.predict(df)[0]
            approved = prediction == 1

            st.markdown(
                f"<h2 style='color:{'green' if approved else 'red'};'>"
                f"{'✅ MINING APPROVED' if approved else '❌ MINING DENIED'}</h2>",
                unsafe_allow_html=True
            )

            if groq_client and use_ai:
                st.markdown("### 🤖 AI Explanation")
                prompt = f"Explain mining decision based on data: {df.to_dict()}"
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300
                )
                st.write(response.choices[0].message.content)
