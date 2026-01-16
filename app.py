import streamlit as st
import pandas as pd
import joblib
from groq import Groq
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mountain Mining Approval System",
    page_icon="⛏️",
    layout="centered"
)

# ---------------- GROQ CLIENT ----------------
def load_groq_client():
    api_key = st.sidebar.text_input(
        "🔑 Groq API Key (optional)",
        type="password",
        help="Used for AI explanation & impact analysis"
    )
    if api_key:
        return Groq(api_key=api_key)
    return None

groq_client = load_groq_client()

def groq_explanation(input_df, prediction, client):
    if client is None:
        return None

    prompt = f"""
You are an environmental sustainability expert.

Given the following mining site data:
{input_df.to_dict(orient="records")[0]}

The machine learning model predicted:
Mining Allowed = {prediction}

Explain:
1. Why this decision was made
2. Environmental risks involved
3. Long-term consequences if mining is carried out
4. Sustainability recommendation

Keep explanation concise and professional.
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# ---------------- ENCODING MAPS ----------------
SEISMIC_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
RISK_MAP = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}

# ---------------- SESSION STATE ----------------
if "input_data" not in st.session_state:
    st.session_state.input_data = pd.DataFrame()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⛏️ Mining System")
page = st.sidebar.radio("Navigation", ["Input Details", "CSV Upload", "Result"])

# ---------------- INPUT PAGE ----------------
if page == "Input Details":
    st.title("📝 Mining Site Input Details")
    with st.form("mining_form"):
        elevation = st.number_input("Elevation (m)", 100, 5000, 1200)
        slope = st.slider("Slope (°)", 0, 60, 20)
        forest = st.slider("Forest Cover (%)", 0, 100, 40)
        rainfall = st.number_input("Annual Rainfall (mm)", 500, 5000, 1800)
        population = st.number_input("Population Density (per km²)", 1, 1000, 120)

        protected = st.selectbox("Protected Area", [0, 1])
        wildlife = st.selectbox("Wildlife Sanctuary", [0, 1])

        river_dist = st.number_input("Distance to River (km)", 0.0, 20.0, 5.0)
        road_dist = st.number_input("Distance to Road (km)", 0.0, 20.0, 3.0)

        ndvi = st.slider("NDVI Index", 0.0, 1.0, 0.45)

        seismic = st.selectbox("Seismic Zone", list(SEISMIC_MAP.keys()))
        deforestation = st.selectbox("Deforestation Risk", list(RISK_MAP.keys()))
        water = st.selectbox("Water Pollution Risk", list(RISK_MAP.keys()))
        air = st.selectbox("Air Pollution Risk", list(RISK_MAP.keys()))

        submit = st.form_submit_button("Save & Continue")

    if submit:
        st.session_state.input_data = pd.DataFrame([{
            "Elevation_m": elevation,
            "Slope_deg": slope,
            "Forest_Cover_Percent": forest,
            "Protected_Area": protected,
            "Wildlife_Sanctuary": wildlife,
            "Annual_Rainfall_mm": rainfall,
            "Population_Density_per_km2": population,
            "Distance_to_River_km": river_dist,
            "Distance_to_Road_km": road_dist,
            "NDVI": ndvi,
            "Seismic_Zone": SEISMIC_MAP[seismic],
            "Deforestation_Risk": RISK_MAP[deforestation],
            "Water_Pollution_Risk": RISK_MAP[water],
            "Air_Pollution_Risk": RISK_MAP[air]
        }])
        st.success("✅ Input saved. Go to Result page.")

# ---------------- CSV UPLOAD ----------------
elif page == "CSV Upload":
    st.title("📂 Batch Mining Assessment")
    file = st.file_uploader("Upload CSV File", type=["csv", "xlsx"])
    if file:
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_excel(file)

        # Encode categorical columns if needed
        for col, mapping in [("Seismic_Zone", SEISMIC_MAP),
                             ("Deforestation_Risk", RISK_MAP),
                             ("Water_Pollution_Risk", RISK_MAP),
                             ("Air_Pollution_Risk", RISK_MAP)]:
            if col in df.columns:
                df[col] = df[col].map(mapping)

        df["Mining_Allowed"] = model.predict(df)
        st.success("✅ Predictions generated")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "mining_results.csv")

# ---------------- RESULT PAGE ----------------
elif page == "Result":
    st.title("📊 Mining Approval Result")
    if st.session_state.input_data.empty:
        st.warning("Please enter inputs first.")
    else:
        df = st.session_state.input_data
        st.dataframe(df)

        if st.button("Run Mining Model"):
            pred = model.predict(df)[0]

            # Color-coded result
            st.markdown(
                f"<h2 style='color: {'green' if pred=='Yes' else 'red'};'>"
                f"{'✅ MINING APPROVED' if pred=='Yes' else '❌ MINING DENIED'}</h2>",
                unsafe_allow_html=True
            )

            # ---------------- AI Explanation ----------------
            st.markdown("### 🧠 Rule-based Explanation")
            reasons = []
            if df["Protected_Area"][0] == 1:
                reasons.append("The region is under protected environmental status.")
            if df["Forest_Cover_Percent"][0] > 60:
                reasons.append("High forest cover increases ecological risk.")
            if df["Seismic_Zone"][0] >= 4:
                reasons.append("High seismic activity increases disaster risk.")
            if df["Distance_to_River_km"][0] < 1:
                reasons.append("Proximity to river increases water pollution risk.")
            if not reasons:
                reasons.append("Environmental and infrastructural risks are within acceptable limits.")

            for r in reasons:
                st.write("•", r)

            st.markdown("### 🌱 Environmental Consequences")
            st.write("""
- Possible deforestation and habitat loss  
- Increased risk of water and air pollution  
- Long-term impact on biodiversity and local communities  
""")

            # ---------------- Groq AI Explanation ----------------
            if groq_client:
                st.markdown("### 🤖 Groq AI Insight")
                with st.spinner("Running AI analysis..."):
                    groq_text = groq_explanation(df, pred, groq_client)
                    st.write(groq_text)
