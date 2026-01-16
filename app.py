import streamlit as st
import pandas as pd
import joblib
import os
from groq import Groq

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI-Assisted Mining Feasibility System",
    page_icon="⛏️",
    layout="centered"
)

# =====================================================
# LOAD MODEL AND PREPROCESSING TOOLS
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    features = joblib.load("model_features.pkl")  # feature order
    encoders = joblib.load("label_encoders.pkl")  # dict of LabelEncoders
    return model, features, encoders

model, FEATURES, LABEL_ENCODERS = load_model()

# =====================================================
# GROQ CLIENT (OPTIONAL)
# =====================================================
@st.cache_resource
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

groq_client = load_groq_client()

# =====================================================
# FUNCTION: Preprocess Input for Prediction
# =====================================================
def preprocess_input(df):
    df = df.copy()
    # Encode categorical columns using saved encoders
    for col, le in LABEL_ENCODERS.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    # Keep only features used in training
    df = df[FEATURES]
    return df

# =====================================================
# FUNCTION: Groq AI Explanation
# =====================================================
def groq_explanation(df, prediction):
    if groq_client is None:
        return None

    prompt = f"""
You are an environmental sustainability expert.

Mining site data:
{df.to_dict(orient="records")[0]}

Model decision:
Mining Allowed = {"Yes" if prediction == 1 else "No"}

Explain briefly:
1. Reason for the decision
2. Environmental risks
3. Long-term consequences
4. Sustainability recommendations
"""

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=400
    )

    return response.choices[0].message.content

# =====================================================
# SESSION STATE
# =====================================================
if "input_df" not in st.session_state:
    st.session_state.input_df = None

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⛏️ Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Single Site Input", "CSV Upload", "Result"]
)
use_ai = st.sidebar.checkbox("🤖 Enable AI Explanation", value=True)

# =====================================================
# PAGE: Single Site Input
# =====================================================
if page == "Single Site Input":
    st.title("📝 Mining Site Details")

    with st.form("mining_form"):
        data = {
            "Region_ID": "R_NEW",
            "State": st.text_input("State", "Karnataka"),
            "Mountain_Range": st.text_input("Mountain Range", "Western Ghats"),
            "Nearby_River": st.text_input("Nearby River", "Sharavathi"),
            "Wildlife_Sanctuary": st.text_input("Wildlife Sanctuary", "None"),
            "Restriction_Type": st.selectbox("Restriction Type",
                                             ["None", "Wildlife Sanctuary", "National Park", "Biosphere Reserve"]),
            "Dominant_Trees": st.text_input("Dominant Trees", "Teak"),
            "Key_Animals": st.text_input("Key Animals", "Deer"),
            "Minerals_Present": st.text_input("Minerals Present", "Iron Ore"),

            "Elevation_m": st.number_input("Elevation (m)", 50, 5000, 500),
            "Slope_deg": st.slider("Slope (°)", 0, 60, 15),
            "NDVI": st.slider("NDVI", 0.0, 1.0, 0.45),
            "Forest_Cover_Percent": st.slider("Forest Cover (%)", 0, 100, 40),
            "Distance_to_River_km": st.number_input("Distance to River (km)", 0.0, 50.0, 5.0),
            "Protected_Area": st.selectbox("Protected Area", [0, 1]),
            "Existing_Mining": st.selectbox("Existing Mining", [0, 1]),

            "Deforestation_Risk": st.selectbox("Deforestation Risk", ["Low", "Medium", "High", "Very High"]),
            "Water_Pollution_Risk": st.selectbox("Water Pollution Risk", ["Low", "Medium", "High", "Very High"]),
            "Air_Pollution_Risk": st.selectbox("Air Pollution Risk", ["Low", "Medium", "High", "Very High"]),

            "Annual_Rainfall_mm": st.number_input("Annual Rainfall (mm)", 200, 5000, 1500),
            "Seasonal_Rainfall_Variability": st.slider("Rainfall Variability", 0.0, 2.0, 0.5),
            "Avg_Temperature_C": st.slider("Avg Temperature (°C)", 5, 40, 25),
            "Max_Temperature_C": st.slider("Max Temperature (°C)", 10, 50, 35),
            "Min_Temperature_C": st.slider("Min Temperature (°C)", 0, 30, 15),

            "Seismic_Zone": st.selectbox("Seismic Zone", ["I", "II", "III", "IV", "V"]),
            "Population_Density_per_km2": st.number_input("Population Density", 1, 5000, 500),
            "Land_Use_Type": st.selectbox("Land Use Type", ["Forest", "Agriculture", "Urban", "Mining"]),
            "Mining_Employment_Dependency_%": st.slider("Mining Employment Dependency (%)", 0, 100, 20),
            "Past_Mining_Accidents": st.number_input("Past Mining Accidents", 0, 20, 0),
            "Previous_Hazard_Report": st.selectbox("Previous Hazard Report", ["Low", "Medium", "High"]),

            "Distance_to_Road_km": st.number_input("Distance to Road (km)", 0.0, 100.0, 10.0),
            "Distance_to_Rail_km": st.number_input("Distance to Rail (km)", 0.0, 200.0, 25.0),
            "Distance_to_Town_km": st.number_input("Distance to Town (km)", 0.0, 200.0, 50.0),
            "Road_Connectivity": st.selectbox("Road Connectivity", ["Poor", "Moderate", "Good"]),
            "Rail_Connectivity": st.selectbox("Rail Connectivity", ["None", "Low", "Moderate", "High"])
        }

        submit = st.form_submit_button("Save Input")

    if submit:
        st.session_state.input_df = pd.DataFrame([data])
        st.success("✅ Data saved. Go to Result page.")

# =====================================================
# PAGE: CSV Upload
# =====================================================
elif page == "CSV Upload":
    st.title("📂 Batch Mining Assessment")
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        # Preprocess
        df_proc = preprocess_input(df)
        preds = model.predict(df_proc)
        df["Mining_Allowed_Prediction"] = ["Yes" if p == 1 else "No" for p in preds]

        st.success("✅ Predictions generated")
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv, "mining_predictions.csv")

# =====================================================
# PAGE: Result
# =====================================================
elif page == "Result":
    st.title("📊 Mining Approval Result")
    if st.session_state.input_df is None:
        st.warning("Please enter site details first.")
    else:
        df = st.session_state.input_df
        df_proc = preprocess_input(df)
        if st.button("Run Prediction"):
            pred = model.predict(df_proc)[0]
            allowed = pred == 1
            color = "green" if allowed else "red"
            label = "✅ MINING APPROVED" if allowed else "❌ MINING DENIED"
            st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)

            st.markdown("### 🌱 Environmental Notes")
            if df["Protected_Area"][0] == 1:
                st.write("• Site lies in a protected area.")
            if df["Forest_Cover_Percent"][0] > 60:
                st.write("• High forest cover may cause ecological damage.")
            if df["Seismic_Zone"][0] in ["IV", "V"]:
                st.write("• High seismic risk zone.")

            if groq_client and use_ai:
                st.markdown("### 🤖 AI Sustainability Insight")
                with st.spinner("Analyzing..."):
                    explanation = groq_explanation(df_proc, pred)
                    st.write(explanation)
