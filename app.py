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
# LOAD MODEL + FEATURES + ENCODERS
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    features = joblib.load("model_features.pkl")
    encoders = joblib.load("label_encoders.pkl")
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
# GROQ AI EXPLANATION
# =====================================================
def groq_explanation(df, decision, reasons=None):
    if groq_client is None:
        return "Groq AI not enabled."

    site_data = df.to_dict(orient="records")[0]

    if decision == "DENIED":
        prompt = f"""
You are an environmental clearance officer.

Mining site details:
{site_data}

Final decision: MINING DENIED

Reasons:
{reasons}

Explain clearly:
1. Why mining cannot be approved
2. Environmental and legal violations
3. Long-term ecological risks
4. Impact on wildlife, water, and population
5. Suggest safer alternatives
"""
    else:
        prompt = f"""
You are an environmental clearance officer.

Mining site details:
{site_data}

Final decision: MINING APPROVED

Explain clearly:
1. Why mining is permitted
2. Environmental conditions satisfied
3. Risk mitigation measures
4. Legal compliance justification
5. Sustainability safeguards required
"""

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=450
    )

    return response.choices[0].message.content
# =====================================================
# PREPROCESS INPUT (NO ERRORS)
# =====================================================
def preprocess_input(df):
    df = df.copy()

    for col, le in LABEL_ENCODERS.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    df = df[FEATURES]
    return df

# =====================================================
# RULE-BASED DENIAL (ENVIRONMENTAL LAW)
# =====================================================
if st.button("Run Prediction"):
    denial_reasons = rule_based_denial(df)

    # ---------------- DENIED ----------------
    if denial_reasons:
        st.markdown(
            "<h2 style='color:red'>❌ MINING DENIED</h2>",
            unsafe_allow_html=True
        )

        st.markdown("### ❗ Reasons for Denial")
        for r in denial_reasons:
            st.write(f"• {r}")

        if groq_client and use_ai:
            st.markdown("### 🤖 AI Justification (Groq)")
            with st.spinner("Generating expert explanation..."):
                explanation = groq_explanation(
                    df,
                    decision="DENIED",
                    reasons=denial_reasons
                )
                st.write(explanation)

    # ---------------- APPROVED ----------------
    else:
        pred = model.predict(df_proc)[0]

        if pred == 1:
            st.markdown(
                "<h2 style='color:green'>✅ MINING APPROVED</h2>",
                unsafe_allow_html=True
            )

            st.markdown("### ✅ Approval Basis")
            st.write("• No critical environmental or legal violations detected")
            st.write("• Risks are within permissible limits")
            st.write("• Location satisfies regulatory thresholds")

            if groq_client and use_ai:
                st.markdown("### 🤖 AI Justification (Groq)")
                with st.spinner("Generating expert explanation..."):
                    explanation = groq_explanation(
                        df,
                        decision="APPROVED"
                    )
                    st.write(explanation)

        else:
            st.markdown(
                "<h2 style='color:red'>❌ MINING DENIED (ML Risk)</h2>",
                unsafe_allow_html=True
            )

            if groq_client and use_ai:
                st.markdown("### 🤖 AI Risk Explanation (Groq)")
                explanation = groq_explanation(
                    df,
                    decision="DENIED",
                    reasons=["High predicted environmental risk by ML model"]
                )
                st.write(explanation)
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
# PAGE: SINGLE SITE INPUT
# =====================================================
if page == "Single Site Input":
    st.title("📝 Mining Site Details")

    with st.form("mining_form"):
        data = {
            "Region_ID": "R_NEW",
            "State": st.text_input("State", "Karnataka"),
            "Mountain_Range": st.text_input("Mountain Range", "Western Ghats"),
            "Nearby_River": st.text_input("Nearby River", "Sharavathi"),
            "Wildlife_Sanctuary": st.selectbox(
                "Wildlife Sanctuary",
                [
                    "Dandeli Wildlife Sanctuary",
                    "Achanakmar Wildlife Sanctuary",
                    "Similipal Wildlife Sanctuary",
                    "Kawal Wildlife Sanctuary",
                    "Kaziranga Wildlife Sanctuary"
                ]
            ),
            "Restriction_Type": st.selectbox(
                "Restriction Type",
                ["None", "Wildlife Sanctuary", "National Park", "Biosphere Reserve"]
            ),
            "Dominant_Trees": st.text_input("Dominant Trees", "Teak"),
            "Key_Animals": st.text_input("Key Animals", "Deer"),
            "Minerals_Present": st.text_input("Minerals Present", "Iron Ore"),

            "Elevation_m": st.number_input("Elevation (m)", 50, 5000, 500),
            "Slope_deg": st.slider("Slope (°)", 0, 60, 15),
            "NDVI": st.slider("NDVI", 0.0, 1.0, 0.45),
            "Forest_Cover_Percent": st.slider("Forest Cover (%)", 0, 100, 70),
            "Distance_to_River_km": st.number_input("Distance to River (km)", 0.0, 50.0, 2.0),
            "Protected_Area": st.selectbox("Protected Area", [0, 1]),
            "Existing_Mining": st.selectbox("Existing Mining", [0, 1]),

            "Deforestation_Risk": st.selectbox(
                "Deforestation Risk", ["Low", "Medium", "High", "Very High"]
            ),
            "Water_Pollution_Risk": st.selectbox(
                "Water Pollution Risk", ["Low", "Medium", "High", "Very High"]
            ),
            "Air_Pollution_Risk": st.selectbox(
                "Air Pollution Risk", ["Low", "Medium", "High", "Very High"]
            ),

            "Annual_Rainfall_mm": st.number_input("Annual Rainfall (mm)", 200, 5000, 2500),
            "Seasonal_Rainfall_Variability": st.slider("Rainfall Variability", 0.0, 2.0, 1.2),
            "Avg_Temperature_C": st.slider("Avg Temperature (°C)", 5, 40, 28),
            "Max_Temperature_C": st.slider("Max Temperature (°C)", 10, 50, 38),
            "Min_Temperature_C": st.slider("Min Temperature (°C)", 0, 30, 18),

            "Seismic_Zone": st.selectbox("Seismic Zone", ["I", "II", "III", "IV", "V"]),
            "Population_Density_per_km2": st.number_input("Population Density", 1, 5000, 900),
            "Land_Use_Type": st.selectbox(
                "Land Use Type", ["Forest", "Agriculture", "Urban", "Mining"]
            ),
            "Mining_Employment_Dependency_%": st.slider(
                "Mining Employment Dependency (%)", 0, 100, 30
            ),
            "Past_Mining_Accidents": st.number_input("Past Mining Accidents", 0, 20, 2),
            "Previous_Hazard_Report": st.selectbox(
                "Previous Hazard Report", ["Low", "Medium", "High"]
            ),

            "Distance_to_Road_km": st.number_input("Distance to Road (km)", 0.0, 100.0, 6.0),
            "Distance_to_Rail_km": st.number_input("Distance to Rail (km)", 0.0, 200.0, 30.0),
            "Distance_to_Town_km": st.number_input("Distance to Town (km)", 0.0, 200.0, 40.0),
            "Road_Connectivity": st.selectbox("Road Connectivity", ["Poor", "Moderate", "Good"]),
            "Rail_Connectivity": st.selectbox(
                "Rail Connectivity", ["None", "Low", "Moderate", "High"]
            )
        }

        submit = st.form_submit_button("Save Input")

    if submit:
        st.session_state.input_df = pd.DataFrame([data])
        st.success("✅ Data saved. Go to Result page.")

# =====================================================
# PAGE: CSV UPLOAD
# =====================================================
elif page == "CSV Upload":
    st.title("📂 Batch Mining Assessment")

    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df_proc = preprocess_input(df)

        preds = model.predict(df_proc)
        df["Mining_Allowed_Prediction"] = ["Yes" if p == 1 else "No" for p in preds]

        st.dataframe(df.head())
        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False).encode("utf-8"),
            "mining_predictions.csv"
        )

# =====================================================
# PAGE: RESULT
# =====================================================
elif page == "Result":
    st.title("📊 Mining Approval Result")

    if st.session_state.input_df is None:
        st.warning("Please enter site details first.")
    else:
        df = st.session_state.input_df
        df_proc = preprocess_input(df)

        if st.button("Run Prediction"):
            denial_reasons = rule_based_denial(df)

            if denial_reasons:
                st.markdown(
                    "<h2 style='color:red'>❌ MINING DENIED</h2>",
                    unsafe_allow_html=True
                )
                st.markdown("### ❗ Reasons for Denial")
                for r in denial_reasons:
                    st.write(f"• {r}")

                if groq_client and use_ai:
                    st.markdown("### 🤖 AI Sustainability Insight")
                    with st.spinner("Analyzing..."):
                        st.write(groq_explanation(df, denial_reasons))

            else:
                pred = model.predict(df_proc)[0]
                st.markdown(
                    "<h2 style='color:green'>✅ MINING APPROVED</h2>",
                    unsafe_allow_html=True
                )
