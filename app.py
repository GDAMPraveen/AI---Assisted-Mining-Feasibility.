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
# GROQ CLIENT
# =====================================================
@st.cache_resource
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

groq_client = load_groq_client()

# =====================================================
# PREPROCESS INPUT
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
    return df[FEATURES]

# =====================================================
# RULE-BASED DENIAL (LAW + ENVIRONMENT)
# =====================================================
def rule_based_denial(df):
    r = []
    row = df.iloc[0]

    if row["Protected_Area"] == 1:
        r.append("Mining is prohibited inside protected areas.")

    if row["Restriction_Type"] in [
        "Wildlife Sanctuary",
        "National Park",
        "Biosphere Reserve"
    ]:
        r.append(f"Restricted under {row['Restriction_Type']} regulations.")

    if row["Forest_Cover_Percent"] > 60:
        r.append("Forest cover exceeds sustainable mining threshold.")

    if row["Distance_to_River_km"] < 3:
        r.append("Too close to river – high water contamination risk.")

    if row["Deforestation_Risk"] in ["High", "Very High"]:
        r.append("Unacceptable deforestation risk.")

    if row["Water_Pollution_Risk"] in ["High", "Very High"]:
        r.append("High probability of water pollution.")

    if row["Air_Pollution_Risk"] in ["High", "Very High"]:
        r.append("Severe air quality impact predicted.")

    if row["Seismic_Zone"] in ["IV", "V"]:
        r.append("Region lies in a high seismic risk zone.")

    if row["Slope_deg"] > 30:
        r.append("Steep slope increases landslide probability.")

    if row["Population_Density_per_km2"] > 1000:
        r.append("High population density increases human risk.")

    return r

# =====================================================
# GROQ EXPLANATION
# =====================================================
def groq_explanation(df, decision, reasons=None):
    if groq_client is None:
        return "Groq AI not enabled."

    site = df.to_dict(orient="records")[0]

    prompt = f"""
You are an environmental clearance authority.

Site details:
{site}

Decision: {decision}

Reasons:
{reasons}

Explain clearly:
• Why this decision was taken
• Legal & environmental justification
• Long-term ecological impact
• Risks to water, wildlife, and people
• Sustainability recommendations
"""

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
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
page = st.sidebar.radio("Select Page", ["Single Site Input", "Result"])
use_ai = st.sidebar.checkbox("🤖 Enable AI Explanation", value=True)

# =====================================================
# PAGE: INPUT
# =====================================================
if page == "Single Site Input":
    st.title("📝 Mining Site Details")

    with st.form("form"):
        data = {
            "Region_ID": "R_NEW",
            "State": "Karnataka",
            "Mountain_Range": "Western Ghats",
            "Nearby_River": "Sharavathi",
            "Restriction_Type": "Wildlife Sanctuary",
            "Forest_Cover_Percent": 70,
            "Distance_to_River_km": 2.0,
            "Protected_Area": 1,
            "Deforestation_Risk": "High",
            "Water_Pollution_Risk": "High",
            "Air_Pollution_Risk": "Medium",
            "Seismic_Zone": "IV",
            "Slope_deg": 35,
            "Population_Density_per_km2": 1200
        }
        submit = st.form_submit_button("Save")

    if submit:
        st.session_state.input_df = pd.DataFrame([data])
        st.success("Saved. Go to Result page.")

# =====================================================
# PAGE: RESULT
# =====================================================
elif page == "Result":
    st.title("📊 Mining Decision")

    if st.session_state.input_df is None:
        st.warning("Enter site details first.")
    else:
        df = st.session_state.input_df
        df_proc = preprocess_input(df)

        if st.button("Run Prediction"):
            denial = rule_based_denial(df)

            if denial:
                st.error("❌ MINING DENIED")
                for r in denial:
                    st.write("•", r)

                if use_ai:
                    st.markdown("### 🤖 Groq AI Explanation")
                    st.write(groq_explanation(df, "DENIED", denial))

            else:
                pred = model.predict(df_proc)[0]
                if pred == 1:
                    st.success("✅ MINING APPROVED")
                    if use_ai:
                        st.write(groq_explanation(df, "APPROVED"))
                else:
                    st.error("❌ MINING DENIED (ML Risk)")
                    if use_ai:
                        st.write(
                            groq_explanation(
                                df,
                                "DENIED",
                                ["High predicted environmental risk"]
                            )
                        )
