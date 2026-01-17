# -*- coding: utf-8 -*-
"""
AI-Assisted Mining Feasibility System – 2025
Policy + Environmental + AI Justification
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Mining Assessment 2025",
    page_icon="⛏️",
    layout="centered"
)

# =====================================================
# LOAD TRAINED ASSETS
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_list.pkl")
    return model, scaler, features

model, scaler, FEATURES = load_assets()

# =====================================================
# GROQ CLIENT
# =====================================================
@st.cache_resource
def load_groq():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)

groq = load_groq()

# =====================================================
# RISK + DECISION LOGIC (UNCHANGED)
# =====================================================
risk_mapping = {'Low': 2, 'Medium': 5, 'High': 8, 'Very High': 10}
seismic_mapping = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}

def to_binary(v):
    return 1 if str(v).lower() in ['1','yes','true'] else 0

def evaluate_and_impact(row):
    reasons, impacts = [], []

    legal_flag = row["Protected_Area"] == 1
    risk_flag = row["Environmental_Risk_Index"] > 0.65
    safety_flag = row["Slope_deg"] > 35 and row["Seismic_Zone"] >= 4
    policy_flag = row["AI_Predicted_Allowed"] == 0

    if legal_flag:
        reasons.append("Inviolate protected / sanctuary zone.")
        impacts.append("Permanent biodiversity loss.")

    if risk_flag:
        reasons.append("Environmental risk index exceeds 0.65.")
        impacts.append("Water, forest, and soil degradation likely.")

    if safety_flag:
        reasons.append("High landslide risk due to slope & seismicity.")
        impacts.append("Threat to nearby human settlements.")

    if policy_flag:
        reasons.append("AI sustainability policy violation.")
        impacts.append("Conflicts with national environmental planning.")

    if legal_flag:
        status = "DENIED (Legal)"
    elif risk_flag:
        status = "REJECTED (High Risk)"
    elif safety_flag:
        status = "REJECTED (Safety)"
    elif policy_flag:
        status = "DENIED (Policy)"
    else:
        status = "APPROVED"
        reasons.append("All regulatory thresholds satisfied.")
        impacts.append("Mitigation plans mandatory.")

    return status, "; ".join(reasons), "; ".join(impacts)

# =====================================================
# UI – INPUT
# =====================================================
st.title("⛏️ Mining Feasibility Assessment")

data = {
    "Elevation_m": st.number_input("Elevation (m)", 50, 5000, 600),
    "Slope_deg": st.slider("Slope (°)", 0, 60, 38),
    "Forest_Cover_Percent": st.slider("Forest Cover (%)", 0, 100, 72),
    "Protected_Area": st.selectbox("Protected Area", [0, 1]),
    "Annual_Rainfall_mm": st.number_input("Rainfall (mm)", 500, 5000, 2800),
    "Seismic_Zone": seismic_mapping[
        st.selectbox("Seismic Zone", ["I","II","III","IV","V"])
    ],
    "Population_Density_per_km2": st.number_input("Population Density", 10, 5000, 1400),
    "Distance_to_River_km": st.number_input("Distance to River (km)", 0.1, 50.0, 1.8),
    "Distance_to_Road_km": st.number_input("Distance to Road (km)", 0.1, 100.0, 5.0),
    "NDVI": st.slider("NDVI", 0.0, 1.0, 0.42),

    "Deforestation_Risk": risk_mapping[
        st.selectbox("Deforestation Risk", risk_mapping.keys())
    ],
    "Water_Pollution_Risk": risk_mapping[
        st.selectbox("Water Pollution Risk", risk_mapping.keys())
    ],
    "Air_Pollution_Risk": risk_mapping[
        st.selectbox("Air Pollution Risk", risk_mapping.keys())
    ],
}

# =====================================================
# RUN ASSESSMENT
# =====================================================
if st.button("Run Assessment"):
    df = pd.DataFrame([data])

    df["Environmental_Risk_Index"] = (
        df[["Deforestation_Risk","Water_Pollution_Risk","Air_Pollution_Risk"]]
        .mean(axis=1) / 10
    )

    X = scaler.transform(df[FEATURES])
    df["AI_Predicted_Allowed"] = model.predict(X)

    status, justification, impacts = evaluate_and_impact(df.iloc[0])

    st.subheader(f"📌 Final Decision: {status}")
    st.write("### 🧾 Justification")
    st.write(justification)
    st.write("### 🌍 Environmental Impact")
    st.write(impacts)

    if groq:
        st.subheader("🤖 Groq AI Expert Opinion")
        st.write(
            groq.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{
                    "role": "user",
                    "content": f"""
Decision: {status}
Justification: {justification}
Impacts: {impacts}

Explain scientifically and legally.
"""
                }]
            ).choices[0].message.content
        )
