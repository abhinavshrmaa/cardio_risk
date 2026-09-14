"""
app.py
------
Streamlit app for a cardiovascular risk detector.

Run with:
    streamlit run app.py

Make sure you've run `python train_model.py` first so that model.pkl exists.
"""

import os
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cardiovascular Risk Detector", page_icon="🫀", layout="centered")

# -----------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# -----------------------------------------------------------------------
@st.cache_resource
def load_model():
    # On a fresh deploy, model.pkl won't exist yet — train it on first run
    # so you don't have to remember to do it manually on the server.
    if not os.path.exists("model.pkl"):
        with st.spinner("First-time setup: training model..."):
            subprocess.run(["python", "train_model.py"], check=True)
    return joblib.load("model.pkl")

@st.cache_resource
def load_background():
    # A sample of scaled training rows, used as the SHAP background distribution
    df = pd.read_csv("framingham.csv")[["male", "age", "currentSmoker", "BMI"]].dropna()
    return df.sample(min(200, len(df)), random_state=1)

model = load_model()
scaler = model.named_steps["scaler"]
clf = model.named_steps["clf"]
background = load_background()
background_scaled = scaler.transform(background)
explainer = shap.LinearExplainer(clf, background_scaled)

# -----------------------------------------------------------------------
# AQI lookup (Open-Meteo, free, no API key required)
# -----------------------------------------------------------------------
def fetch_aqi(city: str):
    """Returns (aqi_value, place_label) or (None, error_message)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=8,
        ).json()
        if not geo.get("results"):
            return None, "City not found."
        result = geo["results"][0]
        lat, lon = result["latitude"], result["longitude"]
        label = f"{result['name']}, {result.get('country', '')}"

        aq = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "current": "us_aqi"},
            timeout=8,
        ).json()
        aqi = aq.get("current", {}).get("us_aqi")
        if aqi is None:
            return None, "AQI unavailable for this location."
        return aqi, label
    except requests.RequestException as e:
        return None, f"Network error: {e}"


def aqi_risk_bump(aqi: float) -> float:
    """Heuristic extra risk (in probability points, 0-1 scale) added on
    top of the model's output for poor air quality. This is NOT learned
    from data — the Framingham dataset has no AQI column — it's a
    transparent, separately-labelled adjustment."""
    if aqi <= 50:
        return 0.00
    if aqi <= 100:
        return 0.01
    if aqi <= 150:
        return 0.03
    if aqi <= 200:
        return 0.05
    if aqi <= 300:
        return 0.08
    return 0.12


def risk_category(pct: float) -> str:
    if pct < 10:
        return "Low"
    if pct < 20:
        return "Moderate"
    if pct < 35:
        return "Elevated"
    return "High"


# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------
st.title("🫀 Cardiovascular Risk Detector")
st.caption(
    "A prototype 10-year coronary heart disease risk estimator, trained on the "
    "Framingham Heart Study dataset. **Not a diagnostic tool** — for educational "
    "and portfolio purposes only."
)

with st.form("risk_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    with col2:
        height_cm = st.number_input("Height (cm)", min_value=100, max_value=230, value=170)
        weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)

    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.caption(f"Calculated BMI: **{bmi:.1f}**")

    smoker = st.radio("Smoking status", ["Non-smoker", "Smoker"], horizontal=True)

    city = st.text_input("City (optional — used to look up local AQI)", "")

    submitted = st.form_submit_button("Run assessment")

if submitted:
    # ---- Build model input -------------------------------------------------
    sample = pd.DataFrame([{
        "male": 1 if gender == "Male" else 0,
        "age": age,
        "currentSmoker": 1 if smoker == "Smoker" else 0,
        "BMI": bmi,
    }])

    model_proba = model.predict_proba(sample)[:, 1][0]

    # ---- Optional AQI lookup and adjustment --------------------------------
    aqi_value, aqi_label = (None, None)
    bump = 0.0
    if city.strip():
        with st.spinner("Looking up air quality..."):
            aqi_value, aqi_label = fetch_aqi(city.strip())
        if aqi_value is not None:
            bump = aqi_risk_bump(aqi_value)
            st.info(f"📍 {aqi_label} — US AQI **{aqi_value}**")
        else:
            st.warning(f"Couldn't fetch AQI ({aqi_label}). Continuing without it.")

    final_proba = min(1.0, model_proba + bump)
    pct = final_proba * 100
    category = risk_category(pct)

    # ---- Display result ------------------------------------------------
    st.subheader("Result")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Estimated 10-year CHD risk", f"{pct:.1f}%")
        st.write(f"**Category: {category}**")
    with c2:
        st.progress(min(1.0, final_proba))
        if bump > 0:
            st.caption(
                f"Includes a +{bump*100:.1f} point air-quality adjustment "
                "(heuristic, not model-learned — see note below)."
            )

    # ---- SHAP explanation -----------------------------------------------
    st.subheader("Why this score? (model contribution per factor)")
    sample_scaled = scaler.transform(sample)
    shap_values = explainer(sample_scaled)
    shap_values.feature_names = ["Male", "Age", "Smoker", "BMI"]

    fig, ax = plt.subplots(figsize=(6, 3))
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Bars show how much each input pushed the model's *raw score* up or "
        "down relative to the average patient in the training data, before "
        "the air-quality adjustment above."
    )

    st.divider()
    st.caption(
        "⚠️ **Educational prototype only.** This model is trained on the "
        "Framingham Heart Study dataset with a small feature set (sex, age, "
        "smoking, BMI) and is not a validated clinical risk score such as "
        "Framingham's own multi-variable equation or ASCVD. It does not "
        "constitute medical advice — consult a physician for an actual "
        "cardiovascular risk assessment. The air-quality adjustment is a "
        "simple heuristic, not something the model learned from data."
    )
