import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Barley Advisor", page_icon="🌾", layout="centered")
st.title("🌾 Barley Advisor — Yield & Quality Predictor")

@st.cache_resource
def load_model():
    return joblib.load(Path("barley_model/model.pkl"))

model = load_model()

st.caption("Prototype model trained on Teagasc dataset. For demo only — not agronomic advice.")

# === Inputs matching training columns ===
col1, col2 = st.columns(2)

final_n_timing_gs = col1.number_input("Final N timing (GS)", min_value=0, max_value=200, value=39)
year              = col2.number_input("Year", min_value=2000, max_value=2100, value=2020)
n_rate_kg_ha      = col1.number_input("Nitrogen rate (kg/ha)", min_value=0.0, max_value=350.0, value=120.0)

site              = col2.selectbox("Site", options=["A", "B", "C", "D"])  # adapt to your dataset categories
block             = col1.selectbox("Block", options=["1", "2", "3", "4"]) # adapt

n_timing_gs       = col2.number_input("N timing (GS)", min_value=0, max_value=200, value=30)

season_rain_mm    = col1.number_input("Season rainfall (mm)", min_value=0.0, max_value=1000.0, value=120.0)
season_tmax_c     = col2.number_input("Season Tmax (°C)", min_value=-5.0, max_value=40.0, value=18.0)
season_tmin_c     = col1.number_input("Season Tmin (°C)", min_value=-10.0, max_value=25.0, value=5.0)
season_srad_mj_m2 = col2.number_input("Season solar rad (MJ/m²)", min_value=0.0, max_value=2000.0, value=500.0)

sowing_doy        = st.number_input("Sowing date (DOY)", min_value=1, max_value=366, value=120)

if st.button("Predict"):
    X = pd.DataFrame([{
        "final_n_timing_gs": final_n_timing_gs,
        "year": year,
        "n_rate_kg_ha": n_rate_kg_ha,
        "site": site,
        "block": block,
        "n_timing_gs": n_timing_gs,
        "season_rain_mm": season_rain_mm,
        "season_tmax_c": season_tmax_c,
        "season_tmin_c": season_tmin_c,
        "season_srad_mj_m2": season_srad_mj_m2,
        "sowing_doy": sowing_doy
    }])

    try:
        y_pred = model.predict(X)[0]
        st.success(f"Predicted yield: **{y_pred:.2f} t/ha**")
    except Exception as e:
        st.error("Prediction failed — check feature inputs or categories.")
        st.code(repr(e))
