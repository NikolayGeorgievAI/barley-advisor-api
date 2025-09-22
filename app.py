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

# === Replace with your actual features ===
col1, col2 = st.columns(2)
sowing_date   = col1.number_input("Sowing date (DOY)", min_value=1.0, max_value=366.0, value=120.0)
rainfall_mm   = col2.number_input("Total rainfall (mm)", min_value=0.0, max_value=800.0, value=120.0)
nitrogen_kg   = col1.number_input("Nitrogen (kg/ha)", min_value=0.0, max_value=350.0, value=120.0)
avg_temp_c    = col2.number_input("Average temp (°C)", min_value=-5.0, max_value=35.0, value=12.0)
# add any other features you had in training!

if st.button("Predict"):
    # Make sure the column names & order match training!
    X = pd.DataFrame([{
        "sowing_date": sowing_date,
        "rainfall_mm": rainfall_mm,
        "nitrogen_kg_ha": nitrogen_kg,
        "temp_c": avg_temp_c
    }])
    try:
        y_pred = model.predict(X)[0]
        st.success(f"Predicted yield: **{y_pred:.2f} t/ha**")
    except Exception as e:
        st.error("Prediction failed. Likely wrong feature list/order.")
        st.code(repr(e))
