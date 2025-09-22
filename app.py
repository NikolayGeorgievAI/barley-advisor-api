import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load("barley_model/model.pkl")

st.set_page_config(page_title="Barley Advisor", page_icon="🌾")
st.title("🌾 Barley Advisor — Yield & Quality Predictor")
st.caption("Prototype model trained on Teagasc dataset. For demo only — not agronomic advice.")

st.header("Inputs")

# Hidden field: default source_doc
source_doc = "Focus2022"

year = st.number_input("Year", min_value=2000, max_value=2030, value=2020, step=1)
site_id = st.selectbox("Site ID", ["S1", "S2", "S3", "S4", "S5", "S6", "T1", "T2", "T3"])
end_use = st.selectbox("End Use", ["feed", "malting"])
n_rate_kg_ha = st.number_input("N Rate (kg/ha)", min_value=0, max_value=300, value=120, step=10)
n_split_first_prop = st.number_input("N Split First Prop", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
final_n_timing_gs = st.selectbox("Final N Timing (GS)", ["GS25", "GS31", "GS37", "GS61", "emergence", "sowing"])

if st.button("Predict"):
    # Build input DataFrame (with hidden source_doc)
    input_df = pd.DataFrame([{
        "source_doc": source_doc,
        "year": year,
        "site_id": site_id,
        "end_use": end_use,
        "n_rate_kg_ha": n_rate_kg_ha,
        "n_split_first_prop": n_split_first_prop,
        "final_n_timing_gs": final_n_timing_gs
    }])

    # Run prediction
    pred = model.predict(input_df)[0]

    # Show results
    st.success(f"🌱 Predicted yield: {pred[0]:.2f} t/ha")
    st.info(f"💡 Predicted grain protein: {pred[1]:.2f} %")
