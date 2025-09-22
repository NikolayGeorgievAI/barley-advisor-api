import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------- UI BASICS ----------
st.set_page_config(page_title="Barley Advisor — Yield & Quality", page_icon="🌾", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0'>🌾 Barley Advisor — Yield & Quality</h1>"
    "<div style='color:#666;margin-top:4px'>Prototype model trained on Teagasc dataset. "
    "For demo only — not agronomic advice.</div>",
    unsafe_allow_html=True,
)

# ---------- MODEL LOADING ----------
MODEL_PATH = Path("barley_model/model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

model_error = None
model = None
try:
    model = load_model()
except Exception as e:
    model_error = str(e)

# Expected feature order used during training
FEATURE_ORDER = [
    "source_doc",
    "year",
    "site_id",
    "end_use",
    "n_rate_kg_ha",
    "n_split_first_prop",
    "final_n_timing_gs",
]

# For clean UX in the main column
col_main, = st.columns([1])

# ---------- INPUTS / PREDICTION ----------
with st.container():
    st.subheader("Inputs")

    # Keep “hidden” source_doc for consistency with trained pipeline (we set Focus2022)
    source_doc_default = "Focus2022"  # safest default from your inspection
    source_doc = source_doc_default

    # End-use choices (align with your encoder categories)
    END_USE = ["feed", "malting"]
    # GS milestones translated to friendly labels (keep original codes for model)
    GS_OPTIONS = {
        "GS25": "Tillering (GS25)",
        "GS31": "Stem elongation (GS31)",
        "GS55": "Ear half-emerged (GS55)",
        "GS57": "Ear three-quarters (GS57)",
        "GS61": "Flowering (GS61)",
        "emergence": "Emergence",
        "sowing": "Sowing",
    }
    gs_labels = [GS_OPTIONS[k] for k in GS_OPTIONS]

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", value=2020, min_value=2010, max_value=2035, step=1)
        n_rate = st.number_input("Nitrogen rate (kg/ha)", value=120.0, min_value=0.0, max_value=300.0, step=1.0)
        gs_label = st.selectbox("Final N timing", options=gs_labels, index=list(GS_OPTIONS.keys()).index("GS25"))
    with c2:
        end_use = st.selectbox("End use", END_USE, index=END_USE.index("feed"))
        n_split_first_prop = st.number_input("N split — first application proportion", value=0.50, min_value=0.0, max_value=1.0, step=0.05)

    # We hide site_id for users; pick a stable default used during training (e.g., “S1”/“A”/etc.)
    site_id = "S1"

    # Map label back to code
    inverse_gs = {v: k for k, v in GS_OPTIONS.items()}
    final_n_timing_gs = inverse_gs[gs_label]

    def make_input_df():
        row = {
            "source_doc": source_doc,
            "year": int(year),
            "site_id": site_id,
            "end_use": end_use,
            "n_rate_kg_ha": float(n_rate),
            "n_split_first_prop": float(n_split_first_prop),
            "final_n_timing_gs": final_n_timing_gs,
        }
        # Ensure exact column order the pipeline expects
        df = pd.DataFrame([{k: row[k] for k in FEATURE_ORDER}])
        return df

    pred_df = make_input_df()

    predict_btn = st.button("Predict", type="primary", use_container_width=False)

    last_prediction = None
    if predict_btn:
        if model_error:
            st.error(f"Model error: {model_error}")
        else:
            try:
