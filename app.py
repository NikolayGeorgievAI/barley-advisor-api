# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------- Page setup ----------------
st.set_page_config(page_title="Barley Advisor", page_icon="🌾", layout="centered")
st.title("🌾 Barley Advisor — Yield & Quality Predictor")
st.caption("Prototype model trained on Teagasc dataset. For demo only — not agronomic advice.")

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    return joblib.load(Path("barley_model/model.pkl"))

model = load_model()

# ---------------- Hidden defaults ----------------
# These match categories learned by your model (from your notebook output)
DEFAULT_SOURCE_DOC = "Focus2022"
DEFAULT_SITE_ID = "S1"

source_doc = DEFAULT_SOURCE_DOC
site_id = DEFAULT_SITE_ID

# Optional developer controls (kept out of the way for users)
with st.sidebar.expander("⚙️ Developer settings", expanded=False):
    dev_toggle = st.checkbox("Enable overrides", value=False, help="For testing only.")
    if dev_toggle:
        source_doc = st.selectbox(
            "Source document",
            ["Focus2022", "Hackett2011_i4", "Hackett2010_i9"],
            index=0,
            help="Internal dataset identifier used during training."
        )
        site_id = st.selectbox(
            "Site ID",
            ["S1", "S2", "S3", "S4", "S5", "S6", "T1", "T2", "T3"],
            index=0,
            help="Trial site code from the Teagasc dataset."
        )

# ---------------- Friendly GS mapping ----------------
GS_OPTIONS = {
    "GS25": "Tillering (GS25)",
    "GS31": "Stem elongation (GS31)",
    "GS37": "Flag leaf visible (GS37)",
    "GS61": "Flowering (GS61)",
    "emergence": "Emergence",
    "sowing": "At sowing",
}

def ui_label_to_gs(value: str) -> str:
    """Map UI-friendly label back to model code."""
    for k, v in GS_OPTIONS.items():
        if v == value:
            return k
    # fallback if already a code
    return value

# ---------------- Input form ----------------
st.subheader("Inputs")

with st.form("inputs"):
    col1, col2 = st.columns(2)

    year = col1.number_input(
        "Year", min_value=2000, max_value=2030, value=2020, step=1,
        help="Trial year or season year."
    )

    end_use = col2.selectbox(
        "End use", ["feed", "malting"],
        help="Intended grain use category."
    )

    n_rate_kg_ha = col1.number_input(
        "Nitrogen rate (kg/ha)", min_value=0, max_value=300, value=120, step=10,
        help="Total N applied across the season."
    )

    n_split_first_prop = col2.number_input(
        "N split — first application proportion", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Share of total N applied in the first split (0–1)."
    )

    final_n_timing_label = col1.selectbox(
        "Final N timing",
        list(GS_OPTIONS.values()),
        index=0,
        help="Growth stage when the last nitrogen was applied (Zadoks scale)."
    )
    final_n_timing_gs = ui_label_to_gs(final_n_timing_label)

    submitted = st.form_submit_button("Predict")

# ---------------- Predict ----------------
if submitted:
    try:
        # Build input row in the exact order your model expects
        row = {
            "source_doc": source_doc,                 # hidden default (or dev override)
            "year": year,
            "site_id": site_id,                       # hidden default (or dev override)
            "end_use": end_use,
            "n_rate_kg_ha": float(n_rate_kg_ha),
            "n_split_first_prop": float(n_split_first_prop),
            "final_n_timing_gs": final_n_timing_gs,   # mapped back to model code
        }

        X = pd.DataFrame([row])

        preds = model.predict(X)
        preds = np.array(preds).flatten()

        if len(preds) == 1:
            st.success(f"Prediction: **{preds[0]:.2f}**")
        else:
            col_y, col_p = st.columns(2)
            col_y.success(f"🌾 Predicted yield: **{preds[0]:.2f} t/ha**")
            col_p.success(f"🧬 Predicted grain protein: **{preds[1]:.2f} %**")

        st.caption("Results are model estimates based on the inputs provided.")

    except Exception as e:
        st.error("Prediction failed.")
        st.code(repr(e))
