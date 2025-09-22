# app.py (polished)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any

# --------------------- Page config ---------------------
st.set_page_config(page_title="Barley Advisor", page_icon="🌾", layout="centered")

st.markdown(
    """
    <style>
      .small-note {font-size:0.9rem;color:#6b7280}
      .metric {font-size:1.15rem}
      .section {margin-top:1rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌾 Barley Advisor — Yield & Quality Predictor")
st.markdown(
    '<p class="small-note">Prototype model trained on Teagasc dataset. '
    'For demo only — not agronomic advice.</p>',
    unsafe_allow_html=True,
)

# --------------------- Load model ----------------------
@st.cache_resource
def load_model():
    return joblib.load(Path("barley_model/model.pkl"))

model = load_model()

# --------------------- Discover expected features ----------------------
EXPECTED = getattr(model, "feature_names_in_", None)
if EXPECTED is None and hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        cols = getattr(step, "feature_names_in_", None)
        if cols is not None:
            EXPECTED = cols
            break

if EXPECTED is None or len(EXPECTED) == 0:
    st.error(
        "Model does not expose expected input feature names. "
        "Please refit on a pandas DataFrame or provide the column list."
    )
    st.stop()

EXPECTED = list(EXPECTED)

# --------------- Pull categorical choices from encoders ----------------
def discover_categorical_options(m) -> Dict[str, List[Any]]:
    options: Dict[str, List[Any]] = {}

    def to_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return [x]

    transformers = []
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "transformers_"):
                transformers = step.transformers_
                break
    elif hasattr(m, "transformers_"):
        transformers = m.transformers_

    for _, transformer, cols in transformers:
        if transformer in ("drop", "passthrough") or transformer is None:
            continue
        inner = transformer
        if hasattr(transformer, "named_steps"):
            inner = list(transformer.named_steps.values())[-1]
        if inner.__class__.__name__ == "OneHotEncoder" and hasattr(inner, "categories_"):
            cats = inner.categories_
            cols_list = to_list(cols)
            if len(cols_list) == len(cats):
                for feat, cat_list in zip(cols_list, cats):
                    options[str(feat)] = [
                        c.item() if hasattr(c, "item") else c for c in list(cat_list)
                    ]
    return options

CAT_OPTS = discover_categorical_options(model)

# ------------------ Friendly labels & tooltips -------------------
LABELS = {
    "source_doc": "Source document",
    "site_id": "Site ID",
    "block": "Trial block",
    "end_use": "End use",
    "n_rate_kg_ha": "Nitrogen rate (kg/ha)",
    "n_split_first_prop": "N split — first application proportion",
    "n_timing_gs": "N timing (growth stage)",
    "final_n_timing_gs": "Final N timing (growth stage)",
    "sowing_doy": "Sowing day of year",
    "season_rain_mm": "Growing season rainfall (mm)",
    "season_tmax_c": "Growing season max temp (°C)",
    "season_tmin_c": "Growing season min temp (°C)",
    "season_srad_mj_m2": "Growing season solar radiation (MJ/m²)",
    "year": "Year",
}

TOOLTIPS = {
    "source_doc": "Trial/source identifier from the dataset.",
    "site_id": "Location code of the trial site.",
    "block": "Experimental block within the site (randomization).",
    "end_use": "Intended grain use (e.g., malting/feed).",
    "n_rate_kg_ha": "Total nitrogen applied across the season.",
    "n_split_first_prop": "Share of N applied in the first split (0–1).",
    "n_timing_gs": "Growth stage of the main N application.",
    "final_n_timing_gs": "Growth stage of the final N application.",
    "sowing_doy": "Calendar day of sowing (1–366).",
    "season_rain_mm": "Cumulative rainfall during the growing season.",
    "season_tmax_c": "Average daily maximum temperature during season.",
    "season_tmin_c": "Average daily minimum temperature during season.",
    "season_srad_mj_m2": "Cumulative solar radiation during season.",
    "year": "Trial year.",
}

NUMERIC_HINTS = ("_mm", "_c", "_kg", "_kg_ha", "_doy", "year", "prop", "gs", "srad", "rate")
def looks_numeric(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in NUMERIC_HINTS)

def numeric_default_for(feat: str) -> float:
    f = feat.lower()
    if "year" in f: return 2020.0
    if "kg_ha" in f or f.endswith("_kg") or "rate" in f: return 120.0
    if f.endswith("_mm"): return 120.0
    if f.endswith("_doy"): return 120.0
    if "tmax" in f and f.endswith("_c"): return 18.0
    if "tmin" in f and f.endswith("_c"): return 5.0
    if "srad" in f: return 500.0
    if f.endswith("prop"): return 0.5
    if f.endswith("gs"): return 30.0
    return 0.0

# -------------------------- Form UI --------------------------
st.subheader("Inputs")

# Group: management & trial meta vs climate
mgmt_feats = [f for f in EXPECTED if any(k in f for k in ["n_", "sowing", "end_use", "block", "site", "source", "year"])]
climate_feats = [f for f in EXPECTED if f not in mgmt_feats]

user_vals: Dict[str, Any] = {}

with st.form("input_form"):
    # Management / trial
    st.markdown("**Management & Trial**")
    cols = st.columns(2)
    for i, feat in enumerate(mgmt_feats):
        col = cols[i % 2]
        label = LABELS.get(feat, feat.replace("_", " ").title())
        help_ = TOOLTIPS.get(feat, None)
        if feat in CAT_OPTS and len(CAT_OPTS[feat]) > 0:
            user_vals[feat] = col.selectbox(label, options=CAT_OPTS[feat], help=help_)
        else:
            if looks_numeric(feat):
                user_vals[feat] = col.number_input(label, value=float(numeric_default_for(feat)), help=help_)
            else:
                user_vals[feat] = col.text_input(label, value="", help=help_)

    # Climate / season
    if climate_feats:
        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.markdown("**Climate (Growing Season)**")
        cols = st.columns(2)
        for i, feat in enumerate(climate_feats):
            col = cols[i % 2]
            label = LABELS.get(feat, feat.replace("_", " ").title())
            help_ = TOOLTIPS.get(feat, None)
            if looks_numeric(feat):
                user_vals[feat] = col.number_input(label, value=float(numeric_default_for(feat)), help=help_)
            else:
                # Rare case: climate feature is categorical
                if feat in CAT_OPTS and len(CAT_OPTS[feat]) > 0:
                    user_vals[feat] = col.selectbox(label, options=CAT_OPTS[feat], help=help_)
                else:
                    user_vals[feat] = col.text_input(label, value="", help=help_)

    submitted = st.form_submit_button("Predict")

# ----------------------- Predict & show -----------------------
if submitted:
    try:
        row = []
        for feat in EXPECTED:
            v = user_vals.get(feat)
            if isinstance(v, (list, tuple, np.ndarray)):
                v = v[0] if len(v) else None
            row.append(v)

        X = pd.DataFrame([row], columns=EXPECTED)
        preds = model.predict(X)
        preds = np.array(preds).flatten()

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        if len(preds) == 1:
            st.success(f"Prediction: **{preds[0]:.2f}**")
        else:
            col_y, col_p = st.columns(2)
            col_y.success(f"🌾 Predicted yield: **{preds[0]:.2f} t/ha**")
            col_p.success(f"🧬 Predicted grain protein: **{preds[1]:.2f} %**")

        st.markdown('<p class="small-note">Results are model estimates based on inputs provided.</p>', unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.code(repr(e))

# -------------------- Optional debug panel -------------------
with st.expander("Debug (optional)"):
    st.write("Expected feature order:", EXPECTED)
    if CAT_OPTS:
        st.write("Categorical options discovered from OneHotEncoder:")
        st.json(CAT_OPTS)
