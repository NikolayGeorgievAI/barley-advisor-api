import re
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Barley Advisor", page_icon="🌾", layout="centered")
st.title("🌾 Barley Advisor — Yield & Quality Predictor")
st.caption("Prototype on Teagasc data. For demo only.")

@st.cache_resource
def load_model():
    return joblib.load(Path("barley_model/model.pkl"))

model = load_model()

# ---------- helper: figure out expected columns ----------
def expected_columns_from_model(m):
    # 1) most sklearn estimators/pipelines store this when fit with a DataFrame
    cols = getattr(m, "feature_names_in_", None)
    if cols is not None:
        return list(cols)

    # 2) some nested pipelines store it on the final step
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                return list(cols)

    # 3) otherwise unknown; return empty and we’ll fall back to error parsing
    return []

EXPECTED = expected_columns_from_model(model)

# ---------- heuristics to decide numeric vs categorical ----------
NUMERIC_HINTS = ("_mm", "_c", "_kg_ha", "_prop", "_doy", "year", "gs", "tmax", "tmin", "srad")
def looks_numeric(name:str) -> bool:
    name_l = name.lower()
    return any(h in name_l for h in NUMERIC_HINTS)

# Preset options if we recognize common categoricals
KNOWN_OPTIONS = {
    "site": ["A","B","C","D"],      # replace with your real sites if you know them
    "block": ["1","2","3","4"],     # replace with your real blocks if you know them
    "end_use": ["malting","feed"],  # guess
    "source": ["trial","farm"],     # guess
}

# ---------- UI builder ----------
st.subheader("Inputs")

user_vals = {}
cols = st.columns(2)

def num_input(key, label, default):
    return cols[0 if num_input.idx % 2 == 0 else 1].number_input(label, value=default, key=key)
num_input.idx = 0

def cat_input(key, label, options=None, default=None):
    c = cols[0 if cat_input.idx % 2 == 0 else 1]
    if options:
        return c.selectbox(label, options=options, index=options.index(default) if default in options else 0, key=key)
    return c.text_input(label, value=default or "", key=key)
cat_input.idx = 0

# If we know exactly what columns are expected, build UI for them
if EXPECTED:
    for name in EXPECTED:
        if looks_numeric(name):
            # set sensible defaults
            d = 120.0
            if name.endswith("_prop"): d = 0.5
            if name in ("season_tmax_c",): d = 18.0
            if name in ("season_tmin_c",): d = 5.0
            if name in ("season_srad_mj_m2",): d = 500.0
            if name in ("year",): d = 2020
            if name.endswith("_gs"): d = 30.0
            if name.endswith("_kg_ha"): d = 120.0
            if name.endswith("_mm"): d = 120.0
            if name.endswith("_doy"): d = 120.0
            val = num_input(name, name.replace("_"," ").title(), float(d))
            user_vals[name] = float(val)
            num_input.idx += 1
        else:
            options = KNOWN_OPTIONS.get(name)
            default = (options[0] if options else "")
            val = cat_input(name, name.replace("_"," ").title(), options=options, default=default)
            user_vals[name] = val
            cat_input.idx += 1
else:
    st.info("The model didn’t expose `feature_names_in_`. Click Predict once to capture the error and I’ll infer the missing columns from it.")

# ---------- Predict ----------
if st.button("Predict"):
    def predict_with(df):
        return model.predict(df)[0]

    # If we have an expected list, use it in that exact order
    if EXPECTED:
        X = pd.DataFrame([[user_vals.get(c, np.nan) for c in EXPECTED]], columns=EXPECTED)
        try:
            y = predict_with(X)
            st.success(f"Predicted yield: **{float(y):.2f} t/ha**")
        except Exception as e:
            st.error("Prediction failed — see details below. I’ll try to infer any other required columns.")
            st.code(repr(e))
    else:
        # try with what we have to get the model’s exact “missing columns” message
        try:
            X = pd.DataFrame([user_vals])
            y = predict_with(X)
            st.success(f"Predicted yield: **{float(y):.2f} t/ha**")
        except Exception as e:
            st.error("Model expects specific columns. I parsed the error below:")
            msg = repr(e)
            st.code(msg)

            # Attempt to extract a set like: ("col1", "col2", ...)
            m = re.search(r"Columns are missing:\s*\(([^)]*)\)", msg)
            if m:
                raw = m.group(1)
                # split by quotes
                missing = [c.strip().strip("'").strip('"') for c in raw.split(",") if c.strip().strip("'").strip('"')]
                missing = [c for c in missing if c]  # clean
                st.warning("Add these to your UI (I can do this automatically next edit):")
                st.write(missing)
            else:
                st.info("Couldn’t parse missing columns list. Share the error above and I’ll wire them in.")
