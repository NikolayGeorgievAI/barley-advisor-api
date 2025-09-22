# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any

st.set_page_config(page_title="Barley Advisor", page_icon="🌾", layout="centered")
st.title("🌾 Barley Advisor — Yield & Quality Predictor")
st.caption("Prototype model trained on Teagasc dataset. For demo only — not agronomic advice.")

# ---------------- Model loader ----------------
@st.cache_resource
def load_model():
    return joblib.load(Path("barley_model/model.pkl"))

model = load_model()

# ---------------- Discover expected features ----------------
EXPECTED = getattr(model, "feature_names_in_", None)

if EXPECTED is None:
    # Try to find feature_names_in_ on a step inside a pipeline
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                EXPECTED = cols
                break

# Safe emptiness check (works for list or numpy array)
if EXPECTED is None or len(EXPECTED) == 0:
    st.error(
        "Model does not expose expected input feature names. "
        "Refit on a pandas DataFrame or provide the column list."
    )
    st.stop()

# Ensure plain Python list
EXPECTED = list(EXPECTED)

# ---------------- Introspect categories from encoders ----------------
def discover_categorical_options(m) -> Dict[str, List[Any]]:
    """
    Walk a typical sklearn pipeline with a ColumnTransformer -> OneHotEncoder
    and return {feature_name: [allowed_categories...]} for categoricals.
    """
    options: Dict[str, List[Any]] = {}

    def to_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return [x]

    transformers = []
    # Pull out a ColumnTransformer if present
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "transformers_"):
                transformers = step.transformers_
                break
    elif hasattr(m, "transformers_"):
        transformers = m.transformers_

    # Each entry like ('cat', OneHotEncoder(...), [col1, col2, ...])
    for _, transformer, cols in transformers:
        if transformer in ("drop", "passthrough") or transformer is None:
            continue

        # If inner pipeline, take last step
        inner = transformer
        if hasattr(transformer, "named_steps"):
            inner = list(transformer.named_steps.values())[-1]

        if inner.__class__.__name__ == "OneHotEncoder" and hasattr(inner, "categories_"):
            cats = inner.categories_
            cols_list = to_list(cols)
            if len(cols_list) == len(cats):
                for feat, cat_list in zip(cols_list, cats):
                    clean = []
                    for c in cat_list:
                        # convert numpy scalars to python types
                        if hasattr(c, "item"):
                            clean.append(c.item())
                        else:
                            clean.append(c)
                    options[str(feat)] = clean

    return options

CATEGORICAL_OPTIONS = discover_categorical_options(model)

# ---------------- Heuristics for numeric vs categorical ----------------
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

# ---------------- Build the input form ----------------
st.subheader("Inputs")

user_vals: Dict[str, Any] = {}
cols = st.columns(2)

for i, feat in enumerate(EXPECTED):
    col = cols[i % 2]
    label = feat.replace("_", " ").title()

    if feat in CATEGORICAL_OPTIONS and len(CATEGORICAL_OPTIONS[feat]) > 0:
        user_vals[feat] = col.selectbox(label, options=CATEGORICAL_OPTIONS[feat])
    else:
        # Fallback: numeric vs text guess
        if looks_numeric(feat):
            user_vals[feat] = col.number_input(label, value=float(numeric_default_for(feat)))
        else:
            user_vals[feat] = col.text_input(label, "")

# ---------------- Predict ----------------
if st.button("Predict"):
    try:
        # Build single-row DataFrame in exact expected order
        row = []
        for feat in EXPECTED:
            v = user_vals.get(feat)
            # Flatten arrays/lists just in case
            if isinstance(v, (list, tuple, np.ndarray)):
                v = v[0] if len(v) else None
            row.append(v)

        X = pd.DataFrame([row], columns=EXPECTED)

        y = model.predict(X)
        y = np.array(y).flatten()  # single- or multi-target

        if len(y) == 1:
            st.success(f"Prediction: **{y[0]:.2f}**")
        else:
            # Rename these to your actual target names if you like
            st.success(f"🌾 Predicted yield: **{y[0]:.2f} t/ha**")
            st.success(f"🧬 Predicted grain protein: **{y[1]:.2f} %**")

    except Exception as e:
        st.error("Prediction failed.")
        st.code(repr(e))

# ---------------- Optional debug toggle ----------------
with st.expander("Debug (optional)"):
    st.write("Expected feature order:", EXPECTED)
    if CATEGORICAL_OPTIONS:
        st.write("Categorical options discovered from model encoders:")
        st.json(CATEGORICAL_OPTIONS)
