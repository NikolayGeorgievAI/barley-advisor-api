import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Barley Advisor", page_icon="🌾", layout="centered")
st.title("🌾 Barley Advisor — Yield & Quality Predictor")
st.caption("Prototype model trained on Teagasc dataset. For demo only — not agronomic advice.")

@st.cache_resource
def load_model():
    return joblib.load(Path("barley_model/model.pkl"))

model = load_model()

# Expected features from the trained pipeline
EXPECTED = getattr(model, "feature_names_in_", None)
if EXPECTED is None:
    st.error("Model does not expose expected features. Please update pipeline or UI.")
    st.stop()

# ---------- Input form ----------
st.subheader("Inputs")

user_vals = {}
cols = st.columns(2)

# Example mappings for categorical features (adapt these to your dataset)
KNOWN_OPTIONS = {
    "end_use": ["malting", "feed"],
    "source": ["trial", "farm"],
}

for i, feat in enumerate(EXPECTED):
    col = cols[i % 2]

    # numeric heuristic
    if any(h in feat.lower() for h in ["_mm", "_c", "_kg", "_doy", "year", "prop", "gs", "srad"]):
        default = 0.0
        if "year" in feat: default = 2020
        if "kg" in feat: default = 120
        if "mm" in feat: default = 120
        if "doy" in feat: default = 120
        if "c" in feat: default = 15
        if "prop" in feat: default = 0.5
        val = col.number_input(feat.replace("_"," ").title(), value=float(default))
        user_vals[feat] = val
    else:
        opts = KNOWN_OPTIONS.get(feat)
        if opts:
            val = col.selectbox(feat.replace("_"," ").title(), opts)
        else:
            val = col.text_input(feat.replace("_"," ").title())
        user_vals[feat] = val

# ---------- Prediction ----------
if st.button("Predict"):
    try:
        # build row in correct order
        clean_row = []
        for c in EXPECTED:
            v = user_vals.get(c)
            # flatten numpy / list
            if isinstance(v, (list, tuple, np.ndarray)):
                v = v[0]
            clean_row.append(v)

        X = pd.DataFrame([clean_row], columns=EXPECTED)

        y = model.predict(X)
        y = np.array(y).flatten()

        if len(y) == 1:
            st.success(f"Prediction: **{y[0]:.2f}**")
        elif len(y) >= 2:
            st.success(f"🌾 Predicted yield: **{y[0]:.2f} t/ha**")
            st.success(f"🧬 Predicted grain protein: **{y[1]:.2f} %**")

    except Exception as e:
        st.error("Prediction failed")
        st.code(repr(e))
