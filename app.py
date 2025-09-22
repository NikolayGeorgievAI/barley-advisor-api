import streamlit as st
from pathlib import Path
import sys, os, platform

st.set_page_config(page_title="Barley Advisor (diag)", layout="centered")
st.title("🌾 Barley Advisor — diagnostic")

# --- Basic render proof ---
st.success("UI is alive ✅")

# --- Repo / files ---
cwd = Path(".").resolve()
st.write("**Working directory:**", str(cwd))
st.write("**Top-level files:**", sorted([p.name for p in cwd.iterdir()]))

model_pkl = Path("barley_model/model.pkl")
model_skops = Path("barley_model/model.skops")

st.write("**barley_model folder exists?**", model_pkl.parent.exists())
st.write("**model.pkl exists?**", model_pkl.exists(), f"({model_pkl})")
st.write("**model.skops exists?**", model_skops.exists(), f"({model_skops})")

# --- Versions ---
import pandas, numpy, joblib
try:
    import sklearn
except Exception:
    sklearn = None

st.write({
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "pandas": pandas.__version__,
    "numpy": numpy.__version__,
    "joblib": joblib.__version__,
    "sklearn": getattr(sklearn, "__version__", "NOT INSTALLED"),
})

# --- Try to load model (joblib first, then skops) ---
model = None
load_errors = []

if model_pkl.exists():
    try:
        import joblib
        model = joblib.load(model_pkl)
        st.success("Loaded model.pkl with joblib ✅")
    except Exception as e:
        load_errors.append(("joblib/pkl", repr(e)))

if model is None and model_skops.exists():
    try:
        import skops.io as sio
        model = sio.load(model_skops, trusted=True)
        st.success("Loaded model.skops with skops ✅")
    except Exception as e:
        load_errors.append(("skops/skops", repr(e)))

if model is None:
    if load_errors:
        st.error("Model failed to load. Details below:")
        for where, err in load_errors:
            st.code(f"{where}: {err}")
        st.info("Fix: pin exact sklearn/numpy versions in requirements.txt OR export with skops.")
    else:
        st.warning("No model file found. Commit barley_model/model.pkl (or .skops) to the repo.")

# --- Minimal UI so page is not blank ---
with st.expander("Example input form (placeholder)"):
    sowing_date = st.number_input("Sowing date (DOY)", 1.0, 366.0, 120.0)
    rainfall_mm = st.number_input("Total rainfall (mm)", 0.0, 800.0, 120.0)
    if st.button("Predict (if model loaded)"):
        if model is None:
            st.error("No model loaded.")
        else:
            import pandas as pd
            X = pd.DataFrame([{"sowing_date": sowing_date, "rainfall_mm": rainfall_mm}])
            try:
                y = model.predict(X)[0]
                st.success(f"Predicted: {y}")
            except Exception as e:
                st.error("Prediction failed (likely wrong feature names/order).")
                st.code(repr(e))
