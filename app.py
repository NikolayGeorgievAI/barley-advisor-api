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

st.write("**barley**
