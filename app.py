# app.py — predictor + rec engine + Azure OpenAI chatbot
import os
import requests
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
DEFAULT_SOURCE_DOC = "Focus2022"
DEFAULT_SITE_ID = "S1"
source_doc = DEFAULT_SOURCE_DOC
site_id = DEFAULT_SITE_ID

# Optional developer controls
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
    for k, v in GS_OPTIONS.items():
        if v == value:
            return k
    return value

# ---------------- Inputs ----------------
st.subheader("Inputs")
with st.form("inputs"):
    c1, c2 = st.columns(2)
    year = c1.number_input("Year", min_value=2000, max_value=2030, value=2020, step=1,
                           help="Trial year or season year.")
    end_use = c2.selectbox("End use", ["feed", "malting"],
                           help="Intended grain use category.")

    n_rate_kg_ha = c1.number_input("Nitrogen rate (kg/ha)", min_value=0, max_value=300, value=120, step=10,
                                   help="Total N applied across the season.")
    n_split_first_prop = c2.number_input("N split — first application proportion", min_value=0.0, max_value=1.0,
                                         value=0.5, step=0.05,
                                         help="Share of total N applied in the first split (0–1).")

    final_n_timing_label = c1.selectbox("Final N timing", list(GS_OPTIONS.values()), index=0,
                                        help="Growth stage when the last nitrogen was applied (Zadoks scale).")
    final_n_timing_gs = ui_label_to_gs(final_n_timing_label)

    submitted = st.form_submit_button("Predict")

# ---------------- Predict ----------------
pred_yield = None
pred_protein = None
if submitted:
    try:
        row = {
            "source_doc": source_doc,
            "year": year,
            "site_id": site_id,
            "end_use": end_use,
            "n_rate_kg_ha": float(n_rate_kg_ha),
            "n_split_first_prop": float(n_split_first_prop),
            "final_n_timing_gs": final_n_timing_gs,
        }
        X = pd.DataFrame([row])
        preds = model.predict(X)
        preds = np.array(preds).flatten()
        if len(preds) == 1:
            pred_yield = float(preds[0])
            st.success(f"🌾 Predicted yield: **{pred_yield:.2f} t/ha**")
        else:
            pred_yield = float(preds[0])
            pred_protein = float(preds[1])
            col_y, col_p = st.columns(2)
            col_y.success(f"🌾 Predicted yield: **{pred_yield:.2f} t/ha**")
            col_p.success(f"🧬 Predicted grain protein: **{pred_protein:.2f} %**")
        st.caption("Results are model estimates based on the inputs provided.")
    except Exception as e:
        st.error("Prediction failed.")
        st.code(repr(e))

# ---------------- Recommendation engine (rule-based core) ----------------
def recommend(end_use: str, n_rate: float, n_split_first: float, final_gs: str,
              y_pred: float | None, p_pred: float | None) -> str:
    """
    Simple transparent rules:
      - Malting target protein window ~9.5–11.0%
      - If protein > 11 -> suggest reducing N (and/or earlier timing)
      - If protein < 9.5 -> suggest increasing N (and/or later timing)
      - Feed: prioritize yield, protein less constrained
    Percent adjustments are heuristic; keep conservative (10–20%).
    """
    lines = []

    # Context header
    if y_pred is not None:
        lines.append(f"Predicted yield: {y_pred:.2f} t/ha.")
    if p_pred is not None:
        lines.append(f"Predicted protein: {p_pred:.2f}%.")

    if end_use == "malting" and p_pred is not None:
        tgt_low, tgt_high = 9.5, 11.0
        if p_pred > tgt_high:
            # protein too high for malting
            # proportional but bounded suggestion
            overshoot = min(max((p_pred - tgt_high) / 2.0, 0.05), 0.20)  # 5–20%
            kgha = int(round(n_rate * overshoot))
            lines.append(
                f"Protein is above typical malting targets ({tgt_low}–{tgt_high}%). "
                f"Consider **reducing total N by ~{int(overshoot*100)}% (~{kgha} kg/ha)**."
            )
            if final_gs in ["GS37", "GS61"]:
                lines.append("Also consider **earlier final N timing** (e.g., GS25–GS31) to mitigate late protein uplift.")
        elif p_pred < tgt_low:
            undershoot = min(max((tgt_low - p_pred) / 2.0, 0.05), 0.20)  # 5–20%
            kgha = int(round(n_rate * undershoot))
            lines.append(
                f"Protein is below typical malting targets. "
                f"Consider **increasing total N by ~{int(undershoot*100)}% (~{kgha} kg/ha)**."
            )
            if final_gs in ["sowing", "emergence", "GS25"]:
                lines.append("If quality allows, **slightly later final N timing** (e.g., GS31) can help lift protein.")
        else:
            lines.append("Protein is within the typical malting window — consider **fine-tuning N splits** rather than total N.")
            if 0.35 <= n_split_first <= 0.55:
                lines.append("Your first-split share looks balanced for malting (≈40–50%).")

    elif end_use == "feed":
        # No strict protein constraint; aim for yield with N efficiency
        if y_pred is not None and y_pred < 7.0 and n_rate < 150:
            lines.append("Yield is modest; consider **increasing N rate** toward 150–180 kg/ha if soil/residual N is low.")
        if final_gs in ["sowing", "emergence"]:
            lines.append("For feed, avoid placing too much N at sowing; ensure **adequate in-season splits** (e.g., GS25/GS31).")

    # Generic safety notes
    lines.append("Always adjust for local soil N supply, previous crop, and nitrate leaching limits.")
    return " ".join(lines)

if pred_yield is not None or pred_protein is not None:
    rec = recommend(end_use, float(n_rate_kg_ha), float(n_split_first_prop), final_n_timing_gs,
                    pred_yield, pred_protein)
    st.markdown("### 📋 Recommendation")
    st.info(rec)

# ---------------- Azure OpenAI Chatbot ----------------
def have_azure():
    try:
        _ = st.secrets["AZURE_OPENAI_ENDPOINT"]
        _ = st.secrets["AZURE_OPENAI_API_KEY"]
        _ = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
        return True
    except Exception:
        return False

def azure_chat(messages):
    """
    Minimal Azure OpenAI Chat Completions call.
    Expects secrets: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
    """
    endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_key = st.secrets["AZURE_OPENAI_API_KEY"]
    deployment = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview"

    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 500,
        "top_p": 0.9,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

st.markdown("---")
st.subheader("💬 Ask the advisor")

# Seed the chat with context (inputs + predictions) for better answers
context_blob = {
    "source_doc": source_doc,
    "site_id": site_id,
    "year": year,
    "end_use": end_use,
    "n_rate_kg_ha": float(n_rate_kg_ha),
    "n_split_first_prop": float(n_split_first_prop),
    "final_n_timing_gs": final_n_timing_gs,
    "predicted_yield_t_ha": pred_yield,
    "predicted_protein_pct": pred_protein,
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system",
         "content": (
             "You are an agronomy assistant for spring barley. "
             "Be practical, concise, and transparent about uncertainty. "
             "Targets: malting protein ~9.5–11.0%. "
             "If asked for N changes, give ballpark ranges (±10–20%) with caveats "
             "about soil N, regulations, and lodging risk."
         )},
        {"role": "system", "content": f"Context: {context_blob}"}
    ]

# show history
for m in st.session_state.chat_history:
    if m["role"] in ("user", "assistant"):
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["content"])

user_msg = st.chat_input("Ask a question (e.g., 'What if I reduce N by 15%?')")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    try:
        if have_azure():
            reply = azure_chat(st.session_state.chat_history)
        else:
            reply = ("(Azure OpenAI not configured) "
                     "Add AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT in Streamlit Secrets "
                     "to enable the chatbot.")
    except Exception as e:
        reply = f"Azure call failed: {e}"
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
