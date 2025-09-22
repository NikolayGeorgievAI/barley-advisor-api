# app.py — Barley Advisor with predictions, recommendations, and Azure OpenAI chatbot
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------- Optional: Azure OpenAI client --------
AZURE_OK = False
try:
    from openai import AzureOpenAI
    if "azure" in st.secrets:
        AZURE_OK = all(k in st.secrets["azure"] for k in ("api_key", "endpoint", "deployment", "api_version"))
except Exception:
    AZURE_OK = False

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

# Optional developer controls (for your testing only)
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
    n_split_first_prop = c2.number_input("N split — first application proportion",
                                         min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                         help="Share of total N applied in the first split (0–1).")

    final_n_timing_label = c1.selectbox("Final N timing", list(GS_OPTIONS.values()), index=0,
                                        help="Growth stage when the last nitrogen was applied (Zadoks scale).")
    final_n_timing_gs = ui_label_to_gs(final_n_timing_label)

    submitted = st.form_submit_button("Predict")

# ---------------- Predict ----------------
pred_yield = None
pred_protein = None
input_row = None

if submitted:
    try:
        input_row = {
            "source_doc": source_doc,
            "year": year,
            "site_id": site_id,
            "end_use": end_use,
            "n_rate_kg_ha": float(n_rate_kg_ha),
            "n_split_first_prop": float(n_split_first_prop),
            "final_n_timing_gs": final_n_timing_gs,
        }
        X = pd.DataFrame([input_row])
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

# ---------------- Recommendation engine ----------------
def simple_recommendation(end_use: str, n_rate: float, n_split_first: float, final_gs: str,
                          y_pred: float | None, p_pred: float | None) -> str:
    """
    Transparent heuristics:
      - Malting target protein window ~9.5–11.0%.
      - If protein > 11: suggest reducing N by 5–20% and/or earlier final timing.
      - If protein < 9.5: suggest increasing N by 5–20% and/or slightly later timing.
      - Feed: prioritize yield and efficiency.
    """
    notes = []

    if y_pred is not None:
        notes.append(f"Predicted yield: {y_pred:.2f} t/ha.")
    if p_pred is not None:
        notes.append(f"Predicted protein: {p_pred:.2f}%.")

    if end_use == "malting" and p_pred is not None:
        lo, hi = 9.5, 11.0
        if p_pred > hi:
            overshoot = min(max((p_pred - hi) / 2.0, 0.05), 0.20)  # 5–20%
            kgha = int(round(n_rate * overshoot))
            notes.append(
                f"Protein is above malting target ({lo}–{hi}%). "
                f"Consider **reducing total N by ~{int(overshoot*100)}% (~{kgha} kg/ha)**."
            )
            if final_gs in ["GS37", "GS61"]:
                notes.append("Also consider **earlier final N timing** (e.g., GS25–GS31) to reduce late protein uplift.")
        elif p_pred < lo:
            undershoot = min(max((lo - p_pred) / 2.0, 0.05), 0.20)  # 5–20%
            kgha = int(round(n_rate * undershoot))
            notes.append(
                f"Protein is below malting target. Consider **increasing total N by ~{int(undershoot*100)}% (~{kgha} kg/ha)**."
            )
            if final_gs in ["sowing", "emergence", "GS25"]:
                notes.append("A **slightly later final N timing** (e.g., GS31) can help lift protein.")
        else:
            notes.append("Protein is within the typical malting window — consider **fine-tuning splits** rather than total N.")
            if 0.35 <= n_split_first <= 0.55:
                notes.append("First-split share is reasonable for malting (~40–50%).")
    else:
        # feed
        if y_pred is not None and y_pred < 7.0 and n_rate < 150:
            notes.append("Yield looks modest; consider **raising N** toward 150–180 kg/ha if soil N is low.")
        if final_gs in ["sowing", "emergence"]:
            notes.append("Avoid concentrating too much N at sowing; ensure **in-season splits** (GS25/GS31).")

    notes.append("Always adjust for local soil N, previous crop, and regulatory constraints.")
    return " ".join(notes)

# Show recommendation
if input_row is not None:
    rec_text = simple_recommendation(
        end_use=end_use,
        n_rate=float(n_rate_kg_ha),
        n_split_first=float(n_split_first_prop),
        final_gs=final_n_timing_gs,
        y_pred=pred_yield,
        p_pred=pred_protein,
    )
    st.markdown("### 📋 Recommendation")
    st.info(rec_text)

# ---------------- Azure OpenAI chatbot ----------------
st.markdown("---")
st.subheader("💬 Ask the advisor")

# Prepare context for the model
context_blob = {
    "inputs": {
        "source_doc": source_doc,
        "site_id": site_id,
        "year": year,
        "end_use": end_use,
        "n_rate_kg_ha": float(n_rate_kg_ha),
        "n_split_first_prop": float(n_split_first_prop),
        "final_n_timing_gs": final_n_timing_gs,
    },
    "predictions": {
        "yield_t_ha": pred_yield,
        "protein_pct": pred_protein,
    },
    "guidance": {
        "malting_protein_target": "9.5–11.0%",
        "typical_n_split": "first split ~40–50%",
        "gs_notes": "GS25=tillering, GS31=stem elongation, GS37=flag leaf, GS61=flowering",
    },
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system",
         "content": (
             "You are an agronomy assistant for spring barley. "
             "Be practical, concise, and transparent about uncertainty. "
             "If asked for N adjustments, give ballpark ranges (±10–20%) and caveats "
             "about soil N, local regulations, lodging risk, and timing."
         )},
        {"role": "system", "content": f"Context: {json.dumps(context_blob)}"}
    ]

# Render chat history
for m in st.session_state.chat_history:
    if m["role"] in ("user", "assistant"):
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["content"])

# Input + respond
user_msg = st.chat_input("Ask a question (e.g., 'What if I reduce N by 15%?')")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    if AZURE_OK:
        try:
            client = AzureOpenAI(
                api_key=st.secrets["azure"]["api_key"],
                api_version=st.secrets["azure"]["api_version"],
                azure_endpoint=st.secrets["azure"]["endpoint"],
            )
            deployment = st.secrets["azure"]["deployment"]

            resp = client.chat.completions.create(
                model=deployment,
                messages=st.session_state.chat_history,
                temperature=0.2,
                top_p=0.9,
                max_tokens=600,
            )
            reply = resp.choices[0].message.content
        except Exception as e:
            reply = f"Azure OpenAI error: {e}"
    else:
        reply = ("(Azure OpenAI not configured) Add [azure] api_key, endpoint, deployment, api_version "
                 "in Streamlit secrets to enable the chatbot.")

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
