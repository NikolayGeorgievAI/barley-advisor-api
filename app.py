# app.py
from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------
# Page config
# ----------
st.set_page_config(
    page_title="Barley Advisor — Yield & Quality Predictor",
    page_icon="🌾",
    layout="wide",
)

# -----------------
# Styling / helpers
# -----------------
HIDE_DECORATION = """
<style>
/* Tighten cards/panels a bit and make inputs roomy */
section.main > div { padding-top: 1.2rem; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
div[data-baseweb="select"] > div { border-radius: 10px; }
.stAlert { border-radius: 12px; }

/* Right-align the Advisor settings header cell content */
.advisor-settings-col > div:nth-child(1) { display: flex; justify-content: end; }

/* Footer link styling */
.footer a { text-decoration: none; }
.footer .name { font-weight: 600; }
</style>
"""
st.markdown(HIDE_DECORATION, unsafe_allow_html=True)

def pill(text: str, color: str = "#16a34a") -> str:
    """Small rounded pill for result chips."""
    return f"""
    <span style="
        display:inline-block;
        padding:0.35rem 0.6rem;
        border-radius:999px;
        font-weight:600;
        background:{color}1A;
        color:{color};
        border:1px solid {color}33;">
        {text}
    </span>
    """

# ---------------
# Load the model
# ---------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Expecting joblib pipeline at barley_model/model.pkl
    path = os.path.join("barley_model", "model.pkl")
    model = joblib.load(path)

    # Try to detect expected order (helps avoid column mismatch)
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        # Fallback to the order we used during training (based on your earlier logs)
        expected = [
            "source_doc",
            "year",
            "site_id",
            "end_use",
            "n_rate_kg_ha",
            "n_split_first_prop",
            "final_n_timing_gs",
        ]
    return model, expected

MODEL, EXPECTED = load_model()

# --------------------
# Domain dictionaries
# --------------------
END_USE = {
    "feed": "feed",
    "malting": "malting",
}

# Use simple labels for BBCH/GS – map to training categories
FINAL_N_TIMING_LABELS = {
    "Tillering (GS25)": "GS25",
    "Stem extension (GS31–32)": "GS31",
    "Flag leaf (GS37–39)": "GS37",
    "Ear emergence (GS51–55)": "GS51",
    "Flowering (GS59)": "GS59",
}

# Hidden defaults (not shown to user, but required by the model)
HIDDEN_DEFAULTS = {
    "source_doc": "Focus2022",
    "site_id": "S1",
    "year": 2020,  # kept but hidden in UI
}

# -----------------
# Header + subtext
# -----------------
st.title("Barley Advisor — Yield & Quality Predictor")
st.caption("Prototype model trained on Teagasc data. For demo only — not agronomic advice.")

# -------------
# Input controls
# -------------
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        # Year is hidden from UI — using stored default
        year = HIDDEN_DEFAULTS["year"]

        n_rate = st.number_input(
            "Nitrogen rate (kg/ha)",
            min_value=0.0, max_value=350.0, step=5.0, value=120.0,
            help="Total nitrogen rate applied across the season (kg/ha).",
        )

        final_n_label = st.selectbox(
            "Final N timing",
            options=list(FINAL_N_TIMING_LABELS.keys()),
            index=0,
            help="When the last nitrogen is planned. Labels map to growth stages used in the model.",
        )
        final_n_timing_gs = FINAL_N_TIMING_LABELS[final_n_label]

    with c2:
        end_use_label = st.selectbox(
            "End use",
            options=list(END_USE.keys()),
            index=0,
            help="Target market. Malting barley prefers lower grain protein.",
        )
        end_use = END_USE[end_use_label]

        n_split_first_prop = st.number_input(
            "N split — first application proportion",
            min_value=0.0, max_value=1.0, step=0.05, value=0.50,
            help="Share of total N applied in the first pass (e.g., 0.50 = 50%).",
        )

# Assemble single-row dataframe in expected order
def build_features_df() -> pd.DataFrame:
    row = {
        "source_doc": HIDDEN_DEFAULTS["source_doc"],
        "year": HIDDEN_DEFAULTS["year"],
        "site_id": HIDDEN_DEFAULTS["site_id"],
        "end_use": end_use,
        "n_rate_kg_ha": float(n_rate),
        "n_split_first_prop": float(n_split_first_prop),
        "final_n_timing_gs": final_n_timing_gs,
    }
    # Respect model's expected column order
    safe_row = {col: row[col] for col in EXPECTED}
    return pd.DataFrame([safe_row])

# ----------
# Predict UI
# ----------
left, right = st.columns([1, 2])
with left:
    if st.button("Predict", type="primary", use_container_width=True):
        df = build_features_df()
        try:
            raw = MODEL.predict(df)
            # Expecting shape (1, 2): [yield_t_ha, protein_pct]
            # Fall back gracefully if 1D
            if isinstance(raw, (list, tuple, np.ndarray)):
                arr = np.array(raw)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    yld = float(arr[0, 0])
                    prot = float(arr[0, 1])
                elif arr.ndim == 1 and arr.size >= 2:
                    yld = float(arr[0])
                    prot = float(arr[1])
                else:
                    # If model returns a single target, assume it's yield
                    yld = float(arr[0])
                    prot = np.nan
            else:
                yld, prot = float(raw), np.nan

            st.session_state["last_prediction"] = {"yield_t_ha": yld, "protein_pct": prot}
        except Exception as e:
            st.error(f"Prediction failed. Please check inputs. Details: {e}")

with right:
    pred = st.session_state.get("last_prediction")
    if pred:
        y_txt = f"Predicted yield: {pred['yield_t_ha']:.2f} t/ha"
        p_html = pill(y_txt, "#16a34a")
        st.markdown(p_html, unsafe_allow_html=True)

        if not np.isnan(pred.get("protein_pct", np.nan)):
            pr_txt = f"Predicted grain protein: {pred['protein_pct']:.2f} %"
            st.markdown(pill(pr_txt, "#2563eb"), unsafe_allow_html=True)

        st.caption("Results are model estimates based on the inputs provided.")

# -----------------------
# Advisor (Azure OpenAI)
# -----------------------
st.divider()

row_title, row_settings = st.columns([1.0, 1.0], gap="small")
with row_title:
    st.subheader("Ask the advisor")
with row_settings:
    # Right-side compact settings box
    st.markdown('<div class="advisor-settings-col">', unsafe_allow_html=True)
  with st.expander("Advisor settings (optional)"):
    creativity = st.slider(
        "Response style",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Controls how the advisor responds:\n"
             "- Lower values = more conservative, consistent, and factual\n"
             "- Higher values = more creative, exploratory, and varied"
    )


# Build a helpful, safe prompt
def build_advisor_prompt(question: str) -> str:
    pred = st.session_state.get("last_prediction", {})
    yld = pred.get("yield_t_ha")
    prt = pred.get("protein_pct")

    # Compact context
    context = {
        "end_use": end_use,
        "n_rate_kg_ha": n_rate,
        "n_split_first_prop": n_split_first_prop,
        "final_n_timing_gs": final_n_timing_gs,
        "assumed_year": HIDDEN_DEFAULTS["year"],
        "assumed_site": HIDDEN_DEFAULTS["site_id"],
        "predicted_yield_t_ha": round(yld, 3) if isinstance(yld, (int, float)) else None,
        "predicted_protein_pct": round(prt, 3) if isinstance(prt, (int, float)) else None,
    }

    instructions = (
        "You are a cautious agronomy assistant for spring barley in a temperate climate. "
        "Use the provided context and the model estimates as indicative only. "
        "Offer practical, concise suggestions. Avoid medical/chemical safety advice; instead, "
        "recommend consulting local agronomy guidance for any regulatory or safety constraints. "
        "Prefer neutral language and quantify changes (e.g., ±10–20 kg/ha). Keep to 6 bullets max, then a one-sentence summary."
    )

    return (
        f"{instructions}\n\nCONTEXT:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "If the question asks about increasing/decreasing N, respond with expected trade-offs for yield & protein, "
        "a suggested split ratio, and any watch-outs (lodging, timing drift)."
    )

# Azure OpenAI client (if configured)
def get_azure_client():
    try:
        az = st.secrets["azure"]
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=az["api_key"],
            api_version=az["api_version"],
            azure_endpoint=az["endpoint"],
        )
        deployment = az["deployment"]
        return client, deployment
    except Exception:
        return None, None

client, deployment = get_azure_client()

# Chat box
question = st.text_input(
    "Ask a question (e.g., 'What if I reduce N by 15%?')",
    value="",
    placeholder="Type your agronomy question…",
)

def call_azure_chat(q: str) -> tuple[bool, str]:
    prompt = build_advisor_prompt(q)
    try:
        resp = client.chat.completions.create(
            model=deployment,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a helpful, cautious agronomy assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=450,
        )
        text = resp.choices[0].message.content.strip()
        return True, text
    except Exception as e:
        return False, str(e)

if question.strip():
    if client is None:
        st.info("Azure OpenAI not configured. Add `[azure] api_key, endpoint, deployment, api_version` in Streamlit secrets to enable the chatbot.")
    else:
        ok, answer = call_azure_chat(question.strip())
        if ok:
            st.success(answer)
        else:
            with st.expander("Azure call failed. See error details below."):
                st.code(str(answer))

# -------
# Footer
# -------
st.markdown("---")
st.markdown(
    """
<div class="footer" style="display:flex;align-items:center;gap:0.5rem;opacity:0.9;">
  <span>Developed by <span class="name">Nikolay Georgiev</span></span>
  <a href="https://www.linkedin.com/in/nikolaygeorgiev/" target="_blank" rel="noopener noreferrer" title="LinkedIn">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="18" height="18" style="vertical-align:-3px;opacity:0.9;" />
  </a>
</div>
""",
    unsafe_allow_html=True,
)

