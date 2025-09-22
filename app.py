# app.py
# Barley Advisor — Yield & Quality Predictor (clean version)
# - No Year, no debug panes
# - Azure OpenAI chat advisor (with a clear "Response style" control instead of "temperature")
# - Footer credit with LinkedIn (icon shown AFTER name)
# - Hidden defaults for model features not shown to the user (source_doc, site_id)

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- App config ----------
st.set_page_config(
    page_title="Barley Advisor — Yield & Quality",
    page_icon="🌾",
    layout="wide",
)

# ---------- Constants / UI dictionaries ----------
END_USE_OPTIONS = ["feed", "malting"]

# Friendly labels for final N timing, mapped to GS codes used by the model
GS_LABELS = {
    "Tillering (GS25)": "GS25",
    "Stem extension (GS31)": "GS31",
    "Flag leaf (GS37)": "GS37",
    "Booting (GS41)": "GS41",
    "Emergence": "emergence",
    "Sowing": "sowing",
}
GS_FRIENDLY = list(GS_LABELS.keys())

# Hidden defaults for features the model expects but we don’t show
DEFAULT_SOURCE_DOC = "Focus2022"   # consistent default doc used during training
DEFAULT_SITE_ID = "S1"             # single-site default to keep UI simple

MODEL_PATH = os.path.join("barley_model", "model.pkl")


# ---------- Utilities ----------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load the trained sklearn pipeline (joblib)."""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model at '{path}'.\n\n{e}")
        return None


def make_feature_row(
    end_use: str,
    n_rate_kg_ha: float,
    n_split_first_prop: float,
    final_n_timing_label: str,
) -> pd.DataFrame:
    """
    Build a single-row dataframe in the exact feature order the model expects.
    Expected columns (from training): 
      ['source_doc','year','site_id','end_use','n_rate_kg_ha','n_split_first_prop','final_n_timing_gs']
    - We removed 'year' from UI; set to a neutral default (e.g., 2020).
    """
    final_gs = GS_LABELS[final_n_timing_label]

    data = {
        "source_doc": [DEFAULT_SOURCE_DOC],
        "year": [2020],  # neutral constant; model won’t mind as long as trained for multiple years
        "site_id": [DEFAULT_SITE_ID],
        "end_use": [end_use],
        "n_rate_kg_ha": [float(n_rate_kg_ha)],
        "n_split_first_prop": [float(n_split_first_prop)],
        "final_n_timing_gs": [final_gs],
    }
    cols_in_order = [
        "source_doc",
        "year",
        "site_id",
        "end_use",
        "n_rate_kg_ha",
        "n_split_first_prop",
        "final_n_timing_gs",
    ]
    df = pd.DataFrame(data)[cols_in_order]
    return df


def safe_number(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)


# ---------- Azure OpenAI client (lazy) ----------
def get_azure_client():
    """
    Build an Azure OpenAI client from Streamlit secrets.
    Returns (client, deployment, api_ok: bool, msg_if_not_ok).
    """
    try:
        from openai import AzureOpenAI
    except Exception:
        return None, None, False, "Python package 'openai' is not installed."

    az = st.secrets.get("azure", {})
    api_key = az.get("api_key")
    endpoint = az.get("endpoint")
    deployment = az.get("deployment")
    api_version = az.get("api_version")

    if not all([api_key, endpoint, deployment, api_version]):
        return None, None, False, (
            "Add [azure] api_key, endpoint, deployment, api_version in Streamlit secrets to enable the advisor."
        )

    try:
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        return client, deployment, True, ""
    except Exception as e:
        return None, None, False, f"Failed to initialize Azure OpenAI client: {e}"


def ask_advisor(client, deployment, messages, temperature):
    """
    Call Azure OpenAI chat completion with safety and return string content or error text.
    """
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=float(temperature),
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Azure call failed. See error details below.\n\n{e}"


# ---------- UI: Header ----------
st.markdown(
    """
    # Barley Advisor — Yield & Quality

    Prototype model trained on Teagasc data. For demo only — **not** agronomic advice.
    """,
    help="Outputs are model-based estimates and should not be used as sole agronomic advice.",
)

# ---------- Left & right columns for inputs ----------
left, right = st.columns(2)

with left:
    end_use = st.selectbox(
        "End use",
        END_USE_OPTIONS,
        index=0,
        help="Target use of the barley crop, used by the model to estimate protein and yield.",
    )

with right:
    n_split_first_prop = st.number_input(
        "N split — first application proportion",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=0.50,
        help="Fraction of total N applied at the first timing (e.g., 0.50 = 50%).",
    )

with left:
    n_rate_kg_ha = st.number_input(
        "Nitrogen rate (kg/ha)",
        min_value=0.0,
        max_value=400.0,
        step=5.0,
        value=120.0,
        help="Total nitrogen applied across the season.",
    )

with right:
    final_n_choice = st.selectbox(
        "Final N timing",
        GS_FRIENDLY,
        index=0,
        help="Choose the last N timing; internally the model uses GS codes.",
    )

# Predict button
pred_btn = st.button("Predict", type="primary")

# ---------- Load model ----------
model = load_model(MODEL_PATH)

yield_pred = None
protein_pred = None

if pred_btn:
    if model is None:
        st.error("Model not loaded. Cannot predict.")
    else:
        X = make_feature_row(
            end_use=end_use,
            n_rate_kg_ha=n_rate_kg_ha,
            n_split_first_prop=n_split_first_prop,
            final_n_timing_label=final_n_choice,
        )
        try:
            # Expecting a 2-value output: [yield_t_ha, protein_pct]
            yhat = model.predict(X)
            # Handle different shapes robustly
            arr = np.array(yhat).reshape(-1)
            if arr.size >= 2:
                yield_pred = float(arr[0])
                protein_pred = float(arr[1])
            else:
                st.error("Model returned an unexpected output shape.")
        except Exception as e:
            st.error(f"Prediction failed — check inputs or categories.\n\n{e}")

# ---------- Show predictions ----------
if (yield_pred is not None) and (protein_pred is not None):
    m1, m2 = st.columns(2)
    with m1:
        st.success(f"**Predicted yield:** {yield_pred:.2f} t/ha")
    with m2:
        st.success(f"**Predicted grain protein:** {protein_pred:.2f} %")


# ---------- Advisor (chat) row: Ask + Settings aligned ----------
st.markdown("---")
row_left, row_right = st.columns([3, 1])

with row_left:
    st.subheader("Ask the advisor")

with row_right:
    with st.expander("Advisor settings (optional)", expanded=False):
        creativity = st.slider(
            "Response style",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            help=(
                "Controls how the advisor responds (this is **not** field temperature):\n"
                "• Lower values = more conservative, consistent, and factual\n"
                "• Higher values = more creative, exploratory, and varied"
            ),
        )

# Chat input
user_q = st.chat_input("Ask a question (e.g., 'What if I reduce N by 15%?')")

if user_q:
    # Prepare a short context for the advisor using current UI values/predictions
    context_lines = [
        f"End use: {end_use}",
        f"Total N: {n_rate_kg_ha:.0f} kg/ha",
        f"First application proportion (split): {n_split_first_prop:.2f}",
        f"Final N timing: {final_n_choice} (code {GS_LABELS[final_n_choice]})",
    ]
    if (yield_pred is not None) and (protein_pred is not None):
        context_lines.append(f"Model predicted yield: {yield_pred:.2f} t/ha")
        context_lines.append(f"Model predicted protein: {protein_pred:.2f} %")
    context = "\n".join(context_lines)

    # System instruction keeps the advisor grounded and non-prescriptive.
    system_prompt = (
        "You are an agronomy assistant. Provide practical, cautious, and non-prescriptive "
        "advice for barley management based on the user’s inputs and the model’s predicted "
        "yield and protein. Always remind users to consult local agronomy advice and consider "
        "trialing changes at small scale first."
    )

    client, deployment, ok, msg = get_azure_client()

    if not ok:
        st.warning(
            "Azure OpenAI not configured. "
            "Add `[azure] api_key, endpoint, deployment, api_version` in Streamlit secrets to enable the advisor."
        )
        with st.chat_message("assistant"):
            st.write(
                "If Azure were configured, I would answer here. Meanwhile, consider:\n"
                "• For malting, reducing protein often means moderating N late in the season.\n"
                "• For feed, a bit more N can lift yield, but watch lodging and split timing.\n"
                "• Always adapt to local soils, rainfall, and lodging risk."
            )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here are my inputs and model outputs:\n{context}\n\nMy question: {user_q}",
            },
        ]
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            answer = ask_advisor(client, deployment, messages, temperature=creativity)
            st.write(answer)


# ---------- Footer credit ----------
st.markdown("---")
footer_col1, footer_col2 = st.columns([1, 3])
with footer_col1:
    st.caption("Version: demo")
with footer_col2:
    # LinkedIn after name, as requested
    st.markdown(
        """
        <div style="text-align:right;">
            Developed by <b>Nikolay Georgiev</b>
            <a href="https://www.linkedin.com/in/nikolaygeorgiev/" target="_blank" rel="noopener">
                <img alt="LinkedIn" src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" 
                     style="height: 0.95em; vertical-align: text-bottom; margin-left: 6px;"/>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
