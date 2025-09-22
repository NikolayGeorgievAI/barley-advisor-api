# app.py — Barley Advisor (predictor + recommendations + Azure advisor chat)
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page setup ----------------
st.set_page_config(page_title="Barley Advisor — Yield & Quality", page_icon="🌾", layout="wide")
st.markdown(
    "<h1 style='margin-bottom:0'>🌾 Barley Advisor — Yield & Quality</h1>"
    "<div style='color:#666;margin-top:4px'>Prototype model trained on Teagasc data. "
    "For demo only — not agronomic advice.</div>",
    unsafe_allow_html=True,
)

# ---------------- Model loading ----------------
MODEL_PATH = Path("barley_model/model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = None
model_error = None
try:
    model = load_model()
except Exception as e:
    model_error = str(e)

# Expected column order for your pipeline
FEATURE_ORDER = [
    "source_doc",
    "year",
    "site_id",
    "end_use",
    "n_rate_kg_ha",
    "n_split_first_prop",
    "final_n_timing_gs",
]

# ---------------- Inputs & Prediction ----------------
st.subheader("Inputs")

# Hidden defaults (kept out of the UI for end users)
SOURCE_DOC_DEFAULT = "Focus2022"
SITE_ID_DEFAULT = "S1"

# Friendly GS mapping (UI ←→ model code)
GS_OPTIONS = {
    "GS25": "Tillering (GS25)",
    "GS31": "Stem elongation (GS31)",
    "GS37": "Flag leaf visible (GS37)",
    "GS61": "Flowering (GS61)",
    "emergence": "Emergence",
    "sowing": "At sowing",
}
GS_LABELS = [GS_OPTIONS[k] for k in GS_OPTIONS]
INV_GS = {v: k for k, v in GS_OPTIONS.items()}

END_USE = ["feed", "malting"]

c1, c2 = st.columns(2)
with c1:
    year = st.number_input("Year", value=2020, min_value=2010, max_value=2035, step=1,
                           help="Trial/season year.")
    n_rate = st.number_input("Nitrogen rate (kg/ha)", value=120.0, min_value=0.0, max_value=300.0, step=1.0,
                             help="Total N applied across the season.")
    gs_label = st.selectbox("Final N timing", options=GS_LABELS, index=GS_LABELS.index("Tillering (GS25)"),
                            help="Growth stage of the **final** N application (Zadoks).")
with c2:
    end_use = st.selectbox("End use", END_USE, index=END_USE.index("feed"),
                           help="Intended grain use.")
    n_split_first_prop = st.number_input("N split — first application proportion",
                                         value=0.50, min_value=0.0, max_value=1.0, step=0.05,
                                         help="Share of total N applied in the first split (0–1).")

# Prepare input row (respecting model feature order)
def make_input_df():
    row = {
        "source_doc": SOURCE_DOC_DEFAULT,
        "year": int(year),
        "site_id": SITE_ID_DEFAULT,
        "end_use": end_use,
        "n_rate_kg_ha": float(n_rate),
        "n_split_first_prop": float(n_split_first_prop),
        "final_n_timing_gs": INV_GS[gs_label],
    }
    return pd.DataFrame([{k: row[k] for k in FEATURE_ORDER}])

pred_df = make_input_df()

predict_btn = st.button("Predict", type="primary")
pred_yield = None
pred_protein = None

if predict_btn:
    if model_error:
        st.error(f"Model error: {model_error}")
    else:
        try:
            raw = model.predict(pred_df)
            raw = np.array(raw).reshape(-1)
            if raw.size >= 2:
                pred_yield = float(raw[0])
                pred_protein = float(raw[1])
                c_y, c_p = st.columns(2)
                c_y.success(f"**Predicted yield:** {pred_yield:.2f} t/ha")
                c_p.success(f"**Predicted grain protein:** {pred_protein:.2f} %")
            else:
                pred_yield = float(raw[0])
                st.success(f"**Predicted yield:** {pred_yield:.2f} t/ha")
                st.info("Model returned a single target.")
            st.caption("Results are model estimates based on the inputs provided.")
            st.session_state["last_prediction"] = {
                "yield_t_ha": pred_yield,
                "protein_pct": pred_protein,
                "inputs": pred_df.to_dict(orient="records")[0],
            }
        except Exception as e:
            st.error("Prediction failed — check inputs/categories.")
            st.code(repr(e), language="text")

# ---------------- Recommendation engine (rule-based) ----------------
def simple_recommendation(end_use: str, n_rate: float, n_split_first: float, final_gs: str,
                          y_pred: float | None, p_pred: float | None) -> str:
    """
    Transparent heuristics:
      - Malting protein window ~9.5–11.0%.
      - If protein > 11: reduce N by ~5–20% and/or earlier final timing.
      - If protein < 9.5: increase N by ~5–20% and/or slightly later timing.
      - Feed: prioritize yield and N efficiency.
    """
    lines = []
    if y_pred is not None:
        lines.append(f"Predicted yield: {y_pred:.2f} t/ha.")
    if p_pred is not None:
        lines.append(f"Predicted protein: {p_pred:.2f}%.")

    if end_use == "malting" and p_pred is not None:
        lo, hi = 9.5, 11.0
        if p_pred > hi:
            overshoot = min(max((p_pred - hi) / 2.0, 0.05), 0.20)  # 5–20%
            kgha = int(round(n_rate * overshoot))
            lines.append(
                f"Protein is above malting target ({lo}–{hi}%). "
                f"Consider **reducing total N by ~{int(overshoot*100)}% (~{kgha} kg/ha)**."
            )
            if final_gs in ["GS37", "GS61"]:
                lines.append("Consider **earlier final N timing** (e.g., GS25–GS31) to reduce late protein uplift.")
        elif p_pred < lo:
            undershoot = min(max((lo - p_pred) / 2.0, 0.05), 0.20)
            kgha = int(round(n_rate * undershoot))
            lines.append(
                f"Protein is below malting target. "
                f"Consider **increasing total N by ~{int(undershoot*100)}% (~{kgha} kg/ha)**."
            )
            if final_gs in ["sowing", "emergence", "GS25"]:
                lines.append("A **slightly later final N timing** (e.g., GS31) can help lift protein.")
        else:
            lines.append("Protein is within the typical malting window — consider **fine-tuning splits** rather than total N.")
            if 0.35 <= n_split_first <= 0.55:
                lines.append("First-split share looks balanced for malting (~40–50%).")
    else:
        # feed use
        if y_pred is not None and y_pred < 7.0 and n_rate < 150:
            lines.append("Yield looks modest; consider **raising N** toward 150–180 kg/ha if soil N is low.")
        if final_gs in ["sowing", "emergence"]:
            lines.append("Avoid concentrating too much N at sowing; ensure **in-season splits** (GS25/GS31).")

    lines.append("Always adjust for local soil N, previous crop, lodging risk, and regulations.")
    return " ".join(lines)

if "last_prediction" in st.session_state:
    lp = st.session_state["last_prediction"]
    rec_text = simple_recommendation(
        end_use=end_use,
        n_rate=float(n_rate),
        n_split_first=float(n_split_first_prop),
        final_gs=INV_GS[gs_label],
        y_pred=lp.get("yield_t_ha"),
        p_pred=lp.get("protein_pct"),
    )
    st.markdown("### 📋 Recommendation")
    st.info(rec_text)

# ---------------- Advisor (Azure OpenAI) ----------------
st.markdown("---")

# Header row: left title, right settings on the SAME line
col_left, col_right = st.columns([3, 1], vertical_alignment="center")
with col_left:
    st.subheader("Ask the advisor")
with col_right:
    with st.expander("Advisor settings", expanded=False):
        if "advisor_tone" not in st.session_state:
            st.session_state.advisor_tone = "Concise"
        if "advisor_use_context" not in st.session_state:
            st.session_state.advisor_use_context = True
        st.session_state.advisor_use_context = st.checkbox(
            "Include prediction context",
            value=st.session_state.advisor_use_context,
        )
        st.session_state.advisor_tone = st.selectbox(
            "Tone", ["Concise", "Detailed"],
            index=0 if st.session_state.advisor_tone == "Concise" else 1,
        )

# Helper: read Azure secrets safely
def read_azure_secrets():
    data = st.secrets.get("azure", {})
    key = data.get("api_key") or data.get("key")
    endpoint = (data.get("endpoint") or "").rstrip("/")
    deployment = data.get("deployment")
    api_version = data.get("api_version") or "2024-08-01-preview"
    return key, endpoint, deployment, api_version

azure_key, azure_endpoint, azure_deployment, azure_api_version = read_azure_secrets()

def azure_configured() -> bool:
    missing = []
    if not azure_key:        missing.append("azure.api_key")
    if not azure_endpoint:   missing.append("azure.endpoint")
    if not azure_deployment: missing.append("azure.deployment")
    if not azure_api_version:missing.append("azure.api_version")
    if missing:
        st.info(":orange[Azure OpenAI not configured] — add "
                + ", ".join(missing)
                + " in **Streamlit → Settings → Secrets** to enable the advisor.")
        return False
    return True

# Initialize chat history (session memory)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "system",
            "content": (
                "You are Barley Advisor, a helpful agronomy assistant for spring barley. "
                "Context is agronomy only; no unsafe intent. "
                "Be practical, cautious, and clear. If suggesting N changes, keep to ±10–20% with caveats "
                "about soil N, regulations, lodging risk, and timing."
            ),
        }
    ]

# Helper: optional context message from prediction
def build_context_message():
    lp = st.session_state.get("last_prediction")
    if not lp:
        return None
    ctx = {
        "predicted_yield_t_ha": lp.get("yield_t_ha"),
        "predicted_protein_pct": lp.get("protein_pct"),
        "inputs": lp.get("inputs"),
        "guidance": {
            "malting_protein_target": "9.5–11.0%",
            "typical_first_split_share": "40–50%",
            "gs_notes": "GS25=tillering, GS31=stem elongation, GS37=flag leaf, GS61=flowering",
        },
    }
    return {"role": "system", "content": f"Prediction context (JSON): {json.dumps(ctx)}"}

# Render chat history so far
for m in st.session_state.chat_messages:
    if m["role"] == "user":
        st.chat_message("user").markdown(m["content"])
    elif m["role"] == "assistant":
        st.chat_message("assistant").markdown(m["content"])

# Chat input (immediately under advisor header)
user_prompt = st.chat_input("Ask a question (e.g., 'What if I reduce N by 15%?')")
if user_prompt:
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").markdown(user_prompt)

    if not azure_configured():
        st.stop()

    messages = list(st.session_state.chat_messages)
    if st.session_state.advisor_use_context:
        ctx_msg = build_context_message()
        if ctx_msg:
            messages.insert(1, ctx_msg)

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=azure_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
        )

        temperature = 0.2 if st.session_state.advisor_tone == "Concise" else 0.5
        resp = client.chat.completions.create(
            model=azure_deployment,  # this is the Deployment name in Azure
            messages=messages,
            temperature=temperature,
            max_tokens=600,
        )
        answer = resp.choices[0].message.content if resp and resp.choices else "I couldn't generate a response."
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)

    except Exception as e:
        msg = str(e)
        if "filtered" in msg.lower():
            st.warning("⚠️ Your question was blocked by Azure’s safety filters. Try rephrasing in agronomy terms (e.g., 'Effect of raising nitrogen by 15% on yield/protein').")
        else:
            hint = ""
            if "401" in msg or "Access denied" in msg:
                hint = "Check API key & subscription; ensure endpoint region matches your deployment."
            elif "404" in msg:
                hint = "Verify the `deployment` name in secrets matches your Azure OpenAI Deployment."
            elif "Name or service not known" in msg or "ENOTFOUND" in msg:
                hint = "Endpoint may be wrong. It should look like https://<resource>.openai.azure.com/"
            elif "429" in msg:
                hint = "You may be rate-limited. Try again shortly."
            st.warning("Azure call failed. " + (hint or "See details below."))
            with st.expander("Error details"):
                st.code(msg, language="text")

# ---------------- Optional safe debug (no secrets printed) ----------------
with st.expander("Debug (safe)"):
    try:
        import importlib.metadata as md
        st.write("openai version:", md.version("openai"))
    except Exception as e:
        st.write("openai import failed:", e)
    st.write("[azure] in secrets:", "azure" in st.secrets)
    if "azure" in st.secrets:
        st.write("azure keys present:", sorted(list(st.secrets["azure"].keys())))

# ---------------- Footer ----------------
st.markdown(
    """
    <hr style="margin-top: 2em; margin-bottom: 1em;">
    <div style="text-align: center; color: grey; font-size: 0.9em;">
        Developed by 
        <a href="https://www.linkedin.com/in/nikolaygeorgiev/" target="_blank" style="color: grey; text-decoration: none;">
            <b>Nikolay Georgiev</b>
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" style="vertical-align: middle; margin-left: 6px;">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
