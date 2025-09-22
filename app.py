import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------- UI BASICS ----------
st.set_page_config(page_title="Barley Advisor — Yield & Quality", page_icon="🌾", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0'>🌾 Barley Advisor — Yield & Quality</h1>"
    "<div style='color:#666;margin-top:4px'>Prototype model trained on Teagasc dataset. "
    "For demo only — not agronomic advice.</div>",
    unsafe_allow_html=True,
)

# ---------- MODEL LOADING ----------
MODEL_PATH = Path("barley_model/model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

model_error = None
model = None
try:
    model = load_model()
except Exception as e:
    model_error = str(e)

# Expected feature order used during training
FEATURE_ORDER = [
    "source_doc",
    "year",
    "site_id",
    "end_use",
    "n_rate_kg_ha",
    "n_split_first_prop",
    "final_n_timing_gs",
]

# For clean UX in the main column
col_main, = st.columns([1])

# ---------- INPUTS / PREDICTION ----------
with st.container():
    st.subheader("Inputs")

    # Keep “hidden” source_doc for consistency with trained pipeline (we set Focus2022)
    source_doc_default = "Focus2022"  # safest default from your inspection
    source_doc = source_doc_default

    # End-use choices (align with your encoder categories)
    END_USE = ["feed", "malting"]
    # GS milestones translated to friendly labels (keep original codes for model)
    GS_OPTIONS = {
        "GS25": "Tillering (GS25)",
        "GS31": "Stem elongation (GS31)",
        "GS55": "Ear half-emerged (GS55)",
        "GS57": "Ear three-quarters (GS57)",
        "GS61": "Flowering (GS61)",
        "emergence": "Emergence",
        "sowing": "Sowing",
    }
    gs_labels = [GS_OPTIONS[k] for k in GS_OPTIONS]

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year", value=2020, min_value=2010, max_value=2035, step=1)
        n_rate = st.number_input("Nitrogen rate (kg/ha)", value=120.0, min_value=0.0, max_value=300.0, step=1.0)
        gs_label = st.selectbox("Final N timing", options=gs_labels, index=list(GS_OPTIONS.keys()).index("GS25"))
    with c2:
        end_use = st.selectbox("End use", END_USE, index=END_USE.index("feed"))
        n_split_first_prop = st.number_input("N split — first application proportion", value=0.50, min_value=0.0, max_value=1.0, step=0.05)

    # We hide site_id for users; pick a stable default used during training (e.g., “S1”/“A”/etc.)
    site_id = "S1"

    # Map label back to code
    inverse_gs = {v: k for k, v in GS_OPTIONS.items()}
    final_n_timing_gs = inverse_gs[gs_label]

    def make_input_df():
        row = {
            "source_doc": source_doc,
            "year": int(year),
            "site_id": site_id,
            "end_use": end_use,
            "n_rate_kg_ha": float(n_rate),
            "n_split_first_prop": float(n_split_first_prop),
            "final_n_timing_gs": final_n_timing_gs,
        }
        # Ensure exact column order the pipeline expects
        df = pd.DataFrame([{k: row[k] for k in FEATURE_ORDER}])
        return df

    pred_df = make_input_df()

    predict_btn = st.button("Predict", type="primary", use_container_width=False)

    last_prediction = None
    if predict_btn:
        if model_error:
            st.error(f"Model error: {model_error}")
        else:
            try:
                raw = model.predict(pred_df)  # expecting [yield_t_ha, protein_pct] or similar
                raw = np.array(raw).reshape(-1)
                # Heuristic mapping; adjust if your pipeline returns named outputs
                if raw.size == 2:
                    yld, prot = float(raw[0]), float(raw[1])
                else:
                    # If model returns e.g. dict or 2D, try best-effort:
                    yld, prot = float(raw[0]), float(raw[1]) if raw.size > 1 else (float(raw[0]), np.nan)

                st.success(f"**Predicted yield:** {yld:.2f} t/ha")
                st.success(f"**Predicted grain protein:** {prot:.2f} %")
                last_prediction = {
                    "yield_t_ha": yld,
                    "protein_pct": prot,
                    "inputs": pred_df.to_dict(orient="records")[0],
                }
                st.session_state["last_prediction"] = last_prediction
            except Exception as e:
                st.error("Prediction failed — check feature inputs or categories.")
                st.code(repr(e), language="text")

# ---------- ADVISOR (Azure OpenAI) ----------
st.markdown("---")
st.subheader("Ask the advisor")

# Small helper to read Azure secrets safely
def read_azure_secrets():
    data = st.secrets.get("azure", {})
    # Normalize keys that users often vary
    key = data.get("api_key") or data.get("key")
    endpoint = (data.get("endpoint") or "").rstrip("/")
    deployment = data.get("deployment")
    api_version = data.get("api_version") or data.get("version") or "2024-02-15-preview"
    return key, endpoint, deployment, api_version

azure_key, azure_endpoint, azure_deployment, azure_api_version = read_azure_secrets()

# Nice banner when misconfigured
def azure_configured() -> bool:
    ok = True
    missing = []
    if not azure_key:
        missing.append("azure.api_key")
    if not azure_endpoint:
        missing.append("azure.endpoint")
    if not azure_deployment:
        missing.append("azure.deployment")
    if not azure_api_version:
        missing.append("azure.api_version")
    if missing:
        st.info(
            ":orange[Azure OpenAI not configured] — add " + ", ".join(missing) +
            " in **Streamlit → Settings → Secrets** to enable the chatbot."
        )
        ok = False
    return ok

# Advisor options
with st.expander("Advisor settings (optional)", expanded=False):
    use_context = st.checkbox("Include my latest prediction & inputs as context", value=True)
    tone = st.selectbox("Tone", ["Concise", "Detailed"], index=0)

# Initialize chat state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "system",
            "content": (
                "You are Barley Advisor, a helpful agronomy assistant. "
                "You provide practical, cautious guidance for spring barley in temperate climates. "
                "Use simple language and provide specific, actionable suggestions. "
                "Never claim certainty; recommend consulting local agronomy guidance."
            ),
        }
    ]

# Helper to add contextual system message
def add_context_message():
    lp = st.session_state.get("last_prediction")
    if not lp:
        return None
    inputs = lp["inputs"]
    ctx = {
        "predicted_yield_t_ha": lp.get("yield_t_ha"),
        "predicted_protein_pct": lp.get("protein_pct"),
        "inputs": inputs,
    }
    return {
        "role": "system",
        "content": (
            "Context from the prediction tool (if available). "
            f"JSON: {json.dumps(ctx)}. "
            "Use this context to tailor recommendations (e.g., protein too high for malting)."
        ),
    }

# Render previous messages (assistant/user)
def render_history():
    for m in st.session_state.chat_messages:
        if m["role"] == "user":
            st.chat_message("user").markdown(m["content"])
        elif m["role"] == "assistant":
            st.chat_message("assistant").markdown(m["content"])

render_history()

prompt = st.chat_input("Ask a question (e.g., “What if I reduce N by 15%?”)")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    if not azure_configured():
        st.stop()

    # Build request messages
    messages = list(st.session_state.chat_messages)
    if use_context:
        ctx_msg = add_context_message()
        if ctx_msg:
            messages.insert(1, ctx_msg)  # after the base system prompt

    # Call Azure OpenAI with clean error handling
    try:
        # We use the new OpenAI SDK naming for Azure
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=azure_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
        )

        temp = 0.2 if tone == "Concise" else 0.5
        resp = client.chat.completions.create(
            model=azure_deployment,  # this is your Deployment name in Azure
            messages=messages,
            temperature=temp,
            max_tokens=500,
        )

        answer = resp.choices[0].message.content if resp and resp.choices else "I couldn’t generate a response."
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)

    except Exception as e:
        # Parse common Azure errors for clarity
        msg = str(e)
        hint = ""
        if "401" in msg or "Access denied" in msg:
            hint = "Check your API key & subscription are active, and the endpoint region matches your deployment."
        elif "404" in msg:
            hint = "Check the `deployment` name in secrets — it must match the Azure OpenAI Deployment."
        elif "Name or service not known" in msg or "getaddrinfo ENOTFOUND" in msg:
            hint = "Your `endpoint` looks wrong. It should look like: https://<resource-name>.openai.azure.com/"
        elif "429" in msg:
            hint = "You may be rate limited. Try again in a bit or lower the frequency."

        st.warning("Azure call failed. " + (hint or "See error details below."))
        with st.expander("Error details"):
            st.code(msg, language="text")
