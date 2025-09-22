# app.py
import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# Page config & small CSS tweaks
# ==============================
st.set_page_config(
    page_title="Barley Advisor — Yield & Quality",
    page_icon="🌾",
    layout="wide",
)

# Subtle style and a highlighted card for the advisor
st.markdown(
    """
    <style>
    .advisor-card {
        background: #f8f9fc;
        border: 1px solid #eef0f6;
        padding: 18px 18px 14px 18px;
        border-radius: 14px;
    }
    .small-muted { color: #6c757d; font-size: 0.92rem; }
    .footer a { text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
    .linkedInIcon {
        display:inline-block;
        width:18px; height:18px;
        margin-left:6px; position:relative; top:3px;
    }
    .metric-box {
        padding:12px 14px; border:1px solid #eef0f6; border-radius:12px; background:#fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =======================
# Load model and metadata
# =======================
@st.cache_resource(show_spinner=False)
def load_model():
    # Adjust if your model path differs
    model_path = os.path.join("barley_model", "model.pkl")
    mdl = joblib.load(model_path)
    return mdl

model = load_model()

# The model was trained with these features (hidden defaults for some)
EXPECTED_COLS = [
    "source_doc",          # hidden default
    "year",                # hidden default (we'll set a fixed value)
    "site_id",             # hidden default
    "end_use",
    "n_rate_kg_ha",
    "n_split_first_prop",
    "final_n_timing_gs",
]

# Reasonable hidden defaults – adjust if your model expects different tokens
HIDDEN_DEFAULTS = dict(
    source_doc="Focus2022",
    year=2020,
    site_id="S1",
)

END_USE_OPTIONS = ["malting", "feed"]
FINAL_N_OPTIONS = [
    "Tillering (GS25)",
    "Stem extension (GS31)",
    "Flag leaf (GS39)",
]

# ==========================
# Helper: make predictions
# ==========================
def predict_yield_quality(end_use: str, n_rate: float, n_split_prop: float, final_n_label: str):
    # Map the final N timing label to a numeric GS code if your model expects that
    # Here we parse the number inside parentheses, e.g. "Tillering (GS25)" -> 25
    gs_val = 25
    if "(" in final_n_label and "GS" in final_n_label:
        try:
            gs_val = int(final_n_label.split("GS")[1].split(")")[0])
        except Exception:
            pass

    row = {
        **HIDDEN_DEFAULTS,
        "end_use": end_use,
        "n_rate_kg_ha": float(n_rate),
        "n_split_first_prop": float(n_split_prop),
        "final_n_timing_gs": float(gs_val),
    }
    X = pd.DataFrame([row], columns=EXPECTED_COLS)

    # Your pipeline likely returns a 2-vector: [yield_t_ha, protein_pct]
    y_pred = model.predict(X)
    yld, prot = float(y_pred[0][0]), float(y_pred[0][1])
    return yld, prot

# ====================================
# Simple rule-based recommendation text
# ====================================
def rule_of_thumb_advice(end_use: str, n_rate: float, n_split_prop: float, final_n_label: str, yld: float, prot: float) -> str:
    target_protein = 10.5 if end_use == "malting" else 11.5  # malting lower target; feed a bit higher is fine
    delta_p = prot - target_protein

    tips = []
    if end_use == "malting":
        if prot > target_protein + 0.3:
            tips.append("Protein looks on the high side for malting. Consider **reducing total N** or **bringing more N earlier** (keep late N modest).")
        elif prot < target_protein - 0.3:
            tips.append("Protein may be a bit low. If quality allows, consider **slightly later N** or **a small top-up** to raise protein.")
        else:
            tips.append("Protein is close to a typical malting window. **Maintain a balanced split** and avoid late heavy N.")
    else:  # feed
        if prot < 8.5:
            tips.append("Protein is quite low for feed; a **modest increase in total N** or **slightly later N** may improve grain protein.")
        else:
            tips.append("For feed barley this protein looks acceptable. Focus on **maximizing yield** with a balanced N plan.")

    # Split guidance
    split_pct = n_split_prop * 100
    if split_pct < 35:
        tips.append("Your first split is relatively small. Many programs use **40–50% at GS25** to support tillering.")
    elif split_pct > 60:
        tips.append("Your first split is quite large. Consider **40–50% at GS25** with the remainder later for flexibility.")

    # Lodging caution if late timing
    if "GS39" in final_n_label:
        tips.append("With **late N (GS39)**, watch lodging risk; ensure variety and canopy support it.")

    summary = f"- Predicted yield: **{yld:.2f} t/ha**\n- Predicted grain protein: **{prot:.2f}%**"
    return "#### Quick take\n" + summary + "\n\n" + "#### Suggestions\n" + "\n".join([f"- {t}" for t in tips])

# ==================================
# Azure OpenAI (via Streamlit secret)
# ==================================
def get_azure_client():
    try:
        az = st.secrets["azure"]
        api_key = az.get("api_key", "")
        endpoint = az.get("endpoint", "").rstrip("/")
        deployment = az.get("deployment", "")
        api_version = az.get("api_version", "")

        if not (api_key and endpoint and deployment and api_version):
            return None, "Missing one or more Azure secrets (api_key, endpoint, deployment, api_version)."

        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        return (client, deployment), None
    except Exception as e:
        return None, f"Azure config error: {e}"

def ask_azure_advisor(prompt: str, system_prompt: str, temperature: float):
    pack, err = get_azure_client()
    if err:
        raise RuntimeError(err)
    (client, deployment) = pack

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=max(0.0, min(1.0, float(temperature))),
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Azure call failed: {e}")

# ============
# UI — Header
# ============
st.title("🌾 Barley Advisor — Yield & Quality")
st.caption("Prototype model trained on Teagasc data. For demo only — not agronomic advice.")

# ============
# Inputs
# ============
st.header("Inputs")

col1, col2 = st.columns(2)

with col1:
    n_rate = st.number_input("Nitrogen rate (kg/ha)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
with col2:
    end_use = st.selectbox("End use", END_USE_OPTIONS, index=END_USE_OPTIONS.index("feed"))

with col1:
    n_split = st.number_input("N split — first application proportion", min_value=0.0, max_value=1.0, value=0.50, step=0.05, help="Fraction of total N applied in the first split (e.g., 0.50 means 50% at GS25).")
with col2:
    final_n_label = st.selectbox("Final N timing", FINAL_N_OPTIONS, index=0)

# Predict button
pred_btn = st.button("Predict", type="primary")

# Output placeholders
yld, prot = None, None
if pred_btn:
    try:
        yld, prot = predict_yield_quality(end_use, n_rate, n_split, final_n_label)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.subheader("Predicted yield")
            st.markdown(f"<h3>{yld:.2f} t/ha</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.subheader("Predicted grain protein")
            st.markdown(f"<h3>{prot:.2f}%</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed — check inputs. Details: {e}")

# =========================
# Big, inviting Advisor UI
# =========================
st.subheader("Ask the advisor")

# Right-aligned advisor settings
right = st.container()
with right:
    with st.expander("Advisor settings (optional)", expanded=False):
        # Rename with domain-safe language + clear tooltip
        resp_style = st.slider(
            "Response style",
            min_value=0.0, max_value=1.0, value=0.20, step=0.05,
            help="This controls how *creative vs. cautious* the AI writing is (AI parameter). "
                 "Lower = more conservative/grounded, Higher = more exploratory. "
                 "This is **not** crop temperature."
        )

# Advisor card with large text area and bold button
st.markdown("<div class='advisor-card'>", unsafe_allow_html=True)

default_hint = "What if I reduce N by 15%?" if yld is None else f"Given the predictions ({yld:.2f} t/ha, {prot:.2f}%), what should I adjust?"

user_q = st.text_area(
    " ",
    placeholder=f"Ask a question (e.g., '{default_hint}')",
    height=90,
    label_visibility="collapsed",
)
ask_btn = st.button("🚀 Ask Advisor", type="primary", use_container_width=True)

# System grounding for the model
SYSTEM_PROMPT = (
    "You are an agronomy assistant for spring barley. You respond succinctly, with clear bullet points and concrete, "
    "practical advice. You base suggestions on the user's inputs and the predicted yield and protein. You do not provide "
    "safety-critical advice. Always include a short summary and then 3–6 actionable bullets. If the user asks for something "
    "outside barley N management, state what's out of scope and gently redirect."
)

if ask_btn:
    if not user_q.strip():
        st.warning("Please type a question for the advisor.")
    else:
        # If we have predictions, include them; otherwise, still include the raw inputs
        context_lines = [
            f"End use: {end_use}",
            f"N rate (kg/ha): {n_rate}",
            f"First split proportion: {n_split} (≈ {n_split*100:.0f}%)",
            f"Final N timing: {final_n_label}",
        ]
        if yld is not None and prot is not None:
            context_lines.append(f"Predicted yield: {yld:.2f} t/ha")
            context_lines.append(f"Predicted protein: {prot:.2f}%")

        base_context = " | ".join(context_lines)
        full_prompt = (
            f"Context: {base_context}\n\n"
            f"User question: {user_q}\n\n"
            "Please answer for a typical Irish/UK spring barley context, noting that local conditions and rules vary. "
            "Keep it brief with a summary and 3–6 bullets."
        )

        # Attempt Azure call; if misconfigured, fall back to local rule-of-thumb
        try:
            answer = ask_azure_advisor(full_prompt, SYSTEM_PROMPT, resp_style)
            st.markdown(answer)
        except Exception as e:
            st.warning(f"Azure advisor not available. Showing rule-of-thumb guidance instead. ({e})")
            # Fallback uses our quick heuristic if we have predictions
            if yld is None or prot is None:
                # Get a temporary prediction so the advice has numbers
                try:
                    _y, _p = predict_yield_quality(end_use, n_rate, n_split, final_n_label)
                except Exception:
                    _y, _p = (math.nan, math.nan)
                st.markdown(rule_of_thumb_advice(end_use, n_rate, n_split, final_n_label, _y, _p))
            else:
                st.markdown(rule_of_thumb_advice(end_use, n_rate, n_split, final_n_label, yld, prot))

st.markdown("</div>", unsafe_allow_html=True)

# =========
# Footer
# =========
st.markdown("---")
left, right = st.columns([1, 1])
with left:
    st.caption("Version: demo")
with right:
    st.markdown(
        """
        <div class='footer' style='text-align:right;'>
          Developed by <a href="https://www.linkedin.com/in/nikolaygeorgiev/" target="_blank">Nikolay Georgiev</a>
          <a href="https://www.linkedin.com/in/nikolaygeorgiev/" target="_blank" class="linkedInIcon">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="#0A66C2">
              <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.762 2.239 5 5 5h14c2.762 0 5-2.238 5-5v-14c0-2.761-2.238-5-5-5zm-11 20h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764 0-.974.784-1.768 1.75-1.768s1.75.794 1.75 1.768c0 .974-.784 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-1.337-.027-3.059-1.865-3.059-1.866 0-2.152 1.458-2.152 2.965v5.698h-3v-11h2.881v1.507h.041c.401-.761 1.379-1.563 2.84-1.563 3.04 0 3.602 2.003 3.602 4.609v6.447z"/>
            </svg>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
