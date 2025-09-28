# app.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ================== Author / Branding ==================
AUTHOR_NAME = "Nikolay Georgiev"
LINKEDIN_URL = "https://www.linkedin.com/in/nikolaygeorgiev/"

# ================== Page setup & styles ==================
st.set_page_config(page_title="Barley Advisor — Yield & Quality Predictor", page_icon="🌾", layout="wide")

# ---------- CSS (compact header) ----------
st.markdown(
    """
    <style>
      .block-container {padding-top: 0.6rem; padding-bottom: 1.2rem; max-width: 1200px;}
      .top-banner {display:flex; justify-content:space-between; align-items:center;
        font-size:13px; padding:4px 8px; border-radius:10px; background:#f5f7fb;
        border:1px solid #e7ebf5; margin-bottom:6px; white-space:nowrap; gap:8px;}
      .pill {display:inline-block; padding:2px 8px; border-radius:999px; background:#eef6ff; border:1px solid #dbeafe; font-size:12px;}
      .pill-ok {background:#e8f7ee; border-color:#c8ecd6;}
      .pill-warn {background:#f1f5ff; border-color:#dbeafe;}

      .kpi-grid {display:grid; grid-template-columns: repeat(4, minmax(210px, 1fr)); gap:10px;}
      .kpi {border:1px solid #eef1f6; border-radius:12px; padding:10px; background:white; box-shadow:0 1px 2px rgba(0,0,0,0.03);}
      .kpi h4 {margin:0 0 4px 0; font-size:12px; color:#667085; font-weight:600;}
      .kpi .val {font-size:22px; font-weight:800; letter-spacing:-0.2px;}
      .muted {color:#667085; font-size:12px;}
      .good {color:#117a37;}

      .card {border:1px solid #eef1f6; border-radius:14px; padding:14px; background:white;}
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Secrets / Azure status ==================
def get_secret_or_env(name: str, default: str = "") -> str:
    """Prefer Streamlit secrets (flat or in [azure] section), else env vars."""
    try:
        if name in st.secrets:
            v = str(st.secrets[name]).strip()
            if v:
                return v
        if "azure" in st.secrets:
            # Expect keys: endpoint, deployment, api_key, api_version
            key_map = {
                "AZURE_OPENAI_ENDPOINT": "endpoint",
                "AZURE_OPENAI_DEPLOYMENT": "deployment",
                "AZURE_OPENAI_API_KEY": "api_key",
                "AZURE_OPENAI_API_VERSION": "api_version",
            }
            k = key_map.get(name)
            if k and k in st.secrets["azure"]:
                v = str(st.secrets["azure"][k]).strip()
                if v:
                    return v
    except Exception:
        pass
    return os.getenv(name, default)

AZURE_ENDPOINT    = get_secret_or_env("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT  = get_secret_or_env("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_KEY     = get_secret_or_env("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = get_secret_or_env("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OK = bool(AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_KEY)

# ================== Top banner ==================
azure_badge = (
    '<span class="pill pill-ok">Azure OpenAI connected</span>'
    if AZURE_OK else
    '<span class="pill pill-warn">Local mode (Azure optional)</span>'
)

st.markdown(
    f"""
    <div class="top-banner">
      <div><strong>Barley Advisor</strong> — ML predictions + Azure Generative AI advisor</div>
      <div style="display:flex; align-items:center; gap:8px;">
        {azure_badge}
        <span class="pill">Demo — N. Georgiev</span>
        <a href="{LINKEDIN_URL}" target="_blank" rel="noopener"
           style="font-size:12px; color:#2563eb; text-decoration:none;">
           Developed by {AUTHOR_NAME} ↗
        </a>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== Helpers & defaults ==================
MODEL_PATH = "models/barley_model.pkl"

@dataclass
class Inputs:
    n_rate: float
    p_rate: float
    moisture_pct: float
    field_size: float
    barley_type: str         # "Malting" or "Feed"
    barley_price: float
    urea_price: float
    phosphate_price: float
    inhibitor_on: bool
    inhibitor_cost_per_t_urea: float
    protector_on: bool
    protector_cost_per_t_phosphate: float
    currency: str

CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "CHF": "CHF"}
def symbol(cur: str) -> str: return CURRENCY_SYMBOLS.get(cur, cur)

def load_model():
    """Load your trained model [yield_t_ha, protein_as_is_pct]. Fallback keeps the app usable."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    class FallbackModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            n = X["n_rate"].values
            yld = 6.5 + 0.035*n - 0.00007*(n**2)              # t/ha
            yld = np.clip(yld, 3.5, 12.0)
            prot_as_is = np.clip(7.5 + 0.010*n, 7.0, 12.0)    # % as-is
            return np.c_[yld, prot_as_is]
    return FallbackModel()

MODEL = load_model()

def predict_yield_and_protein_as_is(n_rate: float, p_rate: float) -> Tuple[float, float]:
    X = pd.DataFrame({"n_rate": [n_rate], "p_rate": [p_rate]})
    yld, prot = MODEL.predict(X)[0]
    return float(yld), float(prot)

def as_is_to_dm(protein_as_is_pct: float, moisture_pct: float) -> float:
    """Convert protein measured on wet (as-is) basis to dry matter basis."""
    m = max(0.0, min(40.0, moisture_pct)) / 100.0
    return protein_as_is_pct / (1.0 - m)

def estimate_starch_dm(protein_dm_pct: float, protector_on: bool) -> Tuple[float, float]:
    """
    Simple proxy: starch inversely tracks protein. Base 64% starch DM at 10% protein DM.
    Protector ON nudges starch up by +0.6 abs. Returns a band ±0.8 abs.
    """
    base = 64.0 - 0.7*(protein_dm_pct - 10.0)
    if protector_on: base += 0.6
    return max(55.0, min(72.0, base - 0.8)), max(55.0, min(72.0, base + 0.8))

def gross_margin_simple(inp: Inputs, yield_t_ha: float):
    cur = symbol(inp.currency)
    # Urea: 46% N
    kg_urea_per_ha = inp.n_rate / 0.46
    t_urea_per_ha = kg_urea_per_ha / 1000.0
    urea_cost = t_urea_per_ha * inp.urea_price
    inhibitor_cost = t_urea_per_ha * inp.inhibitor_cost_per_t_urea if inp.inhibitor_on else 0.0

    t_phosphate_per_ha = inp.p_rate / 1000.0
    phosphate_cost = t_phosphate_per_ha * inp.phosphate_price
    protector_cost = t_phosphate_per_ha * inp.protector_cost_per_t_phosphate if inp.protector_on else 0.0

    revenue = yield_t_ha * inp.barley_price
    variable_costs = urea_cost + inhibitor_cost + phosphate_cost + protector_cost
    margin_per_ha = revenue - variable_costs
    margin_total = margin_per_ha * inp.field_size

    rows = [
        (f"Revenue ({inp.barley_type})", f"{cur}{revenue:,.0f} /ha", f"Price: {cur}{inp.barley_price}/t"),
        ("Urea cost", f"{cur}{urea_cost:,.0f} /ha", f"{kg_urea_per_ha:,.0f} kg urea/ha"),
        ("Nitrogen inhibitor" + (" (ON)" if inp.inhibitor_on else " (OFF)"),
         f"{cur}{inhibitor_cost:,.0f} /ha",
         f"{inp.inhibitor_cost_per_t_urea:.0f}{cur}/t × {t_urea_per_ha:.3f} t/ha"),
        ("Phosphate cost", f"{cur}{phosphate_cost:,.0f} /ha", f"{inp.p_rate:.0f} kg P₂O₅/ha"),
        ("Phosphate protector" + (" (ON)" if inp.protector_on else " (OFF)"),
         f"{cur}{protector_cost:,.0f} /ha",
         f"{inp.protector_cost_per_t_phosphate:.0f}{cur}/t × {t_phosphate_per_ha:.3f} t/ha"),
        ("Margin (per ha)", f"{cur}{margin_per_ha:,.0f} /ha", ""),
        ("Margin (field)", f"{cur}{margin_total:,.0f} total", f"{inp.field_size:.2f} ha"),
    ]
    df = pd.DataFrame(rows, columns=["Item", "Amount", "Notes"])
    return margin_per_ha, margin_total, df

# ================== Sidebar inputs ==================
with st.sidebar:
    st.header("Inputs")
    currency = st.selectbox("Currency", ["EUR", "USD", "CHF"], index=0)
    cur = symbol(currency)

    c1, c2 = st.columns(2)
    with c1:
        n_rate = st.number_input("Nitrogen rate (kg N/ha)", 0.0, 400.0, 160.0, 5.0)
        barley_type = st.radio("Barley type", options=["Malting", "Feed"], index=0,
                               help="Switches the reference price between malting and feed.")
        default_malting, default_feed = 320.0, 260.0
        price_label = "Malting barley price" if barley_type == "Malting" else "Feed barley price"
        default_price = default_malting if barley_type == "Malting" else default_feed
        barley_price = st.number_input(f"{price_label} [{cur}/t]", 50.0, 1000.0, default_price, 5.0)

        moisture_pct = st.number_input(
            "Grain moisture at measurement (%)", 8.0, 28.0, 12.0, 0.5,
            help="Used to convert model protein (as-is) to Dry Matter (DM)."
        )
    with c2:
        p_rate = st.number_input("Phosphate rate (kg P₂O₅/ha)", 0.0, 200.0, 60.0, 5.0)
        urea_price = st.number_input(f"Urea price [{cur}/t]", 100.0, 1500.0, 450.0, 10.0)
        phosphate_price = st.number_input(f"Phosphate price [{cur}/t]", 100.0, 1500.0, 700.0, 10.0)

    st.divider()
    st.subheader("Treatment options")
    inhibitor_on = st.toggle("Use nitrogen inhibitor", value=False, help=f"Adds cost per tonne of urea. Default {cur}35/t.")
    inhibitor_cost = st.number_input(f"Inhibitor cost [{cur}/t urea]", 0.0, 200.0, 35.0, 1.0, disabled=not inhibitor_on)

    protector_on = st.toggle("Use phosphate protector", value=False, help=f"Adds cost per tonne of phosphate and may nudge starch DM.")
    protector_cost = st.number_input(f"Protector cost [{cur}/t phosphate]", 0.0, 200.0, 45.0, 1.0, disabled=not protector_on)

    st.divider()
    field_size = st.number_input("Field size (ha)", 0.1, 5000.0, 10.0, 0.5)

    inputs = Inputs(
        n_rate=n_rate, p_rate=p_rate, moisture_pct=moisture_pct, field_size=field_size,
        barley_type=barley_type, barley_price=barley_price,
        urea_price=urea_price, phosphate_price=phosphate_price,
        inhibitor_on=inhibitor_on, inhibitor_cost_per_t_urea=inhibitor_cost,
        protector_on=protector_on, protector_cost_per_t_phosphate=protector_cost,
        currency=currency,
    )

# ================== Predictions ==================
yield_t_ha, protein_as_is = predict_yield_and_protein_as_is(inputs.n_rate, inputs.p_rate)
protein_dm = as_is_to_dm(protein_as_is, inputs.moisture_pct)
prot_low, prot_high = max(0.0, protein_dm - 0.5), protein_dm + 0.5
starch_low, starch_high = estimate_starch_dm(protein_dm, inputs.protector_on)

# ================== Compact KPI row ==================
st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
st.markdown('<div class="kpi"><h4>Predicted yield</h4><div class="val good">{:.2f} t/ha</div></div>'.format(yield_t_ha), unsafe_allow_html=True)
st.markdown(
    '<div class="kpi"><h4>Predicted protein (DM)</h4>'
    f'<div class="val">{protein_dm:.2f} %</div>'
    f'<div class="muted">Converted from as-is @ {inputs.moisture_pct:.1f}% moisture</div></div>',
    unsafe_allow_html=True
)
st.markdown('<div class="kpi"><h4>Estimated starch (DM)</h4><div class="val">{:.1f}–{:.1f}%</div></div>'.format(starch_low, starch_high), unsafe_allow_html=True)
margin_per_ha, margin_total, _tmp = gross_margin_simple(inputs, yield_t_ha)
st.markdown('<div class="kpi"><h4>Gross margin (per ha)</h4><div class="val">{sym}{val:,.0f}</div></div>'.format(sym=symbol(inputs.currency), val=margin_per_ha), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================== Quality & Economics Sections ==================
c1, c2 = st.columns((1.1, 1.0), gap="large")

with c1:
    st.subheader("Estimated quality (proxy)")
    st.caption("Protein range shown is ±0.5% abs around the model’s point estimate (DM). Starch inversely tracks protein; phosphate protector adds ≈+0.6 pp in this proxy.")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown('<div class="card"><div class="muted">Estimated Protein (DM)</div><div class="val" style="font-weight:800; font-size:26px;">{:.1f}–{:.1f}%</div></div>'.format(prot_low, prot_high), unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="card"><div class="muted">Estimated Starch (DM)</div><div class="val" style="font-weight:800; font-size:26px;">{:.1f}–{:.1f}%</div></div>'.format(starch_low, starch_high), unsafe_allow_html=True)

with c2:
    st.subheader("Gross margin (simple)")
    margin_per_ha, margin_total, breakdown = gross_margin_simple(inputs, yield_t_ha)
    st.dataframe(breakdown, hide_index=True, use_container_width=True)
    st.markdown(
        f'<div class="card" style="margin-top:10px;"><div class="muted">Summary (per ha / field)</div>'
        f'<div style="font-size:22px; font-weight:800;">Per ha: {symbol(inputs.currency)}{margin_per_ha:,.0f} &nbsp;|&nbsp; Field: {symbol(inputs.currency)}{margin_total:,.0f}</div></div>',
        unsafe_allow_html=True
    )

# ================== Azure GenAI (advisor) ==================
st.subheader("Generative AI advisor (optional)")
if AZURE_OK:
    st.caption('✅ Azure OpenAI connected. Try: “What if I reduce N by 15%?”')
else:
    st.caption('🟦 Local mode: predictions run without Azure OpenAI. Enable Azure secrets to use the advisor.')

def call_azure_genai(prompt: str, context: dict) -> str:
    if not AZURE_OK:
        return ("Azure advisor is not connected. Heuristic: lowering N by ~15% typically reduces protein (DM) by ~0.1–0.3 pp; "
                "yield response is season-dependent. Urea savings can improve margin when grain prices are soft. "
                "Protector adds cost but may nudge starch DM upward (~+0.6 pp here).")
    try:
        import requests
        url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
        headers = {"api-key": AZURE_API_KEY, "Content-Type": "application/json"}
        sys_prompt = (
            "You are an agronomy advisor. Be concise and practical. "
            "Use the provided context (yield, protein DM, starch DM band, prices, costs) to reason about margin trade-offs. "
            "Explain DM vs as-is clearly if asked."
        )
        payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "top_p": 0.95,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(Azure call failed; local fallback) {e}"

def sanitize_md(s: str) -> str:
    return s.replace("~", r"\\~")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

user_msg = st.chat_input("Ask a question (e.g., “What if I reduce N by 15%?”)")
if user_msg:
    st.session_state.chat_history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    context = {
        "yield_t_ha": round(yield_t_ha, 2),
        "protein_dm_pct": round(protein_dm, 2),
        "starch_dm_band": (round(starch_low, 1), round(starch_high, 1)),
        "currency": inputs.currency,
        "barley_type": inputs.barley_type,
        "barley_price_per_t": inputs.barley_price,
        "urea_price_per_t": inputs.urea_price,
        "phosphate_price_per_t": inputs.phosphate_price,
        "inhibitor_on": inputs.inhibitor_on,
        "protector_on": inputs.protector_on,
        "moisture_pct": inputs.moisture_pct,
        "n_rate": inputs.n_rate,
        "p_rate": inputs.p_rate,
        "margin_per_ha": round(margin_per_ha, 0),
    }
    reply = call_azure_genai(user_msg, context)
    safe_reply = sanitize_md(reply)
    st.session_state.chat_history.append(("assistant", safe_reply))
    with st.chat_message("assistant"):
        st.markdown(safe_reply)

# ================== Footer ==================
st.markdown(
    f"""
    <div class="muted" style="margin-top:16px;">
      Data note: model training used public agronomy datasets from TEAGASC. Prototype for learning and discussion.
      Always validate with your own field data, local specs, and advisor guidance.
      <br/>
      Built with Python + Streamlit. Optional Azure OpenAI integration for what-if guidance.
      <br/>
      Developed by <a href="{LINKEDIN_URL}" target="_blank" rel="noopener">{AUTHOR_NAME}</a>.
    </div>
    """,
    unsafe_allow_html=True
)
