# app.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ================== Page setup & styles ==================
st.set_page_config(
    page_title="Barley Advisor — Yield & Quality Predictor",
    page_icon="🌾",
    layout="wide",
)

# ---------- CSS ----------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1300px;}
      .top-banner {
        display:flex; justify-content:space-between; align-items:center;
        font-size:14px; padding:8px 12px; border-radius:10px;
        background:#f3f8ff; border:1px solid #e3ecff; margin-bottom:8px;
      }
      .kpi-grid {display:grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap:14px;}
      .kpi {border:1px solid #eef1f6; border-radius:14px; padding:14px; background:white; box-shadow:0 1px 2px rgba(0,0,0,0.03);}
      .kpi h4 {margin:0 0 6px 0; font-size:13px; color:#667085; font-weight:600;}
      .kpi .val {font-size:26px; font-weight:800; letter-spacing:-0.3px;}
      .good {color:#117a37;}
      .warn {color:#b54708;}
      .bad {color:#b42318;}
      .pill {display:inline-block; padding:3px 8px; border-radius:999px; background:#eef6ff; border:1px solid #dbeafe; font-size:12px;}
      .card {border:1px solid #eef1f6; border-radius:16px; padding:16px; background:white;}
      .muted {color:#667085; font-size:12px;}
      .help {font-size:12px; color:#475467;}
      .successbox {background:#edfdf3; border:1px solid #d1fadf; padding:10px 12px; border-radius:10px; margin-bottom:8px;}
      .successbox strong {color:#05603a;}
      .infobox {background:#eff8ff; border:1px solid #d1e9ff; padding:10px 12px; border-radius:10px; margin-bottom:8px;}
    </style>
    """,
    unsafe_allow_html=True
)

# One-line banner (prevents wrapping on small glitches)
st.markdown(
    """
    <div class="top-banner">
      <div><strong>Barley Advisor</strong> — demo prototype connecting ML predictions with an Azure Generative AI advisor</div>
      <div class="pill">Demo version — N. Georgiev</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== Helpers & defaults ==================
MODEL_PATH = "models/barley_model.pkl"

@dataclass
class Inputs:
    n_rate: float             # kg N/ha (as urea-N equivalent)
    p_rate: float             # kg P2O5/ha
    moisture_pct: float       # grain moisture at measurement (%)
    field_size: float         # ha
    barley_price: float       # price per tonne of barley (malting) in selected currency
    urea_price: float         # price per tonne of urea
    phosphate_price: float    # price per tonne of phosphate (P2O5 carrier or proxy)
    inhibitor_on: bool
    inhibitor_cost_per_t_urea: float
    protector_on: bool
    protector_cost_per_t_phosphate: float
    currency: str

CURRENCY_SYMBOLS = {
    "EUR": "€",
    "USD": "$",
    "CHF": "CHF"
}

def symbol(cur: str) -> str:
    return CURRENCY_SYMBOLS.get(cur, cur)

def load_model():
    """Load your trained model. If not available, use a transparent fallback."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    # Fallback: a simple, explainable pseudo-model to keep the app running.
    class FallbackModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            # Toy function: yield driven by N rate with diminishing returns; protein as-is correlates with N
            n = X["n_rate"].values
            base_yield = 6.5 + 0.035 * n - 0.00007 * (n ** 2)  # t/ha
            base_yield = np.clip(base_yield, 3.5, 12.0)
            protein_as_is = 7.5 + 0.010 * n  # %
            protein_as_is = np.clip(protein_as_is, 7.0, 12.0)
            return np.c_[base_yield, protein_as_is]
    return FallbackModel()

MODEL = load_model()

def predict_yield_and_protein_as_is(n_rate: float, p_rate: float) -> Tuple[float, float]:
    X = pd.DataFrame({"n_rate": [n_rate], "p_rate": [p_rate]})
    out = MODEL.predict(X)
    # Expect array with two columns: [yield_t_ha, protein_as_is_pct]
    yld, prot_as_is = float(out[0, 0]), float(out[0, 1])
    return yld, prot_as_is

def as_is_to_dm(protein_as_is_pct: float, moisture_pct: float) -> float:
    """Convert protein measured on wet (as-is) basis to dry matter basis."""
    m = max(0.0, min(40.0, moisture_pct)) / 100.0
    return protein_as_is_pct / (1.0 - m)

def dm_to_as_is(protein_dm_pct: float, moisture_pct: float) -> float:
    m = max(0.0, min(40.0, moisture_pct)) / 100.0
    return protein_dm_pct * (1.0 - m)

def estimate_starch_dm(protein_dm_pct: float, protector_on: bool) -> Tuple[float, float]:
    """
    Very simple proxy: starch and protein often move inversely.
    Start from 64.0% starch DM at 10% protein DM and adjust slope modestly.
    Protector (if ON) nudges starch up by +0.6% abs. (tunable)
    """
    base = 64.0 - 0.7 * (protein_dm_pct - 10.0)
    if protector_on:
        base += 0.6
    # Provide a band ±0.8 abs to reflect uncertainty
    return max(55.0, min(72.0, base - 0.8)), max(55.0, min(72.0, base + 0.8))

def gross_margin_simple(inp: Inputs, yield_t_ha: float) -> Tuple[float, float, pd.DataFrame]:
    """
    Revenue = yield * barley_price
    Cost = urea + phosphate (+ inhibitors if toggled)
    """
    cur = symbol(inp.currency)

    # --- Urea ---
    # Assume urea product contains 46% N. Convert kg N/ha to kg urea/ha.
    kg_urea_per_ha = inp.n_rate / 0.46
    t_urea_per_ha = kg_urea_per_ha / 1000.0
    urea_cost = t_urea_per_ha * inp.urea_price

    inhibitor_cost = 0.0
    if inp.inhibitor_on:
        inhibitor_cost = t_urea_per_ha * inp.inhibitor_cost_per_t_urea

    # --- Phosphate ---
    # Simplify: treat p_rate as kg P2O5/ha; convert to tonnes of product proxy
    t_phosphate_per_ha = (inp.p_rate / 1000.0)
    phosphate_cost = t_phosphate_per_ha * inp.phosphate_price

    protector_cost = 0.0
    if inp.protector_on:
        protector_cost = t_phosphate_per_ha * inp.protector_cost_per_t_phosphate

    # --- Revenue & Margin ---
    revenue = yield_t_ha * inp.barley_price
    variable_costs = urea_cost + inhibitor_cost + phosphate_cost + protector_cost
    margin_per_ha = revenue - variable_costs
    margin_total = margin_per_ha * inp.field_size

    # breakdown table
    rows = [
        ("Revenue", f"{cur}{revenue:,.0f} /ha", ""),
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

    col_a, col_b = st.columns(2)
    with col_a:
        n_rate = st.number_input("Nitrogen rate (kg N/ha)", min_value=0.0, max_value=400.0, value=160.0, step=5.0)
        barley_price = st.number_input(f"Barley price (malting) [{cur}/t]", min_value=50.0, max_value=1000.0, value=320.0, step=5.0)
        moisture_pct = st.number_input("Grain moisture at measurement (%)", min_value=8.0, max_value=28.0, value=12.0, step=0.5,
                                       help="Used to convert protein from as-is to dry matter. All displays use DM.")
    with col_b:
        p_rate = st.number_input("Phosphate rate (kg P₂O₅/ha)", min_value=0.0, max_value=200.0, value=60.0, step=5.0)
        urea_price = st.number_input(f"Urea price [{cur}/t]", min_value=100.0, max_value=1500.0, value=450.0, step=10.0)
        phosphate_price = st.number_input(f"Phosphate price [{cur}/t]", min_value=100.0, max_value=1500.0, value=700.0, step=10.0)

    st.divider()
    st.subheader("Treatment options")
    inhibitor_on = st.toggle("Use nitrogen inhibitor", value=False, help=f"Adds cost per tonne of urea. Default {cur}35/t.")
    inhibitor_cost = st.number_input(f"Inhibitor cost [{cur}/t urea]", min_value=0.0, max_value=200.0, value=35.0, step=1.0, disabled=not inhibitor_on)

    protector_on = st.toggle("Use phosphate protector", value=False, help=f"Adds cost per tonne of phosphate and nudges starch DM.")
    protector_cost = st.number_input(f"Protector cost [{cur}/t phosphate]", min_value=0.0, max_value=200.0, value=45.0, step=1.0, disabled=not protector_on)

    st.divider()
    field_size = st.number_input("Field size (ha)", min_value=0.1, max_value=5000.0, value=10.0, step=0.5)

    inputs = Inputs(
        n_rate=n_rate, p_rate=p_rate, moisture_pct=moisture_pct, field_size=field_size,
        barley_price=barley_price, urea_price=urea_price, phosphate_price=phosphate_price,
        inhibitor_on=inhibitor_on, inhibitor_cost_per_t_urea=inhibitor_cost,
        protector_on=protector_on, protector_cost_per_t_phosphate=protector_cost,
        currency=currency,
    )

# ================== Predictions ==================
yield_t_ha, protein_as_is = predict_yield_and_protein_as_is(inputs.n_rate, inputs.p_rate)
protein_dm = as_is_to_dm(protein_as_is, inputs.moisture_pct)

# Uncertainty bands (simple, adjustable)
prot_low, prot_high = max(0.0, protein_dm - 0.5), protein_dm + 0.5
starch_low, starch_high = estimate_starch_dm(protein_dm, inputs.protector_on)

# ================== Header KPIs ==================
st.markdown('<div class="successbox"><strong>All protein values shown are on a Dry Matter (DM) basis.</strong> (Converted from model as-is using your moisture input.)</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="kpi"><h4>Predicted yield</h4><div class="val good">{:.2f} t/ha</div></div>'.format(yield_t_ha), unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi"><h4>Predicted grain protein (DM)</h4><div class="val">{:.2f} %</div><div class="muted">Moisture assumed: {:.1f}%</div></div>'.format(protein_dm, inputs.moisture_pct), unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi"><h4>Estimated starch (DM)</h4><div class="val">{:.1f}–{:.1f}%</div></div>'.format(starch_low, starch_high), unsafe_allow_html=True)
with k4:
    margin_per_ha, margin_total, _tmp = gross_margin_simple(inputs, yield_t_ha)
    st.markdown('<div class="kpi"><h4>Gross margin (per ha)</h4><div class="val">{sym}{val:,.0f}</div></div>'.format(sym=symbol(inputs.currency), val=margin_per_ha), unsafe_allow_html=True)

# ================== Quality & Economics Sections ==================
c1, c2 = st.columns((1.1, 1.0), gap="large")

with c1:
    st.subheader("Estimated quality (proxy)")
    st.caption("Protein range is ±0.5% abs around prediction (DM). Starch proxy is a simple inverse relation to protein; phosphate protector adds +0.6% abs.")
    qc1, qc2 = st.columns(2)
    with qc1:
        st.markdown(
            '<div class="card"><div class="muted">Estimated Protein (DM)</div>'
            f'<div class="val" style="font-weight:800; font-size:28px;">{prot_low:.1f}–{prot_high:.1f}%</div></div>',
            unsafe_allow_html=True
        )
    with qc2:
        st.markdown(
            '<div class="card"><div class="muted">Estimated Starch (DM)</div>'
            f'<div class="val" style="font-weight:800; font-size:28px;">{starch_low:.1f}–{starch_high:.1f}%</div></div>',
            unsafe_allow_html=True
        )
    st.markdown(
        '<div class="infobox">Note: Model protein is trained on as-is measurements. The app converts to DM using your moisture input to keep the basis consistent.</div>',
        unsafe_allow_html=True
    )

with c2:
    st.subheader("Gross margin (simple)")
    margin_per_ha, margin_total, breakdown = gross_margin_simple(inputs, yield_t_ha)
    st.dataframe(
        breakdown,
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(
        f'<div class="card" style="margin-top:10px;"><div class="muted">Summary</div>'
        f'<div style="font-size:24px; font-weight:800;">Per ha: {symbol(inputs.currency)}{margin_per_ha:,.0f} &nbsp;&nbsp;|&nbsp;&nbsp; Field: {symbol(inputs.currency)}{margin_total:,.0f}</div></div>',
        unsafe_allow_html=True
    )

# ================== What-if Advisor (Azure GenAI) ==================
st.subheader("Ask the Generative AI advisor")
st.caption("Powered by Azure OpenAI when configured. Try: *“What if I reduce N by 15%?”* or *“How does a phosphate protector impact starch and margin?”*")

# Env-based configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

def call_azure_genai(prompt: str, context: dict) -> str:
    """
    If Azure env vars are present, call the chat completion endpoint.
    Otherwise, return a local heuristic explanation so the UI still works.
    """
    if AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_KEY:
        try:
            # Lazy import to avoid dependency if not configured
            import requests
            url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
            headers = {"api-key": AZURE_API_KEY, "Content-Type": "application/json"}
            sys_prompt = (
                "You are an agronomy advisor. Be concise and practical. "
                "Use the provided context (yield, protein DM, starch DM band, prices, costs) to reason about margin trade-offs. "
                "If the user asks about DM vs as-is, explain clearly."
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
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"(Azure call failed, falling back locally) {e}\n\nGiven your inputs, here’s a quick heuristic: reducing N typically lowers protein and sometimes yield; margin impact depends on urea & barley price spread."
    # Local fallback
    return (
        "Azure advisor not configured. Heuristic guidance:\n"
        "- Lowering N by ~15% often reduces protein (DM) by ~0.1–0.3 abs and yield slightly; savings in urea may improve margin when grain prices are weak.\n"
        "- Phosphate protector adds a small cost but can nudge starch DM upward (~+0.6 abs here), potentially helpful for malting specs.\n"
        "Configure Azure env vars to enable full Generative AI responses."
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show prior messages
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
    st.session_state.chat_history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)

# ================== Footer ==================
st.markdown(
    """
    <div class="muted" style="margin-top:24px;">
      Data note: model training used public agronomy datasets from TEAGASC. This is a prototype for learning and discussion.
      Always validate with your own field data, local specs, and advisor guidance.
    </div>
    """,
    unsafe_allow_html=True
)
