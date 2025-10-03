# app.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ================== Small compat helpers ==================
def st_divider():
    """Safe divider for older/newer Streamlit versions."""
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.write("---")

# ================== Author / Branding ==================
AUTHOR_NAME = "Nikolay Georgiev"
LINKEDIN_URL = "https://www.linkedin.com/in/nikolaygeorgiev/"
APP_VERSION = "v2.0 (Oct 2025)"

# ================== Page setup & styles ==================
st.set_page_config(page_title="Barley Advisor — Yield & Quality Predictor", page_icon="🌾", layout="wide")

# ---------- CSS (compact header + v2 summary styling) ----------
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

      .summary-bar {position: sticky; top: 0; z-index: 50; background:#ffffffcc; backdrop-filter: blur(6px);
        padding: 8px 12px; border:1px solid #eef1f6; border-radius:12px; margin-bottom:8px;}
      .metric-box {border:1px solid #eef1f6; border-radius:10px; padding:8px 10px; background:white;}
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
      <div><strong>Barley Advisor</strong> — ML predictions + Azure Generative AI advisor <span class="muted">· {APP_VERSION}</span></div>
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

# ================== Core predictions for current Advisor ==================
yield_t_ha, protein_as_is = predict_yield_and_protein_as_is(inputs.n_rate, inputs.p_rate)
protein_dm = as_is_to_dm(protein_as_is, inputs.moisture_pct)
prot_low, prot_high = max(0.0, protein_dm - 0.5), protein_dm + 0.5
starch_low, starch_high = estimate_starch_dm(protein_dm, inputs.protector_on)
margin_per_ha, margin_total, _tmp = gross_margin_simple(inputs, yield_t_ha)

# ================== V2 Tabs ==================
st_divider()
st.markdown("## 🌾 Barley Advisor v2.0 — New Features")

tab_current, tab_compare, tab_econ, tab_model = st.tabs(
    ["Advisor (current)", "Scenario Compare (NEW)", "Economics (NEW)", "Model Card"]
)

# ---------- Sticky summary (shows current advisor quick facts) ----------
with tab_current:
    st.markdown(
        f"""
        <div class="summary-bar">
          <b>Summary:</b>
          <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin-top:6px;">
            <div class="metric-box"><div class="muted">Yield</div><div style="font-weight:800;">{yield_t_ha:.2f} t/ha</div></div>
            <div class="metric-box"><div class="muted">Protein (DM)</div><div style="font-weight:800;">{protein_dm:.2f} %</div></div>
            <div class="metric-box"><div class="muted">Starch (DM est.)</div><div style="font-weight:800;">{starch_low:.1f}–{starch_high:.1f} %</div></div>
            <div class="metric-box"><div class="muted">Margin (per ha)</div><div style="font-weight:800;">{symbol(inputs.currency)}{margin_per_ha:,.0f}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===== your original KPI row =====
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    st.markdown('<div class="kpi"><h4>Predicted yield</h4><div class="val good">{:.2f} t/ha</div></div>'.format(yield_t_ha), unsafe_allow_html=True)
    st.markdown(
        '<div class="kpi"><h4>Predicted protein (DM)</h4>'
        f'<div class="val">{protein_dm:.2f} %</div>'
        f'<div class="muted">Converted from as-is @ {inputs.moisture_pct:.1f}% moisture</div></div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="kpi"><h4>Estimated starch (DM)</h4><div class="val">{:.1f}–{:.1f}%</div></div>'.format(starch_low, starch_high), unsafe_allow_html=True)
    st.markdown('<div class="kpi"><h4>Gross margin (per ha)</h4><div class="val">{sym}{val:,.0f}</div></div>'.format(sym=symbol(inputs.currency), val=margin_per_ha), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== Quality & Economics Sections =====
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

# ---------- Scenario Compare (NEW) ----------
with tab_compare:
    st.subheader("Scenario Compare (A vs B)")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Scenario A")
        nA = st.slider("N rate A (kg/ha)", 60, 240, 160, 10)
        pA = st.slider("P rate A (kg/ha)", 0, 120, 60, 10)
        yA, protA_as_is = predict_yield_and_protein_as_is(nA, pA)
        protA_dm = as_is_to_dm(protA_as_is, inputs.moisture_pct)
    with colB:
        st.markdown("#### Scenario B")
        nB = st.slider("N rate B (kg/ha)", 60, 240, 180, 10)
        pB = st.slider("P rate B (kg/ha)", 0, 120, 60, 10)
        yB, protB_as_is = predict_yield_and_protein_as_is(nB, pB)
        protB_dm = as_is_to_dm(protB_as_is, inputs.moisture_pct)

    st.markdown("#### Results (B vs A)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Yield B (t/ha)", f"{yB:.2f}", f"{yB - yA:+.2f}")
    c2.metric("Protein B (DM, %)", f"{protB_dm:.2f}", f"{protB_dm - protA_dm:+.2f}")
    # quick econ compare using your current prices
    inputs_A = Inputs(**{**inputs.__dict__, "n_rate": nA, "p_rate": pA})
    inputs_B = Inputs(**{**inputs.__dict__, "n_rate": nB, "p_rate": pB})
    mA, _, _ = gross_margin_simple(inputs_A, yA)
    mB, _, _ = gross_margin_simple(inputs_B, yB)
    c3.metric(f"Margin Δ ({symbol(inputs.currency)}/ha)", f"{mB:,.0f}", f"{(mB - mA):+,.0f}")

# ---------- Economics (NEW) ----------
with tab_econ:
    st.subheader("Economics & Sustainability")
    price_per_t = st.number_input("Grain price (€/t)", 80.0, 400.0, float(inputs.barley_price), step=5.0)
    n_price_per_kg = st.number_input("N price (€/kg N)", 0.5, 3.0, 1.2, step=0.05)
    co2_factor = st.number_input("CO₂e factor (kg CO₂e/kg N)", 3.6, 9.0, 6.5, step=0.1)
    inhibitor_prem = st.number_input("Inhibitor premium (€/ha)", 0.0, 80.0, 25.0, step=1.0)

    # Simple econ recalc using current yield and your N
    # (kept intentionally simple to avoid changing your gross_margin_simple)
    revenue = yA * price_per_t  # use A as reference here
    n_cost = inputs.n_rate * n_price_per_kg
    inhibitor_cost = inhibitor_prem if inputs.inhibitor_on else 0.0
    gross_margin = revenue - (n_cost + inhibitor_cost)

    em_kg = inputs.n_rate * co2_factor
    e1, e2, e3 = st.columns(3)
    e1.metric("Revenue A (€/ha)", f"{revenue:,.0f}")
    e2.metric("Margin A (€/ha)", f"{gross_margin:,.0f}")
    e3.metric("Emissions A (kg CO₂e/ha)", f"{em_kg:,.0f}")

    st.caption("Note: Above is a compact economics view. Full per-item breakdown is in the *Advisor (current)* tab.")

    # --- Export simple report (HTML) ---
    if st.button("⬇️ Download simple HTML report"):
        report = f"""
        <html><head><meta charset='utf-8'><title>Barley Advisor v2 Report</title></head><body>
        <h2>Barley Advisor — {APP_VERSION}</h2>
        <p><b>Inputs</b>: N={inputs.n_rate} kg/ha, P={inputs.p_rate} kg/ha, Moisture={inputs.moisture_pct}%</p>
        <p><b>Results</b>: Yield={yield_t_ha:.2f} t/ha, Protein (DM)={protein_dm:.2f}%,
           Margin/ha={symbol(inputs.currency)}{margin_per_ha:,.0f}</p>
        <p><b>Economics (compact)</b>: Grain price={price_per_t} €/t, N price={n_price_
