# app.py
from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import requests
import streamlit as st

# ============= Page setup & styles =============
st.set_page_config(
    page_title="Barley Advisor — Yield & Quality Predictor",
    page_icon="🌾",
    layout="wide",
)

# Global CSS: bigger ask box, nicer cards, subtle hovers, footer, etc.
st.markdown(
    """
    <style>
      /* Tighten the page a bit */
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

      /* Make the chat input larger */
      div[data-testid="stTextInput"] input[type="text"]{
          height: 56px;
          font-size: 18px;
          padding-left: 16px;
      }

      /* Card hover polish for KPI containers */
      .kpi-card:hover {
          transition: box-shadow 0.15s ease-in-out;
          box-shadow: 0 6px 24px rgba(0,0,0,0.06);
      }

      /* Align the advisor header row */
      .advisor-row { display:flex; align-items:center; justify-content:space-between; }
      .advisor-title { font-size: 28px; font-weight: 800; color: #111827; margin: 0.5rem 0 0.75rem; }

      /* Footer */
      .footer-wrap { display:flex; gap:.5rem; align-items:center; justify-content:flex-end; color:#374151; }
      .footer-wrap a { color:#111827; text-decoration:none; font-weight:700; }
      .footer-wrap a:hover { text-decoration:underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============= Model loading =============
@st.cache_resource(show_spinner=False)
def load_model(path: str = "barley_model/model.pkl"):
    return joblib.load(path)

model = None
model_load_error = None
try:
    model = load_model()
except Exception as e:
    model_load_error = str(e)

# The original pipeline expects these columns in this order.
FEATURE_ORDER = [
    "source_doc", "year", "site_id", "end_use",
    "n_rate_kg_ha", "n_split_first_prop", "final_n_timing_gs"
]

# Default hidden values for model compatibility
DEFAULT_SOURCE_DOC = "Focus2022"
DEFAULT_YEAR = 2020
DEFAULT_SITE_ID = "S1"

GS_OPTIONS = {
    "T tillering (GS25)": "GS25",
    "Stem extension (GS31–32)": "GS31",
    "Flag leaf (GS37–39)": "GS39",
    "Ear emergence (GS51–59)": "GS51",
    "Sowing": "sowing",
}
GS_LABELS = list(GS_OPTIONS.keys())

# ============= Helpers =============
def predict_yield_protein(
    end_use: str,
    n_rate_kg_ha: float,
    n_split_first_prop: float,
    final_n_label: str,
    source_doc: str = DEFAULT_SOURCE_DOC,
    year: int = DEFAULT_YEAR,
    site_id: str = DEFAULT_SITE_ID,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Run the sklearn pipeline and return yield (t/ha) and protein (%)"""
    if model is None:
        return None, None, model_load_error or "Model not loaded."

    try:
        final_n_gs = GS_OPTIONS.get(final_n_label, "GS25")
        row = pd.DataFrame([{
            "source_doc": source_doc,
            "year": int(year),
            "site_id": site_id,
            "end_use": end_use,
            "n_rate_kg_ha": float(n_rate_kg_ha),
            "n_split_first_prop": float(n_split_first_prop),
            "final_n_timing_gs": final_n_gs,
        }])[FEATURE_ORDER]

        pred = model.predict(row)
        # Assume model outputs [yield_t_ha, protein_percent] in that order.
        yld, prot = float(pred[0][0]), float(pred[0][1])
        return yld, prot, None
    except Exception as e:
        return None, None, f"Prediction failed: {e}"

# ----- Price fetchers (best-effort with graceful fallback) -----
@st.cache_data(ttl=24*3600, show_spinner=False)
def fetch_best_effort_prices() -> Dict[str, Optional[float]]:
    """
    Try to fetch indicative prices (USD/t) for barley (feed/malting) and urea.
    If online fetch fails, return None and we’ll show manual fallback values/UI.
    """
    prices = {"barley_feed": None, "barley_malting": None, "urea": None}

    # These are "best effort" demo endpoints; they may change or be blocked.
    # We intentionally keep them simple & permissive.
    sources = {
        "urea": "https://prices.openag.io/urea/latest.json",
        "barley_feed": "https://prices.openag.io/barley/feed/latest.json",
        "barley_malting": "https://prices.openag.io/barley/malting/latest.json",
    }
    headers = {"User-Agent": "Mozilla/5.0 (BarleyAdvisor Demo)"}

    for key, url in sources.items():
        try:
            r = requests.get(url, timeout=8, headers=headers)
            if r.ok:
                data = r.json()
                # Expecting {"price_usd_per_tonne": 600} (demo format)
                val = data.get("price_usd_per_tonne")
                if isinstance(val, (int, float)) and val > 0:
                    prices[key] = float(val)
        except Exception:
            pass  # swallow: we’ll use fallback path below

    return prices

def fmt_money(x: float, curr: str = "USD") -> str:
    return f"{curr} {x:,.0f}"

def kpi_card(label: str, value: str, color: str = "#111827"):
    st.markdown(
        f"""
        <div class="kpi-card" style="
            border:1px solid rgba(0,0,0,0.07);
            border-radius:14px;
            padding:16px 18px;
            background: #fff;
        ">
            <div style="font-size:14px; color:#6b7280; margin-bottom:6px;">{label}</div>
            <div style="font-size:36px; font-weight:800; color:{color}; line-height:1.1;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============= Title & intro =============
st.title("🌾 Barley Advisor — Yield & Quality Predictor")
st.caption("Prototype model trained on Teagasc dataset. For demo only — not agronomic advice.")

# ============= Inputs =============
with st.container():
    st.subheader("Inputs")
    col1, col2 = st.columns(2)

    with col1:
        end_use = st.selectbox("End use", ["feed", "malting"], index=0)
        n_rate = st.number_input("Nitrogen rate (kg/ha)", min_value=0.0, max_value=400.0, value=120.0, step=5.0)

    with col2:
        n_split_first = st.number_input(
            "N split — first application proportion",
            min_value=0.0, max_value=1.0, value=0.50, step=0.05,
            help="If you split N, what fraction is applied in the first pass?"
        )

    final_n_timing = st.selectbox("Final N timing", GS_LABELS, index=0)

# Optional: small developer drawer (hidden values)
with st.sidebar:
    st.markdown("### ⚙️ Developer settings")
    with st.expander("Developer settings", expanded=False):
        source_doc = st.selectbox("Source document", ["Focus2022", "Hackett2019", "Hackett2011_14"], index=0)
        site_id = st.text_input("Site ID", value=DEFAULT_SITE_ID)
        year_fix = st.number_input("Year (hidden to user)", min_value=2010, max_value=2030, value=DEFAULT_YEAR, step=1)

# ============= Prediction =============
btn = st.button("Predict", use_container_width=True)
pred_yield, pred_protein = None, None
pred_error = None

if btn:
    pred_yield, pred_protein, pred_error = predict_yield_protein(
        end_use=end_use,
        n_rate_kg_ha=n_rate,
        n_split_first_prop=n_split_first,
        final_n_label=final_n_timing,
        source_doc=source_doc,
        year=year_fix,
        site_id=site_id
    )

if pred_error:
    st.error(pred_error)

if pred_yield is not None and pred_protein is not None and pred_error is None:
    st.success(f"**Predicted yield:** {pred_yield:0.2f} t/ha")
    st.success(f"**Predicted grain protein:** {pred_protein:0.2f} %")

# ============= Simple gross margin =============
st.markdown("### Gross margin (simple)")

# Fetch “best effort” online prices (cached), then allow manual override.
price_data = fetch_best_effort_prices()
# Fallback defaults if online fetch missing:
fallbacks = {
    "barley_feed": 190.0,       # USD/t
    "barley_malting": 220.0,    # USD/t
    "urea": 600.0,              # USD/t
}

# Pick barley price by end use
barley_price_online = price_data.get(f"barley_{end_use}") or None
urea_price_online = price_data.get("urea") or None

cur = "USD"  # display label only

# Manual overrides (small)
ov1, ov2 = st.columns(2)
with ov1:
    barley_price = st.number_input(
        f"Barley price ({end_use}) [USD/t]",
        value=float(barley_price_online or fallbacks[f"barley_{end_use}"]),
        min_value=0.0, step=5.0, key="barley_price_override"
    )
with ov2:
    urea_price = st.number_input(
        "Urea price [USD/t]",
        value=float(urea_price_online or fallbacks["urea"]),
        min_value=0.0, step=5.0, key="urea_price_override"
    )

# Notes on whether we used online or manual
notes = []
if barley_price_online is None:
    notes.append(f"• Barley manual fallback: {fallbacks[f'barley_{end_use}']:.0f}/t")
if urea_price_online is None:
    notes.append(f"• Urea manual fallback: {fallbacks['urea']:.0f}/t")
if notes:
    st.write("\n".join(notes))

# Compute revenue & cost if we have a predicted yield
if pred_yield is not None and pred_error is None:
    # Revenue
    revenue_ha = pred_yield * barley_price  # USD/ha

    # N cost: assume urea is 46% N
    if n_rate > 0:
        urea_needed_kg = n_rate / 0.46
        n_cost_ha = (urea_needed_kg / 1000.0) * urea_price
    else:
        n_cost_ha = 0.0

    gross_margin_ha = revenue_ha - n_cost_ha

    # Color-coded KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Revenue/ha", fmt_money(revenue_ha, cur), "#16a34a")  # green
    with c2:
        kpi_card("N cost/ha", fmt_money(n_cost_ha, cur), "#dc2626")    # red
    with c3:
        gm_color = "#16a34a" if gross_margin_ha >= 0 else "#dc2626"
        kpi_card("Gross margin/ha", fmt_money(gross_margin_ha, cur), gm_color)
else:
    st.info("Enter inputs and click **Predict** to see the gross margin KPIs.")

# ============= Ask the advisor (Azure OpenAI) =============
st.markdown("## Ask the advisor")

col_left, col_right = st.columns([3, 1])  # wide left, narrow right
with col_left:
    st.markdown("###")  # spacer so alignment is nice

with col_right:
    with st.expander("Advisor settings (optional)", expanded=False):
        temperature = st.slider(
            "Response style",
            min_value=0.0, max_value=1.0, step=0.05, value=0.20,
            help="AI generation setting (not weather): lower = more concise/grounded, higher = more exploratory."
        )

# Input line under the header
user_q = st.text_input(
    "Ask a question (e.g., 'What if I reduce N by 15%?')",
    label_visibility="collapsed",
    key="ask_box"
)


def azure_chat_completion(
    prompt: str,
    context: Dict[str, str],
    temperature: float = 0.2
) -> Tuple[Optional[str], Optional[str]]:
    """Call Azure OpenAI if configured; return (answer, error)."""

    # Check secrets
    if "azure" not in st.secrets:
        return None, "Azure OpenAI not configured. Add [azure] api_key, endpoint, deployment, api_version in Secrets."

    az = st.secrets["azure"]
    required = ["api_key", "endpoint", "deployment", "api_version"]
    if any(k not in az or not az[k] for k in required):
        return None, "Azure OpenAI not configured. Missing one of api_key, endpoint, deployment, api_version."

    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=az["api_key"],
            azure_endpoint=az["endpoint"],
            api_version=az["api_version"],
        )

        sys_prompt = (
            "You are a cautious agronomy assistant for barley. "
            "Use short, practical notes. Avoid unsafe, off-label, or definitive agronomy instructions. "
            "If uncertain, suggest consulting a local agronomist. "
            "When discussing nitrogen, relate to rate/splits/timing at a high level and mention potential trade-offs "
            "(yield, protein, lodging, quality)."
        )

        # Build context string
        ctx_lines = []
        for k, v in context.items():
            ctx_lines.append(f"- {k}: {v}")
        ctx_text = "Context:\n" + "\n".join(ctx_lines)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{ctx_text}\n\nUser question: {prompt}"},
        ]

        resp = client.chat.completions.create(
            model=az["deployment"],
            messages=messages,
            temperature=float(temperature),
            max_tokens=400,
        )
        answer = resp.choices[0].message.content
        return answer, None
    except Exception as e:
        return None, f"Azure call failed: {e}"

if user_q:
    ctx = {
        "end_use": end_use,
        "N rate (kg/ha)": f"{n_rate}",
        "N split first prop": f"{n_split_first}",
        "Final N timing": final_n_timing,
    }
    if pred_yield is not None and pred_error is None:
        ctx["Predicted yield (t/ha)"] = f"{pred_yield:0.2f}"
        ctx["Predicted protein (%)"] = f"{pred_protein:0.2f}"
    if 'barley_price' in locals():
        ctx["Barley price (USD/t)"] = f"{barley_price:0.0f}"
    if 'urea_price' in locals():
        ctx["Urea price (USD/t)"] = f"{urea_price:0.0f}"

    with st.spinner("Thinking..."):
        answer, err = azure_chat_completion(user_q, ctx, temperature=temperature)
    if err:
        with st.expander("Error details"):
            st.error(err)
    elif answer:
        st.info(answer)

# ============= Footer =============
st.markdown("---")
footer_cols = st.columns([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
with footer_cols[-4]:
    st.caption("Version: demo")
with footer_cols[-1]:
    st.markdown(
        """
        <div class="footer-wrap">
          <span>Developed by</span>
          <a href="https://www.linkedin.com/in/nikolaygeorgiev/" target="_blank" rel="noopener">Nikolay Georgiev</a>
          <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg"
               alt="LinkedIn" width="16" height="16" />
        </div>
        """,
        unsafe_allow_html=True
    )

