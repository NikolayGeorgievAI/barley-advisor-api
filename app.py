# app.py — Barley Advisor with live(ish) price feeds + gross margin
import os
import math
import json
import typing as t
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============== Page config & light CSS =================
st.set_page_config(page_title="Barley Advisor — Yield & Quality", page_icon="🌾", layout="wide")
st.markdown(
    """
    <style>
      .advisor-card { background: #f8f9fc; border: 1px solid #eef0f6; padding: 18px; border-radius: 14px; }
      .metric-box { padding:12px 14px; border:1px solid #eef0f6; border-radius:12px; background:#fff; }
      .right-rail { border: 1px solid #eef0f6; border-radius: 12px; padding: 12px; background: #fff; }
      .footer a { text-decoration: none; }
      .footer img { width:18px; height:18px; margin-left: 6px; vertical-align: -3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============== Model loading =================
MODEL_PATH = os.getenv("MODEL_PATH", "barley_model/model.pkl")
EXPECTED_COLS = [
    "source_doc", "year", "site_id", "end_use",
    "n_rate_kg_ha", "n_split_first_prop", "final_n_timing_gs"
]
HIDDEN = dict(source_doc="Focus2022", year=2020, site_id="S1")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

MODEL = None
MODEL_ERR = None
try:
    MODEL = load_model(MODEL_PATH)
except Exception as e:
    MODEL_ERR = str(e)

# ============== Price fetchers (live-ish) =================
@dataclass
class PriceFetchSpec:
    url: str
    kind: str               # "csv" | "excel" | "json"
    sheet: str | None = None
    price_col: str | None = None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_csv_or_excel(spec: PriceFetchSpec) -> float | None:
    """Generic CSV/XLSX fetch: returns the last non-NaN value from the specified column (or first numeric col)."""
    try:
        if spec.kind == "csv":
            df = pd.read_csv(spec.url)
        else:
            df = pd.read_excel(spec.url, sheet_name=spec.sheet) if spec.sheet else pd.read_excel(spec.url)
        if spec.price_col and spec.price_col in df.columns:
            s = pd.to_numeric(df[spec.price_col], errors="coerce").dropna()
        else:
            # find first numeric column
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return None
            s = pd.to_numeric(df[num_cols[-1]], errors="coerce").dropna()
        return float(s.iloc[-1]) if not s.empty else None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_urea_from_tradingeconomics() -> float | None:
    """
    TradingEconomics API (public demo creds 'guest:guest').
    Returns latest Urea price (USD per metric ton).
      Docs: https://api.tradingeconomics.com/
      Example: https://api.tradingeconomics.com/commodity/urea?c=guest:guest&format=json
    """
    try:
        url = "https://api.tradingeconomics.com/commodity/urea?c=guest:guest&format=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Usually returns a list; take last/first with a "Last" or "Price" field
        if isinstance(data, list) and data:
            rec = data[0]
            val = rec.get("Last") or rec.get("Price") or rec.get("Value")
            if val is not None:
                return float(val)
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_barley_from_fred() -> float | None:
    """
    FRED: Global price of Barley (PBARLUSDM) — USD per metric ton, monthly.
    Stable CSV: https://fred.stlouisfed.org/series/PBARLUSDM/downloaddata/PBARLUSDM.csv
    """
    try:
        url = "https://fred.stlouisfed.org/series/PBARLUSDM/downloaddata/PBARLUSDM.csv"
        df = pd.read_csv(url)
        s = pd.to_numeric(df["PBARLUSDM"], errors="coerce").dropna()
        return float(s.iloc[-1]) if not s.empty else None
    except Exception:
        return None

def urea_cost_per_kgN(urea_price_per_tonne: float) -> float:
    """Convert urea price (per tonne) to cost per kg of N (urea = 46% N)."""
    if urea_price_per_tonne <= 0: 
        return 0.0
    return (urea_price_per_tonne / 1000.0) / 0.46

# ============== Azure OpenAI helper (optional) =================
def get_azure_client():
    try:
        from openai import AzureOpenAI
    except Exception:
        return None, "openai SDK not installed"
    cfg = st.secrets.get("azure", {})
    req = ("api_key", "endpoint", "deployment", "api_version")
    if not all(k in cfg and cfg[k] for k in req):
        return None, "Azure OpenAI not configured"
    try:
        client = AzureOpenAI(
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
            azure_endpoint=cfg["endpoint"],
        )
        return client, None
    except Exception as e:
        return None, f"Azure init error: {e}"

def ask_advisor(system_msg: str, user_msg: str, temperature: float) -> str:
    client, err = get_azure_client()
    if err or not client:
        raise RuntimeError(err or "Azure not available")
    resp = client.chat.completions.create(
        model=st.secrets["azure"]["deployment"],
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=max(0.0, min(1.0, float(temperature))),
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()

# ============== UI: Title =================
st.title("🌾 Barley Advisor — Yield & Quality")
st.caption("Prototype model trained on Teagasc data. For demo only — not agronomic advice.")

# ============== Inputs =================
c1, c2 = st.columns(2)
with c1:
    n_rate = st.number_input("Nitrogen rate (kg/ha)", 0.0, 300.0, 120.0, 1.0)
    n_split = st.number_input("N split — first application proportion", 0.0, 1.0, 0.50, 0.05,
                              help="Fraction of total N applied at the first pass (e.g., 0.50 = 50%).")
with c2:
    end_use = st.selectbox("End use", ["feed", "malting"], index=0)
    final_n_label = st.selectbox("Final N timing (GS)", 
                                 ["Tillering (GS25)", "Stem elongation (GS31)", "Flag leaf (GS37)", "Emergence", "Sowing"], index=0)

def gs_code(lbl: str) -> str:
    s = lbl.lower()
    if "gs25" in s or "tiller" in s: return "GS25"
    if "gs31" in s or "stem" in s: return "GS31"
    if "gs37" in s or "flag" in s: return "GS37"
    if "emerg" in s: return "emergence"
    if "sow" in s: return "sowing"
    return "GS25"

def predict(end_use: str, n_rate: float, n_split: float, final_label: str) -> tuple[float, float]:
    if MODEL_ERR: raise RuntimeError(MODEL_ERR)
    row = {
        "source_doc": HIDDEN["source_doc"],
        "year": HIDDEN["year"],
        "site_id": HIDDEN["site_id"],
        "end_use": end_use,
        "n_rate_kg_ha": float(n_rate),
        "n_split_first_prop": float(n_split),
        "final_n_timing_gs": gs_code(final_label),
    }
    X = pd.DataFrame([row], columns=EXPECTED_COLS)
    y = MODEL.predict(X)  # [[yield, protein]] expected
    return float(y[0][0]), float(y[0][1])

pred_btn = st.button("Predict", type="primary")
yld, prot = None, None
if pred_btn:
    try:
        yld, prot = predict(end_use, n_rate, n_split, final_n_label)
        a, b = st.columns(2)
        with a:
            st.markdown(f"<div class='metric-box'><b>Predicted yield:</b> {yld:.2f} t/ha</div>", unsafe_allow_html=True)
        with b:
            st.markdown(f"<div class='metric-box'><b>Predicted grain protein:</b> {prot:.2f} %</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============== Sidebar: data source overrides =================
with st.sidebar:
    st.subheader("Live price sources (optional overrides)")
    st.caption("App auto-uses TradingEconomics (Urea) & FRED (Barley). Override below if you prefer a CSV/XLSX URL.")

    with st.expander("Override Urea source (CSV/XLSX)", expanded=False):
        urea_url = st.text_input("Urea price URL")
        urea_kind = st.radio("File type", ["csv", "excel"], horizontal=True, index=0, key="u_kind")
        urea_sheet = st.text_input("Excel sheet (optional)")
        urea_col = st.text_input("Price column (optional)")

    with st.expander("Override Barley source (CSV/XLSX)", expanded=False):
        barley_url = st.text_input("Barley price URL")
        barley_kind = st.radio("File type ", ["csv", "excel"], horizontal=True, index=0, key="b_kind")
        barley_sheet = st.text_input("Excel sheet (optional)")
        barley_col = st.text_input("Price column (optional)")

    st.markdown("---")
    st.caption("If overrides are empty or fail, the app uses TradingEconomics (Urea) and FRED (Barley).")

# ============== Resolve prices (live -> override -> manual) =================
st.markdown("### Gross margin (simple)")

# Manual fallback values (currency-agnostic numbers; choose € as display by default)
barley_manual = st.number_input("Manual barley price (per tonne)", 0.0, 2000.0, 190.0, 1.0, key="b_manual")
urea_manual   = st.number_input("Manual urea price (per tonne)", 0.0, 4000.0, 600.0, 1.0, key="u_manual")
currency = st.selectbox("Currency display", ["€", "$", "£"], index=0)

# Try live fetches
barley_price = None
urea_price = None
notes = []

# 1) try overrides
if barley_url:
    v = fetch_csv_or_excel(PriceFetchSpec(barley_url, barley_kind, barley_sheet or None, barley_col or None))
    if v: barley_price = float(v); notes.append(f"Barley price fetched from override URL: {barley_price:.2f}/t")
if urea_url:
    v = fetch_csv_or_excel(PriceFetchSpec(urea_url, urea_kind, urea_sheet or None, urea_col or None))
    if v: urea_price = float(v); notes.append(f"Urea price fetched from override URL: {urea_price:.2f}/t")

# 2) if missing, use defaults (live-ish)
if barley_price is None:
    v = fetch_barley_from_fred()
    if v: barley_price = float(v); notes.append(f"Barley price from FRED (PBARLUSDM): {barley_price:.2f} USD/t")
if urea_price is None:
    v = fetch_urea_from_tradingeconomics()
    if v: urea_price = float(v); notes.append(f"Urea price from TradingEconomics: {urea_price:.2f} USD/t")

# 3) fallback to manual
if barley_price is None:
    barley_price = float(barley_manual); notes.append(f"Barley price using manual value: {barley_price:.2f}/t")
if urea_price is None:
    urea_price = float(urea_manual); notes.append(f"Urea price using manual value: {urea_price:.2f}/t")

# If currency is €, but fetched in USD, you might add a quick FX knob (optional simple factor)
fx_hint = 1.0
if currency != "$":
    with st.expander("Currency conversion (optional)", expanded=False):
        st.caption("Fetched FRED/TE prices are in USD. Use a simple FX factor if displaying € or £.")
        fx_hint = st.number_input("FX factor (USD → selected currency)", min_value=0.1, max_value=2.0, value=0.95, step=0.01,
                                  help="E.g., USD→EUR ~0.94; USD→GBP ~0.78 (approx).")
display_barley = barley_price * (fx_hint if currency != "$" else 1.0)
display_urea   = urea_price   * (fx_hint if currency != "$" else 1.0)

for n in notes:
    st.caption("• " + n)

# ============== Economics =================
if yld is not None and prot is not None:
    # Revenue
    revenue = yld * display_barley
    # N cost from urea: convert €/t → €/kg N, then multiply by kg N applied
    n_cost = n_rate * urea_cost_per_kgN(display_urea)
    gross = revenue - n_cost

    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Revenue/ha", f"{currency} {revenue:,.0f}", help=f"{yld:.2f} t/ha × {currency} {display_barley:,.0f}/t")
    with g2:
        st.metric("Nitrogen cost/ha", f"{currency} {n_cost:,.0f}",
                  help=f"{n_rate:.0f} kg N/ha × {currency} {urea_cost_per_kgN(display_urea):.2f}/kg N")
    with g3:
        st.metric("Gross margin/ha", f"{currency} {gross:,.0f}")

    st.caption("Simple partial margin: revenue from barley minus urea cost for N. "
               "Does not include other variable/fixed costs.")

# ============== Advisor (highlighted card) =================
st.markdown("## Ask the advisor")
rail, main = st.columns([0.4, 1.6], gap="large")

with rail:
    st.markdown("<div class='right-rail'>", unsafe_allow_html=True)
    st.subheader("Advisor settings")
    response_style = st.slider(
        "Response style",
        0.0, 1.0, 0.20, 0.05,
        help="Controls the AI's writing style (not crop temperature).\n"
             "Lower = conservative & consistent. Higher = exploratory & varied."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with main:
    st.markdown("<div class='advisor-card'>", unsafe_allow_html=True)
    default_hint = (
        f"Given the predictions ({yld:.2f} t/ha, {prot:.2f}%), "
        "what should I adjust?" if yld is not None else "What if I reduce N by 15%?"
    )
    user_q = st.text_area(" ", placeholder=default_hint, height=90, label_visibility="collapsed")
    ask_btn = st.button("🚀 Ask Advisor", type="primary", use_container_width=True)

    if ask_btn:
        if not user_q.strip():
            st.warning("Please type a question for the advisor.")
        else:
            # Small grounded system prompt
            system_prompt = (
                "You are a cautious agronomy assistant for spring barley. "
                "Use the inputs, predictions, and prices to suggest practical, non-prescriptive actions. "
                "Be concise, use bullet points, and mention caveats (soil N, lodging, regulations)."
            )
            context = {
                "inputs": {
                    "end_use": end_use,
                    "n_rate_kg_ha": n_rate,
                    "n_split_first_prop": n_split,
                    "final_n_timing_gs": gs_code(final_n_label),
                },
                "prediction": {"yield_t_ha": yld, "protein_pct": prot},
                "prices": {
                    "barley_per_t": display_barley,
                    "urea_per_t": display_urea,
                    "currency": currency,
                },
            }
            try:
                answer = ask_advisor(system_prompt, f"Context:\n{json.dumps(context)}\n\nQuestion: {user_q}", response_style)
                st.write(answer)
            except Exception as e:
                st.warning(f"Advisor unavailable. ({e})")
                # Fallback: quick rules
                tips = []
                if end_use == "malting" and prot is not None:
                    if prot > 11.0:
                        tips.append("Protein above typical malting window (9.5–11%). Consider reducing late N or total N by ~10–20%.")
                    elif prot < 9.5:
                        tips.append("Protein below malting window; a small increase in N or slightly later split can help.")
                if yld is not None and yld < 7 and n_rate < 150:
                    tips.append("Yield modest; consider raising N (toward 150–180 kg/ha) if soil N is low and lodging risk is acceptable.")
                st.markdown("**Fallback suggestions:**\n- " + "\n- ".join(tips) if tips else "No fallback suggestions available.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============== Footer =================
st.markdown("---")
f1, f2 = st.columns([1, 1])
with f1:
    st.caption("Version: demo")
with f2:
    st.markdown(
        """
        <div class='footer' style='text-align:right;'>
          Developed by <b>Nikolay Georgiev</b>
          <a href="https://www.linkedin.com/in/nikolaygeorgiev/" target="_blank" rel="noopener">
            <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" alt="LinkedIn" />
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
