# app.py — Barley Advisor with live(ish) price feeds + gross margin
import os
import math
import json
from dataclasses import dataclass

import joblib
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
    kind: str               # "csv" | "excel"
    sheet: str | None = None
    price_col: str | None = None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_csv_or_excel(spec: PriceFetchSpec) -> float | None:
    try:
        if spec.kind == "csv":
            df = pd.read_csv(spec.url)
        else:
            df = pd.read_excel(spec.url, sheet_name=spec.sheet) if spec.sheet else pd.read_excel(spec.url)
        if spec.price_col and spec.price_col in df.columns:
            s = pd.to_numeric(df[spec.price_col], errors="coerce").dropna()
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return None
            s = pd.to_numeric(df[num_cols[-1]], errors="coerce").dropna()
        return float(s.iloc[-1]) if not s.empty else None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_urea_from_tradingeconomics() -> float | None:
    try:
        url = "https://api.tradingeconomics.com/commodity/urea?c=guest:guest&format=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
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
    try:
        url = "https://fred.stlouisfed.org/series/PBARLUSDM/downloaddata/PBARLUSDM.csv"
        df = pd.read_csv(url)
        s = pd.to_numeric(df["PBARLUSDM"], errors="coerce").dropna()
        return float(s.iloc[-1]) if not s.empty else None
    except Exception:
        return None

def urea_cost_per_kgN(urea_price_per_tonne: float) -> float:
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
    n_split = st.number_input("N split — first application proportion", 0.0, 1.0, 0.50, 0.05)
with c2:
    end_use = st.selectbox("End use", ["feed", "malting"], index=0)
    final_n_label = st.selectbox("Final N timing (GS)", 
                                 ["Tillering (GS25)", "Stem elongation (GS31)", "Flag leaf (GS37)", "Emergence", "Sowing"], index=0)

def gs_code(lbl: str) -> str:
    s = lbl.lower()
    if "gs25" in s: return "GS25"
    if "gs31" in s: return "GS31"
    if "gs37" in s: return "GS37"
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
    y = MODEL.predict(X)
    return float(y[0][0]), float(y[0][1])

pred_btn = st.button("Predict", type="primary")
yld, prot = None, None
if pred_btn:
    try:
        yld, prot = predict(end_use, n_rate, n_split, final_n_label)
        a, b = st.columns(2)
        with a: st.markdown(f"<div class='metric-box'><b>Predicted yield:</b> {yld:.2f} t/ha</div>", unsafe_allow_html=True)
        with b: st.markdown(f"<div class='metric-box'><b>Predicted protein:</b> {prot:.2f}%</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============== Sidebar: data source overrides (fixed keys) =================
with st.sidebar:
    st.subheader("Live price sources (optional overrides)")

    with st.expander("Override Urea source (CSV/XLSX)", expanded=False):
        urea_url   = st.text_input("Urea price URL", key="urea_url")
        urea_kind  = st.radio("File type", ["csv", "excel"], horizontal=True, index=0, key="urea_kind")
        urea_sheet = st.text_input("Excel sheet (optional)", key="urea_sheet")
        urea_col   = st.text_input("Price column (optional)", key="urea_col")

    with st.expander("Override Barley source (CSV/XLSX)", expanded=False):
        barley_url   = st.text_input("Barley price URL", key="barley_url")
        barley_kind  = st.radio("File type ", ["csv", "excel"], horizontal=True, index=0, key="barley_kind")
        barley_sheet = st.text_input("Excel sheet (optional)", key="barley_sheet")
        barley_col   = st.text_input("Price column (optional)", key="barley_col")

    st.caption("If overrides are empty or fail, the app uses TradingEconomics (Urea) and FRED (Barley).")

# ============== Resolve prices =================
barley_price = None
urea_price = None
notes = []

if barley_url:
    v = fetch_csv_or_excel(PriceFetchSpec(barley_url, barley_kind, barley_sheet or None, barley_col or None))
    if v: barley_price = float(v); notes.append(f"Barley override: {barley_price:.2f}/t")
if urea_url:
    v = fetch_csv_or_excel(PriceFetchSpec(urea_url, urea_kind, urea_sheet or None, urea_col or None))
    if v: urea_price = float(v); notes.append(f"Urea override: {urea_price:.2f}/t")

if barley_price is None:
    v = fetch_barley_from_fred()
    if v: barley_price = float(v); notes.append(f"Barley (FRED): {barley_price:.2f} USD/t")
if urea_price is None:
    v = fetch_urea_from_tradingeconomics()
    if v: urea_price = float(v); notes.append(f"Urea (TradingEconomics): {urea_price:.2f} USD/t")

if barley_price is None: barley_price = 190.0; notes.append("Barley manual fallback: 190/t")
if urea_price is None: urea_price = 600.0; notes.append("Urea manual fallback: 600/t")

st.markdown("### Gross margin (simple)")
for n in notes: st.caption("• " + n)

if yld is not None:
    revenue = yld * barley_price
    n_cost = n_rate * urea_cost_per_kgN(urea_price)
    gross = revenue - n_cost
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Revenue/ha", f"{revenue:,.0f}")
    with c2: st.metric("N cost/ha", f"{n_cost:,.0f}")
    with c3: st.metric("Gross margin/ha", f"{gross:,.0f}")

# ============== Advisor =================
st.markdown("## Ask the advisor")
rail, main = st.columns([0.4, 1.6])

with rail:
    st.markdown("<div class='right-rail'>", unsafe_allow_html=True)
    response_style = st.slider("Response style", 0.0, 1.0, 0.20, 0.05)
    st.caption("Not crop temperature — controls AI writing style.")
    st.markdown("</div>", unsafe_allow_html=True)

with main:
    st.markdown("<div class='advisor-card'>", unsafe_allow_html=True)
    q = st.text_area(" ", placeholder="Ask a question (e.g. 'What if I reduce N by 15%?')", height=90, label_visibility="collapsed")
    if st.button("🚀 Ask Advisor", use_container_width=True):
        try:
            ctx = {"inputs": {"end_use": end_use, "n_rate": n_rate, "split": n_split}, "pred": {"yield": yld, "protein": prot}}
            answer = ask_advisor("You are a cautious agronomy assistant.", f"Context: {json.dumps(ctx)}\n\nQ: {q}", response_style)
            st.write(answer)
        except Exception as e:
            st.warning(f"Advisor unavailable ({e})")
    st.markdown("</div>", unsafe_allow_html=True)

# ============== Footer =================
st.markdown("---")
st.markdown(
    """
    <div class='footer' style='text-align:right;'>
      Developed by <b>Nikolay Georgiev</b>
      <a href="https://www.linkedin.com/in/nikolaygeorgiev/" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" alt="LinkedIn"/>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)
