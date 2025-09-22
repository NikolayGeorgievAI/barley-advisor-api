# ---------- Color-coded KPI cards ----------
def fmt_money(x: float, curr: str = "USD") -> str:
    return f"{curr} {x:,.0f}"

def kpi_card(label: str, value: str, color: str = "#111827"):
    st.markdown(
        f"""
        <div style="
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

# Choose colors
rev_color = "#16a34a"   # green
cost_color = "#dc2626"  # red
gm_color = "#16a34a" if gross_margin_ha >= 0 else "#dc2626"  # green if positive, red if negative
gm_label = "Gross margin/ha"

# Layout
c1, c2, c3 = st.columns(3)
with c1:
    kpi_card("Revenue/ha", fmt_money(revenue_ha, curr), rev_color)
with c2:
    kpi_card("N cost/ha", fmt_money(n_cost_ha, curr), cost_color)
with c3:
    kpi_card(gm_label, fmt_money(gross_margin_ha, curr), gm_color)
# -------------------------------------------
