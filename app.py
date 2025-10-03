# ================== V2 Enhancements ==================
st.divider()
st.markdown("## 🌾 Barley Advisor v2.0 — New Features")

tab_current, tab_compare, tab_econ, tab_model = st.tabs(
    ["Advisor (current)", "Scenario Compare (NEW)", "Economics (NEW)", "Model Card"]
)

# --- Scenario Compare (A/B) ---
with tab_compare:
    st.subheader("Scenario Compare (A vs B)")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Scenario A")
        nA = st.slider("N rate A (kg/ha)", 60, 240, 160, 10)
        pA = st.slider("P rate A (kg/ha)", 0, 120, 60, 10)
        yieldA, protA = predict_yield_and_protein_as_is(nA, pA)
    with colB:
        st.markdown("#### Scenario B")
        nB = st.slider("N rate B (kg/ha)", 60, 240, 180, 10)
        pB = st.slider("P rate B (kg/ha)", 0, 120, 60, 10)
        yieldB, protB = predict_yield_and_protein_as_is(nB, pB)

    d1, d2 = st.columns(2)
    d1.metric("Yield Δ (t/ha)", f"{yieldB:.2f}", f"{yieldB - yieldA:+.2f}")
    d2.metric("Protein Δ (%)", f"{protB:.2f}", f"{protB - protA:+.2f}")

# --- Economics Tab ---
with tab_econ:
    st.subheader("Economics & Sustainability")
    price = st.number_input("Grain price (€/t)", 100, 400, 200)
    n_price = st.number_input("N price (€/kg)", 0.5, 3.0, 1.2, step=0.1)
    co2_factor = st.number_input("CO₂e factor (kg/kg N)", 3.0, 10.0, 6.5, step=0.1)

    # reuse your gross_margin_simple but with custom economics here
    m_per_ha, m_total, df_econ = gross_margin_simple(inputs, yield_t_ha)
    st.dataframe(df_econ, hide_index=True, use_container_width=True)

    emissions = inputs.n_rate * co2_factor
    st.metric("Estimated emissions (kg CO₂e/ha)", f"{emissions:.0f}")

    # Export report
    if st.button("⬇️ Download simple HTML report"):
        report = f"<h2>Barley Advisor v2 Report</h2><p>Yield: {yield_t_ha:.2f} t/ha<br>Protein DM: {protein_dm:.2f}%<br>Margin: {m_per_ha:,.0f}{symbol(inputs.currency)}/ha</p>"
        st.download_button("Save report", report.encode("utf-8"), file_name="barley_report.html", mime="text/html")

# --- Model Card ---
with tab_model:
    st.subheader("Model Card (Transparency)")
    st.markdown("""
    **Data:** TEAGASC barley agronomy trials.  
    **Targets:** Yield (t/ha), Protein (% as-is → DM).  
    **Version:** v2.0 (Oct 2025).  
    **Intended use:** Scenario comparison & education.  
    **Known limitations:** Simplified, not calibrated for all regions.  
    **Disclaimer:** Prototype — not a substitute for local agronomy advice.
    """)
