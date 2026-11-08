import os, uuid
import streamlit as st
import pandas as pd
import pfe_copilot as pfe

st.set_page_config(page_title="PFE-Copilot Risk Dashboard", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Indexing corpus and building portfolio...")
def init():
    pfe.make_portfolio()
    pfe.set_retriever(pfe.HybridRetriever(pfe.build_chunks(pfe.load_corpus(), 900, 150), embed=True, rerank=True))
    return True


init()


@st.cache_data(ttl=3600, show_spinner="Running Monte-Carlo exposure simulation...")
def snapshot():
    return pfe.exposure_snapshot()


@st.cache_data(ttl=3600, show_spinner="Running stress scenario (re-simulating PFE)...")
def stress(scn, with_pfe, custom=None):
    return pfe.run_stress(scn, with_pfe, custom_returns=dict(custom) if custom else None)


def color_status(v):
    return {"BREACH": "background-color:#FFC7CE", "WARNING": "background-color:#FFEB9C", "OK": "background-color:#C6EFCE"}.get(v, "")


def styled(df, status_cols, money_cols=()):
    sty = df.style.format({c: "{:,.0f}" for c in money_cols})
    return getattr(sty, "map", sty.applymap)(color_status, subset=status_cols)


if "sid" not in st.session_state:
    st.session_state.sid, st.session_state.msgs = uuid.uuid4().hex[:8], []
agent = pfe.CCRAgent(st.session_state.sid)

snap = snapshot()
synthetic = snap.data_source.str.contains("SYNTHETIC").any()
with st.sidebar:
    st.header("PFE-Copilot")
    st.caption(f"LLM: {'live endpoint' if pfe.llm.live else 'DRY-RUN (rule-based)'}")
    st.caption("Data: Yahoo Finance | SEC EDGAR CSAs | BCBS-IOSCO")
    if st.button("Refresh market data & caches"):
        st.cache_data.clear(); pfe._CACHE.clear(); pfe._PX.clear(); st.rerun()
    if st.button("Generate daily pack (xlsx)"):
        st.session_state["pack"] = pfe.generate_daily_pack_tool()["file"]
    if st.session_state.get("pack") and os.path.exists(st.session_state["pack"]):
        with open(st.session_state["pack"], "rb") as fh:
            st.download_button("Download daily pack", fh.read(), file_name=os.path.basename(st.session_state["pack"]), key="dl_pack")
    strat = st.selectbox("Planner strategy (chat)", ["fewshot", "cot", "decomp"])
    if st.button("Reset chat"):
        agent.reset(); st.session_state.msgs = []; st.rerun()

st.title("Counterparty Credit Risk - exposure, future risk and crisis risk")
if synthetic:
    st.error("SYNTHETIC FALLBACK MARKET DATA in use (Yahoo Finance unreachable) - not for business use.")
st.caption("Illustrative portfolio. Margin actions are proposals pending human approval.")
tab_ov, tab_fut, tab_cr, tab_rev, tab_chat, tab_obs = st.tabs(["Overview", "Future risk (PFE)", "Crisis scenarios", "Reverse stress", "Copilot", "Observability"])

with tab_ov:
    c = st.columns(5)
    c[0].metric("Counterparties", len(snap))
    c[1].metric("Limit BREACH", int((snap.limit_status == "BREACH").sum()))
    c[2].metric("Limit WARNING", int((snap.limit_status == "WARNING").sum()))
    c[3].metric("Margin actions proposed", int((snap.margin_action != "NO_ACTION").sum()))
    c[4].metric("Sum peak PFE / sum limits", f"{snap.peak_pfe_collateralized_usd.sum() / snap.credit_limit_usd.sum():.0%}")
    cols = ["counterparty_id", "counterparty_name", "net_mtm_usd", "peak_pfe_collateralized_usd", "credit_limit_usd", "utilization_pct", "limit_status", "margin_action", "margin_amount_usd", "top_driver"]
    st.dataframe(styled(snap[cols], ["limit_status"], ["net_mtm_usd", "peak_pfe_collateralized_usd", "credit_limit_usd", "margin_amount_usd"]), hide_index=True, use_container_width=True)

    with st.expander("Regulatory cross-check: simplified SA-CCR vs Monte-Carlo"):
        st.dataframe(pfe.saccr_cross_check().round(2), hide_index=True, use_container_width=True)
        st.caption("Simplified Basel SA-CCR (BCBS d279) as a sanity check: MC_EAD = 1.4 x effective EPE vs SA-CCR EAD. Large ratios mean the two methods disagree and deserve investigation. Parameters should be verified against d279; not a regulatory calculation.")
    with st.expander("Model validation (closed-form benchmark and CSA maths checks)"):
        if st.button("Run engine validation"):
            vdf = pfe.validate_engine()
            st.dataframe(vdf, hide_index=True, use_container_width=True)
            (st.success if vdf.passed.all() else st.error)(f"{int(vdf.passed.sum())}/{len(vdf)} checks passed")

with tab_fut:
    st.subheader("Future exposure profile (Monte-Carlo)")
    cp = st.selectbox("Counterparty", snap.counterparty_id.tolist(), key="fut_cp")
    a, b = st.columns(2)
    mpor = a.slider("Margin period of risk (days)", 5, 30, int(pfe.cp_table().query("counterparty_id == @cp").mpor_days.iloc[0]))
    vm = b.slider("Volatility multiplier", 0.8, 2.5, 1.0, 0.1)
    vmodel = st.radio("Volatility model", ["ewma", "hist"], horizontal=True, help="ewma: starts at today's EWMA vol and mean-reverts to the 2y level (forward-looking). hist: flat 2y volatility.")
    ex = pfe.simulate_exposure(cp, mpor_days=mpor, vol_mult=vm, vol_model=vmodel)
    lim = float(pfe.cp_table().query("counterparty_id == @cp").credit_limit_usd.iloc[0])
    m = st.columns(4)
    m[0].metric("Peak 95% PFE (collateralised)", f"{ex['peak_pfe_collateralized_usd']:,.0f}", f"{ex['peak_pfe_collateralized_usd'] / lim:.0%} of limit")
    m[1].metric("Peak 95% PFE (uncollateralised)", f"{ex['peak_pfe_uncollateralized_usd']:,.0f}")
    m[2].metric("Expected positive exposure", f"{ex['epe_collateralized_usd']:,.0f}")
    m[3].metric("Peak at day", ex["peak_day"])
    st.pyplot(pfe.fig_future_risk(cp, mpor_days=mpor, vol_mult=vm, vol_model=vmodel))
    with st.expander("Volatility forecast behind this chart"):
        und = sorted(pfe.trades_table().query("counterparty_id == @cp").underlying.unique())
        st.dataframe(pfe.vol_forecast_table(und), hide_index=True)
        st.pyplot(pfe.fig_vol_forecast(und))
    st.caption("Wider MPOR or higher vol = more gap risk between margin calls. Try MPOR 20 and vol x1.5 to see the limit pressure.")

with tab_cr:
    st.subheader("Crisis scenarios: exposure and collateral shocked together")
    for k_, v_ in (("scenario_choice", "equity_crash_20"), ("sl_equity", -25), ("sl_gold", 5), ("sl_rates", 3), ("sl_credit", -8), ("sl_usd", 5)):
        st.session_state.setdefault(k_, v_)
    with st.expander("Describe a scenario in words (AI-assisted)"):
        txt = st.text_area("Scenario", "Equities fall 25%, the dollar rallies 8% and credit spreads widen 150bp")
        if st.button("Propose shocks"):
            prop = pfe.propose_scenario(txt)
            st.session_state["ai_prop"] = prop
            for k_, key_ in (("equity", "sl_equity"), ("gold", "sl_gold"), ("rates", "sl_rates"), ("credit", "sl_credit"), ("usd", "sl_usd")):
                st.session_state[key_] = int(round(prop["shocks"][k_] * 100))
            st.session_state["scenario_choice"] = "custom"
        prop = st.session_state.get("ai_prop")
        if prop:
            st.info(f"Source: {prop['source']}. Rationale: {prop['rationale']}")
            for c_ in prop["clamped"]:
                st.warning(c_)
            st.caption("The AI only proposes. Review and edit the sliders below before reading the results.")
    choice = st.selectbox("Scenario", pfe.SCENARIO_NAMES + ["custom"], key="scenario_choice")
    custom = None
    if choice == "custom":
        sc = st.columns(5)
        sh = dict(equity=sc[0].slider("Equities %", -70, 40, key="sl_equity") / 100, gold=sc[1].slider("Gold %", -40, 60, key="sl_gold") / 100,
                  rates=sc[2].slider("Long bonds %", -40, 40, key="sl_rates") / 100, credit=sc[3].slider("IG credit %", -40, 15, key="sl_credit") / 100,
                  usd=sc[4].slider("USD strength %", -20, 30, key="sl_usd") / 100)
        custom = tuple(sorted(pfe.bucket_returns(sh, pfe.portfolio_tickers()).items()))
    elif choice in pfe.HIST_SCENARIOS:
        st.caption(f"Historical replay: {pfe.HIST_SCENARIOS[choice][2]} ({pfe.HIST_SCENARIOS[choice][0]} to {pfe.HIST_SCENARIOS[choice][1]}) using realised Yahoo Finance returns")
    else:
        st.caption(pfe.PRESET_SCENARIOS[choice][1])
    with_pfe = st.checkbox(f"Also re-simulate PFE under stress (vol x{pfe.STRESS_VOL_MULT}, correlation blend {pfe.STRESS_CORR_BLEND})", True)
    df, ret, desc, notes = stress(choice, with_pfe, custom)
    for n in notes:
        st.warning(n)
    liq = df[df.stressed_margin_signed_usd > 0].stressed_margin_signed_usd.sum()
    m = st.columns(4)
    m[0].metric("Liquidity need (calls we make)", f"{liq:,.0f}")
    m[1].metric("BREACH today", int((df.base_status == "BREACH").sum()))
    m[2].metric("BREACH after stress", int((df.stressed_status == "BREACH").sum()), int((df.stressed_status == "BREACH").sum() - (df.base_status == "BREACH").sum()), delta_color="inverse")
    m[3].metric("Worst gap exposure", f"{df.stressed_exposure_usd.max():,.0f}")
    st.pyplot(pfe.fig_stress_results(df, desc))
    st.pyplot(pfe.fig_liquidity(df))
    show = ["counterparty_id", "base_mtm_usd", "stressed_mtm_usd", "collateral_change_usd", "stressed_exposure_usd", "stressed_util_pct", "base_status", "stressed_status", "stressed_margin_signed_usd", "worst_driver"]
    st.dataframe(styled(df[show], ["base_status", "stressed_status"], ["base_mtm_usd", "stressed_mtm_usd", "collateral_change_usd", "stressed_exposure_usd", "stressed_margin_signed_usd"]), hide_index=True, use_container_width=True)
    with st.expander("Shocks applied (return per ticker)"):
        st.dataframe(pd.DataFrame({"ticker": list(ret), "return_pct": [round(v * 100, 2) for v in ret.values()]}), hide_index=True)
    if choice != "custom" and st.button("Export this scenario to Excel"):
        st.session_state["stress_file"] = pfe.run_stress_test_tool(choice)["file"]
    if st.session_state.get("stress_file") and os.path.exists(st.session_state["stress_file"]):
        with open(st.session_state["stress_file"], "rb") as fh:
            st.download_button("Download stress workbook", fh.read(), file_name=os.path.basename(st.session_state["stress_file"]), key="dl_stress")

with tab_rev:
    st.subheader("Reverse stress: how far must markets move to breach the limit?")
    cp2 = st.selectbox("Counterparty", snap.counterparty_id.tolist(), key="rev_cp")
    fig, info = pfe.fig_reverse_stress(cp2)
    d_, u_ = info["down_breach_shock_pct"], info["up_breach_shock_pct"]
    st.write(f"Exposure after collateral exceeds the {info['limit_usd']:,.0f} limit if all underlyings move " +
             (f"**down {abs(d_):.0f}%**" if d_ is not None else "**no** downward move up to 50%") + " or " +
             (f"**up {u_:.0f}%**" if u_ is not None else "**no** upward move up to 50%") + " (collateral value moves with its own beta).")
    st.pyplot(fig)

with tab_chat:
    for mm in st.session_state.msgs:
        with st.chat_message(mm["role"]):
            st.markdown(mm["content"])
    if q := st.chat_input("e.g. Prepare the margin call notice for CP-002 / What happens in a COVID-style crash? / What does the CSA say about MTA?"):
        st.session_state.msgs.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Planning -> executing -> verifying..."):
                r = agent.ask(q, strategy=strat)
            st.markdown(r["answer"])
            for i, x in enumerate(r["results"]):
                o = x.get("output") if x.get("ok") else None
                for k in ("file", "upload_csv"):
                    if isinstance(o, dict) and o.get(k) and os.path.exists(o[k]):
                        with open(o[k], "rb") as fh:
                            st.download_button("Download " + os.path.basename(o[k]), fh.read(), file_name=os.path.basename(o[k]), key=f"dl_{r['trace_id']}_{i}_{k}")
            with st.expander(f"Trace {r['trace_id']} - {r['n_tool_calls']} tool calls, {r['llm_calls']} LLM calls, {r['tokens']} tokens, {r['latency_s']}s"):
                st.json({"plan": r["plan"], "flags": r["flags"], "tools": [{k: v for k, v in x.items() if k != "output"} for x in r["results"]]})
        st.session_state.msgs.append({"role": "assistant", "content": r["answer"]})

with tab_obs:
    st.subheader("Trace dashboard (real query executions)")
    tr = pfe.tracer.load()
    if tr.empty:
        st.info("No traces yet - ask the copilot something in the Copilot tab (or run the evaluation in the notebook).")
    else:
        m = st.columns(6)
        m[0].metric("Queries", len(tr)); m[1].metric("Avg latency (s)", f"{tr.latency_s.mean():.2f}"); m[2].metric("p95 latency (s)", f"{tr.latency_s.quantile(.95):.2f}")
        m[3].metric("Avg tokens", f"{tr.tokens.mean():.0f}"); m[4].metric("Avg tool calls", f"{tr.n_tool_calls.mean():.2f}"); m[5].metric("Refusal rate", f"{tr.refused.mean():.0%}")
        c1, c2 = st.columns(2)
        tools = pd.Series([t["tool"] for ts in tr["tools"] for t in ts])
        flags = pd.Series([f for fl in tr["flags"] for f in fl])
        c1.caption("Tool usage"); c1.bar_chart(tools.value_counts() if len(tools) else pd.Series(dtype=int))
        c2.caption("Flags (refusals, fallbacks, injection, PII)"); c2.bar_chart(flags.value_counts() if len(flags) else pd.Series(dtype=int))
        st.caption("Latency per query (s)"); st.line_chart(tr.latency_s.reset_index(drop=True))
        view = tr[["trace_id", "query", "n_tool_calls", "llm_calls", "tokens", "latency_s", "refused", "critic_revisions"]].copy()
        view["flags"] = tr["flags"].apply(lambda x: ", ".join(x))
        st.dataframe(view.tail(100), hide_index=True, use_container_width=True)
        sel = st.selectbox("Inspect a trace", tr.trace_id.tolist()[::-1])
        st.json(tr[tr.trace_id == sel].iloc[0].to_dict())
