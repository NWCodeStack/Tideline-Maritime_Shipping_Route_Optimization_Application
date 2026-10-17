"""
app.py - Tideline: live AI shipping-operations simulator (Streamlit) + the original multi-agent stress test.

    python app.py          console stress-test demo (needs only numpy, pandas, scikit-learn, networkx)
    streamlit run app.py   interactive dashboard

Live operations tab
    Plan a voyage, start the simulated clock and watch the ship move along the modelled lanes. Toggle disruptions
    in the incident panel (or ask the copilot) and the remaining voyage is re-routed from the ship's current
    position. Weather (Open-Meteo) and maritime headlines (GDELT / Google News RSS) feed the Alerts & Intelligence panel.
Multi-agent stress test tab
    The original comparison: a container ship and an oil tanker re-routing under the same shocks.

All mutable state lives in st.session_state (one ``ops.Ops`` controller per browser session); the cached
resources (fuel model, base graph) are read-only.
"""
from __future__ import annotations

import pandas as pd

from agents import DEFAULT_CONTAINER, DEFAULT_TANKER
from data_engine import build_maritime_graph
from events import apply_events, describe
from ml_models import load_or_train_model
from routing_engine import RoutingEngine, comparison_frame

SCENARIOS = {
    "Calm seas": {},
    "Strait of Hormuz blockaded": {"hormuz_blockade": True},
    "Suez Canal toll x5": {"suez_toll_multiplier": 5.0},
    "Red Sea escalation (80%)": {"red_sea_escalation": 0.8},
    "Perfect storm (Hormuz + Red Sea + Suez x5 + hurricane)": {
        "hormuz_blockade": True, "red_sea_escalation": 0.8, "suez_toll_multiplier": 5.0,
        "atlantic_hurricane": 0.9, "taiwan_tension": 0.4},
}


def console_demo() -> None:
    """Print Container (Shanghai -> Rotterdam, Apple goods) vs Tanker (Jebel Ali -> Rotterdam, crude) per scenario."""
    model = load_or_train_model()
    engine = RoutingEngine(model)
    G0 = build_maritime_graph()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(f"Fuel model: random forest, R2={model.metrics.get('rf_r2', float('nan')):.3f} on "
          f"{model.metrics.get('n_samples', 0)} synthetic voyages\n")
    for name, events in SCENARIOS.items():
        G1 = apply_events(G0, events)
        comps = [engine.compare(G0, G1, DEFAULT_CONTAINER, "SHA", "RTM"),
                 engine.compare(G0, G1, DEFAULT_TANKER, "DXB", "RTM")]
        print("=" * 100)
        print(f"SCENARIO: {name}   [{', '.join(describe(events)) or 'no events'}]")
        print("=" * 100)
        print(comparison_frame(comps).to_string())
        for c in comps:
            for note in c.stressed.notes:
                print(f"  note ({c.agent.vessel.label}): {note}")
        print()


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def render_ui() -> None:
    """The Streamlit dashboard (imports streamlit lazily so `python app.py` works without it).

    Page hierarchy: header (status) -> Route KPIs -> [Live route map + copilot | five tabs]. The map and copilot sit
    outside the tabs so switching tabs never resets them; the stress test is the second top-level tab."""
    import streamlit as st

    import ui
    from data_engine import hub_ports
    from live import kpi_fragment, map_fragment, seek_fragment
    from ops import Ops
    from page import Page, get_base_graph, get_llm, get_model
    from stress_ui import stress_tab
    from tabs import (RIGHT_TABS, render_alerts, render_change_banner, render_compare, render_copilot,
                      render_header, render_intel_panel, render_operate, render_optimize, render_sources)

    st.set_page_config(page_title="Tideline · Maritime Operations", page_icon="⚓", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)
    ss = st.session_state

    model, G0 = get_model(), get_base_graph()
    llm, llm_status = get_llm()
    if "ops" not in ss:
        ss["ops"] = Ops(model)
    ops: Ops = ss["ops"]
    ops.model = model
    pg = Page(ops, model, G0, hub_ports(G0), llm, llm_status if llm else "Rules-based command mode (no LLM key)")
    pg.sync_widgets()
    pg.show_toasts()

    render_header(pg)
    tab_live, tab_stress = st.tabs(["Live operations", "Multi-agent stress test"])
    with tab_live:
        change = ops.demo if ops.demo else ops.last_change
        render_change_banner(change, demo=bool(ops.demo))
        kpi_fragment(ops)()                           # Route KPIs and telemetry (full width, tick during playback)
        map_col, side_col = st.columns([1.25, 1], gap="medium")
        with map_col:
            map_fragment(pg)()                        # animated in the browser: re-sent only when the route/clock changes
            seek_fragment(pg)()
            render_copilot(pg)
        with side_col:
            t_op, t_opt, t_cmp, t_alert, t_src = st.tabs(RIGHT_TABS, key="right_tabs")
            with t_op:
                render_operate(pg)
            with t_opt:
                render_optimize(pg)
            with t_cmp:
                render_compare(pg, change)
            with t_alert:
                intel_slot = render_alerts(pg)
            with t_src:
                render_sources(pg)
    with tab_stress:
        stress_tab(pg)
    st.markdown(ui.FOOTER, unsafe_allow_html=True)
    render_intel_panel(pg, intel_slot)                # rendered LAST: slow or failing network calls never delay the page


if _in_streamlit():
    render_ui()
elif __name__ == "__main__":
    console_demo()
