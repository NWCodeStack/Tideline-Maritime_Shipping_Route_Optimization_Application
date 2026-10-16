"""
app.py - Multi-Agent Maritime Routing and Supply Chain Stress-Tester.

    python app.py          console simulation (needs only numpy, pandas, scikit-learn, networkx)
    streamlit run app.py   interactive dashboard (also needs streamlit; folium + streamlit-folium for the Leaflet map)

Flow: events stress the synthetic network (``apply_events``) -> ``RoutingEngine`` re-routes a container ship
(Apple goods) and an oil tanker (crude) with their own cost functions -> side-by-side comparison.
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
    """The Streamlit dashboard (imports streamlit lazily so `python app.py` works without it)."""
    import numpy as np
    import streamlit as st
    import streamlit.components.v1 as components

    try:                                    # interactive Leaflet map; falls back to the built-in SVG map without it
        from streamlit_folium import st_folium
        from folium_map import build_folium_map
        HAS_FOLIUM = True
    except ImportError:
        HAS_FOLIUM = False

    import ui
    from agents import CARGO_OWNERS, VESSELS, make_agent, profile_matrix
    from data_engine import build_maritime_graph, edges_dataframe, hub_ports
    from events import EVENT_CATALOG, HURRICANE_TRACKS, PRESETS, apply_events, default_settings, describe
    from map_view import build_map_html
    from ml_models import HULL, load_or_train_model
    from routing_engine import RoutingEngine, comparison_frame

    st.set_page_config(page_title="Maritime Stress-Tester", page_icon="⚓", layout="wide")
    st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)


    # --------------------------------------------------------------------------------------------------
    # Cached resources
    # --------------------------------------------------------------------------------------------------
    @st.cache_resource(show_spinner="Training fuel model on 5,000 synthetic voyages...")
    def get_model():
        return load_or_train_model()


    @st.cache_resource
    def get_base_graph():
        return build_maritime_graph()


    def show_df(df: pd.DataFrame, **kw) -> None:
        """st.dataframe across Streamlit versions (the width argument changed in 1.4x)."""
        try:
            st.dataframe(df, width="stretch", **kw)
        except TypeError:
            st.dataframe(df, use_container_width=True, **kw)


    def toggle(label: str, key: str, help: str):
        return (st.toggle if hasattr(st, "toggle") else st.checkbox)(label, key=key, help=help)


    model, G0 = get_model(), get_base_graph()
    ports = hub_ports(G0)

    # --------------------------------------------------------------------------------------------------
    # Sidebar: global events and voyage set-up
    # --------------------------------------------------------------------------------------------------
    DEFAULTS = default_settings()
    for spec in EVENT_CATALOG:
        st.session_state.setdefault(f"ev_{spec.key}", spec.default)
    st.session_state.setdefault("ev_hurricane_track", DEFAULTS["hurricane_track"])


    def apply_preset() -> None:
        preset = PRESETS[st.session_state["preset"]]
        for spec in EVENT_CATALOG:
            st.session_state[f"ev_{spec.key}"] = preset.get(spec.key, spec.default)


    with st.sidebar:
        st.markdown('<div class="section-name">Global events</div>', unsafe_allow_html=True)
        st.selectbox("Scenario preset", list(PRESETS), key="preset", on_change=apply_preset)
        settings: dict = {}
        for spec in EVENT_CATALOG:
            if spec.kind == "toggle":
                settings[spec.key] = toggle(spec.label, f"ev_{spec.key}", spec.help)
            else:
                fmt = "%.0fx" if spec.key == "suez_toll_multiplier" else "%.2f"
                label = spec.label + (" (x)" if spec.key == "suez_toll_multiplier" else " (severity)")
                settings[spec.key] = st.slider(label, spec.lo, spec.hi, step=spec.step, key=f"ev_{spec.key}",
                                               help=spec.help, format=fmt if spec.key != "suez_toll_multiplier" else "%.1f")
            if spec.key == "atlantic_hurricane":
                settings["hurricane_track"] = st.selectbox(
                    "Hurricane track", list(HURRICANE_TRACKS), key="ev_hurricane_track", disabled=settings[spec.key] == 0,
                    format_func=lambda k: HURRICANE_TRACKS[k][0],
                    help="Only ships whose route crosses the storm are affected.")

        st.markdown('<div class="section-name" style="margin-top:22px">Voyages</div>', unsafe_allow_html=True)
        port_ids = list(ports)
        name_of = lambda p: ports[p]
        c_owner = st.selectbox("Container cargo owner", ["high_margin", "low_margin"],
                               format_func=lambda k: CARGO_OWNERS[k].label,
                               help="High-margin: sponsors speed and tolls. Low-margin: avoids tolls, tolerates delay.")
        col1, col2 = st.columns(2)
        c_o = col1.selectbox("Container from", port_ids, index=port_ids.index("SHA"), format_func=name_of)
        c_d = col2.selectbox("Container to", port_ids, index=port_ids.index("RTM"), format_func=name_of)
        col3, col4 = st.columns(2)
        t_o = col3.selectbox("Tanker from", port_ids, index=port_ids.index("DXB"), format_func=name_of)
        t_d = col4.selectbox("Tanker to", port_ids, index=port_ids.index("RTM"), format_func=name_of)

        st.markdown('<div class="section-name" style="margin-top:22px">Economics</div>', unsafe_allow_html=True)
        fuel_price = st.slider("Fuel price ($/tonne)", 300, 1400, 620, 20)
        algorithm = st.radio("Path algorithm", ["dijkstra", "astar"], horizontal=True,
                             format_func=lambda a: {"dijkstra": "Dijkstra", "astar": "A*"}[a])
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        run = st.button("Simulate and Optimize Route", type="primary")

    # --------------------------------------------------------------------------------------------------
    # Compute
    # --------------------------------------------------------------------------------------------------
    params = {"settings": dict(settings), "owner": c_owner, "container": (c_o, c_d), "tanker": (t_o, t_d),
              "fuel_price": fuel_price, "algorithm": algorithm}


    def simulate(p: dict) -> dict:
        """Stress the network, then route both agents (calm baseline + stressed + holding-course comparison)."""
        engine = RoutingEngine(model, p["fuel_price"])
        G1 = apply_events(G0, p["settings"])
        agents = (make_agent("container", p["owner"]), make_agent("tanker", "energy_trader"))
        comps = [engine.compare(G0, G1, agents[0], *p["container"], p["algorithm"]),
                 engine.compare(G0, G1, agents[1], *p["tanker"], p["algorithm"])]
        return {"params": p, "G1": G1, "comps": comps}


    # Routes are (re)computed when the button is pressed, and once on first load so the page is never empty.
    if run or "sim" not in st.session_state:
        st.session_state["sim"] = simulate(params)
    sim = st.session_state["sim"]
    G1, comps, active_settings = sim["G1"], sim["comps"], sim["params"]["settings"]
    stale = sim["params"] != params
    (c_o, c_d), (t_o, t_d) = sim["params"]["container"], sim["params"]["tanker"]
    active = describe(active_settings)

    # --------------------------------------------------------------------------------------------------
    # Header + headline metrics
    # --------------------------------------------------------------------------------------------------
    st.markdown(ui.topbar(), unsafe_allow_html=True)
    st.markdown(ui.intro(active), unsafe_allow_html=True)

    feasible = [c for c in comps if c.stressed.feasible]
    delta = sum(c.stressed.total_usd - c.calm.total_usd for c in comps if c.stressed.feasible and c.calm.feasible)
    rerouted = sum(c.rerouted for c in comps)
    unreachable = sum(not c.stressed.feasible for c in comps)
    cards = []
    for c, tone in zip(comps, ("blue", "gold")):
        s = c.stressed
        cards.append((f"{c.agent.vessel.label} all-in", ui._usd(s.total_usd) if s.feasible else "No route",
                      f"{ports[s.origin]} → {ports[s.destination]} · {s.days:.1f} days" if s.feasible else s.reason[:70],
                      tone if s.feasible else "red"))
    cards.append(("Stress impact vs calm seas", ("+" if delta >= 0 else "−") + ui._usd(abs(delta)),
                  "Extra all-in cost across both voyages" if delta >= 0 else "Cheaper than calm-sea baseline",
                  "red" if delta > 1e5 else ""))
    cards.append(("Agents rerouted", f"{rerouted} of {len(comps)}",
                  f"{unreachable} with no feasible route" if unreachable else "Everyone still has a route",
                  "gold" if rerouted else ""))
    st.markdown(ui.metric_cards(cards), unsafe_allow_html=True)

    # --------------------------------------------------------------------------------------------------
    # Map + agent cards
    # --------------------------------------------------------------------------------------------------
    left, right = st.columns([2.15, 1])
    with left:
        st.markdown(ui.panel_head("Global route intelligence", "Stressed network"), unsafe_allow_html=True)
        if stale:
            st.markdown(ui.alert("Settings changed", "Press Simulate and Optimize Route to apply them to the map."),
                        unsafe_allow_html=True)
        layers = [{"label": c.agent.vessel.label, "vessel_type": c.agent.vessel_type, "result": c.stressed,
                   "hold": c.hold_course} for c in comps]
        if HAS_FOLIUM:
            fmap = build_folium_map(G1, layers, active_settings)
            try:
                st_folium(fmap, height=540, use_container_width=True, returned_objects=[], key="folium_map")
            except TypeError:                                     # older streamlit-folium: no use_container_width
                st_folium(fmap, height=540, width=1100, returned_objects=[], key="folium_map")
        else:
            st.caption("folium / streamlit-folium not installed: showing the built-in map. "
                       "`pip install folium streamlit-folium` for the interactive Leaflet map.")
            svg_layers = [{"label": l["label"], "result": l["result"], "hold": l["hold"],
                           "color": col} for l, col in zip(layers, (ui.CONTAINER_COLOR, ui.TANKER_COLOR))]
            components.html(build_map_html(G1, svg_layers, active_settings, height=540), height=548)

    with right:
        st.markdown('<div class="section-name">Agent decisions</div>', unsafe_allow_html=True)
        for c, col, (o, d) in zip(comps, (ui.CONTAINER_COLOR, ui.TANKER_COLOR), ((c_o, c_d), (t_o, t_d))):
            st.markdown(ui.agent_card(c, col, ports[o], ports[d]), unsafe_allow_html=True)

        st.markdown('<div class="section-name" style="margin-top:14px">Alerts</div>', unsafe_allow_html=True)
        alerts = []
        for c in comps:
            s = c.stressed
            if not s.feasible:
                alerts.append(ui.alert(f"{c.agent.vessel.label}: no feasible route", s.reason, "critical"))
            elif s.forced_through:
                alerts.append(ui.alert(f"{c.agent.vessel.label}: forced through risk", " ".join(s.notes), ""))
            elif s.overland:
                alerts.append(ui.alert(f"{c.agent.vessel.label}: overland transfer", " ".join(s.notes), "info"))
            if s.feasible and c.hold_course is None:
                alerts.append(ui.alert(f"{c.agent.vessel.label}: calm route now closed",
                                       "The pre-event route is no longer passable, so savings vs holding course are n/a.",
                                       "info"))
        if not active:
            alerts.append(ui.alert("No active events", "Pick a preset or switch on a shock to see agents diverge.", "info"))
        st.markdown("".join(alerts) or ui.alert("All clear", "No warnings for the selected voyages.", "info"),
                    unsafe_allow_html=True)

    # --------------------------------------------------------------------------------------------------
    # Breakdown dashboard
    # --------------------------------------------------------------------------------------------------
    st.markdown('<div class="section-name" style="margin-top:8px">Voyage breakdown</div>', unsafe_allow_html=True)
    for c in comps:
        s_, k_ = c.stressed, c.calm
        st.markdown(f'<div class="section-name" style="color:#e8f3fa;margin-top:14px">{c.agent.name}</div>',
                    unsafe_allow_html=True)
        if not s_.feasible:
            st.markdown(ui.alert("No feasible route", s_.reason, "critical"), unsafe_allow_html=True)
            continue
        m1, m2, m3, m4, m5 = st.columns(5)
        has_calm = k_.feasible
        m1.metric("Distance (NM)", f"{s_.distance_nm:,.0f}", f"{s_.distance_nm - k_.distance_nm:+,.0f} vs calm" if has_calm else None,
                  delta_color="inverse")
        m2.metric("Hours elapsed", f"{s_.hours:,.0f}", f"{s_.hours - k_.hours:+,.0f} h vs calm" if has_calm else None,
                  delta_color="inverse")
        m3.metric("Fuel burned (t)", f"{s_.fuel_t:,.0f}", f"{s_.fuel_t - k_.fuel_t:+,.0f} t vs calm" if has_calm else None,
                  delta_color="inverse")
        m4.metric("Total cost (all-in)", ui._usd(s_.total_usd),
                  f"{(s_.total_usd - k_.total_usd) / 1e6:+,.2f}M vs calm" if has_calm else None, delta_color="inverse")
        m5.metric("Cost score (agent objective)", ui._usd(s_.objective), help="What the agent minimised: tolls are "
                  "weighted by its attitude to them, so this differs from the plain-dollar all-in cost.")

    # --------------------------------------------------------------------------------------------------
    # Side-by-side comparison
    # --------------------------------------------------------------------------------------------------
    st.markdown('<div class="section-name" style="margin-top:8px">Side-by-side comparison</div>', unsafe_allow_html=True)
    show_df(comparison_frame(comps))
    st.caption("All-in cost = fuel + tolls & fees + war-risk insurance + time cost (asset cost + lateness penalty). "
               "'vs holding course' reprices the calm-sea route under today's events; fuel saved and tolls avoided "
               "are measured against that same route.")

    tab_profiles, tab_model, tab_lanes = st.tabs(["Agent profiles", "Fuel model", "Network lanes"])
    with tab_profiles:
        st.caption("Behavioural matrices that drive each agent's edge cost.")
        show_df(profile_matrix().T)
    with tab_model:
        m = model.metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Training voyages", f"{m.get('n_samples', 0):,}")
        k2.metric("Random forest R²", f"{m.get('rf_r2', float('nan')):.3f}")
        k3.metric("Linear baseline R²", f"{m.get('linear_r2', float('nan')):.3f}")
        k4.metric("MAE (t/day)", f"{m.get('rf_mae_t_per_day', float('nan')):.1f}")
        curves = {}
        for vt, (draft, cargo) in {"Container": (12.0, 70_000), "Tanker": (16.0, 140_000)}.items():
            lo, hi = HULL[vt]["speed"]
            speeds = np.linspace(lo, hi, 30)
            curves[vt] = pd.Series(model.speed_curve(vt, speeds, draft, cargo), index=np.round(speeds, 1))
        st.caption("Learned fuel burn (tonnes/day) vs speed (knots): the cubic curve is why slow-steaming pays.")
        st.line_chart(pd.DataFrame(curves))
    with tab_lanes:
        lanes = edges_dataframe(G1)
        lanes["events"] = [", ".join(d["active_events"]) for *_, d in G1.edges(data=True)]
        show_df(lanes)

    st.markdown(ui.METHODOLOGY, unsafe_allow_html=True)
    st.markdown(ui.FOOTER, unsafe_allow_html=True)


if _in_streamlit():
    render_ui()
elif __name__ == "__main__":
    console_demo()
