"""
stress_ui.py - the original multi-agent stress test (a top-level tab; uses the shared scenario settings).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import ui
from agents import CARGO_OWNERS, make_agent, profile_matrix
from data_engine import edges_dataframe
from events import apply_events
from live import HAS_FOLIUM, st_folium
from map_view import build_map_html
from ml_models import HULL
from page import show_df
from routing_engine import RoutingEngine, comparison_frame

if HAS_FOLIUM:
    from folium_map import build_folium_map


def stress_tab(pg):
    ss, ops, model, G0, ports = pg.ss, pg.ops, pg.model, pg.G0, pg.ports
    port_ids = list(ports)
    st.caption("Two agents with different economics face the same scenarios (the incidents in Voyage & Simulation plus the sidebar's advanced scenarios): "
               "a container ship and an oil tanker, each re-routed by its own cost function.")
    c = st.columns([1.4, 1, 1, 1, 1, 1.1, 1])
    owner = c[0].selectbox("Container cargo owner", ["high_margin", "low_margin"], key="st_owner",
                           format_func=lambda k: CARGO_OWNERS[k].label)
    c_o = c[1].selectbox("Container from", port_ids, index=port_ids.index("SHA"), format_func=ports.get, key="st_co")
    c_d = c[2].selectbox("Container to", port_ids, index=port_ids.index("RTM"), format_func=ports.get, key="st_cd")
    t_o = c[3].selectbox("Tanker from", port_ids, index=port_ids.index("DXB"), format_func=ports.get, key="st_to")
    t_d = c[4].selectbox("Tanker to", port_ids, index=port_ids.index("RTM"), format_func=ports.get, key="st_td")
    fuel_price = c[5].slider("Fuel price ($/t)", 300, 1400, 620, 20, key="st_fuel")
    algorithm = c[6].radio("Algorithm", ["dijkstra", "astar"], key="st_algo",
                           format_func=lambda a: {"dijkstra": "Dijkstra", "astar": "A*"}[a])
    run = st.button("Simulate and Optimize Route", type="primary", key="st_run")

    params = {"settings": {k: v for k, v in ops.settings.items()}, "owner": owner, "container": (c_o, c_d),
              "tanker": (t_o, t_d), "fuel_price": fuel_price, "algorithm": algorithm}

    def simulate(p: dict) -> dict:
        engine = RoutingEngine(model, p["fuel_price"])
        G1 = apply_events(G0, p["settings"])
        agents = (make_agent("container", p["owner"]), make_agent("tanker", "energy_trader"))
        comps = [engine.compare(G0, G1, agents[0], *p["container"], p["algorithm"]),
                 engine.compare(G0, G1, agents[1], *p["tanker"], p["algorithm"])]
        return {"params": p, "G1": G1, "comps": comps}

    if run or "sim" not in ss:
        ss["sim"] = simulate(params)
    sim = ss["sim"]
    G1, comps, active_settings = sim["G1"], sim["comps"], sim["params"]["settings"]
    stale = sim["params"] != params
    (c_o, c_d), (t_o, t_d) = sim["params"]["container"], sim["params"]["tanker"]

    delta = sum(x.stressed.total_usd - x.calm.total_usd for x in comps if x.stressed.feasible and x.calm.feasible)
    rerouted = sum(x.rerouted for x in comps)
    unreachable = sum(not x.stressed.feasible for x in comps)
    cards = []
    for x, tone in zip(comps, ("blue", "gold"), strict=False):
        s = x.stressed
        cards.append((f"{x.agent.vessel.label} all-in", ui._usd(s.total_usd) if s.feasible else "No route",
                      f"{ports[s.origin]} → {ports[s.destination]} · {s.days:.1f} days" if s.feasible else s.reason[:70],
                      tone if s.feasible else "red"))
    cards.append(("Stress impact vs calm seas", ("+" if delta >= 0 else "−") + ui._usd(abs(delta)),
                  "Extra all-in cost across both voyages" if delta >= 0 else "Cheaper than calm-sea baseline",
                  "red" if delta > 1e5 else ""))
    cards.append(("Agents rerouted", f"{rerouted} of {len(comps)}",
                  f"{unreachable} with no feasible route" if unreachable else "Everyone still has a route",
                  "gold" if rerouted else ""))
    st.markdown(ui.metric_cards(cards), unsafe_allow_html=True)

    left, right = st.columns([2.15, 1])
    with left:
        st.markdown(ui.panel_head("Global route intelligence", "Stressed network"), unsafe_allow_html=True)
        if stale:
            st.markdown(ui.alert("Settings changed", "Press Simulate and Optimize Route to apply them to the map."),
                        unsafe_allow_html=True)
        layers = [{"label": x.agent.vessel.label, "vessel_type": x.agent.vessel_type, "result": x.stressed,
                   "hold": x.hold_course} for x in comps]
        if HAS_FOLIUM:
            st_folium(build_folium_map(G1, layers, active_settings), height=540, use_container_width=True,
                      returned_objects=[], key="folium_map")
        else:
            svg_layers = [{"label": l["label"], "result": l["result"], "hold": l["hold"], "color": col}
                          for l, col in zip(layers, (ui.CONTAINER_COLOR, ui.TANKER_COLOR), strict=False)]
            components.html(build_map_html(G1, svg_layers, active_settings, height=540), height=548)
    with right:
        st.markdown('<div class="section-name">Agent decisions</div>', unsafe_allow_html=True)
        for x, col, (o, d) in zip(comps, (ui.CONTAINER_COLOR, ui.TANKER_COLOR), ((c_o, c_d), (t_o, t_d)), strict=False):
            st.markdown(ui.agent_card(x, col, ports[o], ports[d]), unsafe_allow_html=True)
        notes = []
        for x in comps:
            s = x.stressed
            if not s.feasible:
                notes.append(ui.alert(f"{x.agent.vessel.label}: no feasible route", s.reason, "critical"))
            elif s.forced_through:
                notes.append(ui.alert(f"{x.agent.vessel.label}: forced through risk", " ".join(s.notes), ""))
            elif s.overland:
                notes.append(ui.alert(f"{x.agent.vessel.label}: overland transfer", " ".join(s.notes), "info"))
        st.markdown("".join(notes), unsafe_allow_html=True)

    st.markdown('<div class="section-name" style="margin-top:8px">Voyage breakdown</div>', unsafe_allow_html=True)
    for x in comps:
        s_, k_ = x.stressed, x.calm
        st.markdown(f'<div class="section-name" style="color:#e8f3fa;margin-top:14px">{x.agent.name}</div>',
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

    st.markdown('<div class="section-name" style="margin-top:8px">Side-by-side comparison</div>', unsafe_allow_html=True)
    show_df(comparison_frame(comps))
    st.caption("All-in cost = fuel + tolls & fees + war-risk insurance + time cost (asset cost + lateness penalty). "
               "'vs holding course' reprices the calm-sea route under today's events.")
    tab_profiles, tab_model, tab_lanes = st.tabs(["Agent profiles", "Fuel model", "Network lanes"])
    with tab_profiles:
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
