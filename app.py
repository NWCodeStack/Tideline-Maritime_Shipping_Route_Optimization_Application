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
TICK_S = 1.0                     # playback refresh interval (real seconds)


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

    Page hierarchy: header (status) -> [Voyage & Simulation | Alerts & Intelligence] -> Route KPIs ->
    [Live route map | Operations copilot] -> Rerouting decision & event timeline. Advanced configuration
    lives in the (collapsed) sidebar."""
    import html
    import json
    from datetime import datetime, timezone

    import numpy as np
    import streamlit as st
    import streamlit.components.v1 as components

    try:                                    # interactive Leaflet map; the offline SVG map is used without it
        from streamlit_folium import st_folium
        from folium_map import build_folium_map, build_ops_base_map, build_ops_layer
        HAS_FOLIUM = True
    except ImportError:
        HAS_FOLIUM = False
        st_folium = build_folium_map = build_ops_base_map = build_ops_layer = None

    import copilot
    import intel
    import ui
    import voyage as V
    from agents import CARGO_OWNERS, VESSELS, make_agent, profile_matrix
    from data_engine import CHOKEPOINTS, edges_dataframe, hub_ports
    from events import EVENT_BY_KEY, EVENT_CATALOG, HURRICANE_TRACKS, PRESETS
    from llm import get_provider
    from map_view import build_map_html, build_ops_map_html
    from ml_models import HULL
    from ops import SPEEDS, Ops

    st.set_page_config(page_title="Tideline · Maritime Operations", page_icon="⚓", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)
    ss = st.session_state

    # ----------------------------------------------------------------------------------------------
    # Cached, read-only resources (shared across sessions; never mutated)
    # ----------------------------------------------------------------------------------------------
    @st.cache_resource(show_spinner="Training fuel model on 5,000 synthetic voyages...")
    def get_model():
        return load_or_train_model()

    @st.cache_resource
    def get_base_graph():
        return build_maritime_graph()

    @st.cache_resource
    def get_llm():
        return get_provider(st.secrets)

    @st.cache_data(ttl=1800, show_spinner=False)
    def cached_marine(coords: tuple) -> dict:
        return intel.fetch_marine(coords)          # raises IntelError -> not cached

    @st.cache_data(ttl=900, show_spinner=False)
    def cached_news() -> dict:
        return intel.fetch_news()

    def show_df(df: pd.DataFrame, **kw) -> None:
        try:
            st.dataframe(df, width="stretch", **kw)
        except TypeError:
            st.dataframe(df, use_container_width=True, **kw)

    def md(text: str) -> str:
        """Escape characters Streamlit markdown would otherwise interpret ($ starts LaTeX)."""
        return text.replace("$", "\\$")

    model, G0 = get_model(), get_base_graph()
    llm, llm_status = get_llm()
    llm_label = llm_status if llm else "Rules-based command mode (no LLM key)"
    if "ops" not in ss:
        ss["ops"] = Ops(model)
    ops: Ops = ss["ops"]
    ops.model = model

    # ----------------------------------------------------------------------------------------------
    # Widget <-> controller sync: widget state is rewritten from the controller on every run (before the widgets
    # are created), so a change made by the copilot or the demo shows up in the controls immediately.
    # ----------------------------------------------------------------------------------------------
    def sync_widgets() -> None:
        c = ops.cfg
        ss["cfg_origin"], ss["cfg_dest"] = c.origin, c.destination
        ss["cfg_vessel"], ss["cfg_owner"] = c.vessel_key, c.owner_key
        ss["cfg_fixed_speed"] = c.cruise_speed_kn is not None
        lo, hi = HULL[VESSELS[c.vessel_key].vessel_type]["speed"]
        ss["cfg_speed"] = float(min(max(c.cruise_speed_kn or VESSELS[c.vessel_key].speed_options_kn[1], lo), hi))
        ss["cfg_fuel"] = float(c.fuel_price)
        ss["cfg_pref"] = c.preference
        ss["cfg_w_time"], ss["cfg_w_cost"], ss["cfg_w_risk"] = (float(w) for w in c.weights)
        ss["cfg_avoid"] = list(c.avoid)
        ss["cfg_port_delay"], ss["cfg_canal_delay"] = float(c.port_delay_hours), float(c.canal_delay_hours)
        ss["cfg_dep_date"], ss["cfg_dep_time"] = c.departure.date(), c.departure.time().replace(tzinfo=None)
        for spec in EVENT_CATALOG:
            ss[f"inc_{spec.key}"] = ops.settings[spec.key]
        ss["inc_hurricane_track"] = ops.settings["hurricane_track"]
        ss["play_speed"] = next((k for k, v in SPEEDS.items() if v == ops.speed), "10 h/s")

    def flash(msg: str) -> None:
        ss.setdefault("_flash", []).append(msg)

    def cb(fn, *a, **kw):
        """Wrap a controller call as a widget callback; its message becomes a toast."""
        def _run():
            try:
                out = fn(*a, **kw)
            except Exception as e:                  # never let a callback crash the page
                out = f"Action failed: {e}"
            if isinstance(out, str) and out:
                flash(out)
        return _run

    def on_voyage() -> None:
        flash(ops.create_voyage(ss["cfg_origin"], ss["cfg_dest"], ss["cfg_vessel"],
                                ss["cfg_owner"] if ss["cfg_owner"] in V.VESSEL_OWNERS[ss["cfg_vessel"]] else None))

    def on_cfg(field: str) -> None:
        val = {
            "cruise_speed_kn": (float(ss["cfg_speed"]) if ss["cfg_fixed_speed"] else None),
            "fuel_price": float(ss["cfg_fuel"]),
            "preference": ss["cfg_pref"],
            "weights": (float(ss["cfg_w_time"]), float(ss["cfg_w_cost"]), float(ss["cfg_w_risk"])),
            "avoid": tuple(ss["cfg_avoid"]),
            "port_delay_hours": float(ss["cfg_port_delay"]),
            "canal_delay_hours": float(ss["cfg_canal_delay"]),
            "departure": datetime.combine(ss["cfg_dep_date"], ss["cfg_dep_time"], tzinfo=timezone.utc),
        }[field]
        flash(ops.update_config(**{field: val}))

    def on_incident(key: str) -> None:
        flash(ops.set_event(key, ss[f"inc_{key}"]))

    def on_speed() -> None:
        ops.set_speed(SPEEDS[ss["play_speed"]])

    def on_seek() -> None:
        ops.seek(float(ss["seek_h"]))

    def current_weather():
        return intel.current_for(ss.get("weather"), V.route_signature(ops.voyage))

    def intel_ctx() -> dict:
        route = ops_route_chokepoints()
        return {"weather_summary": intel.weather_summary(current_weather()),
                "news_summary": lambda region: intel.news_summary(ss.get("news"), route, region)}

    def on_chat(prompt: str | None = None) -> None:
        msg = (prompt or ss.get("chat_in") or "").strip()
        if not msg:
            return
        ops.chat.append({"role": "user", "content": msg})
        try:
            reply = copilot.handle(msg, ops, llm, intel_ctx())
            ops.chat.append({"role": "assistant", "content": reply.text, "mode": reply.mode,
                             "ops": [(n, a) for n, a, _ in reply.ops], "rejected": [r for _, r in reply.rejected]})
        except Exception as e:
            ops.chat.append({"role": "assistant", "content": f"Sorry, that command failed: {e}", "ops": [],
                             "mode": "command", "rejected": []})

    def ops_route_chokepoints() -> list[str]:
        ctx = V.build_context(ops.cfg, ops.settings, model)
        _, ahead, _ = V.split_at(ops.voyage, ctx.G)
        sail = [l for l in ahead if l.kind == "sail"]
        return V.summarise(ctx, sail).chokepoints if sail else []

    sync_widgets()
    for msg in ss.pop("_flash", []):
        st.toast(md(msg[:300]))

    started = ops.started()
    ports = hub_ports(G0)
    port_ids = list(ports)
    v = ops.voyage

    # ----------------------------------------------------------------------------------------------
    # Sidebar: ADVANCED configuration only (collapsed by default). Everything needed for a demo is on the page.
    # ----------------------------------------------------------------------------------------------
    with st.sidebar:
        st.markdown('<div class="panel-title">Advanced settings</div><div class="panel-sub">Vessel economics, '
                    'optimisation, delays and extra scenarios. Changes re-plan from the ship\'s position.</div>',
                    unsafe_allow_html=True)
        with st.expander("Voyage details", expanded=False):
            st.selectbox("Cargo owner", list(V.VESSEL_OWNERS[ops.cfg.vessel_key]), key="cfg_owner",
                         format_func=lambda k: f"{CARGO_OWNERS[k].label} · {CARGO_OWNERS[k].cargo}",
                         on_change=on_voyage, disabled=started)
            d1, d2 = st.columns(2)
            d1.date_input("Departure (UTC)", key="cfg_dep_date", on_change=on_cfg, args=("departure",),
                          disabled=started)
            d2.time_input("Time", key="cfg_dep_time", on_change=on_cfg, args=("departure",), disabled=started,
                          step=3600)
        with st.expander("Vessel & fuel economics", expanded=False):
            lo, hi = HULL[VESSELS[ops.cfg.vessel_key].vessel_type]["speed"]
            st.toggle("Fixed cruising speed", key="cfg_fixed_speed", on_change=on_cfg, args=("cruise_speed_kn",),
                      help="Off: the optimiser picks the cheapest of the vessel's speed options on every lane.")
            st.slider("Cruising speed (kn)", float(lo), float(hi), step=0.5, key="cfg_speed", on_change=on_cfg,
                      args=("cruise_speed_kn",), disabled=not ops.cfg.cruise_speed_kn)
            st.slider("Fuel price ($/t VLSFO)", 300.0, 1600.0, step=10.0, key="cfg_fuel", on_change=on_cfg,
                      args=("fuel_price",))
            agent = ops.cfg.agent()
            spd = agent.speed_options_kn[len(agent.speed_options_kn) // 2]
            from routing_engine import predict_tonnes_per_day
            tpd = float(predict_tonnes_per_day(model, agent.vessel_type, spd, agent.draft_m, agent.cargo_weight_t,
                                               15.0, 1.5)[0])
            st.caption(md(f"Model estimate at {spd:g} kn, calm sea: **{tpd:,.0f} t/day** "
                          f"(≈ ${tpd * ops.cfg.fuel_price / 1e3:,.0f}k/day)."))
        with st.expander("Route optimisation", expanded=False):
            st.radio("Objective", list(V.PREFERENCES), key="cfg_pref", horizontal=True,
                     format_func=V.PREFERENCES.get, on_change=on_cfg, args=("preference",),
                     help=" · ".join(f"{V.PREFERENCES[k]}: {t}" for k, t in V.PREFERENCE_HELP.items()))
            if ops.cfg.preference == "balanced":
                st.caption("Balanced weights")
                st.slider("Time", 0.0, 3.0, step=0.25, key="cfg_w_time", on_change=on_cfg, args=("weights",))
                st.slider("Cost", 0.0, 3.0, step=0.25, key="cfg_w_cost", on_change=on_cfg, args=("weights",))
                st.slider("Risk", 0.0, 3.0, step=0.25, key="cfg_w_risk", on_change=on_cfg, args=("weights",))
            st.multiselect("Avoid chokepoints", list(CHOKEPOINTS), key="cfg_avoid", format_func=CHOKEPOINTS.get,
                           on_change=on_cfg, args=("avoid",), help="Treated as closed for this voyage.")
        with st.expander("Canal & berth delays", expanded=False):
            st.number_input("Berth wait at destination (h)", 0.0, 240.0, step=6.0, key="cfg_port_delay",
                            on_change=on_cfg, args=("port_delay_hours",))
            st.number_input("Extra delay per canal transit (h)", 0.0, 240.0, step=6.0, key="cfg_canal_delay",
                            on_change=on_cfg, args=("canal_delay_hours",))
            p1, p2 = st.columns(2)
            cport = p1.selectbox("Congested port", port_ids, index=port_ids.index(ops.cfg.destination),
                                 key="cong_port", format_func=lambda p: ports[p])
            chrs = p2.number_input("Extra wait (h)", 0.0, 480.0, 48.0, 12.0, key="cong_hours")
            st.button("Apply port congestion (scenario)", on_click=cb(ops.set_congestion, cport, chrs),
                      width="stretch")
        with st.expander("Storm scenario", expanded=False):
            w1, w2 = st.columns(2)
            wave = w1.number_input("Wave height (m)", 2.0, 15.0, 6.0, 0.5, key="storm_wave")
            rad = w2.number_input("Radius (nm)", 100.0, 1000.0, 300.0, 50.0, key="storm_radius")
            s1, s2 = st.columns(2)
            s1.button("Storm ahead", on_click=cb(ops.add_storm, wave, None, None, rad), width="stretch")
            s2.button("Clear storms", on_click=cb(ops.clear_storms), width="stretch",
                      disabled=not ops.settings["storm_cells"])
            st.caption(f"Cores ≥ 5 m waves are avoided when an alternative exists. "
                       f"Active storm cells: {len(ops.settings['storm_cells'])}.")
        with st.expander("More scenarios & fees", expanded=False):
            st.slider("Red Sea security escalation", 0.0, 1.0, step=0.05, key="inc_red_sea_escalation",
                      on_change=on_incident, args=("red_sea_escalation",))
            st.slider("Suez fee ×", 1.0, 10.0, step=0.5, key="inc_suez_toll_multiplier", on_change=on_incident,
                      args=("suez_toll_multiplier",))
            st.slider("Panama fee ×", 1.0, 5.0, step=0.5, key="inc_panama_toll_multiplier", on_change=on_incident,
                      args=("panama_toll_multiplier",))
            st.slider("Taiwan Strait tension", 0.0, 1.0, step=0.05, key="inc_taiwan_tension",
                      on_change=on_incident, args=("taiwan_tension",))
            st.slider("Atlantic hurricane (severity)", 0.0, 1.0, step=0.05, key="inc_atlantic_hurricane",
                      on_change=on_incident, args=("atlantic_hurricane",))
            st.selectbox("Hurricane track", list(HURRICANE_TRACKS), key="inc_hurricane_track",
                         format_func=lambda k: HURRICANE_TRACKS[k][0], on_change=on_incident,
                         args=("hurricane_track",))
            st.toggle("Panama Canal drought", key="inc_panama_drought", on_change=on_incident,
                      args=("panama_drought",))
            preset = st.selectbox("Preset", list(PRESETS), key="preset_pick")
            st.button("Apply preset", on_click=cb(ops.apply_settings, PRESETS[preset], f"preset '{preset}'"),
                      width="stretch")
        with st.expander("Data sources & methodology", expanded=False):
            st.markdown(ui.METHODOLOGY, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------------------------------------
    state = ("blocked" if v.status == "blocked" else "arrived" if v.status == "arrived" else
             "active" if ops.playing else "idle" if v.status == "planned" and v.t == 0 else "paused")
    st.markdown(ui.topbar(state, llm_label), unsafe_allow_html=True)
    active = describe(ops.settings)
    st.markdown(f'<div class="subline">Live operations: plan, sail, disrupt and reroute a <b>simulated</b> vessel. '
                f'{"Active scenarios (simulated): <b>" + html.escape(" · ".join(active)) + "</b>" if active else "No scenarios active."}'
                f' <span class="intel-meta">Advanced settings: sidebar »</span></div>', unsafe_allow_html=True)
    tab_live, tab_stress = st.tabs(["Live operations", "Multi-agent stress test"])

    with tab_live:
        change = ops.demo if ops.demo else ops.last_change
        if change:
            st.markdown(ui.change_banner({"before": change["before"], "after": change["after"]},
                                         demo=bool(ops.demo)), unsafe_allow_html=True)

        # ---------------- Row 1: Voyage & Simulation | Alerts & Intelligence
        top_l, top_r = st.columns([1.05, 1], gap="medium")
        with top_l:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Voyage & Simulation</div><div class="panel-sub">Ports, playback '
                            'and scenario controls. Scenarios are SIMULATED, never real closures.</div>',
                            unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.selectbox("From", port_ids, key="cfg_origin", format_func=lambda p: ports[p],
                             on_change=on_voyage, disabled=started)
                c2.selectbox("To", port_ids, key="cfg_dest", format_func=lambda p: ports[p],
                             on_change=on_voyage, disabled=started)
                c3.selectbox("Vessel", list(VESSELS), key="cfg_vessel", format_func=lambda k: VESSELS[k].label,
                             on_change=on_voyage, disabled=started,
                             help="Bulk carriers use the tanker hull class and LNG carriers the container hull class "
                                  "for fuel estimates.")
                b1, b2, b3, b4 = st.columns([1.1, 1, 1, 1.6])
                if ops.playing:
                    b1.button("⏸ Pause", on_click=cb(ops.pause), width="stretch", key="btn_pause")
                elif v.status == "planned" and v.t == 0:
                    b1.button("▶ Start", type="primary", on_click=cb(ops.start), width="stretch", key="btn_start",
                              disabled=v.status == "blocked")
                else:
                    b1.button("▶ Resume", type="primary", on_click=cb(ops.resume), width="stretch", key="btn_resume",
                              disabled=v.status in ("blocked", "arrived"))
                b2.button("+1 day", on_click=cb(ops.advance_hours, 24.0), width="stretch", key="btn_day",
                          disabled=v.status in ("blocked", "arrived"))
                b3.button("Reset", on_click=cb(ops.reset), width="stretch", key="btn_reset")
                b4.selectbox("Playback (simulated h/s)", list(SPEEDS), key="play_speed", on_change=on_speed,
                             label_visibility="collapsed")
                if started:
                    st.caption("Underway: ports and vessel are locked (Reset to change).")
                dm1, dm2 = st.columns([1.6, 1])
                dm1.button("↻ Replay hackathon demo" if ops.demo else "▶ Hackathon demo: Suez closure",
                           type="primary", on_click=cb(ops.run_demo, "demo", True), width="stretch", key="demo_run",
                           help="Rotterdam → Shanghai time-lapse to the Port Said approach, then a SIMULATED Suez "
                                "closure diverts the ship via the Cape of Good Hope from where it is.")
                if ops._pre_demo is not None:
                    dm2.button("Exit demo", on_click=cb(ops.exit_demo), width="stretch", key="demo_exit",
                               help="Restore the voyage, assumptions and scenarios you had before the demo.")
                else:
                    dm2.button("Instant demo", on_click=cb(ops.run_demo, "demo", False), width="stretch",
                               key="demo_instant", help="Same scenario without the time-lapse.")
                st.markdown('<div class="section-name" style="margin-top:6px">Incidents (simulated)</div>',
                            unsafe_allow_html=True)
                t1, t2 = st.columns(2)
                for i, key in enumerate(("suez_blockage", "bab_el_mandeb_closure", "hormuz_blockade", "panama_closure")):
                    label = {"suez_blockage": "Suez Canal closed", "bab_el_mandeb_closure": "Bab-el-Mandeb closed",
                             "hormuz_blockade": "Hormuz closed", "panama_closure": "Panama Canal closed"}[key]
                    (t1 if i % 2 == 0 else t2).toggle(label, key=f"inc_{key}", on_change=on_incident, args=(key,),
                                                      help=EVENT_BY_KEY[key].help)
                st.button("Clear all scenarios", on_click=cb(ops.clear_all), width="stretch", key="btn_clear",
                          disabled=not describe(ops.settings))

        with top_r:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Alerts & Intelligence</div><div class="panel-sub">Active risks · '
                            'live weather (Open-Meteo) · news & security headlines (advisory)</div>',
                            unsafe_allow_html=True)
                alerts = []
                if v.status == "blocked":
                    alerts.append(ui.alert("Voyage blocked", v.block_reason, "critical"))
                if v.status == "arrived":
                    alerts.append(ui.alert("Arrived", f"Voyage complete after {v.t / 24:.1f} days.", "info"))
                if any(l.note for l in v.legs if l.kind == "sail" and l.note):
                    alerts.append(ui.alert("Forced through", "No route avoids every storm / high-premium lane.", ""))
                if v.reroutes:
                    last = next((e for e in reversed(v.timeline) if e["kind"] == "reroute"), None)
                    if last:
                        alerts.append(ui.alert(f"Rerouted ({v.reroutes}×)", last["title"], "info"))
                for d in active:
                    alerts.append(ui.alert("Simulated scenario", d, ""))
                wx = current_weather()
                if wx and wx.get("worst") and wx["worst"]["level"] >= 2:
                    wr = wx["worst"]
                    alerts.append(ui.alert(f"Forecast hazard ahead: {wr['severity']} (Open-Meteo)",
                                           f"{wr['values'].get('wave_height')} m waves {wr['label'].lower()} at simulated "
                                           f"ETA {wr['eta']:%b %d %H:%M} UTC. {wr['action']}",
                                           "critical" if wr["level"] == 3 else ""))
                news = ss.get("news")
                if news and news.get("articles") and not news.get("demo"):
                    for adv in intel.advisories(intel.rank_news(news["articles"], ops_route_chokepoints())):
                        alerts.append(ui.alert(f"News advisory (unverified): {adv['label']}",
                                               f"\"{adv['headline']}\" ({adv['source']}). No corridor was closed.", ""))
                st.markdown('<div style="max-height:150px;overflow-y:auto">'
                            + ("".join(alerts) or ui.alert("All clear", "No warnings for the current voyage.", "info"))
                            + "</div>", unsafe_allow_html=True)
                intel_slot = st.container(height=300, border=False)

        # ---------------- Row 2: Route KPIs (full width, ticks during playback)
        kpi_fragment(st, ops, ui, V)()

        # ---------------- Row 3: Live route map | Operations copilot
        map_col, chat_col = st.columns([2.3, 1], gap="medium")
        with map_col:
            map_fragment(st, ss, ops, model, ui, V, HAS_FOLIUM, st_folium, build_ops_base_map, build_ops_layer,
                         build_ops_map_html, components, on_seek, current_weather)()
        with chat_col:
            chip = "mode-chip" if llm else "mode-chip cmd"
            st.markdown(f'<div class="copilot-head"><div class="panel-title">Operations Copilot</div>'
                        f'<span class="{chip}">{html.escape(llm_label)}</span></div>', unsafe_allow_html=True)
            with st.container(height=430, border=True):
                if not ops.chat:
                    st.markdown(md("Give me instructions (\"Close Suez and reroute my ship\") or ask questions "
                                   "(\"What if Suez closes?\"). Questions and what-ifs never change the voyage; "
                                   "what-ifs run on a copy and are labelled PREVIEW."))
                for m in ops.chat[-40:]:
                    with st.chat_message(m["role"], avatar="🧭" if m["role"] == "assistant" else "👤"):
                        if m["role"] == "assistant":
                            tag = "LLM" if m.get("mode") == "llm" else "rules"
                            chips = "".join(f'<span class="op-chip">{html.escape(n)}'
                                            f'{html.escape(json.dumps(a)) if a else ""}</span>' for n, a in m.get("ops", []))
                            st.markdown(f'<span class="op-chip">{tag}</span>{chips}', unsafe_allow_html=True)
                        st.markdown(md(m["content"]))
                        if m.get("rejected"):
                            st.caption("Rejected (failed validation): " + "; ".join(m["rejected"]))
            examples = [("Close Suez + reroute", "Close Suez and reroute my ship"),
                        ("What if Suez closes?", "What if the Suez Canal closes?"),
                        ("Advance 5 days", "Advance the voyage by 5 days"),
                        ("6 m storm ahead", "A storm with 6 m waves is affecting our route. Reroute to avoid it."),
                        ("Safest route?", "What is the safest available route?"),
                        ("Pause + risk", "Pause the vessel and explain our current operational risk")]
            ex_cols = st.columns(2)
            for i, (lab, ex) in enumerate(examples):
                ex_cols[i % 2].button(lab, key=f"ex_{i}", on_click=on_chat, args=(ex,), width="stretch", help=ex)
            st.chat_input("Instruction or question…", key="chat_in", on_submit=on_chat)

        # ---------------- Row 4: Rerouting decision & event timeline
        st.markdown('<div class="panel-title" style="margin-top:14px">Rerouting decision & event timeline</div>'
                    '<div class="panel-sub">Recommendation, before/after comparison and incident history '
                    '(model estimates).</div>', unsafe_allow_html=True)
        d_left, d_right = st.columns([1.35, 1], gap="medium")
        with d_left:
            if change:
                st.markdown(ui.recommendation(change, v.status, V.PREFERENCES[ops.cfg.preference]),
                            unsafe_allow_html=True)
                st.markdown(ui.comparison_table(change["before"], change["after"],
                                                "Before (via Suez)" if ops.demo else "Before",
                                                "After closure" if ops.demo else "After"), unsafe_allow_html=True)
            elif v.status == "blocked":
                st.markdown(ui.decision(v.block_reason, True), unsafe_allow_html=True)
            elif v.decision:
                st.markdown(ui.decision(v.decision, rec=v.recommendation), unsafe_allow_html=True)
            opts = v.candidates if v.candidates else ops.options()
            if opts:
                rows = []
                for c in opts:
                    s_ = c.summary
                    rows.append({"Option": c.name, "Selected": "✓" if c.chosen else "", "Corridor": s_.corridor,
                                 "Remaining nm": round(s_.nm), "Days": round(s_.hours / 24, 1),
                                 "All-in $M": round(s_.total_usd / 1e6, 2), "Fuel t": round(s_.fuel_t),
                                 "Risk /100": round(s_.risk_index), "Forced": "yes" if s_.forced else ""})
                with st.expander("Route options considered" + (" (last re-evaluation)" if v.candidates
                                                                else " (from current position)"), expanded=False):
                    show_df(pd.DataFrame(rows), hide_index=True)
                    st.caption(f"Objective: {V.PREFERENCES[ops.cfg.preference]} - "
                               f"{V.PREFERENCE_HELP[ops.cfg.preference]} Model estimates for the remaining voyage.")
                    if v.decision:
                        st.caption(md("Full rationale: " + v.decision))
        with d_right:
            with st.container(height=360, border=False):
                st.markdown(ui.timeline(v.timeline), unsafe_allow_html=True)

    with tab_stress:
        stress_tab(st, ss, ops, model, G0, ports, ui, show_df, HAS_FOLIUM, components, build_folium_map, st_folium,
                   build_map_html, make_agent, profile_matrix, edges_dataframe, HULL, CARGO_OWNERS, np)

    with st.expander("Data sources, model and limitations"):
        st.markdown(ui.METHODOLOGY, unsafe_allow_html=True)
    st.markdown(ui.FOOTER, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------------------
    # Weather / News tabs: rendered LAST into their slot under Alerts, so slow or failing network calls never
    # delay the rest of the page. Refreshes itself every 10 minutes; never runs on animation frames.
    # ----------------------------------------------------------------------------------------------
    @st.fragment(run_every=600)
    def intel_panel():
        t_w, t_n = st.tabs(["Weather", "News & Security"])
        with t_w:
            render_weather(st, ss, ops, V, intel, cached_marine, cb, flash)
        with t_n:
            render_news(st, ss, ops, intel, cached_news, cb, ops_route_chokepoints)

    with intel_slot:
        intel_panel()


# ==================================================================================================
# Fragments that re-run on their own every TICK_S while playing. Both call ops.tick(): the clock advances by REAL
# elapsed time x playback speed, so whichever fragment runs first moves the ship and the other one sees ~0 s.
# ==================================================================================================
def kpi_fragment(st, ops, ui, V):
    @st.fragment(run_every=TICK_S if ops.playing else None)
    def kpis():
        was_playing = ops.playing
        ops.tick()
        if was_playing and not ops.playing:      # arrived / blocked / demo trigger: refresh the whole page
            st.rerun()
        v, G = ops.voyage, ops.G
        s = V.ship_state(v, G)
        p = V.projected(v, G)
        if v.status == "blocked":
            eta = ("BLOCKED", "No feasible passage: ship holding position", "red")
        elif v.status == "arrived":
            eta = (f"{p['eta']:%b %d %H:%M}", f"Arrived after {v.t / 24:.1f} days", "mint")
        else:
            eta = (f"{p['eta']:%b %d %H:%M} UTC", f"{s.remaining_h / 24:.1f} days to go (simulated clock)", "")
        risk = p["remaining_risk_index"] if v.status != "arrived" else p["risk_index"]
        cards = [("ETA", eta[0], eta[1], eta[2], True),
                 ("Distance remaining", f"{s.remaining_nm:,.0f} nm", f"{s.traveled_nm:,.0f} nm sailed", "blue", False),
                 ("Projected cost", ui._usd(p["total_usd"]),
                  f"fuel {p['fuel_t']:,.0f} t · tolls {ui._usd(p['toll_usd'])} · spent {ui._usd(p['spent_usd'])}",
                  "gold", True),
                 ("Risk ahead", f"{risk:.0f}/100", V.risk_label(risk) + " (synthetic risk model)",
                  "red" if risk >= 65 else "gold" if risk >= 45 else "mint", True)]
        st.markdown(ui.metric_cards(cards), unsafe_allow_html=True)
        state = ("blocked" if v.status == "blocked" else "playing" if ops.playing else
                 "idle" if v.status == "planned" and v.t == 0 else "paused")
        clock = V.calendar(v, v.t)
        st.markdown(ui.live_strip(state, [
            ("Sim clock", f"{clock:%Y-%m-%d %H:%M} UTC"), ("Day", f"{v.t / 24:.1f}"),
            ("SIMULATED position", f"{s.lat:.2f}, {s.lon:.2f}"), ("Course", f"{s.course_deg:03.0f}°"),
            ("SOG", f"{s.sog_kn:.1f} kn"), ("Status", s.activity)], jump_h=ops.speed * TICK_S),
            unsafe_allow_html=True)
    return kpis


def map_fragment(st, ss, ops, model, ui, V, has_folium, st_folium, build_base, build_layer, build_svg, components,
                 on_seek, current_weather):
    import json

    @st.fragment(run_every=TICK_S if ops.playing else None)
    def live_map():
        was_playing = ops.playing
        ops.tick()
        if was_playing and not ops.playing:
            st.rerun()
        v, G = ops.voyage, ops.G
        s = V.ship_state(v, G)
        done, ahead, _ = V.split_at(v, G)
        done_pts, ahead_pts = V.legs_polyline(G, done), V.legs_polyline(G, ahead)
        if done_pts and ahead_pts and done_pts[-1] != ahead_pts[0]:
            ahead_pts = [done_pts[-1]] + ahead_pts
        wps = V.waypoints(v, G)
        alts = [(c.name, V.legs_polyline(G, c.summary.legs)) for c in v.candidates
                if not c.chosen and c.name != "Hold course"]
        color = {"Container": "#58aaff", "Tanker": "#f5bf75"}.get(V.VESSELS[ops.cfg.vessel_key].vessel_type)
        modes = ["Interactive map", "Offline map"] if has_folium else ["Offline map"]
        h1, h2 = st.columns([3, 1.3])
        h1.markdown('<div class="panel-title">Live route map <span class="tag-sim">SIMULATED POSITION · NOT AIS</span>'
                    '</div>', unsafe_allow_html=True)
        mode = h2.selectbox("Map", modes, key="map_mode", label_visibility="collapsed",
                            help="Offline map: no tiles or folium needed (use it if map tiles are blocked).")
        if mode == "Interactive map":
            base_key = json.dumps(ops.settings, sort_keys=True, default=str)
            if ss.get("_base_key") != base_key:             # static layers rebuilt only when scenarios change
                ctx = V.build_context(ops.cfg, ops.settings, model)
                ss["_base_map"] = build_base(ctx.G, ops.settings)
                ss["_base_key"] = base_key
            wx = current_weather() or {}
            layer = build_layer(G, v, done_pts, ahead_pts, wps, s, alts, wx.get("points", []), color)
            base = ss["_base_map"]
            out = st_folium(base, feature_group_to_add=layer, key="ops_map", height=520,
                            use_container_width=True, returned_objects=["last_object_clicked"])
            # st_folium attaches the dynamic layer to the map object; detach it so the cached base map's script stays
            # byte-identical: only the voyage layer is pushed each tick (no reload, zoom/pan preserved).
            for k in [k for k in base._children if k.startswith("feature_group_")]:
                del base._children[k]
            click = (out or {}).get("last_object_clicked")
            if click and click != ss.get("_last_click"):
                ss["_last_click"] = click
                near = min(wps, key=lambda w: (w["lat"] - click["lat"]) ** 2 + (w["lon"] - click["lng"]) ** 2,
                           default=None)
                if near and (near["lat"] - click["lat"]) ** 2 + (near["lon"] - click["lng"]) ** 2 < 0.5:
                    ops.seek(near["t"])
                    st.rerun(scope="fragment")
        else:
            ctx = V.build_context(ops.cfg, ops.settings, model)
            components.html(build_svg(ctx.G, ops.settings, done_pts, ahead_pts, v.superseded, alts,
                                      v.blocked_remaining, s, v.status, color, height=500), height=508)
        total = max(v.total_hours, 0.1)
        ss["seek_h"] = float(min(round(v.t, 1), total))
        st.slider("Voyage time (hours since departure): drag to jump, or click a waypoint", 0.0, float(total),
                  step=1.0, key="seek_h", on_change=on_seek, disabled=v.status == "blocked" or not v.legs)
        st.caption("Mint = completed track · solid = remaining plan · grey dashed = previous committed route · amber dotted = "
                   "alternatives considered · red dashed = blocked plan · circles = Open-Meteo forecast at the "
                   "simulated ETA · purple = storm scenario.")
    return live_map


# ==================================================================================================
# Live Maritime Intelligence: Weather and News & Security tabs
# ==================================================================================================
def render_weather(st, ss, ops, V, intel, cached_marine, cb, flash):
    import html
    import time
    G = ops.G
    v = ops.voyage
    now = time.time()
    f1, f2 = st.columns([1.5, 1])
    fi_on = f1.toggle("Forecast-informed simulation", key="fi_mode",
                      help="Opt-in: Open-Meteo hazards ahead of the ship (waves ≥ threshold at the simulated ETA) "
                           "become labelled SIMULATED storm cells and the remaining route is re-assessed. "
                           "Not a navigation recommendation.")
    thr = f2.number_input("Hazard ≥ (m)", 2.0, 12.0, 4.0, 0.5, key="fi_threshold",
                          help="Significant wave height that counts as a hazard.")
    if v.status == "arrived" or not v.legs:
        st.caption("No remaining route to forecast.")
        return
    sig = V.route_signature(v)
    pts = V.remaining_samples(v, G)
    coords = tuple((round(p["lat"], 1), round(p["lon"], 1)) for p in pts)
    if now < ss.get("_wx_backoff", 0):
        w = intel.reuse_or_unavailable(ss.get("_wx_last"), sig, ss.get("_wx_err", "provider unavailable"))
    else:
        try:
            with st.spinner("Fetching Open-Meteo marine forecast…"):
                raw = cached_marine(coords)
            w = intel.with_route(intel.assess_weather(raw, pts, ops.cfg.departure), sig, coords)
            ss["_wx_last"] = w
        except intel.IntelError as e:
            ss["_wx_backoff"], ss["_wx_err"] = now + 120, str(e)
            w = intel.reuse_or_unavailable(ss.get("_wx_last"), sig, str(e))
    if w.get("status") != "unavailable":
        w = dict(w, route_sig=sig)
    ss["weather"] = w
    if fi_on:
        msg = ops.apply_forecast_hazards(w, thr)
        if msg:
            flash(msg)
            st.rerun()                               # re-plan visible everywhere (converges: cells only accumulate)
    status = w["status"] + (" · STALE" if w.get("stale") else "")
    st.markdown(f'<div class="intel-meta"><span class="tag-live">LIVE DATA</span> Status: <b>{html.escape(status)}</b> · '
                f'{html.escape(w["source"])}' + (f' · fetched {html.escape(w["fetched_at"])} UTC' if w.get("fetched_at")
                                                 else "") + "</div>", unsafe_allow_html=True)
    if w.get("note"):
        st.caption(w["note"])
    if not w["points"]:
        st.caption("The simulator keeps running on its synthetic sea-state model.")
        return
    rows = []
    for p in w["points"]:
        vals = p["values"]
        lvl = p["level"]
        sev = f'<span class="sev l{lvl if lvl >= 0 else "n"}">{html.escape(p["severity"])}</span>'
        if p["status"] == "forecast":
            detail = (f"waves {vals.get('wave_height')} m · swell {vals.get('swell_wave_height')} m · "
                      f"period {vals.get('wave_period')} s"
                      + (f" · wind {vals.get('wind_speed_10m')} kn" if vals.get("wind_speed_10m") is not None else "")
                      + (f" · current {vals.get('ocean_current_velocity')} km/h" if vals.get("ocean_current_velocity") is not None else "")
                      + (f" · SST {vals.get('sea_surface_temperature')} °C" if vals.get("sea_surface_temperature") is not None else ""))
            if (vals.get("wave_height") or 0) >= thr:
                detail += f" · HAZARD (≥ {thr:g} m): " + intel.operational_impact(vals.get("wave_height"),
                                                                                  vals.get("wind_speed_10m"))
        else:
            detail = p["status"]
        rows.append(f'<div class="intel-row"><b>{html.escape(p["label"])}</b> {sev}<br>{html.escape(detail)}'
                    f'<div class="intel-meta">simulated ETA {p["eta"]:%b %d %H:%M} UTC · {p["lat"]:.1f}, {p["lon"]:.1f}'
                    + (f' · {html.escape(p["action"])}' if p["level"] >= 1 else "") + "</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    worst = w.get("worst")
    if worst and worst["level"] >= 2 and not fi_on:
        st.button(f"Simulate storm at worst point ({worst['values'].get('wave_height')} m)", key="wx_storm",
                  on_click=cb(ops.add_storm, max(float(worst["values"].get("wave_height") or 5.0), 5.0),
                              worst["lat"], worst["lon"], 250.0, f"forecast {worst['label'].lower()}",
                              "manual", "open-meteo"), width="stretch")
    if any(c.get("source") == ops.FORECAST_SOURCE for c in ops.settings["storm_cells"]):
        st.button("Remove forecast-informed storm cells", key="fi_clear", on_click=cb(ops.clear_forecast_hazards),
                  width="stretch")
    st.caption("Forecasts are matched to the ship's SIMULATED arrival time at each point. Data: Open-Meteo (CC BY 4.0). "
               "Advisory unless forecast-informed simulation is on; even then the result is a modelled scenario.")


def render_news(st, ss, ops, intel, cached_news, cb, route_fn):
    import html
    import time
    now = time.time()
    c1, c2 = st.columns([1, 1])
    demo = c2.toggle("Demo headlines", key="news_demo", help="Clearly labelled sample items for offline demos.")
    if c1.button("Refresh news", key="news_refresh", width="stretch",
                 disabled=now - ss.get("_news_refreshed", 0) < 15):
        ss["_news_refreshed"] = now
        ss["_news_backoff"] = 0
        cached_news.clear()
    if demo:
        news = intel.demo_news()
    elif now < ss.get("_news_backoff", 0):
        news = ss.get("_news_last") or {"articles": [], "error": ss.get("_news_err", "unavailable")}
        if ss.get("_news_last"):
            news = dict(news, stale=True)
    else:
        try:
            with st.spinner("Fetching maritime headlines…"):
                news = cached_news()
            ss["_news_last"] = news
        except intel.IntelError as e:
            ss["_news_backoff"], ss["_news_err"] = now + 180, str(e)
            news = dict(ss["_news_last"], stale=True) if ss.get("_news_last") else {"articles": [], "error": str(e)}
    ss["news"] = news
    if not news.get("articles"):
        st.warning(f"Live news unavailable: {news.get('error', 'no results')}. No headlines are shown rather than "
                   "invented. Use the demo toggle for a labelled sample.")
        return
    meta = (f"Source: {news['provider']}" + (f" · fetched {news['fetched_at']} UTC" if news.get("fetched_at") else "")
            + (" · STALE (refresh failed)" if news.get("stale") else ""))
    if news.get("errors"):
        meta += " · fallback used (" + "; ".join(news["errors"])[:120] + ")"
    st.markdown(f'<div class="intel-meta">{html.escape(meta)}</div>', unsafe_allow_html=True)
    route = route_fn()
    ranked = intel.rank_news(news["articles"], route)
    rows = []
    for a in ranked:
        when = f"{a['published']:%b %d %H:%M} UTC" if a["published"] else "no date (demo)"
        title = html.escape(a["title"])
        link = f'<a href="{html.escape(a["url"])}" target="_blank" rel="noopener">{title}</a>' if a["url"] else title
        tags = " · ".join(a["region_labels"])
        flag = ' <span class="sev l2">ON ROUTE</span>' if a["on_route"] else ""
        rows.append(f'<div class="intel-row">{link}{flag}<div class="intel-meta">{html.escape(a["source"] or "unknown")}'
                    f' · {when} · {html.escape(tags)}</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    if not news.get("demo"):
        for adv in intel.advisories(ranked):
            already = ops.settings.get({"suez_closure": "suez_blockage", "hormuz_closure": "hormuz_blockade"}.get(
                adv["scenario"], adv["scenario"]))
            st.button(f"Simulate {adv['label']} closure (scenario)", key=f"adv_{adv['region']}", disabled=bool(already),
                      on_click=cb(ops.set_disruption, adv["scenario"], True), width="stretch")
    st.caption("Headlines are advisory. A headline is not evidence that a corridor is closed, so nothing is closed "
               "automatically; buttons above only run a labelled simulation scenario.")


# ==================================================================================================
# Original multi-agent stress test (now a tab; uses the shared scenario settings)
# ==================================================================================================
def stress_tab(st, ss, ops, model, G0, ports, ui, show_df, has_folium, components, build_folium_map, st_folium,
               build_map_html, make_agent, profile_matrix, edges_dataframe, HULL, CARGO_OWNERS, np):
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
    for x, tone in zip(comps, ("blue", "gold")):
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
        if has_folium:
            st_folium(build_folium_map(G1, layers, active_settings), height=540, use_container_width=True,
                      returned_objects=[], key="folium_map")
        else:
            svg_layers = [{"label": l["label"], "result": l["result"], "hold": l["hold"], "color": col}
                          for l, col in zip(layers, (ui.CONTAINER_COLOR, ui.TANKER_COLOR))]
            components.html(build_map_html(G1, svg_layers, active_settings, height=540), height=548)
    with right:
        st.markdown('<div class="section-name">Agent decisions</div>', unsafe_allow_html=True)
        for x, col, (o, d) in zip(comps, (ui.CONTAINER_COLOR, ui.TANKER_COLOR), ((c_o, c_d), (t_o, t_d))):
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


if _in_streamlit():
    render_ui()
elif __name__ == "__main__":
    console_demo()
