"""
tabs.py - renderers for the Live operations tab: header, copilot, and the five right-hand tabs.

Each function takes the ``Page`` context (page.py) and draws one region. Nothing here mutates simulation state
except through ``Ops`` (widget callbacks / ``pg.cb``).
"""
from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

import economics as ECO
import intel
import panels as P
import risk as RISK
import sources as SRC
import ui
import voyage as V
from agents import CARGO_OWNERS, VESSELS
from data_engine import CHOKEPOINTS
from events import HURRICANE_TRACKS, PRESETS, describe
from intel_ui import render_news, render_weather
from ml_models import HULL
from ops import SPEEDS
from page import cached_marine, cached_news, md, show_df
from routing_engine import predict_tonnes_per_day

RIGHT_TABS = ["Operate", "Optimize", "Route comparison", "Alerts & intelligence", "Data sources"]

EXAMPLES = [("Close Suez + reroute", "Close Suez and reroute my ship"),
            ("What if Suez closes?", "What if the Suez Canal closes?"),
            ("Advance 5 days", "Advance the voyage by 5 days"),
            ("6 m storm ahead", "A storm with 6 m waves is affecting our route. Reroute to avoid it."),
            ("Safest route?", "What is the safest available route?"),
            ("Pause + risk", "Pause the vessel and explain our current operational risk")]


# ==================================================================================================
# Header and banner
# ==================================================================================================
def render_header(pg) -> None:
    ops, v = pg.ops, pg.ops.voyage
    state = ("blocked" if v.status == "blocked" else "arrived" if v.status == "arrived" else
             "active" if ops.playing else "idle" if v.status == "planned" and v.t == 0 else "paused")
    st.markdown(ui.topbar(state, pg.llm_label), unsafe_allow_html=True)
    active = describe(ops.settings)
    scen = ("Active scenarios (simulated): <b>" + html.escape(" · ".join(active)) + "</b>") if active \
        else "No scenarios active."
    st.markdown(f'<div class="subline">Live operations: plan, sail, disrupt and reroute a <b>simulated</b> vessel. '
                f'{scen}</div>', unsafe_allow_html=True)


def render_change_banner(change, demo: bool) -> None:
    if change:
        st.markdown(ui.change_banner({"before": change["before"], "after": change["after"]}, demo=demo),
                    unsafe_allow_html=True)


# ==================================================================================================
# Copilot (docked under the map, never inside a tab)
# ==================================================================================================
def render_copilot(pg) -> None:
    ops = pg.ops
    chip = "mode-chip" if pg.llm else "mode-chip cmd"
    st.markdown(f'<div class="copilot-head"><div class="panel-title">Operations Copilot</div>'
                f'<span class="{chip}">{html.escape(pg.llm_label)}</span></div>', unsafe_allow_html=True)
    with st.container(height=300, border=True):
        _chat_log(ops)
    if getattr(ops, "proposal", None):
        st.button("Simulate this scenario", key="btn_sim_proposal", type="primary", width="stretch",
                  on_click=pg.on_chat, args=("Simulate the proposed scenario",),
                  help="Applies exactly the scenario proposed from your pasted report. Nothing changes until "
                       "you press this.")
    ex_cols = st.columns(3)
    for i, (lab, ex) in enumerate(EXAMPLES):
        ex_cols[i % 3].button(lab, key=f"ex_{i}", on_click=pg.on_chat, args=(ex,), width="stretch", help=ex)
    st.chat_input("Instruction or question…", key="chat_in", on_submit=pg.on_chat)


def _chat_log(ops) -> None:
    if not ops.chat:
        st.markdown(md("Give me instructions (\"Close Suez and reroute my ship\") or ask questions "
                       "(\"What if Suez closes?\"). Questions and what-ifs never change the voyage; "
                       "what-ifs run on a copy and are labelled PREVIEW. Paste a news report (start a short "
                       "one with \"News:\") and I read it as data and only propose a scenario."))
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


# ==================================================================================================
# Operate: voyage, playback, demo, incidents
# ==================================================================================================
def render_operate(pg) -> None:
    ops, v = pg.ops, pg.ops.voyage
    p_now = V.projected(v, pg.G0)
    risk_now = p_now["risk_index"] if v.status == "arrived" else p_now["remaining_risk_index"]
    st.markdown(P.status_line(v.recommendation, ops.route_corridor(), risk_now), unsafe_allow_html=True)
    _voyage_pickers(pg)
    _playback(pg)
    _demo_buttons(pg)
    _incident_picker(pg)
    with st.expander("Storms, fees and more scenarios", expanded=False):
        _more_scenarios(pg)


def _voyage_pickers(pg) -> None:
    ports, started = pg.ports, pg.ops.started()
    c1, c2, c3 = st.columns(3)
    c1.selectbox("From", pg.port_ids, key="cfg_origin", format_func=lambda p: ports[p],
                 on_change=pg.on_voyage, disabled=started)
    c2.selectbox("To", pg.port_ids, key="cfg_dest", format_func=lambda p: ports[p],
                 on_change=pg.on_voyage, disabled=started)
    c3.selectbox("Vessel", list(VESSELS), key="cfg_vessel", format_func=lambda k: VESSELS[k].label,
                 on_change=pg.on_voyage, disabled=started,
                 help="Ports and vessel lock once underway (Reset to change). Bulk carriers use the tanker "
                      "hull class and LNG carriers the container hull class for fuel estimates.")


def _playback(pg) -> None:
    ops, cb = pg.ops, pg.cb
    v = ops.voyage
    b1, b2, b3, b4 = st.columns([1.1, 1, 1, 1.4])
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
    b4.selectbox("Playback (simulated h/s)", list(SPEEDS), key="play_speed", on_change=pg.on_speed,
                 label_visibility="collapsed")


def _demo_buttons(pg) -> None:
    ops, cb = pg.ops, pg.cb
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


def _incident_picker(pg) -> None:
    ops = pg.ops
    i1, i2 = st.columns([3, 1.2])
    i1.multiselect("Incidents (simulated)", list(P.INCIDENTS), key="inc_picker",
                   format_func=P.INCIDENTS.get, on_change=pg.on_incident_picker,
                   placeholder="No incident active",
                   help="Scenarios are SIMULATED, never real closures. The ship is re-routed from where "
                        "it is whenever you add or remove one.")
    i2.button("Clear all", on_click=pg.cb(ops.clear_all), width="stretch", key="btn_clear",
              disabled=not describe(ops.settings))


def _more_scenarios(pg) -> None:
    ops, cb, ports = pg.ops, pg.cb, pg.ports
    w1, w2, w3 = st.columns([1, 1, 1.2])
    wave = w1.number_input("Wave (m)", 2.0, 15.0, 6.0, 0.5, key="storm_wave")
    rad = w2.number_input("Radius (nm)", 100.0, 1000.0, 300.0, 50.0, key="storm_radius")
    w3.button("Storm ahead", on_click=cb(ops.add_storm, wave, None, None, rad), width="stretch",
              help="Cores of 5 m waves or more are avoided when an alternative exists.")
    st.button("Clear storms", on_click=cb(ops.clear_storms), width="stretch",
              disabled=not ops.settings["storm_cells"])
    for label, key, hi in (("Red Sea security escalation", "red_sea_escalation", 1.0),
                           ("Taiwan Strait tension", "taiwan_tension", 1.0),
                           ("Atlantic hurricane (severity)", "atlantic_hurricane", 1.0)):
        st.slider(label, 0.0, hi, step=0.05, key=f"inc_{key}", on_change=pg.on_incident, args=(key,))
    st.selectbox("Hurricane track", list(HURRICANE_TRACKS), key="inc_hurricane_track",
                 format_func=lambda k: HURRICANE_TRACKS[k][0], on_change=pg.on_incident, args=("hurricane_track",))
    st.slider("Suez fee ×", 1.0, 10.0, step=0.5, key="inc_suez_toll_multiplier",
              on_change=pg.on_incident, args=("suez_toll_multiplier",))
    st.slider("Panama fee ×", 1.0, 5.0, step=0.5, key="inc_panama_toll_multiplier",
              on_change=pg.on_incident, args=("panama_toll_multiplier",))
    st.toggle("Panama Canal drought", key="inc_panama_drought", on_change=pg.on_incident, args=("panama_drought",))
    p1, p2 = st.columns(2)
    cport = p1.selectbox("Congested port", pg.port_ids, index=pg.port_ids.index(ops.cfg.destination),
                         key="cong_port", format_func=lambda p: ports[p])
    chrs = p2.number_input("Extra wait (h)", 0.0, 480.0, 48.0, 12.0, key="cong_hours")
    st.button("Apply port congestion", on_click=cb(ops.set_congestion, cport, chrs), width="stretch")
    preset = st.selectbox("Scenario set", list(PRESETS), key="preset_pick")
    st.button("Apply scenario set", on_click=cb(ops.apply_settings, PRESETS[preset], f"preset '{preset}'"),
              width="stretch")


# ==================================================================================================
# Optimize: objective presets and the economics behind the decision
# ==================================================================================================
def render_optimize(pg) -> None:
    st.radio("Preset", [*ECO.PRESETS, "custom"], key="cfg_preset", horizontal=True, on_change=pg.on_preset,
             format_func=lambda k: ECO.PRESETS[k].label if k in ECO.PRESETS else "Custom",
             help=" · ".join(f"{p.label}: {p.blurb}" for p in ECO.PRESETS.values())
             + " Moving a slider below switches to Custom.")
    st.slider("Value of time (x)", 1.0, 10.0, step=0.5, key="cfg_tvm", on_change=pg.on_econ,
              args=("time_value_mult",),
              help="A delayed hour counts this many times its cash cost in the speed and route decision.")
    st.slider("Carbon price ($/t CO2)", 0.0, 1000.0, step=10.0, key="cfg_carbon", on_change=pg.on_econ,
              args=("carbon_usd_per_t",),
              help="Decision weight only: it slows the ship and favours cleaner routes but is never "
                   "added to cash cost.")
    st.radio("Fuel price source", ["global", "origin_port"], key="cfg_fuelmode", horizontal=True,
             on_change=pg.on_econ, args=("fuel_price_mode",),
             format_func=lambda k: "Fuel-price setting" if k == "global" else "Departure port (sample)")
    with st.expander("Advanced", expanded=False):
        _advanced_objective(pg)
        _advanced_carbon_fuel(pg)
        _advanced_voyage(pg)


def _advanced_objective(pg) -> None:
    cfg = pg.ops.cfg
    st.markdown('<div class="section-name">Objective and avoidance</div>', unsafe_allow_html=True)
    st.radio("Objective", list(V.PREFERENCES), key="cfg_pref", horizontal=True,
             format_func=V.PREFERENCES.get, on_change=pg.on_cfg, args=("preference",),
             help=" · ".join(f"{V.PREFERENCES[k]}: {t}" for k, t in V.PREFERENCE_HELP.items()))
    if cfg.preference == "balanced":
        for col, (label, key) in zip(st.columns(3), (("Time", "cfg_w_time"), ("Cost", "cfg_w_cost"),
                                                     ("Risk", "cfg_w_risk")), strict=False):
            col.slider(label, 0.0, 3.0, step=0.25, key=key, on_change=pg.on_cfg, args=("weights",))
    st.multiselect("Avoid chokepoints", list(CHOKEPOINTS), key="cfg_avoid", format_func=CHOKEPOINTS.get,
                   on_change=pg.on_cfg, args=("avoid",), help="Treated as closed for this voyage.")


def _advanced_carbon_fuel(pg) -> None:
    cfg = pg.ops.cfg
    st.markdown('<div class="section-name">Carbon, ECA and fuel</div>', unsafe_allow_html=True)
    st.toggle("EU ETS (voyages calling at an EU port)", key="cfg_ets_on", on_change=pg.on_econ, args=("ets_enabled",))
    e1, e2 = st.columns(2)
    e1.number_input("Allowance ($/t)", 0.0, 400.0, step=5.0, key="cfg_eua", on_change=pg.on_econ,
                    args=("eua_usd_per_t",), disabled=not cfg.econ.ets_enabled)
    e2.number_input("Coverage (%)", 0.0, 100.0, step=10.0, key="cfg_etsc", on_change=pg.on_econ,
                    args=("ets_coverage_pct",), disabled=not cfg.econ.ets_enabled)
    st.number_input(f"Bunker price at {pg.ports[cfg.origin]} ($/t, override)", 100.0, 3000.0,
                    step=10.0, key="cfg_fuel_ovr", on_change=pg.on_econ, args=("fuel_overrides",),
                    disabled=cfg.econ.fuel_price_mode != "origin_port",
                    help="SAMPLE price table unless you override it here.")
    st.slider("Fuel price ($/t VLSFO)", 300.0, 1600.0, step=10.0, key="cfg_fuel", on_change=pg.on_cfg,
              args=("fuel_price",))
    lo, hi = HULL[VESSELS[cfg.vessel_key].vessel_type]["speed"]
    s1, s2 = st.columns([1, 1.4])
    s1.toggle("Fixed speed", key="cfg_fixed_speed", on_change=pg.on_cfg, args=("cruise_speed_kn",),
              help="Off: the optimiser picks the best of the vessel's speed options on every lane.")
    s2.slider("Cruising speed (kn)", float(lo), float(hi), step=0.5, key="cfg_speed", on_change=pg.on_cfg,
              args=("cruise_speed_kn",), disabled=not cfg.cruise_speed_kn)
    agent = cfg.agent()
    spd = agent.speed_options_kn[len(agent.speed_options_kn) // 2]
    tpd = float(predict_tonnes_per_day(pg.model, agent.vessel_type, spd, agent.draft_m,
                                       agent.cargo_weight_t, 15.0, 1.5)[0])
    st.caption(md(f"Model estimate at {spd:g} kn, calm sea: **{tpd:,.0f} t/day** "
                  f"(≈ ${tpd * cfg.fuel_price / 1e3:,.0f}k/day)."))


def _advanced_voyage(pg) -> None:
    cfg, started = pg.ops.cfg, pg.ops.started()
    st.markdown('<div class="section-name">Voyage and delays</div>', unsafe_allow_html=True)
    st.selectbox("Cargo owner", list(V.VESSEL_OWNERS[cfg.vessel_key]), key="cfg_owner",
                 format_func=lambda k: f"{CARGO_OWNERS[k].label} · {CARGO_OWNERS[k].cargo}",
                 on_change=pg.on_voyage, disabled=started)
    d1, d2 = st.columns(2)
    d1.date_input("Departure (UTC)", key="cfg_dep_date", on_change=pg.on_cfg, args=("departure",), disabled=started)
    d2.time_input("Time", key="cfg_dep_time", on_change=pg.on_cfg, args=("departure",), disabled=started, step=3600)
    q1, q2 = st.columns(2)
    q1.number_input("Berth wait (h)", 0.0, 240.0, step=6.0, key="cfg_port_delay",
                    on_change=pg.on_cfg, args=("port_delay_hours",))
    q2.number_input("Canal delay (h)", 0.0, 240.0, step=6.0, key="cfg_canal_delay",
                    on_change=pg.on_cfg, args=("canal_delay_hours",))


# ==================================================================================================
# Route comparison: options side by side, costs, risk, checkpoints, fuel burn
# ==================================================================================================
def render_compare(pg, change) -> None:
    ops, v = pg.ops, pg.ops.voyage
    _decision(pg, change)
    opts = v.candidates if v.candidates else ops.options()
    if opts:
        st.markdown('<div class="section-name">Route options'
                    + (" (last re-evaluation)" if v.candidates else " (from current position)")
                    + '</div>', unsafe_allow_html=True)
        show_df(pd.DataFrame(P.options_rows(opts)), hide_index=True)
        st.caption(f"Objective: {V.PREFERENCES[ops.cfg.preference]}. Model estimates for the remaining voyage.")
    st.markdown('<div class="section-name">Cost breakdown · every factor, with its data basis</div>',
                unsafe_allow_html=True)
    whole = V.summarise(V.build_context(ops.cfg, ops.settings, pg.model), v.legs)
    st.markdown(ui.cost_breakdown(ECO.breakdown(whole, ops.cfg.econ), whole.co2_t), unsafe_allow_html=True)
    _risk_panel(pg)
    _checkpoints_and_burn(pg)


def _decision(pg, change) -> None:
    ops, v = pg.ops, pg.ops.voyage
    if change:
        st.markdown(ui.recommendation(change, v.status, V.PREFERENCES[ops.cfg.preference]), unsafe_allow_html=True)
        st.markdown(ui.comparison_table(change["before"], change["after"],
                                        "Before (via Suez)" if ops.demo else "Before",
                                        "After closure" if ops.demo else "After"), unsafe_allow_html=True)
    elif v.status == "blocked":
        st.markdown(ui.decision(v.block_reason, True), unsafe_allow_html=True)
    elif v.decision:
        st.markdown(ui.decision(v.decision, rec=v.recommendation), unsafe_allow_html=True)


def _risk_panel(pg) -> None:
    v = pg.ops.voyage
    st.markdown('<div class="section-name">Risk score · what it means and what drives it</div>',
                unsafe_allow_html=True)
    _, ahead_legs, _ = V.split_at(v, pg.G0)
    rb_view = RISK.route_risk(ahead_legs if ahead_legs else v.legs)
    st.markdown(ui.risk_chip(rb_view.score) + " &nbsp;" + ("remaining route" if ahead_legs else "whole voyage"),
                unsafe_allow_html=True)
    st.markdown(ui.risk_panel(rb_view), unsafe_allow_html=True)


def _checkpoints_and_burn(pg) -> None:
    v = pg.ops.voyage
    st.markdown('<div class="section-name">Checkpoints ahead · sea state and source</div>', unsafe_allow_html=True)
    cp_rows = P.checkpoint_rows(pg.current_weather())
    if cp_rows:
        show_df(pd.DataFrame(cp_rows), hide_index=True)
    else:
        st.caption("No live checkpoint data right now (offline or beyond the 7-day forecast). The "
                   "simulator uses its synthetic sea-state model.")
    burn = P.fuel_burn_series(v.legs)
    if burn:
        st.markdown('<div class="section-name">Fuel burned over the voyage '
                    '<span class="est">MODEL ESTIMATE</span></div>', unsafe_allow_html=True)
        st.line_chart(pd.DataFrame(burn).set_index("Voyage day"), height=180)
        st.caption("Cumulative fuel burned by the whole planned voyage. Fuel on board is not modelled.")


# ==================================================================================================
# Alerts & intelligence. Returns the slot the Weather / News tabs are drawn into LAST (render_intel_panel), so slow
# or failing network calls never delay the rest of the page.
# ==================================================================================================
def _alert_list(pg) -> list[str]:
    ops, v, ss = pg.ops, pg.ops.voyage, pg.ss
    alerts = []
    if v.status == "blocked":
        alerts.append(ui.alert("Voyage blocked", v.block_reason, "critical"))
    if v.status == "arrived":
        alerts.append(ui.alert("Arrived", f"Voyage complete after {v.t / 24:.1f} days.", "info"))
    if any(leg.note for leg in v.legs if leg.kind == "sail" and leg.note):
        alerts.append(ui.alert("Forced through", "No route avoids every storm / high-premium lane.", ""))
    if v.reroutes:
        last = next((e for e in reversed(v.timeline) if e["kind"] == "reroute"), None)
        if last:
            alerts.append(ui.alert(f"Rerouted ({v.reroutes}×)", last["title"], "info"))
    for d in describe(ops.settings):
        alerts.append(ui.alert("Simulated scenario", d, ""))
    wx = pg.current_weather()
    if wx and wx.get("worst") and wx["worst"]["level"] >= 2:
        wr = wx["worst"]
        alerts.append(ui.alert(f"Forecast hazard ahead: {wr['severity']} (Open-Meteo)",
                               f"{wr['values'].get('wave_height')} m waves {wr['label'].lower()} at simulated "
                               f"ETA {wr['eta']:%b %d %H:%M} UTC. {wr['action']}",
                               "critical" if wr["level"] == 3 else ""))
    news = ss.get("news")
    if news and news.get("articles") and not news.get("demo"):
        for adv in intel.advisories(intel.rank_news(news["articles"], pg.route_chokepoints())):
            alerts.append(ui.alert(f"News advisory (unverified): {adv['label']}",
                                   f"\"{adv['headline']}\" ({adv['source']}). No corridor was closed.", ""))
    return alerts


def render_alerts(pg):
    v = pg.ops.voyage
    st.markdown('<div style="max-height:150px;overflow-y:auto">'
                + ("".join(_alert_list(pg)) or ui.alert("All clear", "No warnings for the current voyage.", "info"))
                + "</div>", unsafe_allow_html=True)
    intel_slot = st.container(height=300, border=False)
    st.markdown('<div class="section-name">Risk legend</div>', unsafe_allow_html=True)
    st.markdown(" ".join(ui.risk_chip(s) for s in (10, 30, 50, 70, 90)) + '<div class="intel-meta" '
                'style="margin-top:6px">Hover a chip for what the score means. News headlines are '
                'advisory and never change it.</div>', unsafe_allow_html=True)
    with st.expander("Event timeline", expanded=False):
        with st.container(height=260, border=False):
            st.markdown(ui.timeline(v.timeline), unsafe_allow_html=True)
    return intel_slot


def render_intel_panel(pg, slot) -> None:
    """Weather / News sub-tabs; refreshes itself every 10 minutes and never runs on animation frames."""
    @st.fragment(run_every=600)
    def intel_panel():
        t_w, t_n = st.tabs(["Weather", "News & Security"])
        with t_w:
            render_weather(pg.ss, pg.ops, cached_marine, pg.cb, pg.flash)
        with t_n:
            render_news(pg.ss, pg.ops, cached_news, pg.cb, pg.route_chokepoints)

    with slot:
        intel_panel()


# ==================================================================================================
# Data sources: every input behind the plan, with its basis
# ==================================================================================================
def render_sources(pg) -> None:
    ss = pg.ss
    src_rows = SRC.plan_sources(pg.ops, ss.get("weather"), ss.get("news"), pg.llm_label)
    df_src = pd.DataFrame([{"Input": r["input"], "Basis": r["basis"], "Detail": r["detail"],
                            "Date": r["date"], "Link": r["url"] or None} for r in src_rows])
    try:
        st.dataframe(df_src, hide_index=True, width="stretch",
                     column_config={"Link": st.column_config.LinkColumn("Link")})
    except TypeError:
        show_df(df_src, hide_index=True)
    st.caption("Basis: live = fetched now · published = a public reference value · estimated = an "
               "assumption you can edit · sample = illustrative · simulated = produced by the model · "
               "user setting = what you chose.")
    with st.expander("Methodology and limitations", expanded=False):
        st.markdown(ui.METHODOLOGY, unsafe_allow_html=True)
