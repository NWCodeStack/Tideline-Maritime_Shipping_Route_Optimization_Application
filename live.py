"""
live.py - the fragments that follow the simulated clock.

The server stays the clock (``ops.tick()``), but the ship itself is animated in the browser from a small contract
(``anim.build_contract``), so the map is NOT re-sent every second. ``kpi_fragment`` and ``seek_fragment`` re-run every
``TICK_S`` while playing; ``map_fragment`` only runs when something changes the route or the clock (a full rerun).
When a tick changes the clock itself (stall, arrival, demo trigger) the whole page is re-run so the map gets a new
contract.
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

import anim
import panels as P
import ui
import voyage as V
from geo_assets import project as svg_project
from map_view import build_ops_map_html

try:                                    # interactive Leaflet map; the offline SVG map is used without it
    from streamlit_folium import st_folium
    from folium_map import build_ops_base_map, build_ops_layer
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    st_folium = build_ops_base_map = build_ops_layer = None

TICK_S = 1.0                     # clock / telemetry refresh interval (real seconds)


def _tick_and_sync(ops) -> None:
    """Advance the clock; re-run the page when the browser animation must get a new contract."""
    was_playing, epoch = ops.playing, ops.epoch
    ops.tick()
    if (was_playing and not ops.playing) or ops.epoch != epoch:
        st.rerun()


SEEK_ECHO_WINDOW = 5             # how many recently written slider values can come back as a stale echo


def remember_seek(written: tuple, value: float) -> tuple:
    return (written + (value,))[-SEEK_ECHO_WINDOW:]


def seek_is_echo(value: float, written: tuple, playing: bool) -> bool:
    """While the clock runs, the browser can echo back a slider value the server wrote on an earlier tick; that is
    not a drag. Stale echoes only happen while ticks are writing the slider, so a paused slider is never an echo."""
    return playing and any(abs(value - w) < 0.05 for w in written)


def warning_signature(warnings: list[dict]) -> tuple:
    """What the map's port rings show (port and kind), order-free, so a change can be detected cheaply."""
    return tuple(sorted((w["port"], w["kind"]) for w in warnings))


def kpi_fragment(ops):
    @st.fragment(run_every=TICK_S if ops.playing else None)
    def kpis():
        _tick_and_sync(ops)
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
                 ("Risk ahead", f"{risk:.0f}/100", f"{V.risk_label(risk)} · hover for what this means",
                  ui.band(risk)[1], True, ui.RISK_CAPTION)]
        st.markdown(ui.metric_cards(cards), unsafe_allow_html=True)
        state = ("blocked" if v.status == "blocked" else "playing" if ops.playing else
                 "idle" if v.status == "planned" and v.t == 0 else "paused")
        clock = V.calendar(v, v.t)
        st.markdown(ui.live_strip(state, [
            ("Sim clock", f"{clock:%Y-%m-%d %H:%M} UTC"), ("Day", f"{v.t / 24:.1f}"),
            ("SIMULATED position", f"{s.lat:.2f}, {s.lon:.2f}"), ("Course", f"{s.course_deg:03.0f}°"),
            ("SOG", f"{s.sog_kn:.1f} kn"), ("Status", s.activity)]),
            unsafe_allow_html=True)
    return kpis


def _port_warnings(pg, ship_pos):
    ops = pg.ops
    return P.port_warnings(V.build_context(ops.cfg, ops.settings, pg.model), ship_pos, ops.settings, ops.G)


def _refresh_if_port_rings_changed(pg) -> None:
    """The map is no longer re-sent every tick, so the red/amber port rings (which depend on where the ship is) are
    refreshed by re-running the page only when they would actually change."""
    ss, ops = pg.ss, pg.ops
    if not ops.playing or ss.get("map_mode") == "Offline map" or "_warn_sig" not in ss:
        return
    sig = warning_signature(_port_warnings(pg, V.ship_state(ops.voyage, ops.G).pos))
    if sig != ss["_warn_sig"]:
        st.rerun()


def seek_fragment(pg):
    """The 'Voyage time' slider. It follows the clock while playing; a seek re-runs the page so the map snaps."""
    ss, ops = pg.ss, pg.ops

    def on_seek():
        if seek_is_echo(float(ss["seek_h"]), ss.get("_seek_written", ()), ops.playing):
            return
        pg.on_seek()
        ss["_seeked"] = True

    @st.fragment(run_every=TICK_S if ops.playing else None)
    def seek():
        _tick_and_sync(ops)
        if ss.pop("_seeked", False):
            st.rerun()
        v = ops.voyage
        total = max(v.total_hours, 0.1)
        ss["seek_h"] = float(min(round(v.t, 1), total))
        ss["_seek_written"] = remember_seek(ss.get("_seek_written", ()), ss["seek_h"])
        _refresh_if_port_rings_changed(pg)
        st.slider("Voyage time (hours since departure): drag to jump, or click a waypoint", 0.0, float(total),
                  step=1.0, key="seek_h", on_change=on_seek, disabled=v.status == "blocked" or not v.legs)
        st.caption("Mint = completed track · solid = remaining plan · grey dashed = previous committed route · amber "
                   "dotted = alternatives considered · red dashed = blocked plan · circles = Open-Meteo forecast at the "
                   "simulated ETA · purple = storm scenario · tinted boxes = security zones by risk band (hover for the "
                   "numbers, hide them in the layer menu) · red ring = port with no sea passage from the ship · amber "
                   "ring = port congestion scenario.")
    return seek


def _route_points(v, G):
    done, ahead, _ = V.split_at(v, G)
    done_pts, ahead_pts = V.legs_polyline(G, done), V.legs_polyline(G, ahead)
    if done_pts and ahead_pts and done_pts[-1] != ahead_pts[0]:
        ahead_pts = [done_pts[-1]] + ahead_pts
    return done_pts, ahead_pts


def _interactive_map(pg, v, G, s, color, contract, wps, alts) -> None:
    ss, ops, model = pg.ss, pg.ops, pg.model
    base_key = json.dumps(ops.settings, sort_keys=True, default=str)
    if ss.get("_base_key") != base_key:             # static layers rebuilt only when scenarios change
        ctx = V.build_context(ops.cfg, ops.settings, model)
        ss["_base_map"] = build_ops_base_map(ctx.G, ops.settings, hazards=P.hazard_zones(ctx.G, V.base_graph()))
        ss["_base_key"] = base_key
    wx = pg.current_weather() or {}
    warns = _port_warnings(pg, s.pos)
    ss["_warn_sig"] = warning_signature(warns)
    layer = build_ops_layer(G, v, [], [], wps, s, alts, wx.get("points", []), color, port_warnings=warns,
                            contract=contract)
    base = ss["_base_map"]
    out = st_folium(base, feature_group_to_add=layer, key="ops_map", height=520,
                    use_container_width=True, returned_objects=["last_object_clicked"])
    # st_folium attaches the dynamic layer to the map object; detach it so the cached base map's script stays
    # byte-identical: only the voyage layer is pushed (no reload, zoom/pan preserved).
    for k in [k for k in base._children if k.startswith("feature_group_")]:
        del base._children[k]
    click = (out or {}).get("last_object_clicked")
    if click and click != ss.get("_last_click"):
        ss["_last_click"] = click
        near = min(wps, key=lambda w: (w["lat"] - click["lat"]) ** 2 + (w["lon"] - click["lng"]) ** 2, default=None)
        if near and (near["lat"] - click["lat"]) ** 2 + (near["lon"] - click["lng"]) ** 2 < 0.5:
            ops.seek(near["t"])
            st.rerun()


def map_fragment(pg):
    ops, model = pg.ops, pg.model

    @st.fragment
    def live_map():
        ops.tick()
        v, G = ops.voyage, ops.G
        s = V.ship_state(v, G)
        wps = V.waypoints(v, G)
        alts = [(c.name, V.legs_polyline(G, c.summary.legs)) for c in v.candidates
                if not c.chosen and c.name != "Hold course"]
        color = {"Container": "#58aaff", "Tanker": "#f5bf75"}.get(V.VESSELS[ops.cfg.vessel_key].vessel_type)
        perf = bool(st.query_params.get("perf"))
        track = anim.build_track(v, G)
        modes = ["Interactive map", "Offline map"] if HAS_FOLIUM else ["Offline map"]
        h1, h2 = st.columns([3, 1.3])
        h1.markdown('<div class="panel-title">Live route map <span class="tag-sim">SIMULATED POSITION · NOT AIS</span>'
                    '</div>', unsafe_allow_html=True)
        mode = h2.selectbox("Map", modes, key="map_mode", label_visibility="collapsed",
                            help="Offline map: no tiles or folium needed (use it if map tiles are blocked).")
        if mode == "Interactive map":
            _interactive_map(pg, v, G, s, color, anim.build_contract(ops, G, perf=perf, track=track), wps, alts)
        else:
            ctx = V.build_context(ops.cfg, ops.settings, model)
            contract = anim.build_contract(ops, G, project=svg_project, perf=perf, track=track)
            components.html(build_ops_map_html(
                ctx.G, ops.settings, [], V.legs_polyline(G, v.legs), v.superseded, alts, v.blocked_remaining, s,
                v.status, color, height=500, contract=contract), height=508)
    return live_map
