"""
panels.py - the pure helpers behind the right-hand tabs (no Streamlit, so they are unit-testable).

app.py does the drawing; the decisions about what to show live here: which incidents are switched on, the rows of
the route-comparison table, the fuel-burn series, the checkpoint table, and the one-line status.
"""
from __future__ import annotations

import html

import numpy as np

import ui
import voyage as V
from data_engine import GEOPOLITICAL_ZONES

# The four headline incidents shown in the compact picker (the other scenarios live under "More scenarios").
INCIDENTS = {
    "suez_blockage": "Suez Canal closed",
    "bab_el_mandeb_closure": "Bab-el-Mandeb closed",
    "hormuz_blockade": "Hormuz closed",
    "panama_closure": "Panama Canal closed",
}

STATUS_LABELS = {"change_course": ("CHANGE COURSE NOW", "change"), "stay": ("STAY ON COURSE", "stay"),
                 "blocked": ("VOYAGE BLOCKED", "blocked")}


def selected_incidents(settings: dict) -> list[str]:
    """The picker's value for the current scenario settings."""
    return [k for k in INCIDENTS if settings.get(k)]


def incident_changes(selected, settings: dict) -> list[tuple[str, bool]]:
    """(incident, on/off) for every incident whose state differs from the picker. Unknown keys are ignored."""
    chosen = set(selected)
    return [(k, k in chosen) for k in INCIDENTS if bool(settings.get(k)) != (k in chosen)]


def options_rows(candidates) -> list[dict]:
    """One row per route option, side by side: time, distance, fuel, CO2, cash cost, risk score with its band."""
    rows = []
    for c in candidates:
        s = c.summary
        rows.append({"Option": c.name, "Selected": "✓" if c.chosen else "", "Corridor": s.corridor,
                     "ETA days": round(s.hours / 24, 1), "Distance nm": round(s.nm), "Fuel t": round(s.fuel_t),
                     "CO2 t": round(s.co2_t), "All-in $M": round(s.total_usd / 1e6, 2),
                     "Risk score": round(s.risk_index), "Risk band": ui.band(s.risk_index)[0],
                     "Forced": "yes" if s.forced else ""})
    return rows


def fuel_burn_series(legs) -> list[dict]:
    """Cumulative fuel burned at the end of each leg (Tideline does not model fuel on board)."""
    out, hours, burned = [], 0.0, 0.0
    for l in legs:
        hours += l.hours
        burned += l.fuel_t
        out.append({"Voyage day": round(hours / 24, 2), "Fuel burned (t)": round(burned, 1)})
    return out


def checkpoint_rows(weather: dict | None) -> list[dict]:
    """The sampled checkpoints with sea state and where each value came from."""
    if not weather or not weather.get("points"):
        return []
    rows = []
    for p in weather["points"]:
        v = p.get("values") or {}
        live = p.get("status") == "forecast"
        rows.append({"Checkpoint": p["label"], "ETA (UTC)": p["eta"].strftime("%b %d %H:%M"),
                     "Waves (m)": v.get("wave_height") if live else None,
                     "Wind (kn)": v.get("wind_speed_10m") if live else None,
                     "Sea state": p.get("severity", "n/a"),
                     "Source": "live forecast (Open-Meteo)" if live else p.get("status", "n/a")})
    return rows


def hazard_zones(G, base) -> list[dict]:
    """Every security zone with its CURRENT level, split into the base level and what scenarios added, and the band
    that level falls in (score = 100 x level, the same bands as the route risk score). ``G`` is the stressed graph,
    ``base`` the calm one; the level is the highest lane sample inside the zone, as in the risk breakdown."""
    out = []
    for z in GEOPOLITICAL_ZONES:
        cur = bas = None
        for u, v, d in G.edges(data=True):
            m = z.contains(d["sample_lat"], d["sample_lon"])
            if m.any():
                c = float(np.max(d["geo_samples"][m]))
                b = float(np.max(base.edges[u, v]["geo_samples_base"][m]))
                cur = c if cur is None else max(cur, c)
                bas = b if bas is None else max(bas, b)
        if cur is None:                                   # no lane crosses this zone: report its baseline
            cur = bas = z.baseline
        score = round(100 * cur)
        label, tone = ui.band(score)
        out.append({"key": z.key, "label": z.label, "bounds": (z.lat_min, z.lat_max, z.lon_min, z.lon_max),
                    "security": cur, "base": bas, "scenario": max(cur - bas, 0.0), "score": score,
                    "band": label, "tone": tone})
    return out


def hazard_tooltip(z: dict) -> str:
    return (f"{z['label']}: {z['band']} · security {z['security']:.2f} (base {z['base']:.2f} + scenario "
            f"{z['scenario']:.2f}). SIMULATED levels, not a live security assessment.")


def port_warnings(ctx, pos, settings: dict, G) -> list[dict]:
    """Ports to flag on the map: no sea passage from the ship's position (closures), or a congestion scenario.
    There is no 'too shallow for this ship' warning: Tideline has no port-draft data and none is invented."""
    reach = V.reachable_ports(ctx, pos)
    out = []
    for n, d in G.nodes(data=True):
        if d["kind"] != "port":
            continue
        hours = float(settings.get("port_congestion", {}).get(n, 0.0))
        if n not in reach:
            out.append({"port": n, "name": d["name"], "lat": d["lat"], "lon": d["lon"], "kind": "unreachable",
                        "detail": f"{d['name']}: no sea passage from the ship's position (SIMULATED closure)."})
        elif hours > 0:
            out.append({"port": n, "name": d["name"], "lat": d["lat"], "lon": d["lon"], "kind": "congested",
                        "detail": f"{d['name']}: congestion +{hours:g} h berth wait (SIMULATED scenario)."})
    return out


def status_line(recommendation: str, corridor: str, risk_score: float) -> str:
    """One line for the top of the Operate tab: the decision, the corridor and the risk chip."""
    label, tone = STATUS_LABELS.get(recommendation, ("Plan assessed", "stay"))
    return (f'<div class="statusline {tone}"><b>{html.escape(label)}</b> · {html.escape(corridor)} · '
            f'{ui.risk_chip(risk_score)}</div>')
