"""
panels.py - the pure helpers behind the right-hand tabs (no Streamlit, so they are unit-testable).

app.py does the drawing; the decisions about what to show live here: which incidents are switched on, the rows of
the route-comparison table, the fuel-burn series, the checkpoint table, and the one-line status.
"""
from __future__ import annotations

import html

import ui

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


def status_line(recommendation: str, corridor: str, risk_score: float) -> str:
    """One line for the top of the Operate tab: the decision, the corridor and the risk chip."""
    label, tone = STATUS_LABELS.get(recommendation, ("Plan assessed", "stay"))
    return (f'<div class="statusline {tone}"><b>{html.escape(label)}</b> · {html.escape(corridor)} · '
            f'{ui.risk_chip(risk_score)}</div>')
