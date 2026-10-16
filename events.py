"""
events.py - global shocks that can be layered onto the baseline maritime graph.

``apply_events(G, settings)`` returns a NEW graph (the baseline is never mutated) in which lane
attributes have been updated: risks re-scored from the per-lane samples, tolls scaled, lanes blocked,
drafts restricted. The routing engine then simply re-runs on the stressed graph.

settings is a plain dict ``{event_key: value}``; bool events are toggles, float events are sliders.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_engine import GEO_ZONE, RED_SEA_ZONES, WEATHER_BLOCK_THRESHOLD, haversine_nm


@dataclass(frozen=True)
class EventSpec:
    key: str
    label: str
    kind: str            # "toggle" | "slider"
    default: float | bool
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.05
    help: str = ""


EVENT_CATALOG: list[EventSpec] = [
    EventSpec("hormuz_blockade", "Blockade Strait of Hormuz", "toggle", False,
              help="Closes the strait to all shipping; Persian Gulf risk jumps to near-maximum."),
    EventSpec("red_sea_escalation", "Red Sea escalation", "slider", 0.0, 0.0, 1.0, 0.05,
              help="Houthi-style attacks: raises war-risk across the Red Sea, Bab-el-Mandeb and Gulf of Aden."),
    EventSpec("suez_toll_multiplier", "Suez Canal toll increase", "slider", 1.0, 1.0, 10.0, 0.5,
              help="Multiplier on the Suez transit fee (x1 = normal)."),
    EventSpec("suez_blockage", "Suez Canal blocked", "toggle", False,
              help="A grounded mega-ship closes the canal (Ever Given style)."),
    EventSpec("atlantic_hurricane", "Atlantic hurricane", "slider", 0.0, 0.0, 1.0, 0.05,
              help="Severity of a hurricane; lanes through its core become impassable at high severity."),
    EventSpec("panama_drought", "Panama Canal drought", "toggle", False,
              help="Draft limit drops to 13.5 m and transit fees rise 2.5x."),
    EventSpec("taiwan_tension", "Taiwan Strait tension", "slider", 0.0, 0.0, 1.0, 0.05,
              help="Raises war-risk in the Taiwan Strait and South China Sea."),
    EventSpec("bab_el_mandeb_closure", "Bab-el-Mandeb / Red Sea closed", "toggle", False,
              help="Closes the southern entrance of the Red Sea to all shipping."),
    EventSpec("panama_closure", "Panama Canal closed", "toggle", False,
              help="Closes the Panama Canal to all transits."),
    EventSpec("panama_toll_multiplier", "Panama Canal toll increase", "slider", 1.0, 1.0, 5.0, 0.5,
              help="Multiplier on the Panama transit fee (x1 = normal)."),
]
# Settings that are not sliders/toggles: storm cells placed by the user or the copilot, and port congestion.
#   storm_cells:    [{"lat", "lon", "radius_nm", "wave_m", "label", "source"}]
#   port_congestion: {port_id: extra berth-wait hours}
STORM_AVOID_WAVE_M = 5.0          # storm cores at or above this significant wave height are avoided if possible
EVENT_BY_KEY = {e.key: e for e in EVENT_CATALOG}

# Hurricane tracks: (label, centre lat, centre lon, radius nm)
HURRICANE_TRACKS: dict[str, tuple] = {
    "cape_verde": ("Cape Verde (off West Africa)", 16.0, -28.0, 800.0),
    "main_development": ("Main development region", 18.0, -50.0, 900.0),
    "caribbean": ("Caribbean / Gulf of Mexico", 20.0, -75.0, 700.0),
}

PRESETS: dict[str, dict] = {
    "Calm seas": {},
    "Red Sea crisis": {"red_sea_escalation": 0.8},
    "Hormuz blockade": {"hormuz_blockade": True},
    "Perfect storm": {"hormuz_blockade": True, "red_sea_escalation": 0.8, "suez_toll_multiplier": 5.0,
                      "atlantic_hurricane": 0.9, "taiwan_tension": 0.4},
}


def default_settings() -> dict:
    return {e.key: e.default for e in EVENT_CATALOG} | {"hurricane_track": "cape_verde", "storm_cells": [],
                                                         "port_congestion": {}}


def normalise(settings: dict | None) -> dict:
    """Fill in defaults so callers can pass a partial dict."""
    out = default_settings()
    out.update(settings or {})
    return out


def wave_to_severity(wave_m: float) -> float:
    """Inverse of routing_engine.sea_state (wave = 0.8 + 7.5 * w^2): significant wave height -> 0..1 severity."""
    return float(np.clip(np.sqrt(max(float(wave_m) - 0.8, 0.0) / 7.5), 0.0, 0.97))


def describe(settings: dict) -> list[str]:
    """Human-readable list of the events that are actually switched on."""
    s = normalise(settings)
    out = []
    if s["hormuz_blockade"]:
        out.append("Strait of Hormuz blockaded")
    if s["red_sea_escalation"] > 0:
        out.append(f"Red Sea escalation {s['red_sea_escalation']:.0%}")
    if s["suez_toll_multiplier"] > 1:
        out.append(f"Suez toll x{s['suez_toll_multiplier']:g}")
    if s["suez_blockage"]:
        out.append("Suez Canal blocked")
    if s["atlantic_hurricane"] > 0:
        out.append(f"Atlantic hurricane {s['atlantic_hurricane']:.0%} ({HURRICANE_TRACKS[s['hurricane_track']][0]})")
    if s["panama_drought"]:
        out.append("Panama drought")
    if s["taiwan_tension"] > 0:
        out.append(f"Taiwan tension {s['taiwan_tension']:.0%}")
    if s["bab_el_mandeb_closure"]:
        out.append("Bab-el-Mandeb closed")
    if s["panama_closure"]:
        out.append("Panama Canal closed")
    if s["panama_toll_multiplier"] > 1:
        out.append(f"Panama toll x{s['panama_toll_multiplier']:g}")
    for c in s["storm_cells"]:
        out.append(f"Storm {c.get('label', '')} ({c['wave_m']:g} m waves)".replace("  ", " "))
    for port, hours in s["port_congestion"].items():
        if hours > 0:
            out.append(f"Port congestion {port} +{hours:g} h")
    return out


def _refresh(d: dict) -> None:
    d["geopolitical_risk"] = float(d["geo_samples"].mean())
    d["geopolitical_peak"] = float(d["geo_samples"].max())
    d["weather_risk"] = float(d["weather_samples"].mean())
    d["weather_peak"] = float(d["weather_samples"].max())


def _raise_in_zones(d: dict, zone_keys, level: float, strength: float) -> bool:
    """Blend geo samples inside the given zones toward ``level`` by ``strength`` (0..1)."""
    mask = np.zeros(len(d["sample_lat"]), dtype=bool)
    for k in zone_keys:
        mask |= GEO_ZONE[k].contains(d["sample_lat"], d["sample_lon"])
    if not mask.any():
        return False
    g = d["geo_samples"].copy()
    g[mask] = np.maximum(g[mask], g[mask] + strength * (level - g[mask]))
    d["geo_samples"] = g
    return True


def _init_edge(d: dict) -> None:
    d["geo_samples"] = d["geo_samples_base"].copy()
    d["weather_samples"] = d["weather_samples_base"].copy()
    d["active_events"] = []


def _apply_to_edge(d: dict, s: dict) -> None:
    """Apply every active event in ``s`` (a normalised settings dict) to ONE lane's attribute dict, in place."""
    track = HURRICANE_TRACKS[s["hurricane_track"]]
    ev = d["active_events"]
    if s["hormuz_blockade"]:
        if d["chokepoint"] == "hormuz":
            d["blocked"] = True
            ev.append("Hormuz blockade")
        # Inside the Gulf risk is near-maximum; the strait box (which also covers the Fujairah approaches
        # outside it) is tense but passable for ships that are not trying to cross.
        hit = _raise_in_zones(d, ("hormuz",), 0.50, 1.0)
        hit |= _raise_in_zones(d, ("persian_gulf",), 0.90, 1.0)
        if hit:
            ev.append("Hormuz blockade")
    if s["red_sea_escalation"] > 0 and _raise_in_zones(d, RED_SEA_ZONES, 0.85, s["red_sea_escalation"]):
        ev.append("Red Sea escalation")
    if s["taiwan_tension"] > 0 and _raise_in_zones(d, ("taiwan_strait", "south_china_sea"), 0.75,
                                                   s["taiwan_tension"]):
        ev.append("Taiwan tension")
    if d["chokepoint"] == "suez":
        if s["suez_toll_multiplier"] > 1:
            d["toll_multiplier"] *= float(s["suez_toll_multiplier"])
            ev.append(f"Suez toll x{s['suez_toll_multiplier']:g}")
        if s["suez_blockage"]:
            d["blocked"] = True
            ev.append("Suez blocked")
    if d["chokepoint"] == "panama" and s["panama_drought"]:
        d["toll_multiplier"] *= 2.5
        d["max_draft_m"] = min(d["max_draft_m"] or 99.0, 13.5)
        ev.append("Panama drought")
    if d["chokepoint"] == "panama":
        if s["panama_toll_multiplier"] > 1:
            d["toll_multiplier"] *= float(s["panama_toll_multiplier"])
            ev.append(f"Panama toll x{s['panama_toll_multiplier']:g}")
        if s["panama_closure"]:
            d["blocked"] = True
            ev.append("Panama closed")
    if d["chokepoint"] == "bab_el_mandeb" and s["bab_el_mandeb_closure"]:
        d["blocked"] = True
        ev.append("Bab-el-Mandeb closed")
    if d["mode"] == "sea":
        for cell in s["storm_cells"]:
            _apply_storm(d, cell, ev)
    if s["atlantic_hurricane"] > 0 and d["mode"] == "sea":
        _, clat, clon, radius = track
        dist = np.array([haversine_nm(la, lo, clat, clon) for la, lo in zip(d["sample_lat"], d["sample_lon"])])
        inside = dist < radius
        if inside.any():
            storm = s["atlantic_hurricane"] * 0.97 * (1.0 - 0.45 * (dist / radius) ** 2)
            w = d["weather_samples"].copy()
            w[inside] = np.maximum(w[inside], storm[inside])
            d["weather_samples"] = w
            ev.append("Atlantic hurricane")
    d["active_events"] = sorted(set(ev))
    _refresh(d)
    d["events_applied"] = True


def _apply_storm(d: dict, cell: dict, ev: list) -> None:
    """A user/copilot-placed storm: raise sea state inside its radius. Cores with waves >= STORM_AVOID_WAVE_M are
    pushed over the storm-closure threshold, so routing avoids them unless no other route exists."""
    dist = np.array([haversine_nm(la, lo, cell["lat"], cell["lon"]) for la, lo in zip(d["sample_lat"], d["sample_lon"])])
    radius = float(cell.get("radius_nm", 300.0))
    inside = dist < radius
    if not inside.any():
        return
    sev = wave_to_severity(cell["wave_m"]) * (1.0 - 0.35 * (dist / radius) ** 2)
    if cell["wave_m"] >= STORM_AVOID_WAVE_M:
        sev = np.where(dist < 0.75 * radius, np.maximum(sev, WEATHER_BLOCK_THRESHOLD + 0.02), sev)
    w = d["weather_samples"].copy()
    w[inside] = np.maximum(w[inside], sev[inside])
    d["weather_samples"] = w
    ev.append(f"Storm {cell.get('label', '')}".strip())


def stress_edge(edge_data: dict, settings: dict | None) -> dict:
    """Return a stressed copy of a single lane's attributes (used by ``routing_engine.calculate_edge_cost``)."""
    d = dict(edge_data)
    d["toll_multiplier"] = 1.0
    d["blocked"] = False
    _init_edge(d)
    _apply_to_edge(d, normalise(settings))
    return d


def apply_events(G, settings: dict | None):
    """Return a stressed copy of ``G`` with the requested events applied (the baseline is untouched)."""
    s = normalise(settings)
    H = G.copy()
    for _, _, d in H.edges(data=True):
        _init_edge(d)
        _apply_to_edge(d, s)
    return H


if __name__ == "__main__":
    from data_engine import build_maritime_graph
    base = build_maritime_graph()
    for name, preset in PRESETS.items():
        H = apply_events(base, preset)
        hit = sum(1 for *_, d in H.edges(data=True) if d["active_events"])
        blocked = sum(1 for *_, d in H.edges(data=True) if d["blocked"])
        print(f"{name:<16} lanes affected: {hit:>3}   blocked: {blocked}")
