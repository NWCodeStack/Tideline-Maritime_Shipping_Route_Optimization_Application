"""
folium_map.py - interactive Leaflet map (folium) of the stressed network and the chosen routes.

``build_folium_map`` returns a ``folium.Map``; app.py displays it with ``streamlit_folium.st_folium``.
Route lines are colour-coded by vessel: Blue for container ships, Amber for tankers.
"""
from __future__ import annotations

import math

import folium

from data_engine import GEO_ZONE, RED_SEA_ZONES
from events import HURRICANE_TRACKS, normalise

BLUE, AMBER, RED, STORM = "#2f80ed", "#f5a623", "#e5534b", "#a78bfa"
VESSEL_COLORS = {"Container": BLUE, "Tanker": AMBER}
# Keyless dark basemap (CartoDB tiles now require an API key and render an "API KEY REQUIRED" watermark).
DARK_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
DARK_ATTR = "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"


def unwrap(coords):
    """[(lat, lon)] -> same points with longitudes made continuous across the antimeridian (e.g. 175 -> 185)."""
    out, offset, prev = [], 0.0, None
    for lat, lon in coords:
        if prev is not None:
            if lon - prev > 180:
                offset -= 360.0
            elif prev - lon > 180:
                offset += 360.0
        out.append((lat, lon + offset))
        prev = lon
    return out


def _zone(m, key, color, label):
    z = GEO_ZONE[key]
    folium.Rectangle([[z.lat_min, z.lon_min], [z.lat_max, z.lon_max]], color=color, weight=1.5, dash_array="6 5",
                     fill=True, fill_color=color, fill_opacity=0.15, tooltip=label).add_to(m)


def build_folium_map(G, routes: list[dict], settings: dict | None = None) -> folium.Map:
    """``routes`` items: {label, vessel_type, result (RouteResult), hold (RouteResult | None)}."""
    s = normalise(settings)
    m = folium.Map(location=[22, 40], zoom_start=2, min_zoom=2, tiles=DARK_TILES, attr=DARK_ATTR, max_zoom=16,
                   world_copy_jump=True, control_scale=True)

    lanes = folium.FeatureGroup(name="Shipping lanes", show=True)
    for u, v, d in G.edges(data=True):
        if d["blocked"]:
            color, dash, w = RED, "6 6", 2.5
        elif d["mode"] == "overland":
            color, dash, w = "#8ba5b8", "2 6", 2
        elif d["weather_peak"] >= 0.6:
            color, dash, w = STORM, None, 1.5
        elif d["geopolitical_peak"] >= 0.3:
            color, dash, w = AMBER, None, 1.5
        else:
            color, dash, w = "#4a6378", None, 1
        pts = unwrap(list(zip(d["sample_lat"].tolist(), d["sample_lon"].tolist())))
        tip = (f"{G.nodes[u]['name']} to {G.nodes[v]['name']}<br>geopolitical risk {d['geopolitical_risk']:.2f}, "
               f"weather risk {d['weather_risk']:.2f}" + ("<br><b>CLOSED</b>" if d["blocked"] else ""))
        folium.PolyLine(pts, color=color, weight=w, opacity=0.7, dash_array=dash, tooltip=tip).add_to(lanes)
    lanes.add_to(m)

    zones = folium.FeatureGroup(name="Event zones", show=True)
    if s["hormuz_blockade"]:
        _zone(zones, "hormuz", RED, "Strait of Hormuz: blockade")
        _zone(zones, "persian_gulf", RED, "Persian Gulf: war-risk")
    if s["red_sea_escalation"] > 0:
        for k in RED_SEA_ZONES:
            _zone(zones, k, AMBER, GEO_ZONE[k].label + ": escalation")
    if s["taiwan_tension"] > 0:
        _zone(zones, "taiwan_strait", AMBER, "Taiwan Strait tension")
        _zone(zones, "south_china_sea", AMBER, "South China Sea tension")
    if s["atlantic_hurricane"] > 0:
        label, lat, lon, radius_nm = HURRICANE_TRACKS[s["hurricane_track"]]
        folium.Circle([lat, lon], radius=radius_nm * 1852, color=STORM, weight=1.5, dash_array="6 5", fill=True,
                      fill_color=STORM, fill_opacity=0.15, tooltip=f"Hurricane: {label}").add_to(zones)
    zones.add_to(m)

    ships = folium.FeatureGroup(name="Optimal routes", show=True)
    for r in routes:
        res, hold, color = r["result"], r.get("hold"), VESSEL_COLORS[r["vessel_type"]]
        if hold is not None and hold.feasible and res.feasible and hold.path != res.path:
            folium.PolyLine(unwrap(hold.coords), color=color, weight=2, opacity=0.6, dash_array="2 8",
                            tooltip=f"{r['label']}: previous route").add_to(ships)
        if res.feasible:
            folium.PolyLine(unwrap(res.coords), color=color, weight=5, opacity=0.95,
                            tooltip=f"{r['label']}: {res.distance_nm:,.0f} nm, {res.days:.1f} days, "
                                    f"${res.total_usd / 1e6:,.2f}M").add_to(ships)
    ships.add_to(m)

    for n, d in G.nodes(data=True):
        if d["kind"] == "port":
            folium.CircleMarker([d["lat"], d["lon"]], radius=5, color="#e8f3fa", weight=2, fill=True,
                                fill_color="#0d1c2c", fill_opacity=1, tooltip=d["name"]).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m
