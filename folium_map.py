"""
folium_map.py - interactive Leaflet map (folium) of the stressed network and the chosen routes.

``build_folium_map`` returns a ``folium.Map``; app.py displays it with ``streamlit_folium.st_folium``.
Route lines are colour-coded by vessel: Blue for container ships, Amber for tankers.
"""
from __future__ import annotations


import folium

from data_engine import GEO_ZONE, RED_SEA_ZONES
from events import HURRICANE_TRACKS, normalise
from panels import hazard_tooltip

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
    _network_layers(m, G, s)

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
    _ports(m, G)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def _ports(m, G) -> None:
    for _n, d in G.nodes(data=True):
        if d["kind"] == "port":
            folium.CircleMarker([d["lat"], d["lon"]], radius=5, color="#e8f3fa", weight=2, fill=True,
                                fill_color="#0d1c2c", fill_opacity=1, tooltip=d["name"]).add_to(m)


def _network_layers(m, G, s: dict) -> None:
    """Risk-coloured lanes and active scenario zones (shared by the stress-test and live-operations maps)."""
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
        pts = unwrap(list(zip(d["sample_lat"].tolist(), d["sample_lon"].tolist(), strict=False)))
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
    if s.get("bab_el_mandeb_closure"):
        _zone(zones, "bab_el_mandeb", RED, "Bab-el-Mandeb: CLOSED (scenario)")
    for cell in s.get("storm_cells", []):
        src = ("forecast-informed scenario (Open-Meteo)" if str(cell.get("source", "")).startswith("open-meteo")
               else "simulated scenario")
        folium.Circle([cell["lat"], cell["lon"]], radius=cell["radius_nm"] * 1852, color=STORM, weight=1.5,
                      dash_array="6 5", fill=True, fill_color=STORM, fill_opacity=0.18,
                      tooltip=f"Storm {cell.get('label', '')}: {cell['wave_m']:g} m waves ({src})").add_to(zones)
    zones.add_to(m)


# --------------------------------------------------------------------------------------------------
# Live operations map: static base (network + scenarios) and a dynamic voyage layer pushed every tick
# --------------------------------------------------------------------------------------------------
MINT, MUTED, ALT = "#4de0c8", "#829aac", "#f5bf75"
SEV_COLORS = {0: "#4de0c8", 1: "#f5bf75", 2: "#ff9f5a", 3: "#fe7d81"}


HAZARD_COLORS = {"mint": MINT, "blue": BLUE, "gold": AMBER, "red": RED}
HAZARD_FILL = {"Low": 0.04, "Guarded": 0.10, "Elevated": 0.16, "High": 0.22, "Severe": 0.28}


def _hazard_layer(zones) -> folium.FeatureGroup:
    """Security zones tinted by their current risk band (SIMULATED levels), toggleable from the layer control."""
    fg = folium.FeatureGroup(name="Hazard zones (risk band)", show=True)
    for z in zones:
        la0, la1, lo0, lo1 = z["bounds"]
        color = HAZARD_COLORS[z["tone"]]
        folium.Rectangle([[la0, lo0], [la1, lo1]], color=color, weight=1, opacity=0.6, fill=True, fill_color=color,
                         fill_opacity=HAZARD_FILL[z["band"]], tooltip=hazard_tooltip(z)).add_to(fg)
    return fg


def build_ops_base_map(G, settings: dict, center=(25.0, 40.0), zoom: int = 2, hazards=()) -> folium.Map:
    """Static base: network, scenario zones, hazard zones and ports. Anything that depends on where the ship is
    (port warnings) is in the per-tick voyage layer instead, so this map is rebuilt only when scenarios change."""
    s = normalise(settings)
    m = folium.Map(location=list(center), zoom_start=zoom, min_zoom=2, tiles=DARK_TILES, attr=DARK_ATTR,
                   max_zoom=12, world_copy_jump=True, control_scale=True)
    _network_layers(m, G, s)
    if hazards:
        _hazard_layer(hazards).add_to(m)
    _ports(m, G)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def _ship_icon(course: float, status: str) -> folium.DivIcon:
    color = {"blocked": RED, "arrived": MINT}.get(status, "#ffffff")
    svg = (f'<div style="transform:rotate({course:.0f}deg);width:26px;height:26px">'
           f'<svg viewBox="0 0 24 24" width="26" height="26"><path d="M12 2 L18 20 L12 16 L6 20 Z" fill="{color}" '
           f'stroke="#07121f" stroke-width="1.5"/></svg></div>')
    return folium.DivIcon(html=svg, icon_size=(26, 26), icon_anchor=(13, 13))


def build_ops_layer(G, voyage, done_pts, ahead_pts, waypoints, ship, candidates_pts=(), weather_pts=(),
                    route_color=BLUE, port_warnings=()) -> folium.FeatureGroup:
    """Everything that moves: completed track, remaining route, abandoned / blocked routes, alternatives, port
    warnings (they depend on where the ship is), ship."""
    fg = folium.FeatureGroup(name="Voyage")
    for w in port_warnings:
        color = RED if w["kind"] == "unreachable" else AMBER
        folium.CircleMarker([w["lat"], w["lon"]], radius=11, color=color, weight=2, fill=False,
                            tooltip=w["detail"]).add_to(fg)
    for name, pts in candidates_pts:
        if len(pts) > 1:
            folium.PolyLine(unwrap(pts), color=ALT, weight=2, opacity=0.55, dash_array="1 7",
                            tooltip=f"Alternative considered: {name}").add_to(fg)
    for pts in voyage.superseded:
        if len(pts) > 1:
            folium.PolyLine(unwrap(pts), color=MUTED, weight=2.5, opacity=0.8, dash_array="6 8",
                            tooltip="Committed route before the change (dashed); the new route is solid").add_to(fg)
    if voyage.blocked_remaining and len(voyage.blocked_remaining) > 1:
        folium.PolyLine(unwrap(voyage.blocked_remaining), color=RED, weight=3, opacity=0.9, dash_array="4 6",
                        tooltip="Planned route - BLOCKED").add_to(fg)
    if len(ahead_pts) > 1:
        folium.PolyLine(unwrap(ahead_pts), color=route_color, weight=4, opacity=0.95,
                        tooltip="Remaining route (current plan)").add_to(fg)
    if len(done_pts) > 1:
        folium.PolyLine(unwrap(done_pts), color=MINT, weight=4.5, opacity=0.95, tooltip="Completed track").add_to(fg)
    for w in waypoints:
        folium.CircleMarker([w["lat"], w["lon"]], radius=4, color="#cde7ed", weight=1, fill=True,
                            fill_color="#102236", fill_opacity=1,
                            tooltip=f"{w['name']} · voyage day {w['t'] / 24:.1f} (click to jump here)").add_to(fg)
    for p in weather_pts:
        if p.get("level", -1) < 0:
            continue
        v = p["values"]
        folium.CircleMarker([p["lat"], p["lon"]], radius=7, color=SEV_COLORS[p["level"]], weight=2, fill=True,
                            fill_color=SEV_COLORS[p["level"]], fill_opacity=0.35,
                            tooltip=(f"Open-Meteo forecast at simulated ETA {p['eta']:%b %d %H:%M} UTC: "
                                     f"waves {v.get('wave_height')} m"
                                     + (f", wind {v.get('wind_speed_10m')} kn" if v.get('wind_speed_10m') is not None else "")
                                     + f" ({p['severity']})")).add_to(fg)
    folium.Marker([ship.lat, ship.lon], icon=_ship_icon(ship.course_deg, voyage.status),
                  tooltip=(f"SIMULATED position {ship.lat:.2f}, {ship.lon:.2f} · {ship.sog_kn:.1f} kn · "
                           f"course {ship.course_deg:03.0f}° · {ship.activity}")).add_to(fg)
    return fg
