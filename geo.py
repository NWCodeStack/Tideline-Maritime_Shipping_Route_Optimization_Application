"""geo.py - lane geometry: positions along lanes, bearings and polylines (pure functions over the lane graph)."""
from __future__ import annotations

import math

import networkx as nx


def node_ll(G: nx.Graph, n: str) -> tuple[float, float]:
    d = G.nodes[n]
    return d["lat"], d["lon"]


def point_on_lane(G: nx.Graph, x: str, y: str, f: float) -> tuple[float, float]:
    (la, lo), (lb, lb_lon) = node_ll(G, x), node_ll(G, y)
    dlon = lb_lon - lo
    if dlon > 180:
        dlon -= 360
    elif dlon < -180:
        dlon += 360
    lon = lo + dlon * f
    return la + (lb - la) * f, (lon + 180.0) % 360.0 - 180.0


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    yv = math.sin(dl) * math.cos(p2)
    xv = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(yv, xv)) + 360.0) % 360.0


def legs_polyline(G: nx.Graph, legs: list) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for l in legs:
        if l.kind == "hold":
            continue
        a, b = point_on_lane(G, l.x, l.y, l.f0), point_on_lane(G, l.x, l.y, l.f1)
        if not pts or pts[-1] != a:
            pts.append(a)
        pts.append(b)
    return pts
