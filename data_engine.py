"""
data_engine.py - synthetic global maritime network.

This module mocks the ingestion of open geospatial datasets (global shipping lanes, World Port
Index) by generating a NetworkX graph whose *geometry* matches the real world: nodes sit at real
port and chokepoint coordinates, edges follow plausible deep-water lanes, and risk baselines are
driven by hard-coded lat/lon bounding boxes (Strait of Hormuz, Red Sea, Gulf of Aden, Taiwan
Strait, ...).

Graph contract (everything downstream relies on this)
-----------------------------------------------------
Nodes
    name, lat, lon, kind ("port" | "waypoint" | "chokepoint"), chokepoint (key or None),
    geopolitical_risk (0..1, baseline risk at the node's own coordinates)
Edges (undirected)
    base_distance_nm     lane length in nautical miles
    weather_risk         0..1, mean synthetic sea-state severity along the lane
    geopolitical_risk    0..1, mean conflict/piracy exposure along the lane
    geopolitical_peak    0..1, worst point along the lane (drives "avoid if premium spikes")
    weather_peak         0..1, worst point along the lane (drives storm closures)
    mode                 "sea" | "canal" | "overland"
    chokepoint           chokepoint key or None
    tolls                {vessel_type: USD} transit fee, empty for free lanes
    max_draft_m          depth limit (canals), None if unrestricted
    speed_cap_kn         speed limit (canals), None if unrestricted
    wait_hours           queueing/convoy time added to transit
    blocked              bool, True when the lane is closed (set by events.apply_events)
    sample_lat/lon, geo_samples_base, weather_samples_base   per-sample arrays used to re-score
                         the lane when events change the world

``build_maritime_graph`` is deterministic for a given seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

SAMPLE_SPACING_NM = 25.0
AMBIENT_GEO_RISK = 0.02
AMBIENT_WEATHER_RISK = 0.10
WEATHER_BLOCK_THRESHOLD = 0.85          # lanes with a sample this stormy are closed to shipping


# =============================================================================================
# Risk zones: hard-coded bounding boxes
# =============================================================================================
@dataclass(frozen=True)
class RiskZone:
    key: str
    label: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    baseline: float

    def contains(self, lat, lon):
        """Vectorised membership test for arrays (or scalars) of lat/lon."""
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        return (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)


# Geopolitical baselines. Overlapping boxes take the maximum, so Bab-el-Mandeb (a narrow, hotter
# box) sits on top of the wider Red Sea box.
GEOPOLITICAL_ZONES: list[RiskZone] = [
    RiskZone("hormuz", "Strait of Hormuz", 24.0, 27.6, 54.0, 58.6, 0.40),
    RiskZone("persian_gulf", "Persian Gulf", 24.0, 30.5, 47.5, 56.0, 0.25),
    RiskZone("red_sea", "Red Sea", 12.0, 28.5, 32.0, 44.0, 0.20),
    RiskZone("bab_el_mandeb", "Bab-el-Mandeb", 11.5, 13.5, 42.0, 44.5, 0.28),
    RiskZone("gulf_of_aden", "Gulf of Aden", 10.0, 15.5, 43.0, 52.0, 0.22),
    RiskZone("somali_basin", "Somali Basin", -2.0, 10.0, 45.0, 58.0, 0.12),
    RiskZone("taiwan_strait", "Taiwan Strait", 22.0, 26.0, 117.0, 121.5, 0.12),
    RiskZone("south_china_sea", "South China Sea", 5.0, 21.0, 107.0, 118.0, 0.10),
    RiskZone("malacca", "Strait of Malacca", 0.5, 6.5, 95.0, 104.5, 0.10),
    RiskZone("gulf_of_guinea", "Gulf of Guinea", -4.0, 7.0, -8.0, 10.0, 0.14),
    RiskZone("east_med", "Eastern Mediterranean", 31.0, 36.5, 32.0, 36.5, 0.08),
]
GEO_ZONE = {z.key: z for z in GEOPOLITICAL_ZONES}
RED_SEA_ZONES = ("red_sea", "bab_el_mandeb", "gulf_of_aden")

# Synthetic climatology (a "typical bad-ish season"): 0 = glassy, 1 = survival conditions.
WEATHER_ZONES: list[RiskZone] = [
    RiskZone("north_atlantic", "North Atlantic storm track", 40.0, 62.0, -60.0, 0.0, 0.42),
    RiskZone("biscay", "Bay of Biscay", 43.0, 49.0, -10.0, -1.0, 0.45),
    RiskZone("north_pacific_w", "North Pacific (west)", 35.0, 55.0, 150.0, 180.0, 0.40),
    RiskZone("north_pacific_e", "North Pacific (east)", 35.0, 55.0, -180.0, -130.0, 0.40),
    RiskZone("cape_of_good_hope", "Cape of Good Hope", -42.0, -30.0, 12.0, 40.0, 0.50),
    RiskZone("cape_horn", "Cape Horn / Drake Passage", -62.0, -45.0, -85.0, -55.0, 0.70),
    RiskZone("indian_monsoon", "Indian monsoon belt", 5.0, 22.0, 60.0, 95.0, 0.30),
    RiskZone("typhoon_belt", "West Pacific typhoon belt", 10.0, 28.0, 108.0, 140.0, 0.30),
    RiskZone("caribbean", "Caribbean", 8.0, 24.0, -90.0, -60.0, 0.15),
    RiskZone("mexican_pacific", "Mexican Pacific", 8.0, 24.0, -112.0, -92.0, 0.15),
]

CHOKEPOINTS: dict[str, str] = {
    "hormuz": "Strait of Hormuz",
    "bab_el_mandeb": "Bab-el-Mandeb",
    "suez": "Suez Canal",
    "malacca": "Strait of Malacca",
    "gibraltar": "Strait of Gibraltar",
    "dover": "Dover Strait",
    "cape_of_good_hope": "Cape of Good Hope",
    "cape_horn": "Cape Horn",
    "panama": "Panama Canal",
    "taiwan": "Taiwan Strait",
}

# Illustrative canal transit fees by vessel type (USD). Real dues depend on tonnage and category.
CANAL_TOLLS = {
    "suez": {"Container": 450_000, "Tanker": 600_000},
    "panama": {"Container": 400_000, "Tanker": 350_000},
}


def zone_risk(lat, lon, zones: list[RiskZone], default: float) -> np.ndarray:
    """Risk at each (lat, lon): the max baseline of any zone containing the point, else ``default``."""
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    out = np.full(lat.shape, default, dtype=float)
    for z in zones:
        m = z.contains(lat, lon)
        out[m] = np.maximum(out[m], z.baseline)
    return out


# =============================================================================================
# Nodes and lanes
# =============================================================================================
# id: (name, lat, lon, kind, chokepoint)
NODE_DEFS: dict[str, tuple] = {
    # --- Hub ports
    "SHA": ("Shanghai", 31.0, 122.3, "port", None),
    "HKG": ("Hong Kong", 22.2, 114.3, "port", None),
    "SIN": ("Singapore", 1.2, 103.9, "port", None),
    "MUM": ("Mumbai", 18.9, 72.8, "port", None),
    "DXB": ("Dubai (Jebel Ali)", 25.0, 55.05, "port", None),
    "FJR": ("Fujairah", 25.15, 56.4, "port", None),
    "RTM": ("Rotterdam", 51.95, 4.0, "port", None),
    "NYC": ("New York", 40.4, -73.8, "port", None),
    "LAX": ("Los Angeles", 33.6, -118.4, "port", None),
    # --- East Asia
    "ECS": ("East China Sea", 28.5, 123.0, "waypoint", None),
    "TWS": ("Taiwan Strait", 24.3, 119.8, "chokepoint", "taiwan"),
    "PHS": ("East of Taiwan", 23.5, 123.5, "waypoint", None),
    "LUZ": ("Luzon Strait", 20.3, 121.2, "waypoint", None),
    "SCS0": ("South China Sea (north)", 21.5, 117.0, "waypoint", None),
    "SCS1": ("South China Sea (mid)", 17.5, 114.0, "waypoint", None),
    "SCS2": ("South China Sea (south)", 10.5, 109.5, "waypoint", None),
    # --- Malacca and the Bay of Bengal
    "SES": ("Singapore Strait (east)", 1.3, 104.6, "waypoint", None),
    "SGW": ("Singapore Strait (west)", 1.15, 103.2, "waypoint", None),
    "MAL1": ("Strait of Malacca", 2.3, 101.5, "chokepoint", "malacca"),
    "MAL2": ("Malacca (north)", 5.6, 98.0, "waypoint", None),
    "GCH": ("Great Channel", 6.6, 93.4, "waypoint", None),
    "SLK": ("South of Sri Lanka", 5.2, 80.5, "waypoint", None),
    "KON": ("Off the Konkan coast", 15.0, 72.0, "waypoint", None),
    "MUMS": ("Lakshadweep Sea", 8.0, 75.0, "waypoint", None),
    # --- Arabian Sea, Gulf, Red Sea
    "ARB": ("Arabian Sea", 14.5, 66.0, "waypoint", None),
    "GOO": ("Gulf of Oman", 24.8, 58.8, "waypoint", None),
    "HS1": ("Hormuz approach", 25.4, 57.1, "waypoint", None),
    "HOR": ("Strait of Hormuz", 26.5, 56.5, "chokepoint", "hormuz"),
    "PGU": ("Persian Gulf", 26.0, 54.5, "waypoint", None),
    "GOA": ("Gulf of Aden", 12.0, 47.5, "waypoint", None),
    "BAB": ("Bab-el-Mandeb", 12.6, 43.4, "chokepoint", "bab_el_mandeb"),
    "RS1": ("Red Sea (south)", 17.0, 40.5, "waypoint", None),
    "RS2": ("Red Sea (mid)", 22.0, 37.8, "waypoint", None),
    "RS3": ("Red Sea (north)", 27.2, 34.2, "waypoint", None),
    "GS1": ("Gulf of Suez (mouth)", 27.5, 33.6, "waypoint", None),
    "GS2": ("Gulf of Suez (south)", 28.0, 33.4, "waypoint", None),
    "GS3": ("Gulf of Suez (mid)", 28.5, 32.95, "waypoint", None),
    "GS4": ("Gulf of Suez (north)", 29.0, 32.75, "waypoint", None),
    "GS5": ("Gulf of Suez (head)", 29.5, 32.5, "waypoint", None),
    "SUZ": ("Suez (south)", 29.8, 32.4, "chokepoint", "suez"),
    "PSD": ("Port Said", 31.3, 32.35, "chokepoint", "suez"),
    # --- Mediterranean and Atlantic Europe
    "MED1": ("East Mediterranean", 33.8, 27.0, "waypoint", None),
    "MED2": ("Ionian Sea", 34.6, 20.0, "waypoint", None),
    "MED3": ("Strait of Sicily", 37.6, 11.6, "waypoint", None),
    "MED4": ("West Mediterranean", 37.9, 5.0, "waypoint", None),
    "ALB": ("Alboran Sea", 36.0, -2.0, "waypoint", None),
    "GIB": ("Strait of Gibraltar", 35.9, -5.6, "chokepoint", "gibraltar"),
    "ATL1": ("Off Portugal", 36.5, -10.5, "waypoint", None),
    "ATL2": ("Off Galicia", 42.8, -10.5, "waypoint", None),
    "BISC": ("Bay of Biscay", 47.0, -7.5, "waypoint", None),
    "USH": ("Ushant", 49.0, -5.8, "waypoint", None),
    "CHA": ("English Channel", 49.9, -2.0, "waypoint", None),
    "CH2": ("Eastern Channel", 50.2, 0.2, "waypoint", None),
    "DOV": ("Dover Strait", 51.0, 1.5, "chokepoint", "dover"),
    # --- Cape of Good Hope route
    "IO1": ("Indian Ocean (central)", -12.0, 66.0, "waypoint", None),
    "IO2": ("Indian Ocean (southwest)", -28.0, 48.0, "waypoint", None),
    "IO3": ("Agulhas", -36.0, 28.0, "waypoint", None),
    "CAPE": ("Cape of Good Hope", -36.0, 19.0, "chokepoint", "cape_of_good_hope"),
    "ATL3": ("Off Namibia", -26.0, 9.0, "waypoint", None),
    "ATL4": ("Gulf of Guinea", -5.0, -4.0, "waypoint", None),
    "ATL5": ("Off West Africa", 14.0, -20.5, "waypoint", None),
    "ATL6": ("Off Madeira", 33.0, -20.0, "waypoint", None),
    # --- North Atlantic
    "NATL": ("North Atlantic (north)", 42.0, -40.0, "waypoint", None),
    "SA": ("North Atlantic (south)", 35.5, -40.0, "waypoint", None),
    "ANN": ("Atlantic (northeast of Puerto Rico)", 21.5, -65.5, "waypoint", None),
    "GUI1": ("Off the Guianas", 12.0, -50.0, "waypoint", None),
    "MON": ("Mona Passage", 18.2, -67.9, "waypoint", None),
    "CAR1": ("Caribbean Sea", 14.0, -75.0, "waypoint", None),
    "COL": ("Colon (Caribbean mouth)", 9.7, -79.9, "chokepoint", "panama"),
    # --- Panama and the Pacific coast of the Americas
    "BAL": ("Balboa (Pacific mouth)", 8.5, -79.4, "chokepoint", "panama"),
    "GPN": ("Gulf of Panama", 6.5, -80.0, "waypoint", None),
    "CRC": ("Off Costa Rica", 7.0, -84.5, "waypoint", None),
    "MX3": ("Gulf of Tehuantepec (offshore)", 11.5, -90.0, "waypoint", None),
    "MX2": ("Off Acapulco", 14.0, -98.0, "waypoint", None),
    "MX1": ("Off Cabo Corrientes", 17.0, -106.5, "waypoint", None),
    "CAB": ("Off Cabo San Lucas", 21.5, -110.5, "waypoint", None),
    "BJ2": ("Off Baja (south)", 25.0, -114.5, "waypoint", None),
    "BJ1": ("Off Baja (north)", 30.0, -118.0, "waypoint", None),
    # --- Trans-Pacific great-circle lane
    "JP1": ("South of Kyushu", 29.5, 130.0, "waypoint", None),
    "JP2": ("South of Shikoku", 31.0, 135.5, "waypoint", None),
    "JP3": ("South of Honshu", 33.0, 141.5, "waypoint", None),
    "NP1": ("North Pacific 1", 38.5, 155.0, "waypoint", None),
    "NP2": ("North Pacific 2", 44.0, 172.0, "waypoint", None),
    "NP3": ("North Pacific 3", 47.0, -175.0, "waypoint", None),
    "NP4": ("North Pacific 4", 46.0, -155.0, "waypoint", None),
    "NP5": ("North Pacific 5", 40.0, -136.0, "waypoint", None),
    "NP6": ("Off California", 34.8, -123.5, "waypoint", None),
    # --- South America, Cape Horn, South Atlantic
    "SAM1": ("Off Ecuador", 1.0, -82.5, "waypoint", None),
    "SAM2": ("Off Peru (north)", -6.0, -83.5, "waypoint", None),
    "SAM3": ("Off Peru (south)", -15.0, -79.0, "waypoint", None),
    "SAM4": ("Off northern Chile", -25.0, -73.5, "waypoint", None),
    "SAM5": ("Off central Chile", -38.0, -76.5, "waypoint", None),
    "SAM6": ("Off southern Chile", -50.0, -78.5, "waypoint", None),
    "SAM7": ("Southeast Pacific", -57.0, -76.0, "waypoint", None),
    "CHN": ("Cape Horn", -57.5, -68.0, "chokepoint", "cape_horn"),
    "SAT1": ("South Atlantic 1", -54.0, -55.0, "waypoint", None),
    "SAT2": ("South Atlantic 2", -38.0, -46.0, "waypoint", None),
    "SAT3": ("South Atlantic 3", -15.0, -32.0, "waypoint", None),
    "SAT4": ("South Atlantic 4", 0.0, -28.0, "waypoint", None),
}

# (a, b, options). Options: chokepoint, mode, tolls, distance_nm, speed_cap_kn, wait_hours,
# max_draft_m, km, flat_cost_usd.
EDGE_DEFS: list[tuple] = [
    # East Asia
    ("SHA", "ECS", {}),
    ("ECS", "TWS", {"chokepoint": "taiwan"}),
    ("TWS", "SCS0", {}),
    ("SCS0", "HKG", {}),
    ("HKG", "SCS1", {}),
    ("SCS0", "SCS1", {}),
    ("ECS", "PHS", {}),
    ("PHS", "LUZ", {}),
    ("LUZ", "SCS1", {}),
    ("SCS1", "SCS2", {}),
    ("SCS2", "SES", {}),
    ("SES", "SIN", {}),
    # Malacca, Bay of Bengal, India
    ("SIN", "SGW", {}),
    ("SGW", "MAL1", {}),
    ("MAL1", "MAL2", {"chokepoint": "malacca"}),
    ("MAL2", "GCH", {}),
    ("GCH", "SLK", {}),
    ("MUM", "KON", {}),
    ("KON", "MUMS", {}),
    ("MUMS", "SLK", {}),
    ("MUM", "ARB", {}),
    ("SLK", "ARB", {}),
    # Arabian Sea and the Gulf
    ("ARB", "GOO", {}),
    ("GOO", "HS1", {}),
    ("HS1", "HOR", {"chokepoint": "hormuz"}),
    ("HOR", "PGU", {"chokepoint": "hormuz"}),
    ("PGU", "DXB", {}),
    ("GOO", "FJR", {}),
    ("FJR", "DXB", {"mode": "overland", "km": 135.0, "flat_cost_usd": 1_500_000}),
    # Red Sea and Suez
    ("ARB", "GOA", {}),
    ("GOA", "BAB", {"chokepoint": "bab_el_mandeb"}),
    ("BAB", "RS1", {}),
    ("RS1", "RS2", {}),
    ("RS2", "RS3", {}),
    ("RS3", "GS1", {}),
    ("GS1", "GS2", {}),
    ("GS2", "GS3", {}),
    ("GS3", "GS4", {}),
    ("GS4", "GS5", {}),
    ("GS5", "SUZ", {}),
    ("SUZ", "PSD", {"mode": "canal", "chokepoint": "suez", "tolls": CANAL_TOLLS["suez"], "distance_nm": 104.0,
                    "speed_cap_kn": 8.0, "wait_hours": 12.0, "max_draft_m": 20.1}),
    # Mediterranean to the Channel
    ("PSD", "MED1", {}),
    ("MED1", "MED2", {}),
    ("MED2", "MED3", {}),
    ("MED3", "MED4", {}),
    ("MED4", "ALB", {}),
    ("ALB", "GIB", {"chokepoint": "gibraltar"}),
    ("GIB", "ATL1", {}),
    ("ATL1", "ATL2", {}),
    ("ATL2", "BISC", {}),
    ("BISC", "USH", {}),
    ("USH", "CHA", {}),
    ("CHA", "CH2", {}),
    ("CH2", "DOV", {"chokepoint": "dover"}),
    ("DOV", "RTM", {}),
    # Cape of Good Hope route
    ("SLK", "IO1", {}),
    ("ARB", "IO1", {}),
    ("IO1", "IO2", {}),
    ("IO2", "IO3", {}),
    ("IO3", "CAPE", {"chokepoint": "cape_of_good_hope"}),
    ("CAPE", "ATL3", {}),
    ("ATL3", "ATL4", {}),
    ("ATL4", "ATL5", {}),
    ("ATL5", "ATL6", {}),
    ("ATL6", "ATL1", {}),
    # North Atlantic
    ("USH", "NATL", {}),
    ("NATL", "NYC", {}),
    ("NATL", "SA", {}),
    ("ATL1", "SA", {}),
    ("ATL6", "SA", {}),
    ("SA", "NYC", {}),
    # Caribbean and Panama
    ("ANN", "SA", {}),
    ("ANN", "NATL", {}),
    ("ANN", "NYC", {}),
    ("MON", "ANN", {}),
    ("CAR1", "MON", {}),
    ("COL", "CAR1", {}),
    ("BAL", "COL", {"mode": "canal", "chokepoint": "panama", "tolls": CANAL_TOLLS["panama"], "distance_nm": 43.0,
                    "speed_cap_kn": 8.0, "wait_hours": 24.0, "max_draft_m": 15.2}),
    ("BAL", "GPN", {}),
    # Pacific coast of the Americas
    ("GPN", "CRC", {}),
    ("CRC", "MX3", {}),
    ("MX3", "MX2", {}),
    ("MX2", "MX1", {}),
    ("MX1", "CAB", {}),
    ("CAB", "BJ2", {}),
    ("BJ2", "BJ1", {}),
    ("BJ1", "LAX", {}),
    # Trans-Pacific
    ("ECS", "JP1", {}),
    ("PHS", "JP1", {}),
    ("JP1", "JP2", {}),
    ("JP2", "JP3", {}),
    ("JP3", "NP1", {}),
    ("NP1", "NP2", {}),
    ("NP2", "NP3", {}),
    ("NP3", "NP4", {}),
    ("NP4", "NP5", {}),
    ("NP5", "NP6", {}),
    ("NP6", "LAX", {}),
    # Cape Horn route
    ("GPN", "SAM1", {}),
    ("SAM1", "SAM2", {}),
    ("SAM2", "SAM3", {}),
    ("SAM3", "SAM4", {}),
    ("SAM4", "SAM5", {}),
    ("SAM5", "SAM6", {}),
    ("SAM6", "SAM7", {}),
    ("SAM7", "CHN", {"chokepoint": "cape_horn"}),
    ("CHN", "SAT1", {}),
    ("SAT1", "SAT2", {}),
    ("SAT2", "SAT3", {}),
    ("SAT3", "SAT4", {}),
    ("SAT4", "ATL5", {}),
    # South American Atlantic coast, offshore: a ship off Brazil can reach the Caribbean (and Panama) directly
    # instead of backtracking across the Atlantic via West Africa.
    ("ANN", "GUI1", {}),
    ("GUI1", "SAT4", {}),
]


# =============================================================================================
# Geometry helpers
# =============================================================================================
def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 3440.065 * math.asin(min(1.0, math.sqrt(h)))


def interpolate_lane(lat1, lon1, lat2, lon2, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` mid-segment points along a lane, taking the short way round the antimeridian."""
    dlon = lon2 - lon1
    if dlon > 180:
        dlon -= 360
    elif dlon < -180:
        dlon += 360
    t = (np.arange(n) + 0.5) / n
    lat = lat1 + (lat2 - lat1) * t
    lon = lon1 + dlon * t
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


# =============================================================================================
# Public builder
# =============================================================================================
def build_maritime_graph(seed: int = 7) -> nx.Graph:
    """Create the baseline (no-event) maritime network. Deterministic for a given ``seed``."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    for nid, (name, lat, lon, kind, choke) in NODE_DEFS.items():
        geo = float(zone_risk(lat, lon, GEOPOLITICAL_ZONES, AMBIENT_GEO_RISK)[0])
        G.add_node(nid, name=name, lat=lat, lon=lon, kind=kind, chokepoint=choke, geopolitical_risk=geo)

    for a, b, opt in EDGE_DEFS:
        na, nb = G.nodes[a], G.nodes[b]
        mode = opt.get("mode", "sea")
        if mode == "overland":
            nm = 0.0
            km = float(opt["km"])
            n = 2
        else:
            km = 0.0
            nm = float(opt.get("distance_nm") or haversine_nm(na["lat"], na["lon"], nb["lat"], nb["lon"]))
            n = max(2, math.ceil(nm / SAMPLE_SPACING_NM))
        lat_s, lon_s = interpolate_lane(na["lat"], na["lon"], nb["lat"], nb["lon"], n)

        geo = zone_risk(lat_s, lon_s, GEOPOLITICAL_ZONES, AMBIENT_GEO_RISK)
        geo = np.clip(geo + rng.uniform(0.0, 0.015), 0.0, 1.0)            # small synthetic jitter
        wx = zone_risk(lat_s, lon_s, WEATHER_ZONES, AMBIENT_WEATHER_RISK)
        wx = np.clip(wx + rng.normal(0.0, 0.03), 0.0, 0.8)
        if mode in ("canal", "overland"):
            wx = np.full_like(wx, 0.05)                                    # sheltered

        G.add_edge(
            a, b,
            base_distance_nm=round(nm, 1),
            km=km,
            weather_risk=float(wx.mean()),
            weather_peak=float(wx.max()),
            geopolitical_risk=float(geo.mean()),
            geopolitical_peak=float(geo.max()),
            mode=mode,
            chokepoint=opt.get("chokepoint"),
            tolls=dict(opt.get("tolls", {})),
            max_draft_m=opt.get("max_draft_m"),
            speed_cap_kn=opt.get("speed_cap_kn"),
            wait_hours=float(opt.get("wait_hours", 0.0)),
            flat_cost_usd=float(opt.get("flat_cost_usd", 0.0)),
            toll_multiplier=1.0,
            blocked=False,
            active_events=[],
            sample_lat=lat_s,
            sample_lon=lon_s,
            geo_samples_base=geo,
            weather_samples_base=wx,
        )
    return G


def hub_ports(G: nx.Graph) -> dict[str, str]:
    """{node id: display name} for every port, in a stable order."""
    return {n: d["name"] for n, d in G.nodes(data=True) if d["kind"] == "port"}


def edges_dataframe(G: nx.Graph) -> pd.DataFrame:
    """Flat table of lane attributes (handy for inspection and for tests)."""
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "u": u, "v": v, "mode": d["mode"], "chokepoint": d["chokepoint"],
            "distance_nm": d["base_distance_nm"], "km": d["km"],
            "weather_risk": round(d["weather_risk"], 3), "geopolitical_risk": round(d["geopolitical_risk"], 3),
            "geopolitical_peak": round(d["geopolitical_peak"], 3), "blocked": d["blocked"],
        })
    return pd.DataFrame(rows)
