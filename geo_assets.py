"""
geo_assets.py - land geometry and map projection shared by the map renderer and the tests.

Where the coastlines come from
------------------------------
* ``assets/land_natural_earth.svg`` is a low-resolution Natural Earth land layer (public domain)
  that covers lon -31..139 and lat 69..-53, which includes Europe, Africa, the Middle East,
  India, South-East Asia and China - the waters this project cares about most.
* ``EXTRA_LAND`` below fills in everything outside that box (the Americas, Japan east of 139E,
  the Russian Far East, Australia, New Guinea, New Zealand) with simplified hand-built outlines.
  These are schematic: good enough for a chart backdrop, not for navigation.

Both layers are exposed as lon/lat polygons (for point-in-land tests) and as projected SVG paths
(for drawing). The map uses a Mercator projection clipped to lat 69N..58S.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

ASSET_PATH = Path(__file__).parent / "assets" / "land_natural_earth.svg"

# --- how the Natural Earth SVG was projected (plate carree, 1200 x 650 px) --------------------
NE_W, NE_H = 1200.0, 650.0
NE_LON0, NE_LON_SPAN = -31.0, 170.0
NE_LAT_TOP, NE_LAT_SPAN = 69.0, 122.0
NE_LON_MIN, NE_LON_MAX = NE_LON0, NE_LON0 + NE_LON_SPAN          # -31 .. 139
NE_LAT_MAX, NE_LAT_MIN = NE_LAT_TOP, NE_LAT_TOP - NE_LAT_SPAN     # 69 .. -53

# --- world map projection used by this app (Mercator) ----------------------------------------
WORLD_W = 2400.0
MAP_LAT_TOP, MAP_LAT_BOTTOM = 69.0, -58.0


def _merc(lat: float) -> float:
    lat = max(-85.0, min(85.0, lat))
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


_Y_TOP, _Y_BOTTOM = _merc(MAP_LAT_TOP), _merc(MAP_LAT_BOTTOM)
WORLD_H = round(WORLD_W / (2 * math.pi) * (_Y_TOP - _Y_BOTTOM), 1)


def project(lat: float, lon: float) -> tuple[float, float]:
    """(lat, lon) in degrees -> (x, y) in world-map pixels."""
    x = (lon + 180.0) / 360.0 * WORLD_W
    y = (_Y_TOP - _merc(lat)) / (_Y_TOP - _Y_BOTTOM) * WORLD_H
    return x, y


# =============================================================================================
# Natural Earth layer
# =============================================================================================
def _ne_xy_to_lonlat(x: float, y: float) -> tuple[float, float]:
    return NE_LON0 + x / NE_W * NE_LON_SPAN, NE_LAT_TOP - y / NE_H * NE_LAT_SPAN


@lru_cache(maxsize=1)
def natural_earth_polygons() -> list[list[tuple[float, float]]]:
    """Rings of (lon, lat) from the Natural Earth SVG (one ring per <path>)."""
    text = ASSET_PATH.read_text(encoding="utf-8")
    rings = []
    for d in re.findall(r'<path d="([^"]+)"', text):
        pts = [_ne_xy_to_lonlat(float(a), float(b)) for a, b in re.findall(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", d)]
        if len(pts) >= 3:
            rings.append(pts)
    return rings


# =============================================================================================
# Hand-built outlines for everything outside the Natural Earth box  (lon, lat)
# =============================================================================================
NORTH_AMERICA = [
    # Pacific coast, Darien -> Costa Rica -> Mexico
    (-77.8, 7.2), (-78.4, 8.0), (-78.9, 8.4), (-79.5, 8.9), (-79.9, 8.6), (-80.0, 7.55), (-80.9, 7.25),
    (-81.2, 7.8), (-82.0, 8.2), (-82.9, 8.3), (-83.4, 8.6), (-83.7, 9.0), (-84.5, 9.6), (-85.2, 9.8),
    (-85.7, 10.5), (-85.7, 11.1), (-86.8, 12.0), (-87.5, 12.9), (-88.3, 13.2), (-89.4, 13.5), (-90.8, 13.9),
    (-92.2, 14.5), (-93.9, 15.8), (-95.2, 16.2), (-96.5, 15.7), (-98.0, 16.2), (-99.9, 16.8), (-101.5, 17.5),
    (-103.5, 18.4), (-105.2, 19.9), (-105.7, 20.7), (-105.4, 21.7), (-106.3, 23.2), (-108.0, 25.0),
    (-109.4, 26.5), (-110.9, 27.9), (-112.4, 29.0), (-113.1, 31.0), (-114.8, 31.8),
    # Gulf of California down the Baja east coast
    (-114.5, 30.3), (-113.7, 29.2), (-112.9, 28.2), (-112.2, 27.0), (-111.4, 25.9), (-110.3, 24.2),
    (-109.5, 23.2), (-109.9, 22.9),
    # Baja Pacific coast back up to California
    (-110.3, 23.6), (-111.8, 24.7), (-112.2, 25.8), (-113.1, 26.7), (-114.1, 27.8), (-115.0, 28.0),
    (-114.6, 28.7), (-115.5, 29.7), (-116.3, 30.8), (-116.8, 32.0), (-117.15, 32.55),
    # California -> Alaska
    (-117.4, 33.2), (-117.9, 33.6), (-118.25, 33.72), (-118.45, 33.75), (-118.5, 34.03), (-119.0, 34.07),
    (-119.6, 34.4), (-120.5, 34.45), (-120.65, 34.6), (-120.9, 35.4), (-121.9, 36.6), (-122.4, 37.1),
    (-122.5, 37.8), (-123.0, 38.0), (-123.8, 39.4), (-124.4, 40.44), (-124.2, 41.75), (-124.55, 42.8),
    (-124.1, 44.6), (-124.0, 46.2), (-124.7, 48.4), (-125.6, 49.0), (-126.8, 49.9), (-127.9, 51.0),
    (-128.5, 52.3), (-130.0, 53.8), (-131.4, 54.9), (-133.0, 56.4), (-135.5, 58.0), (-137.5, 58.9),
    (-139.5, 59.7), (-141.5, 60.0), (-144.5, 60.1), (-147.5, 60.9), (-149.5, 60.0), (-151.5, 59.6),
    (-153.5, 59.0), (-155.5, 58.0), (-157.5, 56.8), (-159.5, 55.8), (-162.0, 55.0), (-164.4, 54.8),
    (-163.5, 55.6), (-161.0, 56.0), (-158.8, 57.5), (-157.8, 58.7), (-160.5, 58.6), (-162.0, 59.0),
    (-164.5, 60.5), (-166.0, 62.0), (-164.8, 63.0), (-161.5, 63.8), (-166.0, 64.5), (-168.1, 65.6),
    # Arctic coast (above the map edge, kept crude) and Labrador
    (-166.0, 68.4), (-158.0, 71.3), (-141.0, 69.7), (-125.0, 70.0), (-110.0, 68.5), (-95.0, 68.5),
    (-88.0, 69.5), (-81.0, 69.5), (-70.0, 66.0), (-64.4, 60.3), (-61.8, 57.5), (-60.0, 55.5), (-57.0, 53.5),
    (-55.7, 52.0), (-58.0, 51.4), (-60.0, 50.2), (-63.0, 50.2), (-66.5, 50.2), (-68.5, 49.3), (-68.0, 48.5),
    (-66.0, 49.2), (-65.5, 48.2), (-64.8, 47.0), (-64.5, 46.2),
    # Nova Scotia -> New England -> mid-Atlantic
    (-63.5, 45.8), (-61.5, 45.7), (-60.0, 46.2), (-59.8, 45.9), (-61.5, 45.0), (-63.5, 44.5), (-65.8, 43.5),
    (-66.2, 44.4), (-67.0, 44.8), (-68.5, 44.3), (-70.0, 43.7), (-70.6, 42.8), (-70.0, 41.8), (-70.5, 41.6),
    (-71.5, 41.4), (-71.9, 41.1), (-73.0, 40.9), (-73.9, 40.6), (-74.0, 40.45), (-74.0, 39.6), (-74.9, 38.9),
    (-75.1, 38.3), (-75.9, 37.1), (-76.0, 36.9), (-75.5, 35.2), (-76.6, 34.6), (-77.9, 33.9), (-79.0, 33.2),
    (-79.9, 32.7), (-80.8, 32.0), (-81.4, 30.7), (-81.0, 29.5), (-80.6, 28.4), (-80.1, 26.8), (-80.1, 25.8),
    (-80.4, 25.2), (-81.1, 25.15), (-81.8, 26.1),
    # Gulf of Mexico
    (-82.7, 27.5), (-82.7, 28.9), (-83.8, 29.9), (-85.3, 29.7), (-86.5, 30.4), (-88.0, 30.3), (-89.3, 29.0),
    (-90.3, 29.1), (-91.8, 29.5), (-93.8, 29.7), (-95.0, 29.2), (-96.5, 28.3), (-97.3, 27.5), (-97.2, 25.9),
    (-97.7, 24.0), (-97.8, 22.3), (-97.2, 20.7), (-96.1, 19.2), (-94.8, 18.4), (-94.0, 18.2), (-92.7, 18.6),
    (-91.5, 18.7), (-90.6, 19.4), (-90.4, 20.8), (-90.3, 21.3), (-89.5, 21.3), (-88.0, 21.6), (-87.0, 21.5),
    (-86.8, 20.5), (-87.5, 19.0), (-87.9, 18.2), (-88.3, 16.6),
    # Caribbean coast of Central America
    (-88.0, 15.8), (-86.0, 15.9), (-84.2, 15.6), (-83.3, 15.0), (-83.4, 13.2), (-83.6, 11.5), (-83.7, 10.9),
    (-83.0, 9.9), (-82.3, 9.3), (-81.2, 9.0), (-80.0, 9.35), (-79.0, 9.55), (-78.0, 9.2), (-77.4, 8.6),
]

HUDSON_BAY = [  # water cut-out
    (-94.8, 58.8), (-93.0, 61.0), (-90.5, 63.2), (-86.5, 64.0), (-83.0, 63.7), (-81.0, 62.8), (-78.5, 62.5),
    (-77.5, 60.5), (-78.5, 58.0), (-79.5, 55.5), (-82.0, 52.9), (-85.0, 55.0), (-88.0, 56.5), (-92.0, 57.2),
]

LONG_ISLAND = [(-73.6, 40.6), (-72.0, 41.0), (-71.9, 41.1), (-73.9, 40.6)]
CUBA = [(-84.9, 21.9), (-83.0, 22.9), (-81.0, 23.1), (-79.0, 22.5), (-77.5, 21.6), (-75.7, 20.7), (-74.2, 20.2),
        (-75.2, 19.9), (-77.2, 19.9), (-78.5, 21.4), (-81.0, 22.0), (-84.5, 21.8)]
HISPANIOLA = [(-74.4, 18.4), (-73.0, 19.9), (-71.7, 19.9), (-70.0, 19.8), (-68.4, 18.6), (-69.0, 18.4),
              (-70.5, 18.2), (-71.4, 17.7), (-72.9, 18.1)]
PUERTO_RICO = [(-67.2, 18.5), (-65.7, 18.4), (-65.6, 18.0), (-67.2, 17.95)]
JAMAICA = [(-78.3, 18.4), (-76.3, 18.2), (-76.8, 17.8), (-78.2, 18.0)]

GREENLAND = [(-73.0, 78.5), (-58.0, 82.0), (-30.0, 83.5), (-12.0, 81.5), (-18.0, 76.0), (-20.0, 73.0),
             (-22.0, 70.0), (-27.0, 68.5), (-35.0, 66.0), (-40.0, 65.0), (-43.0, 60.0), (-48.0, 61.0),
             (-51.0, 64.0), (-54.0, 67.0), (-54.0, 70.0), (-58.0, 75.0), (-67.0, 76.0)]

SOUTH_AMERICA = [
    (-77.4, 7.2), (-77.4, 8.6), (-76.0, 9.4), (-74.8, 11.0), (-72.5, 11.9), (-71.7, 12.4), (-71.0, 11.5),
    (-69.5, 11.5), (-67.0, 10.6), (-64.0, 10.6), (-62.0, 10.7), (-61.0, 9.0), (-60.0, 8.5), (-57.0, 6.0),
    (-52.0, 5.0), (-50.0, 1.0), (-48.5, -1.0), (-44.0, -2.5), (-38.5, -3.7), (-35.0, -5.5), (-35.0, -9.0),
    (-38.5, -13.0), (-39.0, -18.0), (-41.0, -22.0), (-43.2, -23.0), (-48.0, -26.0), (-49.5, -30.0),
    (-52.5, -33.0), (-54.5, -34.8), (-56.0, -34.9), (-57.0, -37.0), (-62.0, -39.0), (-65.0, -41.0),
    (-65.5, -45.0), (-67.5, -46.5), (-68.5, -50.5), (-68.3, -52.5), (-65.0, -54.8), (-67.3, -55.9),
    (-70.5, -55.2), (-73.0, -53.5), (-75.5, -50.0), (-75.5, -46.5), (-73.7, -42.0), (-73.5, -37.0),
    (-72.0, -32.0), (-71.6, -30.0), (-70.3, -18.3), (-71.5, -17.0), (-75.5, -15.5), (-76.3, -13.7),
    (-77.2, -12.0), (-79.5, -8.0), (-81.2, -6.0), (-81.3, -4.6), (-80.2, -3.0), (-81.0, -2.2), (-80.5, -0.8),
    (-80.1, 0.5), (-79.0, 1.3), (-78.5, 2.0), (-77.8, 3.5), (-77.1, 3.9), (-77.4, 6.0),
]

FAR_EAST_RUSSIA = [
    (139.0, 46.2), (139.0, 70.0), (150.0, 71.5), (160.0, 69.7), (170.0, 70.0), (180.0, 69.0), (180.0, 65.0),
    (177.0, 64.5), (170.0, 60.0), (163.0, 59.8), (162.0, 57.0), (160.5, 54.5), (156.7, 51.0), (155.9, 55.0),
    (156.5, 57.5), (159.5, 61.0), (155.0, 59.3), (150.5, 59.2), (143.0, 59.3), (140.5, 57.5), (139.0, 54.0),
    (140.5, 53.5), (141.5, 50.5), (140.5, 48.0),
]
SAKHALIN = [(142.0, 46.0), (143.2, 49.0), (143.5, 52.0), (142.7, 54.0), (142.0, 53.0), (142.2, 49.0)]
HONSHU_KYUSHU = [(130.9, 34.0), (135.0, 33.5), (139.8, 35.2), (141.0, 38.0), (141.9, 40.5), (140.2, 41.0),
                 (139.9, 40.0), (136.7, 37.3), (133.0, 35.5)]
HOKKAIDO = [(140.0, 41.5), (141.5, 45.4), (145.5, 43.3), (143.0, 42.0)]
NEW_GUINEA = [(131.0, -0.8), (135.0, -3.3), (138.0, -1.6), (141.0, -2.6), (144.5, -3.8), (147.0, -6.0),
              (150.5, -10.5), (147.5, -10.1), (143.5, -9.0), (141.0, -9.1), (138.0, -8.2), (137.5, -5.5),
              (135.0, -4.2), (133.0, -4.0)]
AUSTRALIA = [(114.0, -22.0), (113.5, -26.0), (115.0, -34.3), (118.0, -35.0), (124.0, -33.8), (129.0, -31.6),
             (134.0, -32.8), (138.0, -35.0), (140.0, -38.0), (146.0, -39.0), (150.0, -37.5), (153.5, -28.5),
             (153.0, -25.0), (150.0, -22.0), (146.0, -18.5), (145.5, -14.5), (143.5, -11.0), (141.5, -13.0),
             (141.5, -17.0), (139.5, -17.5), (136.5, -12.0), (132.5, -11.5), (130.0, -12.5), (127.0, -14.0),
             (125.0, -15.0), (122.0, -17.5), (118.0, -20.0)]
NZ_NORTH = [(172.7, -34.5), (174.5, -36.0), (176.0, -37.5), (178.5, -37.7), (177.0, -39.5), (175.5, -41.5),
            (174.8, -41.3), (175.0, -39.6), (173.8, -39.2), (174.6, -37.5), (173.0, -35.4)]
NZ_SOUTH = [(172.6, -40.6), (174.2, -41.3), (173.0, -43.0), (171.2, -44.5), (170.0, -46.0), (168.0, -46.5),
            (166.5, -45.5), (168.0, -44.0), (171.0, -42.0)]

EXTRA_LAND = [NORTH_AMERICA, LONG_ISLAND, CUBA, HISPANIOLA, PUERTO_RICO, JAMAICA, GREENLAND, SOUTH_AMERICA,
              FAR_EAST_RUSSIA, SAKHALIN, HONSHU_KYUSHU, HOKKAIDO, NEW_GUINEA, AUSTRALIA, NZ_NORTH, NZ_SOUTH]
EXTRA_WATER = [HUDSON_BAY]


# =============================================================================================
# Point-in-land test (used by tests to prove that no sea lane crosses land)
# =============================================================================================
def _in_ring(lon: float, lat: float, ring) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@lru_cache(maxsize=1)
def _ne_index():
    out = []
    for ring in natural_earth_polygons():
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        out.append((min(xs), max(xs), min(ys), max(ys), ring))
    return out


def in_natural_earth_box(lat: float, lon: float) -> bool:
    return NE_LON_MIN <= lon <= NE_LON_MAX and NE_LAT_MIN <= lat <= NE_LAT_MAX


def on_land(lat: float, lon: float) -> bool:
    """True if (lat, lon) is on land according to the combined coastline layers."""
    if in_natural_earth_box(lat, lon):
        return any(x0 <= lon <= x1 and y0 <= lat <= y1 and _in_ring(lon, lat, ring)
                   for x0, x1, y0, y1, ring in _ne_index())
    if any(_in_ring(lon, lat, r) for r in EXTRA_WATER):
        return False
    return any(_in_ring(lon, lat, r) for r in EXTRA_LAND)


# =============================================================================================
# SVG output
# =============================================================================================
def _path(points) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in (project(lat, lon) for lon, lat in points)) + " Z"


def land_svg_markup() -> str:
    """SVG <g> elements for the land layer, in world-map pixel coordinates (fill only, no stroke).

    Natural Earth data is drawn inside its own box; the hand-built outlines are drawn through an
    even-odd clip that removes that box, so the two layers never overlap.
    """
    ne_d = " ".join(_path(r) for r in natural_earth_polygons())
    x0, y0 = project(NE_LAT_MAX, NE_LON_MIN)
    x1, y1 = project(NE_LAT_MIN, NE_LON_MAX)
    outside = f"M0,0 H{WORLD_W:.0f} V{WORLD_H:.0f} H0 Z M{x0:.1f},{y0:.1f} H{x1:.1f} V{y1:.1f} H{x0:.1f} Z"
    extra_d = " ".join(_path(r) for r in EXTRA_LAND)
    hudson_d = " ".join(_path(r) for r in EXTRA_WATER)
    return (
        f'<clipPath id="outsideNE"><path clip-rule="evenodd" d="{outside}"/></clipPath>'
        f'<g class="land"><path d="{ne_d}"/>'
        f'<g clip-path="url(#outsideNE)"><path d="{extra_d}"/><path class="sea" d="{hudson_d}"/></g></g>'
    )
