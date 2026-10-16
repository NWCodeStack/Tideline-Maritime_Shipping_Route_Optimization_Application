"""
economics.py - cost factors beyond fuel / tolls / war-risk / time, and the objective presets.

Everything here is a SIMULATED or ESTIMATED assumption, labelled as such wherever it is shown. Not for
commercial, regulatory or navigation decisions.

Decision score for one lane (the optimiser minimises this when it picks a speed per lane):

    cash  = fuel + canal tolls + war-risk insurance + time cost + ECA fuel switch + EU ETS
    score = cash + (time_value_mult - 1) x time cost + carbon shadow price x CO2

``cash`` is money that leaves the company and is what the cost breakdown sums to. The carbon shadow price and the
time-value multiplier are decision weights: they steer the optimiser (slower, cleaner steaming) but are never
presented as cash. The idea of presets built from sliders (value of time, carbon price) and the sample bunker
prices follow the Pelagic reference prototype (optimizer.py PRESETS, data/network.json).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CO2_PER_T_FUEL = 3.114                 # t CO2 per t of fuel oil burned (IMO default factor for heavy fuel oil)
ECA_FUEL_PREMIUM_USD_PER_T = 250.0     # extra cost per t of fuel burned inside an ECA (low-sulphur fuel), ESTIMATE
EU_PORTS = frozenset({"RTM"})          # ports inside the EU ETS scope in this network
DEFAULT_EUA_USD_PER_T = 70.0           # EU carbon allowance price, ESTIMATE (user-editable)
DEFAULT_ETS_COVERAGE_PCT = 50.0        # share of a voyage's emissions surrendered (extra-EU voyages), ESTIMATE

# Simplified emission control areas as lat/lon boxes (name, lat_min, lat_max, lon_min, lon_max). These are
# deliberately coarse rectangles, not the legal boundaries.
ECA_BOXES = (
    ("North Sea & English Channel", 48.4, 62.0, -6.0, 9.0),
    ("Baltic Sea", 53.5, 66.0, 9.4, 30.0),
    ("North American east coast", 24.0, 48.0, -80.0, -66.0),
    ("North American west coast", 30.0, 49.0, -127.0, -117.0),
    ("Mediterranean", 30.0, 46.0, -5.3, 36.0),
)

# Bunker price (VLSFO, USD/t) at each port. SAMPLE values: the Pelagic normal-times table where it has the port,
# Tideline estimates otherwise. Not quotes.
PORT_FUEL_PRICES = {
    "RTM": {"usd_per_t": 575.0, "basis": "SAMPLE (Pelagic normal-times table)"},
    "SHA": {"usd_per_t": 610.0, "basis": "SAMPLE (Pelagic normal-times table)"},
    "SIN": {"usd_per_t": 585.0, "basis": "SAMPLE (Pelagic normal-times table)"},
    "DXB": {"usd_per_t": 615.0, "basis": "SAMPLE (Pelagic normal-times table, Jebel Ali)"},
    "FJR": {"usd_per_t": 600.0, "basis": "SAMPLE (Pelagic normal-times table)"},
    "HKG": {"usd_per_t": 600.0, "basis": "SAMPLE (Tideline estimate)"},
    "MUM": {"usd_per_t": 620.0, "basis": "SAMPLE (Tideline estimate)"},
    "NYC": {"usd_per_t": 640.0, "basis": "SAMPLE (Tideline estimate)"},
    "LAX": {"usd_per_t": 650.0, "basis": "SAMPLE (Tideline estimate)"},
}


@dataclass(frozen=True)
class Economics:
    """User-editable economic assumptions (hashable, so it can sit inside the pricing cache key)."""
    time_value_mult: float = 1.0          # >1: a delayed hour counts for more than its cash cost (decision weight)
    carbon_usd_per_t: float = 0.0         # carbon shadow price, USD per t CO2 (decision weight, never cash)
    ets_enabled: bool = True
    eua_usd_per_t: float = DEFAULT_EUA_USD_PER_T
    ets_coverage_pct: float = DEFAULT_ETS_COVERAGE_PCT
    fuel_price_mode: str = "global"       # "global" = the fuel-price slider; "origin_port" = the departure port's price
    fuel_overrides: tuple = ()            # ((port id, USD/t), ...) user prices that beat the sample table


@dataclass(frozen=True)
class Preset:
    label: str
    preference: str                       # the routing objective it selects
    time_value_mult: float
    carbon_usd_per_t: float
    blurb: str


PRESETS = {
    "balanced": Preset("Balanced", "balanced", 1.0, 0.0, "Weighted mix of time, cost and risk."),
    "cheapest": Preset("Cheapest", "cheapest", 1.0, 0.0, "Lowest expenses, time only as a tie-break."),
    "fastest": Preset("Fastest", "fastest", 4.0, 0.0, "Shortest transit; a delayed hour counts 4x."),
    "safest": Preset("Safest", "safest", 1.0, 0.0, "Least exposure to weather and security risk."),
    "greenest": Preset("Greenest", "balanced", 1.0, 200.0, "Pays $200/t CO2 in the decision, so ships steam slower."),
}


def apply_preset(key: str) -> tuple[str, dict]:
    """(routing objective, Economics field changes) for a preset. Raises KeyError for an unknown key."""
    p = PRESETS[key]
    return p.preference, {"time_value_mult": p.time_value_mult, "carbon_usd_per_t": p.carbon_usd_per_t}


def preset_key(preference: str, econ: Economics) -> str:
    """The preset these settings equal, or "custom" once any slider has been moved off a preset."""
    for key, p in PRESETS.items():
        if (p.preference == preference and abs(p.time_value_mult - econ.time_value_mult) < 1e-9
                and abs(p.carbon_usd_per_t - econ.carbon_usd_per_t) < 1e-9):
            return key
    return "custom"


def effective_fuel_price(global_price: float, origin: str, econ: Economics) -> float:
    """USD/t used to price the voyage's fuel: the slider, or (origin_port mode) an override / the port's sample price."""
    if econ.fuel_price_mode == "origin_port":
        overrides = dict(econ.fuel_overrides)
        if origin in overrides:
            return float(overrides[origin])
        if origin in PORT_FUEL_PRICES:
            return float(PORT_FUEL_PRICES[origin]["usd_per_t"])
    return float(global_price)


def ets_applies(origin: str, destination: str, econ: Economics) -> bool:
    return bool(econ.ets_enabled and (origin in EU_PORTS or destination in EU_PORTS))


def eca_fraction(sample_lat, sample_lon) -> float:
    """Share of a lane's sample points inside any emission control area (0..1)."""
    lat, lon = np.asarray(sample_lat, dtype=float), np.asarray(sample_lon, dtype=float)
    if lat.size == 0:
        return 0.0
    inside = np.zeros(lat.shape, dtype=bool)
    for _, la0, la1, lo0, lo1 in ECA_BOXES:
        inside |= (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    return float(inside.mean())


def breakdown(summary, econ: Economics) -> list[dict]:
    """The cost lines of a route, each with its USD, whether it is cash, and the basis of its data.

    Lines with ``cash`` True sum to ``summary.total_usd``; the carbon shadow price is shown but not added."""
    price_basis = ("departure-port price, SAMPLE table" if econ.fuel_price_mode == "origin_port"
                   else "price from the fuel-price setting")
    return [
        {"label": "Fuel", "usd": summary.fuel_usd, "cash": True,
         "basis": f"MODEL ESTIMATE (fuel model trained on SIMULATED data); {price_basis}"},
        {"label": "Canal tolls", "usd": summary.toll_usd, "cash": True,
         "basis": "SAMPLE (illustrative tariffs, user-adjustable multipliers)"},
        {"label": "War-risk insurance", "usd": summary.war_usd, "cash": True,
         "basis": "MODEL ESTIMATE (synthetic risk model)"},
        {"label": "Time cost (ship + cargo delay)", "usd": summary.time_usd, "cash": True,
         "basis": "MODEL ESTIMATE (assumed daily cost and lateness penalty)"},
        {"label": "ECA fuel switch", "usd": summary.eca_usd, "cash": True,
         "basis": f"ESTIMATED (${ECA_FUEL_PREMIUM_USD_PER_T:,.0f}/t premium inside simplified ECA boxes)"},
        {"label": "EU ETS", "usd": summary.ets_usd, "cash": True,
         "basis": (f"ESTIMATED (${econ.eua_usd_per_t:,.0f}/t allowance x {econ.ets_coverage_pct:g}% coverage; "
                   "EU-port voyages only)")},
        {"label": "Carbon shadow price (decision weight, not cash)", "usd": summary.carbon_usd, "cash": False,
         "basis": f"USER SETTING (${econ.carbon_usd_per_t:,.0f}/t CO2; steers speed and route only)"},
    ]
