"""
voyage.py - simulated vessel movement and mid-voyage dynamic rerouting.

A ``Voyage`` is an itinerary of ``Leg``s over the maritime graph plus a simulated clock ``t`` (hours since
departure). Every leg is a piece of ONE lane of the graph (a fraction ``f0 -> f1`` of the lane ``x -> y``), so
the ship can only ever be on a drawn, land-checked lane: no straight lines across continents, no teleporting.

Position at time t: walk the legs by duration (canal queue time first, then sailing time) and interpolate along
the lane by distance. Lane lengths are great-circle (haversine) nautical miles; the position is interpolated
along the same lat/lon segment the map draws, which is the geometry the land-crossing test validates.

Rerouting (``reevaluate``) works from the ship's CURRENT position:
  1. cut the itinerary at t (everything already sailed is kept unchanged: distance, time, fuel, cost);
  2. look at the remaining lanes only - a closure the ship has already passed, or that its remaining corridor
     never touches, does not trigger a reroute;
  3. if the remaining route is affected, run Dijkstra from a virtual "vessel" node that sits on the current lane
     (it may continue forward or turn back along that lane) under the selected preference;
  4. if nothing is reachable, the voyage is BLOCKED and the ship holds position - no invented passages.

Vessels never use overland lanes (cargo trucking is a cargo decision, not a ship movement).
All numbers are modelled estimates on a synthetic network. Positions are simulated, never AIS observations.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import networkx as nx

from agents import CARGO_OWNERS, VESSEL_OWNERS, VESSELS, RoutingAgent, make_agent, premium_multiple
from data_engine import CHOKEPOINTS, build_maritime_graph, haversine_nm
from economics import Economics, effective_fuel_price, ets_applies
from events import apply_events, describe, normalise
from risk import TRAFFIC_FULL_RISK_DELAY_H, WEATHER_WEIGHT, band, route_risk
from routing_engine import DEFAULT_FUEL_PRICE_USD_PER_T, FUEL_MODEL, EdgeCost, price_edge_batch

PREFERENCES = {
    "fastest": "Fastest",
    "cheapest": "Cheapest",
    "safest": "Safest",
    "balanced": "Balanced",
}
PREFERENCE_HELP = {
    "fastest": "Minimise hours to arrival (sailing + canal queues).",
    "cheapest": "Minimise voyage expenses: fuel + tolls + war-risk insurance + ship operating cost.",
    "safest": "Minimise risk exposure: hours x (geopolitical risk + 0.5 x weather risk).",
    "balanced": "Weighted sum of time, expenses and risk, each divided by its network-wide average per lane.",
}
VIRTUAL = "__VESSEL__"
FORCED_PENALTY = 2.0               # a route through storm-closed / over-premium lanes must be 2x better to win
REROUTE_TOLERANCE = 0.005          # keep the current route unless the alternative is >0.5% better
CHANGE_COURSE_THRESHOLD_USD = 25_000   # a different route must beat staying by this much all-in (hysteresis)
TURN_PENALTY_H = 4.0               # time to bring the ship about and settle on the reverse heading (simulated)
SAFEST_SWITCH_RISK_POINTS = 3.0    # "safest" objective: switch if the new route's 0-100 risk score is this much lower


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------
@dataclass
class VoyageConfig:
    origin: str = "RTM"
    destination: str = "SHA"
    vessel_key: str = "container"
    owner_key: str = "high_margin"
    cruise_speed_kn: float | None = None          # None -> optimiser picks per lane from the vessel's options
    fuel_price: float = DEFAULT_FUEL_PRICE_USD_PER_T
    departure: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(minute=0, second=0,
                                                                                             microsecond=0))
    preference: str = "balanced"
    weights: tuple = (1.0, 1.0, 1.0)              # balanced mode: (time, cost, risk)
    avoid: tuple = ()                             # chokepoint keys the operator refuses to transit
    port_delay_hours: float = 12.0                # baseline berth wait at the destination
    canal_delay_hours: float = 0.0                # extra queue at every canal on top of the modelled wait
    change_course_threshold_usd: float = CHANGE_COURSE_THRESHOLD_USD   # all-in saving needed to leave the committed route
    econ: Economics = field(default_factory=Economics)   # ECA / EU ETS / carbon / value-of-time / per-port fuel

    def agent(self) -> RoutingAgent:
        owner = self.owner_key if self.owner_key in VESSEL_OWNERS[self.vessel_key] else VESSEL_OWNERS[self.vessel_key][0]
        return make_agent(self.vessel_key, owner, self.cruise_speed_kn)


# --------------------------------------------------------------------------------------------------
# Itinerary pieces
# --------------------------------------------------------------------------------------------------
@dataclass
class Leg:
    x: str                     # lane start node (in travel direction)
    y: str                     # lane end node
    f0: float                  # fraction of the lane x->y where this leg starts
    f1: float                  # ... and ends (f1 > f0; a stationary hold leg has f0 == f1)
    nm: float
    wait_h: float              # queue / berth time spent at the start of the leg
    sea_h: float
    fuel_t: float = 0.0
    fuel_usd: float = 0.0
    toll_usd: float = 0.0
    war_usd: float = 0.0
    time_usd: float = 0.0
    risk: float = 0.0          # sea hours x (geo + 0.5 wx)
    chokepoint: str | None = None
    kind: str = "sail"         # "sail" | "hold" (berth wait / held position)
    note: str = ""
    entered: bool = False      # canal queue + toll already paid (leg starts mid-lane or was cut mid-transit)
    queued_h: float = 0.0      # part of the queue already waited (leg cut while the ship was queueing)
    turn: bool = False         # starts with a U-turn: ``wait_h`` is the turn penalty, not a queue
    eca_usd: float = 0.0       # low-sulphur fuel premium inside emission control areas (cash)
    ets_usd: float = 0.0       # EU ETS allowances (cash)
    co2_t: float = 0.0
    carbon_usd: float = 0.0    # carbon shadow price x CO2: a decision weight, NOT cash
    speed_kn: float = 0.0      # speed through the water the optimiser chose for this lane
    p_sec_base: float = 0.0    # risk signals of this lane (0..1), see risk.py: base security level of its zones,
    p_sec_scen: float = 0.0    # what a simulated scenario added on top,
    p_wx: float = 0.0          # weather (half the lane's sea-state severity),
    p_traf: float = 0.0        # and the simulated canal slowdown
    zone: str = ""             # human label of the lane (chokepoint name, else "A - B")

    @property
    def hours(self) -> float:
        return self.wait_h + self.sea_h

    @property
    def total_usd(self) -> float:
        return self.fuel_usd + self.toll_usd + self.war_usd + self.time_usd + self.eca_usd + self.ets_usd


@dataclass
class RouteSummary:
    label: str
    feasible: bool
    legs: list = field(default_factory=list)
    nm: float = 0.0
    hours: float = 0.0
    fuel_t: float = 0.0
    fuel_usd: float = 0.0
    toll_usd: float = 0.0
    war_usd: float = 0.0
    time_usd: float = 0.0
    risk_index: float = 0.0
    peak_premium: float = 1.0
    chokepoints: list = field(default_factory=list)
    forced: bool = False
    reason: str = ""
    eca_usd: float = 0.0
    ets_usd: float = 0.0
    co2_t: float = 0.0
    carbon_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        """Cash cost (the carbon shadow price is a decision weight and is not included)."""
        return self.fuel_usd + self.toll_usd + self.war_usd + self.time_usd + self.eca_usd + self.ets_usd

    @property
    def corridor(self) -> str:
        major = [c for c in self.chokepoints if c in ("Suez Canal", "Bab-el-Mandeb", "Cape of Good Hope",
                                                      "Strait of Hormuz", "Panama Canal", "Cape Horn",
                                                      "Strait of Malacca")]
        return " > ".join(major) if major else "Direct"


@dataclass
class Candidate:
    name: str                  # "Hold course", "Fastest", ...
    summary: RouteSummary
    chosen: bool = False


@dataclass
class Voyage:
    config: VoyageConfig
    legs: list
    t: float = 0.0
    status: str = "planned"    # planned | underway | arrived | blocked
    settings: dict = field(default_factory=dict)
    superseded: list = field(default_factory=list)       # [(lat, lon)] polylines of abandoned remaining routes
    blocked_remaining: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    decision: str = ""
    recommendation: str = ""   # "stay" | "change_course" | "blocked" ("" before the first assessment)
    block_reason: str = ""
    timeline: list = field(default_factory=list)
    reroutes: int = 0

    @property
    def total_hours(self) -> float:
        return sum(l.hours for l in self.legs)


# --------------------------------------------------------------------------------------------------
# Network context: stressed graph + per-lane prices for one ship and one set of disruptions
# --------------------------------------------------------------------------------------------------
@dataclass
class Context:
    G: nx.Graph
    costs: dict
    agent: RoutingAgent
    config: VoyageConfig
    settings: dict
    means: tuple               # network-average (hours, cost, risk) per lane, for the balanced objective


_BASE: nx.Graph | None = None


def base_graph() -> nx.Graph:
    global _BASE
    if _BASE is None:
        _BASE = build_maritime_graph()
    return _BASE


def _econ_key(cfg: VoyageConfig, settings: dict) -> str:
    s = normalise(settings)
    e = cfg.econ
    return json.dumps({"s": s, "v": cfg.vessel_key, "o": cfg.owner_key, "spd": cfg.cruise_speed_kn,
                       "fp": round(effective_fuel_price(cfg.fuel_price, cfg.origin, e), 2),
                       "av": sorted(cfg.avoid), "cd": cfg.canal_delay_hours,
                       "tvm": e.time_value_mult, "cp": e.carbon_usd_per_t, "eua": e.eua_usd_per_t,
                       "etsc": e.ets_coverage_pct, "ets": ets_applies(cfg.origin, cfg.destination, e)},
                      sort_keys=True, default=str)


_MODELS: dict[int, object] = {}


def build_context(cfg: VoyageConfig, settings: dict, model=None) -> Context:
    model = model or FUEL_MODEL
    _MODELS[id(model)] = model
    G, costs = _priced_graph(_econ_key(cfg, settings), id(model))
    agent = cfg.agent()
    hrs, usd, rsk = [], [], []
    for (u, v), c in ((tuple(k), c) for k, c in costs.items()):
        if c.hard_block:
            continue
        h, e, r = lane_metrics(c, G.edges[u, v], agent)
        hrs.append(h), usd.append(e), rsk.append(r)
    mean = lambda xs: max(sum(xs) / max(len(xs), 1), 1e-9)
    return Context(G, costs, agent, cfg, normalise(settings), (mean(hrs), mean(usd), mean(rsk)))


@lru_cache(maxsize=64)
def _priced_graph(key: str, model_id: int):
    """(stressed sea-only graph, {frozenset(u, v): EdgeCost}). Cached on the full economic key; read-only."""
    k = json.loads(key)
    cfg = VoyageConfig(vessel_key=k["v"], owner_key=k["o"], cruise_speed_kn=k["spd"], fuel_price=k["fp"],
                       avoid=tuple(k["av"]), canal_delay_hours=k["cd"])
    G = apply_events(base_graph(), k["s"])
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["mode"] == "overland"])
    if cfg.canal_delay_hours:
        for _, _, d in G.edges(data=True):
            if d["mode"] == "canal":
                d["wait_hours"] = d["wait_hours"] + cfg.canal_delay_hours
    econ = Economics(time_value_mult=k["tvm"], carbon_usd_per_t=k["cp"], eua_usd_per_t=k["eua"],
                     ets_coverage_pct=k["etsc"])
    costs = price_edge_batch(G.edges(data=True), cfg.agent(), cfg.fuel_price, _MODELS.get(model_id),
                             econ=econ, ets_active=k["ets"])
    avoid = set(cfg.avoid)
    if avoid:
        for key_, c in costs.items():
            u, v = tuple(key_)
            if G.edges[u, v]["chokepoint"] in avoid or G.nodes[u]["chokepoint"] in avoid \
                    or G.nodes[v]["chokepoint"] in avoid:
                costs[key_] = replace(c, hard_block="avoided by operator")
    return G, costs


def lane_metrics(c: EdgeCost, d: dict, agent: RoutingAgent, frac: float = 1.0, with_entry: bool = True):
    """(hours, expenses USD, risk) for a lane, or a fraction of it. Expenses exclude the cargo owner's lateness
    penalty (that is a time preference, not an expense)."""
    sea_h = (c.hours - d["wait_hours"]) * frac
    wait = d["wait_hours"] if with_entry else 0.0
    hours = sea_h + wait
    expenses = c.fuel_usd * frac + (c.toll_usd if with_entry else 0.0) + c.war_usd * frac \
        + (c.eca_usd + c.ets_usd + c.carbon_usd) * frac + hours * agent.vessel.daily_asset_cost_usd / 24.0
    risk = sea_h * (d["geopolitical_risk"] + 0.5 * d["weather_risk"])
    return hours, expenses, risk


def _weight(ctx: Context, c: EdgeCost, d: dict, pref: str, frac: float = 1.0, with_entry: bool = True) -> float:
    h, e, r = lane_metrics(c, d, ctx.agent, frac, with_entry)
    if pref == "fastest":
        return h + 1e-6 * e
    if pref == "cheapest":
        return e + 1e-3 * h
    if pref == "safest":
        return r + 1e-3 * h
    wt, wc, wr = ctx.config.weights
    mh, me, mr = ctx.means
    return wt * h / mh + wc * e / me + wr * r / mr


# --------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# Building legs from a path and summarising them
# --------------------------------------------------------------------------------------------------
def _make_leg(ctx: Context, x: str, y: str, f0: float, f1: float, entered: bool, queued_h: float = 0.0) -> Leg:
    """A piece of lane x->y. Entering a lane at its start costs its queue time and toll; a ship that is already
    on the lane (``entered``) pays neither again."""
    G, agent = ctx.G, ctx.agent
    d = G.edges[x, y]
    c = ctx.costs[frozenset((x, y))]
    frac = f1 - f0
    sea_h = (c.hours - d["wait_hours"]) * frac
    wait = 0.0 if entered else max(d["wait_hours"] - queued_h, 0.0)
    hours = wait + sea_h
    return Leg(x=x, y=y, f0=f0, f1=f1, nm=c.nm * frac, wait_h=wait, sea_h=sea_h,
               fuel_t=c.fuel_t * frac, fuel_usd=c.fuel_usd * frac, toll_usd=0.0 if entered else c.toll_usd,
               war_usd=c.war_usd * frac, time_usd=hours * agent.delay_penalty_usd_per_hr,
               risk=sea_h * (d["geopolitical_risk"] + 0.5 * d["weather_risk"]), chokepoint=d["chokepoint"],
               note=c.soft_block or "", entered=entered, queued_h=queued_h,
               eca_usd=c.eca_usd * frac, ets_usd=c.ets_usd * frac, co2_t=c.co2_t * frac,
               carbon_usd=c.carbon_usd * frac, speed_kn=c.speed_kn, **_risk_signals(ctx, x, y, d))


def _risk_signals(ctx: Context, x: str, y: str, d: dict) -> dict:
    """The lane's inputs to the risk score (risk.py), with the scenario part of security kept apart from its base."""
    base = base_graph().edges[x, y]["geopolitical_risk"]
    G = ctx.G
    traffic = (min(ctx.config.canal_delay_hours / TRAFFIC_FULL_RISK_DELAY_H, 1.0) if d["mode"] == "canal" else 0.0)
    return {"p_sec_base": base, "p_sec_scen": max(d["geopolitical_risk"] - base, 0.0),
            "p_wx": WEATHER_WEIGHT * d["weather_risk"], "p_traf": traffic,
            "zone": CHOKEPOINTS.get(d["chokepoint"]) or f"{G.nodes[x]['name']} - {G.nodes[y]['name']}"}


def _turn_leg(ctx: Context, leg: Leg, extra_h: float = TURN_PENALTY_H) -> Leg:
    """Make ``leg`` start with a U-turn: the ship waits ``extra_h`` more (the penalty, or what is left of it) where
    it is (position stays on the lane) before sailing back, on top of any queue the leg already has. Idle fuel
    during the manoeuvre is not modelled; the time cost is."""
    return replace(leg, wait_h=leg.wait_h + extra_h, turn=True, note="turning around (U-turn penalty)",
                   time_usd=leg.time_usd + extra_h * ctx.agent.delay_penalty_usd_per_hr)


def _turn_weight(ctx: Context, pref: str) -> float:
    """The U-turn's cost in the units of the active objective (hours / expenses / risk-free time)."""
    h = TURN_PENALTY_H
    e = h * ctx.agent.vessel.daily_asset_cost_usd / 24.0
    if pref == "fastest":
        return h
    if pref == "cheapest":
        return e
    if pref == "safest":
        return 1e-3 * h
    wt, wc, _ = ctx.config.weights
    mh, me, _ = ctx.means
    return wt * h / mh + wc * e / me


def _berth_leg(ctx: Context, node: str, hours: float, note: str) -> Leg:
    return Leg(x=node, y=node, f0=0.0, f1=0.0, nm=0.0, wait_h=hours, sea_h=0.0,
               time_usd=hours * ctx.agent.delay_penalty_usd_per_hr, kind="hold", note=note)


def berth_hours(ctx: Context, dest: str) -> float:
    return ctx.config.port_delay_hours + float(ctx.settings["port_congestion"].get(dest, 0.0))


def summarise(ctx: Context, legs: list, label: str = "", forced: bool = False) -> RouteSummary:
    s = RouteSummary(label=label, feasible=True, legs=legs, forced=forced)
    seen: list[str] = []
    peak = 0.0
    for l in legs:
        s.nm += l.nm
        s.hours += l.hours
        s.fuel_t += l.fuel_t
        s.fuel_usd += l.fuel_usd
        s.toll_usd += l.toll_usd
        s.war_usd += l.war_usd
        s.time_usd += l.time_usd
        s.eca_usd += l.eca_usd
        s.ets_usd += l.ets_usd
        s.co2_t += l.co2_t
        s.carbon_usd += l.carbon_usd
        if l.kind == "sail":
            keys = [ctx.G.nodes[l.x]["chokepoint"] if l.f0 == 0.0 else None, l.chokepoint,
                    ctx.G.nodes[l.y]["chokepoint"] if l.f1 >= 1.0 - 1e-9 else None]
            for ck in keys:
                if ck and ck in CHOKEPOINTS and CHOKEPOINTS[ck] not in seen:
                    seen.append(CHOKEPOINTS[ck])
            if ctx.G.has_edge(l.x, l.y):
                peak = max(peak, ctx.G.edges[l.x, l.y]["geopolitical_peak"])
    s.risk_index = 100.0 * route_risk(legs).total
    s.peak_premium = premium_multiple(peak)
    s.chokepoints = seen
    return s


def risk_label(idx: float) -> str:
    return band(idx)[0]


# --------------------------------------------------------------------------------------------------
# Routing from an arbitrary position
# --------------------------------------------------------------------------------------------------
@dataclass
class Position:
    """Where the ship is: at ``node`` or on lane x->y at fraction f (``entered`` = canal queue already done)."""
    node: str | None = None
    x: str | None = None
    y: str | None = None
    f: float = 0.0


def arrival_lane(done: list, pos: Position) -> tuple[str, str] | None:
    """(u, v) of the lane the ship just sailed when it is sitting at node v, else None. Leaving v back toward u is a
    U-turn and pays the turn penalty (a ship part-way along a lane is handled in ``route_from`` itself)."""
    if pos.node is None:
        return None
    last = next((l for l in reversed(done) if l.kind == "sail" and l.nm > 0), None)
    return (last.x, last.y) if last is not None and last.y == pos.node else None


def nearest_reachable_port(G: nx.Graph, H: nx.DiGraph, start: str, dest: str) -> tuple[str, float] | None:
    """(port, great-circle nm to ``dest``) of the port closest to an unreachable ``dest`` that the ship can still
    reach, or None. ``H`` is the lane graph the search used (closures removed) and ``start`` the ship's node in it.
    The port the ship is already sitting at does not count: that case is reported separately."""
    if start not in H:
        return None
    d = G.nodes[dest]
    ports = [n for n in nx.descendants(H, start) if n != dest and n in G and G.nodes[n]["kind"] == "port"]
    if not ports:
        return None
    best = min(ports, key=lambda n: haversine_nm(G.nodes[n]["lat"], G.nodes[n]["lon"], d["lat"], d["lon"]))
    return best, haversine_nm(G.nodes[best]["lat"], G.nodes[best]["lon"], d["lat"], d["lon"])


def reachable_ports(ctx: Context, pos: Position) -> set[str]:
    """Ports the ship could still sail to from ``pos`` with the current hard closures (storm and insurance soft
    blocks are ignored: they can be forced through). A ship part-way along a lane can go on to the next node, or turn
    back to the previous one unless it is inside a canal."""
    G = ctx.G
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    for key, c in ctx.costs.items():
        if not c.hard_block:
            u, v = tuple(key)
            H.add_edge(u, v)
            H.add_edge(v, u)
    if pos.node is not None:
        starts = [pos.node]
    else:
        starts = [pos.y] + ([] if G.edges[pos.x, pos.y]["mode"] == "canal" else [pos.x])
    reach: set[str] = set()
    for s in starts:
        reach |= {s} | nx.descendants(H, s)
    return {n for n in reach if G.nodes[n]["kind"] == "port"}


def route_from(ctx: Context, pos: Position, dest: str, pref: str, via: str | None = None,
               add_berth: bool = True, prev_lane: tuple[str, str] | None = None) -> RouteSummary:
    """Best route from ``pos`` to ``dest`` under ``pref``. Strict pass first (no storm-closed / over-premium lanes),
    then a "forced through" pass that allows those; closures, draft limits and avoided chokepoints never relax.
    ``prev_lane`` = the lane the ship arrived by (see ``arrival_lane``), so reversing out of a node costs a U-turn."""
    if via:
        first = route_from(ctx, pos, via, pref, add_berth=False, prev_lane=prev_lane)
        if not first.feasible:
            return RouteSummary(label=pref, feasible=False, reason=first.reason)
        second = route_from(ctx, Position(node=via), dest, pref, add_berth=add_berth)
        if not second.feasible:
            return RouteSummary(label=pref, feasible=False, reason=second.reason)
        return summarise(ctx, first.legs + second.legs, pref, first.forced or second.forced)

    G = ctx.G
    for relaxed in (False, True):
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes)
        for key, c in ctx.costs.items():
            if c.hard_block or (c.soft_block and not relaxed):
                continue
            u, v = tuple(key)
            w = _weight(ctx, c, G.edges[u, v], pref)
            H.add_edge(u, v, w=w)
            H.add_edge(v, u, w=w)
        start = pos.node
        if pos.node is None:                       # on a lane: may continue to y or turn back to x
            start = VIRTUAL
            c = ctx.costs[frozenset((pos.x, pos.y))]
            d = G.edges[pos.x, pos.y]
            in_canal = d["mode"] == "canal"        # a transit in progress is completed, and nobody turns round in a canal
            if in_canal or not (c.hard_block or (c.soft_block and not relaxed)):
                H.add_edge(VIRTUAL, pos.y, w=_weight(ctx, c, d, pref, 1.0 - pos.f, with_entry=False))
            if not in_canal:
                back = replace(c, hard_block=None, soft_block=None)     # retreating along the lane is always allowed
                H.add_edge(VIRTUAL, pos.x, w=_weight(ctx, back, d, pref, pos.f, with_entry=False)
                           + _turn_weight(ctx, pref))
                if H.has_edge(pos.y, pos.x):       # ...and "forward to y, then straight back" is the same U-turn
                    H[pos.y][pos.x]["w"] += _turn_weight(ctx, pref)
        if start == dest:
            legs = [_berth_leg(ctx, dest, berth_hours(ctx, dest), "berth wait")] if add_berth else []
            return summarise(ctx, legs, pref)
        reverses = (prev_lane is not None and pos.node is not None and pos.node == prev_lane[1]
                    and H.has_edge(pos.node, prev_lane[0]))
        if reverses:
            H[pos.node][prev_lane[0]]["w"] += _turn_weight(ctx, pref)
        try:
            path = nx.dijkstra_path(H, start, dest, weight="w")
        except nx.NetworkXNoPath:
            continue
        legs: list[Leg] = []
        for a, b in zip(path, path[1:]):
            if a == VIRTUAL:
                if b == pos.y:
                    legs.append(_make_leg(ctx, pos.x, pos.y, pos.f, 1.0, entered=True))
                else:                               # turning back: sail the lane y->x from (1-f) to 1
                    legs.append(_turn_leg(ctx, _make_leg(ctx, pos.y, pos.x, 1.0 - pos.f, 1.0, entered=True)))
            else:
                leg = _make_leg(ctx, a, b, 0.0, 1.0, entered=False)
                prev = legs[-1] if legs else None
                if (reverses and a == pos.node and b == prev_lane[0] and not legs) or \
                        (prev is not None and prev.kind == "sail" and (prev.x, prev.y) == (b, a)):
                    leg = _turn_leg(ctx, leg)
                legs.append(leg)
        if add_berth:
            legs.append(_berth_leg(ctx, dest, berth_hours(ctx, dest), "berth wait"))
        return summarise(ctx, legs, pref, forced=relaxed)

    closed = sorted({f"{CHOKEPOINTS.get(ctx.G.edges[tuple(k)]['chokepoint'], '/'.join(sorted(k)))}: {c.hard_block}"
                     for k, c in ctx.costs.items() if c.hard_block})
    where = pos.node and G.nodes[pos.node]["name"] or f"the ship's position near {G.nodes[pos.x]['name']}"
    near = nearest_reachable_port(G, H, start, dest)
    if near:
        hint = f" Nearest port still reachable: {G.nodes[near[0]]['name']}, {near[1]:,.0f} nm from {G.nodes[dest]['name']}."
    elif pos.node is not None and G.nodes[pos.node]["kind"] == "port":
        hint = f" The ship is at {where}, the only port it can still reach."
    else:
        hint = ""
    return RouteSummary(label=pref, feasible=False,
                        reason=f"No feasible sea passage from {where} to {G.nodes[dest]['name']}. "
                               + ("Closed: " + "; ".join(closed[:4]) + "." if closed else "") + hint)


def reprice(ctx: Context, legs: list, label: str = "Hold course") -> RouteSummary:
    """Re-price an existing list of remaining legs under ``ctx``. Infeasible if any remaining lane is closed."""
    out: list[Leg] = []
    forced = False
    for l in legs:
        if l.kind == "hold":
            if l.note == "berth wait":
                out.append(_berth_leg(ctx, l.x, berth_hours(ctx, l.x), "berth wait"))
            else:
                out.append(l)
            continue
        c = ctx.costs.get(frozenset((l.x, l.y)))
        mid_canal = bool(l.entered and c is not None and ctx.G.edges[l.x, l.y]["mode"] == "canal")   # transit completes
        if c is None or (c.hard_block and not mid_canal):
            name = CHOKEPOINTS.get(ctx.G.edges[l.x, l.y]["chokepoint"]) if ctx.G.has_edge(l.x, l.y) else None
            why = c.hard_block if c else "closed"
            return RouteSummary(label=label, feasible=False,
                                reason=f"{name or ctx.G.nodes[l.x]['name'] + ' - ' + ctx.G.nodes[l.y]['name']}: {why}")
        forced |= bool(c.soft_block)
        leg = _make_leg(ctx, l.x, l.y, l.f0, l.f1, entered=l.entered, queued_h=l.queued_h)
        left = max(l.wait_h - leg.wait_h, 0.0) if l.turn else 0.0       # turn penalty still to serve
        out.append(_turn_leg(ctx, leg, left) if left > 0 else leg)
    return summarise(ctx, out, label, forced)


def score(ctx: Context, s: RouteSummary, pref: str) -> float:
    if not s.feasible:
        return math.inf
    agent = ctx.agent
    expenses = s.fuel_usd + s.toll_usd + s.war_usd + s.eca_usd + s.ets_usd + s.carbon_usd \
        + s.hours * agent.vessel.daily_asset_cost_usd / 24.0
    risk = sum(l.risk for l in s.legs)
    penalty = FORCED_PENALTY if s.forced else 1.0
    if pref == "fastest":
        return s.hours * penalty
    if pref == "cheapest":
        return expenses * penalty
    if pref == "safest":
        return risk * penalty + 1e-3 * s.hours
    wt, wc, wr = ctx.config.weights
    mh, me, mr = ctx.means
    # Per-lane averages make the three terms comparable; summing lanes keeps it identical to the Dijkstra weight.
    return penalty * (wt * s.hours / mh + wc * expenses / me + wr * risk / mr)


# --------------------------------------------------------------------------------------------------
# Clock and position
# --------------------------------------------------------------------------------------------------
@dataclass
class ShipState:
    lat: float
    lon: float
    course_deg: float
    sog_kn: float
    leg_index: int
    traveled_nm: float
    remaining_nm: float
    elapsed_h: float
    remaining_h: float
    activity: str              # "sailing" | "canal queue" | "berth wait" | "holding" | "arrived" | "not departed"
    pos: Position


def locate(G: nx.Graph, legs: list, t: float) -> tuple[int, float]:
    """(leg index, hours into that leg) at time t. Past the end -> (last index, its duration)."""
    acc = 0.0
    for i, l in enumerate(legs):
        if t < acc + l.hours - 1e-9:
            return i, t - acc
        acc += l.hours
    return max(len(legs) - 1, 0), legs[-1].hours if legs else 0.0


def ship_state(v: Voyage, G: nx.Graph) -> ShipState:
    legs = v.legs
    if not legs:
        lat, lon = node_ll(G, v.config.origin)
        return ShipState(lat, lon, 0.0, 0.0, 0, 0.0, 0.0, v.t, 0.0, "holding", Position(node=v.config.origin))
    i, into = locate(G, legs, v.t)
    l = legs[i]
    traveled = sum(x.nm for x in legs[:i])
    if l.kind == "hold":
        if l.x != l.y and l.f0 > 0.0:                # held part-way along a lane (e.g. turning around)
            lat, lon = point_on_lane(G, l.x, l.y, l.f0)
            pos = Position(x=l.x, y=l.y, f=l.f0)
        else:
            lat, lon = node_ll(G, l.x)
            pos = Position(node=l.x)
        frac_nm, sog, course = 0.0, 0.0, 0.0
        activity = "berth wait" if l.note == "berth wait" else "holding"
    elif into < l.wait_h:
        lat, lon = point_on_lane(G, l.x, l.y, l.f0)
        pos = Position(node=l.x) if l.f0 == 0.0 else Position(x=l.x, y=l.y, f=l.f0)
        frac_nm, sog = 0.0, 0.0
        course = bearing_deg(*node_ll(G, l.x), *node_ll(G, l.y))
        activity = "turning around" if l.turn else "canal queue"
    else:
        p = (into - l.wait_h) / max(l.sea_h, 1e-9)
        f = l.f0 + (l.f1 - l.f0) * min(max(p, 0.0), 1.0)
        lat, lon = point_on_lane(G, l.x, l.y, f)
        pos = Position(node=l.y) if f >= 1.0 - 1e-9 else (Position(node=l.x) if f <= 1e-9 else
                                                            Position(x=l.x, y=l.y, f=f))
        frac_nm = l.nm * min(max(p, 0.0), 1.0)
        sog = l.nm / max(l.sea_h, 1e-9)
        course = bearing_deg(*node_ll(G, l.x), *node_ll(G, l.y))
        activity = "sailing"
    traveled += frac_nm
    total_nm = sum(x.nm for x in legs)
    if v.status == "arrived":
        activity, sog = "arrived", 0.0
    elif v.status == "blocked":
        activity, sog = "holding", 0.0
    elif v.status == "planned" and v.t == 0:
        activity, sog = "not departed", 0.0
    return ShipState(lat, lon, course, sog, i, traveled, max(total_nm - traveled, 0.0), v.t,
                     max(v.total_hours - v.t, 0.0), activity, pos)


def split_at(v: Voyage, G: nx.Graph) -> tuple[list, list, Position]:
    """(legs already sailed, legs still ahead, position). The leg in progress is cut exactly at time t, so the
    sailed part keeps its distance, hours and costs and the remaining part starts at the ship."""
    if not v.legs or v.t <= 0:
        return [], list(v.legs), Position(node=v.config.origin)
    if v.t >= v.total_hours - 1e-9:                 # at the end of the itinerary (arrived, or held when blocked)
        last = v.legs[-1]
        if last.kind == "hold":
            pos = Position(node=last.x)
        else:
            pos = Position(node=last.y) if last.f1 >= 1 - 1e-9 else Position(x=last.x, y=last.y, f=last.f1)
        return list(v.legs), [], pos
    i, into = locate(G, v.legs, v.t)
    l = v.legs[i]
    done, ahead = list(v.legs[:i]), list(v.legs[i + 1:])
    if into <= 1e-9:
        return done, [l] + ahead, ship_state(v, G).pos
    if l.kind == "hold":
        done.append(replace(l, wait_h=into, time_usd=l.time_usd * into / max(l.hours, 1e-9)))
        ahead.insert(0, replace(l, wait_h=l.hours - into, time_usd=l.time_usd * (1 - into / max(l.hours, 1e-9))))
        return done, ahead, Position(node=l.x)
    if into < l.wait_h:                              # in a canal queue (or a U-turn): the wait is spent, sailing still ahead
        at_node = l.f0 == 0.0                        # a mid-lane wait keeps the ship's lane position
        done.append(Leg(x=l.x, y=l.x if at_node else l.y, f0=l.f0, f1=l.f0, nm=0.0, wait_h=into, sea_h=0.0,
                        time_usd=l.time_usd * into / max(l.hours, 1e-9), kind="hold",
                        note=l.note if l.turn else "canal queue"))
        ahead.insert(0, replace(l, wait_h=l.wait_h - into, queued_h=l.queued_h + into,
                                time_usd=l.time_usd * (1 - into / max(l.hours, 1e-9))))
        return done, ahead, Position(node=l.x) if l.f0 == 0.0 else Position(x=l.x, y=l.y, f=l.f0)
    p = (into - l.wait_h) / max(l.sea_h, 1e-9)
    f = l.f0 + (l.f1 - l.f0) * p
    sailed = replace(l, f1=f, nm=l.nm * p, sea_h=l.sea_h * p, fuel_t=l.fuel_t * p, fuel_usd=l.fuel_usd * p,
                     war_usd=l.war_usd * p, risk=l.risk * p, time_usd=(l.wait_h + l.sea_h * p) / max(l.hours, 1e-9)
                     * l.time_usd, eca_usd=l.eca_usd * p, ets_usd=l.ets_usd * p, co2_t=l.co2_t * p,
                     carbon_usd=l.carbon_usd * p)
    rest = replace(l, f0=f, entered=True, nm=l.nm * (1 - p), wait_h=0.0, sea_h=l.sea_h * (1 - p), fuel_t=l.fuel_t * (1 - p),
                   fuel_usd=l.fuel_usd * (1 - p), toll_usd=0.0, war_usd=l.war_usd * (1 - p), risk=l.risk * (1 - p),
                   time_usd=l.time_usd - sailed.time_usd, eca_usd=l.eca_usd * (1 - p), ets_usd=l.ets_usd * (1 - p),
                   co2_t=l.co2_t * (1 - p), carbon_usd=l.carbon_usd * (1 - p))
    done.append(sailed)
    ahead.insert(0, rest)
    pos = Position(node=l.y) if f >= 1 - 1e-9 else Position(x=l.x, y=l.y, f=f)
    return done, ahead, pos


# --------------------------------------------------------------------------------------------------
# Voyage lifecycle
# --------------------------------------------------------------------------------------------------
def calendar(v: Voyage, hours: float) -> datetime:
    return v.config.departure + timedelta(hours=hours)


def log(v: Voyage, kind: str, title: str, detail: str = "", source: str = "") -> None:
    v.timeline.append({"t": round(v.t, 2), "at": calendar(v, v.t).strftime("%Y-%m-%d %H:%M UTC"),
                       "kind": kind, "title": title, "detail": detail, "source": source})


def plan_voyage(cfg: VoyageConfig, settings: dict, model=None) -> Voyage:
    ctx = build_context(cfg, settings, model)
    v = Voyage(config=cfg, legs=[], settings=copy.deepcopy(normalise(settings)))
    best = route_from(ctx, Position(node=cfg.origin), cfg.destination, cfg.preference)
    G = ctx.G
    name = lambda n: G.nodes[n]["name"]
    if best.feasible:
        v.legs = best.legs
        log(v, "plan", f"Voyage planned: {name(cfg.origin)} → {name(cfg.destination)}",
            f"{PREFERENCES[cfg.preference]} route via {best.corridor}: {best.nm:,.0f} nm, "
            f"{best.hours / 24:.1f} days, est. ${best.total_usd / 1e6:,.2f}M.")
    else:
        v.status = "blocked"
        v.block_reason = best.reason
        log(v, "blocked", "No feasible route at departure", best.reason)
    return v


def advance(v: Voyage, hours: float) -> bool:
    """Move the simulated clock forward. Returns True when the ship arrives during this step."""
    if v.status in ("arrived", "blocked") or hours <= 0 or not v.legs:
        return False
    if v.status == "planned":
        v.status = "underway"
    v.t = min(v.t + hours, v.total_hours)
    if v.t >= v.total_hours - 1e-9:
        v.status = "arrived"
        log(v, "arrival", "Arrived", f"Voyage complete after {v.t / 24:.1f} days.")
        return True
    return False


def seek(v: Voyage, t: float) -> None:
    """Jump to a voyage time along the current itinerary (used by the time slider and waypoint clicks)."""
    if not v.legs:
        return
    v.t = min(max(float(t), 0.0), v.total_hours)
    if v.status != "blocked":
        v.status = "arrived" if v.t >= v.total_hours - 1e-9 else ("underway" if v.t > 0 else "planned")


def _changed_lanes(old: Context, new: Context, legs: list) -> tuple[list, list]:
    """(lanes on the remaining route whose conditions changed, of which hard-closed) by display name."""
    changed, closed = [], []
    for l in legs:
        if l.kind == "hold":
            if l.note == "berth wait" and berth_hours(old, l.x) != berth_hours(new, l.x):
                changed.append(f"{new.G.nodes[l.x]['name']} berth")
            continue
        k = frozenset((l.x, l.y))
        a, b = old.costs.get(k), new.costs.get(k)
        lane = f"{new.G.nodes[l.x]['name']} - {new.G.nodes[l.y]['name']}"
        if b is None or b.hard_block:
            closed.append(lane)
            continue
        if a is None or a.soft_block != b.soft_block or abs(a.objective - b.objective) > 1e-6 * max(abs(a.objective), 1) \
                or abs(a.total_usd - b.total_usd) > 1.0:
            changed.append(lane)
    return changed, closed


def _improved_lanes(old: Context, new: Context) -> list[str]:
    """Lanes ANYWHERE in the network that reopened or became >0.5% cheaper: a better route may now exist even
    though nothing on the current remaining route changed (e.g. a closure was lifted)."""
    out = []
    for k, b in new.costs.items():
        a = old.costs.get(k)
        if a is None:
            continue
        reopened = (a.hard_block and not b.hard_block) or (a.soft_block and not b.soft_block)
        cheaper = not b.hard_block and b.objective < a.objective * (1 - REROUTE_TOLERANCE)
        if reopened or cheaper:
            u, w = tuple(k)
            ck = new.G.edges[u, w]["chokepoint"]
            out.append(CHOKEPOINTS.get(ck) if ck else f"{new.G.nodes[u]['name']} - {new.G.nodes[w]['name']}")
    return list(dict.fromkeys(out))


def reevaluate(v: Voyage, settings: dict, model=None, cause: str = "", force: bool = False,
               via: str | None = None, prev_config: VoyageConfig | None = None, source: str = "") -> str:
    """Re-plan the rest of the voyage from where the ship is now. Returns a one-line decision summary."""
    new_settings = copy.deepcopy(normalise(settings))
    old_ctx = build_context(prev_config or v.config, v.settings, model)
    ctx = build_context(v.config, new_settings, model)
    G = ctx.G
    v.settings = new_settings
    cfg = v.config

    if v.status == "arrived":
        msg = "Ship has already arrived; nothing to reroute."
        log(v, "info", cause or "Re-evaluation", msg, source)
        return msg

    done, ahead, pos = split_at(v, G)
    prev_lane = arrival_lane(done, pos)
    was_blocked = v.status == "blocked"
    changed, closed = _changed_lanes(old_ctx, ctx, ahead) if ahead else ([], [])
    improved = _improved_lanes(old_ctx, ctx)
    hold = reprice(ctx, ahead) if ahead else RouteSummary("Hold course", False, reason="No remaining route.")
    objective_changed = prev_config is not None and prev_config.preference != cfg.preference
    affected = bool(changed or closed or improved) or was_blocked or force or via is not None or objective_changed

    if not affected:
        events = ", ".join(describe(new_settings)) or "no active disruptions"
        msg = (f"STAY ON COURSE. Remaining route NOT affected ({hold.corridor}); no reroute needed. "
               f"Active scenarios: {events}.")
        v.candidates = [Candidate("Hold course", hold, chosen=True)]
        v.recommendation = "stay"
        v.decision = msg
        log(v, "assessment", cause or "Scenario assessed", msg, source)
        return msg

    cands: list[Candidate] = []
    if hold.feasible:
        cands.append(Candidate("Hold course", hold))
    for pref in PREFERENCES:
        s = route_from(ctx, pos, cfg.destination, pref, via=via, prev_lane=prev_lane)
        s.label = PREFERENCES[pref] + (f" via {G.nodes[via]['name']}" if via else "")
        if s.feasible and not any(_same(s, c.summary) for c in cands):
            cands.append(Candidate(s.label, s))
        elif s.feasible:
            for c in cands:
                if _same(s, c.summary) and c.name != "Hold course" and PREFERENCES[pref] not in c.name:
                    c.name += f" / {PREFERENCES[pref]}"
    feasible = [c for c in cands if c.summary.feasible]
    name = lambda n: G.nodes[n]["name"]

    if not feasible:
        reason = route_from(ctx, pos, cfg.destination, cfg.preference, prev_lane=prev_lane).reason
        v.legs = done
        v.t = sum(l.hours for l in done)
        v.blocked_remaining = legs_polyline(G, ahead)
        v.status = "blocked"
        v.block_reason = reason
        v.candidates = []
        v.recommendation = "blocked"
        v.decision = f"VOYAGE BLOCKED. {reason}"
        log(v, "blocked", cause or "Voyage blocked", v.decision, source)
        return v.decision

    if via is not None:
        pool = [c for c in feasible if c.name != "Hold course"] or feasible
    else:
        pool = feasible
    best = min(pool, key=lambda c: score(ctx, c.summary, cfg.preference))
    rejected = None          # a route the objective prefers that still does not beat staying by the threshold
    if not (force or objective_changed) and via is None and hold.feasible and best.name != "Hold course" \
            and not _worth_changing(ctx, hold, best.summary):
        rejected, best = best, cands[0]
    best.chosen = True
    v.candidates = cands

    if was_blocked:
        v.status = "underway" if v.t > 0 else "planned"
        v.block_reason = ""
        v.blocked_remaining = []

    if best.name == "Hold course":
        n = len(changed) + len(closed)
        if n:
            what = "closures" if closed else "costs or conditions"
            lead = (f"Remaining route affected: {what} changed on {n} lane(s) "
                    f"({', '.join((closed + changed)[:3])}{'...' if n > 3 else ''}); ")
        elif improved:
            lead = f"Conditions improved elsewhere ({', '.join(improved[:2])}); alternatives re-checked: "
        else:
            lead = "The remaining route does not use the affected lanes; alternatives re-checked: "
        msg = ("STAY ON COURSE. " + lead
               + f"holding course is still the best {PREFERENCES[cfg.preference].lower()} option. "
               f"Remaining: {hold.nm:,.0f} nm, {hold.hours / 24:.1f} d, est. ${hold.total_usd / 1e6:,.2f}M, "
               f"risk {hold.risk_index:.0f}/100.")
        if rejected is not None:
            alt = rejected.summary
            msg += (f" A different route ({alt.corridor}) scores slightly better but does not beat staying by the "
                    f"${cfg.change_course_threshold_usd:,.0f} change-course threshold.")
        v.legs = done + hold.legs
        v.recommendation = "stay"
        v.decision = msg
        log(v, "assessment", cause or "Route re-assessed", msg, source)
        return msg

    new = best.summary
    v.superseded = [legs_polyline(G, ahead)] if ahead else []
    v.legs = done + new.legs
    v.reroutes += 1
    why = explain_choice(ctx, best, cands)
    turning = bool(new.legs and new.legs[0].turn)
    where = name(pos.node) if pos.node else f"its position near {name(pos.x)}"
    headline = _change_headline(G, pos, new, hold if hold.feasible else None)
    turn_note = f" The ship turns around (+{TURN_PENALTY_H:g} h)." if turning else ""
    msg = (headline + f"REROUTED from {where} ({ship_state(v, G).lat:.2f}, {ship_state(v, G).lon:.2f}) "
           f"via {new.corridor}: {new.nm:,.0f} nm remaining, {new.hours / 24:.1f} d, est. "
           f"${new.total_usd / 1e6:,.2f}M, risk {new.risk_index:.0f}/100.{turn_note} {why}")
    if closed:
        msg = f"Remaining route closed at {', '.join(closed[:2])}. " + msg
    elif improved and not changed:
        msg = f"Conditions improved ({', '.join(improved[:2])}). " + msg
    v.recommendation = "change_course"
    v.decision = msg
    log(v, "reroute", cause or "Rerouted", msg, source)
    return msg


def _worth_changing(ctx: Context, hold: RouteSummary, alt: RouteSummary) -> bool:
    """Hysteresis: leave the committed route only if the alternative beats it by ``change_course_threshold_usd`` in
    the terms of the active objective (sunk costs excluded, since only remaining legs are compared):
    fastest = hours saved x the cost of a delayed hour; cheapest = expenses (no lateness penalty); balanced and
    safest = all-in cost incl. time. Cheapest, balanced and safest also count the carbon shadow price, because the
    route search does: otherwise a cleaner route the objective prefers could never clear the threshold.
    'Safest' also switches on a clear risk reduction."""
    cfg = ctx.config
    if cfg.preference == "fastest":
        gain = (hold.hours - alt.hours) * ctx.agent.delay_penalty_usd_per_hr
    elif cfg.preference == "cheapest":
        gain = (hold.total_usd - hold.time_usd + hold.carbon_usd) - (alt.total_usd - alt.time_usd + alt.carbon_usd)
    else:
        gain = (hold.total_usd + hold.carbon_usd) - (alt.total_usd + alt.carbon_usd)
    if gain >= cfg.change_course_threshold_usd:
        return True
    return cfg.preference == "safest" and hold.risk_index - alt.risk_index >= SAFEST_SWITCH_RISK_POINTS


def _change_headline(G: nx.Graph, pos: Position, new: RouteSummary, hold: RouteSummary | None) -> str:
    """"CHANGE COURSE NOW: turning north from off X, saves 9.1 d and $1.4M vs continuing via Y." One line, so a
    reader sees the decision and the reason before the detail. ``hold`` is None when staying is no longer possible."""
    name = lambda n: G.nodes[n]["name"]
    turning = bool(new.legs and new.legs[0].turn)
    off = name(pos.node) if pos.node else name(pos.x if pos.f < 0.5 else pos.y)
    place = off[0].lower() + off[1:] if off.startswith("Off ") else f"off {off}"
    first = next((l for l in new.legs if l.kind == "sail"), None)
    if turning and first:
        heading = _compass(bearing_deg(*node_ll(G, first.x), *node_ll(G, first.y)))
        action = f"turning {heading} from {place}"
    else:
        action = f"changing course toward {name(first.y)}" if first else "changing course"
    if hold is None:
        gain, verdict = "is the only way on", "because the committed route is no longer possible"
    else:
        dd, dc = (hold.hours - new.hours) / 24, (hold.total_usd - new.total_usd) / 1e6
        gain = (f"saves {dd:.1f} d and ${dc:.2f}M" if dd >= 0 and dc >= 0
                else f"changes the voyage by {-dd:+.1f} d and {-dc:+.2f}M USD")
        verdict = f"vs continuing via {hold.corridor}"
    return f"CHANGE COURSE NOW: {action}, {gain} {verdict}. "


def _compass(deg: float) -> str:
    return ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")[
        int(((deg % 360) + 22.5) // 45) % 8]


def _same(a: RouteSummary, b: RouteSummary) -> bool:
    key = lambda s: [(l.x, l.y) for l in s.legs if l.kind == "sail"]
    return key(a) == key(b)


def explain_choice(ctx: Context, best: Candidate, cands: list[Candidate]) -> str:
    pref = ctx.config.preference
    s = best.summary
    others = [c for c in cands if c is not best and c.summary.feasible]
    if not others:
        return f"It is the only feasible route ({PREFERENCES[pref]} objective)."
    parts = []
    for o in others[:2]:
        os_ = o.summary
        dd = (s.hours - os_.hours) / 24
        dc = (s.total_usd - os_.total_usd) / 1e6
        dr = s.risk_index - os_.risk_index
        parts.append(f"vs {o.name}: {dd:+.1f} d, {dc:+.2f}M USD, risk {dr:+.0f}")
    return f"Chosen under the {PREFERENCES[pref]} objective ({PREFERENCE_HELP[pref]}) " + "; ".join(parts) + "."


def projected(v: Voyage, G: nx.Graph) -> dict:
    """Totals for the whole voyage (sailed + remaining) and for the remaining part alone."""
    done, ahead, _ = split_at(v, G)
    tot = lambda legs, k: sum(getattr(l, k) for l in legs)
    out = {k: tot(v.legs, k) for k in ("nm", "fuel_t", "fuel_usd", "toll_usd", "war_usd", "time_usd", "eca_usd",
                                       "ets_usd", "co2_t", "carbon_usd")}
    out["total_usd"] = sum(l.total_usd for l in v.legs)
    out["remaining_usd"] = sum(l.total_usd for l in ahead)
    out["spent_usd"] = sum(l.total_usd for l in done)
    out["remaining_fuel_t"] = tot(ahead, "fuel_t")
    out["risk_index"] = 100.0 * route_risk(v.legs).total
    out["remaining_risk"] = route_risk(ahead)
    out["remaining_risk_index"] = 100.0 * out["remaining_risk"].total
    out["eta"] = calendar(v, v.total_hours)
    return out


def waypoints(v: Voyage, G: nx.Graph) -> list[dict]:
    """Named waypoints on the itinerary with the voyage time the ship reaches them (for click-to-seek)."""
    out, acc = [], 0.0
    for l in v.legs:
        acc += l.hours
        if l.kind == "sail" and l.f1 >= 1.0 - 1e-9:
            lat, lon = node_ll(G, l.y)
            out.append({"node": l.y, "name": G.nodes[l.y]["name"], "lat": lat, "lon": lon, "t": acc})
    return out


def remaining_samples(v: Voyage, G: nx.Graph, every_nm: float = 900.0, limit: int = 8) -> list[dict]:
    """Points on the remaining route every ~``every_nm`` with simulated arrival times (for weather sampling)."""
    st = ship_state(v, G)
    pts = [{"lat": st.lat, "lon": st.lon, "t": v.t, "label": "Ship position"}]
    _, ahead, _ = split_at(v, G)
    acc_t, acc_nm, next_at = v.t, 0.0, every_nm
    for l in ahead:
        if l.kind == "hold":
            acc_t += l.hours
            continue
        start_t = acc_t + l.wait_h
        while l.nm > 0 and acc_nm + l.nm >= next_at and len(pts) < limit:
            p = (next_at - acc_nm) / l.nm
            lat, lon = point_on_lane(G, l.x, l.y, l.f0 + (l.f1 - l.f0) * p)
            pts.append({"lat": lat, "lon": lon, "t": start_t + l.sea_h * p,
                        "label": f"Near {G.nodes[l.y]['name'] if p > 0.5 else G.nodes[l.x]['name']}"})
            next_at += every_nm
        acc_nm += l.nm
        acc_t += l.hours
    return pts


def route_signature(v: Voyage) -> str:
    """Identity of the itinerary (endpoints, departure, lanes and their sailed fractions). Changes on every reroute,
    reset or new voyage; used to decide whether cached route-specific data (e.g. a weather assessment) still applies."""
    lanes = ";".join(f"{l.x}-{l.y}:{l.f0:.4f}-{l.f1:.4f}" for l in v.legs if l.kind == "sail")
    return f"{v.config.origin}>{v.config.destination}@{v.config.departure.isoformat()}|{v.status == 'blocked'}|{lanes}"


def vessel_choices() -> dict[str, str]:
    return {k: v.label for k, v in VESSELS.items()}


def owner_choices(vessel_key: str) -> dict[str, str]:
    return {k: CARGO_OWNERS[k].label for k in VESSEL_OWNERS[vessel_key]}
