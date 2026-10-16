"""
routing_engine.py - agent-aware shortest paths over the maritime graph.

For every lane (edge) the engine computes, for a given agent:

    objective = fuel_tonnes(ML) * fuel_price
              + toll_weight * toll_cost
              + geopolitical_risk * insurance_sensitivity * days_exposed
              + travel_hours * delay_penalty_per_hour

and runs Dijkstra (or A*) on that weight. Each agent therefore sees a *different* graph: the same Suez
Canal is a bargain to an Apple container ship and a toll trap to an IKEA one.

Two kinds of numbers are kept apart on purpose:
  * ``objective``  - what the agent optimises (toll_weight applied; used only to choose the route)
  * ``total_usd``  - the plain all-in dollars of the voyage (fuel + tolls + war-risk + time cost)

Feasibility is handled in two passes. Pass 1 is strict: closed lanes, draft limits, storm-closed lanes and
lanes whose insurance premium exceeds the agent's walk-away multiple are removed. If that leaves no route,
pass 2 relaxes the *voluntary* constraints (storm closure, premium limit) and flags the voyage as
"forced through". Hard closures (blockades, blocked canals, draft limits) are never relaxed - if they cut
the network the answer is honestly "no feasible route".
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from agents import RoutingAgent, premium_multiple
from events import stress_edge
from data_engine import CHOKEPOINTS, WEATHER_BLOCK_THRESHOLD, haversine_nm
from economics import CO2_PER_T_FUEL, ECA_FUEL_PREMIUM_USD_PER_T, eca_fraction
import sklearn

from ml_models import FEATURES, MODEL_PATH, FuelModel, true_fuel_tonnes_per_day

DEFAULT_FUEL_PRICE_USD_PER_T = 620.0     # VLSFO, illustrative
OVERLAND_SPEED_KMH = 50.0
SPEED_LOSS_WEATHER = 0.5                 # speed over ground = speed * (1 - 0.5 * weather^2)
MAJOR_CORRIDORS = {"Strait of Hormuz", "Bab-el-Mandeb", "Suez Canal", "Strait of Malacca", "Panama Canal",
                   "Cape of Good Hope", "Cape Horn"}
Algorithm = str                          # "dijkstra" | "astar"


# --------------------------------------------------------------------------------------------------
# Fuel model: load the trained pickle if it exists, otherwise fall back to a physics formula
# --------------------------------------------------------------------------------------------------
def load_pickled_model(path=MODEL_PATH) -> FuelModel | None:
    """Unpickle ``models/fuel_model.pkl``. Returns None if it is missing, unreadable or from another sklearn."""
    try:
        with open(path, "rb") as fh:
            model = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(model, FuelModel) or model.sklearn_version != sklearn.__version__:
        return None
    try:                                     # tiny per-lane batches: one thread beats spinning up joblib workers
        model.pipeline.set_params(model__n_jobs=1)
    except (ValueError, AttributeError):
        pass
    return model


FUEL_MODEL: FuelModel | None = load_pickled_model()      # None -> physics fallback (see below)


def build_feature_matrix(vessel_type, speed, draft, cargo, wind, wave) -> pd.DataFrame:
    """Feature matrix in the column order the model was trained on.

    Columns: vessel_type, speed_kn, draft_m, cargo_weight_t, wind_speed_kn, wave_height_m. The vessel type is
    passed as a string and one-hot encoded inside the model's pipeline, so no manual encoding is needed.
    (Cargo weight is a sixth input beyond the five in the brief: it drives displacement and so fuel burn.)
    """
    speed, draft, cargo, wind, wave = (np.atleast_1d(np.asarray(a, dtype=float)).ravel()
                                       for a in np.broadcast_arrays(speed, draft, cargo, wind, wave))
    return pd.DataFrame({"vessel_type": [vessel_type] * len(speed), "speed_kn": speed, "draft_m": draft,
                         "cargo_weight_t": cargo, "wind_speed_kn": wind, "wave_height_m": wave})[FEATURES]


def fallback_tonnes_per_day(vessel_type, speed, draft, cargo, wind, wave) -> np.ndarray:
    """Closed-form estimate used when no trained model is available: cubic speed law x displacement^(2/3)
    x weather factor + hotel load. This is the same physics the training data was generated from."""
    speed, _, cargo, wind, wave = (np.atleast_1d(np.asarray(a, dtype=float)).ravel()
                                   for a in np.broadcast_arrays(speed, draft, cargo, wind, wave))
    return true_fuel_tonnes_per_day(np.array([vessel_type] * len(speed)), speed, cargo, wind, wave)


def predict_tonnes_per_day(model: FuelModel | None, vessel_type, speed, draft, cargo, wind, wave) -> np.ndarray:
    """ML prediction (tonnes of fuel per day at sea; divide by 24 for hourly burn) with a robust fallback."""
    model = model or FUEL_MODEL
    if model is not None:
        try:
            out = np.asarray(model.pipeline.predict(build_feature_matrix(vessel_type, speed, draft, cargo, wind, wave)))
            if np.all(np.isfinite(out)):
                return out
        except Exception:
            pass
    return fallback_tonnes_per_day(vessel_type, speed, draft, cargo, wind, wave)


def sea_state(weather_risk) -> tuple[np.ndarray, np.ndarray]:
    """Map a 0-1 weather risk to (wind knots, wave height m) fed to the fuel model."""
    w = np.asarray(weather_risk, dtype=float)
    return 8.0 + 45.0 * w, 0.8 + 7.5 * w ** 2


# --------------------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------------------
@dataclass
class EdgeCost:
    u: str
    v: str
    mode: str
    nm: float
    hours: float
    speed_kn: float
    fuel_t: float
    fuel_usd: float
    toll_usd: float
    war_usd: float
    time_usd: float
    objective: float
    hard_block: str | None = None        # never relaxed
    soft_block: str | None = None        # relaxed in the "forced through" pass
    co2_t: float = 0.0                   # fuel_t x 3.114
    eca_usd: float = 0.0                 # low-sulphur fuel premium for the share of the lane inside an ECA (cash)
    ets_usd: float = 0.0                 # EU ETS allowances for this lane's CO2 (cash)
    carbon_usd: float = 0.0              # carbon shadow price x CO2 (decision weight, NOT cash)

    @property
    def total_usd(self) -> float:
        """Cash cost of the lane (the carbon shadow price is excluded on purpose)."""
        return self.fuel_usd + self.toll_usd + self.war_usd + self.time_usd + self.eca_usd + self.ets_usd


@dataclass
class RouteResult:
    agent: str
    origin: str
    destination: str
    feasible: bool
    path: list = field(default_factory=list)
    path_names: list = field(default_factory=list)
    coords: list = field(default_factory=list)          # [(lat, lon)] polyline for the map
    distance_nm: float = 0.0
    hours: float = 0.0
    fuel_t: float = 0.0
    fuel_usd: float = 0.0
    toll_usd: float = 0.0
    war_usd: float = 0.0
    time_usd: float = 0.0
    total_usd: float = 0.0
    objective: float = 0.0
    avg_speed_kn: float = 0.0
    chokepoints: list = field(default_factory=list)
    overland: bool = False
    forced_through: bool = False
    notes: list = field(default_factory=list)
    reason: str = ""

    @property
    def days(self) -> float:
        return self.hours / 24.0


@dataclass
class Comparison:
    """Calm baseline vs stressed network for one agent."""
    agent: RoutingAgent
    calm: RouteResult
    stressed: RouteResult
    hold_course: RouteResult | None          # calm route re-priced under the events (None if now impossible)
    fuel_saved_t: float | None
    fuel_saved_usd: float | None
    tolls_avoided_usd: float | None
    cost_vs_hold_usd: float | None
    rerouted: bool


def price_edge_batch(items, agent: RoutingAgent, fuel_price: float, model: FuelModel | None = None,
                     econ=None, ets_active: bool = False) -> dict[frozenset, EdgeCost]:
    """Price lanes for one agent. ``items`` is an iterable of (u, v, edge_data); one ML call per speed option.

    objective = fuel_t * fuel_price + toll_weight * toll + geopolitical_risk * insurance_sensitivity * days
                + hours * delay_penalty * time_value_mult + ECA fuel switch + EU ETS + carbon_price * CO2

    ``econ`` (economics.Economics) switches on the factors after the first four; without it the lane is priced
    exactly as before (the stress-test tab). The speed of each lane is the option with the lowest objective.
    """
    tvm = econ.time_value_mult if econ else 1.0
    carbon_price = econ.carbon_usd_per_t if econ else 0.0
    ets_rate = econ.eua_usd_per_t * econ.ets_coverage_pct / 100.0 if (econ and ets_active) else 0.0
    edges = list(items)
    sea = [(u, v, d) for u, v, d in edges if d["mode"] != "overland"]
    out: dict[frozenset, EdgeCost] = {}

    if sea:
        nm = np.array([d["base_distance_nm"] for *_, d in sea])
        wx = np.array([d["weather_risk"] for *_, d in sea])
        geo = np.array([d["geopolitical_risk"] for *_, d in sea])
        cap = np.array([d["speed_cap_kn"] or np.inf for *_, d in sea])
        wait = np.array([d["wait_hours"] for *_, d in sea])
        toll = np.array([d["tolls"].get(agent.vessel_type, 0.0) * d["toll_multiplier"] + d["flat_cost_usd"]
                         for *_, d in sea])
        wind, wave = sea_state(wx)
        slow = 1.0 - SPEED_LOSS_WEATHER * wx ** 2
        eca = (np.array([eca_fraction(d["sample_lat"], d["sample_lon"]) for *_, d in sea]) if econ
               else np.zeros(len(sea)))

        best = None
        for speed in agent.speed_options_kn:
            stw = np.minimum(speed, cap)
            sea_h = nm / np.maximum(stw * slow, 0.5)
            hours = sea_h + wait
            tpd = predict_tonnes_per_day(model, agent.vessel_type, stw, agent.draft_m, agent.cargo_weight_t, wind, wave)
            fuel_t = tpd * sea_h / 24.0
            fuel_usd = fuel_t * fuel_price
            war = geo * agent.insurance_sensitivity_usd * hours / 24.0
            time_usd = hours * agent.delay_penalty_usd_per_hr
            co2 = fuel_t * CO2_PER_T_FUEL
            eca_usd = fuel_t * eca * ECA_FUEL_PREMIUM_USD_PER_T
            ets_usd = co2 * ets_rate
            carbon_usd = co2 * carbon_price
            obj = fuel_usd + agent.toll_weight * toll + war + time_usd * tvm + eca_usd + ets_usd + carbon_usd
            cur = dict(stw=stw, hours=hours, fuel_t=fuel_t, fuel_usd=fuel_usd, war=war, time=time_usd, obj=obj,
                       co2=co2, eca=eca_usd, ets=ets_usd, carbon=carbon_usd)
            if best is None:
                best = cur
            else:
                better = cur["obj"] < best["obj"]
                for k in best:
                    best[k] = np.where(better, cur[k], best[k])

        for i, (u, v, d) in enumerate(sea):
            hard = soft = None
            if d["blocked"]:
                hard = "closed"
            elif d["max_draft_m"] is not None and agent.draft_m > d["max_draft_m"]:
                hard = f"draft {agent.draft_m:g} m > {d['max_draft_m']:g} m limit"
            pm = premium_multiple(d["geopolitical_peak"])
            if d["weather_peak"] >= WEATHER_BLOCK_THRESHOLD:
                soft = "storm"
            elif pm > agent.max_premium_multiple:
                soft = f"insurance {pm:.1f}x"
            out[frozenset((u, v))] = EdgeCost(
                u, v, d["mode"], float(nm[i]), float(best["hours"][i]), float(best["stw"][i]),
                float(best["fuel_t"][i]), float(best["fuel_usd"][i]), float(toll[i]), float(best["war"][i]),
                float(best["time"][i]), float(best["obj"][i]), hard, soft,
                co2_t=float(best["co2"][i]), eca_usd=float(best["eca"][i]), ets_usd=float(best["ets"][i]),
                carbon_usd=float(best["carbon"][i]))

    for u, v, d in edges:
        if d["mode"] != "overland":
            continue
        hours = d["km"] / OVERLAND_SPEED_KMH + d["wait_hours"]
        fee = d["flat_cost_usd"]
        time_usd = hours * agent.delay_penalty_usd_per_hr
        out[frozenset((u, v))] = EdgeCost(
            u, v, "overland", d["km"] / 1.852, hours, 0.0, 0.0, 0.0, fee, 0.0, time_usd,
            agent.toll_weight * fee + time_usd, "closed" if d["blocked"] else None, None)
    return out



# --------------------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------------------
class RoutingEngine:
    def __init__(self, fuel_model: FuelModel | None = None,
                 fuel_price_usd_per_t: float = DEFAULT_FUEL_PRICE_USD_PER_T):
        self.fuel_model = fuel_model or FUEL_MODEL           # None -> physics fallback
        self.fuel_price = float(fuel_price_usd_per_t)
        # Tiny batches: a single-threaded forest is faster than spinning up workers.
        try:
            self.fuel_model.pipeline.set_params(model__n_jobs=1)
        except (ValueError, AttributeError):
            pass

    # ---- pricing every lane for one agent -------------------------------------------------------
    def price_edges(self, G: nx.Graph, agent: RoutingAgent) -> dict[frozenset, EdgeCost]:
        return price_edge_batch(G.edges(data=True), agent, self.fuel_price, self.fuel_model)

    # ---- routing --------------------------------------------------------------------------------
    def route(self, G: nx.Graph, agent: RoutingAgent, origin: str, destination: str,
              algorithm: Algorithm = "dijkstra", costs: dict | None = None) -> RouteResult:
        costs = costs or self.price_edges(G, agent)
        res = RouteResult(agent=agent.name, origin=origin, destination=destination, feasible=False)
        if origin == destination:
            res.reason = "Origin and destination are the same port."
            return res

        for relaxed in (False, True):
            H = nx.Graph()
            H.add_nodes_from(G.nodes)
            for key, c in costs.items():
                if c.hard_block or (c.soft_block and not relaxed):
                    continue
                H.add_edge(c.u, c.v, objective=c.objective)
            if not nx.has_path(H, origin, destination):
                continue
            if algorithm == "astar":
                path = nx.astar_path(H, origin, destination, heuristic=self._heuristic(G, costs, destination),
                                     weight="objective")
            else:
                path = nx.dijkstra_path(H, origin, destination, weight="objective")
            res = self._summarise(G, agent, path, costs, forced=relaxed)
            return res

        hard = [c for c in costs.values() if c.hard_block]
        res.reason = ("No feasible sea route: closures/draft limits cut the network between "
                      f"{G.nodes[origin]['name']} and {G.nodes[destination]['name']}.")
        res.notes = sorted({f"{c.u}-{c.v}: {c.hard_block}" for c in hard})[:6]
        return res

    @staticmethod
    def _heuristic(G, costs, destination):
        """Admissible A* heuristic: cheapest observed $ per great-circle nm, times the remaining distance."""
        ratios = []
        for key, c in costs.items():
            if c.hard_block:
                continue
            gc = haversine_nm(G.nodes[c.u]["lat"], G.nodes[c.u]["lon"], G.nodes[c.v]["lat"], G.nodes[c.v]["lon"])
            if gc > 1e-6:
                ratios.append(c.objective / gc)
        r = min(ratios) if ratios else 0.0
        dlat, dlon = G.nodes[destination]["lat"], G.nodes[destination]["lon"]

        def h(n, _t):
            return r * haversine_nm(G.nodes[n]["lat"], G.nodes[n]["lon"], dlat, dlon)
        return h

    # ---- summarising a path ---------------------------------------------------------------------
    def evaluate_path(self, G: nx.Graph, agent: RoutingAgent, path: list, costs: dict | None = None) -> RouteResult:
        """Re-price a fixed path on (possibly stressed) graph G. Infeasible only on hard closures."""
        costs = costs or self.price_edges(G, agent)
        for a, b in zip(path, path[1:]):
            c = costs.get(frozenset((a, b)))
            if c is None or c.hard_block:
                res = RouteResult(agent.name, path[0], path[-1], False, reason=f"Lane {a}-{b} is closed.")
                return res
        return self._summarise(G, agent, path, costs, forced=False)

    @staticmethod
    def _summarise(G, agent, path, costs, forced: bool) -> RouteResult:
        res = RouteResult(agent=agent.name, origin=path[0], destination=path[-1], feasible=True, path=list(path),
                          path_names=[G.nodes[n]["name"] for n in path], forced_through=forced)
        seen: list[str] = []
        violated: set[str] = set()
        coords: list[tuple] = []
        for n in path:
            ck = G.nodes[n]["chokepoint"]
            if ck and ck not in seen:
                seen.append(ck)
        for a, b in zip(path, path[1:]):
            c = costs[frozenset((a, b))]
            d = G.edges[a, b]
            res.distance_nm += c.nm
            res.hours += c.hours
            res.fuel_t += c.fuel_t
            res.fuel_usd += c.fuel_usd
            res.toll_usd += c.toll_usd
            res.war_usd += c.war_usd
            res.time_usd += c.time_usd
            res.objective += c.objective
            if c.mode == "overland":
                res.overland = True
            if d["chokepoint"] and d["chokepoint"] not in seen:
                seen.append(d["chokepoint"])
            if c.soft_block and forced:
                violated.add(c.soft_block)
            la, lo = np.asarray(d["sample_lat"]), np.asarray(d["sample_lon"])
            if G.nodes[a]["lat"] != la[0] or G.nodes[a]["lon"] != lo[0]:
                la, lo = la[::-1], lo[::-1]
            coords.extend(zip(la.tolist(), lo.tolist()))
        res.coords = coords
        res.total_usd = res.fuel_usd + res.toll_usd + res.war_usd + res.time_usd
        sea_nm = sum(costs[frozenset((a, b))].nm for a, b in zip(path, path[1:])
                     if costs[frozenset((a, b))].mode != "overland")
        res.avg_speed_kn = sea_nm / max(res.hours, 1e-9)
        res.chokepoints = [CHOKEPOINTS[k] for k in seen if k in CHOKEPOINTS]
        if forced:
            res.notes.append("Forced through: no route avoids " + ", ".join(sorted(violated)) + ".")
        if res.overland:
            res.notes.append("Overland transfer between ports (cargo re-loaded outside the blockade).")
        return res

    # ---- calm vs stressed -----------------------------------------------------------------------
    def compare(self, G_calm: nx.Graph, G_stressed: nx.Graph, agent: RoutingAgent, origin: str, destination: str,
                algorithm: Algorithm = "dijkstra") -> Comparison:
        calm = self.route(G_calm, agent, origin, destination, algorithm)
        stressed = self.route(G_stressed, agent, origin, destination, algorithm)
        hold = self.evaluate_path(G_stressed, agent, calm.path) if calm.feasible else None
        if hold is not None and not hold.feasible:
            hold = None
        fuel_t = fuel_usd = tolls = cost = None
        if hold is not None and stressed.feasible:
            fuel_t = hold.fuel_t - stressed.fuel_t
            fuel_usd = hold.fuel_usd - stressed.fuel_usd
            tolls = hold.toll_usd - stressed.toll_usd
            cost = hold.total_usd - stressed.total_usd
        return Comparison(agent, calm, stressed, hold, fuel_t, fuel_usd, tolls, cost,
                          rerouted=bool(calm.feasible and stressed.feasible and calm.path != stressed.path))


# --------------------------------------------------------------------------------------------------
# Single-lane cost + Dijkstra wired directly to it
# --------------------------------------------------------------------------------------------------
def calculate_edge_cost(u, v, edge_data: dict, vessel_agent: RoutingAgent, current_fuel_price: float,
                        global_events: dict | None = None, fuel_model: FuelModel | None = None) -> EdgeCost:
    """Cost of sailing ONE lane for one agent.

        objective = ML_fuel_tonnes * fuel_price + toll_weight * toll
                    + geopolitical_risk * insurance_sensitivity * days + travel_hours * delay_penalty

    * Fuel comes from the pickled random forest (one call covering all of the agent's speed options; the
      cheapest speed wins) or, if no model is available, from the physics fallback.
    * ``global_events`` ({"hormuz_blockade": True, ...}) is applied to this lane on the fly unless
      ``edge_data`` already comes from ``apply_events`` (flagged ``events_applied``).
    * The returned EdgeCost carries ``hard_block`` (closed / draft limit) and ``soft_block`` (storm / insurance
      above the agent's walk-away multiple) so callers can decide what is passable; ``objective`` is the weight.
    """
    if global_events is not None and not edge_data.get("events_applied"):
        edge_data = stress_edge(edge_data, global_events)
    return price_edge_batch([(u, v, edge_data)], vessel_agent, current_fuel_price, fuel_model)[frozenset((u, v))]


def find_optimal_route(G: nx.Graph, vessel_agent: RoutingAgent, origin: str, destination: str,
                       current_fuel_price: float = DEFAULT_FUEL_PRICE_USD_PER_T, global_events: dict | None = None,
                       fuel_model: FuelModel | None = None) -> RouteResult:
    """Dijkstra where every edge weight comes from ``calculate_edge_cost`` for this agent.

    Pass 1 skips closed lanes and lanes the agent refuses (storms, insurance above its limit); if that
    disconnects the pair, pass 2 allows the refused lanes and marks the voyage "forced through".
    Closed lanes and draft limits are never allowed.
    """
    cache: dict[frozenset, EdgeCost] = {}

    def cost_of(u, v, d) -> EdgeCost:
        key = frozenset((u, v))
        if key not in cache:
            cache[key] = calculate_edge_cost(u, v, d, vessel_agent, current_fuel_price, global_events, fuel_model)
        return cache[key]

    for relaxed in (False, True):
        def weight(u, v, d, _relaxed=relaxed):
            c = cost_of(u, v, d)
            if c.hard_block or (c.soft_block and not _relaxed):
                return None                                  # networkx treats None as "edge not usable"
            return c.objective
        try:
            path = nx.dijkstra_path(G, origin, destination, weight=weight)
        except nx.NetworkXNoPath:
            continue
        return RoutingEngine._summarise(G, vessel_agent, path, cache, forced=relaxed)
    return RouteResult(agent=vessel_agent.name, origin=origin, destination=destination, feasible=False,
                       reason="No feasible sea route: closures/draft limits cut the network.")


# --------------------------------------------------------------------------------------------------
# Presentation helpers (used by app.py and the CLI demo)
# --------------------------------------------------------------------------------------------------
def route_label(res: RouteResult) -> str:
    """Compact path label: corridor name rather than every waypoint."""
    if not res.feasible:
        return "No feasible route"
    names = [n for n in res.chokepoints if n in MAJOR_CORRIDORS]
    label = " > ".join(names) if names else "Direct"
    return label + (" (forced)" if res.forced_through else "")


def comparison_frame(comps: list[Comparison]) -> pd.DataFrame:
    """Side-by-side table: one column per agent, one row per metric."""
    cols = {}
    for c in comps:
        s, k = c.stressed, c.calm
        col = {
            "Route": route_label(s),
            "Rerouted vs calm": "Yes" if c.rerouted else ("No" if s.feasible else "-"),
            "Distance (nm)": f"{s.distance_nm:,.0f}" if s.feasible else "-",
            "Travel time (days)": f"{s.days:.1f}" if s.feasible else "-",
            "Avg speed (kn)": f"{s.avg_speed_kn:.1f}" if s.feasible else "-",
            "Fuel burned (t)": f"{s.fuel_t:,.0f}" if s.feasible else "-",
            "Fuel cost": _usd(s.fuel_usd) if s.feasible else "-",
            "Tolls & fees": _usd(s.toll_usd) if s.feasible else "-",
            "War-risk insurance": _usd(s.war_usd) if s.feasible else "-",
            "Time cost": _usd(s.time_usd) if s.feasible else "-",
            "All-in cost": _usd(s.total_usd) if s.feasible else "No feasible route",
            "vs holding course": _signed(c.cost_vs_hold_usd, "saved", "extra") if s.feasible else "-",
            "Fuel saved (t)": _signed_num(c.fuel_saved_t) if s.feasible else "-",
            "Tolls avoided": _usd(c.tolls_avoided_usd) if c.tolls_avoided_usd is not None and s.feasible else "-",
            "Stress vs calm": (_signed(k.total_usd - s.total_usd, "cheaper", "more expensive")
                               if s.feasible and k.feasible else "-"),
        }
        cols[c.agent.name] = col
    return pd.DataFrame(cols)


def _usd(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    a = abs(x)
    if a >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if a >= 1e3:
        return f"${x / 1e3:,.0f}k"
    return f"${x:,.0f}"


def _signed(x, pos: str, neg: str) -> str:
    if x is None:
        return "n/a (calm route now closed)"
    if abs(x) < 500:
        return "same"
    return f"{_usd(abs(x))} {pos if x > 0 else neg}"


def _signed_num(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:+,.0f}" if abs(x) >= 0.5 else "0"


if __name__ == "__main__":
    from agents import DEFAULT_CONTAINER, DEFAULT_TANKER
    from data_engine import build_maritime_graph
    from events import PRESETS, apply_events
    from ml_models import load_or_train_model

    eng = RoutingEngine(load_or_train_model())
    G0 = build_maritime_graph()
    pd.set_option("display.width", 200)
    for name, preset in PRESETS.items():
        G1 = apply_events(G0, preset)
        comps = [eng.compare(G0, G1, DEFAULT_CONTAINER, "SHA", "RTM"), eng.compare(G0, G1, DEFAULT_TANKER, "DXB", "RTM")]
        print(f"\n=== {name} ===")
        print(comparison_frame(comps).to_string())
