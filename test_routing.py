"""
Core agent-behaviour tests. Run with either:

    pytest -v test_routing.py
    python test_routing.py
"""
try:
    import pytest  # noqa: F401
except ImportError:      # plain `python test_routing.py` works without pytest
    pytest = None

from agents import DEFAULT_CONTAINER, DEFAULT_TANKER
from data_engine import build_maritime_graph
from events import apply_events
from ml_models import train_fuel_model
from routing_engine import (FUEL_MODEL, calculate_edge_cost, fallback_tonnes_per_day, find_optimal_route,
                            predict_tonnes_per_day)

FUEL_PRICE = 620.0
HORMUZ_NODES = {"HS1", "HOR"}          # the strait itself: Gulf of Oman side -> Persian Gulf side


def _tanker_route(events):
    """Tanker Jebel Ali -> Rotterdam on the baseline graph with `events` applied lane by lane."""
    return find_optimal_route(build_maritime_graph(), DEFAULT_TANKER, "DXB", "RTM", FUEL_PRICE, events)


def test_tanker_routes_straight_through_hormuz_when_open():
    """Test Case 1: hormuz_blockade False -> the tanker sails through the Strait of Hormuz nodes."""
    route = _tanker_route({"hormuz_blockade": False})
    assert route.feasible
    assert HORMUZ_NODES <= set(route.path), route.path_names
    assert "Strait of Hormuz" in route.chokepoints
    assert not route.overland


def test_tanker_bypasses_hormuz_when_blockaded():
    """Test Case 2: hormuz_blockade True -> no Hormuz node on the path; the alternate track is taken."""
    open_route = _tanker_route({"hormuz_blockade": False})
    route = _tanker_route({"hormuz_blockade": True})
    assert route.feasible, route.reason
    assert not (HORMUZ_NODES & set(route.path)), route.path_names
    assert "Strait of Hormuz" not in route.chokepoints
    assert route.overland and "FJR" in route.path          # cargo trucked to Fujairah, outside the strait
    assert route.total_usd > open_route.total_usd          # the detour costs more


def test_blockaded_hormuz_lane_is_closed_for_every_agent():
    G = build_maritime_graph()
    for agent in (DEFAULT_TANKER, DEFAULT_CONTAINER):
        c = calculate_edge_cost("HS1", "HOR", G.edges["HS1", "HOR"], agent, FUEL_PRICE, {"hormuz_blockade": True})
        assert c.hard_block == "closed"
        ok = calculate_edge_cost("HS1", "HOR", G.edges["HS1", "HOR"], agent, FUEL_PRICE, {"hormuz_blockade": False})
        assert ok.hard_block is None


def test_tanker_pays_more_for_war_risk_than_container_on_same_lane():
    G = build_maritime_graph()
    e = G.edges["HS1", "HOR"]
    tanker = calculate_edge_cost("HS1", "HOR", e, DEFAULT_TANKER, FUEL_PRICE, {})
    box = calculate_edge_cost("HS1", "HOR", e, DEFAULT_CONTAINER, FUEL_PRICE, {})
    assert tanker.war_usd / tanker.hours > box.war_usd / box.hours


def test_pre_stressed_graph_gives_same_route_as_on_the_fly_events():
    events = {"hormuz_blockade": True}
    G1 = apply_events(build_maritime_graph(), events)
    assert find_optimal_route(G1, DEFAULT_TANKER, "DXB", "RTM", FUEL_PRICE).path == _tanker_route(events).path


def test_fallback_formula_is_close_to_trained_model():
    model = FUEL_MODEL or train_fuel_model(n_estimators=60)
    ml = predict_tonnes_per_day(model, "Container", 20.0, 12.0, 70_000, 15.0, 1.5)
    fb = fallback_tonnes_per_day("Container", 20.0, 12.0, 70_000, 15.0, 1.5)
    assert abs(ml[0] - fb[0]) / fb[0] < 0.15


def test_routing_works_without_any_model_file():
    """With no pickle (FUEL_MODEL is None) the physics fallback keeps the engine running."""
    import routing_engine
    saved = routing_engine.FUEL_MODEL
    routing_engine.FUEL_MODEL = None
    try:
        route = find_optimal_route(build_maritime_graph(), DEFAULT_CONTAINER, "SHA", "RTM", FUEL_PRICE)
        assert route.feasible and route.fuel_t > 0
    finally:
        routing_engine.FUEL_MODEL = saved


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
