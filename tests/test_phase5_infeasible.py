"""
Phase 5A: honest infeasibility. When no sea passage exists the message says which port the ship CAN still reach.

    python -m pytest tests/test_phase5_infeasible.py -v
"""
import unittest

import networkx as nx

import voyage as V
from ops import Ops


def _blocked_rtm_dxb():
    o = Ops()
    o.set_disruption("hormuz_closure", True)
    msg = o.create_voyage("RTM", "DXB")
    return o, msg


class TestNearestReachablePort(unittest.TestCase):
    def test_dubai_behind_a_closed_hormuz_points_to_fujairah(self):
        o, msg = _blocked_rtm_dxb()
        self.assertEqual(o.voyage.status, "blocked")
        self.assertIn("Nearest port still reachable: Fujairah", o.voyage.block_reason)
        self.assertIn("Fujairah", msg)

    def test_distance_is_given_from_the_destination(self):
        o, _ = _blocked_rtm_dxb()
        self.assertRegex(o.voyage.block_reason, r"Fujairah, \d+ nm from Dubai")

    def test_the_destination_is_never_its_own_suggestion(self):
        o, _ = _blocked_rtm_dxb()
        self.assertNotIn("still reachable: Dubai", o.voyage.block_reason)

    def test_still_says_what_is_closed(self):
        o, _ = _blocked_rtm_dxb()
        self.assertIn("Closed:", o.voyage.block_reason)
        self.assertIn("Strait of Hormuz", o.voyage.block_reason)

    def test_ship_trapped_inside_the_gulf_is_told_where_it_can_shelter(self):
        o = Ops()
        o.create_voyage("DXB", "RTM", vessel_key="tanker")
        o.advance_hours(6)                                   # out of Dubai, still inside the Gulf
        o.set_disruption("hormuz_closure", True)
        self.assertEqual(o.voyage.status, "blocked")
        self.assertIn("Nearest port still reachable: Dubai", o.voyage.block_reason)

    def test_no_suggestion_when_a_route_exists(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.advance_hours(24)
        o.set_disruption("suez_closure", True)               # reroutes round the Cape; nothing is infeasible
        self.assertNotEqual(o.voyage.status, "blocked")
        self.assertNotIn("still reachable", o.voyage.decision)

    def test_route_from_reports_it_on_the_summary_reason(self):
        o, _ = _blocked_rtm_dxb()
        ctx = V.build_context(o.cfg, o.settings, o.model)
        s = V.route_from(ctx, V.Position(node="RTM"), "DXB", "balanced")
        self.assertFalse(s.feasible)
        self.assertIn("Fujairah", s.reason)

    def test_helper_returns_none_when_nothing_else_is_reachable(self):
        G = V.base_graph()
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes)                            # no lanes at all: only the start node is reachable
        self.assertIsNone(V.nearest_reachable_port(G, H, "RTM", "DXB"))

    def test_helper_prefers_the_port_closest_to_the_destination(self):
        G = V.base_graph()
        H = nx.DiGraph()
        H.add_edges_from([("RTM", "FJR"), ("RTM", "SHA"), ("RTM", "LAX")])
        node, nm = V.nearest_reachable_port(G, H, "RTM", "DXB")
        self.assertEqual(node, "FJR")
        self.assertLess(nm, 300)


if __name__ == "__main__":
    unittest.main()
