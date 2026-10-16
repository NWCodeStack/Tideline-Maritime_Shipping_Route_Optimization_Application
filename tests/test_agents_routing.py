"""Agent, event and routing tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import DEFAULT_CONTAINER, DEFAULT_TANKER, make_agent, premium_multiple
from data_engine import build_maritime_graph
from events import PRESETS, apply_events
from ml_models import train_fuel_model
from routing_engine import RoutingEngine, comparison_frame


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G0 = build_maritime_graph()
        cls.eng = RoutingEngine(train_fuel_model(seed=3, n_estimators=60))
        cls.apple = DEFAULT_CONTAINER
        cls.ikea = make_agent("container", "low_margin")

    def test_calm_container_uses_suez(self):
        r = self.eng.route(self.G0, self.apple, "SHA", "RTM")
        self.assertTrue(r.feasible)
        self.assertIn("Suez Canal", r.chokepoints)
        self.assertEqual(r.toll_usd, 450_000)

    def test_events_do_not_mutate_baseline(self):
        before = self.G0.edges["SUZ", "PSD"]["toll_multiplier"]
        apply_events(self.G0, {"suez_toll_multiplier": 5})
        self.assertEqual(self.G0.edges["SUZ", "PSD"]["toll_multiplier"], before)

    def test_toll_x5_splits_the_agents(self):
        G1 = apply_events(self.G0, {"suez_toll_multiplier": 5})
        apple = self.eng.route(G1, self.apple, "SHA", "RTM")
        ikea = self.eng.route(G1, self.ikea, "SHA", "RTM")
        self.assertIn("Suez Canal", apple.chokepoints)           # time-critical: pays
        self.assertNotIn("Suez Canal", ikea.chokepoints)         # cost-sensitive: avoids
        self.assertEqual(ikea.toll_usd, 0)

    def test_comparison_reports_tolls_avoided(self):
        G1 = apply_events(self.G0, {"suez_toll_multiplier": 5})
        c = self.eng.compare(self.G0, G1, self.ikea, "SHA", "RTM")
        self.assertTrue(c.rerouted)
        self.assertAlmostEqual(c.tolls_avoided_usd, 450_000 * 5)
        self.assertGreater(c.cost_vs_hold_usd, 0)

    def test_hormuz_blockade_tanker_goes_overland(self):
        G1 = apply_events(self.G0, {"hormuz_blockade": True})
        r = self.eng.route(G1, DEFAULT_TANKER, "DXB", "RTM")
        self.assertTrue(r.feasible and r.overland)
        self.assertNotIn("Strait of Hormuz", r.chokepoints)

    def test_hard_closure_means_no_route(self):
        G1 = apply_events(self.G0, {"hormuz_blockade": True})
        for e in list(G1.edges("FJR", data=True)):
            if e[2]["mode"] == "overland":
                e[2]["blocked"] = True
        r = self.eng.route(G1, DEFAULT_TANKER, "DXB", "RTM")
        self.assertFalse(r.feasible)

    def test_astar_matches_dijkstra(self):
        for preset in PRESETS.values():
            G1 = apply_events(self.G0, preset)
            for ag, o in ((self.apple, "SHA"), (DEFAULT_TANKER, "DXB"), (self.ikea, "LAX")):
                a = self.eng.route(G1, ag, o, "RTM", "dijkstra")
                b = self.eng.route(G1, ag, o, "RTM", "astar")
                self.assertEqual(a.feasible, b.feasible)
                if a.feasible:
                    self.assertAlmostEqual(a.objective, b.objective, places=3)

    def test_premium_threshold_for_containers(self):
        self.assertGreater(premium_multiple(0.45), 3.0)
        self.assertLess(premium_multiple(0.40), 3.0)

    def test_tanker_more_insurance_sensitive(self):
        self.assertGreater(DEFAULT_TANKER.insurance_sensitivity_usd, DEFAULT_CONTAINER.insurance_sensitivity_usd)

    def test_comparison_frame(self):
        G1 = apply_events(self.G0, PRESETS["Perfect storm"])
        df = comparison_frame([self.eng.compare(self.G0, G1, self.apple, "SHA", "RTM"),
                               self.eng.compare(self.G0, G1, DEFAULT_TANKER, "DXB", "RTM")])
        self.assertEqual(df.shape[1], 2)
        self.assertIn("Tolls avoided", df.index)


if __name__ == "__main__":
    unittest.main()
