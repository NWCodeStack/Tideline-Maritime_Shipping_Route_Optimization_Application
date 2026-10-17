"""Geometry and ML tests. Run: python -m unittest discover -s tests -v"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

import data_engine as de
import geo_assets as ga
from ml_models import HULL, generate_voyage_dataset, train_fuel_model


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = de.build_maritime_graph()

    def test_connected_and_hubs_present(self):
        self.assertTrue(nx.is_connected(self.G))
        for p in ("SHA", "RTM", "LAX", "SIN", "DXB"):
            self.assertEqual(self.G.nodes[p]["kind"], "port")

    def test_edge_attributes_in_range(self):
        for *_, d in self.G.edges(data=True):
            self.assertTrue(0 <= d["weather_risk"] <= 1 and 0 <= d["geopolitical_risk"] <= 1)
            self.assertGreaterEqual(d["base_distance_nm"], 0)

    def test_bounding_boxes_give_high_baseline_risk(self):
        ambient = de.AMBIENT_GEO_RISK
        hormuz = self.G.edges["HS1", "HOR"]
        red = self.G.edges["RS1", "RS2"]
        calm = self.G.edges["SIN", "MAL1"] if self.G.has_edge("SIN", "MAL1") else None
        self.assertGreater(hormuz["geopolitical_risk"], 0.3)
        self.assertGreater(red["geopolitical_risk"], 0.15)
        self.assertGreater(hormuz["geopolitical_risk"], 10 * ambient)
        self.assertTrue(calm is None or calm["geopolitical_risk"] < hormuz["geopolitical_risk"])

    def test_no_node_or_lane_on_land(self):
        ports = {n for n, d in self.G.nodes(data=True) if d["kind"] == "port"}
        for n, d in self.G.nodes(data=True):
            if d["kind"] != "port":
                self.assertFalse(ga.on_land(d["lat"], d["lon"]), f"{n} is on land")
        for u, v, d in self.G.edges(data=True):
            if d["mode"] != "sea":
                continue
            a, b = self.G.nodes[u], self.G.nodes[v]
            la, lo = de.interpolate_lane(a["lat"], a["lon"], b["lat"], b["lon"], max(4, int(d["base_distance_nm"] / 4)))
            for y, x in zip(la, lo, strict=False):
                near_port = any(math.hypot(y - self.G.nodes[p]["lat"], x - self.G.nodes[p]["lon"]) < 0.35
                                for p in (u, v) if p in ports)
                self.assertTrue(near_port or not ga.on_land(y, x), f"lane {u}-{v} crosses land near {y:.1f},{x:.1f}")


class FuelModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_voyage_dataset(5000, seed=1)
        cls.model = train_fuel_model(cls.df, seed=1, n_estimators=60)

    def test_dataset_shape(self):
        self.assertEqual(len(self.df), 5000)
        self.assertEqual(set(self.df.vessel_type), {"Container", "Tanker"})

    def test_forest_beats_linear_and_is_accurate(self):
        self.assertGreater(self.model.metrics["rf_r2"], 0.95)
        self.assertGreater(self.model.metrics["rf_r2"], self.model.metrics["linear_r2"])

    def test_speed_fuel_is_nonlinear(self):
        lo, hi = HULL["Container"]["speed"]
        f = self.model.speed_curve("Container", [lo + 1, (lo + hi) / 2, hi - 1], 12.0, 70_000)
        self.assertTrue(f[0] < f[1] < f[2])
        self.assertGreater(f[2] - f[1], f[1] - f[0])       # convex: each extra knot costs more


if __name__ == "__main__":
    unittest.main()
