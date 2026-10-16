"""
Phase 5B: hazard overlay - zones tinted by computed risk band, and port warnings (congested / unreachable).

    python -m pytest tests/test_phase5_hazards.py -v

"Too shallow for this ship" markers are deliberately absent: Tideline has no port-draft data and none is invented.
"""
import unittest

import folium_map as FM
import panels as P
import voyage as V
from data_engine import GEOPOLITICAL_ZONES
from ops import Ops

BANDS = {"Low", "Guarded", "Elevated", "High", "Severe"}


def _ctx(o):
    return V.build_context(o.cfg, o.settings, o.model)


def _zones(o):
    return {z["key"]: z for z in P.hazard_zones(_ctx(o).G, V.base_graph())}


class TestHazardZones(unittest.TestCase):
    def test_every_zone_is_reported_with_a_band(self):
        zones = _zones(Ops())
        self.assertEqual(set(zones), {z.key for z in GEOPOLITICAL_ZONES})
        for z in zones.values():
            self.assertIn(z["band"], BANDS)
            self.assertEqual(z["score"], round(100 * z["security"]))
            for k in ("label", "bounds", "base", "scenario", "tone"):
                self.assertIn(k, z)

    def test_calm_network_has_no_scenario_term(self):
        for z in _zones(Ops()).values():
            self.assertEqual(z["scenario"], 0.0, z["key"])
            self.assertAlmostEqual(z["security"], z["base"], places=9)

    def test_base_levels_follow_the_zone_baselines(self):
        zones = _zones(Ops())
        self.assertGreaterEqual(zones["hormuz"]["base"], 0.40)
        self.assertGreater(zones["hormuz"]["score"], zones["east_med"]["score"])
        self.assertIn(zones["hormuz"]["band"], {"Elevated", "High", "Severe"})

    def test_red_sea_escalation_raises_the_red_sea_zones_as_a_scenario_term(self):
        calm = _zones(Ops())
        o = Ops()
        o.set_disruption("red_sea_escalation", True, value=0.45)
        hot = _zones(o)
        for k in ("red_sea", "bab_el_mandeb", "gulf_of_aden"):
            self.assertGreater(hot[k]["scenario"], 0.0, k)
            self.assertGreater(hot[k]["score"], calm[k]["score"], k)
            self.assertAlmostEqual(hot[k]["base"], calm[k]["base"], places=9)
        self.assertEqual(hot["taiwan_strait"]["scenario"], 0.0)          # untouched elsewhere

    def test_hormuz_blockade_makes_the_gulf_severe(self):
        o = Ops()
        o.set_disruption("hormuz_closure", True)
        self.assertEqual(_zones(o)["persian_gulf"]["band"], "Severe")

    def test_tooltip_states_the_parts_and_that_it_is_simulated(self):
        o = Ops()
        o.set_disruption("red_sea_escalation", True, value=0.45)
        z = _zones(o)["bab_el_mandeb"]
        tip = P.hazard_tooltip(z)
        for needle in (z["label"], z["band"], "security", "base", "scenario", "SIMULATED"):
            self.assertIn(needle, tip)


class TestReachablePorts(unittest.TestCase):
    def test_every_port_is_reachable_in_calm_seas(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        reach = V.reachable_ports(_ctx(o), V.Position(node="RTM"))
        self.assertTrue({"RTM", "SHA", "SIN", "MUM", "DXB", "FJR", "NYC", "LAX", "HKG"} <= reach)

    def test_dubai_is_cut_off_when_hormuz_is_closed_but_fujairah_is_not(self):
        o = Ops()
        o.set_disruption("hormuz_closure", True)
        reach = V.reachable_ports(_ctx(o), V.Position(node="RTM"))
        self.assertNotIn("DXB", reach)
        self.assertIn("FJR", reach)

    def test_a_ship_inside_the_gulf_can_only_reach_gulf_ports(self):
        o = Ops()
        o.create_voyage("DXB", "RTM", vessel_key="tanker")
        o.advance_hours(6)
        o.set_disruption("hormuz_closure", True)
        _, _, pos = V.split_at(o.voyage, o.G)
        reach = V.reachable_ports(_ctx(o), pos)
        self.assertEqual(reach, {"DXB"})


class TestPortWarnings(unittest.TestCase):
    def test_none_in_calm_seas(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        self.assertEqual(P.port_warnings(_ctx(o), V.Position(node="RTM"), o.settings, o.G), [])

    def test_unreachable_port_is_flagged_with_the_reason(self):
        o = Ops()
        o.set_disruption("hormuz_closure", True)
        w = {x["port"]: x for x in P.port_warnings(_ctx(o), V.Position(node="RTM"), o.settings, o.G)}
        self.assertEqual(w["DXB"]["kind"], "unreachable")
        self.assertIn("no sea passage", w["DXB"]["detail"].lower())
        self.assertNotIn("FJR", w)

    def test_congested_port_is_flagged_with_the_extra_wait(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.set_congestion("SHA", 36)
        w = {x["port"]: x for x in P.port_warnings(_ctx(o), V.Position(node="RTM"), o.settings, o.G)}
        self.assertEqual(w["SHA"]["kind"], "congested")
        self.assertIn("36", w["SHA"]["detail"])
        self.assertIn("SIMULATED", w["SHA"]["detail"])

    def test_no_too_shallow_markers_exist(self):
        o = Ops()
        o.set_disruption("hormuz_closure", True)
        o.set_congestion("SHA", 12)
        kinds = {x["kind"] for x in P.port_warnings(_ctx(o), V.Position(node="RTM"), o.settings, o.G)}
        self.assertTrue(kinds <= {"congested", "unreachable"})


class TestOnTheMap(unittest.TestCase):
    def test_base_map_has_a_hazard_layer_and_a_layer_control(self):
        o = Ops()
        o.set_disruption("red_sea_escalation", True, value=0.45)
        m = FM.build_ops_base_map(_ctx(o).G, o.settings, hazards=P.hazard_zones(_ctx(o).G, V.base_graph()))
        html_ = m.get_root().render()
        self.assertIn("Hazard zones", html_)
        self.assertIn("Bab-el-Mandeb", html_)
        self.assertIn("control.layers", html_)

    def test_base_map_still_builds_without_hazards(self):
        o = Ops()
        FM.build_ops_base_map(_ctx(o).G, o.settings)

    def test_voyage_layer_draws_the_port_warnings(self):
        import folium
        o = Ops()
        o.set_disruption("hormuz_closure", True)
        o.set_congestion("SHA", 24)
        ctx = _ctx(o)
        warns = P.port_warnings(ctx, V.Position(node="RTM"), o.settings, o.G)
        s = V.ship_state(o.voyage, o.G)
        layer = FM.build_ops_layer(o.G, o.voyage, [], [], [], s, (), (), "#58aaff", port_warnings=warns)
        m = folium.Map()
        layer.add_to(m)
        html_ = m.get_root().render()
        self.assertIn("no sea passage", html_.lower())
        self.assertIn("congestion", html_.lower())


if __name__ == "__main__":
    unittest.main()
