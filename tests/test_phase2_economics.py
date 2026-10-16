"""
Phase 2 (first cut): per-port fuel price, speed per leg, ECA / EU ETS / carbon lines, objective presets.

    python -m pytest tests/test_phase2_economics.py -v

Money that leaves the company (fuel, tolls, war-risk, time, ECA fuel switch, EU ETS) is "cash" and sums to
``total_usd``. The carbon shadow price is a decision weight only: it steers the optimiser but is never cash.
"""
import unittest

import economics as E
import voyage as V
from agents import VESSELS
from events import default_settings
from ops import Ops


def _plan(origin="RTM", dest="SHA", preference="balanced", **econ):
    cfg = V.VoyageConfig(origin=origin, destination=dest, preference=preference, econ=E.Economics(**econ))
    v = V.plan_voyage(cfg, default_settings())
    ctx = V.build_context(cfg, default_settings())
    return cfg, v, V.summarise(ctx, v.legs)


def _avg_speed(legs):
    sail = [l for l in legs if l.kind == "sail"]
    return sum(l.nm for l in sail) / sum(l.sea_h for l in sail)


class TestECA(unittest.TestCase):
    def test_lane_fraction_inside_and_outside(self):
        G = V.base_graph()
        d = G.edges["USH", "CHA"]
        self.assertGreater(E.eca_fraction(d["sample_lat"], d["sample_lon"]), 0.5)          # English Channel
        d = G.edges["SA", "NATL"]
        self.assertEqual(E.eca_fraction(d["sample_lat"], d["sample_lon"]), 0.0)            # mid-Atlantic

    def test_line_only_when_route_touches_an_eca(self):
        _, _, rtm_sha = _plan("RTM", "SHA")
        _, _, sin_sha = _plan("SIN", "SHA")
        self.assertGreater(rtm_sha.eca_usd, 0)
        self.assertEqual(sin_sha.eca_usd, 0)

    def test_cost_is_fuel_burned_inside_times_premium(self):
        _, v, s = _plan("RTM", "SHA")
        G = V.base_graph()
        inside_t = sum(l.fuel_t * E.eca_fraction(G.edges[l.x, l.y]["sample_lat"], G.edges[l.x, l.y]["sample_lon"])
                       for l in v.legs if l.kind == "sail")
        self.assertAlmostEqual(s.eca_usd, inside_t * E.ECA_FUEL_PREMIUM_USD_PER_T, delta=1.0)


class TestEUETS(unittest.TestCase):
    def test_line_only_for_voyages_calling_at_an_eu_port(self):
        self.assertGreater(_plan("RTM", "SHA")[2].ets_usd, 0)
        self.assertGreater(_plan("SHA", "RTM")[2].ets_usd, 0)
        self.assertEqual(_plan("SIN", "SHA")[2].ets_usd, 0)
        self.assertEqual(_plan("NYC", "LAX")[2].ets_usd, 0)

    def test_formula_and_user_controls(self):
        _, _, base = _plan("RTM", "SHA", eua_usd_per_t=70.0, ets_coverage_pct=50.0)
        self.assertAlmostEqual(base.ets_usd, base.co2_t * 0.5 * 70.0, delta=1.0)
        _, _, full = _plan("RTM", "SHA", eua_usd_per_t=70.0, ets_coverage_pct=100.0)
        self.assertAlmostEqual(full.ets_usd, 2 * base.ets_usd, delta=1.0)
        _, _, off = _plan("RTM", "SHA", ets_enabled=False)
        self.assertEqual(off.ets_usd, 0)

    def test_ets_is_cash(self):
        _, _, s = _plan("RTM", "SHA")
        self.assertAlmostEqual(s.total_usd, s.fuel_usd + s.toll_usd + s.war_usd + s.time_usd + s.eca_usd + s.ets_usd,
                               delta=1.0)


class TestCarbon(unittest.TestCase):
    def test_co2_from_fuel(self):
        _, _, s = _plan("RTM", "SHA")
        self.assertAlmostEqual(s.co2_t, s.fuel_t * E.CO2_PER_T_FUEL, delta=0.5)

    def test_shadow_price_is_a_line_but_not_cash(self):
        _, _, s = _plan("RTM", "SHA", carbon_usd_per_t=200.0)
        self.assertAlmostEqual(s.carbon_usd, s.co2_t * 200.0, delta=1.0)
        self.assertAlmostEqual(s.total_usd, s.fuel_usd + s.toll_usd + s.war_usd + s.time_usd + s.eca_usd + s.ets_usd,
                               delta=1.0)
        _, _, zero = _plan("RTM", "SHA")
        self.assertEqual(zero.carbon_usd, 0)


class TestSpeedPerLeg(unittest.TestCase):
    def test_every_sailing_leg_has_a_speed_the_ship_can_do(self):
        _, v, _ = _plan("RTM", "SHA")
        opts = VESSELS["container"].speed_options_kn
        for l in v.legs:
            if l.kind == "sail":
                self.assertGreater(l.speed_kn, 0)
                self.assertLessEqual(l.speed_kn, max(opts) + 1e-6)

    def test_greenest_steams_slower_than_fastest_on_the_same_route(self):
        _, fast, _ = _plan("RTM", "SHA", preference="fastest", time_value_mult=4.0)
        pref, changes = E.apply_preset("greenest")
        _, green, _ = _plan("RTM", "SHA", preference=pref, **changes)
        self.assertEqual(V.route_signature(fast).split("|")[-1], V.route_signature(green).split("|")[-1])
        self.assertLess(_avg_speed(green.legs), _avg_speed(fast.legs))

    def test_carbon_price_lowers_emissions(self):
        _, _, base = _plan("RTM", "SHA")
        _, _, green = _plan("RTM", "SHA", carbon_usd_per_t=200.0)
        self.assertLess(green.co2_t, base.co2_t)


class TestPresets(unittest.TestCase):
    def test_the_five_presets(self):
        self.assertEqual(set(E.PRESETS), {"balanced", "cheapest", "fastest", "safest", "greenest"})

    def test_apply_and_recognise(self):
        for key in E.PRESETS:
            pref, changes = E.apply_preset(key)
            self.assertEqual(E.preset_key(pref, E.Economics(**changes)), key)

    def test_moving_a_slider_switches_to_custom(self):
        pref, changes = E.apply_preset("greenest")
        changes = dict(changes, carbon_usd_per_t=changes["carbon_usd_per_t"] + 1)
        self.assertEqual(E.preset_key(pref, E.Economics(**changes)), "custom")

    def test_default_config_is_the_balanced_preset(self):
        cfg = V.VoyageConfig()
        self.assertEqual(E.preset_key(cfg.preference, cfg.econ), "balanced")

    def test_ops_apply_preset(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        msg = o.apply_preset("greenest")
        self.assertEqual(o.cfg.econ.carbon_usd_per_t, E.PRESETS["greenest"].carbon_usd_per_t)
        self.assertEqual(E.preset_key(o.cfg.preference, o.cfg.econ), "greenest")
        self.assertIn("Greenest", msg)
        self.assertIn("Unknown preset", o.apply_preset("nope"))

    def test_ops_slider_change_reads_as_custom_and_reprices(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.set_economics(carbon_usd_per_t=123.0)
        self.assertEqual(E.preset_key(o.cfg.preference, o.cfg.econ), "custom")
        self.assertGreater(V.projected(o.voyage, o.G)["carbon_usd"], 0)


class TestPerPortFuelPrice(unittest.TestCase):
    def test_global_price_is_the_default(self):
        cfg, _, s = _plan("RTM", "SHA")
        self.assertAlmostEqual(s.fuel_usd / s.fuel_t, cfg.fuel_price, places=2)

    def test_origin_port_price_and_override(self):
        _, _, s = _plan("RTM", "SHA", fuel_price_mode="origin_port")
        self.assertAlmostEqual(s.fuel_usd / s.fuel_t, E.PORT_FUEL_PRICES["RTM"]["usd_per_t"], places=2)
        _, _, s = _plan("RTM", "SHA", fuel_price_mode="origin_port", fuel_overrides=(("RTM", 900.0),))
        self.assertAlmostEqual(s.fuel_usd / s.fuel_t, 900.0, places=2)

    def test_every_port_has_a_price_with_a_basis(self):
        from ops import PORTS
        for pid in PORTS:
            self.assertIn(pid, E.PORT_FUEL_PRICES)
            self.assertTrue(E.PORT_FUEL_PRICES[pid]["basis"])


class TestBreakdown(unittest.TestCase):
    def test_every_factor_is_its_own_labelled_line(self):
        cfg, _, s = _plan("RTM", "SHA", carbon_usd_per_t=200.0)
        lines = E.breakdown(s, cfg.econ)
        labels = [l["label"] for l in lines]
        for needle in ("Fuel", "Canal tolls", "War-risk insurance", "Time cost", "ECA fuel switch", "EU ETS",
                       "Carbon shadow price"):
            self.assertTrue(any(needle in x for x in labels), (needle, labels))
        for l in lines:
            self.assertTrue(l["basis"], l)
        cash = sum(l["usd"] for l in lines if l["cash"])
        self.assertAlmostEqual(cash, s.total_usd, delta=1.0)
        shadow = next(l for l in lines if "Carbon shadow" in l["label"])
        self.assertFalse(shadow["cash"])

    def test_untouched_factors_show_zero_not_missing(self):
        cfg, _, s = _plan("SIN", "SHA")
        lines = {l["label"]: l for l in E.breakdown(s, cfg.econ)}
        self.assertEqual(lines["EU ETS"]["usd"], 0)
        self.assertEqual(lines["ECA fuel switch"]["usd"], 0)


class TestChangeThresholdCountsCarbon(unittest.TestCase):
    """The route search minimises cash + carbon shadow price, so the change-course threshold must too."""

    def _ctx(self, carbon):
        cfg = V.VoyageConfig(econ=E.Economics(carbon_usd_per_t=carbon))
        return V.build_context(cfg, default_settings())

    def test_cleaner_but_dearer_route_is_worth_taking_when_carbon_is_priced(self):
        hold = V.RouteSummary("hold", True, hours=300, fuel_usd=1.0e6, carbon_usd=0.5e6)
        alt = V.RouteSummary("alt", True, hours=300, fuel_usd=1.2e6, carbon_usd=0.2e6)     # +$200k cash, -$300k carbon
        self.assertTrue(V._worth_changing(self._ctx(200.0), hold, alt))

    def test_without_a_carbon_price_the_dearer_route_is_not(self):
        hold = V.RouteSummary("hold", True, hours=300, fuel_usd=1.0e6)
        alt = V.RouteSummary("alt", True, hours=300, fuel_usd=1.2e6)
        self.assertFalse(V._worth_changing(self._ctx(0.0), hold, alt))


class TestLegacyEngineUnchanged(unittest.TestCase):
    def test_stress_test_pricing_has_no_new_lines(self):
        from agents import DEFAULT_CONTAINER
        from routing_engine import price_edge_batch
        G = V.base_graph()
        costs = price_edge_batch(G.edges(data=True), DEFAULT_CONTAINER, 620.0)
        self.assertTrue(all(c.eca_usd == 0 and c.ets_usd == 0 and c.carbon_usd == 0 for c in costs.values()))


class TestSplitKeepsTheNewLines(unittest.TestCase):
    def test_sailed_plus_remaining_equals_total(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.advance_hours(5 * 24)
        done, ahead, _ = V.split_at(o.voyage, o.G)
        for k in ("eca_usd", "ets_usd", "co2_t", "carbon_usd"):
            whole = sum(getattr(l, k) for l in o.voyage.legs)
            parts = sum(getattr(l, k) for l in done) + sum(getattr(l, k) for l in ahead)
            self.assertAlmostEqual(whole, parts, delta=max(1.0, abs(whole) * 1e-6), msg=k)


if __name__ == "__main__":
    unittest.main()
