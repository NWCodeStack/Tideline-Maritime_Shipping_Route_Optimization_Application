"""
Phase 3: what the risk score means - bands, caption, and a breakdown that adds up.

    python -m pytest tests/test_phase3_risk.py -v

Score = round(100 x modelled route risk). Per leg the weather, security and traffic signals are combined with a
noisy-OR; over the route the legs are weighted by the time spent at sea. It is a relative comparison between
routes, not a calibrated probability.
"""
import unittest

import risk as R
import ui
import voyage as V
from ops import Ops


def _ops(origin="RTM", dest="SHA", **scenario):
    o = Ops()
    o.create_voyage(origin, dest)
    for name, value in scenario.items():
        o.set_disruption(name, True, value=value)
    return o


class TestBands(unittest.TestCase):
    def test_boundaries(self):
        expected = {0: "Low", 19: "Low", 20: "Guarded", 39: "Guarded", 40: "Elevated", 59: "Elevated",
                    60: "High", 79: "High", 80: "Severe", 100: "Severe"}
        for score, label in expected.items():
            self.assertEqual(R.band(score)[0], label, score)

    def test_fractional_scores_round_first(self):
        self.assertEqual(R.band(19.4)[0], "Low")
        self.assertEqual(R.band(19.6)[0], "Guarded")

    def test_every_band_has_a_colour(self):
        for score in (5, 25, 45, 65, 95):
            self.assertIn(R.band(score)[1], {"mint", "blue", "gold", "red"})

    def test_old_label_function_follows_the_bands(self):
        for score in (3, 22, 41, 66, 90):
            self.assertEqual(V.risk_label(score), R.band(score)[0])

    def test_caption_text(self):
        self.assertTrue(R.CAPTION.startswith("Risk score (0-100): modelled chance that something disrupts this voyage"))
        self.assertIn("not a calibrated probability", R.CAPTION)


class TestNoisyOr(unittest.TestCase):
    def test_combination(self):
        self.assertAlmostEqual(R.leg_risk(0.45, 0.10, 0.0), 1 - 0.55 * 0.90)
        self.assertEqual(R.leg_risk(0, 0, 0), 0.0)
        self.assertAlmostEqual(R.leg_risk(1.0, 0.3, 0.2), 1.0)

    def test_never_below_the_largest_signal_nor_above_one(self):
        for a, b, c in ((0.1, 0.2, 0.3), (0.9, 0.9, 0.9), (0.0, 0.05, 0.0)):
            p = R.leg_risk(a, b, c)
            self.assertGreaterEqual(p, max(a, b, c) - 1e-12)
            self.assertLessEqual(p, 1.0)


class TestBreakdownAddsUp(unittest.TestCase):
    def test_parts_sum_to_the_total(self):
        o = _ops(red_sea_escalation=0.45)
        rb = R.route_risk(o.voyage.legs)
        self.assertAlmostEqual(rb.weather + rb.security + rb.traffic, rb.total, places=9)
        self.assertGreater(rb.security, 0.0)
        self.assertAlmostEqual(rb.score, round(100 * rb.total))

    def test_total_is_the_time_weighted_mean_of_leg_noisy_or(self):
        o = _ops(red_sea_escalation=0.45)
        sail = [l for l in o.voyage.legs if l.kind == "sail"]
        hours = sum(l.sea_h for l in sail)
        expected = sum(l.sea_h * R.leg_risk(l.p_sec_base + l.p_sec_scen, l.p_wx, l.p_traf) for l in sail) / hours
        self.assertAlmostEqual(R.route_risk(o.voyage.legs).total, expected, places=9)

    def test_summary_and_projection_use_the_same_score(self):
        o = _ops(red_sea_escalation=0.45)
        ctx = V.build_context(o.cfg, o.settings, o.model)
        s = V.summarise(ctx, o.voyage.legs)
        rb = R.route_risk(o.voyage.legs)
        self.assertAlmostEqual(s.risk_index, 100 * rb.total, places=6)
        self.assertAlmostEqual(V.projected(o.voyage, o.G)["risk_index"], 100 * rb.total, places=6)

    def test_still_adds_up_after_part_of_the_voyage_is_sailed(self):
        o = _ops()
        o.advance_hours(6 * 24)
        _, ahead, _ = V.split_at(o.voyage, o.G)
        rb = R.route_risk(ahead)
        self.assertAlmostEqual(rb.weather + rb.security + rb.traffic, rb.total, places=9)
        self.assertAlmostEqual(V.projected(o.voyage, o.G)["remaining_risk_index"], 100 * rb.total, places=6)

    def test_empty_route_is_zero(self):
        rb = R.route_risk([])
        self.assertEqual((rb.total, rb.score, rb.security), (0.0, 0, 0.0))


class TestTopZone(unittest.TestCase):
    def test_scenario_shows_up_as_its_own_term(self):
        o = _ops(red_sea_escalation=0.45)
        rb = R.route_risk(o.voyage.legs)
        self.assertGreater(rb.top_scenario, 0.0)
        self.assertAlmostEqual(rb.top_base + rb.top_scenario, rb.top_security, places=6)
        self.assertTrue(rb.top_zone)

    def test_calm_network_has_no_scenario_term(self):
        rb = R.route_risk(_ops().voyage.legs)
        self.assertEqual(rb.top_scenario, 0.0)

    def test_explanation_names_the_zone_and_its_parts(self):
        o = _ops(red_sea_escalation=0.45)
        rb = R.route_risk(o.voyage.legs)
        text = R.explain(rb)
        for needle in (rb.top_zone, "security", "base", "scenario", "weather", "traffic"):
            self.assertIn(needle, text)

    def test_headlines_are_advisory_only(self):
        text = R.explain(R.route_risk(_ops().voyage.legs))
        self.assertIn("headlines are advisory and do not change the score", text.lower())


class TestTrafficSignal(unittest.TestCase):
    def test_zero_without_a_slowdown_setting(self):
        self.assertEqual(R.route_risk(_ops().voyage.legs).traffic, 0.0)

    def test_comes_only_from_the_simulated_canal_delay(self):
        o = _ops()
        o.update_config(canal_delay_hours=72.0)
        rb = R.route_risk(o.voyage.legs)
        self.assertGreater(rb.traffic, 0.0)
        self.assertIn("SIMULATED", R.BASIS["traffic"])

    def test_not_a_live_feed(self):
        self.assertIn("no live traffic feed", R.BASIS["traffic"])


class TestWhereTheScoreAppears(unittest.TestCase):
    def test_chip_has_band_score_and_caption(self):
        chip = ui.risk_chip(27.4)
        for needle in ("27", "Guarded", "not a calibrated probability"):
            self.assertIn(needle, chip)

    def test_chip_colours_follow_the_band(self):
        self.assertIn("mint", ui.risk_chip(5))
        self.assertIn("red", ui.risk_chip(85))

    def test_copilot_risk_answer_gives_band_and_breakdown(self):
        o = _ops(red_sea_escalation=0.45)
        text = o.explain("risk")
        for needle in (R.band(V.projected(o.voyage, o.G)["remaining_risk_index"])[0], "weather", "security",
                       "not a calibrated probability"):
            self.assertIn(needle, text)

    def test_change_banner_shows_band_labels(self):
        o = _ops()
        o.advance_hours(4 * 24)                       # still short of Suez, so closing it changes the route
        o.set_disruption("suez_closure", True)
        banner = ui.change_banner(o.last_change)
        self.assertIn("Risk ahead", banner)
        self.assertRegex(banner, r"Risk ahead <b>\d+ (Low|Guarded|Elevated|High|Severe) → \d+ "
                                 r"(Low|Guarded|Elevated|High|Severe)</b>")


class TestRiskPanelAndTooltip(unittest.TestCase):
    def test_panel_shows_caption_bands_parts_and_top_lane(self):
        o = _ops(red_sea_escalation=0.45)
        rb = R.route_risk(o.voyage.legs)
        html_ = ui.risk_panel(rb)
        for needle in ("not a calibrated probability", "Weather", "Security", "Traffic", "SIMULATED",
                       f"{rb.score}/100", rb.band_label, rb.top_zone, "20+ Guarded", "80+ Severe",
                       "do not change the score"):
            self.assertIn(needle, html_)

    def test_panel_markup_is_single_line_and_escaped(self):
        rb = R.RiskBreakdown(total=0.3, weather=0.1, security=0.2, top_zone="<b>x</b> & y", top_security=0.2,
                             top_base=0.2)
        html_ = ui.risk_panel(rb)
        self.assertNotIn("\n", html_)
        self.assertIn("&lt;b&gt;x&lt;/b&gt; &amp; y", html_)
        self.assertNotIn("<b>x</b>", html_)

    def test_kpi_card_tooltip_is_attribute_safe(self):
        card = ui.metric_cards([("Risk ahead", "27/100", "Guarded", "blue", True, 'say "hi" <now>')])
        self.assertIn('title="say &quot;hi&quot; &lt;now&gt;"', card)

    def test_arrived_voyage_reports_the_whole_route_not_zero(self):
        o = _ops()
        o.advance_hours(10_000)
        self.assertEqual(o.voyage.status, "arrived")
        text = o.explain("risk")
        self.assertIn("Whole-voyage risk score", text)
        self.assertNotIn("risk score 0/100", text)


class TestRoutingObjectiveUnchanged(unittest.TestCase):
    def test_safest_route_is_not_riskier_than_fastest(self):
        o = _ops(red_sea_escalation=0.8)
        ctx = V.build_context(o.cfg, o.settings, o.model)
        pos = V.Position(node="RTM")
        safe = V.route_from(ctx, pos, "SHA", "safest")
        fast = V.route_from(ctx, pos, "SHA", "fastest")
        self.assertLessEqual(safe.risk_index, fast.risk_index + 1e-6)


if __name__ == "__main__":
    unittest.main()
