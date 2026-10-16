"""
Phase 4 building blocks: the pure helpers behind the right-hand tabs, and the Data sources list.

    python -m pytest tests/test_phase4_panels.py -v
"""
import unittest
from datetime import datetime

import panels as P
import sources as S
from ops import Ops


def _ops(origin="RTM", dest="SHA"):
    o = Ops()
    o.create_voyage(origin, dest)
    return o


class TestIncidentPicker(unittest.TestCase):
    def test_the_four_headline_incidents(self):
        self.assertEqual(set(P.INCIDENTS), {"suez_blockage", "bab_el_mandeb_closure", "hormuz_blockade",
                                            "panama_closure"})

    def test_selected_follows_settings(self):
        o = _ops()
        o.set_disruption("panama_closure", True)
        self.assertEqual(P.selected_incidents(o.settings), ["panama_closure"])

    def test_changes_are_the_difference_only(self):
        settings = _ops().settings
        self.assertEqual(P.incident_changes(["suez_blockage"], settings), [("suez_blockage", True)])
        settings = dict(settings, suez_blockage=True, hormuz_blockade=True)
        self.assertEqual(sorted(P.incident_changes(["suez_blockage"], settings)), [("hormuz_blockade", False)])
        self.assertEqual(P.incident_changes(["suez_blockage", "hormuz_blockade"], settings), [])

    def test_unknown_keys_are_ignored(self):
        self.assertEqual(P.incident_changes(["not_an_incident"], _ops().settings), [])


class TestOptionsTable(unittest.TestCase):
    def test_rows_have_cost_co2_risk_score_and_band(self):
        rows = P.options_rows(_ops().options())
        self.assertTrue(rows)
        for r in rows:
            for k in ("Option", "Corridor", "ETA days", "Distance nm", "Fuel t", "CO2 t", "All-in $M",
                      "Risk score", "Risk band"):
                self.assertIn(k, r)
            self.assertIn(r["Risk band"], {"Low", "Guarded", "Elevated", "High", "Severe"})

    def test_chosen_option_is_ticked(self):
        o = _ops()
        o.advance_hours(4 * 24)
        o.set_disruption("suez_closure", True)
        rows = P.options_rows(o.voyage.candidates)
        self.assertEqual(sum(1 for r in rows if r["Selected"]), 1)


class TestFuelBurnSeries(unittest.TestCase):
    def test_cumulative_and_matches_the_voyage_total(self):
        o = _ops()
        series = P.fuel_burn_series(o.voyage.legs)
        self.assertTrue(series)
        burn = [p["Fuel burned (t)"] for p in series]
        self.assertEqual(burn, sorted(burn))
        self.assertAlmostEqual(burn[-1], sum(l.fuel_t for l in o.voyage.legs), delta=0.5)
        days = [p["Voyage day"] for p in series]
        self.assertEqual(days, sorted(days))

    def test_empty_voyage(self):
        self.assertEqual(P.fuel_burn_series([]), [])


class TestCheckpointRows(unittest.TestCase):
    def _weather(self):
        return {"status": "ok", "source": "Open-Meteo Marine API + Open-Meteo Forecast API (wind)",
                "points": [{"label": "Near Off Galicia", "eta": datetime(2026, 10, 20, 6), "status": "forecast",
                            "severity": "Moderate", "values": {"wave_height": 3.46, "wind_speed_10m": 29.7}},
                           {"label": "Near Suez", "eta": datetime(2026, 11, 2, 6),
                            "status": "beyond forecast horizon", "severity": "n/a", "values": {}}]}

    def test_rows_show_sea_state_and_source(self):
        rows = P.checkpoint_rows(self._weather())
        self.assertEqual(rows[0]["Waves (m)"], 3.46)
        self.assertEqual(rows[0]["Source"], "live forecast (Open-Meteo)")
        self.assertIsNone(rows[1]["Waves (m)"])
        self.assertIn("beyond", rows[1]["Source"])

    def test_no_weather_means_no_rows(self):
        self.assertEqual(P.checkpoint_rows(None), [])
        self.assertEqual(P.checkpoint_rows({"status": "unavailable", "points": []}), [])


class TestStatusLine(unittest.TestCase):
    def test_recommendation_corridor_and_risk(self):
        html_ = P.status_line("change_course", "Panama Canal", 31.2)
        for needle in ("CHANGE COURSE NOW", "Panama Canal", "31", "Guarded"):
            self.assertIn(needle, html_)
        self.assertIn("STAY ON COURSE", P.status_line("stay", "Cape Horn", 10))

    def test_blank_before_first_assessment(self):
        self.assertIn("Plan assessed", P.status_line("", "Direct", 5))

    def test_single_line_and_escaped(self):
        html_ = P.status_line("stay", "<b>x</b>", 5)
        self.assertNotIn("\n", html_)
        self.assertNotIn("<b>x</b>", html_)


class TestDataSources(unittest.TestCase):
    def test_every_row_has_a_basis_and_detail(self):
        rows = S.plan_sources(_ops(), None, None, "Rules-based command mode (no LLM key)")
        self.assertGreaterEqual(len(rows), 10)
        for r in rows:
            self.assertIn(r["basis"], S.BASES, r)
            self.assertTrue(r["input"] and r["detail"], r)

    def test_nothing_simulated_is_called_live(self):
        rows = {r["input"]: r for r in S.plan_sources(_ops(), None, None, "")}
        for name in ("Vessel position and clock", "Lane network", "Disruption scenarios", "Risk score"):
            self.assertEqual(rows[name]["basis"], "simulated", name)

    def test_weather_row_follows_the_fetch(self):
        w = {"status": "ok", "source": "Open-Meteo Marine API + Open-Meteo Forecast API (wind)",
             "fetched_at": "2026-10-16T22:34:24+00:00", "points": []}
        row = next(r for r in S.plan_sources(_ops(), w, None, "") if r["input"].startswith("Weather"))
        self.assertEqual(row["basis"], "live")
        self.assertIn("2026-10-16", row["date"])
        self.assertTrue(row["url"].startswith("https://open-meteo.com"))
        down = next(r for r in S.plan_sources(_ops(), {"status": "unavailable", "points": [], "note": "offline",
                                                       "fetched_at": None}, None, "")
                    if r["input"].startswith("Weather"))
        self.assertEqual(down["basis"], "simulated")
        self.assertIn("unavailable", down["detail"].lower())

    def test_news_row_is_advisory(self):
        news = {"provider": "Google News RSS", "fetched_at": "2026-10-16T22:00:00+00:00", "articles": [{"t": 1}]}
        row = next(r for r in S.plan_sources(_ops(), None, news, "") if r["input"].startswith("News"))
        self.assertEqual(row["basis"], "live")
        self.assertIn("advisory", row["detail"].lower())

    def test_fuel_price_row_follows_the_setting(self):
        o = _ops()
        row = next(r for r in S.plan_sources(o, None, None, "") if r["input"] == "Fuel price")
        self.assertEqual(row["basis"], "user setting")
        o.set_economics(fuel_price_mode="origin_port")
        row = next(r for r in S.plan_sources(o, None, None, "") if r["input"] == "Fuel price")
        self.assertEqual(row["basis"], "sample")

    def test_economics_rows_are_estimates(self):
        rows = {r["input"]: r for r in S.plan_sources(_ops(), None, None, "")}
        self.assertEqual(rows["EU ETS"]["basis"], "estimated")
        self.assertEqual(rows["ECA fuel switch"]["basis"], "estimated")
        self.assertEqual(rows["CO2 per tonne of fuel"]["basis"], "published")

    def test_llm_row_reports_the_mode(self):
        row = next(r for r in S.plan_sources(_ops(), None, None, "Rules-based command mode (no LLM key)")
                   if r["input"] == "Copilot")
        self.assertIn("no LLM key", row["detail"])


if __name__ == "__main__":
    unittest.main()
