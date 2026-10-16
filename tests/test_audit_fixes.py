"""
Regression tests for the pre-deployment audit fixes:
  Fix 1 - chatbot intent: questions / what-ifs / negations never mutate state
  Fix 4 - forecast-informed simulation + route-bound (non-stale) weather
  Fix 5 - offline SVG voyage map
  Fix 7 - animated, replayable demo

    python -m pytest tests/test_audit_fixes.py -v
"""
import copy
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import copilot as C
import intel
import voyage as V
from map_view import build_ops_map_html
from ops import Ops


def fingerprint(o: Ops):
    """Everything a chat message could change."""
    return (copy.deepcopy(o.settings), o.cfg, round(o.voyage.t, 9), o.voyage.status, V.route_signature(o.voyage),
            len(o.voyage.timeline), o.version, o.playing, o.voyage.reroutes)


def underway() -> Ops:
    o = Ops()
    o.create_voyage("RTM", "SHA")
    o.advance_hours(100)                      # in the Mediterranean, Suez still ahead
    return o


# ------------------------------------------------------------------------------------------------- Fix 1
class TestIntentSafety(unittest.TestCase):
    NON_MUTATING = [
        "What happens if the Suez Canal closes?",
        "Do not close Suez, just explain the effect of a closure",
        "Show news about a Suez closure",
        "What if fuel prices increase by 25%?",
        "Is the Strait of Hormuz closed?",
        "The Strait of Hormuz has closed. Does that impact my Rotterdam to Shanghai shipment?",
        "Don't change anything, but what would a 6 m storm ahead do?",
        "Suez closure",                                        # ambiguous: no instruction
        "The Suez Canal has closed.",                          # statement: clarify, don't act
        "Would closing Bab-el-Mandeb delay us?",
        "Without changing anything, close Panama and tell me the cost",
    ]

    def test_reviewer_examples_leave_state_unchanged(self):
        for msg in self.NON_MUTATING:
            o = underway()
            before = fingerprint(o)
            reply = C.handle(msg, o)
            self.assertEqual(fingerprint(o), before, msg)
            self.assertFalse(reply.changed, msg)

    def test_hypotheticals_return_a_labelled_preview(self):
        o = underway()
        r = C.handle("What happens if the Suez Canal closes?", o)
        self.assertEqual([n for n, *_ in r.ops], ["preview"])
        self.assertIn("PREVIEW ONLY", r.text)
        self.assertIn("Cape of Good Hope", r.text)               # the counterfactual was actually computed
        r = C.handle("What if fuel prices increase by 25%?", o)
        self.assertIn("PREVIEW ONLY", r.text)
        self.assertAlmostEqual(o.cfg.fuel_price, 620.0)

    def test_information_requests(self):
        o = underway()
        r = C.handle("Is the Strait of Hormuz closed?", o)
        self.assertEqual([n for n, *_ in r.ops], ["scenario_status"])
        self.assertIn("not active", r.text)
        self.assertIn("does not verify real-world", r.text)
        r = C.handle("Show news about a Suez closure", o)
        self.assertEqual([n for n, *_ in r.ops], ["news"])

    def test_ambiguous_statement_asks_for_clarification(self):
        o = underway()
        r = C.handle("Suez closure", o)
        self.assertEqual(r.ops, [])
        self.assertIn("didn't change anything", r.text)

    def test_explicit_instructions_still_apply(self):
        cases = {
            "Simulate a Suez closure and reroute my ship": lambda o: o.settings["suez_blockage"],
            "Close Suez and reroute my ship.": lambda o: o.settings["suez_blockage"],
            "The Suez Canal just closed. Find the best alternative route.": lambda o: o.settings["suez_blockage"],
            "Increase fuel prices by 25% and recalculate costs.": lambda o: abs(o.cfg.fuel_price - 775) < 1e-6,
            "Advance the voyage by 2 days": lambda o: abs(o.voyage.t - 148) < 1e-6,
            "Please pause the vessel": lambda o: not o.playing,
        }
        for msg, check in cases.items():
            o = underway()
            o.playing = True
            C.handle(msg, o)
            self.assertTrue(check(o), msg)
            self.assertEqual(C.classify(msg), "action", msg)

    def test_preview_matches_what_applying_would_do(self):
        o = underway()
        preview = C.handle("What if Suez closes?", o).text
        C.handle("Simulate a Suez closure", o)
        after = o.snapshot()
        self.assertIn(after["corridor"], preview)

    def test_llm_proposals_go_through_the_same_gate(self):
        class FakeLLM:
            name = "fake"

            def extract_ops(self, message, state):
                return {"operations": [{"op": "set_disruption", "args": {"name": "suez_closure", "active": True}}],
                        "reply": ""}

            def explain(self, message, facts):
                return None

        o = underway()
        before = fingerprint(o)
        r = C.handle("is the canal ok right now?", o, llm=FakeLLM())       # question -> status only
        self.assertEqual(fingerprint(o), before)
        self.assertEqual([n for n, *_ in r.ops], ["scenario_status"])
        r = C.handle("hmm the canal", o, llm=FakeLLM())                    # ambiguous -> nothing
        self.assertEqual(fingerprint(o), before)
        r = C.handle("Please shut the canal for this exercise", o, llm=FakeLLM())   # explicit -> applied
        self.assertTrue(o.settings["suez_blockage"])


# ------------------------------------------------------------------------------------------------- Fix 4
def fake_assessment(o: Ops, wave: float, idx: int = 2, stale: bool = False) -> dict:
    pts = V.remaining_samples(o.voyage, o.G)
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    out = []
    for i, p in enumerate(pts):
        w = wave if i == idx else 1.0
        out.append({"label": p["label"], "lat": p["lat"], "lon": p["lon"], "eta": now + timedelta(hours=p["t"]),
                    "voyage_h": p["t"], "status": "forecast", "values": {"wave_height": w, "wind_speed_10m": 20.0},
                    "severity": intel.severity(w, 20)[0], "level": intel.severity(w, 20)[1], "action": ""})
    return {"status": "ok", "points": out, "worst": max(out, key=lambda r: r["values"]["wave_height"]),
            "near_ship": out[0], "fetched_at": "2026-10-16T00:00:00+00:00", "units": {}, "note": "",
            "source": "Open-Meteo (test)", "stale": stale}


class TestForecastInformed(unittest.TestCase):
    def test_hazard_ahead_becomes_labelled_scenario_and_reassesses(self):
        o = underway()
        sig = V.route_signature(o.voyage)
        msg = o.apply_forecast_hazards(fake_assessment(o, 7.5), wave_threshold_m=4.0)
        self.assertIn("FORECAST-INFORMED SCENARIO", msg)
        cells = [c for c in o.settings["storm_cells"] if c["source"] == o.FORECAST_SOURCE]
        self.assertEqual(len(cells), 1)
        self.assertIn("Open-Meteo", cells[0]["label"])
        self.assertIn("FORECAST-INFORMED", o.voyage.timeline[-1]["title"])
        self.assertEqual(o.voyage.timeline[-1]["source"], "forecast")
        if V.route_signature(o.voyage) != sig:                  # if the route moved, the change was recorded
            self.assertIsNotNone(o.last_change)

    def test_no_reapply_no_oscillation(self):
        o = underway()
        a = fake_assessment(o, 7.5)
        self.assertIsNotNone(o.apply_forecast_hazards(a, 4.0))
        n = len(o.voyage.timeline)
        self.assertIsNone(o.apply_forecast_hazards(a, 4.0))      # same hazard -> nothing new
        self.assertEqual(len(o.voyage.timeline), n)

    def test_below_threshold_ship_position_and_stale_are_ignored(self):
        o = underway()
        before = fingerprint(o)
        self.assertIsNone(o.apply_forecast_hazards(fake_assessment(o, 3.0), 4.0))             # below threshold
        self.assertIsNone(o.apply_forecast_hazards(fake_assessment(o, 9.0, idx=0), 4.0))      # at the ship itself
        self.assertIsNone(o.apply_forecast_hazards(fake_assessment(o, 9.0, stale=True), 4.0))  # stale data
        self.assertIsNone(o.apply_forecast_hazards(intel.weather_unavailable("down"), 4.0))
        self.assertEqual(fingerprint(o), before)

    def test_clear_forecast_cells_keeps_user_storms(self):
        o = underway()
        o.add_storm(6.0)
        o.apply_forecast_hazards(fake_assessment(o, 7.5, idx=3), 4.0)
        o.clear_forecast_hazards()
        self.assertEqual([c["source"] for c in o.settings["storm_cells"]], ["scenario"])

    def test_impact_estimate_uses_model_physics(self):
        txt = intel.operational_impact(6.0, 30.0)
        self.assertIn("model estimate", txt)
        self.assertIn("speed loss", txt)


class TestWeatherStaleness(unittest.TestCase):
    def test_signature_changes_on_reroute_not_on_sailing(self):
        o = underway()
        sig = V.route_signature(o.voyage)
        o.advance_hours(10)
        self.assertEqual(V.route_signature(o.voyage), sig)
        o.set_disruption("suez_closure", True)
        self.assertNotEqual(V.route_signature(o.voyage), sig)

    def test_failed_refresh_never_reuses_another_routes_forecast(self):
        o = underway()
        old_sig = V.route_signature(o.voyage)
        last = intel.with_route(fake_assessment(o, 2.0), old_sig, ())
        same = intel.reuse_or_unavailable(last, old_sig, "timeout")
        self.assertTrue(same["stale"])
        self.assertIn("last successful forecast for this route", same["note"])
        o.set_disruption("suez_closure", True)                 # route changes
        other = intel.reuse_or_unavailable(last, V.route_signature(o.voyage), "timeout")
        self.assertEqual(other["status"], "unavailable")
        self.assertEqual(other["points"], [])
        self.assertIn("previous route", other["note"])
        self.assertIsNone(intel.current_for(last, V.route_signature(o.voyage)))
        self.assertIs(intel.current_for(last, old_sig), last)


# ------------------------------------------------------------------------------------------------- Fix 5
class TestOfflineMap(unittest.TestCase):
    def test_svg_map_has_route_context(self):
        o = underway()
        o.set_disruption("suez_closure", True)
        G = o.G
        done, ahead, _ = V.split_at(o.voyage, G)
        alts = [(c.name, V.legs_polyline(G, c.summary.legs)) for c in o.voyage.candidates if not c.chosen]
        page = build_ops_map_html(G, o.settings, V.legs_polyline(G, done), V.legs_polyline(G, ahead),
                                  o.voyage.superseded, alts, o.voyage.blocked_remaining, o.state(), o.voyage.status)
        for needle in ('class="done"', 'class="route"', 'class="sup"', "SIMULATED POSITION", 'class="xmark"',
                       "Suez CLOSED (simulated scenario)"):
            self.assertIn(needle, page)

    def test_offline_map_mode_in_app(self):
        from streamlit.testing.v1 import AppTest
        with mock.patch.dict(os.environ, {"TIDELINE_OFFLINE": "1"}):
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            at = AppTest.from_file(os.path.join(here, "app.py"), default_timeout=180)
            at.run()
            at.selectbox(key="map_mode").set_value("Offline map").run()
            self.assertEqual([e.value for e in at.exception], [])
            at.button(key="demo_instant").click().run()
            self.assertEqual([e.value for e in at.exception], [])


# ------------------------------------------------------------------------------------------------- Fix 7
class TestAnimatedDemo(unittest.TestCase):
    def test_time_lapse_triggers_closure_at_the_approach(self):
        o = Ops()
        o.run_demo(animate=True)
        self.assertTrue(o.playing)
        self.assertIsNone(o.demo)
        self.assertFalse(o.settings["suez_blockage"])           # nothing closed yet
        target = o._demo_target
        now = 0.0
        o.tick(now)
        while o.demo is None and now < 60:
            now += 1.0
            o.tick(now)
            self.assertLessEqual(o.voyage.t, target + 1e-6)      # never overshoots past the approach point
        self.assertIsNotNone(o.demo)
        self.assertFalse(o.playing)
        self.assertAlmostEqual(o.demo["before"]["t"], target, places=6)
        instant = Ops()
        instant.run_demo()
        self.assertEqual(o.demo["after"]["total_usd"], instant.demo["after"]["total_usd"])
        self.assertEqual(o.demo["after"]["eta"], instant.demo["after"]["eta"])

    def test_manual_advance_during_time_lapse_stops_at_trigger(self):
        o = Ops()
        o.run_demo(animate=True)
        o.advance_hours(10_000)
        self.assertIsNotNone(o.demo)
        self.assertTrue(o.demo["via_cape"])

    def test_scrubbing_cancels_and_exit_restores(self):
        o = Ops()
        o.create_voyage("DXB", "RTM", vessel_key="tanker")
        cfg = o.cfg
        o.run_demo(animate=True)
        o.seek(10.0)
        self.assertIsNone(o._demo_target)
        o.exit_demo()
        self.assertEqual(o.cfg, cfg)
        self.assertIsNone(o._demo_target)
        self.assertFalse(o.playing)

    def test_demo_runs_without_network_or_llm(self):
        with mock.patch.dict(os.environ, {"TIDELINE_OFFLINE": "1"}):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            o = Ops()
            o.run_demo()
            self.assertTrue(o.demo["via_cape"])


if __name__ == "__main__":
    unittest.main()
