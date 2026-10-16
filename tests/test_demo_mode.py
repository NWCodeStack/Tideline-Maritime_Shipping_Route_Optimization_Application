"""
Regression tests for the one-click Hackathon Demo Mode (simulated Suez closure -> Cape diversion).

    python -m pytest tests/test_demo_mode.py -v
"""
import os
import unittest
from unittest import mock

import copilot as C
import voyage as V
from geo_assets import on_land
from ops import Ops


def _sail(legs):
    return [frozenset((l.x, l.y)) for l in legs if l.kind == "sail"]


class TestDemoScenario(unittest.TestCase):
    def setUp(self):
        self.o = Ops()
        self.msg = self.o.run_demo()
        self.d = self.o.demo

    def test_voyage_and_labelled_simulated_closure(self):
        o = self.o
        self.assertEqual((o.cfg.origin, o.cfg.destination, o.cfg.vessel_key), ("RTM", "SHA", "container"))
        self.assertTrue(o.settings["suez_blockage"])
        self.assertIn("simulated", self.msg.lower())
        kinds = [e["kind"] for e in o.voyage.timeline]
        self.assertIn("demo", kinds)
        self.assertEqual(kinds[-1], "reroute")
        self.assertIn("SCENARIO (simulated)", o.voyage.timeline[-1]["title"])
        self.assertTrue(all(e["source"] == "demo" for e in o.voyage.timeline if e["kind"] in ("demo", "reroute")))

    def test_position_is_valid_and_before_suez(self):
        b = self.d["before"]
        self.assertFalse(on_land(b["lat"], b["lon"]), (b["lat"], b["lon"]))
        self.assertLess(b["lon"], 32.35)                     # west of Port Said: still in the Mediterranean
        self.assertTrue(31.0 < b["lat"] < 34.5)
        self.assertEqual(b["corridor"].split(" > ")[0], "Suez Canal")   # canal still ahead before the closure
        done, _, _ = V.split_at(self.o.voyage, self.o.G)
        self.assertNotIn(frozenset(("SUZ", "PSD")), _sail(done))       # never transited the canal

    def test_diversion_is_cape_route_from_current_location(self):
        o, b, a = self.o, self.d["before"], self.d["after"]
        _, ahead, _ = V.split_at(o.voyage, o.G)
        self.assertTrue(self.d["via_cape"])
        self.assertIn("CAPE", [l.y for l in ahead])
        self.assertNotIn(frozenset(("SUZ", "PSD")), _sail(ahead))
        first = next(l for l in ahead if l.kind == "sail")
        start = V.point_on_lane(o.G, first.x, first.y, first.f0)
        self.assertAlmostEqual(start[0], b["lat"], places=4)
        self.assertAlmostEqual(start[1], b["lon"], places=4)
        self.assertAlmostEqual(a["t"], b["t"], places=6)                 # progress preserved
        self.assertEqual(o.voyage.status, "underway")
        self.assertEqual(o.voyage.reroutes, 1)
        self.assertTrue(o.voyage.superseded and len(o.voyage.superseded[0]) > 1)   # old route kept for the map

    def test_before_after_metrics(self):
        b, a = self.d["before"], self.d["after"]
        self.assertGreater(a["eta"], b["eta"])
        self.assertGreater(a["nm"], b["nm"])
        self.assertGreater(a["remaining_nm"], b["remaining_nm"])
        self.assertGreater(a["total_usd"], b["total_usd"])
        self.assertIn("Cape of Good Hope", a["corridor"])
        for k in ("risk", "fuel_t", "toll_usd"):
            self.assertIn(k, a)
        # the "after" figures are the live voyage state, not a separate calculation
        self.assertEqual(a["eta"], V.projected(self.o.voyage, self.o.G)["eta"])

    def test_replay_is_deterministic_and_clean(self):
        o = self.o
        first = dict(self.d["after"])
        o.advance_hours(100)
        o.add_storm(6.0)
        o.run_demo()
        self.assertEqual(o.demo["after"]["total_usd"], first["total_usd"])
        self.assertEqual(o.demo["after"]["eta"], first["eta"])
        self.assertEqual(o.settings["storm_cells"], [])               # replay starts from clean scenarios
        self.assertEqual(o.voyage.reroutes, 1)

    def test_playback_continues_after_demo(self):
        o = self.o
        t = o.voyage.t
        o.advance_hours(24)
        self.assertAlmostEqual(o.voyage.t, t + 24)


class TestDemoResetAndExit(unittest.TestCase):
    def test_exit_restores_previous_session(self):
        o = Ops()
        o.create_voyage("DXB", "RTM", vessel_key="tanker")
        o.set_disruption("red_sea_escalation", True)
        o.advance_hours(30)
        before_cfg, before_settings, before_t = o.cfg, dict(o.settings), o.voyage.t
        o.run_demo()
        o.run_demo()                                              # replay must not overwrite the saved session
        self.assertIn("restored", o.exit_demo())
        self.assertEqual(o.cfg, before_cfg)
        self.assertEqual(o.settings, before_settings)
        self.assertAlmostEqual(o.voyage.t, before_t)
        self.assertIsNone(o.demo)
        self.assertIn("not active", o.exit_demo())

    def test_reset_clears_demo_panel(self):
        o = Ops()
        o.run_demo()
        o.reset()
        self.assertIsNone(o.demo)
        self.assertEqual(o.voyage.t, 0.0)

    def test_sessions_are_isolated(self):
        a, b = Ops(), Ops()
        a.run_demo()
        self.assertFalse(b.settings["suez_blockage"])
        self.assertIsNone(b.demo)


class TestDemoCopilotAndApp(unittest.TestCase):
    def test_copilot_runs_and_exits_demo(self):
        o = Ops()
        r = C.handle("Run the hackathon demo", o)
        self.assertEqual([n for n, *_ in r.ops], ["run_demo"])
        self.assertTrue(o.demo["via_cape"])
        r = C.handle("Exit demo mode", o)
        self.assertEqual([n for n, *_ in r.ops], ["exit_demo"])
        self.assertIsNone(o.demo)

    def test_demo_button_in_app(self):
        from streamlit.testing.v1 import AppTest
        with mock.patch.dict(os.environ, {"TIDELINE_OFFLINE": "1"}):
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            at = AppTest.from_file(os.path.join(here, "app.py"), default_timeout=180)
            at.run()
            at.button(key="demo_instant").click().run()
            self.assertEqual([e.value for e in at.exception], [])
            self.assertIsNotNone(at.session_state["ops"].demo)
            html_out = " ".join(m.value for m in at.markdown)
            self.assertIn("Hackathon demo", html_out)
            self.assertIn("Before (via Suez)", html_out)
            self.assertIn("Recommended: continue via Cape of Good Hope", html_out)
            at.button(key="demo_run").click().run()               # animated replay: time-lapse armed, closure pending
            self.assertEqual([e.value for e in at.exception], [])
            o = at.session_state["ops"]
            self.assertTrue(o.playing and o._demo_target is not None and o.demo is None)
            at.button(key="demo_exit").click().run()
            self.assertEqual([e.value for e in at.exception], [])
            self.assertIsNone(at.session_state["ops"].demo)


if __name__ == "__main__":
    unittest.main()
