"""
Acceptance tests for the live operations simulator, dynamic rerouting, copilot and live-data feeds.

    python -m pytest tests/test_live_ops.py -v
"""
import os
import unittest
from datetime import datetime, timezone
from unittest import mock

import requests

import copilot as C
import intel
import voyage as V
from geo_assets import on_land
from ops import Ops


def _time_on_lane(v, lane):
    """Voyage hour when the ship is halfway along ``lane`` (a 2-node set)."""
    acc = 0.0
    for l in v.legs:
        if l.kind == "sail" and {l.x, l.y} == lane:
            return acc + l.wait_h + l.sea_h * 0.5
        acc += l.hours
    raise AssertionError(f"lane {lane} not on route")


def _sail_lanes(legs):
    return [frozenset((l.x, l.y)) for l in legs if l.kind == "sail"]


class Test1VesselAnimation(unittest.TestCase):
    def test_ship_follows_route_and_metrics_stay_consistent(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.start()
        G = o.G
        total_nm = sum(l.nm for l in o.voyage.legs)
        total_h = o.voyage.total_hours
        eta0 = V.projected(o.voyage, G)["eta"]
        last_traveled, last_t = -1.0, -1.0
        for _ in range(40):
            o.advance_hours(12)
            s = o.state()
            self.assertGreater(s.traveled_nm, last_traveled - 1e-6)
            self.assertGreater(o.voyage.t, last_t)
            self.assertAlmostEqual(s.traveled_nm + s.remaining_nm, total_nm, delta=1.0)
            self.assertAlmostEqual(s.elapsed_h + s.remaining_h, total_h, delta=1e-6)
            leg = o.voyage.legs[s.leg_index]
            if leg.kind == "sail" and G.edges[leg.x, leg.y]["mode"] == "sea":    # canals cut through land
                self.assertFalse(on_land(s.lat, s.lon), (s.lat, s.lon))
            self.assertEqual(V.projected(o.voyage, G)["eta"], eta0)     # no disruption -> ETA unchanged
            last_traveled, last_t = s.traveled_nm, o.voyage.t
            if o.voyage.status == "arrived":
                break
        o.advance_hours(10_000)
        self.assertEqual(o.voyage.status, "arrived")
        s = o.state()
        self.assertAlmostEqual((s.lat, s.lon), V.node_ll(G, "SHA"), places=4)

    def test_position_is_on_the_current_lane_segment(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.advance_hours(100)
        s = o.state()
        G = o.G
        leg = o.voyage.legs[s.leg_index]
        a = V.point_on_lane(G, leg.x, leg.y, leg.f0)
        b = V.point_on_lane(G, leg.x, leg.y, leg.f1)
        lo_lat, hi_lat = sorted((a[0], b[0]))
        self.assertTrue(lo_lat - 1e-6 <= s.lat <= hi_lat + 1e-6)


class Test2MidVoyageDisruption(unittest.TestCase):
    def test_bab_el_mandeb_closure_reroutes_from_current_position(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.start()
        o.advance_hours(_time_on_lane(o.voyage, {"RS2", "RS3"}))       # in the middle of the Red Sea
        before = o.state()
        t_before = o.voyage.t
        msg = o.set_disruption("bab_el_mandeb_closure", True)
        after = o.state()
        self.assertIn("REROUTED", msg)
        self.assertEqual(o.voyage.status, "underway")
        # progress preserved exactly
        self.assertAlmostEqual(o.voyage.t, t_before, places=6)
        self.assertAlmostEqual(after.traveled_nm, before.traveled_nm, delta=0.5)
        self.assertAlmostEqual((after.lat, after.lon), (before.lat, before.lon), places=4)
        # the new remaining route starts at the ship, not at Rotterdam, and avoids the closed strait
        _, ahead, _ = V.split_at(o.voyage, o.G)
        first = next(l for l in ahead if l.kind == "sail")
        start = V.point_on_lane(o.G, first.x, first.y, first.f0)
        self.assertAlmostEqual(start[0], before.lat, places=4)
        self.assertAlmostEqual(start[1], before.lon, places=4)
        self.assertNotIn(frozenset(("GOA", "BAB")), _sail_lanes(ahead))
        self.assertIn("CAPE", [l.y for l in ahead])
        self.assertTrue(o.voyage.superseded and len(o.voyage.superseded[0]) > 1)
        self.assertEqual(o.voyage.timeline[-1]["kind"], "reroute")

    def test_suez_closure_in_mediterranean_turns_back(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.advance_hours(_time_on_lane(o.voyage, {"MED3", "MED2"}))
        msg = o.set_disruption("suez_closure", True)
        self.assertIn("REROUTED", msg)
        _, ahead, _ = V.split_at(o.voyage, o.G)
        self.assertNotIn(frozenset(("SUZ", "PSD")), _sail_lanes(ahead))


class Test3NoFalseHormuzImpact(unittest.TestCase):
    def test_hormuz_closure_does_not_reroute_rotterdam_shanghai(self):
        for hours in (0.0, 60.0, 250.0):
            o = Ops()
            o.create_voyage("RTM", "SHA")
            if hours:
                o.advance_hours(hours)
            legs_before = [(l.x, l.y, round(l.f0, 6), round(l.f1, 6)) for l in o.voyage.legs]
            msg = o.set_disruption("hormuz_closure", True)
            self.assertIn("NOT affected", msg, hours)
            self.assertEqual(o.voyage.reroutes, 0)
            legs_after = [(l.x, l.y, round(l.f0, 6), round(l.f1, 6)) for l in o.voyage.legs]
            self.assertEqual(legs_before, legs_after)

    def test_closure_already_passed_is_not_rerouted(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.advance_hours(_time_on_lane(o.voyage, {"GOA", "ARB"}))        # past Suez and Bab-el-Mandeb
        msg = o.set_disruption("suez_closure", True)
        self.assertIn("NOT affected", msg)
        self.assertEqual(o.voyage.reroutes, 0)


class Test4PhysicallyBlockedVoyage(unittest.TestCase):
    def test_ship_trapped_in_gulf_is_blocked_not_teleported(self):
        o = Ops()
        o.create_voyage("DXB", "RTM", vessel_key="tanker")
        o.start()
        o.advance_hours(6)
        s = o.state()
        self.assertTrue(24.0 < s.lat < 30.5 and s.lon < 56.5, (s.lat, s.lon))   # still inside the Persian Gulf
        t = o.voyage.t
        msg = o.set_disruption("hormuz_closure", True)
        self.assertEqual(o.voyage.status, "blocked")
        self.assertIn("BLOCKED", msg)
        self.assertIn("No feasible sea passage", o.voyage.block_reason)
        # the ship holds position: time does not advance and it is not moved anywhere
        held = o.state()
        o.advance_hours(48)
        self.assertEqual(o.voyage.t, t)
        self.assertAlmostEqual((o.state().lat, o.state().lon), (held.lat, held.lon), places=6)
        self.assertNotIn("FJR", [l.y for l in o.voyage.legs])            # no overland "sea passage" invented
        # reopening restores a voyage from where the ship is
        o.set_disruption("hormuz_closure", False)
        self.assertEqual(o.voyage.status, "underway")
        self.assertIn("HOR", [l.y for l in o.voyage.legs])

    def test_departure_blocked_when_already_closed(self):
        o = Ops()
        o.set_disruption("hormuz_closure", True)
        o.create_voyage("DXB", "RTM", vessel_key="tanker")
        self.assertEqual(o.voyage.status, "blocked")
        self.assertIn("Cannot start", o.start())


class Test5ChatbotControlsMap(unittest.TestCase):
    def test_close_suez_and_reroute(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.advance_hours(_time_on_lane(o.voyage, {"MED2", "MED1"}))
        version = o.version
        reply = C.handle("Close Suez and reroute my ship.", o)
        self.assertTrue(o.settings["suez_blockage"])                       # structured state updated
        self.assertEqual([n for n, *_ in reply.ops], ["set_disruption"])
        self.assertGreater(o.version, version)                             # map layer redraw key bumped
        self.assertEqual(o.voyage.reroutes, 1)                              # routing engine ran
        self.assertIn("REROUTED", reply.text)                               # explanation
        self.assertIn("SCENARIO (simulated)", reply.text)
        self.assertEqual(o.voyage.timeline[-1]["source"], "copilot")

    def test_example_commands_parse(self):
        o = Ops()
        cases = {
            "My ship is traveling from Rotterdam to Shanghai. Start the voyage.": ["start"],
            "Advance the voyage by 5 days.": ["advance"],
            "The Suez Canal just closed. Find the best alternative route.": ["set_disruption"],
            "A storm with 6-meter waves is affecting our current route. Reroute to avoid it.": ["add_storm"],
            "What is the safest available route?": ["explain"],
            "Increase fuel prices by 25% and recalculate costs.": ["set_fuel_price"],
            "How much more expensive is the Cape of Good Hope route?": ["compare_route"],
            "Pause the vessel and explain our current operational risk.": ["pause", "explain"],
            "Show me recent maritime news affecting the Red Sea.": ["news"],
            "The Strait of Hormuz has closed. Does that impact my Rotterdam to Shanghai shipment?":
                ["set_disruption", "explain"],
            "Create a new voyage from Jebel Ali to Rotterdam with a tanker": ["create_voyage"],
        }
        for msg, expected in cases.items():
            got = [op.name for op in C._merge(C.parse(msg, o))]
            self.assertEqual(got, expected, msg)
        adv = C.parse("Advance the voyage by 5 days.", o)[0]
        self.assertEqual(adv.args["hours"], 120.0)

    def test_fuel_price_command_uses_same_code_path(self):
        o = Ops()
        C.handle("Increase fuel prices by 25%", o)
        self.assertAlmostEqual(o.cfg.fuel_price, 775.0)

    def test_validation_rejects_unknown_or_out_of_range_ops(self):
        with self.assertRaises(C.InvalidOp):
            C.validate(C.Op("exec_python", {"code": "import os"}))
        with self.assertRaises(C.InvalidOp):
            C.validate(C.Op("advance", {"hours": 1e9}))
        with self.assertRaises(C.InvalidOp):
            C.validate(C.Op("set_disruption", {"name": "close_everything"}))
        with self.assertRaises(C.InvalidOp):
            C.validate(C.Op("advance", {"hours": 5, "shell": "rm -rf /"}))

    def test_llm_proposals_are_validated_before_execution(self):
        class FakeLLM:
            name = "fake"

            def extract_ops(self, message, state):
                return {"operations": [{"op": "run_shell", "args": {}},
                                       {"op": "advance", "args": {"hours": 24}}], "reply": "ok"}

            def explain(self, message, facts):
                return None

        o = Ops()
        reply = C.handle("please do the needful", o, llm=FakeLLM())
        self.assertEqual([n for n, *_ in reply.ops], ["advance"])
        self.assertEqual(len(reply.rejected), 1)
        self.assertAlmostEqual(o.voyage.t, 24.0)

    def test_unknown_message_without_llm_returns_help(self):
        o = Ops()
        reply = C.handle("sing me a sea shanty", o)
        self.assertEqual(reply.ops, [])
        self.assertIn("command mode", reply.text)


class Test6FeedOutages(unittest.TestCase):
    def setUp(self):
        os.environ.pop("TIDELINE_OFFLINE", None)
        intel._DOWN_UNTIL.clear()

    def test_weather_network_failure_raises_cleanly(self):
        with mock.patch("intel.requests.get", side_effect=requests.ConnectionError("down")), \
                mock.patch("intel.time.sleep"):
            with self.assertRaises(intel.IntelError):
                intel.fetch_marine(((10.0, 60.0),))
        w = intel.weather_unavailable("network error")
        self.assertEqual(w["status"], "unavailable")
        self.assertIn("unavailable", intel.weather_summary(w))
        self.assertIn("unavailable", intel.weather_summary(None))

    def test_rate_limit_is_not_retried(self):
        resp = mock.Mock(status_code=429, text="Please limit requests")
        with mock.patch("intel.requests.get", return_value=resp) as g:
            with self.assertRaises(intel.IntelError):
                intel.fetch_news_gdelt()
            self.assertEqual(g.call_count, 1)

    def test_news_falls_back_to_rss_when_gdelt_returns_non_json(self):
        gdelt = mock.Mock(status_code=200, content=b"Please limit requests")
        gdelt.json.side_effect = ValueError("not json")
        rss = mock.Mock(status_code=200, content=(
            b"<rss><channel><item><title>Shipping lines pause Red Sea transits - Example Wire</title>"
            b"<link>https://example.com/a</link><pubDate>Wed, 14 Oct 2026 09:10:00 GMT</pubDate>"
            b"<source url='https://example.com'>Example Wire</source></item></channel></rss>"))

        def fake_get(url, **kw):
            return gdelt if "gdelt" in url else rss
        with mock.patch("intel.requests.get", side_effect=fake_get):
            news = intel.fetch_news()
        self.assertEqual(news["provider"], "Google News RSS")
        a = news["articles"][0]
        self.assertEqual(a["title"], "Shipping lines pause Red Sea transits")
        self.assertEqual(a["source"], "Example Wire")
        self.assertIn("bab_el_mandeb", a["regions"])
        ranked = intel.rank_news(news["articles"], ["Bab-el-Mandeb"])
        self.assertEqual(ranked[0]["on_route"], ["Bab-el-Mandeb"])
        self.assertEqual(intel.advisories(ranked)[0]["scenario"], "bab_el_mandeb_closure")

    def test_news_total_failure_is_reported_not_invented(self):
        with mock.patch("intel.requests.get", side_effect=requests.Timeout("slow")), \
                mock.patch("intel.time.sleep"):
            with self.assertRaises(intel.IntelError):
                intel.fetch_news()
        self.assertIn("unavailable", intel.news_summary(None, []))
        self.assertIn("No headlines are invented", intel.news_summary({"articles": [], "error": "x"}, []))

    def test_headlines_never_close_corridors(self):
        o = Ops()
        news = {"articles": [intel._article("Suez Canal closed after ship attack", "https://e.com", "E",
                                            datetime.now(timezone.utc), "test")], "provider": "test"}
        text = intel.news_summary(news, ["Suez Canal"])
        self.assertIn("advisory only", text)
        self.assertFalse(o.settings["suez_blockage"])

    def test_weather_beyond_horizon_is_labelled(self):
        now = datetime(2026, 10, 16, tzinfo=timezone.utc)
        times = [f"2026-10-16T{h:02d}:00" for h in range(24)]
        raw = {"marine": [{"hourly": {"time": times, "wave_height": [6.5] * 24}}] * 2,
               "wind": None, "fetched_at": "2026-10-16T00:00:00+00:00"}
        pts = [{"lat": 10, "lon": 60, "t": 2, "label": "Ship position"},
               {"lat": 12, "lon": 62, "t": 400, "label": "Near X"}]
        w = intel.assess_weather(raw, pts, now)
        self.assertEqual(w["points"][0]["status"], "forecast")
        self.assertEqual(w["points"][0]["severity"], "Severe")
        self.assertEqual(w["points"][1]["status"], "beyond forecast horizon")
        self.assertEqual(w["status"], "partial")                           # wind missing -> partial, flagged


class Test7SimulationControls(unittest.TestCase):
    def test_pause_resume_reset(self):
        o = Ops()
        o.create_voyage("RTM", "SHA")
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)                                                   # first tick arms the clock
        o.tick(now=101.0)
        self.assertAlmostEqual(o.voyage.t, 10.0)
        o.pause()
        p = o.state()
        o.tick(now=105.0)
        self.assertAlmostEqual(o.voyage.t, 10.0)                            # no movement while paused
        self.assertEqual((o.state().lat, o.state().lon), (p.lat, p.lon))
        o.resume()
        o.tick(now=200.0)
        o.tick(now=202.0)
        self.assertAlmostEqual(o.voyage.t, 30.0)                            # resumes from where it paused
        o.tick(now=260.0)                                                   # long stall is capped (3 s)
        self.assertAlmostEqual(o.voyage.t, 60.0)
        o.reset()
        self.assertEqual(o.voyage.t, 0.0)
        self.assertEqual(o.voyage.status, "planned")
        self.assertFalse(o.playing)
        self.assertAlmostEqual((o.state().lat, o.state().lon), V.node_ll(o.G, "RTM"))

    def test_sessions_do_not_share_state(self):
        a, b = Ops(), Ops()
        a.set_disruption("suez_closure", True)
        a.advance_hours(50)
        self.assertFalse(b.settings["suez_blockage"])
        self.assertEqual(b.voyage.t, 0.0)


class Test8Deployment(unittest.TestCase):
    def test_app_starts_without_network_or_api_keys(self):
        from streamlit.testing.v1 import AppTest
        env = {"TIDELINE_OFFLINE": "1"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            at = AppTest.from_file(os.path.join(here, "app.py"), default_timeout=180)
            at.run()
            self.assertEqual([e.value for e in at.exception], [])
            text = " ".join(m.value for m in at.markdown)
            self.assertIn("Command mode", text)
            self.assertIn("SIMULATED", text.upper())
            # drive the chat through the real UI callback path
            at.chat_input(key="chat_in").set_value("Advance the voyage by 2 days").run()
            self.assertEqual([e.value for e in at.exception], [])
            self.assertAlmostEqual(at.session_state["ops"].voyage.t, 48.0)

    def test_requirements_cover_imports(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        req = open(os.path.join(here, "requirements.txt"), encoding="utf-8").read().lower()
        for pkg in ("streamlit", "networkx", "pandas", "scikit-learn", "folium", "streamlit-folium", "requests"):
            self.assertIn(pkg, req)


if __name__ == "__main__":
    unittest.main()
