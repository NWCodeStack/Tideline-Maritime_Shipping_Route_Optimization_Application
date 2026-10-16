"""
Coverage-gap tests: LLM provider (fake SDK), model persistence, weather/news edge cases, and Ops read-only helpers.

    python -m pytest tests/test_coverage_gaps.py -v
"""
import json
import os
import pickle
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import requests

import intel
import llm
import ml_models
from ops import Ops, default_settings


# ------------------------------------------------------------------------------------------------- llm.py
class _FakeAPIError(Exception):
    pass


class _FakeBadRequest(_FakeAPIError):
    pass


def _fake_anthropic(create):
    """A stand-in for the anthropic SDK: ``create`` handles both beta and plain message calls."""
    mod = types.SimpleNamespace(APIError=_FakeAPIError, BadRequestError=_FakeBadRequest)
    client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(create=create)),
                                   messages=types.SimpleNamespace(create=create))
    mod.Anthropic = lambda **kw: client
    return mod


def _resp(text, stop_reason="end_turn"):
    return types.SimpleNamespace(stop_reason=stop_reason, content=[types.SimpleNamespace(type="text", text=text)])


def _provider(create):
    with mock.patch.dict(sys.modules, {"anthropic": _fake_anthropic(create)}):
        return llm.AnthropicProvider("sk-test")


class TestAnthropicProvider(unittest.TestCase):
    def test_extract_ops_parses_structured_json(self):
        payload = {"operations": [{"op": "start", "args": {}}], "reply": "ok"}
        p = _provider(lambda **kw: _resp(json.dumps(payload)))
        self.assertEqual(p.extract_ops("go", "{}"), payload)
        self.assertIn(llm.DEFAULT_MODEL, p.name)

    def test_extract_ops_returns_none_on_invalid_json_and_records_error(self):
        p = _provider(lambda **kw: _resp("not json"))
        self.assertIsNone(p.extract_ops("go", "{}"))
        self.assertEqual(p.last_error, "JSONDecodeError")

    def test_refusal_and_empty_content_give_none(self):
        self.assertIsNone(_provider(lambda **kw: _resp("x", stop_reason="refusal")).explain("hi", "facts"))
        empty = types.SimpleNamespace(stop_reason="end_turn", content=[])
        self.assertIsNone(_provider(lambda **kw: empty).extract_ops("go", "{}"))

    def test_api_errors_degrade_to_none(self):
        def boom(**kw):
            raise _FakeAPIError("down")
        p = _provider(boom)
        self.assertIsNone(p.extract_ops("go", "{}"))
        self.assertIsNone(p.explain("hi", "facts"))
        self.assertEqual(p.last_error, "_FakeAPIError")

    def test_beta_unsupported_falls_back_to_plain_request(self):
        calls = []

        def create(**kw):
            calls.append("beta" if "betas" in kw else "plain")
            if "betas" in kw:
                raise TypeError("unexpected keyword 'betas'")
            return _resp("answer")
        p = _provider(create)
        self.assertEqual(p.explain("hi", "facts"), "answer")
        self.assertEqual(calls, ["beta", "plain"])

    def test_output_schema_is_closed(self):
        self.assertFalse(llm.OUTPUT_SCHEMA["additionalProperties"])
        self.assertFalse(llm.ARGS_SCHEMA["additionalProperties"])


class TestGetProvider(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        self._env.stop()

    def test_no_key_means_command_mode(self):
        p, status = llm.get_provider({})
        self.assertIsNone(p)
        self.assertIn("no LLM key", status)

    def test_secrets_access_error_is_swallowed(self):
        class Broken:
            def __contains__(self, k):
                raise RuntimeError("no secrets.toml")
        p, _ = llm.get_provider(Broken())
        self.assertIsNone(p)

    def test_key_from_secrets_builds_provider(self):
        with mock.patch.dict(sys.modules, {"anthropic": _fake_anthropic(lambda **kw: _resp("x"))}):
            p, status = llm.get_provider({"ANTHROPIC_API_KEY": "sk-test", "TIDELINE_LLM_MODEL": "m-1"})
        self.assertEqual(p.model, "m-1")
        self.assertIn("LLM assist", status)

    def test_missing_sdk_degrades_to_command_mode(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        with mock.patch.dict(sys.modules, {"anthropic": None}):          # import raises ImportError
            p, status = llm.get_provider(None)
        self.assertIsNone(p)
        self.assertIn("not installed", status)


# ------------------------------------------------------------------------------------------- ml_models.py
class TestModelPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ml_models.train_fuel_model()

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = ml_models.save_model(self.model, Path(d) / "sub" / "m.pkl")
            loaded = ml_models.load_model(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.sklearn_version, self.model.sklearn_version)

    def test_load_missing_corrupt_or_foreign_pickle_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ml_models.load_model(Path(d) / "absent.pkl"))
            bad = Path(d) / "bad.pkl"
            bad.write_bytes(b"not a pickle")
            self.assertIsNone(ml_models.load_model(bad))
            foreign = Path(d) / "foreign.pkl"
            foreign.write_bytes(pickle.dumps({"a": 1}))
            self.assertIsNone(ml_models.load_model(foreign))

    def test_load_rejects_other_sklearn_version(self):
        with tempfile.TemporaryDirectory() as d:
            path = ml_models.save_model(self.model, Path(d) / "m.pkl")
            with mock.patch.object(ml_models.sklearn, "__version__", "0.0.0"):
                self.assertIsNone(ml_models.load_model(path))

    def test_load_or_train_reuses_valid_pickle_and_tolerates_readonly(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pkl"
            ml_models.save_model(self.model, path)
            with mock.patch.object(ml_models, "train_fuel_model") as train:
                self.assertIsNotNone(ml_models.load_or_train_model(path))
                train.assert_not_called()
            with mock.patch.object(ml_models, "load_model", return_value=None), \
                    mock.patch.object(ml_models, "train_fuel_model", return_value=self.model), \
                    mock.patch.object(ml_models, "save_model", side_effect=OSError("read-only")):
                self.assertIs(ml_models.load_or_train_model(path), self.model)


# ---------------------------------------------------------------------------------------------- intel.py
def _http(status=200, json_data=None, content=b"", json_error=False):
    r = mock.Mock(status_code=status, content=content)
    if json_error:
        r.json.side_effect = ValueError("bad json")
    else:
        r.json.return_value = json_data
    return r


class TestIntelHttp(unittest.TestCase):
    def setUp(self):
        self._off = mock.patch.dict(os.environ, {"TIDELINE_OFFLINE": "0"})
        self._off.start()
        intel._DOWN_UNTIL.clear()

    def tearDown(self):
        self._off.stop()
        intel._DOWN_UNTIL.clear()

    def test_offline_flag_blocks_network(self):
        with mock.patch.dict(os.environ, {"TIDELINE_OFFLINE": "1"}):
            self.assertTrue(intel.offline())
            with self.assertRaises(intel.IntelError):
                intel._get("http://x", {})

    def test_get_status_handling(self):
        with mock.patch("intel.requests.get", return_value=_http(429)):
            with self.assertRaisesRegex(intel.IntelError, "429"):
                intel._get("http://x", {})
        with mock.patch("intel.requests.get", return_value=_http(404)):
            with self.assertRaisesRegex(intel.IntelError, "HTTP 404"):
                intel._get("http://x", {})
        with mock.patch("intel.requests.get", return_value=_http(503)) as g, mock.patch("intel.time.sleep"):
            with self.assertRaisesRegex(intel.IntelError, "HTTP 503"):
                intel._get("http://x", {})
            self.assertEqual(g.call_count, 2)                     # one retry on 5xx

    def test_fetch_marine_ok_with_wind_failure_tolerated(self):
        marine = {"hourly": {"time": ["2026-01-01T00:00"], "wave_height": [1.0]}}

        def fake(url, **kw):
            if "marine" in url:
                return _http(200, [marine])
            raise requests.ConnectionError("wind down")
        with mock.patch("intel.requests.get", side_effect=fake), mock.patch("intel.time.sleep"):
            raw = intel.fetch_marine(((10.0, 60.0),))
        self.assertIsNone(raw["wind"])
        self.assertEqual(len(raw["marine"]), 1)

    def test_fetch_marine_error_shapes(self):
        with mock.patch("intel.requests.get", return_value=_http(200, json_error=True)):
            with self.assertRaisesRegex(intel.IntelError, "non-JSON"):
                intel.fetch_marine(((10.0, 60.0),))
        with mock.patch("intel.requests.get", return_value=_http(200, {"reason": "out of range"})):
            with self.assertRaisesRegex(intel.IntelError, "out of range"):
                intel.fetch_marine(((10.0, 60.0),))

    def test_gdelt_parses_articles_and_skips_incomplete(self):
        data = {"articles": [
            {"title": "Suez Canal closed", "url": "http://a", "domain": "a.com", "seendate": "20260101T120000Z"},
            {"title": "Bad date", "url": "http://b", "seendate": "garbage"},
            {"title": "", "url": "http://c"},
        ]}
        with mock.patch("intel.requests.get", return_value=_http(200, data)):
            arts = intel.fetch_news_gdelt()
        self.assertEqual([a["title"] for a in arts], ["Suez Canal closed", "Bad date"])
        self.assertIsNotNone(arts[0]["published"])
        self.assertIsNone(arts[1]["published"])
        self.assertTrue(arts[0]["advisory"])

    def test_rss_parses_and_strips_source_suffix(self):
        xml = (b"<rss><channel><item><title>Red Sea attack hits tanker - Reuters</title><link>http://r</link>"
               b"<pubDate>Mon, 01 Jan 2026 12:00:00 GMT</pubDate><source>Reuters</source></item>"
               b"<item><title>No link</title></item>"
               b"<item><title>Hormuz tension</title><link>http://h</link><pubDate>nonsense</pubDate></item>"
               b"</channel></rss>")
        with mock.patch("intel.requests.get", return_value=_http(200, content=xml)):
            arts = intel.fetch_news_rss()
        self.assertEqual([a["title"] for a in arts], ["Red Sea attack hits tanker", "Hormuz tension"])
        self.assertEqual(arts[0]["source"], "Reuters")
        self.assertIsNone(arts[1]["published"])
        with mock.patch("intel.requests.get", return_value=_http(200, content=b"<not xml")):
            with self.assertRaisesRegex(intel.IntelError, "parsed"):
                intel.fetch_news_rss()

    def test_fetch_news_cooldown_and_all_failed(self):
        art = intel._article("Suez Canal closed", "http://a", "A", None, "x")
        with mock.patch.object(intel, "fetch_news_gdelt", side_effect=intel.IntelError("429")), \
                mock.patch.object(intel, "fetch_news_rss", return_value=[art]):
            self.assertEqual(intel.fetch_news()["provider"], "Google News RSS")
            self.assertIn("GDELT", intel._DOWN_UNTIL)
            res = intel.fetch_news()                              # GDELT now skipped
            self.assertIn("skipped", res["errors"][0])
        intel._DOWN_UNTIL.clear()
        with mock.patch.object(intel, "fetch_news_gdelt", return_value=[]), \
                mock.patch.object(intel, "fetch_news_rss", return_value=[]):
            with self.assertRaisesRegex(intel.IntelError, "no results"):
                intel.fetch_news()


class TestIntelPure(unittest.TestCase):
    def test_severity_thresholds(self):
        self.assertEqual(intel.severity(None, None), ("Low", 0))
        self.assertEqual(intel.severity(2.5, 0), ("Moderate", 1))
        self.assertEqual(intel.severity(0, 34), ("High", 2))
        self.assertEqual(intel.severity(6, 0), ("Severe", 3))
        self.assertEqual(intel.severity(0, 48), ("Severe", 3))

    def test_operational_impact(self):
        self.assertEqual(intel.operational_impact(None), "")
        self.assertIn("model estimate", intel.operational_impact(3.0))

    def test_assess_weather_window_edges_and_missing_data(self):
        dep = datetime(2026, 1, 1, tzinfo=timezone.utc)
        times = [(dep + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(0, 24)]
        hourly = {"time": times, **{v: [1.0] * 24 for v in intel.MARINE_VARS}}
        coastal = {"time": times, **{v: [None] * 24 for v in intel.MARINE_VARS}}
        raw = {"marine": [{"hourly": hourly}, {"hourly": coastal}, {"hourly": hourly}], "wind": None,
               "fetched_at": "2026-01-01T00:00:00"}
        pts = [{"lat": 0, "lon": 0, "t": 5, "label": "A"},          # inside window
               {"lat": 1, "lon": 1, "t": 5, "label": "B"},          # coastal: no wave data
               {"lat": 2, "lon": 2, "t": 500, "label": "C"},        # beyond horizon
               {"lat": 3, "lon": 3, "t": 5, "label": "D"}]          # no marine record at all
        w = intel.assess_weather(raw, pts, dep)
        statuses = [p["status"] for p in w["points"]]
        self.assertEqual(statuses[0], "forecast")
        self.assertIn("no marine data", statuses[1])
        self.assertEqual(statuses[2], "beyond forecast horizon")
        self.assertEqual(statuses[3], "beyond forecast horizon")
        self.assertEqual(w["status"], "partial")                    # wind is None
        self.assertIn("Wind unavailable", w["note"])
        self.assertIsNotNone(w["worst"])
        early = intel.assess_weather(raw, [{"lat": 0, "lon": 0, "t": -48, "label": "E"}], dep)
        self.assertEqual(early["points"][0]["status"], "before forecast window")
        self.assertEqual(early["status"], "unavailable")
        self.assertIn("forecast window", early["note"])

    def test_weather_summary_branches(self):
        dep = datetime(2026, 1, 1, tzinfo=timezone.utc)
        hourly = {"time": [dep.strftime("%Y-%m-%dT%H:%M")], **{v: [3.0] for v in intel.MARINE_VARS}}
        raw = {"marine": [{"hourly": hourly}, {"hourly": hourly}], "wind": None,
               "fetched_at": "2026-01-01T00:00:00"}
        w = intel.assess_weather(raw, [{"lat": 0, "lon": 0, "t": 0, "label": "Ship"},
                                       {"lat": 1, "lon": 1, "t": 900, "label": "Far"}], dep)
        text = intel.weather_summary(dict(w, stale=True))
        self.assertIn("Near the ship", text)
        self.assertIn("Worst ahead", text)
        self.assertIn("outside the forecast window", text)
        self.assertIn("stale cached copy", text)

    def test_news_summary_region_filter_demo_and_empty(self):
        demo = intel.demo_news()
        self.assertTrue(demo["demo"])
        self.assertIn("DEMO sample headlines", intel.news_summary(demo, ["Suez Canal"]))
        self.assertIn("No recent headlines matched", intel.news_summary(demo, [], region="malacca"))
        self.assertIn("Suez", intel.news_summary(demo, ["Suez Canal"], region="suez"))

    def test_rank_news_dedupes_and_prefers_on_route(self):
        a = intel._article("Suez Canal queue grows", "u1", "s", datetime(2026, 1, 2, tzinfo=timezone.utc), "p")
        dup = intel._article("SUEZ CANAL queue grows", "u2", "s", None, "p")
        other = intel._article("Generic shipping news", "u3", "s", datetime(2026, 1, 3, tzinfo=timezone.utc), "p")
        ranked = intel.rank_news([other, a, dup], ["Suez Canal"])
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["title"], "Suez Canal queue grows")
        self.assertEqual(intel.advisories(ranked), [])               # no disruption wording -> no advisory


# ------------------------------------------------------------------------------------------------ ops.py
class TestOpsHelpers(unittest.TestCase):
    def setUp(self):
        self.o = Ops()
        self.o.create_voyage("RTM", "SHA")
        self.o.advance_hours(50)

    def test_storm_and_congestion_roundtrip(self):
        self.assertEqual(self.o.clear_storms(), "No storms to clear.")
        self.o.add_storm(wave_m=7.0, radius_nm=200)
        self.assertEqual(len(self.o.settings["storm_cells"]), 1)
        self.assertIn("cleared", self.o.clear_storms())
        self.assertEqual(self.o.set_congestion("ZZZ", 10), "Unknown port.")
        self.o.set_congestion("SHA", 24)
        self.assertEqual(self.o.settings["port_congestion"]["SHA"], 24.0)
        self.o.set_congestion("SHA", 0)
        self.assertNotIn("SHA", self.o.settings["port_congestion"])

    def test_storm_without_remaining_route(self):
        self.o.advance_hours(10_000)
        self.assertEqual(self.o.voyage.status, "arrived")
        self.assertIn("No remaining route", self.o.add_storm())

    def test_clear_all_and_apply_settings(self):
        self.assertEqual(self.o.clear_all(), "No active scenarios.")
        self.o.set_disruption("suez_closure", True)
        self.assertIn("Cleared scenarios", self.o.clear_all())
        self.assertEqual(self.o.settings, default_settings())
        msg = self.o.apply_settings({"suez_closure": True}, label="test preset")
        self.assertIn("SCENARIO (simulated): test preset", msg)

    def test_apply_settings_rolls_back_on_error(self):
        before = self.o.settings
        with mock.patch.object(self.o, "_reevaluate", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.o.apply_settings({"suez_closure": True}, label="x")
        self.assertEqual(self.o.settings, before)

    def test_compare_via_and_options(self):
        self.assertEqual(self.o.compare_via("nowhere"), "Unknown corridor.")
        text = self.o.compare_via("cape")
        self.assertTrue(text.startswith(("Route via", "Via", "The current", "No feasible")), text)
        opts = self.o.options()
        self.assertTrue(opts)
        self.assertTrue(any(c.chosen for c in opts))

    def test_reroute_changes_preference(self):
        self.o.reroute(preference="cheapest")
        self.assertEqual(self.o.cfg.preference, "cheapest")

    def test_explain_topics(self):
        for topic, needle in (("eta", "ETA"), ("cost", "Projected all-in"), ("risk", "risk score"),
                              ("options", "Routes from the current position"), ("status", "SIMULATED position")):
            self.assertIn(needle, self.o.explain(topic), topic)

    def test_explain_risk_reports_active_scenarios(self):
        self.o.set_disruption("red_sea_escalation", True)
        self.assertIn("Active scenarios", self.o.explain("risk"))


if __name__ == "__main__":
    unittest.main()
