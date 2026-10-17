"""
Phase 7: continuous playback.

    python -m pytest tests/test_phase7_playback.py -v

Covers the playback speed range, the settle-the-clock rule (no lost time on pause / speed change / reroute), the
animation track and contract the browser interpolates, the copilot playback op, the slider, and that the map layer
is byte-stable between ticks. The browser-side loop (frame pacing, loop stacking) is checked in the built-in browser.
"""
import json
import os
import random
import time
import unittest

os.environ.setdefault("TIDELINE_OFFLINE", "1")

import anim  # noqa: E402
import copilot as C  # noqa: E402
import ops as OPS  # noqa: E402
import ui  # noqa: E402
import voyage as V  # noqa: E402
from data_engine import haversine_nm  # noqa: E402
from ops import Ops  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _turned_voyage():
    """Rotterdam -> Los Angeles with Panama closed then reopened: done legs, a Cape Horn detour and a U-turn."""
    o = Ops()
    o.create_voyage("RTM", "LAX")
    o.advance_hours(10 * 24)
    o.set_disruption("panama_closure", True)
    o.advance_hours(4 * 24)
    o.set_disruption("panama_closure", False)
    return o


# ------------------------------------------------------------------------------------------------ speed range
class TestSpeedRange(unittest.TestCase):
    def test_fractional_speed_is_accepted(self):
        o = Ops()
        o.set_speed(2.5)
        self.assertEqual(o.speed, 2.5)

    def test_speed_is_clamped_to_real_time_and_200(self):
        o = Ops()
        o.set_speed(0.0)
        self.assertEqual(o.speed, OPS.MIN_SPEED_H_PER_S)
        o.set_speed(-5)
        self.assertEqual(o.speed, OPS.MIN_SPEED_H_PER_S)
        o.set_speed(1e6)
        self.assertEqual(o.speed, 200.0)
        self.assertAlmostEqual(OPS.MIN_SPEED_H_PER_S, 1 / 3600)

    def test_quarter_hour_per_second_for_four_seconds_is_one_hour(self):
        o = Ops()
        o.set_speed(0.25)
        o.start()
        o.tick(now=100.0)                       # arms the clock
        for k in range(1, 5):
            o.tick(now=100.0 + k)
        self.assertAlmostEqual(o.voyage.t, 1.0, places=6)

    def test_stops_cover_real_time_to_200_and_are_sorted(self):
        values = [v for _, v in OPS.SPEED_STOPS]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[0], 1 / 3600)
        self.assertEqual(values[-1], 200.0)
        self.assertEqual(OPS.SPEED_STOPS[0][0], "Real time")
        self.assertEqual(OPS.SPEEDS["10 h/s"], 10.0)            # the old presets keep their labels
        self.assertLessEqual(values[1], 0.1)

    def test_nearest_stop_uses_log_distance_and_never_throws(self):
        self.assertEqual(OPS.nearest_stop_label(10.0), "10 h/s")
        self.assertEqual(OPS.nearest_stop_label(7.0), "5 h/s")
        self.assertEqual(OPS.nearest_stop_label(3.0), "2 h/s")
        self.assertEqual(OPS.nearest_stop_label(OPS.MIN_SPEED_H_PER_S), "Real time")
        self.assertEqual(OPS.nearest_stop_label(1000.0), "200 h/s")
        for v in (1e-9, 0.0, 0.0001, 0.37, 3.3, 99.0, 150.0, 5e3):
            self.assertIn(OPS.nearest_stop_label(v), OPS.SPEEDS)

    def test_captions_are_plain_words(self):
        self.assertIn("10 h/s", OPS.speed_caption(10.0))
        self.assertIn("2.4 s", OPS.speed_caption(10.0))                # 1 day = 24 h / 10 h/s
        self.assertIn("real time", OPS.speed_caption(OPS.MIN_SPEED_H_PER_S).lower())
        self.assertIn("SIMULATED", OPS.speed_caption(OPS.MIN_SPEED_H_PER_S))


# ------------------------------------------------------------------------------------------------ settle the clock
class TestSettleTheClock(unittest.TestCase):
    def test_pause_settles_time_since_the_last_tick(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)
        o.pause(now=102.5)
        self.assertAlmostEqual(o.voyage.t, 25.0, places=6)
        self.assertFalse(o.playing)

    def test_speed_change_uses_the_old_speed_for_the_elapsed_time(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)
        o.set_speed(2.0, now=101.0)
        self.assertAlmostEqual(o.voyage.t, 10.0, places=6)
        o.tick(now=102.0)
        self.assertAlmostEqual(o.voyage.t, 12.0, places=6)

    def test_resume_while_playing_keeps_the_time_played_since_the_last_tick(self):
        import time
        o = Ops()
        o.set_speed(100.0)
        o.start()
        o.tick()                                 # real clock: first tick only records the reference
        time.sleep(0.05)
        o.resume()                               # copilot "Resume" while already playing
        self.assertGreater(o.voyage.t, 1.0)      # ~5 h played at 100 h/s, not dropped

    def test_a_manual_clock_is_left_alone_when_no_now_is_given(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)
        o.tick(now=101.0)
        t = o.voyage.t
        o.pause()                                # last tick used a caller-supplied clock: no settle against real time
        self.assertEqual(o.voyage.t, t)

    def test_scenario_change_reroutes_from_the_settled_position(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick()                                 # real clock: arms
        time.sleep(0.3)
        t0 = o.voyage.t
        o.set_event("red_sea_escalation", 0.5)
        self.assertGreaterEqual(o.voyage.t, t0 + 2.0)       # ~0.3 s x 10 h/s was not lost

    def test_stall_cap_asks_the_browser_to_resync(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)
        before = o.resync
        o.tick(now=101.0)
        self.assertEqual(o.resync, before)       # normal tick: nothing to correct
        epoch = o.epoch
        o.tick(now=200.0)                        # long stall: capped at 3 s, so the browser's clock is now ahead
        self.assertEqual(o.resync, before + 1)
        self.assertGreater(o.epoch, epoch)
        self.assertAlmostEqual(o.anchor_h, o.voyage.t, places=6)

    def test_epoch_bumps_on_every_clock_event(self):
        o = Ops()
        seen = [o.epoch]
        for fn in (o.start, o.pause, lambda: o.set_speed(3.0), lambda: o.seek(5.0),
                   lambda: o.advance_hours(2.0), o.reset):
            fn()
            seen.append(o.epoch)
        self.assertEqual(seen, sorted(set(seen)), seen)            # strictly increasing: every event bumped it

    def test_anchor_follows_the_clock_and_snap_flags_are_set(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)
        o.pause(now=101.0)
        self.assertAlmostEqual(o.anchor_h, 10.0, places=6)
        self.assertFalse(o.epoch_snap)                            # pause/resume/speed: ease, never jump
        o.seek(20.0)
        self.assertTrue(o.epoch_snap)                             # seek: snap
        self.assertAlmostEqual(o.anchor_h, 20.0, places=6)
        o.epoch_snap = False
        before = (o.version, o.epoch)
        o.set_disruption("suez_closure", True)
        self.assertGreater(o.version, before[0])                  # the itinerary really changed
        self.assertGreater(o.epoch, before[1])
        self.assertTrue(o.epoch_snap)                             # reroute: new itinerary, snap


# ------------------------------------------------------------------------------------------------ animation track
class TestAnimationTrack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _turned_voyage()
        cls.v, cls.G = cls.o.voyage, cls.o.G
        cls.track = anim.build_track(cls.v, cls.G)

    def test_times_never_decrease_and_cover_the_itinerary(self):
        ts = [s[0] for s in self.track]
        self.assertEqual(ts, sorted(ts))
        self.assertAlmostEqual(ts[0], 0.0)
        self.assertAlmostEqual(ts[-1], self.v.total_hours, places=6)

    def test_first_sample_is_the_origin_and_last_is_the_destination(self):
        la, lo = V.node_ll(self.G, self.v.config.origin)
        self.assertAlmostEqual(self.track[0][1], la, places=6)
        self.assertLess(_angle_diff(self.track[0][2], lo), 1e-6)
        la, lo = V.node_ll(self.G, self.v.config.destination)
        self.assertAlmostEqual(self.track[-1][1], la, places=4)
        self.assertLess(_angle_diff(self.track[-1][2], lo), 1e-4)

    def _check_against_ship_state(self, t):
        v = self.v
        saved = (v.t, v.status)
        try:
            v.t, v.status = t, "underway"
            s = V.ship_state(v, self.G)
        finally:
            v.t, v.status = saved
        got = anim.interpolate(self.track, t)
        self.assertAlmostEqual(got["lat"], s.lat, places=5, msg=f"t={t}")
        self.assertLess(_angle_diff(got["lon"], s.lon), 1e-5, f"t={t}")
        self.assertEqual(got["activity"], s.activity, f"t={t}")
        self.assertLess(_angle_diff(got["course"], s.course_deg), 1e-4, f"t={t}")

    def test_fifty_random_times_match_ship_state(self):
        rng = random.Random(7)
        for _ in range(50):
            self._check_against_ship_state(rng.uniform(1e-3, self.v.total_hours - 1e-3))

    def test_waits_and_holds_match_ship_state(self):
        acc, checked = 0.0, 0
        for leg in self.v.legs:
            if leg.wait_h > 1e-6 or leg.kind == "hold":
                self._check_against_ship_state(acc + (leg.wait_h if leg.kind != "hold" else leg.hours) / 2)
                checked += 1
            acc += leg.hours
        self.assertGreater(checked, 0, "scenario produced no queue / turn / berth waits to check")

    def test_the_scenario_contains_a_turn_around(self):
        self.assertTrue(any(leg.turn for leg in self.v.legs))

    def test_stationary_periods_have_zero_speed(self):
        for a, b in zip(self.track, self.track[1:], strict=False):
            if a[5] in ("canal queue", "turning around", "berth wait", "holding"):
                self.assertAlmostEqual(a[1], b[1], places=6, msg=str(a))
                self.assertLess(_angle_diff(a[2], b[2]), 1e-6, str(a))
                self.assertEqual(a[4], 0.0)

    def test_longitudes_are_unwrapped_across_the_antimeridian(self):
        o = Ops()
        o.create_voyage("SHA", "LAX")
        tr = anim.build_track(o.voyage, o.G)
        lons = [s[2] for s in tr]
        for a, b in zip(lons, lons[1:], strict=False):
            self.assertLessEqual(abs(b - a), 180.0)
        self.assertTrue(max(lons) > 180.0 or min(lons) < -180.0, "Pacific crossing should be unwrapped past 180")

    def test_moving_samples_are_dense_enough_to_stay_on_the_lane(self):
        for a, b in zip(self.track, self.track[1:], strict=False):
            if a[4] > 0:
                self.assertLessEqual(haversine_nm(a[1], a[2], b[1], b[2]), 320.0)

    def test_track_is_json_friendly_and_compact(self):
        text = json.dumps(self.track)
        self.assertLess(len(text), 60_000)


# ------------------------------------------------------------------------------------------------ contract
class TestContract(unittest.TestCase):
    def _c(self, o, **kw):
        return anim.contract_json(anim.build_contract(o, o.G, **kw))

    def test_identical_between_ticks_when_nothing_changed(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)
        a = self._c(o)
        o.tick(now=101.0)
        o.tick(now=102.0)
        b = self._c(o)
        self.assertGreater(o.voyage.t, 0)
        self.assertEqual(a, b)

    def test_changes_on_play_speed_seek_reroute_and_resync(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)
        seen = {self._c(o)}
        for step in (lambda: o.set_speed(25.0, now=101.0),
                     lambda: o.pause(now=102.0),
                     lambda: o.seek(30.0),
                     lambda: o.set_disruption("suez_closure", True),
                     lambda: (o.start(), o.tick(now=300.0), o.tick(now=900.0))):
            step()
            c = self._c(o)
            self.assertNotIn(c, seen)
            seen.add(c)

    def test_fields(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        c = anim.build_contract(o, o.G)
        for k in ("v", "route_version", "epoch", "resync", "snap", "t_anchor_h", "speed_h_per_s", "playing",
                  "stop_h", "status", "track", "acts"):
            self.assertIn(k, c)
        self.assertEqual(c["speed_h_per_s"], 10.0)
        self.assertTrue(c["playing"])
        self.assertEqual(c["route_version"], o.version)
        self.assertAlmostEqual(c["stop_h"], o.voyage.total_hours, places=3)
        self.assertNotIn("perf", c)
        self.assertTrue(anim.build_contract(o, o.G, perf=True)["perf"])

    def test_status_does_not_flip_on_the_first_tick(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        before = anim.build_contract(o, o.G)["status"]
        o.tick(now=1.0)
        o.tick(now=2.0)
        self.assertEqual(anim.build_contract(o, o.G)["status"], before)
        self.assertEqual(before, "active")
        self.assertEqual(anim.build_contract(Ops(), Ops().G)["status"], "ready")

    def test_demo_stop_point_is_in_the_contract_and_the_ship_never_passes_it(self):
        o = Ops()
        o.run_demo(animate=True)
        target = o._demo_target
        c = anim.build_contract(o, o.G)
        self.assertAlmostEqual(c["stop_h"], target, places=3)
        self.assertLess(c["stop_h"], o.voyage.total_hours)
        self.assertLessEqual(anim.clamp_time(c, 1e9), target + 1e-3)
        o.seek(5.0)                                              # scrubbing cancels the pending trigger
        self.assertAlmostEqual(anim.build_contract(o, o.G)["stop_h"], o.voyage.total_hours, places=3)

    def test_clamp_time_extrapolates_from_the_anchor(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        c = anim.build_contract(o, o.G)
        self.assertAlmostEqual(anim.clamp_time(c, 2.0), c["t_anchor_h"] + 20.0, places=6)
        c["playing"] = False
        self.assertAlmostEqual(anim.clamp_time(c, 2.0), c["t_anchor_h"], places=6)
        c["playing"] = True
        self.assertEqual(anim.clamp_time(c, 1e9), c["stop_h"])                # 200 h/s near arrival: no overshoot

    def test_json_is_script_safe(self):
        o = Ops()
        text = self._c(o)
        for bad in ("<", ">", "&"):
            self.assertNotIn(bad, text)

    def test_projected_variant_for_the_svg_map(self):
        o = Ops()
        c = anim.build_contract(o, o.G, project=lambda lat, lon: (lon * 2.0, -lat * 2.0))
        first = c["track"][0]
        la, lo = V.node_ll(o.G, o.voyage.config.origin)
        self.assertAlmostEqual(first[1], lo * 2.0, places=3)
        self.assertAlmostEqual(first[2], -la * 2.0, places=3)


# ------------------------------------------------------------------------------------------------ browser script
class TestBrowserScript(unittest.TestCase):
    def test_script_has_no_network_calls_and_one_loop_guard(self):
        js = anim.JS_CORE
        for banned in ("fetch(", "XMLHttpRequest", "sendBeacon", "WebSocket", "eval(", "innerHTML"):
            self.assertNotIn(banned, js, banned)
        self.assertIn("requestAnimationFrame", js)
        self.assertIn("cancelAnimationFrame", js)
        self.assertIn("__tl", js)

    def test_frame_pacing_constants_are_documented_in_the_script(self):
        js = anim.JS_CORE
        for needle in ("MIN_PX", "SLOW_MS", "FAST_MS", "MIN_FPS", "MAX_FPS"):
            self.assertIn(needle, js)

    def test_leaflet_and_svg_adapters_share_the_core(self):
        self.assertIn(anim.JS_CORE, anim.leaflet_script("{}"))
        self.assertIn(anim.JS_CORE, anim.svg_script("{}"))


# ------------------------------------------------------------------------------------------------ map layer
class TestMapLayerStability(unittest.TestCase):
    def _layer_js(self, o, G):
        import folium
        import streamlit_folium as sf

        import folium_map as FM
        c = anim.build_contract(o, G)
        s = V.ship_state(o.voyage, G)
        m = FM.build_ops_base_map(G, o.settings)
        layer = FM.build_ops_layer(G, o.voyage, [], [], [], s, (), (), "#58aaff", contract=c)
        self.assertIsInstance(layer, folium.FeatureGroup)
        return sf._get_feature_group_string(layer, m, 0)

    def test_layer_string_is_identical_between_ticks_and_changes_with_the_contract(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick(now=100.0)
        a = self._layer_js(o, o.G)
        o.tick(now=101.0)
        o.tick(now=102.0)
        b = self._layer_js(o, o.G)
        self.assertEqual(a, b)
        o.set_speed(50.0, now=103.0)
        self.assertNotEqual(a, self._layer_js(o, o.G))

    def test_animated_layer_has_no_per_tick_ship_marker_or_growing_polylines(self):
        o = Ops()
        o.start()
        import folium
        import folium_map as FM
        s = V.ship_state(o.voyage, o.G)
        layer = FM.build_ops_layer(o.G, o.voyage, [], [], [], s, (), (), "#58aaff", contract=anim.build_contract(o, o.G))
        kinds = {type(c).__name__ for c in layer._children.values()}
        self.assertNotIn("Marker", kinds)                        # the ship and the mint track are drawn by the browser script
        self.assertFalse(any(isinstance(c, folium.PolyLine) and "4de0c8" in str(c.options)
                             for c in layer._children.values()))
        self.assertIn("__tl", self._layer_js(o, o.G))

    def test_legacy_signature_still_draws_a_static_ship(self):
        import folium_map as FM
        o = Ops()
        s = V.ship_state(o.voyage, o.G)
        layer = FM.build_ops_layer(o.G, o.voyage, [], [], [], s, (), (), "#58aaff")
        self.assertTrue(any(type(c).__name__ == "Marker" for c in layer._children.values()))


class TestSvgMap(unittest.TestCase):
    def test_svg_page_embeds_the_contract_and_animates(self):
        from map_view import build_ops_map_html
        o = Ops()
        c = anim.build_contract(o, o.G, project=lambda lat, lon: (lon, -lat))
        s = V.ship_state(o.voyage, o.G)
        page = build_ops_map_html(o.G, o.settings, [], [], ship=s, ship_status="planned", contract=c)
        self.assertIn("requestAnimationFrame", page)
        self.assertIn('"route_version"', page)
        self.assertNotIn("</script><script>alert", page)


# ------------------------------------------------------------------------------------------------ telemetry strip
class TestLiveStrip(unittest.TestCase):
    def test_time_lapse_warning_is_gone(self):
        self.assertNotIn("TIME-LAPSE", ui.live_strip("playing", [("Sim clock", "x")]))


# ------------------------------------------------------------------------------------------------ copilot
class TestCopilotPlayback(unittest.TestCase):
    def test_set_playback_speed(self):
        o = Ops()
        r = C.handle("Set playback speed to 2.5 h/s", o)
        self.assertEqual(o.speed, 2.5, r.text)
        self.assertEqual([n for n, _, _ in r.ops], ["set_playback"])

    def test_play_in_real_time(self):
        o = Ops()
        C.handle("Play in real time", o)
        self.assertEqual(o.speed, OPS.MIN_SPEED_H_PER_S)

    def test_slow_down_and_clamp(self):
        o = Ops()
        C.handle("Slow the simulation down to 1 h/s", o)
        self.assertEqual(o.speed, 1.0)
        r = C.handle("Set playback to 900 h/s", o)
        self.assertLessEqual(o.speed, 200.0, r.text)

    def test_a_question_about_playback_changes_nothing(self):
        o = Ops()
        before = (o.speed, o.playing, o.epoch)
        r = C.handle("What is the playback speed?", o)
        self.assertEqual((o.speed, o.playing, o.epoch), before)
        self.assertTrue(r.text)

    def test_a_what_if_about_playback_changes_nothing(self):
        o = Ops()
        before = o.speed
        C.handle("What if I set playback to 50 h/s?", o)
        self.assertEqual(o.speed, before)

    def test_vessel_knots_still_mean_the_ship_not_the_playback(self):
        o = Ops()
        before = o.speed
        C.handle("Set speed to 14 knots", o)
        self.assertEqual(o.cfg.cruise_speed_kn, 14.0)
        self.assertEqual(o.speed, before)

    def test_schema_exposes_set_playback_with_a_range(self):
        self.assertIn("set_playback", C.SCHEMA)
        with self.assertRaises(C.InvalidOp):
            C.validate(C.Op("set_playback", {"hours_per_s": "fast"}))


# ------------------------------------------------------------------------------------------------ the slider (AppTest)
class TestSlider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        cls.AppTest = AppTest

    def _fresh(self):
        return self.AppTest.from_file(APP, default_timeout=180).run()

    def test_slider_replaces_the_selectbox_on_the_operate_tab(self):
        at = self._fresh()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertIn("play_speed", {w.key for w in at.select_slider})
        self.assertNotIn("play_speed", {w.key for w in at.selectbox})
        sl = at.select_slider(key="play_speed")
        self.assertEqual(sl.options[0], "Real time")
        self.assertEqual(sl.options[-1], "200 h/s")

    def test_moving_the_slider_sets_ops_speed_and_reruns_do_not_change_it(self):
        at = self._fresh()
        at.select_slider(key="play_speed").set_value("2 h/s").run()
        self.assertEqual(at.session_state["ops"].speed, 2.0)
        playing = at.session_state["ops"].playing
        at.run()                                                  # any rerun, as a tab switch would trigger none at all
        self.assertEqual(at.session_state["ops"].speed, 2.0)
        self.assertEqual(at.session_state["ops"].playing, playing)

    def test_a_non_preset_speed_maps_to_the_nearest_stop_without_error(self):
        at = self._fresh()
        at.session_state["ops"].set_speed(2.7)
        at.run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(at.select_slider(key="play_speed").value, "2 h/s")

    def test_no_time_lapse_warning_at_200_hours_per_second(self):
        at = self._fresh()
        o = at.session_state["ops"]
        o.set_speed(200.0)
        o.start()
        at.run()
        self.assertNotIn("TIME-LAPSE", " ".join(m.value for m in at.markdown))


if __name__ == "__main__":
    unittest.main()
