"""
Phase 7 review follow-ups: the open /code-review findings, one test class each.

    python -m pytest tests/test_phase7_review.py -v

Copilot parsing of playback vs vessel speed, the seek-slider echo guard, port-warning refresh while sailing,
the tick-and-rerun rule, and the shared antimeridian helper.
"""
import os
import time
import unittest
from unittest import mock

os.environ.setdefault("TIDELINE_OFFLINE", "1")

import copilot as C  # noqa: E402
import live  # noqa: E402
import panels as P  # noqa: E402
import voyage as V  # noqa: E402
from ops import Ops  # noqa: E402


def _names(text):
    return [op.name for op in C.parse(text, Ops())]


class TestCopilotPlaybackParsing(unittest.TestCase):
    def test_a_duration_after_playback_is_not_a_speed(self):
        self.assertNotIn("set_playback", _names("Set playback to advance 2 days"))
        self.assertNotIn("set_playback", _names("playback 5 hours ahead"))

    def test_playback_and_knots_in_one_message_both_apply(self):
        names = _names("Set playback to 10 h/s and set speed to 14 knots")
        self.assertIn("set_playback", names)
        self.assertIn("set_speed", names)

    def test_knots_alone_still_means_the_vessel(self):
        self.assertEqual(_names("Set speed to 14 knots"), ["set_speed"])

    def test_statements_starting_with_speed_slow_or_play_are_not_instructions(self):
        for text in ("Speed 14 knots is fine for now.", "Slow steaming at 12 knots saves fuel.",
                     "Play it safe, the Red Sea is risky."):
            self.assertNotEqual(C.classify(text), "action", text)

    def test_a_statement_about_speed_changes_nothing(self):
        o = Ops()
        before = (o.cfg.cruise_speed_kn, o.speed)
        C.handle("Speed 14 knots is fine for now.", o)
        self.assertEqual((o.cfg.cruise_speed_kn, o.speed), before)

    def test_real_playback_instructions_are_still_actions(self):
        for text in ("Play in real time", "Slow the simulation down to 1 h/s", "Speed up playback to 50 h/s",
                     "Set playback to 25 h/s"):
            self.assertEqual(C.classify(text), "action", text)

    def test_speed_up_playback_sets_the_clock(self):
        o = Ops()
        C.handle("Speed up playback to 50 h/s", o)
        self.assertEqual(o.speed, 50.0)


class TestSeekEchoGuard(unittest.TestCase):
    def test_a_value_the_server_just_wrote_is_an_echo_while_playing(self):
        self.assertTrue(live.seek_is_echo(56.1, (50.0, 56.1, 61.0), playing=True))

    def test_nothing_is_an_echo_while_paused(self):
        self.assertFalse(live.seek_is_echo(56.1, (56.1,), playing=False))

    def test_a_different_value_is_a_real_seek(self):
        self.assertFalse(live.seek_is_echo(120.0, (50.0, 56.1, 61.0), playing=True))

    def test_only_the_last_few_written_values_count(self):
        written = ()
        for v in range(0, 100, 10):
            written = live.remember_seek(written, float(v))
        self.assertLessEqual(len(written), live.SEEK_ECHO_WINDOW)
        self.assertFalse(live.seek_is_echo(0.0, written, playing=True))     # an old position can be sought again
        self.assertTrue(live.seek_is_echo(90.0, written, playing=True))


class TestPortWarningSignature(unittest.TestCase):
    def test_signature_ignores_order_and_detail_text(self):
        a = [{"port": "RTM", "kind": "unreachable", "detail": "x"}, {"port": "SHA", "kind": "congested", "detail": "y"}]
        b = [{"port": "SHA", "kind": "congested", "detail": "other"}, {"port": "RTM", "kind": "unreachable", "detail": "z"}]
        self.assertEqual(live.warning_signature(a), live.warning_signature(b))

    def test_signature_changes_when_a_port_becomes_unreachable(self):
        a = [{"port": "RTM", "kind": "congested", "detail": ""}]
        b = [{"port": "RTM", "kind": "unreachable", "detail": ""}]
        self.assertNotEqual(live.warning_signature(a), live.warning_signature(b))

    def test_signature_of_real_warnings_is_hashable_and_stable(self):
        o = Ops()
        ctx = V.build_context(o.cfg, o.settings, o.model)
        s = V.ship_state(o.voyage, o.G)
        sig = live.warning_signature(P.port_warnings(ctx, s.pos, o.settings, o.G))
        self.assertEqual(sig, live.warning_signature(P.port_warnings(ctx, s.pos, o.settings, o.G)))
        hash(sig)


class TestTickAndSync(unittest.TestCase):
    def _playing(self):
        o = Ops()
        o.set_speed(10.0)
        o.start()
        o.tick()                                  # real clock: records the reference
        return o

    def test_no_rerun_on_an_ordinary_tick(self):
        o = self._playing()
        o._last_tick = time.monotonic() - 0.5
        with mock.patch("live.st.rerun") as rerun:
            live._tick_and_sync(o)
        rerun.assert_not_called()

    def test_a_stall_resync_reruns_the_page(self):
        o = self._playing()
        o._last_tick = time.monotonic() - 10.0
        epoch = o.epoch
        with mock.patch("live.st.rerun") as rerun:
            live._tick_and_sync(o)
        self.assertGreater(o.epoch, epoch)
        rerun.assert_called_once()

    def test_arrival_reruns_the_page(self):
        o = self._playing()
        o.set_speed(200.0)
        o.advance_hours(o.voyage.total_hours - 1.0)
        o._last_tick = time.monotonic() - 2.0
        with mock.patch("live.st.rerun") as rerun:
            live._tick_and_sync(o)
        self.assertFalse(o.playing)
        rerun.assert_called_once()

    def test_a_paused_ops_never_reruns(self):
        o = Ops()
        with mock.patch("live.st.rerun") as rerun:
            live._tick_and_sync(o)
        rerun.assert_not_called()


class TestSharedUnwrap(unittest.TestCase):
    def test_geo_unwrap_makes_longitudes_continuous(self):
        from geo import unwrap
        self.assertEqual(unwrap([(0.0, 175.0), (1.0, -175.0), (2.0, -170.0)]),
                         [(0.0, 175.0), (1.0, 185.0), (2.0, 190.0)])

    def test_folium_map_and_anim_use_the_same_helper(self):
        import anim
        import folium_map
        from geo import unwrap
        self.assertIs(folium_map.unwrap, unwrap)
        self.assertIs(anim.unwrap, unwrap)


if __name__ == "__main__":
    unittest.main()
