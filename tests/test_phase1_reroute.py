"""
Phase 1: re-planning from where the ship is (Panama close / reopen), turn rules, stay-vs-change hysteresis.

    python -m pytest tests/test_phase1_reroute.py -v

Scenario: Rotterdam -> Los Angeles normally goes via Panama.
"""
import unittest

import voyage as V
from geo_assets import on_land
from ops import Ops

MID_ATLANTIC = {"SA", "NATL", "ATL1", "ATL5", "ATL6"}      # the detour must not backtrack across the Atlantic


def _ops_rtm_lax(close_day=10, reopen_after_days=4):
    """Panama closed at ``close_day``; returns Ops positioned ``reopen_after_days`` later, still closed."""
    o = Ops()
    o.create_voyage("RTM", "LAX")
    o.advance_hours(close_day * 24)
    o.set_disruption("panama_closure", True)
    o.advance_hours(reopen_after_days * 24)
    return o


def _sail_nodes(legs):
    return {n for l in legs if l.kind == "sail" for n in (l.x, l.y)}


def _dist_to(o, node):
    s = o.state()
    n = o.G.nodes[node]
    return V.haversine_nm(s.lat, s.lon, n["lat"], n["lon"])


class TestLaneNetwork(unittest.TestCase):
    def test_south_american_atlantic_lanes_exist(self):
        G = V.base_graph()
        self.assertTrue(G.has_edge("ANN", "GUI1"))
        self.assertTrue(G.has_edge("GUI1", "SAT4"))

    def test_new_lane_nodes_are_offshore(self):
        G = V.base_graph()
        self.assertFalse(on_land(G.nodes["GUI1"]["lat"], G.nodes["GUI1"]["lon"]))

    def test_cape_horn_detour_hugs_south_america(self):
        o = _ops_rtm_lax()
        self.assertEqual(o.route_corridor(), "Cape Horn")
        _, ahead, _ = V.split_at(o.voyage, o.G)
        self.assertFalse(_sail_nodes(ahead) & MID_ATLANTIC, _sail_nodes(ahead) & MID_ATLANTIC)


class TestReopenTurnsTowardPanama(unittest.TestCase):
    def setUp(self):
        self.o = _ops_rtm_lax()
        self.d_before = _dist_to(self.o, "COL")
        self.msg = self.o.set_disruption("panama_closure", False)
        self.v = self.o.voyage

    def test_recommendation_and_corridor(self):
        self.assertEqual(self.v.recommendation, "change_course")
        self.assertEqual(self.o.route_corridor(), "Panama Canal")

    def test_ship_moves_toward_panama_after_reopen(self):
        self.o.advance_hours(24)
        self.assertLess(_dist_to(self.o, "COL"), self.d_before - 100)

    def test_eta_beats_continuing_to_cape_horn(self):
        hold = next(c for c in self.v.candidates if c.name == "Hold course")
        chosen = next(c for c in self.v.candidates if c.chosen)
        self.assertLess(chosen.summary.hours, hold.summary.hours)
        self.assertEqual(hold.summary.corridor, "Cape Horn")

    def test_u_turn_costs_time(self):
        _, ahead, _ = V.split_at(self.v, self.o.G)
        first = ahead[0]
        self.assertGreaterEqual(first.wait_h, V.TURN_PENALTY_H)
        self.assertIn("turn", first.note.lower())

    def test_banner_states_decision_and_why(self):
        for needle in ("CHANGE COURSE NOW", "turning", "saves", "vs continuing via Cape Horn"):
            self.assertIn(needle, self.v.decision)
        self.assertIn("REROUTED", self.msg)


class TestStayOnCourse(unittest.TestCase):
    def _deep_south(self):
        o = Ops()
        o.create_voyage("RTM", "LAX")
        o.advance_hours(10 * 24)
        o.set_disruption("panama_closure", True)
        for _ in range(200):                                    # sail the detour until well past the Brazil bulge
            o.advance_hours(12)
            _, ahead, _ = V.split_at(o.voyage, o.G)
            if "SAT2" not in _sail_nodes(ahead):
                break
        return o

    def test_reopen_after_point_of_no_return_stays(self):
        o = self._deep_south()
        o.set_disruption("panama_closure", False)
        self.assertEqual(o.voyage.recommendation, "stay")
        self.assertEqual(o.route_corridor(), "Cape Horn")
        self.assertIn("STAY ON COURSE", o.voyage.decision)

    def test_huge_threshold_keeps_course_even_when_turning_would_save(self):
        o = _ops_rtm_lax()
        o.update_config(change_course_threshold_usd=1e9)
        o.set_disruption("panama_closure", False)
        self.assertEqual(o.voyage.recommendation, "stay")
        self.assertEqual(o.route_corridor(), "Cape Horn")

    def test_zero_threshold_turns(self):
        o = _ops_rtm_lax()
        o.update_config(change_course_threshold_usd=0.0)
        o.set_disruption("panama_closure", False)
        self.assertEqual(o.voyage.recommendation, "change_course")


class TestNoOscillation(unittest.TestCase):
    def test_replanning_twice_gives_same_answer(self):
        o = _ops_rtm_lax()
        o.set_disruption("panama_closure", False)
        sig, n = V.route_signature(o.voyage), o.voyage.reroutes
        o.reroute()
        self.assertEqual(V.route_signature(o.voyage), sig)
        self.assertEqual(o.voyage.reroutes, n)
        self.assertEqual(o.voyage.recommendation, "stay")

    def test_irrelevant_toggles_leave_route_alone(self):
        o = Ops()
        o.create_voyage("RTM", "LAX")
        o.advance_hours(10 * 24)
        lanes = lambda: [(l.x, l.y) for l in V.split_at(o.voyage, o.G)[1] if l.kind == "sail"]
        before = lanes()
        for _ in range(3):
            o.set_disruption("hormuz_closure", True)
            o.advance_hours(2)
            o.set_disruption("hormuz_closure", False)
            o.advance_hours(2)
        after = lanes()
        self.assertEqual(after, before[len(before) - len(after):])      # same lanes, just fewer of them left
        self.assertEqual(o.voyage.reroutes, 0)

    def test_close_reopen_close_is_consistent(self):
        o = Ops()
        o.create_voyage("RTM", "LAX")
        o.advance_hours(10 * 24)
        corridors = []
        for state in (True, False, True):
            o.set_disruption("panama_closure", state)
            corridors.append(o.route_corridor())
            o.advance_hours(6)
        self.assertEqual(corridors, ["Cape Horn", "Panama Canal", "Cape Horn"])


class TestSunkCosts(unittest.TestCase):
    def test_reroute_does_not_change_money_already_spent(self):
        o = _ops_rtm_lax()
        G = o.G
        before = V.projected(o.voyage, G)
        o.set_disruption("panama_closure", False)
        after = V.projected(o.voyage, G)
        self.assertAlmostEqual(after["spent_usd"], before["spent_usd"], delta=1.0)
        self.assertAlmostEqual(after["total_usd"], after["spent_usd"] + after["remaining_usd"], delta=1.0)
        self.assertGreater(after["spent_usd"], 0)

    def test_fuel_and_distance_sailed_are_continuous(self):
        o = _ops_rtm_lax()
        G = o.G
        sailed_before = o.state().traveled_nm
        fuel_before = sum(l.fuel_t for l in V.split_at(o.voyage, G)[0])
        o.set_disruption("panama_closure", False)
        self.assertAlmostEqual(o.state().traveled_nm, sailed_before, delta=1.0)
        self.assertAlmostEqual(sum(l.fuel_t for l in V.split_at(o.voyage, G)[0]), fuel_before, delta=0.5)


class TestCanalRules(unittest.TestCase):
    def _inside_panama(self):
        o = Ops()
        o.create_voyage("RTM", "LAX")
        G = o.G
        for _ in range(600):
            o.advance_hours(1)
            _, _, pos = V.split_at(o.voyage, G)
            if pos.node is None and (pos.x, pos.y) == ("COL", "BAL") and 0.2 < pos.f < 0.8:
                return o
        self.fail("ship never found mid-canal")

    def test_no_u_turn_inside_a_canal(self):
        o = self._inside_panama()
        o.set_disruption("panama_closure", True)
        self.assertNotEqual(o.voyage.status, "blocked")
        _, ahead, _ = V.split_at(o.voyage, o.G)
        first = next(l for l in ahead if l.kind == "sail")
        self.assertEqual((first.x, first.y), ("COL", "BAL"))          # finishes the transit, does not reverse
        self.assertIn("Panama Canal", o.route_corridor())


class TestReviewFindings(unittest.TestCase):
    """Regressions found in the Phase 1 code review."""

    def _ctx(self, **cfg):
        return V.build_context(V.VoyageConfig(**cfg), __import__("events").default_settings())

    def test_turn_penalty_does_not_double_count_a_canal_queue(self):
        ctx = self._ctx()
        leg = V._make_leg(ctx, "COL", "BAL", 0.0, 1.0, entered=False)          # Panama lane: queued entry
        self.assertGreater(leg.wait_h, 0)
        t = V._turn_leg(ctx, leg)
        self.assertAlmostEqual(t.wait_h, leg.wait_h + V.TURN_PENALTY_H)
        self.assertAlmostEqual(t.time_usd, t.hours * ctx.agent.delay_penalty_usd_per_hr, delta=1.0)

    def test_repricing_a_partly_served_turn_keeps_only_the_rest(self):
        ctx = self._ctx()
        leg = V._turn_leg(ctx, V._make_leg(ctx, "ANN", "GUI1", 0.5, 1.0, entered=True), 1.5)
        again = V.reprice(ctx, [leg]).legs[0]
        self.assertAlmostEqual(again.wait_h, 1.5)
        self.assertTrue(again.turn)

    def test_threshold_is_judged_in_the_active_objective(self):
        hold = V.RouteSummary("hold", True, hours=300, fuel_usd=1.0e6, time_usd=1.0e6)
        alt = V.RouteSummary("alt", True, hours=250, fuel_usd=1.2e6, time_usd=0.8e6)   # faster, same all-in cost
        pen = None
        for pref, expected in (("fastest", True), ("cheapest", False), ("balanced", False)):
            ctx = self._ctx(preference=pref)
            pen = ctx.agent.delay_penalty_usd_per_hr
            self.assertEqual(V._worth_changing(ctx, hold, alt), expected, (pref, pen))

    def test_changing_the_objective_midvoyage_bypasses_the_threshold(self):
        o = _ops_rtm_lax()
        o.update_config(change_course_threshold_usd=1e9)
        o.set_disruption("panama_closure", False)
        self.assertEqual(o.voyage.recommendation, "stay")
        o.update_config(preference="cheapest")                  # an explicit new objective is a new decision
        self.assertEqual(o.voyage.recommendation, "change_course")
        self.assertEqual(o.route_corridor(), "Panama Canal")


class TestNothingTeleports(unittest.TestCase):
    def test_position_continuous_across_reroute(self):
        o = _ops_rtm_lax()
        a = o.state()
        o.set_disruption("panama_closure", False)
        b = o.state()
        self.assertLess(V.haversine_nm(a.lat, a.lon, b.lat, b.lon), 1.0)

    def test_every_leg_of_new_route_is_a_graph_lane(self):
        o = _ops_rtm_lax()
        o.set_disruption("panama_closure", False)
        G = o.G
        for l in o.voyage.legs:
            if l.kind == "sail":
                self.assertTrue(G.has_edge(l.x, l.y), (l.x, l.y))


if __name__ == "__main__":
    unittest.main()
