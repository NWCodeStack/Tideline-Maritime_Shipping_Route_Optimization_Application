"""
Phase 4 layout, driven through the real app with Streamlit's headless AppTest (no browser needed).

    python -m pytest tests/test_phase4_layout.py -v

What this cannot see (checked in the built-in browser instead): that the Leaflet map element is the same element
and keeps its zoom while the right-hand tabs are switched.
"""
import os
import unittest

os.environ.setdefault("TIDELINE_OFFLINE", "1")

from streamlit.testing.v1 import AppTest  # noqa: E402

RIGHT_TABS = ["Operate", "Optimize", "Route comparison", "Alerts & intelligence", "Data sources"]


APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _fresh() -> AppTest:
    return AppTest.from_file(APP, default_timeout=180).run()


def _ops(at):
    return at.session_state["ops"]


class TestShell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = _fresh()

    def test_runs_without_exceptions(self):
        self.assertEqual(len(self.at.exception), 0, [e.value for e in self.at.exception])

    def test_top_level_tabs_keep_their_names(self):
        labels = [t.label for t in self.at.tabs]
        self.assertEqual(labels[0], "Live operations")
        self.assertIn("Multi-agent stress test", labels)

    def test_right_hand_tabs_in_order(self):
        # nested sub-tabs (Weather, News & Security) sit between them in the tree, so compare the subsequence
        labels = [t.label for t in self.at.tabs if t.label in RIGHT_TABS]
        self.assertEqual(labels, RIGHT_TABS)

    def test_stress_test_is_not_a_right_hand_tab(self):
        labels = [t.label for t in self.at.tabs]
        self.assertEqual(labels.count("Multi-agent stress test"), 1)
        self.assertLess(labels.index("Live operations"), labels.index(RIGHT_TABS[0]))
        self.assertGreater(labels.index("Multi-agent stress test"), labels.index(RIGHT_TABS[-1]))

    def test_sidebar_holds_no_controls(self):
        sb = self.at.sidebar
        for kind in ("radio", "slider", "selectbox", "toggle", "number_input", "multiselect", "button"):
            self.assertEqual(len(getattr(sb, kind)), 0, kind)

    def test_settings_moved_into_the_main_area(self):
        keys = {w.key for w in self.at.radio} | {w.key for w in self.at.slider} | {w.key for w in self.at.toggle}
        for k in ("cfg_preset", "cfg_tvm", "cfg_carbon", "cfg_ets_on"):
            self.assertIn(k, keys, k)

    def test_the_sidebar_hint_is_gone(self):
        text = " ".join(m.value for m in self.at.markdown)
        self.assertNotIn("Advanced settings: sidebar", text)

    def test_copilot_header_and_input_are_present(self):
        text = " ".join(m.value for m in self.at.markdown)
        self.assertIn("Operations Copilot", text)
        self.assertEqual(len(self.at.chat_input), 1)

    def test_data_sources_table_lists_every_input_with_its_basis(self):
        frames = [d.value for d in self.at.dataframe]
        sources = [f for f in frames if "Basis" in getattr(f, "columns", [])]
        self.assertTrue(sources)
        self.assertGreaterEqual(len(sources[0]), 10)

    def test_declutter_one_help_per_control_not_repeated_sentences(self):
        text = " ".join(m.value for m in self.at.markdown)
        # said once, as the incident picker's help, not as a heading plus a caption plus a banner
        self.assertLessEqual(text.count("Scenarios are SIMULATED, never real closures"), 1)
        self.assertLessEqual(text.count("Advanced settings"), 0)


class TestWiring(unittest.TestCase):
    def test_incident_picker_closes_and_reopens_a_corridor(self):
        at = _fresh()
        picker = next(m for m in at.multiselect if m.key == "inc_picker")
        picker.set_value(["panama_closure"]).run()
        self.assertTrue(_ops(at).settings["panama_closure"])
        next(m for m in at.multiselect if m.key == "inc_picker").set_value([]).run()
        self.assertFalse(_ops(at).settings["panama_closure"])
        self.assertEqual(len(at.exception), 0)

    def test_two_incidents_at_once(self):
        at = _fresh()
        next(m for m in at.multiselect if m.key == "inc_picker").set_value(["suez_blockage", "hormuz_blockade"]).run()
        s = _ops(at).settings
        self.assertTrue(s["suez_blockage"] and s["hormuz_blockade"])

    def test_preset_radio_still_drives_economics(self):
        at = _fresh()
        next(r for r in at.radio if r.key == "cfg_preset").set_value("greenest").run()
        self.assertEqual(_ops(at).cfg.econ.carbon_usd_per_t, 200.0)
        self.assertEqual(len(at.exception), 0)

    def test_copilot_answers_from_the_chat_box(self):
        at = _fresh()
        at.chat_input[0].set_value("What is the safest available route?").run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any(m["role"] == "assistant" for m in _ops(at).chat))

    def test_start_button_starts_the_voyage(self):
        at = _fresh()
        next(b for b in at.button if b.key == "btn_start").click().run()
        self.assertTrue(_ops(at).playing)

    def test_map_state_is_not_reset_by_a_settings_change(self):
        at = _fresh()
        base_before = at.session_state["_base_map"]
        next(r for r in at.radio if r.key == "cfg_preset").set_value("fastest").run()
        self.assertIs(at.session_state["_base_map"], base_before)       # same cached base map object, no rebuild


if __name__ == "__main__":
    unittest.main()
