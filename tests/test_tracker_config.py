"""Reject silent no-op association settings and protect experiment controls."""
import unittest

from tacticalai.tracker_config import configure_tracker


class TrackerConfigTests(unittest.TestCase):
    def setUp(self):
        self.defaults = dict(tracker_type="tracktrack", lost_match_thr=0.0,
                             min_track_len=3, track_buffer=30, with_reid=False,
                             model="auto", gmc_method="sparseOptFlow")
        self.config = dict(with_reid=True, reid_model="yolo26s-reid.onnx",
                           track_buffer=60, gmc_method="sparseOptFlow")

    def test_overrides_and_controls_are_preserved_without_mutating_defaults(self):
        self.config["tracker_overrides"] = {"tracktrack": {"lost_match_thr": 0.85}}
        settings = configure_tracker(self.defaults, self.config, "tracktrack")
        self.assertEqual(settings["lost_match_thr"], 0.85)
        self.assertEqual(settings["track_buffer"], 60)
        self.assertEqual(settings["model"], "yolo26s-reid.onnx")
        self.assertEqual(self.defaults["lost_match_thr"], 0.0)

    def test_unused_typo_managed_and_invalid_settings_fail(self):
        for changes in ({"appearance_thresh": 0.9}, {"lost_match_th": 0.85},
                        {"model": "other.onnx"}, {"lost_match_thr": float("nan")},
                        {"lost_match_thr": True}, {"min_track_len": 1.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configure_tracker(self.defaults, dict(self.config,
                    tracker_overrides={"tracktrack": changes}), "tracktrack")


if __name__ == "__main__":
    unittest.main()
