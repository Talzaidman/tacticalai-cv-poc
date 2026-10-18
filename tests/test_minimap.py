import unittest

import numpy as np

from tacticalai.minimap import apply_roles, compose


class MinimapTests(unittest.TestCase):
    def test_manual_role_interval_handles_known_wrong_person_switch_without_linking_ids(self):
        source = [dict(track_id=96, role="unknown")]
        annotations = dict(intervals=[dict(track_id=96, start_frame=0, end_frame=130, role="staff"),
                                      dict(track_id=96, start_frame=131, end_frame=149, role="player")])
        self.assertEqual(apply_roles(source, annotations, 130)[0]["role"], "staff")
        self.assertEqual(apply_roles(source, annotations, 131)[0]["role"], "player")
        self.assertEqual(source[0]["role"], "unknown")
        self.assertEqual(apply_roles(source, annotations, 150)[0]["role"], "unknown")
        self.assertEqual(apply_roles(source, annotations, 131)[0]["track_id"], 96)
        annotations["intervals"].append(dict(track_id=96, start_frame=120, end_frame=140, role="unknown"))
        with self.assertRaises(ValueError):
            apply_roles(source, annotations, 131)

    def test_minimap_only_displays_current_inside_observed_players(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        court = dict(length_m=28., width_m=15., key_depth_m=5.8, key_width_m=4.9)
        row = dict(track_id=7, role="player", court_xy_m=[5., 7.], court_region="inside", footpoint_image_xy=[200, 250])
        rows = [row, dict(row, track_id=8, role="referee"),
                dict(row, track_id=9, court_xy_m=None, court_region="unknown"),
                dict(row, track_id=10, court_xy_m=[-1, 7], court_region="outside")]
        result, displayed = compose(frame, rows, court, {"method": "manual_keyframe"}, 3., players_only=True, flip_x=True)
        self.assertEqual(displayed, [7])
        self.assertEqual(result.shape, (720, 1280, 3))
        self.assertFalse(frame.any())
        # No renderer state carries a vanished dot into the following frame.
        _, displayed = compose(frame, [], court, {"method": "unavailable"}, 3.04, players_only=True)
        self.assertEqual(displayed, [])


if __name__ == "__main__":
    unittest.main()
