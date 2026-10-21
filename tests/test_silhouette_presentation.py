"""Display masks must not change tracking or fabricate missing silhouettes."""
import copy
import unittest

import numpy as np

from tacticalai.silhouette_presentation import associate_masks


class SilhouetteTests(unittest.TestCase):
    def test_one_mask_cannot_color_two_overlapping_tracks(self):
        rows = [dict(track_id=1, bbox_xyxy=[0, 0, 10, 20]),
                dict(track_id=2, bbox_xyxy=[1, 0, 11, 20])]
        before = copy.deepcopy(rows)
        polygon = np.array([[0, 0], [10, 0], [10, 20], [0, 20]])
        associate_masks(rows, [[0, 0, 10, 20]], [polygon], [.9])
        self.assertIsNotNone(rows[0]['silhouette_xy'])
        self.assertIsNone(rows[1]['silhouette_xy'])
        for a, b in zip(before, rows):
            self.assertTrue(all(b[k] == v for k, v in a.items()))

    def test_distant_mask_is_unknown_not_a_rectangle_fallback(self):
        rows = [dict(track_id=1, bbox_xyxy=[0, 0, 10, 20])]
        associate_masks(rows, [[100, 100, 110, 120]],
                        [np.array([[100, 100], [110, 100], [110, 120]])], [.9])
        self.assertIsNone(rows[0]['silhouette_xy'])
        self.assertIsNone(rows[0]['silhouette_confidence'])
        associate_masks(rows, [], [], [])
        self.assertIsNone(rows[0]['silhouette_xy'])


if __name__ == '__main__':
    unittest.main()
