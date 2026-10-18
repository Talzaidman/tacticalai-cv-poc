import unittest

import cv2
import numpy as np

from tacticalai.auto_calibration import FloorTracker, fit_candidate, grid_points
from tacticalai.calibration import transform, project_track


class AutomaticCalibrationTests(unittest.TestCase):
    def test_known_projective_mapping_with_outliers(self):
        world = grid_points()
        expected = np.array([[18., 3., 40.], [1., 12., 60.], [.001, .003, 1.]])
        image = transform(world, expected)
        image[::7] += [110, -80]
        points = [dict(index=i, world_xy_m=w.tolist(), image_xy=p.tolist())
                  for i, (w, p) in enumerate(zip(world, image))]
        inverse, quality = fit_candidate(points, (640, 360))
        self.assertIsNotNone(inverse)
        self.assertLess(quality['inliers'], len(points))
        checks = [[2, 3], [8, 9], [20, 4]]
        np.testing.assert_allclose(transform(transform(checks, expected), inverse), checks, atol=1e-5)
        self.assertEqual(len(world), 91)  # Elevated basket channels never fit.

    def test_rejected_mapping_cannot_project_players(self):
        matrix, quality = fit_candidate([], (640, 360))
        self.assertIsNone(matrix)
        row = project_track(dict(bbox_xyxy=[20, 30, 40, 80]), matrix, quality,
                            dict(length_m=28, width_m=15), (640, 360))
        self.assertIsNone(row['court_xy_m'])
        rng = np.random.default_rng(8)
        points = [dict(index=i, world_xy_m=w.tolist(), image_xy=p.tolist())
                  for i, (w, p) in enumerate(zip(grid_points(), rng.uniform([0, 0], [640, 360], (91, 2))))]
        inverse, quality = fit_candidate(points, (640, 360))
        self.assertIsNone(inverse)
        self.assertEqual(quality['method'], 'unavailable')

    def test_floor_flow_follows_camera_translation_and_expires(self):
        rng = np.random.default_rng(4)
        gray = rng.integers(0, 256, (360, 640), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        court_to_image = np.array([[18., 0, 50.], [0, 18., 30.], [0, 0, 1.]])
        tracker = FloorTracker(max_age=1)
        tracker.reset(gray, np.linalg.inv(court_to_image), [], {})
        shifted = cv2.warpAffine(gray, np.array([[1., 0, 3.], [0, 1., 2.]]), (640, 360))
        inverse, quality = tracker.advance(shifted, [])
        self.assertIsNotNone(inverse)
        self.assertEqual(quality['method'], 'automatic_floor_flow_unverified')
        test_world = [[4, 5], [15, 8]]
        expected_pixels = transform(test_world, court_to_image)+[3, 2]
        np.testing.assert_allclose(transform(expected_pixels, inverse), test_world, atol=.04)
        inverse, quality = tracker.advance(shifted, [])
        self.assertIsNone(inverse)
        self.assertEqual(quality['method'], 'unavailable')

    def test_floor_feature_selection_masks_people(self):
        rng = np.random.default_rng(1)
        gray = rng.integers(0, 256, (360, 640), dtype=np.uint8)
        tracker = FloorTracker()
        tracker.reset(gray, np.linalg.inv(np.array([[18., 0, 50.], [0, 18., 30.], [0, 0, 1.]])), [[180, 100, 300, 250]], {})
        points = tracker.features.reshape(-1, 2)
        self.assertFalse(((points[:, 0] >= 175) & (points[:, 0] <= 305) &
                          (points[:, 1] >= 95) & (points[:, 1] <= 255)).any())
        self.assertLess(points[:, 1].max(), 306)


if __name__ == '__main__':
    unittest.main()
