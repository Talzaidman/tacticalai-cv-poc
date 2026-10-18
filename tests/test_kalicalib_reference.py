import unittest

import cv2
import numpy as np

from tacticalai.auto_calibration import grid_points
from tacticalai.calibration import transform
from tacticalai.kalicalib_reference import extract_centroids, estimate, refine_camera


class KaliReferenceTests(unittest.TestCase):
    def test_centroids_include_all_winning_regions_and_exclude_baskets(self):
        heatmaps = np.zeros((94, 135, 240), np.float32)
        heatmaps[-1] = .2
        heatmaps[3, 10:12, 20:22] = .3
        heatmaps[3, 30:32, 40:42] = .3
        heatmaps[91, 60:70, 80:90] = .9
        points = extract_centroids(heatmaps)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['index'], 3)
        np.testing.assert_allclose(points[0]['image_xy_960'], [122, 82])
        self.assertEqual(points[0]['winning_pixels'], 8)

    def test_camera_refinement_and_source_scaling_against_known_geometry(self):
        world = grid_points()*100
        rotation = cv2.Rodrigues(np.array([.9, 0., 0.]))[0]
        intrinsic = np.array([[1500., 0, 480.], [0, 1500., 270.], [0, 0, 1.]])
        expected = intrinsic @ np.column_stack((rotation[:, :2], [-1400, -400, 3500]))
        pixels = transform(world, expected)
        points = [dict(index=i, world_xy_cm=w.tolist(), image_xy_960=p.tolist())
                  for i, (w, p) in enumerate(zip(world, pixels))]
        for size in ((640, 360), (1920, 1080)):
            result = estimate(points, size)
            self.assertIsNotNone(result['refined_image_to_court_h'])
            sample_world = [[4, 3], [16, 8], [23, 11]]
            sample_pixels = transform(np.array(sample_world)*100, expected)*[size[0]/960, size[1]/540]
            np.testing.assert_allclose(transform(sample_pixels, np.array(result['refined_image_to_court_h'])),
                                       sample_world, atol=1e-4)

    def test_missing_measurements_never_return_mean_camera(self):
        self.assertIsNone(estimate([], (640, 360))['raw_image_to_court_h'])
        matrix, diagnostic = refine_camera(np.zeros((5, 2)), np.zeros((5, 2)))
        self.assertIsNone(matrix)
        self.assertEqual(diagnostic['reason'], 'fewer_than_six_camera_points')


if __name__ == '__main__':
    unittest.main()
