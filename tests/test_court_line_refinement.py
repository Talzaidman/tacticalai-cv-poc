import unittest

import numpy as np

from tacticalai.calibration import transform
from tacticalai.calibration_cli import polylines
from tacticalai.court import floor_lines

try:
    from tacticalai.court_line_refinement import correction_matrix, refine
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@unittest.skipUnless(HAVE_SCIPY, 'optional geometry extra requires scipy')
class CourtLineRefinementTests(unittest.TestCase):
    def test_correction_maps_control_points_without_float_rounding(self):
        delta = np.array([[1, 2], [-2, 1], [3, -2], [-1, -3]], float)
        source = np.array([[0, 0], [960, 0], [960, 540], [0, 540]], float)
        np.testing.assert_allclose(transform(source, correction_matrix(delta)), source+delta, atol=1e-9)

    def test_known_offset_reduces_against_synthetic_painted_court(self):
        court = dict(length_m=28, width_m=15, key_depth_m=5.8, key_width_m=4.9)
        expected = np.array([[25., 0, 100.], [0, 22., 100.], [0, 0, 1.]])
        image = np.full((540, 960, 3), (70, 40, 50), np.uint8)
        polylines(image, floor_lines(court), expected, (0, 255, 255))
        offset = np.array([[1., 0, 3.], [0, 1., 4.], [0, 0, 1.]]) @ expected
        corrected, diagnostic, mask = refine(image, np.linalg.inv(offset), court, [])
        self.assertIsNotNone(corrected, diagnostic)
        checks = [[4, 4], [15, 8], [24, 11]]
        recovered = transform(transform(checks, expected), corrected)
        self.assertLess(np.mean(np.linalg.norm(recovered-checks, axis=1)), .07)
        self.assertLess(diagnostic['mean_capped_line_distance_after_960_px'],
                        diagnostic['mean_capped_line_distance_before_960_px'])

    def test_missing_camera_and_masked_court_remain_unknown(self):
        image = np.zeros((540, 960, 3), np.uint8)
        matrix, quality, mask = refine(image, None, {}, [])
        self.assertIsNone(matrix)
        self.assertEqual(quality['reason'], 'no_camera_initialization')
        court = dict(length_m=28, width_m=15, key_depth_m=5.8, key_width_m=4.9)
        seed = np.array([[25., 0, 100.], [0, 22., 100.], [0, 0, 1.]])
        polylines(image, floor_lines(court), seed, (0, 255, 255))
        matrix, quality, mask = refine(image, np.linalg.inv(seed), court, [[0, 0, 960, 540]])
        self.assertIsNone(matrix)
        self.assertEqual(quality['reason'], 'insufficient_line_pixels')


if __name__ == '__main__':
    unittest.main()
