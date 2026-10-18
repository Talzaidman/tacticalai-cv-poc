import unittest

import numpy as np

from tacticalai.temporal_projection import kalman_rts, image_probes, smooth_cameras, smooth_players
from tacticalai.calibration import transform


class TemporalProjectionTests(unittest.TestCase):
    def test_constant_velocity_preserved_without_phase_lag(self):
        rng = np.random.default_rng(3)
        times = np.arange(150)/25
        truth = np.column_stack((3+7*times, 8-2*times))
        noisy = truth+rng.normal(0, .25, truth.shape)
        filtered = kalman_rts(noisy, 25)
        self.assertLess(np.sqrt(np.mean((filtered-truth)**2)), .10)
        np.testing.assert_allclose(np.mean(filtered[10:-10]-truth[10:-10], axis=0), 0, atol=.055)

    def test_camera_outlier_bounded_gaps_and_cut_reset(self):
        rng = np.random.default_rng(5)
        matrices = []
        for index in range(110):
            offset = index*.04+(15 if index >= 70 else 0)+rng.normal(0, .06)
            matrices.append(np.array([[.03, 0, offset], [0, .04, -1.], [0, 0, 1.]]))
        matrices[30] = None
        matrices[31] = None
        for index in range(45, 52):
            matrices[index] = None
        matrices[60] = matrices[60].copy()
        matrices[60][0, 2] += 4
        cuts = np.zeros(110, bool)
        cuts[[0, 70]] = True
        smoothed, qualities, _, rejected = smooth_cameras(matrices, (640, 360), 25, cuts)
        self.assertTrue(rejected[60])
        self.assertEqual(qualities[30]['method'], 'kalman_rts_camera_gap_estimate')
        self.assertTrue(all(smoothed[i] is None for i in range(45, 52)))
        self.assertGreater(smoothed[70][0, 2]-smoothed[69][0, 2], 14)
        self.assertAlmostEqual(smoothed[60][0, 2], 60*.04, delta=.15)
        self.assertIsNone(matrices[30])
        probes = image_probes((640, 360))
        expected = transform(probes, np.array([[.03,0,2.4],[0,.04,-1],[0,0,1.]]))
        np.testing.assert_allclose(transform(probes, smoothed[60]), expected, atol=.15)

    def test_no_extrapolation_before_first_or_after_last_camera(self):
        matrices = [None, np.eye(3), np.eye(3), None]
        # Scale to metric range: identity pixels would violate the sanity bound.
        matrices[1] = matrices[2] = np.diag([.02,.02,1.])
        smoothed, _, _, _ = smooth_cameras(matrices, (640,360), 25, [True,False,False,False])
        self.assertIsNone(smoothed[0])
        self.assertIsNone(smoothed[3])

    def test_players_missing_roles_ids_and_large_jumps_not_joined(self):
        def row(track, xy, role='player'):
            return dict(track_id=track, court_xy_m=xy, role=role)
        frames = [dict(tracks=[row(1,[1,2])]), dict(tracks=[row(1,[1.1,2])]), dict(tracks=[]),
                  dict(tracks=[row(1,[10,2])]), dict(tracks=[row(1,[20,2])]),
                  dict(tracks=[row(2,[3,4])]), dict(tracks=[row(1,[2,2],'staff')])]
        smooth_players(frames,25,[True]+[False]*6)
        self.assertEqual(frames[2]['tracks'],[])
        self.assertEqual(frames[3]['tracks'][0]['court_xy_m'],[10,2])
        self.assertEqual(frames[4]['tracks'][0]['court_xy_m'],[20,2])
        self.assertEqual(frames[5]['tracks'][0]['track_id'],2)
        self.assertNotIn('position_temporal_filter',frames[6]['tracks'][0])


if __name__ == '__main__':
    unittest.main()
