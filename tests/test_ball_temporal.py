"""Behavioral checks for image-space gating, bounded inference and trail breaks."""
import importlib.util
import sys
from pathlib import Path
import unittest
import ast
import json
import tempfile
import time
import contextlib
import io

import numpy as np
import cv2

spec = importlib.util.spec_from_file_location('ball_temporal', Path(__file__).parents[1]/'scripts/ball_temporal.py')
temporal = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = temporal
spec.loader.exec_module(temporal)


def detection(x, y=100, conf=.8):
    return dict(x=x, y=y, conf=conf)


class TemporalTests(unittest.TestCase):
    def test_notebook_cached_comparison_runs_without_inference(self):
        notebook = json.loads((Path(__file__).parents[1]/'notebooks/colab_ball_benchmark.ipynb').read_text())
        helper = ''.join(notebook['cells'][2]['source'])
        rf_cell = ast.parse(''.join(notebook['cells'][6]['source']))
        # Skip writing the child script into /content; execute the real cached branch.
        rf_cell.body = rf_cell.body[2:]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, out = root/'cache', root/'out'
            cache.mkdir(); out.mkdir()
            dets = [[detection(10+i*.1, 12)] for i in range(175)]
            raw = dict(provenance=dict(model_id='test-model/13'), variants={
                'RF-DETR-tiles': dict(detections=dets, seconds=40.0, config=dict(
                    mode='tiles', tracker_min_conf=.25, candidate_confidence=.1))})
            (cache/'rf_raw_attack_clip2.json').write_text(json.dumps(raw))
            cache1 = root/'cache1'; cache1.mkdir()
            (cache1/'rf_raw_attack_clip.json').write_text(json.dumps(raw))
            scope = dict(__name__='__main__', OUT=out, json=json, np=np, cv2=cv2,
                         math=__import__('math'), time=time, Path=Path,
                         MIN_CONF={}, RUN_CONFIGS={}, RF_SELECTED_VARIANTS=('RF-DETR-tiles',),
                         RF_SWEEP=False, RF_CACHE_AVAILABLE=True, KALMAN_SETTINGS={},
                         RF_CACHE_SOURCES={'attack_clip2': cache/'rf_raw_attack_clip2.json',
                                           'attack_clip': cache1/'rf_raw_attack_clip.json'}, RF_MISSING_CLIPS={},
                         SHOW_RAW_CANDIDATES=False, SHOW_BASELINE=False,
                         EXPERIMENT='rfdet', RUN_OPTIONAL_MODELS=False, COMPARE_KALMAN=True,
                         RF_MODEL_ID='test-model/13', ROBOFLOW_API_KEY=None,
                         CLIPS={'attack_clip': root/'video1.mp4', 'attack_clip2': root/'video2.mp4'})
            exec(compile(helper, '<notebook-helpers>', 'exec'), scope)
            # Rejected/raw candidates are absent unless the explicit debug flag is on.
            canvas = np.zeros((80,80,3), dtype=np.uint8)
            candidates, unknown = [[detection(40,40)]], [dict(state='none')]
            clean = scope['draw'](canvas, candidates, unknown, 0, 'test')
            self.assertFalse(np.any(np.all(clean == [0,140,255], axis=2)))
            scope['SHOW_RAW_CANDIDATES'] = True
            debug = scope['draw'](canvas, candidates, unknown, 0, 'test')
            self.assertTrue(np.any(np.all(debug == [0,140,255], axis=2)))
            scope['SHOW_RAW_CANDIDATES'] = False
            frame = np.zeros((36,64,3), dtype=np.uint8)
            scope['read_video'] = lambda path: ([frame]*175, 25)
            rendered = []
            def consume_video(path, frames, fps, frame_shape):
                rendered.append(sum(1 for image in frames if image.shape == (36,64,3)))
            scope['write_video'] = consume_video
            scope['show'] = lambda *args: None
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(rf_cell, '<cached-comparison>', 'exec'), scope)
            self.assertEqual(rendered, [175,175,175,175])
            self.assertEqual(set(scope['RESULTS']), {'RF-DETR-tiles', 'RF-DETR-tiles-Kalman'})
            result = json.loads((out/'RF-DETR-tiles-Kalman_attack_clip2.json').read_text())
            self.assertEqual(set(scope['RESULTS']['RF-DETR-tiles-Kalman']), {'attack_clip', 'attack_clip2'})
            self.assertTrue((out/'RF-DETR-tiles-Kalman_attack_clip.json').is_file())
            self.assertFalse(result['config']['raw_candidates_visible'])
            self.assertEqual(result['stats']['detector_time_origin'], 'cached_original_run')
            self.assertTrue(result['config']['reused_detections'])
            self.assertEqual(result['config']['temporal_config']['confirmation_frames'], 3)
            self.assertEqual(result['tracked'][2]['position_source'], 'measurement_filtered')

    def track(self, frames, fps=25, **config):
        return temporal.track_temporal(frames, 1280, 720, fps, temporal.TemporalConfig(**config))

    def test_high_confidence_far_outlier_does_not_teleport_and_gap_interpolates(self):
        result = self.track([[detection(100)], [detection(105)], [detection(110)],
                             [detection(1000, 600, .99)], [detection(120)]])
        self.assertEqual(result[2]['state'], 'detected')
        self.assertEqual(result[3]['state'], 'interpolated')
        self.assertAlmostEqual(result[3]['x'], 115, places=4)
        self.assertIsNone(result[3]['conf'])
        self.assertIsNone(result[3]['measurement_xy'])
        self.assertEqual(result[3]['rejected'][0]['reason'], 'speed_gate')
        self.assertEqual(result[4]['segment_id'], result[2]['segment_id'])
        self.assertLess(result[4]['x'], 130)

    def test_near_measurement_beats_more_confident_distant_candidate(self):
        result = self.track([[detection(100)], [detection(105)], [detection(110)],
                             [detection(1000, conf=.99), detection(115, conf=.4)]])
        self.assertEqual(result[-1]['accepted_candidate_index'], 1)
        self.assertEqual(result[-1]['measurement_xy'], [115, 100])
        self.assertEqual(len(result[-1]['rejected']), 1)

    def test_long_gap_and_reacquisition_start_new_segment_without_interpolation(self):
        frames = [[detection(100)], [detection(105)], [detection(110)]] + [[]]*12
        frames += [[detection(800)], [detection(805)], [detection(810)]]
        result = self.track(frames)
        self.assertEqual(result[-1]['state'], 'detected')
        self.assertNotEqual(result[2]['segment_id'], result[-1]['segment_id'])
        self.assertNotIn('interpolated', [r['state'] for r in result])
        self.assertTrue(all(r['state']=='none' for r in result[8:15]))
        self.assertIsNone(result[8]['x'])
        self.assertIsNone(result[8]['measurement_xy'])

    def test_confirmation_rejects_single_frame_initial_false_positive(self):
        result = self.track([[detection(1000, conf=.99)], [], [detection(100)],
                             [detection(105)], [detection(110)]])
        self.assertEqual([r['state'] for r in result[:4]], ['none']*4)
        self.assertEqual(result[4]['state'], 'detected')

    def test_prediction_horizon_uses_seconds_at_different_frame_rates(self):
        for fps in (25, 50):
            result = self.track([[detection(100)], [detection(100)], [detection(100)]] + [[]]*20, fps=fps)
            self.assertEqual(sum(r['state']=='predicted' for r in result), int(.2*fps))
            self.assertFalse(any(r['state']=='interpolated' for r in result))

    def test_covariance_remains_finite_and_symmetric(self):
        result = self.track([[detection(100+i*3+(-1)**i)] for i in range(100)])
        for r in result:
            if r['covariance_xy'] is not None:
                covariance = np.asarray(r['covariance_xy'])
                self.assertTrue(np.isfinite(covariance).all())
                np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
                self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), 0)

    def test_trails_never_connect_gaps_or_segments(self):
        points = [dict(state='detected', x=i, y=0, segment_id=1) for i in range(2)]
        points += [dict(state='interpolated', x=2, y=0, segment_id=1)]
        points += [dict(state='detected', x=i, y=0, segment_id=1) for i in (3,4)]
        points += [dict(state='detected', x=i, y=0, segment_id=2) for i in (500,501)]
        self.assertEqual(temporal.trail_segments(points, len(points)-1),
                         [[(0,0),(1,0)],[(3,0),(4,0)],[(500,0),(501,0)]])

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(self.track([]), [])
        self.assertEqual(self.track([[detection(float('nan'))]])[0]['state'], 'none')
        for settings in [dict(confirmation_frames=1), dict(max_gap_s=0), dict(min_confidence=2)]:
            with self.assertRaises(ValueError): temporal.TemporalConfig(**settings)
        with self.assertRaises(ValueError): self.track([], fps=0)


if __name__=='__main__':
    unittest.main()
