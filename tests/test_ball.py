"""Measured thresholds, abstention, provenance and map semantics for ball baseline."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from tacticalai.ball import (BallConfig, BallTracker, PossessionEstimator,
                             ball_candidates, tile_windows)
from tacticalai.ball_cli import COURT, detect_ball, encode, flat_ball, load_source
from tacticalai.court import floor_lines
from tacticalai.export import sha256, write_json
from tacticalai.silhouette_presentation import draw_ball_detections, render


def detection(x=100, y=100, confidence=.9):
    return dict(bbox_xyxy=[x-3, y-3, x+3, y+3], det_confidence=confidence,
                detection_source='test')


def person(identity='a', x=100, court=True):
    return dict(player_id=identity, track_id=1 if identity == 'a' else 2,
                identity_segment_id=identity, bbox_xyxy=[x-20, 50, x+20, 150],
                role=None, team_id=None, silhouette_xy=None,
                display_label=identity, court_xy_m=[10., 7.] if court else None,
                court_region='inside' if court else 'unknown')


def observed(x=100, y=100, track_id=1):
    return dict(status='observed', center_image_xy=[x, y], track_id=track_id)


class BallTests(unittest.TestCase):
    def test_configuration_rejects_invalid_thresholds(self):
        for change in ({'max_gap_s': 0}, {'possession_margin': float('nan')},
                       {'detection_confidence': 1.1}, {'confirmation_frames': 1.5}):
            with self.assertRaises(ValueError):
                BallConfig(**change)

    def test_tiles_cover_edges_and_small_video(self):
        self.assertEqual(tile_windows(640, 360), [(0, 0, 640, 360)])
        windows = tile_windows(1920, 1080)
        cover = np.zeros((1080, 1920), bool)
        for x1, y1, x2, y2 in windows:
            self.assertLessEqual(x2-x1, 960)
            self.assertLessEqual(y2-y1, 960)
            cover[y1:y2, x1:x2] = True
        self.assertTrue(cover.all())
        self.assertIn((960, 120, 1920, 1080), windows)

    def test_nms_merges_tiles_and_retains_source(self):
        a, b = detection(), detection(101, confidence=.8)
        b['detection_source'] = 'tile'
        result = ball_candidates([a, b, detection(200, confidence=.1)], (640, 360), BallConfig())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['detection_source'], 'test')
        self.assertEqual(result[0]['center_image_xy'], [100., 100.])
        self.assertEqual(a, detection())

    def test_detector_uses_only_selected_class_and_maps_tiles_to_source_pixels(self):
        boxes = Mock()
        boxes.xyxy.cpu.return_value.tolist.return_value = [[10., 20., 16., 26.]]
        boxes.conf.cpu.return_value.tolist.return_value = [.8]
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        args = SimpleNamespace(tiles=True, tile_size=960, device='cpu', imgsz=1280, conf=.15, ball_class=0)
        image = np.zeros((1080, 1920, 3), np.uint8)
        detections = detect_ball(model, image, args)
        self.assertEqual(len(detections), 1+len(tile_windows(1920, 1080)))
        self.assertEqual(detections[0]['detection_source'], 'full_frame')
        self.assertEqual(detections[-1]['bbox_xyxy'], [970., 140., 976., 146.])
        self.assertTrue(all(call.kwargs['classes'] == [0] for call in model.predict.call_args_list))

    def test_person_and_ball_coordinates_do_not_change_with_source_resolution(self):
        for factor in (1, 3):
            estimator = PossessionEstimator(25)
            row = person()
            row['bbox_xyxy'] = [v*factor for v in row['bbox_xyxy']]
            for _ in range(3):
                result = estimator.update(observed(100*factor, 100*factor), [row])
            self.assertEqual(result['controller_person_id'], 'a')
            self.assertEqual(result['minimap_xy_m'], [10., 7.])

    def test_invalid_and_implausibly_large_boxes(self):
        config = BallConfig()
        self.assertEqual(ball_candidates([dict(bbox_xyxy=[1, 1, 50, 50], det_confidence=.9)],
                                         (640, 360), config), [])
        for box in ([100, 1, 1, 20], [-1, 1, 2, 3], [0, 1, float('nan'), 3]):
            with self.assertRaises(ValueError):
                ball_candidates([dict(bbox_xyxy=box, det_confidence=.9)], (640, 360), config)

    def test_confirmation_predictions_expiration_and_reacquisition(self):
        tracker = BallTracker((640, 360), 25)
        self.assertEqual(tracker.update(0, [detection()])['status'], 'tentative')
        ball = tracker.update(1, [detection(102)])
        self.assertEqual(ball['status'], 'observed')
        first_id = ball['track_id']
        for index in (2, 3, 4):
            ball = tracker.update(index, [])
            self.assertEqual(ball['status'], 'predicted')
            self.assertIsNone(ball['center_image_xy'])
            self.assertIsNone(ball['det_confidence'])
            self.assertIsNone(ball['court_xy_m'])
        self.assertEqual(tracker.update(5, [])['status'], 'unknown')
        ball = tracker.update(6, [detection(400)])
        self.assertEqual(ball['status'], 'tentative')
        self.assertNotEqual(ball['track_id'], first_id)

    def test_ambiguity_and_motion_gate(self):
        tracker = BallTracker((640, 360), 25)
        self.assertEqual(tracker.update(0, [detection(100), detection(200)])['reason'], 'ambiguous_candidates')
        tracker.update(1, [detection(100)])
        self.assertEqual(tracker.update(2, [detection(600)])['reason'], 'outside_motion_gate')

    def test_cut_resets_and_gap_requires_reconfirmation(self):
        tracker = BallTracker((640, 360), 25)
        tracker.update(0, [detection()])
        before = tracker.update(1, [detection()])
        tracker.update(2, [])
        self.assertEqual(tracker.update(3, [detection()])['status'], 'tentative')
        after = tracker.update(4, [detection()], cut=True)
        self.assertNotEqual(before['track_id'], after['track_id'])
        self.assertEqual(after['status'], 'tentative')
        with self.assertRaises(ValueError):
            tracker.update(6, [])

    def test_control_requires_exact_temporal_threshold_and_uses_person_not_ball_xy(self):
        estimator = PossessionEstimator(25)
        row = person()
        before = copy.deepcopy(row)
        for _ in range(2):
            self.assertIsNone(estimator.update(observed(), [row])['controller_person_id'])
        result = estimator.update(observed(), [row])
        self.assertEqual(result['status'], 'estimated_control')
        self.assertEqual(result['controller_person_id'], 'a')
        self.assertEqual(result['minimap_xy_m'], row['court_xy_m'])
        self.assertEqual(result['minimap_position_source'], 'estimated_controller_floor_position')
        self.assertIsNone(result['controller_role'])
        self.assertEqual(row, before)

    def test_possession_abstains_on_overlap_prediction_unknown_identity_and_distant_ball(self):
        for ball, rows, reason in (
            (observed(), [person(), person('b')], 'ambiguous_people'),
            (dict(status='predicted'), [person()], 'ball_not_observed'),
            (observed(), [dict(person(), player_id=None)], 'no_supported_controller'),
            (observed(500), [person()], 'no_supported_controller')):
            estimator = PossessionEstimator(25)
            for _ in range(5):
                result = estimator.update(ball, rows)
                self.assertIsNone(result['controller_person_id'])
                self.assertIsNone(result['minimap_xy_m'])
            self.assertEqual(result['reason'], reason)

    def test_missing_frame_new_ball_identity_segment_or_cut_resets_control(self):
        for change in ('gap', 'segment', 'ball', 'cut', 'person'):
            estimator = PossessionEstimator(25)
            row = person()
            for _ in range(3):
                estimator.update(observed(), [row])
            if change == 'gap':
                estimator.update(dict(status='unknown'), [row])
            if change == 'segment':
                row['identity_segment_id'] = 'new'
            if change == 'person':
                row['player_id'] = 'new'
            result = estimator.update(observed(track_id=2 if change == 'ball' else 1), [row], cut=change == 'cut')
            self.assertIsNone(result['controller_person_id'])

    def test_missing_or_outside_camera_never_maps_ball_even_when_control_is_estimated(self):
        for row in (person(court=False), dict(person(), court_region='outside')):
            estimator = PossessionEstimator(25)
            for _ in range(3):
                result = estimator.update(observed(), [row])
            self.assertEqual(result['controller_person_id'], 'a')
            self.assertIsNone(result['minimap_xy_m'])

    def test_owner_change_requires_new_evidence_without_retaining_old_marker(self):
        estimator = PossessionEstimator(25)
        rows = [person(), person('b', x=400)]
        for _ in range(3):
            result = estimator.update(observed(), rows)
        self.assertEqual(result['controller_person_id'], 'a')
        for _ in range(2):
            result = estimator.update(observed(400), rows)
            self.assertIsNone(result['controller_person_id'])
            self.assertIsNone(result['minimap_xy_m'])
        self.assertEqual(estimator.update(observed(400), rows)['controller_person_id'], 'b')

    def test_map_marker_render_and_flat_export(self):
        row = person()
        estimator = PossessionEstimator(25)
        for _ in range(3):
            result = estimator.update(observed(), [row])
        ball = BallTracker((640, 360), 25).update(0, [detection()])
        frame = dict(frame_idx=0, timestamp_s=0., tracks=[row], calibration=dict(method='test'),
                     ball=ball, possession=result)
        plain = dict(frame)
        plain.pop('ball')
        image = np.zeros((360, 640, 3), np.uint8)
        baseline = render(image, plain, floor_lines(COURT))
        enhanced = render(image, frame, floor_lines(COURT))
        x, y = np.rint(np.array([10, 7])*9.6+[17, 29]).astype(int)
        self.assertTrue(np.any(enhanced[10+y-8:10+y+9, 10+x-8:10+x+9] !=
                               baseline[10+y-8:10+y+9, 10+x-8:10+x+9]))
        flat = flat_ball('clip', frame)
        self.assertEqual((flat['marker_x_m'], flat['marker_y_m']), (10., 7.))
        self.assertIsNone(flat['ball_court_x_m'])
        self.assertIsNone(flat['ball_court_y_m'])
        json.dumps(frame, allow_nan=False)

    def test_all_raw_detections_orange_even_when_tracker_unknown_and_prediction_gray(self):
        orange = np.array([0, 165, 255], np.uint8)
        for status in ('unknown', 'tentative', 'observed', 'predicted'):
            image = np.zeros((720, 1280, 3), np.uint8)
            frame = dict(ball=dict(status=status, center_image_xy=None,
                                  predicted_center_image_xy=[500, 400] if status == 'predicted' else None),
                         ball_raw_detections=[detection(200, 200), detection(800, 300)])
            before = copy.deepcopy(frame)
            draw_ball_detections(image, frame, 1, 1)
            for x, y in ((200, 200), (800, 300)):
                np.testing.assert_array_equal(image[y, x], orange)
                self.assertGreater(np.count_nonzero(np.all(image[y-22:y+23, x-22:x+23] == orange, axis=2)), 100)
            if status == 'predicted':
                self.assertFalse(np.any(np.all(image[385:416, 485:516] == orange, axis=2)))
            self.assertEqual(frame, before)

    def test_source_requires_hash_alignment_and_automatic_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'clip'
            folder.mkdir()
            info = dict(frames=1, fps=25., width=640, height=360, sha256='video')
            frame = dict(frame_idx=0, timestamp_s=0., tracks=[person()], calibration=dict(method='automatic'))
            path = folder/'frames.jsonl'
            path.write_text(json.dumps(frame)+'\n')
            write_json(root/'experiment.json', dict(status='complete', manual_labels_used=False))
            def metadata():
                write_json(folder/'metadata.json', dict(status='complete', video=info,
                           artifacts_sha256={'frames.jsonl': sha256(path)}))
            metadata()
            frames, _ = load_source(root, 'clip', info)
            self.assertEqual(frames, [frame])
            write_json(root/'experiment.json', dict(status='complete', automatic_only=True, manual_labels_used=True))
            with self.assertRaisesRegex(ValueError, 'explicitly automatic'):
                load_source(root, 'clip', info)
            write_json(root/'experiment.json', dict(status='complete', manual_labels_used=False))
            frame['calibration']['method'] = 'manual_keyframe'
            path.write_text(json.dumps(frame)+'\n')
            with self.assertRaises(ValueError):
                load_source(root, 'clip', info)
            metadata()
            with self.assertRaisesRegex(ValueError, 'Manual calibration'):
                load_source(root, 'clip', info)

    def test_actual_mp4_encoding_preserves_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root/'input.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25, (160, 90))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((90, 160, 3), np.uint8))
            writer.release()
            tracker = BallTracker((160, 90), 25)
            possession = PossessionEstimator(25)
            frames = []
            for index in range(3):
                ball = tracker.update(index, [])
                frames.append(dict(frame_idx=index, timestamp_s=index/25, tracks=[],
                                   calibration=dict(method='unavailable'), ball=ball,
                                   possession=possession.update(ball, [])))
            self.assertEqual(encode(video, frames, root, dict(fps=25)), 3)


if __name__ == '__main__':
    unittest.main()
