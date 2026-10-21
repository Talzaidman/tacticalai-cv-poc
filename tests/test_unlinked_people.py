import copy
import json
from pathlib import Path
import unittest

from tacticalai.unlinked_people import add_unlinked, box_matches, same_head_duplicate


CONFIG = dict(json.loads(Path('configs/unlinked_people.json').read_text()), court_filter=False)


def detection(box):
    return dict(bbox_xyxy=box, det_confidence=.4, class_id=0, class_name='person')


def frame(tracks=None, detections=None):
    return dict(frame_idx=0, timestamp_s=0., tracks=tracks or [], detections=detections or [],
                calibration=dict(method='unavailable'), image_to_court_h=None)


def masks(boxes=None):
    boxes = boxes or []
    return dict(frame_idx=0, timestamp_s=0., boxes=boxes, confidences=[.6]*len(boxes),
                polygons=[[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]] for b in boxes])


def track(box):
    return dict(detection(box), track_id=7, player_id='7.0', role='player', team_id='A',
                court_xy_m=[3, 4], display_caption='P7 / T7 | player A',
                silhouette_xy=None, silhouette_confidence=None, silhouette_box_iou=None)


class UnlinkedPeopleTests(unittest.TestCase):
    def test_an_actual_detection_survives_without_identity_or_mask(self):
        source = frame(detections=[detection([10, 10, 30, 80])])
        out = add_unlinked(source, masks(), (100, 100), CONFIG)
        row = out['tracks'][0]
        for key in ('track_id', 'player_id', 'role', 'team_id', 'court_xy_m', 'silhouette_xy'):
            self.assertIsNone(row[key])
        self.assertTrue(row['draw_detection_box_fallback'])
        self.assertEqual(row['display_caption'], 'Unlinked')
        self.assertEqual(row['bbox_xyxy'], [10, 10, 30, 80])

    def test_existing_tracks_are_not_duplicated_and_missing_masks_are_filled(self):
        box = [10, 10, 30, 80]
        source = frame([track(box)], [detection(box)])
        saved = copy.deepcopy(source)
        out = add_unlinked(source, masks([box]), (100, 100), CONFIG)
        self.assertEqual(len(out['tracks']), 1)
        self.assertEqual(out['tracks'][0]['player_id'], '7.0')
        self.assertEqual(out['tracks'][0]['role'], 'player')
        self.assertEqual(out['tracks'][0]['team_id'], 'A')
        self.assertEqual(out['tracks'][0]['court_xy_m'], [3, 4])
        self.assertIsNotNone(out['tracks'][0]['silhouette_xy'])
        self.assertFalse(out['tracks'][0]['draw_detection_box_fallback'])
        self.assertEqual(source, saved)

    def test_partly_overlapping_second_person_is_not_consumed_by_first_track(self):
        a, b = [10, 10, 40, 90], [18, 10, 48, 90]
        out = add_unlinked(frame([track(a)], [detection(a), detection(b)]), masks([a, b]), (100, 100), CONFIG)
        self.assertEqual(len(out['tracks']), 2)
        self.assertEqual(out['tracks'][1]['bbox_xyxy'], b)
        self.assertIsNone(out['tracks'][1]['player_id'])
        self.assertEqual(out['tracks'][1]['segmentation_detection_index'], 1)

    def test_segmenter_only_detection_is_real_evidence_and_stays_unlinked(self):
        box = [10, 10, 30, 80]
        out = add_unlinked(frame(), masks([box]), (100, 100), dict(CONFIG, include_segmentation_detections=True))
        self.assertEqual(len(out['tracks']), 1)
        self.assertEqual(out['tracks'][0]['detector_source'], 'yolo26s_seg_detection')
        self.assertIsNone(out['tracks'][0]['track_id'])
        self.assertIsNotNone(out['tracks'][0]['silhouette_xy'])

    def test_maskless_duplicate_requires_same_head_and_overlap(self):
        a = dict(detection([10, 10, 40, 90]), silhouette_xy=None)
        same = dict(detection([10, 10, 40, 75]), silhouette_xy=None)
        different = dict(detection([18, 10, 48, 90]), silhouette_xy=None)
        self.assertTrue(same_head_duplicate(same, a))
        self.assertFalse(same_head_duplicate(different, a))
        same['silhouette_xy'] = [[10, 10], [40, 10], [40, 75]]
        self.assertFalse(same_head_duplicate(same, a))

    def test_missing_detection_and_mask_never_generate_a_person(self):
        self.assertEqual(add_unlinked(frame(), masks(), (100, 100), CONFIG)['tracks'], [])

    def test_one_to_one_masks_and_existing_mask_preservation(self):
        a, b = [10, 10, 40, 90], [18, 10, 48, 90]
        original = track(a)
        original['silhouette_xy'] = [[11, 11], [38, 12], [37, 85]]
        out = add_unlinked(frame([original], [detection(a), detection(b)]), masks([a]), (100, 100), CONFIG)
        self.assertEqual(out['tracks'][0]['silhouette_xy'], original['silhouette_xy'])
        self.assertIsNone(out['tracks'][1]['silhouette_xy'])
        self.assertEqual(len(box_matches([a], [a, b], .3)), 1)

    def test_boundary_clipping_and_missing_camera_remain_visible(self):
        cfg = dict(CONFIG, court_filter=True)
        out = add_unlinked(frame(detections=[detection([90, 10, 100, 80])]), masks(), (100, 100), cfg)
        self.assertEqual(len(out['tracks']), 1)
        self.assertEqual(out['tracks'][0]['court_gate'], 'uncertain')

    def test_misaligned_masks_are_rejected(self):
        bad = dict(masks(), frame_idx=1)
        with self.assertRaises(ValueError):
            add_unlinked(frame(), bad, (100, 100), CONFIG)


if __name__ == '__main__':
    unittest.main()
