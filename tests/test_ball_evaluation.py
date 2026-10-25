"""Draft localization metrics and automatic motion evidence stay separate."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from scripts.evaluate_ball import evaluate, score_samples
from tacticalai.ball_cli import detect_ball
from tacticalai.ball_motion import (MotionConfig, align_background, analyze_motion,
                                    filter_detections)
from tacticalai.export import sha256, write_json
from tacticalai.silhouette_presentation import draw_ball_detections


def frame(index=0, centers=((100, 100),), status='observed'):
    boxes = [dict(bbox_xyxy=[x-3, y-3, x+3, y+3], center_image_xy=[x, y],
                  det_confidence=.9) for x, y in centers]
    return dict(frame_idx=index, ball_raw_detections=boxes,
                ball=dict(status=status, candidates=boxes,
                          center_image_xy=list(centers[0]) if centers else None,
                          predicted_center_image_xy=[200, 200] if status == 'predicted' else None))


def sample(index=0, visibility='visible', center=(200, 200)):
    return dict(frame_idx=index, visibility=visibility, center_xy=center, split='development')


def score(frames, samples, mode='accepted'):
    return score_samples(frames, samples, [640, 360], [1280, 720], 14, mode)


@pytest.mark.parametrize('mode', ['raw', 'accepted', 'observed'])
def test_metric_scaling_and_exact_radius(mode):
    result = score([frame(centers=((107, 100),))], [sample()], mode)
    assert result['true_positives'] == 1
    assert result['matched_mean_error_review_px'] == 14
    assert score([frame(centers=((107.01, 100),))], [sample()], mode)['false_negatives'] == 1


def test_duplicates_are_false_positives_unknowns_are_not_negatives():
    frames = [frame(centers=((100, 100), (101, 100), (300, 300))), frame(1)]
    result = score(frames, [sample(), sample(1, 'unknown', None)])
    assert (result['true_positives'], result['false_positives'], result['false_negatives']) == (1, 2, 0)
    assert result['precision'] == 1/3
    assert result['recall'] == 1
    assert result['excluded_unknown'] == 1


def test_absent_prediction_and_empty_measurements():
    result = score([frame()], [sample(visibility='absent', center=None)])
    assert result['false_positives'] == 1
    assert result['recall'] is None
    result = score([frame(status='predicted')], [sample()], 'observed')
    assert result['false_negatives'] == 1
    assert result['predictions'] == 0
    assert result['precision'] is None


def test_nonfinite_predictions_are_errors_not_misses():
    item = frame()
    item['ball']['candidates'][0]['center_image_xy'] = [float('nan'), 100]
    with pytest.raises(ValueError, match='predicted'):
        score([item], [sample()])


@pytest.mark.parametrize('frames,samples', [
    ([frame()], [sample(1)]), ([frame(), frame()], [sample()]),
    ([frame()], [sample(), sample()]), ([frame()], [sample(center=None)]),
    ([frame()], [sample(center=(float('nan'), 0))]),
    ([frame()], [sample(visibility='absent')]), ([frame()], [sample(visibility='occluded')])])
def test_invalid_or_incomplete_evidence_fails(frames, samples):
    with pytest.raises(ValueError):
        score(frames, samples)


def test_evaluation_rejects_result_or_source_video_hash_changes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folder = root/'clip'
        folder.mkdir()
        path = folder/'frames.jsonl'
        path.write_text(json.dumps(frame())+'\n', encoding='utf8')
        reference = root/'reference.json'
        write_json(reference, dict(quality='draft', coordinate_size=[1280, 720],
            match_radius_px=14, clips={'clip': dict(video_sha256='video', samples=[sample()])}))
        write_json(root/'experiment.json', dict(status='complete'))
        metadata = dict(status='complete', video=dict(sha256='video', width=640, height=360),
                        artifacts_sha256={'frames.jsonl': sha256(path)})
        write_json(folder/'metadata.json', metadata)
        assert evaluate(root, reference, 'development')['clips']['clip']['accepted']['true_positives'] == 1
        path.write_text(json.dumps(frame(centers=()))+'\n', encoding='utf8')
        with pytest.raises(ValueError, match='hash'):
            evaluate(root, reference, 'development')
        metadata['artifacts_sha256']['frames.jsonl'] = sha256(path)
        metadata['video']['sha256'] = 'other_video'
        write_json(folder/'metadata.json', metadata)
        with pytest.raises(ValueError, match='hash'):
            evaluate(root, reference, 'development')


def test_person_context_crops_are_opt_in_clipped_and_ignore_manual_roles():
    boxes = Mock()
    boxes.xyxy.cpu.return_value.tolist.return_value = [[1., 2., 4., 5.]]
    boxes.conf.cpu.return_value.tolist.return_value = [.8]
    model = Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    args = SimpleNamespace(tiles=False, tile_size=960, device='cpu', imgsz=640,
                           conf=.15, ball_class=32, person_crops=True)
    image = np.zeros((360, 640, 3), np.uint8)
    rows = [dict(bbox_xyxy=[0, 10, 20, 110], court_region='inside', role='anything'),
            dict(bbox_xyxy=[400, 20, 420, 120], court_region='outside')]
    result = detect_ball(model, image, args, rows)
    assert len(result) == 2
    assert result[1]['tile_xyxy'] == [0, 0, 95, 145]
    assert result[1]['detection_source'] == 'person_context_crop'
    assert all(call.kwargs['classes'] == [32] for call in model.predict.call_args_list)
    rows[0]['role'] = None
    assert detect_ball(model, image, args, rows) == result
    args.person_crops = False
    assert len(detect_ball(model, image, args, rows)) == 1
    args.person_crops = True
    rows[0]['bbox_xyxy'] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match='crop'):
        detect_ball(model, image, args, rows)


@pytest.mark.parametrize('change', [
    {'lag': 0}, {'lag': 1.5}, {'working_width': 10}, {'min_detection_motion': float('nan')},
    {'min_area': 200}, {'min_detection_motion': 256}])
def test_motion_configuration_validation(change):
    with pytest.raises(ValueError):
        MotionConfig(**change)


def test_motion_support_scales_boxes_and_keeps_threshold_without_mutation():
    raw = frame(centers=((100, 100), (300, 300)))['ball_raw_detections']
    before = copy.deepcopy(raw)
    difference = np.zeros((360, 640), float)
    difference[48:52, 48:52] = 25
    accepted, evidence = filter_detections(raw, [difference, difference+5], (1280, 720))
    assert accepted == raw[:1]
    assert [r['accepted'] for r in evidence] == [True, False]
    assert evidence[0]['median_residual_0_255'] == 25
    assert raw == before
    assert filter_detections(raw, [], (1280, 720))[0] == []
    assert filter_detections(raw, [], (1280, 720))[1][0]['median_residual_0_255'] is None


def test_camera_failure_and_cuts_do_not_invent_motion_support():
    image = np.zeros((360, 640, 3), np.uint8)
    difference, quality = align_background(image, image)
    assert difference is None
    assert quality['reason'] == 'insufficient_features'
    results = analyze_motion([image]*3, [True, True, False], (640, 360),
                             detections=[frame()['ball_raw_detections']]*3)
    assert all(not r['accepted_detections'] for r in results)
    assert all(r['cameras'] == [] for r in results)
    with pytest.raises(ValueError):
        analyze_motion([image], [], (640, 360))


def test_camera_pan_rejected_but_airborne_moving_object_retained():
    rng = np.random.default_rng(41)
    background = cv2.GaussianBlur(rng.integers(40, 180, (360, 640, 3), dtype=np.uint8), (5, 5), 1)
    previous = background.copy()
    cv2.circle(previous, (90, 80), 6, (20, 30, 110), -1)
    current = cv2.warpAffine(background, np.float32([[1, 0, 4], [0, 1, 2]]), (640, 360))
    cv2.circle(current, (130, 60), 6, (20, 30, 110), -1)
    difference, quality = align_background(previous, current)
    assert quality['reason'] is None
    assert difference is not None
    np.testing.assert_allclose(np.asarray(quality['affine'])[:, 2], [4, 2], atol=.3)
    raw = frame(centers=((130, 60), (300, 100)))['ball_raw_detections']
    accepted, _ = filter_detections(raw, [difference], (640, 360))
    assert accepted == raw[:1]


def test_filtered_overlay_retains_raw_evidence_without_showing_rejected_hits():
    item = frame(centers=((200, 200), (800, 300)), status='unknown')
    item['ball']['center_image_xy'] = None
    item['ball_motion_accepted_detections'] = item['ball_raw_detections'][:1]
    image = np.zeros((720, 1280, 3), np.uint8)
    before = copy.deepcopy(item)
    draw_ball_detections(image, item, 1, 1)
    np.testing.assert_array_equal(image[200, 200], [0, 165, 255])
    np.testing.assert_array_equal(image[300, 800], [0, 0, 0])
    assert item == before
