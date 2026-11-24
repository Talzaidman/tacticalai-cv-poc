from types import SimpleNamespace

import pytest

from tacticalai.person_diagnostic_cli import analyze_frame, configured_tracker


def test_tracker_thresholds_are_explicit_and_defaults_are_not_mutated():
    defaults = dict(tracker_type='tracktrack', track_high_thresh=.6,
                    track_low_thresh=.25, new_track_thresh=.7,
                    min_track_len=3, track_buffer=30, with_reid=False,
                    model='auto', gmc_method='none')
    tuned = configured_tracker(defaults, high=.35, low=.15, new=.35,
        minimum_length=2, buffer_frames=50, with_reid=True, reid_model='reid.pt')
    assert defaults['new_track_thresh'] == .7
    assert tuned['new_track_thresh'] == .35
    assert tuned['track_low_thresh'] == .15
    assert tuned['min_track_len'] == 2
    assert tuned['with_reid'] is True
    assert tuned['model'] == 'reid.pt'


@pytest.mark.parametrize('low,high,new', [(.4, .3, .3), (-.1, .3, .3), (.1, 1.1, .3), (.1, .3, 1.1)])
def test_tracker_thresholds_reject_invalid_values(low, high, new):
    with pytest.raises(ValueError, match='thresholds'):
        configured_tracker({}, high=high, low=low, new=new,
            minimum_length=2, buffer_frames=30, with_reid=False, reid_model='unused')


def test_frame_analysis_separates_track_confirmation_from_court_rejection():
    raw = [
        dict(bbox_xyxy=[1, 1, 5, 5], det_confidence=.8, track_id=None),
        dict(bbox_xyxy=[6, 1, 10, 5], det_confidence=.4, track_id=None),
        dict(bbox_xyxy=[70, 70, 80, 80], det_confidence=.9, track_id=None),
    ]
    tracks = [dict(bbox_xyxy=[1, 1, 5, 5], det_confidence=.8, track_id=1)]
    source = dict(image_to_court_h=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  calibration=dict(method='automatic_test'))
    detections, tracks, counts = analyze_frame(raw, tracks, source, (100, 100))
    assert counts == dict(raw_detections=3, on_court_detections=2,
                          raw_confirmed_tracks=1, confirmed_tracks=1,
                          on_court_detections_matched_to_track=1,
                          on_court_detections_not_confirmed=1, outside_court_gate=1)
    assert [row['track_match_status'] for row in detections] == [
        'confirmed_track', 'detected_not_confirmed', 'outside_court_gate']
    assert tracks[0]['diagnostic_court_keep'] is True
