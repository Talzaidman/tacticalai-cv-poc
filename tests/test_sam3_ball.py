import math

import pytest

from tacticalai.sam3_ball import Sam3SelectionConfig, native_track_timeline, select_dynamic_track


def candidate(object_id, x, y, radius=12., confidence=.7):
    return dict(sam3_id=object_id, x=x, y=y, r=radius, conf=confidence)


def test_selects_small_moving_ball_and_rejects_static_and_large_graphics():
    frames = []
    for index in range(20):
        rows = [candidate(0, 900, 510, radius=20)]
        rows.append(candidate(2, 200+8*index, 250+2*index, radius=10))
        if index >= 5: rows.append(candidate(3, 350, 520, radius=150))
        frames.append(rows)
    selected, evidence = select_dynamic_track(frames, 1000, 600)
    assert evidence['selected_native_track_id'] == 2
    assert all(len(rows) == 1 and rows[0]['sam3_id'] == 2 for rows in selected)
    assert evidence['native_track_diagnostics']['0']['eligible'] is False
    assert evidence['native_track_diagnostics']['3']['eligible'] is False


def test_no_supported_track_stays_unknown_instead_of_selecting_overlay():
    frames = [[candidate(0, 900, 510)] for _ in range(20)]
    selected, evidence = select_dynamic_track(frames, 1000, 600)
    assert evidence['selected_native_track_id'] is None
    assert all(not rows for rows in selected)
    assert all(row['state'] == 'none' for row in native_track_timeline(selected, 1000, 600, 25))


def test_native_timeline_keeps_measurements_distinct_from_short_predictions():
    frames = [[candidate(2, 100, 100)], [candidate(2, 110, 100)], [], [],
              [candidate(2, 140, 100)]]
    tracked = native_track_timeline(frames, 1000, 600, 25,
                                    Sam3SelectionConfig(maximum_prediction_s=.12))
    assert [row['state'] for row in tracked] == ['detected','detected','predicted','predicted','detected']
    assert tracked[0]['accepted_candidate_index'] == 0
    assert tracked[2]['measurement_xy'] is None and tracked[2]['conf'] is None
    assert tracked[2]['position_source'] == 'bounded_native_velocity_prediction'
    assert math.isclose(tracked[2]['x'], 120)


def test_selection_config_rejects_invalid_values():
    with pytest.raises(ValueError): Sam3SelectionConfig(minimum_coverage=0)
    with pytest.raises(ValueError): Sam3SelectionConfig(maximum_radius_diagonal=2)
