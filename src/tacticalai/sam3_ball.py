"""Pure postprocessing for SAM 3 basketball candidates.

SAM 3 supplies native video object IDs.  This module selects one small moving
track and converts it to the common per-frame ball schema.  It deliberately
rejects static broadcast overlays and never turns a prediction into a measured
ball observation.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Sam3SelectionConfig:
    minimum_coverage: float = .05
    maximum_radius_diagonal: float = .04
    minimum_speed_diagonal_frame: float = .0015
    maximum_prediction_s: float = .20

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError('SAM 3 selection parameters must be positive and finite')
        if self.minimum_coverage > 1 or self.maximum_radius_diagonal > 1:
            raise ValueError('Coverage and relative radius must not exceed one')


def select_dynamic_track(detections, width, height, config=Sam3SelectionConfig()):
    """Return candidates from one small moving native track plus diagnostics."""
    if min(width, height) <= 0 or not detections:
        raise ValueError('Invalid video dimensions or empty timeline')
    diagonal = math.hypot(width, height)
    histories = {}
    for frame_idx, rows in enumerate(detections):
        for row in rows:
            if not all(key in row for key in ('sam3_id', 'x', 'y', 'r', 'conf')):
                raise ValueError('Incomplete SAM 3 candidate')
            values = [row['x'], row['y'], row['r'], row['conf']]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Non-finite SAM 3 candidate')
            histories.setdefault(int(row['sam3_id']), []).append((frame_idx, row))
    minimum_observations = max(3, round(config.minimum_coverage * len(detections)))
    ranked, diagnostics = [], {}
    for object_id, samples in histories.items():
        radii = [row['r'] for _, row in samples]
        speeds = [math.hypot(right['x']-left['x'], right['y']-left['y']) /
                  max(1, right_frame-left_frame)
                  for (left_frame, left), (right_frame, right) in zip(samples, samples[1:])]
        median_radius = float(np.median(radii))
        mean_speed = float(np.mean(speeds)) if speeds else 0.
        eligible = (len(samples) >= minimum_observations and
                    median_radius <= config.maximum_radius_diagonal*diagonal and
                    mean_speed >= config.minimum_speed_diagonal_frame*diagonal)
        coverage = len(samples)/len(detections)
        diagnostics[str(object_id)] = dict(observations=len(samples), coverage=coverage,
            median_radius_px=median_radius, mean_speed_px_frame=mean_speed, eligible=eligible)
        if eligible:
            score = mean_speed*math.sqrt(coverage)/max(1., median_radius)
            ranked.append((score, object_id))
    selected_id = max(ranked)[1] if ranked else None
    selected = [[dict(row) for row in rows if int(row['sam3_id']) == selected_id]
                for rows in detections] if selected_id is not None else [[] for _ in detections]
    return selected, dict(selected_native_track_id=selected_id,
                          native_track_diagnostics=diagnostics,
                          selection='small moving native track; static broadcast graphics rejected')


def native_track_timeline(detections, width, height, fps,
                          config=Sam3SelectionConfig()):
    """Convert one selected native ID to measured/predicted/unknown records."""
    if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        raise ValueError('Invalid dimensions or FPS')
    maximum_gap = max(1, math.floor(config.maximum_prediction_s*fps + 1e-9))
    output, segment = [], 0
    last_frame = last_point = previous_frame = previous_point = None
    for frame_idx, rows in enumerate(detections):
        if len(rows) > 1:
            raise ValueError('Selected SAM 3 timeline must contain at most one candidate per frame')
        base = dict(frame_idx=frame_idx, timestamp_s=frame_idx/fps, state='none',
            x=None, y=None, conf=None, segment_id=None, position_source='unknown',
            measurement_xy=None, filtered_xy=None, covariance_xy=None,
            accepted_candidate_index=None, rejected=[], reason='no_native_measurement')
        if rows:
            row = rows[0]
            if last_frame is None or frame_idx-last_frame > maximum_gap:
                segment += 1
                previous_frame = previous_point = None
            else:
                previous_frame, previous_point = last_frame, last_point
            last_frame = frame_idx
            last_point = np.asarray([row['x'], row['y']], dtype=float)
            base.update(state='detected', x=float(last_point[0]), y=float(last_point[1]),
                conf=float(row['conf']), segment_id=segment,
                position_source='sam3_native_measurement', measurement_xy=last_point.tolist(),
                filtered_xy=last_point.tolist(), accepted_candidate_index=0,
                reason='selected_native_sam3_track')
        elif last_frame is not None and 0 < frame_idx-last_frame <= maximum_gap and previous_frame is not None:
            velocity = (last_point-previous_point)/max(1, last_frame-previous_frame)
            point = last_point + velocity*(frame_idx-last_frame)
            if 0 <= point[0] < width and 0 <= point[1] < height:
                base.update(state='predicted', x=float(point[0]), y=float(point[1]),
                    segment_id=segment, position_source='bounded_native_velocity_prediction',
                    filtered_xy=point.tolist(), reason='short_native_track_gap')
        if last_frame is not None and frame_idx-last_frame > maximum_gap:
            last_frame = last_point = previous_frame = previous_point = None
        output.append(base)
    return output
