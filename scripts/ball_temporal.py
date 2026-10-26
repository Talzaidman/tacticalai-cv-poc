"""CPU-only image-space Kalman association and bounded offline gap interpolation.

No court calibration or evaluation labels enter this module. Motion limits are
configurable image-space heuristics, not physical basketball speed measurements.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TemporalConfig:
    min_confidence: float = 0.25
    confirmation_frames: int = 3
    max_gap_s: float = 0.20
    interpolation_gap_s: float = 0.16
    max_speed_diagonal_s: float = 1.0
    max_innovation_diagonal: float = 0.04
    measurement_std_diagonal: float = 0.005
    acceleration_std_diagonal_s2: float = 2.0
    mahalanobis_squared_gate: float = 9.21

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Temporal parameters must be positive and finite')
        if not isinstance(self.confirmation_frames, int) or self.confirmation_frames < 2:
            raise ValueError('Require at least two confirmation frames')
        if self.min_confidence > 1 or self.interpolation_gap_s > self.max_gap_s:
            raise ValueError('Invalid confidence or interpolation horizon')


def _valid_candidates(detections, width, height, config):
    valid = []
    for index, d in enumerate(detections):
        if all(k in d and math.isfinite(d[k]) for k in ('x', 'y', 'conf')):
            if 0 <= d['x'] < width and 0 <= d['y'] < height and config.min_confidence <= d['conf'] <= 1:
                valid.append((index, d))
    return valid


def track_temporal(detections, width, height, fps, config=TemporalConfig()):
    """Return one record per frame; never promote an inferred point to a measurement."""
    if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        raise ValueError('Invalid dimensions or FPS')
    diag = math.hypot(width, height)
    dt = 1.0 / fps
    transition = np.eye(4)
    transition[0, 2] = transition[1, 3] = dt
    observation = np.eye(2, 4)
    noise_map = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]])
    process_noise = noise_map @ noise_map.T * (diag*config.acceleration_std_diagonal_s2)**2
    measurement_noise = np.eye(2) * (diag*config.measurement_std_diagonal)**2
    state = covariance = last_point = None
    last_frame = None
    pending = []
    segment = 0
    output = []
    max_gap = max(1, math.floor(config.max_gap_s*fps + 1e-9))
    for frame, raw in enumerate(detections):
        candidates = _valid_candidates(raw, width, height, config)
        record = dict(frame_idx=frame, timestamp_s=frame/fps,
                      state='none', x=None, y=None, conf=None, segment_id=None,
                      position_source='unknown', measurement_xy=None, filtered_xy=None,
                      covariance_xy=None, accepted_candidate_index=None, rejected=[],
                      reason='no_supported_measurement')
        if state is not None and frame-last_frame > max_gap:
            state = covariance = last_point = None
            pending = []
        if state is None:
            # A new segment needs a coherent streak; confidence alone cannot teleport it.
            if pending:
                previous_frame, previous = pending[-1]
                radius = diag*config.max_speed_diagonal_s*(frame-previous_frame)/fps
                eligible = [(index, d) for index, d in candidates
                            if math.hypot(d['x']-previous['x'], d['y']-previous['y']) <= radius]
            else:
                eligible = candidates
            if eligible:
                index, chosen = max(eligible, key=lambda c: c[1]['conf'])
                pending.append((frame, chosen))
            else:
                pending = []
                if candidates:
                    index, chosen = max(candidates, key=lambda c: c[1]['conf'])
                    pending = [(frame, chosen)]
            if len(pending) >= config.confirmation_frames:
                first_frame, first = pending[0]
                elapsed = (frame-first_frame)/fps
                velocity = (np.array([chosen['x'], chosen['y']])-np.array([first['x'], first['y']]))/elapsed
                state = np.array([chosen['x'], chosen['y'], *velocity], dtype=float)
                covariance = np.diag([(diag*config.measurement_std_diagonal)**2]*2 + [(diag*.2)**2]*2)
                last_point = np.array([chosen['x'], chosen['y']])
                last_frame = frame
                segment += 1
                pending = []
                record.update(state='detected', x=float(state[0]), y=float(state[1]),
                              conf=chosen['conf'], segment_id=segment, measurement_xy=last_point.tolist(),
                              filtered_xy=state[:2].tolist(), accepted_candidate_index=index,
                              covariance_xy=covariance[:2, :2].tolist(),
                              position_source='measurement_filtered', reason='confirmed_new_segment')
            elif pending:
                record['reason'] = 'pending_temporal_confirmation'
        else:
            predicted = transition @ state
            prior_covariance = transition @ covariance @ transition.T + process_noise
            innovation_covariance = observation @ prior_covariance @ observation.T + measurement_noise
            ranked = []
            for index, candidate in candidates:
                point = np.array([candidate['x'], candidate['y']])
                residual = point-predicted[:2]
                distance = float(np.linalg.norm(residual))
                motion = float(np.linalg.norm(point-last_point))
                elapsed = (frame-last_frame)/fps
                mahalanobis = float(residual @ np.linalg.solve(innovation_covariance, residual))
                if motion > diag*config.max_speed_diagonal_s*elapsed:
                    reason = 'speed_gate'
                elif distance > diag*config.max_innovation_diagonal:
                    reason = 'innovation_distance_gate'
                elif mahalanobis > config.mahalanobis_squared_gate:
                    reason = 'mahalanobis_gate'
                else:
                    reason = None
                    ranked.append((mahalanobis-.5*candidate['conf'], index, candidate, residual))
                if reason:
                    record['rejected'].append(dict(candidate_index=index, reason=reason,
                                                    innovation_px=distance, mahalanobis_squared=mahalanobis))
            if ranked:
                _, index, chosen, residual = min(ranked, key=lambda item: item[0])
                gain = np.linalg.solve(innovation_covariance, observation @ prior_covariance).T
                state = predicted + gain @ residual
                correction = np.eye(4)-gain @ observation
                covariance = correction @ prior_covariance @ correction.T + gain @ measurement_noise @ gain.T
                last_point = np.array([chosen['x'], chosen['y']])
                last_frame = frame
                record.update(state='detected', conf=chosen['conf'], measurement_xy=last_point.tolist(),
                              accepted_candidate_index=index, position_source='measurement_filtered',
                              reason='accepted_motion_gates')
            else:
                state, covariance = predicted, prior_covariance
                record.update(state='predicted', position_source='kalman_extrapolation',
                              reason='motion_rejected' if candidates else 'missing_measurement')
            if 0 <= state[0] < width and 0 <= state[1] < height:
                record.update(x=float(state[0]), y=float(state[1]), segment_id=segment,
                              filtered_xy=state[:2].tolist(), covariance_xy=covariance[:2, :2].tolist())
            else:
                state = covariance = last_point = None
                pending = []
                record.update(state='none', x=None, y=None, segment_id=None,
                              conf=None, measurement_xy=None, filtered_xy=None,
                              covariance_xy=None, accepted_candidate_index=None,
                              position_source='unknown', reason='prediction_outside_image')
        output.append(record)

    # Offline interpolation uses two accepted endpoints, only inside the same segment.
    maximum = math.floor(config.interpolation_gap_s*fps + 1e-9)
    accepted = [i for i, r in enumerate(output) if r['state']=='detected']
    for left, right in zip(accepted, accepted[1:]):
        a, b = output[left], output[right]
        if not (0 < right-left-1 <= maximum and a['segment_id']==b['segment_id']):
            continue
        gap = output[left+1:right]
        if not all(r['state']=='predicted' and r['segment_id']==a['segment_id'] for r in gap):
            continue
        for index in range(left+1, right):
            weight = (index-left)/(right-left)
            point = (1-weight)*np.array([a['x'], a['y']])+weight*np.array([b['x'], b['y']])
            output[index].update(state='interpolated', x=float(point[0]), y=float(point[1]),
                                 conf=None, measurement_xy=None, covariance_xy=None,
                                 position_source='offline_linear_interpolation', reason='bounded_gap_between_measurements',
                                 interpolation_endpoints=[left, right])
    return output


def trail_segments(tracked, index, length=25):
    """Only connect consecutive accepted measurements in the same segment."""
    lines, current = [], []
    previous = None
    for i in range(max(0, index-length), index+1):
        point = tracked[i]
        if point['state'] != 'detected' or (previous is not None and
                point.get('segment_id') != previous.get('segment_id')):
            if len(current)>1:
                lines.append(current)
            current = []
        if point['state']=='detected':
            current.append((point['x'], point['y']))
        previous = point
    if len(current)>1:
        lines.append(current)
    return lines
