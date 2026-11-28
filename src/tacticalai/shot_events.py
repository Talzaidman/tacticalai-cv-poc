"""Infer shot candidates from measured image-space basket approaches."""
from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np

from .court import BASKET_OFFSET_M


@dataclass(frozen=True)
class ShotConfig:
    maximum_holder_age_s: float = 1.25
    minimum_flight_s: float = .08
    maximum_release_observation_delay_s: float = .24
    maximum_approach_gap_s: float = .20
    minimum_rival_wrist_score: float = .55
    contact_cooldown_s: float = .65
    endpoint_hold_s: float = .35
    maximum_camera_age_s: float = .25
    maximum_target_image_error_fraction: float = .04

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Shot settings must be positive and finite')
        if self.minimum_flight_s > self.maximum_holder_age_s:
            raise ValueError('Minimum shot flight exceeds the holder cutoff')
        if max(self.minimum_rival_wrist_score, self.maximum_target_image_error_fraction) > 1:
            raise ValueError('Shot score and image error fractions must not exceed one')


def _measurement(frame):
    ball = frame['ball_temporal']
    if ball['state'] != 'detected' or ball.get('measurement_xy') is None:
        return None
    candidates = frame.get('ball_raw_candidates', [])
    accepted = ball.get('accepted_candidate_index')
    radius = candidates[accepted].get('r', 0.) if accepted is not None and 0 <= accepted < len(candidates) else 0.
    return np.asarray(ball['measurement_xy'], dtype=float), max(1., radius)


def _distance_to_box(point, box):
    return float(np.linalg.norm(np.maximum(0., np.maximum(box[:2]-point, point-box[2:]))))


def _contacts(frames, recent, index, measurement, fps, config):
    point, radius = measurement
    for hardware in frames[index].get('basket_hardware', []):
        box = np.asarray(hardware['bbox_xyxy'])
        distance = _distance_to_box(point, box)
        if distance > .75*radius:
            continue
        for prior_index, (prior_point, prior_radius) in reversed(recent):
            if (index-prior_index)/fps > config.maximum_approach_gap_s+1e-9:
                break
            prior_hardware = next((row for row in frames[prior_index].get('basket_hardware', [])
                                   if row['track_id'] == hardware['track_id'] and
                                   row['kind'] == hardware['kind']), None)
            if prior_hardware is None:
                continue
            previous_distance = _distance_to_box(prior_point, np.asarray(prior_hardware['bbox_xyxy']))
            if previous_distance >= 1.5*prior_radius and previous_distance-distance >= radius:
                yield hardware, prior_index, distance, previous_distance
                break


def _control(frame):
    return frame.get('possession_raw', frame['possession'])


def _shooter(frames, index, holder_index, approach_index, recent, fps, config):
    if holder_index is None:
        return 'no_measured_holder_in_ball_segment', None
    age = (index-holder_index)/fps
    if age > config.maximum_holder_age_s+1e-9:
        return 'stale_holder', None
    if age+1e-9 < config.minimum_flight_s:
        return 'insufficient_time_after_last_control', None
    flight = [i for i, _ in recent if i > holder_index]
    if (approach_index <= holder_index or not flight or
            (flight[0]-holder_index)/fps > config.maximum_release_observation_delay_s+1e-9):
        return 'missing_ball_evidence_after_control', None
    holder = _control(frames[holder_index])['controller_person_id']
    for i in flight:
        if any(candidate.get('person_id') != holder and
               candidate.get('geometry_source') == 'wrist_keypoints' and
               candidate.get('contact_score', 0.) >= config.minimum_rival_wrist_score
               for candidate in _control(frames[i]).get('candidates', [])):
            return 'competing_wrist_contact', i
    return 'recent_measured_holder', None


def _basket_end(frames, index, hardware, fps, width, height, court, config):
    """Match observed hardware against projections of two nominal 3D targets."""
    frame = frames[index]
    matrix = frame.get('image_to_court_h')
    if matrix is None or frame['calibration'].get('shot') is None:
        return None, 'missing_camera_at_contact', None
    camera_shot = frame['calibration']['shot']
    first = max(0, index-math.floor(config.maximum_camera_age_s*fps+1e-9))
    for anchor in range(index, first-1, -1):
        calibration = frames[anchor]['calibration']
        if calibration.get('shot') != camera_shot:
            return None, 'missing_recent_camera_pose', None
        camera = (calibration.get('evidence') or {}).get('camera') or {}
        anchor_h = frames[anchor].get('image_to_court_h')
        if anchor_h is not None and camera.get('rotation') is not None and camera.get('reason') is None:
            break
        if frames[anchor].get('ball_camera_cut_reset'):
            return None, 'missing_recent_camera_pose', None
    else:
        return None, 'missing_recent_camera_pose', None

    length, center = court['length_m'], court['width_m']/2
    # Configured FIBA-style geometry for the rim center and backboard face center.
    offset, z = (BASKET_OFFSET_M, 3.05) if hardware['kind'] == 'hoop' else (1.2, 3.45)
    world_cm = 100*np.array([[offset, center, z], [length-offset, center, z]])
    rotated = world_cm @ np.asarray(camera['rotation']).T + np.asarray(camera['translation_cm'])
    source_scale = np.diag([width/960, height/540, 1.])
    anchor_pixels = rotated @ (source_scale @ np.asarray(camera['intrinsic'])).T
    # Carry the nearby calibrated image projection through the intervening floor flow.
    try:
        pixels = np.linalg.solve(np.asarray(matrix), np.asarray(anchor_h) @ anchor_pixels.T).T
    except np.linalg.LinAlgError:
        return None, 'invalid_contact_camera', anchor
    box = np.asarray(hardware['bbox_xyxy'])
    tolerance = config.maximum_target_image_error_fraction*math.hypot(width, height)
    distances = []
    for side, point, depth in zip(('left', 'right'), pixels, rotated[:, 2]):
        if depth <= 0 or not np.isfinite(point).all() or abs(point[2]) < 1e-9:
            continue
        xy = point[:2]/point[2]
        if _distance_to_box(xy, box) <= tolerance:
            distances.append(side)
    if len(distances) != 1:
        return None, 'ambiguous_basket_end' if distances else 'hardware_camera_mismatch', anchor
    side = distances[0]
    return (side, [BASKET_OFFSET_M if side == 'left' else length-BASKET_OFFSET_M, center]), 'camera_matched', anchor


def infer_shot_events(frames, info, court, config=ShotConfig()):
    """Annotate an endpoint animation and retain contact and attribution evidence."""
    fps = info['fps']
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid FPS')
    events = []
    recent = deque()
    history_window = max(config.maximum_holder_age_s, config.maximum_approach_gap_s)
    continuity = last_holder = last_contact = None
    for frame in frames:
        frame['shot_event'] = None
    for index, frame in enumerate(frames):
        scene = frame['calibration'].get('shot')
        segment = frame['ball_temporal'].get('segment_id')
        if index == 0 or scene != frames[index-1]['calibration'].get('shot') or frame.get('ball_camera_cut_reset'):
            continuity = last_holder = last_contact = None
            recent.clear()
        if segment is None or (scene, segment) != continuity:
            continuity = (scene, segment)
            last_holder = None
            recent.clear()
        measurement = _measurement(frame)
        if measurement is None or segment is None:
            continue
        while recent and (index-recent[0][0])/fps > history_window+1e-9:
            recent.popleft()
        control = _control(frame)
        if control['status'] == 'estimated_control' and control['controller_person_id'] is not None:
            last_holder = index
        # A new observed release can qualify before the cooldown for the previous shot ends.
        if (last_contact is not None and (index-last_contact)/fps < config.contact_cooldown_s-1e-9
                and (last_holder is None or (index-last_holder)/fps+1e-9 < config.minimum_flight_s)):
            recent.append((index, measurement))
            continue
        contact = min(_contacts(frames, recent, index, measurement, fps, config),
                      key=lambda hit: (hit[0]['kind'] != 'hoop', hit[2], -hit[0]['confidence']), default=None)
        if contact is not None:
            hardware, approach_frame, distance, previous_distance = contact
            reason, rival_frame = _shooter(frames, index, last_holder, approach_frame, recent, fps, config)
            holder = _control(frames[last_holder]) if last_holder is not None else None
            shooter = holder['controller_person_id'] if reason == 'recent_measured_holder' else None
            start = holder.get('minimap_xy_m') if shooter is not None else None
            target, target_reason, camera_frame = _basket_end(
                frames, index, hardware, fps, info['width'], info['height'], court, config)
            side, end = target if target is not None else (None, None)
            release = last_holder+1 if shooter is not None else None
            display_start = release if start is not None and end is not None else index
            display_end = min(len(frames)-1, index+math.floor(config.endpoint_hold_s*fps+1e-9))
            event = dict(event_id=f'shot-{len(events)+1:04d}', contact_frame=index,
                         approach_frame=approach_frame, contact_kind=hardware['kind'],
                         hardware_track_id=hardware['track_id'], hardware_confidence=hardware['confidence'],
                         ball_segment_id=segment, ball_image_xy=measurement[0].tolist(),
                         ball_box_distance_px=distance, approach_box_distance_px=previous_distance,
                         shooter_player_id=shooter, attribution_reason=reason, rival_wrist_frame=rival_frame,
                         last_measured_holder_frame=last_holder,
                         last_measured_holder_age_s=(index-last_holder)/fps if holder is not None else None,
                         last_measured_holder_player_id=holder['controller_person_id'] if holder is not None else None,
                         release_frame=release, start_xy_m=start, end_xy_m=end, basket_end=side,
                         basket_end_reason=target_reason, camera_pose_frame=camera_frame,
                         display_end_frame=display_end, made=None,
                         method='offline_measured_ball_approach_to_sam3_hardware_box',
                         true_airborne_trajectory_known=False)
            for i in range(display_start, display_end+1):
                progress = (min(i, index)-display_start)/max(1, index-display_start) if i < index else 1.
                frames[i]['shot_event'] = dict(event, progress=progress,
                    phase='inferred_in_flight' if i < index else 'possible_contact' if i == index else 'endpoint_hold')
            events.append(event)
            last_contact = index
            last_holder = None
            recent.clear()
        recent.append((index, measurement))
    return dict(configuration=asdict(config), events=events, contact_candidates=len(events),
                attributed_shots=sum(event['shooter_player_id'] is not None for event in events),
                offline_future_frames_used=True)
