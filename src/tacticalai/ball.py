"""Image-space ball association and conservative, experimental control estimates."""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BallConfig:
    detection_confidence: float = .15
    nms_iou: float = .4
    max_ball_height_fraction: float = .08
    max_gap_s: float = .12
    association_radius_diagonal: float = .035
    max_speed_diagonal_s: float = 1.5
    confirmation_frames: int = 2
    candidate_margin: float = .08
    possession_confirmation_s: float = .12
    possession_min_score: float = .55
    possession_margin: float = .12

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.confirmation_frames, int):
            raise ValueError('confirmation_frames must be an integer')
        for name in ('detection_confidence', 'nms_iou', 'max_ball_height_fraction',
                     'association_radius_diagonal', 'candidate_margin',
                     'possession_min_score', 'possession_margin'):
            if getattr(self, name) > 1:
                raise ValueError(f'{name} must be <= 1')


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = (a[2:]-a[:2]).prod() + (b[2:]-b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.


def ball_candidates(detections, image_size, config):
    """Merge full-frame/tile duplicates; retain provenance of the winning box."""
    width, height = image_size
    candidates = []
    for row in detections:
        box = np.asarray(row['bbox_xyxy'], float)
        confidence = float(row['det_confidence'])
        if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(confidence):
            raise ValueError('Nonfinite or malformed ball detection')
        if not 0 <= confidence <= 1 or np.any(box[2:] <= box[:2]):
            raise ValueError('Invalid ball confidence or box')
        if (np.any(box[:2] < 0) or box[2] > width or box[3] > height):
            raise ValueError('Ball box outside source image')
        if confidence < config.detection_confidence:
            continue
        if max(box[2:]-box[:2]) > height*config.max_ball_height_fraction:
            continue
        candidates.append(dict(row, center_image_xy=((box[:2]+box[2:])/2).tolist()))
    kept = []
    for row in sorted(candidates, key=lambda r: r['det_confidence'], reverse=True):
        if all(iou(row['bbox_xyxy'], other['bbox_xyxy']) < config.nms_iou for other in kept):
            kept.append(row)
    return kept


class BallTracker:
    """One active ball hypothesis; bounded predictions are never observations."""

    def __init__(self, image_size, fps, config=BallConfig()):
        if min(image_size) <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid source dimensions/FPS')
        self.config, self.fps = config, fps
        self.diagonal = math.hypot(*image_size)
        self.image_size = image_size
        self.last_index = None
        self.next_id = 1
        self.reset()

    def reset(self):
        self.position = self.velocity = None
        self.observed_index = None
        self.streak = 0
        self.track_id = None

    def update(self, frame_idx, detections, cut=False):
        if self.last_index is not None and frame_idx != self.last_index+1:
            raise ValueError('Ball tracker requires consecutive frames')
        self.last_index = frame_idx
        if cut:
            self.reset()
        config = self.config
        candidates = ball_candidates(detections, self.image_size, config)
        max_gap = max(1, round(config.max_gap_s*self.fps))
        if self.position is not None and frame_idx-self.observed_index > max_gap:
            self.reset()
        predicted = None
        selected = None
        reason = 'no_candidates'
        if self.position is not None:
            elapsed = frame_idx-self.observed_index
            predicted = self.position + elapsed*self.velocity
            radius = self.diagonal*(config.association_radius_diagonal +
                                     config.max_speed_diagonal_s*elapsed/self.fps)
            ranked = []
            for candidate in candidates:
                distance = np.linalg.norm(np.asarray(candidate['center_image_xy'])-predicted)
                if distance <= radius:
                    ranked.append((candidate['det_confidence']-.5*distance/radius, candidate))
            ranked.sort(key=lambda item: item[0], reverse=True)
            if ranked:
                if len(ranked) == 1 or ranked[0][0]-ranked[1][0] >= config.candidate_margin:
                    selected = ranked[0][1]
                else:
                    reason = 'ambiguous_candidates'
            elif candidates:
                reason = 'outside_motion_gate'
        elif candidates:
            if len(candidates) == 1 or candidates[0]['det_confidence']-candidates[1]['det_confidence'] >= config.candidate_margin:
                selected = candidates[0]
            else:
                reason = 'ambiguous_candidates'
        if selected is not None:
            point = np.asarray(selected['center_image_xy'], float)
            if self.position is None:
                self.track_id = self.next_id
                self.next_id += 1
                self.velocity = np.zeros(2)
            else:
                delta = frame_idx-self.observed_index
                velocity = (point-self.position)/delta
                self.velocity = .5*self.velocity + .5*velocity
                if delta > 1:
                    self.streak = 0
            self.position, self.observed_index = point, frame_idx
            self.streak += 1
            status = 'observed' if self.streak >= config.confirmation_frames else 'tentative'
            return dict(status=status, reason='temporally_confirmed' if status == 'observed' else 'pending_confirmation',
                        track_id=self.track_id, center_image_xy=point.tolist(),
                        bbox_xyxy=selected['bbox_xyxy'], det_confidence=selected['det_confidence'],
                        predicted_center_image_xy=None, candidates=candidates, court_xy_m=None)
        self.streak = 0
        if predicted is not None:
            width, height = self.image_size
            if not (0 <= predicted[0] < width and 0 <= predicted[1] < height):
                predicted = None
                self.reset()
        return dict(status='predicted' if predicted is not None else 'unknown', reason=reason,
                    track_id=self.track_id, center_image_xy=None, bbox_xyxy=None, det_confidence=None,
                    predicted_center_image_xy=predicted.tolist() if predicted is not None else None,
                    candidates=candidates, court_xy_m=None)


class PossessionEstimator:
    """Repeated body-contact evidence, not verified hand contact or possession."""

    def __init__(self, fps, config=BallConfig()):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid FPS')
        self.required = max(2, math.ceil(fps*config.possession_confirmation_s))
        self.config = config
        self.key, self.streak = None, 0

    def update(self, ball, rows, cut=False):
        result = dict(status='unknown', controller_person_id=None, controller_track_id=None,
                      controller_identity_segment_id=None, controller_role=None, score=None,
                      score_kind='uncalibrated_body_proximity_heuristic',
                      reason='ball_not_observed', candidates=[], minimap_xy_m=None,
                      minimap_position_source=None)
        ranked = []
        if ball['status'] == 'observed':
            point = np.asarray(ball['center_image_xy'])
            for row in rows:
                person_id = row.get('player_id')
                if person_id is None or row.get('track_id') is None:
                    continue
                x1, y1, x2, y2 = row['bbox_xyxy']
                height, width = y2-y1, x2-x1
                if height <= 0 or width <= 0:
                    raise ValueError('Invalid person box')
                # Allow lateral hands and low dribbling, without calling it hand evidence.
                lower = np.array([x1-.25*width, y1+.12*height])
                upper = np.array([x2+.25*width, y2+.08*height])
                distance = np.linalg.norm(np.maximum(0, np.maximum(lower-point, point-upper)))/height
                score = max(0., 1-distance/.25-.25*abs(point[0]-(x1+x2)/2)/width)
                ranked.append((score, row))
            ranked.sort(key=lambda item: item[0], reverse=True)
            result['candidates'] = [dict(person_id=r['player_id'], track_id=r['track_id'], score=float(s))
                                    for s, r in ranked[:3]]
            result['reason'] = 'no_supported_controller'
        selected = None
        if ranked and ranked[0][0] >= self.config.possession_min_score:
            if len(ranked) > 1 and ranked[0][0]-ranked[1][0] < self.config.possession_margin:
                result['reason'] = 'ambiguous_people'
            else:
                selected = ranked[0][1]
        if cut:
            self.key, self.streak = None, 0
        if selected is None:
            self.key, self.streak = None, 0
            return result
        key = (selected['player_id'], selected.get('identity_segment_id'),
               selected['track_id'], ball['track_id'])
        self.streak = self.streak+1 if key == self.key else 1
        self.key = key
        result.update(score=float(ranked[0][0]), reason='pending_temporal_contact')
        if self.streak < self.required:
            return result
        result.update(status='estimated_control', reason='repeated_unambiguous_body_proximity',
                      controller_person_id=selected['player_id'],
                      controller_track_id=selected['track_id'],
                      controller_identity_segment_id=selected.get('identity_segment_id'),
                      controller_role=selected.get('role'))
        xy = selected.get('court_xy_m')
        if xy is not None and selected.get('court_region') == 'inside' and np.isfinite(xy).all():
            result.update(minimap_xy_m=list(xy), minimap_position_source='estimated_controller_floor_position')
        return result


def tile_windows(width, height, tile_size=960, overlap=.2):
    if min(width, height, tile_size) <= 0 or not 0 <= overlap < 1:
        raise ValueError('Invalid tile dimensions/overlap')
    def starts(length):
        step = max(1, round(tile_size*(1-overlap)))
        return sorted(set([*range(0, max(1, length-tile_size+1), step), max(0, length-tile_size)]))
    return [(x, y, min(x+tile_size, width), min(y+tile_size, height))
            for y in starts(height) for x in starts(width)]
