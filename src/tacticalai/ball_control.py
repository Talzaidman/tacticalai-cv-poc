"""Silhouette contact and short-window controller hypotheses, entirely on CPU.

This is an image-space heuristic. It does not detect hands or prove possession.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ControlConfig:
    support_window_s: float = .24
    minimum_measurements: int = 3
    minimum_contact_score: float = .35
    ambiguity_margin: float = .10
    max_contact_distance_heights: float = .45
    max_relative_speed_heights_s: float = 8.0
    continuity_s: float = .16
    pose_required: bool = False
    keypoint_confidence: float = .25
    wrist_distance_heights: float = .25
    dribble_continuity_s: float = .28

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for k,v in asdict(self).items() if k!='pose_required'):
            raise ValueError('Controller parameters must be positive and finite')
        if not isinstance(self.minimum_measurements, int) or self.minimum_measurements < 2:
            raise ValueError('Require at least two measured support frames')
        if self.minimum_contact_score > 1 or self.ambiguity_margin > 1:
            raise ValueError('Scores/margins must be <= 1')
        if self.continuity_s > self.support_window_s:
            raise ValueError('Continuity must not exceed the evidence window')


def contact_geometry(point, row, config=ControlConfig()):
    x1,y1,x2,y2 = map(float,row['bbox_xyxy'])
    height,width = y2-y1,x2-x1
    if height <= 0 or width <= 0:
        raise ValueError('Invalid person box')
    xy = np.asarray(point, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError('Invalid ball point')
    polygon = np.asarray(row.get('silhouette_xy') or [],dtype=np.float32)
    if len(polygon)>=3 and polygon.ndim==2 and polygon.shape[1]==2 and np.isfinite(polygon).all():
        distance = max(0.,-cv2.pointPolygonTest(polygon,tuple(map(float,xy)),True))/height
        source = 'visible_silhouette'
        penalty = 0.
    else:
        distance = float(np.linalg.norm(np.maximum(0.,np.maximum(
            np.array([x1,y1])-xy,xy-np.array([x2,y2])))))/height
        source = 'box_fallback'
        penalty = .12
    vertical = (xy[1]-y1)/height
    relative = ((xy-np.array([(x1+x2)/2,y1]))/height).tolist()
    score = max(0.,1-distance/.55-penalty)
    if vertical < .12:
        score *= .75  # Head-level projection is weaker hand/contact evidence.
    if distance > config.max_contact_distance_heights or not -.2 <= vertical <= 1.2:
        score = 0.
    body_score = score
    pose = row.get('control_pose')
    wrist_distance = wrist_confidence = None
    if pose is not None:
        keypoints = np.asarray(pose['keypoints_xy_conf'],dtype=float)
        wrists = [keypoints[index] for index in (9,10)
                  if keypoints.shape==(17,3) and np.isfinite(keypoints[index]).all()
                  and keypoints[index,2]>=config.keypoint_confidence]
        if wrists:
            wrist = min(wrists,key=lambda p:np.linalg.norm(p[:2]-xy))
            wrist_distance = float(np.linalg.norm(wrist[:2]-xy)/height)
            wrist_confidence = float(wrist[2])
            score = max(0.,1-wrist_distance/.35)*min(1.,wrist_confidence/.6)
            if wrist_distance>config.wrist_distance_heights or body_score==0:
                score=0.
            source='wrist_keypoints'
        else:
            score=0.;source='wrists_unavailable'
    elif config.pose_required:
        score=0.;source='pose_unavailable'
    return dict(contact_score=float(score),body_contact_score=float(body_score),distance_heights=float(distance),
                wrist_distance_heights=wrist_distance,wrist_confidence=wrist_confidence,
                geometry_source=source,relative_xy=relative,vertical_fraction=float(vertical))


def identity(row):
    return (row['player_id'],row.get('identity_segment_id'),row['track_id'])


def empty_control(reason):
    return dict(status='unknown',controller_person_id=None,controller_track_id=None,
                controller_identity_segment_id=None,controller_role=None,score=None,
                score_kind='uncalibrated_silhouette_temporal_heuristic',reason=reason,
                candidates=[],minimap_xy_m=None,minimap_position_source=None,
                evidence_source='none',support_measurements=0)


class TemporalControlEstimator:
    def __init__(self,fps,config=ControlConfig()):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid FPS')
        self.fps,self.config = fps,config
        self.index = -1
        self.history = deque()
        self.relative = {}
        self.owner = None
        self.segment = None
        self.last_owner_measurement = None

    def _continuity(self,result,by_key,candidates,frame,inferred):
        config=self.config
        if self.owner not in by_key or self.last_owner_measurement is None:
            return result
        row,candidate=by_key[self.owner]
        if (candidate['relative_speed_heights_s'] is not None and
                candidate['relative_speed_heights_s']>config.max_relative_speed_heights_s):
            result['reason']='relative_motion_breaks_continuity';return result
        horizon=config.continuity_s if inferred else config.dribble_continuity_s
        if frame-self.last_owner_measurement>math.floor(horizon*self.fps+1e-9):
            result['reason']='continuity_expired';return result
        if candidate['body_contact_score']<config.minimum_contact_score:
            result['reason']='owner_not_near_ball';return result
        if any(key!=self.owner and other['contact_score']>=config.minimum_contact_score
               for _,key,other in candidates):
            result['reason']='ambiguous_continuity';return result
        return self._assigned(result,row,candidate,'estimated_control_continuity',
                              'recent_wrist_measurement_memory' if config.pose_required else 'recent_measurement_memory')

    def update(self, ball, rows, cut=False):
        self.index += 1
        frame = self.index
        config = self.config
        segment = ball.get('track_id')
        if cut or segment != self.segment:
            self.history.clear();self.relative.clear();self.owner=None
            self.last_owner_measurement=None
        self.segment = segment
        window = max(config.minimum_measurements,math.ceil(config.support_window_s*self.fps))
        while self.history and frame-self.history[0][0]>=window:
            self.history.popleft()
        measured = ball.get('status')=='observed'
        point = ball.get('center_image_xy') if measured else ball.get('inferred_center_image_xy')
        result = empty_control('ball_unavailable' if point is None else 'no_supported_controller')
        if config.pose_required:
            result['score_kind']='uncalibrated_wrist_temporal_heuristic'
        if point is None:
            self.owner = None
            return result
        evidence = {}
        by_key = {}
        candidates = []
        for row in rows:
            key = identity(row)
            geometry = contact_geometry(point,row,config)
            score = geometry['contact_score']
            relative_speed = None
            if measured:
                prior = self.relative.get(key)
                if prior is not None:
                    previous_frame,previous_relative = prior
                    elapsed = (frame-previous_frame)/self.fps
                    if 0 < elapsed <= config.support_window_s:
                        relative_speed = float(np.linalg.norm(np.asarray(geometry['relative_xy'])-previous_relative)/elapsed)
                        if relative_speed > config.max_relative_speed_heights_s:
                            score = 0.
                self.relative[key] = (frame,np.asarray(geometry['relative_xy']))
                if score >= config.minimum_contact_score:
                    evidence[key] = score
            support = [entry[key] for _,entry in self.history if key in entry]
            if key in evidence:
                support.append(score)
            # Missing measurements reduce support rather than resetting an otherwise coherent window.
            temporal_strength = sum(support)/window
            rank = .6*score + .4*temporal_strength
            candidate = dict(person_id=key[0],track_id=key[2],score=float(rank),
                             contact_score=float(score),support_measurements=len(support),
                             relative_speed_heights_s=relative_speed,**{k:v for k,v in geometry.items() if k!='contact_score'})
            candidates.append((rank,key,candidate))
            by_key[key] = (row,candidate)
        if measured:
            self.history.append((frame,evidence))
        candidates.sort(key=lambda value:value[0],reverse=True)
        result['candidates'] = [entry[2] for entry in candidates]
        if not measured:
            # Continuity can only retain an already established owner, never create/change one.
            return self._continuity(result,by_key,candidates,frame,True)
        supported = [entry for entry in candidates if entry[2]['contact_score']>=config.minimum_contact_score
                     and entry[2]['support_measurements']>=config.minimum_measurements]
        if not supported:
            if config.pose_required and not evidence:
                continuity=self._continuity(result,by_key,candidates,frame,False)
                if continuity['controller_person_id'] is not None:
                    return continuity
            self.owner = None
            result['reason']='pending_temporal_support' if evidence else 'no_contact_or_relative_motion'
            return result
        rank,key,candidate = supported[0]
        # A stronger but unconfirmed rival also blocks assignment, rather than being ignored.
        rivals = [other_rank for other_rank,other_key,other in candidates
                  if other_key != key and other['contact_score']>=config.minimum_contact_score]
        if rivals and rank-max(rivals)<config.ambiguity_margin:
            self.owner = None
            result['reason']='ambiguous_people';return result
        self.owner=key;self.last_owner_measurement=frame
        return self._assigned(result,by_key[key][0],candidate,'estimated_control','current_measurement_window')

    @staticmethod
    def _assigned(result,row,candidate,status,evidence_source):
        result.update(status=status,controller_person_id=row['player_id'],controller_track_id=row['track_id'],
                      controller_identity_segment_id=row.get('identity_segment_id'),controller_role=row.get('role'),
                      score=candidate['score'],reason=('wrist_and_temporal_support' if candidate['geometry_source']=='wrist_keypoints'
                                                      else 'silhouette_and_temporal_support') if status=='estimated_control'
                      else 'bounded_continuity_from_measured_support',evidence_source=evidence_source,
                      support_measurements=candidate['support_measurements'])
        xy=row.get('court_xy_m')
        if xy is not None and row.get('court_region')=='inside' and np.isfinite(xy).all():
            result.update(minimap_xy_m=list(xy),minimap_position_source='estimated_controller_floor_position')
        return result
