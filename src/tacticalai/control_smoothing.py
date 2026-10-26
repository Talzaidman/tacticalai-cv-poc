"""Bounded, offline smoothing of categorical ball-controller assignments.

Only unknown intervals bracketed by the same measured controller can be filled.
No ball measurements, player tracks, identity links or poses are modified.
"""
from dataclasses import asdict, dataclass
import copy
import math

import numpy as np

from .ball_control import contact_geometry


@dataclass(frozen=True)
class ControlSmoothingConfig:
    max_gap_s: float = .60
    max_missing_player_s: float = .12
    max_body_distance_heights: float = .65
    max_horizontal_heights: float = 1.0
    minimum_vertical_fraction: float = -.15
    maximum_vertical_fraction: float = 1.35
    strong_rival_contact: float = .70
    repeated_rival_contact: float = .35
    repeated_rival_measurements: int = 2

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Smoothing settings must be finite')
        if min(self.max_gap_s,self.max_missing_player_s,self.max_body_distance_heights,self.max_horizontal_heights)<=0:
            raise ValueError('Smoothing durations/distances must be positive')
        if not 0 <= self.repeated_rival_contact <= self.strong_rival_contact <= 1:
            raise ValueError('Invalid rival thresholds')
        if not isinstance(self.repeated_rival_measurements,int) or self.repeated_rival_measurements<1:
            raise ValueError('Invalid rival support count')
        if self.minimum_vertical_fraction>=self.maximum_vertical_fraction:
            raise ValueError('Invalid dribble envelope')


def owner_key(control):
    if control.get('controller_person_id') is None:
        return None
    return (control['controller_person_id'],control.get('controller_identity_segment_id'),control['controller_track_id'])


def find_row(frame,key):
    return next((row for row in frame['tracks'] if
                 (row.get('player_id'),row.get('identity_segment_id'),row.get('track_id'))==key
                 and row.get('role')=='player' and row.get('display_on_video',True)),None)


def continuity_row(frames,index,key,fps,config):
    row=find_row(frames[index],key)
    if row is not None:
        return row,True
    limit=math.floor(config.max_missing_player_s*fps+1e-9)
    left=index-1
    while left>=0 and index-left<=limit and find_row(frames[left],key) is None:
        left-=1
    right=index+1
    while right<len(frames) and right-index<=limit and find_row(frames[right],key) is None:
        right+=1
    if left<0 or right>=len(frames) or right-left-1>limit:
        return None,False
    a,b=find_row(frames[left],key),find_row(frames[right],key)
    if a is None or b is None:
        return None,False
    if frames[left]['calibration'].get('shot')!=frames[right]['calibration'].get('shot'):
        return None,False
    weight=(index-left)/(right-left)
    box=((1-weight)*np.asarray(a['bbox_xyxy'])+weight*np.asarray(b['bbox_xyxy'])).tolist()
    # Geometry only: an interpolated box is never inserted into the detection/track list.
    return dict(a,bbox_xyxy=box,silhouette_xy=None,court_xy_m=None,court_region='unknown'),False


def smooth_control_gaps(frames,fps,config=ControlSmoothingConfig()):
    if not math.isfinite(fps) or fps<=0:
        raise ValueError('Invalid FPS')
    output=copy.deepcopy(frames)
    for frame in output:
        frame['possession_raw']=copy.deepcopy(frame['possession'])
    maximum=math.floor(config.max_gap_s*fps+1e-9)
    decisions=[]
    index=0
    while index<len(frames):
        if owner_key(frames[index]['possession']) is not None:
            index+=1;continue
        first=index
        while index<len(frames) and owner_key(frames[index]['possession']) is None:
            index+=1
        last=index-1
        if first==0 or index==len(frames):
            continue
        key=owner_key(frames[first-1]['possession'])
        if key!=owner_key(frames[index]['possession']) or last-first+1>maximum:
            continue
        # Anchor in measured wrist-supported assignments, not a chain of filled gaps.
        left=first-1
        while left>=0 and owner_key(frames[left]['possession'])==key and frames[left]['possession']['status']!='estimated_control':
            left-=1
        right=index
        while right<len(frames) and owner_key(frames[right]['possession'])==key and frames[right]['possession']['status']!='estimated_control':
            right+=1
        if (left<0 or right>=len(frames) or owner_key(frames[left]['possession'])!=key
                or owner_key(frames[right]['possession'])!=key or right-left-1>maximum):
            continue
        segment=frames[left]['ball_temporal'].get('segment_id')
        shot=frames[left]['calibration'].get('shot')
        supported=[];reason=None
        for j in range(left,right+1):
            frame=frames[j];ball=frame['ball_temporal']
            if (segment is None or ball.get('segment_id')!=segment or ball['state']=='none'
                    or frame['calibration'].get('shot')!=shot):
                reason='ball_segment_scene_or_measurement_gap';break
            row,observed=continuity_row(frames,j,key,fps,config)
            if row is None:
                reason='player_not_supported';break
            point=ball.get('measurement_xy') or [ball['x'],ball['y']]
            geometry=contact_geometry(point,row)
            if (geometry['distance_heights']>config.max_body_distance_heights
                    or abs(geometry['relative_xy'][0])>config.max_horizontal_heights
                    or not config.minimum_vertical_fraction<=geometry['vertical_fraction']<=config.maximum_vertical_fraction):
                reason='ball_left_dribble_envelope';break
            for candidate in frame['possession'].get('candidates',[]):
                if candidate['person_id']==key[0] and candidate['track_id']==key[2]:
                    continue
                score=candidate.get('contact_score',0.)
                if (score>=config.strong_rival_contact or
                        score>=config.repeated_rival_contact and
                        candidate.get('support_measurements',0)>=config.repeated_rival_measurements):
                    reason='rival_wrist_support';break
            if reason:break
            if first<=j<=last:
                supported.append((j,row,observed,geometry))
        decisions.append(dict(first=first,last=last,player_id=key[0],accepted=reason is None,reason=reason or 'same_owner_supported_endpoints',anchors=[left,right]))
        if reason:continue
        for j,row,observed,geometry in supported:
            control=copy.deepcopy(frames[left]['possession'])
            control.update(status='estimated_control_continuity',score=None,
                           reason='bounded_same_owner_gap',evidence_source='offline_same_owner_gap_bridge',
                           candidates=frames[j]['possession'].get('candidates',[]),
                           offline_future_frames_used=True,continuity_anchor_frames=[left,right],
                           continuity_duration_s=(right-left-1)/fps,current_player_box_observed=observed,
                           controller_display_label=row.get('display_label',str(key[0])),
                           inferred_controller_bbox_xyxy=None if observed else row['bbox_xyxy'],
                           body_distance_heights=geometry['distance_heights'],support_measurements=0,
                           minimap_xy_m=None,minimap_position_source=None)
            xy=row.get('court_xy_m')
            if observed and xy is not None and row.get('court_region')=='inside' and np.isfinite(xy).all():
                control.update(minimap_xy_m=list(xy),minimap_position_source='estimated_controller_floor_position')
            output[j]['possession']=control
    return output,dict(configuration=asdict(config),filled_frames=sum(f['possession']!=f['possession_raw'] for f in output),decisions=decisions)
