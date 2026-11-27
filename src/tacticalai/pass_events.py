"""Conservative offline pass inference from controller-change endpoints.

This module does not reconstruct an airborne ball in court coordinates.  It
looks for a stable controller, a bounded interval without a controller, and a
stable different controller on the same automatic team.  The displayed path
is explicitly an interpolation between the two player floor positions.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PassConfig:
    minimum_control_s: float = .12
    minimum_flight_s: float = .12
    maximum_flight_s: float = 2.0
    minimum_court_distance_m: float = .9
    minimum_measured_ball_fraction: float = .15
    minimum_visible_ball_fraction: float = .50
    endpoint_hold_s: float = .35
    maximum_endpoint_interpolation_s: float = .50

    def __post_init__(self):
        values = asdict(self).values()
        if any(not math.isfinite(value) for value in values):
            raise ValueError('Pass settings must be finite')
        if min(values) < 0 or self.minimum_control_s <= 0 or self.maximum_flight_s <= 0:
            raise ValueError('Pass durations and distance must be non-negative')
        if self.minimum_flight_s > self.maximum_flight_s:
            raise ValueError('Minimum pass duration exceeds maximum')
        if not 0 <= self.minimum_measured_ball_fraction <= 1:
            raise ValueError('Invalid measured-ball fraction')
        if not 0 <= self.minimum_visible_ball_fraction <= 1:
            raise ValueError('Invalid visible-ball fraction')


def _owner(frame):
    return frame['possession'].get('controller_person_id')


def _runs(frames):
    result=[]
    first=0
    while first < len(frames):
        owner=_owner(frames[first])
        last=first
        while last+1 < len(frames) and _owner(frames[last+1]) == owner:
            last += 1
        result.append((first,last,owner))
        first=last+1
    return result


def _owner_team(frames, run):
    first,last,owner=run
    teams=[]
    for frame in frames[first:last+1]:
        teams.extend(row.get('team_id') for row in frame['tracks']
                     if row.get('player_id') == owner and row.get('role') == 'player'
                     and row.get('team_id') is not None)
    return Counter(teams).most_common(1)[0][0] if teams else None


def _endpoint(frames, run, reverse=False):
    first,last,_=run
    indexes=range(last,first-1,-1) if reverse else range(first,last+1)
    for index in indexes:
        point=frames[index]['possession'].get('minimap_xy_m')
        if (point is not None and len(point) == 2 and
                all(math.isfinite(float(value)) for value in point)):
            return index,[float(point[0]),float(point[1])],dict(
                source='observed_controller_minimap',anchor_frames=[index],
                offline_future_frames_used=False)
    return None,None,None


def _row_position(frame,owner,identity_segment=None):
    for row in frame['tracks']:
        if (row.get('player_id') != owner or row.get('role') != 'player' or
                not row.get('display_on_video',True)):
            continue
        if identity_segment is not None and row.get('identity_segment_id') != identity_segment:
            continue
        point=row.get('minimap_xy_m')
        if (point is not None and len(point)==2 and
                all(math.isfinite(float(value)) for value in point)):
            return [float(point[0]),float(point[1])],row.get('identity_segment_id')
    return None,None


def _interpolated_endpoint(frames,anchor,owner,fps,maximum_s):
    """Interpolate only between nearby valid positions of the same identity."""
    shot=frames[anchor].get('calibration',{}).get('shot')
    identity=frames[anchor]['possession'].get('controller_identity_segment_id')
    if identity is None:
        row=next((row for row in frames[anchor]['tracks']
                  if row.get('player_id')==owner),None)
        identity=None if row is None else row.get('identity_segment_id')
    limit=max(1,math.floor(maximum_s*fps+1e-9))
    anchors=[]
    for direction in (-1,1):
        found=None
        for offset in range(1,limit+1):
            index=anchor+direction*offset
            if not 0<=index<len(frames):
                break
            if frames[index].get('calibration',{}).get('shot') != shot:
                break
            point,row_identity=_row_position(frames[index],owner,identity)
            if point is not None and (identity is None or row_identity==identity):
                found=(index,point);break
        anchors.append(found)
    left,right=anchors
    if left is None or right is None or right[0]-left[0] > limit:
        return None,None
    weight=(anchor-left[0])/(right[0]-left[0])
    point=[(1-weight)*a+weight*b for a,b in zip(left[1],right[1])]
    quality=dict(source='bounded_same_identity_interpolation',
                 anchor_frames=[left[0],right[0]],identity_segment_id=identity,
                 interpolation_span_s=(right[0]-left[0])/fps,
                 offline_future_frames_used=True)
    return point,quality


def infer_pass_events(frames, fps, config=PassConfig()):
    """Annotate frames with endpoint-derived same-team pass visualizations.

    Returns an audit dictionary. Rejections remain available for development
    review and no rejected candidate is rendered.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid FPS')
    for frame in frames:
        frame['pass_event']=None
    minimum_control=max(1,math.ceil(config.minimum_control_s*fps-1e-9))
    minimum_flight=max(1,math.ceil(config.minimum_flight_s*fps-1e-9))
    maximum_flight=max(minimum_flight,math.floor(config.maximum_flight_s*fps+1e-9))
    hold=math.floor(config.endpoint_hold_s*fps+1e-9)
    decisions=[];events=[]
    runs=_runs(frames)
    for source,gap,target in zip(runs,runs[1:],runs[2:]):
        if source[2] is None or gap[2] is not None or target[2] is None:
            continue
        if source[2] == target[2]:
            continue
        decision=dict(source_player_id=source[2],target_player_id=target[2],
                      release_frame=gap[0],receive_frame=target[0],accepted=False)
        source_count=source[1]-source[0]+1;target_count=target[1]-target[0]+1
        flight_count=gap[1]-gap[0]+1
        violations=[]
        if source_count < minimum_control or target_count < minimum_control:
            violations.append('unstable_control_endpoint')
        if not minimum_flight <= flight_count <= maximum_flight:
            violations.append('flight_duration_out_of_range')
        shots={frames[index].get('calibration',{}).get('shot')
               for index in range(source[1],target[0]+1)}
        if len(shots) != 1 or None in shots:
            violations.append('scene_changed_or_unknown')
        source_team=_owner_team(frames,source);target_team=_owner_team(frames,target)
        decision.update(source_team=source_team,target_team=target_team)
        if source_team is None or target_team is None:
            violations.append('unknown_team')
        elif source_team != target_team:
            violations.append('different_team')
        source_position_frame,start,source_position_quality=_endpoint(frames,source,reverse=True)
        target_position_frame,end,target_position_quality=_endpoint(frames,target)
        if start is None:
            start,source_position_quality=_interpolated_endpoint(
                frames,source[1],source[2],fps,config.maximum_endpoint_interpolation_s)
            source_position_frame=source[1] if start is not None else None
        if end is None:
            end,target_position_quality=_interpolated_endpoint(
                frames,target[0],target[2],fps,config.maximum_endpoint_interpolation_s)
            target_position_frame=target[0] if end is not None else None
        decision.update(source_position_frame=source_position_frame,
                        target_position_frame=target_position_frame,
                        source_position_quality=source_position_quality,
                        target_position_quality=target_position_quality)
        distance=math.dist(start,end) if start is not None and end is not None else None
        decision['court_distance_m']=distance
        if distance is None or distance < config.minimum_court_distance_m:
            violations.append('missing_or_too_close_court_endpoints')
        ball_states=[frames[index]['ball_temporal'].get('state','none')
                     for index in range(gap[0],target[0]+1)]
        measured=sum(state == 'detected' for state in ball_states)/len(ball_states)
        visible=sum(state != 'none' for state in ball_states)/len(ball_states)
        decision.update(measured_ball_fraction=measured,visible_ball_fraction=visible)
        if (measured < config.minimum_measured_ball_fraction or
                visible < config.minimum_visible_ball_fraction):
            violations.append('insufficient_ball_evidence')
        # User-requested recall tier: include every candidate with an unstable
        # endpoint, plus candidates where exactly one/both automatic team labels
        # are unavailable. Known opposing-team changes remain rejected.
        tentative=('unstable_control_endpoint' in violations or 'unknown_team' in violations)
        drawable=start is not None and end is not None
        if violations and not (tentative and drawable):
            decision.update(reason=violations[0],reasons=violations)
            decisions.append(decision);continue
        interpolated_endpoint=any(quality is not None and
            quality.get('source')=='bounded_same_identity_interpolation'
            for quality in (source_position_quality,target_position_quality))
        event_warnings=list(violations)
        if interpolated_endpoint:
            event_warnings.append('interpolated_endpoint')
        confidence='tentative' if event_warnings else 'conservative'
        event_id=f"pass-{len(events)+1:04d}"
        display_end=min(len(frames)-1,target[0]+hold)
        event=dict(event_id=event_id,source_player_id=source[2],target_player_id=target[2],
                   team_id=source_team or target_team,release_frame=gap[0],receive_frame=target[0],
                   display_end_frame=display_end,start_xy_m=start,end_xy_m=end,
                   source_position_frame=source_position_frame,
                   target_position_frame=target_position_frame,
                   source_position_quality=source_position_quality,
                   target_position_quality=target_position_quality,
                   court_distance_m=distance,measured_ball_fraction=measured,
                   visible_ball_fraction=visible,
                   method='offline_controller_change_endpoint_interpolation',
                   confidence=confidence,relaxed_reasons=event_warnings,
                   true_airborne_trajectory_known=False)
        for index in range(gap[0],display_end+1):
            in_flight=index <= target[0]
            progress=(min(index,target[0])-gap[0])/max(1,target[0]-gap[0])
            frames[index]['pass_event']=dict(event,progress=float(progress),
                phase='inferred_in_flight' if in_flight else 'confirmed_endpoint_hold')
        events.append(event)
        decision.update(accepted=True,confidence=confidence,reasons=event_warnings,
                        reason=('user_requested_relaxed_candidate' if event_warnings else
                                'same_team_stable_controller_change'))
        decisions.append(decision)
    return dict(configuration=asdict(config),events=events,decisions=decisions,
                accepted_events=len(events),airborne_ball_depth_estimated=False,
                conservative_events=sum(e['confidence']=='conservative' for e in events),
                tentative_events=sum(e['confidence']=='tentative' for e in events),
                tentative_policy=['unstable_control_endpoint','unknown_team','interpolated_endpoint'],
                offline_future_frames_used=True)
