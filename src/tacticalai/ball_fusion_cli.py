"""CPU-only fusion of exported RF-DETR/Kalman balls and automatic player caches.

Run python -m tacticalai.ball_fusion_cli --help. No neural inference or Colab is
required. Controller assignment is repeated body proximity, not verified possession.
"""
import argparse
from collections import Counter
import copy
import csv
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from .ball import BallConfig, PossessionEstimator
from .ball_control import ControlConfig, TemporalControlEstimator
from .control_smoothing import ControlSmoothingConfig, smooth_control_gaps
from .ball_cli import load_source
from .calibration_cli import video_info
from .court import floor_lines
from .export import sha256, write_json
from .identity_presentation import text
from .silhouette_presentation import render

CLIPS = ('attack_clip', 'attack_clip2')
COURT = dict(length_m=28., width_m=15., key_depth_m=5.8, key_width_m=4.9)


def people_lineage(source):
    """Record the inherited tracker/ReID/GMC configuration and automatic identity stages."""
    lineage = []
    visited = set()
    while source.resolve() not in visited:
        visited.add(source.resolve())
        path = source/'experiment.json'
        if not path.is_file():
            break
        manifest = json.loads(path.read_text())
        lineage.append(dict(path=str(path.resolve()), sha256=sha256(path),
                            tracking_profile=manifest.get('tracking_profile'),
                            identity_stage=manifest.get('identity_stage'),
                            manual_labels_used=manifest.get('manual_labels_used'),
                            manual_identity_corrections=manifest.get('manual_identity_corrections')))
        parent = manifest.get('source')
        if not isinstance(parent, str):
            break
        source = Path(parent)
    return lineage


def load_ball(path, video, info):
    """Check coordinate space/time against the exported overlay and accepted candidates."""
    data = json.loads(path.read_text())
    exported = video_info(video)
    if any(exported[k] != info[k] for k in ('width', 'height', 'frames', 'fps')):
        raise ValueError('Ball export and local source have different dimensions/frame timing')
    if data['config'].get('tracker') != 'kalman_cv_gated_interpolation':
        raise ValueError('Expected the exported Kalman solution')
    if len(data['tracked']) != info['frames'] or len(data['detections']) != info['frames']:
        raise ValueError('Ball frame count differs')
    for index, point in enumerate(data['tracked']):
        if point['frame_idx'] != index or abs(point['timestamp_s']-index/info['fps']) > 1e-6:
            raise ValueError('Ball frame index/timestamp differs')
        if point['state'] not in ('none', 'detected', 'predicted', 'interpolated'):
            raise ValueError('Unknown ball state')
        if point['state'] != 'none':
            if not (math.isfinite(point['x']) and math.isfinite(point['y']) and
                    0 <= point['x'] < info['width'] and 0 <= point['y'] < info['height']):
                raise ValueError('Ball position outside the source coordinate space')
        if point['state'] == 'detected':
            candidate = data['detections'][index][point['accepted_candidate_index']]
            if (not np.allclose([candidate['x'], candidate['y']], point['measurement_xy'], rtol=0, atol=1e-6)
                    or abs(candidate['conf']-point['conf']) > 1e-6):
                raise ValueError('Accepted measurement does not match its raw candidate')
        elif point.get('measurement_xy') is not None or point.get('conf') is not None:
            raise ValueError('Inferred/unknown ball must not contain measured coordinates/confidence')
    return data


def controller_ball(point):
    """Only an accepted detector measurement may establish control, never interpolation."""
    return dict(status='observed' if point['state']=='detected' else 'unknown',
                center_image_xy=point.get('measurement_xy') if point['state']=='detected' else None,
                inferred_center_image_xy=[point['x'],point['y']] if point['state'] in ('predicted','interpolated') else None,
                track_id=point.get('segment_id'))


def eligible_people(rows):
    # Keep role-unknown/referee/staff observations unchanged, but do not assign them as players.
    return [r for r in rows if r.get('role')=='player' and r.get('display_on_video', True)
            and r.get('player_id') is not None and r.get('track_id') is not None
            and 'manual' not in str(r.get('classification_source', '')).lower()]


def load_pose(source,clip,info,ball_path,people_provenance):
    metadata=json.loads((source/clip/'metadata.json').read_text())
    path=source/clip/'poses.jsonl'
    if (metadata['status']!='complete' or metadata['video']!=info
            or metadata['ball_json_sha256']!=sha256(ball_path)
            or metadata['source_people']['sha256']!=people_provenance['sha256']
            or metadata['poses_sha256']!=sha256(path)):
        raise ValueError('Pose cache does not match the ball/player/source video')
    frames=[json.loads(line) for line in path.read_text().splitlines()]
    if len(frames)!=info['frames'] or any(f['frame_idx']!=i or abs(f['timestamp_s']-i/info['fps'])>1e-6
                                         for i,f in enumerate(frames)):
        raise ValueError('Pose cache timeline mismatch')
    return frames


def combine_frames(people, balls, fps, config=BallConfig(), association='legacy', control_config=ControlConfig(), poses=None):
    if len(people) != len(balls['tracked']):
        raise ValueError('Player and ball timelines differ')
    if association not in ('legacy','temporal'):
        raise ValueError('Unknown association method')
    estimator = (TemporalControlEstimator(fps,control_config) if association=='temporal'
                 else PossessionEstimator(fps, config))
    output = []
    previous_shot = None
    for index, source in enumerate(people):
        frame = copy.deepcopy(source)
        ball = copy.deepcopy(balls['tracked'][index])
        shot = frame['calibration'].get('shot')
        cut = index==0 or shot != previous_shot
        previous_shot = shot
        rows=eligible_people(frame['tracks'])
        if poses is not None:
            by_key={(p['player_id'],p.get('identity_segment_id'),p['track_id']):p for p in poses[index]['poses']}
            rows=[dict(row,control_pose=by_key.get((row['player_id'],row.get('identity_segment_id'),row['track_id'])))
                  for row in rows]
            frame['player_pose_evidence']=poses[index]
        possession = estimator.update(controller_ball(ball), rows, cut=cut)
        frame.update(ball_temporal=ball, ball_raw_candidates=balls['detections'][index],
                     possession=possession, ball_court_xy_m=None,
                     ball_court_position_reason='image_space_ball_may_be_airborne')
        output.append(frame)
    return output


def add_ball_trails(frames,info,seconds=.8):
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Trail duration must be positive and finite')
    limit=max(1,round(seconds*info['fps']))
    maximum_step=.06*math.hypot(info['width'],info['height'])
    trail=[]
    previous_segment=previous_shot=None
    for frame in frames:
        ball=frame['ball_temporal']
        segment=ball.get('segment_id')
        shot=frame['calibration'].get('shot')
        if ball['state']=='none':
            trail=[]
        else:
            point=[ball['x'],ball['y']]
            if (segment!=previous_segment or shot!=previous_shot or
                    trail and math.dist(trail[-1],point)>maximum_step):
                trail=[]
            trail.append(point)
            trail=trail[-limit:]
        frame['ball_display_trail_xy']=[list(point) for point in trail]
        previous_segment,previous_shot=segment,shot


def render_combined(image, frame, lines=None, player_presentation='clean'):
    # Clean presentation only; raw/inferred distinctions remain in machine-readable data.
    panel=None
    if player_presentation=='silhouettes':
        view=dict(frame,tracks=[dict(row,draw_detection_box_fallback=False) for row in frame['tracks']])
        out=render(image,view,lines or floor_lines(COURT),color_mode='team')
        panel=out[10:200,10:314].copy()
        xy=frame['possession'].get('minimap_xy_m')
        if xy is not None:
            point=tuple(np.rint(np.asarray(xy)*9.6+np.array([17,29])).astype(int))
            cv2.circle(panel,point,8,(0,0,0),4,cv2.LINE_AA)
            cv2.circle(panel,point,7,(80,255,140),2,cv2.LINE_AA)
    else:
        out=cv2.resize(image,(1280,720))
    sx,sy=1280/image.shape[1],720/image.shape[0]
    ball,control=frame['ball_temporal'],frame['possession']
    owner=control['controller_person_id']
    if owner is not None:
        row=next((r for r in frame['tracks'] if r.get('player_id')==owner and
                  r.get('track_id')==control['controller_track_id']),None)
        box=(row['bbox_xyxy'] if row is not None and control.get('current_player_box_observed',True)
             else control.get('inferred_controller_bbox_xyxy'))
        if box is not None:
            x1,y1,x2,y2=np.rint(np.asarray(box)*[sx,sy,sx,sy]).astype(int)
            cv2.rectangle(out,(x1,y1),(x2,y2),(0,0,0),5,cv2.LINE_AA)
            cv2.rectangle(out,(x1,y1),(x2,y2),(80,255,140),2,cv2.LINE_AA)
    trail=frame.get('ball_display_trail_xy',[])
    if len(trail)>1:
        points=np.rint(np.asarray(trail)*[sx,sy]).astype(np.int32)
        cv2.polylines(out,[points],False,(0,0,0),4,cv2.LINE_AA)
        cv2.polylines(out,[points],False,(0,165,255),2,cv2.LINE_AA)
    if ball['state']!='none':
        point=tuple(np.rint(np.array([ball['x'],ball['y']])*[sx,sy]).astype(int))
        cv2.circle(out,point,10,(0,0,0),4,cv2.LINE_AA)
        cv2.circle(out,point,9,(0,165,255),2,cv2.LINE_AA)
    if panel is not None:
        out[10:200,10:314]=panel
    return out


def encode(video, frames, folder, info,player_presentation='clean'):
    import imageio_ffmpeg
    cap = cv2.VideoCapture(str(video))
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y',
        '-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(info['fps']),
        '-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(folder/'annotated.mp4')],stdin=subprocess.PIPE)
    snapshots = {0,info['frames']//3,2*info['frames']//3,info['frames']-1}
    snapshots.update(f['frame_idx'] for f in frames if f['possession']['status']=='estimated_control'
                     and (f['frame_idx']==0 or frames[f['frame_idx']-1]['possession']['status']!='estimated_control'))
    lines = floor_lines(COURT)
    try:
        for frame in frames:
            ok,image = cap.read()
            if not ok: raise ValueError('Source video ended early')
            out = render_combined(image,frame,lines,player_presentation)
            proc.stdin.write(out.tobytes())
            if frame['frame_idx'] in snapshots:
                cv2.imwrite(str(folder/f"frame-{frame['frame_idx']:04d}.jpg"),out)
        if cap.read()[0]: raise ValueError('Source video has extra frames')
        proc.stdin.close()
        if proc.wait()!=0: raise RuntimeError('Video encoding failed')
    finally:
        cap.release()
        if proc.poll() is None:
            proc.kill();proc.wait()
    check = cv2.VideoCapture(str(folder/'annotated.mp4'))
    decoded = 0
    while check.read()[0]: decoded += 1
    check.release()
    if decoded != info['frames']: raise ValueError('Encoded frame count differs')
    return decoded


def flat_frame(frame):
    ball, control = frame['ball_temporal'], frame['possession']
    measurement = ball.get('measurement_xy') or [None,None]
    return dict(frame_idx=frame['frame_idx'],timestamp_s=frame['timestamp_s'],
                ball_state=ball['state'],ball_x=ball['x'],ball_y=ball['y'],
                measured_x=measurement[0],measured_y=measurement[1],
                ball_confidence=ball['conf'],ball_segment=ball['segment_id'],
                ball_position_source=ball['position_source'],ball_court_xy_m=None,
                control_status=control['status'],controller_person_id=control['controller_person_id'],
                controller_track_id=control['controller_track_id'],control_score=control['score'],
                control_reason=control['reason'],control_evidence_source=control.get('evidence_source'),
                control_support_measurements=control.get('support_measurements'),
                raw_controller_person_id=frame.get('possession_raw',control)['controller_person_id'],
                owner_box_observed=control.get('current_player_box_observed',True),
                controller_floor_xy_m=control['minimap_xy_m'])


def run(args):
    config = BallConfig(possession_confirmation_s=args.confirmation_s,
                        possession_min_score=args.min_control_score,
                        possession_margin=args.control_margin)
    control_config = ControlConfig(support_window_s=args.support_window_s,
                                   minimum_contact_score=args.min_contact_score,
                                   continuity_s=args.continuity_s,ambiguity_margin=args.control_margin,
                                   pose_required=args.pose_results is not None)
    smoothing_config=ControlSmoothingConfig(max_gap_s=args.dribble_gap_s,
                                            max_missing_player_s=args.missing_player_gap_s)
    pose_manifest=json.loads((args.pose_results/'experiment.json').read_text()) if args.pose_results else None
    if pose_manifest is not None and pose_manifest['status']!='complete':
        raise ValueError('Pose cache must be complete')
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    report = dict(status='running',automatic_only=True,manual_labels_used=False,
                  neural_inference_rerun=False,offline_processing=True,device='cpu',
                  ball_source=str(args.ball_results.resolve()),people_source=str(args.people_results.resolve()),
                  people_lineage=people_lineage(args.people_results),
                  possession_configuration=asdict(config),association=args.association,
                  temporal_control_configuration=asdict(control_config),clips={},accuracy_metrics=None,
                  pose_source=str(args.pose_results.resolve()) if args.pose_results else None,
                  pose_provenance=pose_manifest,
                  control_smoothing=asdict(smoothing_config) if args.smooth_control else None,
                  presentation=dict(style='assigned_box_and_ball_trail',player_presentation=args.player_presentation,
                                    minimap_marker='estimated_controller_floor_position_only',ball_trail_s=args.ball_trail_s,
                                    uniform_inferred_and_measured_style=True,raw_candidates_visible=False),
                  versions={p:importlib.metadata.version(p) for p in ('numpy','opencv-python','scipy','imageio-ffmpeg')},
                  limitations=[
                      'Controller is repeated body proximity, not verified hand contact or tactical possession.',
                      'Only accepted ball measurements establish/change control. Brief inferred continuity is labelled separately.',
                      'Silhouette distance, relative-motion limits and temporal windows are development heuristics, not hand-contact evidence.',
                      'Incorrect accepted ball/player detections, roles and identity links can produce wrong control.',
                      'No airborne ball homography is applied; map ring is the estimated controller floor position.',
                      'Historical Colab exports have no source-video hash; coordinate/frame alignment is checked.',
                      'Player identities, automatic classes, segmentation and calibration are inherited unchanged.',
                      'Player tracker/ReID/GMC are inherited from the automatic source; ball GMC/ReID are disabled.',
                      'These development clips and thresholds are not held-out accuracy evaluation.'])
    write_json(args.output/'experiment.json',report)
    try:
        for clip in CLIPS:
            tick=time.perf_counter()
            target=args.output/clip;target.mkdir()
            video=args.videos/(clip+'.mp4');info=video_info(video)
            people,provenance=load_source(args.people_results,clip,info)
            ball_path=args.ball_results/f'RF-DETR-tiles-Kalman_{clip}.json'
            balls=load_ball(ball_path,args.ball_results/f'RF-DETR-tiles-Kalman_{clip}.mp4',info)
            poses=load_pose(args.pose_results,clip,info,ball_path,provenance) if args.pose_results else None
            frames=combine_frames(people,balls,info['fps'],config,args.association,control_config,poses)
            if args.smooth_control:
                frames,smoothing=smooth_control_gaps(frames,info['fps'],smoothing_config)
                write_json(target/'control_smoothing.json',smoothing)
            add_ball_trails(frames,info,args.ball_trail_s)
            assert all(all(frame[key]==value for key,value in original.items())
                       for original,frame in zip(people,frames)), 'Original player/camera data changed'
            with (target/'frames.jsonl').open('w',encoding='utf8') as stream:
                for frame in frames:stream.write(json.dumps(frame,allow_nan=False)+'\n')
            flat=[flat_frame(f) for f in frames]
            with (target/'ball_possession.csv').open('w',newline='',encoding='utf8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
            decoded=encode(video,frames,target,info,args.player_presentation)
            counts=dict(ball_states=dict(Counter(f['ball_temporal']['state'] for f in frames)),
                        control_states=dict(Counter(f['possession']['status'] for f in frames)),
                        unknown_reasons=dict(Counter(f['possession']['reason'] for f in frames
                                                     if f['possession']['status']=='unknown')),
                        controller_frames=dict(Counter(f['possession']['controller_person_id'] for f in frames
                                                       if f['possession']['controller_person_id'] is not None)))
            meta=dict(status='complete',video=info,source=provenance,
                      ball_source=dict(path=str(ball_path.resolve()),sha256=sha256(ball_path),
                                       video_hash_verified=False,config=balls['config'],historical_stats=balls['stats']),
                      original_player_observations_preserved=True,annotated_frames_decoded=decoded,
                      counts=counts,processing_s=time.perf_counter()-tick,accuracy_metrics=None,
                      artifacts_sha256={p.name:sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json',meta);report['clips'][clip]=meta
            write_json(args.output/'experiment.json',report)
            print(clip,counts,flush=True)
        source=args.output/'run_source';source.mkdir()
        for name in ('ball_fusion_cli.py','ball.py','ball_control.py','control_smoothing.py','pose_cache_cli.py','silhouette_presentation.py'):
            shutil.copyfile(Path(__file__).with_name(name),source/name)
        report.update(status='complete',processing_s=time.perf_counter()-started,
                      source_sha256={p.name:sha256(p) for p in source.iterdir()})
        write_json(args.output/'experiment.json',report)
        (args.output/'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Ball and handler — both clips</title>'
            '<body style="margin:20px;background:#111;color:#eee;font:16px Arial">'
            '<main style="display:flex;flex-wrap:wrap;gap:20px">'+
            ''.join(f'<section style="flex:1;min-width:380px"><h2>Clip {i}</h2>'
                    f'<video controls loop preload="auto" style="width:100%" src="{clip}/annotated.mp4"></video></section>'
                    for i,clip in enumerate(CLIPS,1))+
            '</main><p style="color:#aaa;font-size:13px">Offline estimates. Filled assignments and inferred ball positions '
            'share the same visual style; measurement/inference flags are preserved in JSON/CSV. Accuracy remains unverified.</p>'
            '<p style="color:#aaa;font-size:13px">When shown, the mini-map ring marks the assigned player\'s floor position, not the airborne ball.</p>'
            '</body>',encoding='utf8')
    except BaseException as error:
        report.update(status='failed',error=repr(error));write_json(args.output/'experiment.json',report);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ball-results',type=Path,required=True)
    parser.add_argument('--people-results',type=Path,required=True)
    parser.add_argument('--videos',type=Path,default=Path('data/raw'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirmation-s',type=float,default=.12)
    parser.add_argument('--min-control-score',type=float,default=.55)
    parser.add_argument('--control-margin',type=float,default=.12)
    parser.add_argument('--association',choices=('legacy','temporal'),default='temporal')
    parser.add_argument('--support-window-s',type=float,default=.24)
    parser.add_argument('--continuity-s',type=float,default=.16)
    parser.add_argument('--min-contact-score',type=float,default=.35)
    parser.add_argument('--pose-results',type=Path)
    parser.add_argument('--ball-trail-s',type=float,default=.8)
    parser.add_argument('--player-presentation',choices=('clean','silhouettes'),default='silhouettes')
    parser.add_argument('--smooth-control',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--dribble-gap-s',type=float,default=.60)
    parser.add_argument('--missing-player-gap-s',type=float,default=.12)
    run(parser.parse_args())


if __name__=='__main__':main()
