"""Combine immutable tracking identities and cached court cameras for review.

No detector, tracker, calibration network or ReID inference is run here. Manual
role annotations affect display only. Identity review labels never enter fusion.
"""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from .calibration import project_track
from .challenge_presentation import COURT, CLIPS
from .court import floor_lines
from .export import sha256, write_json
from .minimap import apply_roles
from .temporal_projection import smooth_players
from .temporal_reid_cli import analyze


def person_label(identity):
    if identity is None:
        return '?'
    identity = str(identity)
    return 'P' + (identity[:-2] if identity.endswith('.0') else identity)


def person_color(identity):
    if identity is None:
        return (170, 170, 170)
    hue = int.from_bytes(hashlib.sha256(str(identity).encode()).digest()[:2], 'big') % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 175, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def fuse_frames(identity_frames, cameras, roles, court=COURT, image_size=(640,360), fps=25):
    if len(identity_frames) != len(cameras):
        raise ValueError('Identity/camera frame count mismatch')
    result, cuts, previous_shot = [], [], None
    for source, camera in zip(identity_frames, cameras):
        if source['frame_idx'] != camera['frame_idx'] or abs(source['timestamp_s']-camera['timestamp_s']) > 1e-6:
            raise ValueError('Identity/camera timestamp mismatch')
        quality = camera['temporal_quality']
        matrix = camera['refined_image_to_court_h']
        if matrix is not None:
            matrix = np.asarray(matrix)
        rows = []
        for row in apply_roles(source['tracks'], roles, source['frame_idx']):
            projected = project_track(row, matrix, quality, court, image_size)
            projected['court_xy_camera_smoothed_m'] = projected['court_xy_m']
            if projected['court_xy_m'] is not None:
                projected['position_source'] = ('estimated_calibration_floor_projection' if quality['method']=='kalman_rts_camera_gap_estimate'
                                                else 'temporally_smoothed_floor_projection')
            rows.append(projected)
        shot = quality.get('shot')
        cuts.append(not result or shot != previous_shot)
        previous_shot = shot
        result.append(dict(source, tracks=rows, calibration=quality, image_to_court_h=matrix.tolist() if matrix is not None else None))
    smooth_players(result, fps, np.asarray(cuts), identity_field='identity_segment_id')
    for frame in result:
        for row in frame['tracks']:
            row['display_label'] = person_label(row['player_id'])
            color = person_color(row['player_id'])
            row['display_color_hex'] = '#' + ''.join(f'{c:02x}' for c in color[::-1])
            row['shown_on_minimap'] = row['role']=='player' and row['court_xy_m'] is not None and row['court_region']=='inside'
    return result


def text(image, value, at, scale=.55, color=(240,245,250)):
    cv2.putText(image, value, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (12,16,20), 4, cv2.LINE_AA)
    cv2.putText(image, value, at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_frame(image, frame, lines, size=(1600,720)):
    """Unobscured 1280x720 footage plus a separate 320px court/status panel."""
    out = np.full((size[1],size[0],3), (29,23,18), np.uint8)
    out[:,:1280] = cv2.resize(image,(1280,720))
    sx, sy = 1280/image.shape[1], 720/image.shape[0]
    for row in frame['tracks']:
        if row['role'] != 'player':
            continue
        x1,y1,x2,y2 = row['bbox_xyxy']
        color = person_color(row['player_id'])
        a,b = (round(x1*sx),round(y1*sy)),(round(x2*sx),round(y2*sy))
        cv2.rectangle(out,a,b,color,2,cv2.LINE_AA)
        text(out,row['display_label'],(a[0],max(20,a[1]-6)),.62,color)
    origin, scale = np.array([1300.,90.]), 10.
    text(out,'COURT MAP',(1300,35),.6)
    text(out,'Approximate floor positions',(1300,59),.4)
    for line in lines:
        pixels=np.rint(line*scale+origin).astype(np.int32)
        cv2.polylines(out,[pixels],False,(170,185,180),1,cv2.LINE_AA)
    shown = [r for r in frame['tracks'] if r['shown_on_minimap']]
    for row in shown:
        x,y = np.rint(np.asarray(row['court_xy_m'])*scale+origin).astype(int)
        color = person_color(row['player_id'])
        cv2.circle(out,(x,y),5,color,-1,cv2.LINE_AA)
        label_width=cv2.getTextSize(row['display_label'],cv2.FONT_HERSHEY_SIMPLEX,.38,1)[0][0]
        label_x=max(1285,min(x+5,size[0]-label_width-5))
        text(out,row['display_label'],(label_x,max(80,y-6)),.38,color)
    quality = frame['calibration']['method']
    text(out,'Camera: '+('estimated' if 'gap_estimate' in quality else 'smoothed' if quality!='unavailable' else 'unavailable'),(1300,280),.43)
    text(out,f"{len(shown)} current player dots",(1300,310),.48)
    text(out,f"{frame['timestamp_s']:.2f}s  |  frame {frame['frame_idx']}",(1300,340),.45)
    text(out,'Person / source tracker ID',(1300,385),.46)
    players = [r for r in frame['tracks'] if r['role']=='player']
    for i,row in enumerate(players[:13]):
        text(out,f"{row['display_label']}  /  raw {row['track_id']}",(1300,414+21*i),.43,person_color(row['player_id']))
    text(out,'? = unknown identity',(1300,697),.4)
    return out


def id_independence(cache, clip, original):
    """Actual feature-cache audit: adding 5000 to IDs must not change decisions."""
    observations=json.loads((cache/clip/'observations.json').read_text())
    with np.load(cache/clip/'features.npz',allow_pickle=False) as values:
        features=values['features']
    renamed_frames=copy.deepcopy(original)
    for frame in renamed_frames:
        for row in frame['tracks']:
            row['track_id']+=5000
    renamed_obs=[dict(o,track_id=o['track_id']+5000) for o in observations]
    prediction=analyze(renamed_frames,renamed_obs,features)
    def undo(s):
        raw,phase=s.split('.')
        return f'{int(raw)-5000}.{phase}'
    return dict(
        events=[dict(track_id=e['track_id']-5000,last_old_frame=e['last_old_frame'],first_new_frame=e['first_new_frame']) for e in prediction['events']],
        links=sorted((undo(c['before']),undo(c['after'])) for c in prediction['candidates'] if c['accepted']),
        all_raw_ids_offset=5000, appearance_features_unchanged=True)


def run(identity_root, camera_root, features_root, output):
    import imageio_ffmpeg
    started=time.perf_counter()
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',clips={},accuracy_metrics=None,detector_tracker_reid_calibration_rerun=False,
                  identity_logic='Generic temporal appearance consistency + tracking corroboration + segment gallery assignment',
                  role_filter='Existing manually reviewed role intervals, display only',
                  court=COURT,labels_used_for_identity_inference=False,
                  generalization='Thresholds developed on these clips; no independent identity accuracy benchmark')
    write_json(output/'experiment.json',manifest)
    snapshots=output/'run_source';snapshots.mkdir()
    for name in ('identity_presentation.py','identity_presentation.html','temporal_projection.py','calibration.py','court.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    manifest['source_sha256']={p.name:sha256(p) for p in snapshots.iterdir()}
    manifest['versions']={p:importlib.metadata.version(p) for p in ('numpy','scipy','opencv-python','imageio-ffmpeg')}
    lines=floor_lines(COURT)
    for clip in CLIPS:
        source=identity_root/clip
        identity_meta=json.loads((source/'metadata.json').read_text())
        camera_folder=camera_root/clip
        camera_meta=json.loads((camera_folder/'metadata.json').read_text())
        if identity_meta['status']!='complete' or camera_meta['status']!='complete':
            raise ValueError('Incomplete source experiment')
        raw_path=Path(identity_meta['source']['source']['config']['run'])/'frames.jsonl'
        expected=identity_meta['source_frames_sha256']
        if sha256(raw_path)!=expected or camera_meta['tracking_frames_sha256']!=expected:
            raise ValueError('Identity and camera refer to different tracking runs')
        video=Path(identity_meta['source']['source']['video'])
        if sha256(video)!=camera_meta['video']['sha256'] or sha256(video)!=identity_meta['source']['video_sha256']:
            raise ValueError('Video mismatch')
        for name,expected_hash in identity_meta['artifacts_sha256'].items():
            if sha256(source/name)!=expected_hash:
                raise ValueError('Identity artifact changed')
        if sha256(camera_folder/'calibration.jsonl')!=camera_meta['calibration_sha256']:
            raise ValueError('Camera cache changed')
        roles=json.loads((camera_folder/'roles.json').read_text())
        if roles['tracking_frames_sha256']!=expected or sha256(camera_folder/'roles.json')!=camera_meta['roles_sha256']:
            raise ValueError('Role annotations refer to a different source')
        identity_frames=[json.loads(l) for l in (source/'frames.jsonl').read_text().splitlines()]
        originals=[json.loads(l) for l in raw_path.read_text().splitlines()]
        audit=id_independence(features_root,clip,originals)
        events=json.loads((source/'events.json').read_text())
        candidates=json.loads((source/'candidates.json').read_text())
        audit['passed']=audit['events']==[dict(track_id=e['track_id'],last_old_frame=e['last_old_frame'],first_new_frame=e['first_new_frame']) for e in events] and audit['links']==sorted((c['before'],c['after']) for c in candidates if c['accepted'])
        if not audit['passed']:
            raise ValueError('ID-renaming independence check failed')
        cameras=[json.loads(l) for l in (camera_folder/'calibration.jsonl').read_text().splitlines()]
        info=camera_meta['video']
        frames=fuse_frames(identity_frames,cameras,roles,image_size=(info['width'],info['height']),fps=info['fps'])
        # Projection changes position fields only; original boxes/IDs and identity
        # decisions must survive exactly, as must non-player observations.
        for before,after in zip(identity_frames,frames):
            assert len(before['tracks'])==len(after['tracks'])
            for a,b in zip(before['tracks'],after['tracks']):
                for key in ('track_id','bbox_xyxy','det_confidence','player_id','identity_segment_id','identity_status'):
                    assert a[key]==b[key]
        target=output/clip;target.mkdir()
        write_json(target/'id-independence-check.json',audit)
        for name in ('roles.json','calibration.jsonl'):
            shutil.copyfile(camera_folder/name,target/name)
        shutil.copyfile(source/'clean.mp4',target/'clean.mp4')
        with (target/'frames.jsonl').open('w',encoding='utf8') as stream, (target/'positions.csv').open('w',newline='',encoding='utf8') as csv_stream:
            columns=['frame_idx','timestamp_s','track_id','identity_segment_id','player_id','identity_status','role','court_x_m','court_y_m','position_source','shown_on_minimap']
            table=csv.DictWriter(csv_stream,fieldnames=columns);table.writeheader()
            for frame in frames:
                stream.write(json.dumps(frame,allow_nan=False)+'\n')
                for r in frame['tracks']:
                    record={k:r.get(k) for k in columns if k not in ('court_x_m','court_y_m')}
                    record.update(frame_idx=frame['frame_idx'],timestamp_s=frame['timestamp_s'],court_x_m=r['court_xy_m'][0] if r['court_xy_m'] is not None else None,court_y_m=r['court_xy_m'][1] if r['court_xy_m'] is not None else None)
                    table.writerow(record)
        cap=cv2.VideoCapture(str(video))
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1600x720','-r',str(info['fps']),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target/'tracking-minimap.mp4')]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for frame in frames:
                ok,image=cap.read()
                if not ok:
                    raise ValueError('Source decode incomplete')
                rendered=render_frame(image,frame,lines)
                process.stdin.write(rendered.tobytes())
                if frame['frame_idx'] in (0,55,131):
                    cv2.imwrite(str(target/f"frame-{frame['frame_idx']:04d}.jpg"),rendered)
            process.stdin.close()
            if process.wait()!=0:
                raise ValueError('Video encode failed')
        finally:
            cap.release()
            if process.poll() is None:
                process.kill();process.wait()
        cap=cv2.VideoCapture(str(target/'tracking-minimap.mp4'));decoded=0
        while cap.read()[0]:
            decoded+=1
        cap.release()
        if decoded!=len(frames):
            raise ValueError('Rendered frame count mismatch')
        base_review=json.loads((source/'review-data.json').read_text())
        stats=dict(raw_ids=base_review['summary']['raw_ids'],person_hypotheses=base_review['summary']['player_ids'],
                   unique_player_hypotheses=len({r['player_id'] for f in frames for r in f['tracks'] if r['role']=='player' and r['player_id'] is not None}),
                   unknown_identity_rows=sum(r['player_id'] is None for f in frames for r in f['tracks']),
                   source_rows=sum(len(f['tracks']) for f in frames),
                   player_rows=sum(r['role']=='player' for f in frames for r in f['tracks']),
                   map_rows=sum(r['shown_on_minimap'] for f in frames for r in f['tracks']),
                   estimated_camera_frames=sum('gap_estimate' in f['calibration']['method'] for f in frames),rendered_frames=decoded)
        write_json(target/'review-data.json',dict(clip=clip,video=info,frames=frames,summary=stats,court_lines=[l.tolist() for l in lines],
                    identity_events=events,development_review=base_review.get('development_review')))
        write_json(target/'metadata.json',dict(status='complete',summary=stats,identity_metadata=identity_meta,
                   camera_metadata=camera_meta,source_fields_preserved=['track_id','bbox_xyxy','det_confidence'],
                   player_smoothing_key='raw ID + role + identity segment; reset on gaps/cuts/unknown identity',
                   inputs_sha256=dict(identity_frames=sha256(source/'frames.jsonl'),camera=sha256(camera_folder/'calibration.jsonl'),roles=sha256(camera_folder/'roles.json')),
                   artifacts_sha256={n:sha256(target/n) for n in ('frames.jsonl','positions.csv','tracking-minimap.mp4')},accuracy_metrics=None))
        manifest['clips'][clip]=dict(video=info,summary=stats,id_independence_check_passed=True)
        print(clip,json.dumps(stats),flush=True)
    # A self-contained evidence tab, kept separate from inference and display.
    shutil.copytree(identity_root,output/'identity-review')
    shutil.copyfile(Path(__file__).with_name('identity_presentation.html'),output/'index.html')
    manifest.update(status='complete',seconds=time.perf_counter()-started)
    write_json(output/'experiment.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identities',type=Path,default=Path('outputs/temporal-reid-final'))
    p.add_argument('--cameras',type=Path,default=Path('outputs/challenge-player-smoothed-v1'))
    p.add_argument('--features',type=Path,default=Path('outputs/temporal-features-v1'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        run(a.identities,a.cameras,a.features,a.output)
    except BaseException as error:
        path=a.output/'experiment.json'
        if path.is_file():
            record=json.loads(path.read_text())
            if record['status']=='running':
                record.update(status='failed',error=repr(error));write_json(path,record)
        raise


if __name__=='__main__':
    main()
