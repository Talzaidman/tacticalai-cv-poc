"""Automatic detection-coverage comparison using cached, controlled experiments."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from .calibration import project_track
from .challenge_presentation import CLIPS,COURT
from .court import floor_lines
from .export import sha256,write_json
from .identity_presentation import person_color,text
from .silhouette_presentation import associate_masks,render


def web_frames(frames):
    """Small display payload; full-precision measurements remain in JSONL."""
    result={}
    keys=('track_id','role','bbox_xyxy','display_label','display_caption','display_color_hex',
          'court_xy_m','court_region','court_gate','silhouette_xy')
    for mode,values in frames.items():
        items=[]
        for frame in values:
            rows=[]
            for row in frame['tracks']:
                view={key:row.get(key) for key in keys}
                if view['silhouette_xy']:
                    polygon=cv2.approxPolyDP(np.asarray(view['silhouette_xy'],np.float32),.5,True).reshape(-1,2)
                    if len(polygon)>=3:view['silhouette_xy']=np.round(polygon,2).tolist()
                rows.append(view)
            items.append(dict(frame_idx=frame['frame_idx'],timestamp_s=frame['timestamp_s'],tracks=rows,
                calibration=dict(method=frame['calibration']['method']),outside_removed=frame['outside_removed']))
        result[mode]=items
    return result


def observations(source,mask,info,mode):
    """Keep unconfirmed measurements visible without inventing track IDs."""
    rows=source['tracks'] if mode=='before' else source['detections']
    result=[]
    matrix=source['image_to_court_h']
    for row in rows:
        row=dict(row,role=None,player_id=str(row['track_id']) if row['track_id'] is not None else None)
        row['display_label']=f"T{row['track_id']}" if row['track_id'] is not None else '?'
        row['display_caption']=row['display_label'] if row['track_id'] is not None else '? ID pending'
        row['display_color_hex']='#'+''.join(f'{c:02x}' for c in person_color(row['player_id'])[::-1])
        row=project_track(row,np.asarray(matrix) if matrix is not None else None,
            source['calibration'],COURT,(info['width'],info['height']))
        result.append(row)
    associate_masks(result,mask['boxes'],mask['polygons'],mask['confidences'])
    return dict(frame_idx=source['frame_idx'],timestamp_s=source['timestamp_s'],tracks=result,
        calibration=source['calibration'],outside_removed=len(source['rejected_detections']))


def run(investigation,output):
    import imageio_ffmpeg
    output.mkdir(parents=True,exist_ok=False)
    replay=investigation/'tracker-replay-safe'
    experiment=json.loads((replay/'experiment.json').read_text())
    mask_meta=json.loads((investigation/'masks-all/metadata.json').read_text())
    if experiment['status']!='complete' or mask_meta['status']!='complete':
        raise ValueError('Inputs are incomplete')
    manifest=dict(status='running',clips={},automatic_only=True,
        primary_detector_rerun=False,tracker_and_reid_rerun=True,segmentation_rerun=True,
        roles_used=False,identity_gallery_rerun=False,
        label_policy='T = tracker ID, ? = current detection without a confirmed ID. IDs are local to each variant/clip.',
        court_policy='Automatic geometry; 3m safety margin, clipped/uncertain cases retained. Not a role classifier.',
        outside_gate_limit='Approximate cameras and bbox-bottom footpoints can retain spectators or exclude boundary persons. No verified floor mask.',
        offline_future_frames_used=True,court=COURT,
        detector_config='YOLO26s, conf 0.10, imgsz 1280, iou 0.7, max_det 100; identical cached detections in both variants',
        tracking_config=dict(before=dict(high=.6,low=.25,new=.7,min_track_len=3,tai_thr=.55),
                             after=dict(high=.25,low=.1,new=.25,min_track_len=3,tai_thr=.55),
                             reid='yolo26n-cls.pt',gmc='sparseOptFlow',buffer=30),
        masks_config=mask_meta, source_replay=experiment)
    write_json(output/'experiment.json',manifest)
    start=time.perf_counter();lines=floor_lines(COURT)
    try:
        for clip in CLIPS:
            folder=output/clip;folder.mkdir()
            video=Path('data/raw')/(clip+'.mp4')
            if sha256(video)!=mask_meta['clips'][clip]['video_sha256']:
                raise ValueError('Video mismatch')
            masks_path=investigation/'masks-all'/clip/'masks.jsonl'
            if sha256(masks_path)!=mask_meta['clips'][clip]['masks_sha256']:
                raise ValueError('Mask cache changed')
            masks=[json.loads(l) for l in masks_path.read_text().splitlines()]
            variants={}
            metadata={}
            for mode,name in [('before','control'),('after','court')]:
                path=replay/clip/name/'frames.jsonl'
                metadata[mode]=json.loads(path.with_name('metadata.json').read_text())
                if sha256(path)!=metadata[mode]['frames_sha256']:
                    raise ValueError('Tracking cache changed')
                variants[mode]=[json.loads(l) for l in path.read_text().splitlines()]
            cap=cv2.VideoCapture(str(video))
            info=dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if any(len(v)!=info['frames'] for v in variants.values()) or len(masks)!=info['frames']:
                raise ValueError('Frame count mismatch')
            frames={mode:[] for mode in variants}
            processes={mode:subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y',
                '-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(info['fps']),'-i','pipe:0',
                '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/f'{mode}.mp4')],
                stdin=subprocess.PIPE) for mode in variants}
            try:
                for index,mask in enumerate(masks):
                    ok,image=cap.read()
                    if not ok:raise ValueError('Source decode failed')
                    if mask['frame_idx']!=index or abs(mask['timestamp_s']-index/info['fps'])>1e-6:
                        raise ValueError('Mask timestamp mismatch')
                    for mode in variants:
                        frame=observations(variants[mode][index],mask,info,mode)
                        frames[mode].append(frame)
                        rendered=render(image,frame,lines)
                        tracked=sum(r['track_id'] is not None for r in frame['tracks'])
                        text(rendered,f"{mode.upper()} | {len(frame['tracks'])} observations | {tracked} track IDs | ? = unassigned",(330,28),.5)
                        processes[mode].stdin.write(rendered.tobytes())
                        if index in (0,55,104,131):
                            cv2.imwrite(str(folder/f'{mode}-{index:03d}.jpg'),rendered)
                    if index%50==0:print(clip,'render',index,flush=True)
                for process in processes.values():
                    process.stdin.close()
                    if process.wait()!=0:raise ValueError('Encode failed')
            finally:
                cap.release()
                for process in processes.values():
                    if process.poll() is None:process.kill();process.wait()
            summaries={}
            for mode in variants:
                check=cv2.VideoCapture(str(folder/f'{mode}.mp4'));decoded=0
                while check.read()[0]:decoded+=1
                check.release()
                if decoded!=info['frames']:raise ValueError('Output decode mismatch')
                summaries[mode]=dict(frames=decoded,observations=sum(len(f['tracks']) for f in frames[mode]),
                    with_track_id=sum(r['track_id'] is not None for f in frames[mode] for r in f['tracks']),
                    unassigned=sum(r['track_id'] is None for f in frames[mode] for r in f['tracks']),
                    masks=sum(r['silhouette_xy'] is not None for f in frames[mode] for r in f['tracks']),
                    outside_removed=sum(f['outside_removed'] for f in frames[mode]))
                with (folder/f'{mode}-frames.jsonl').open('w',encoding='utf8') as stream:
                    for frame in frames[mode]:stream.write(json.dumps(frame,allow_nan=False)+'\n')
            clean=Path('outputs/automatic-silhouettes-v1')/clip/'clean.mp4'
            shutil.copyfile(clean,folder/'clean.mp4')
            with (folder/'observations.csv').open('w',newline='',encoding='utf8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['frame_idx','timestamp_s','track_id','det_confidence',
                    'court_x_m','court_y_m','court_gate','silhouette_confidence']);writer.writeheader()
                for frame in frames['after']:
                    for row in frame['tracks']:
                        xy=row['court_xy_m'] or [None,None]
                        record={k:row.get(k) for k in writer.fieldnames}
                        record.update(frame_idx=frame['frame_idx'],timestamp_s=frame['timestamp_s'],court_x_m=xy[0],court_y_m=xy[1])
                        writer.writerow(record)
            payload=dict(video=info,clip=clip,frames=web_frames(frames),summary=summaries,
                         web_contour_tolerance_source_px=.5,court_lines=[l.tolist() for l in lines])
            (folder/'review-data.json').write_text(json.dumps(payload,allow_nan=False,separators=(',',':')),encoding='utf8')
            write_json(folder/'metadata.json',dict(status='complete',summary=summaries,automatic_only=True,
                source_tracking_metadata=metadata,video_sha256=sha256(video),masks_sha256=sha256(masks_path),
                artifacts_sha256={p.name:sha256(p) for p in folder.iterdir() if p.is_file()}))
            manifest['clips'][clip]=dict(video=info,summary=summaries)
            print(clip,summaries,flush=True)
        evidence=output/'evidence';evidence.mkdir()
        shutil.copyfile(investigation/'evaluation.json',evidence/'evaluation.json')
        for name in ('detector-small','detector-medium','detector-small-e2e'):
            shutil.copytree(investigation/name,evidence/name)
        write_json(evidence/'tracking-comparison.json',experiment)
        snapshots=output/'run_source';snapshots.mkdir()
        for name in ('coverage_presentation.py','coverage_presentation.html','court_gate.py','silhouette_presentation.py'):
            shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
        for name in ('replay_person_coverage.py','probe_detection_coverage.py','cache_person_masks.py','evaluate_person_coverage.py'):
            shutil.copyfile(Path('scripts')/name,snapshots/name)
        shutil.copyfile(Path(__file__).with_name('coverage_presentation.html'),output/'index.html')
        manifest.update(status='complete',render_processing_s=time.perf_counter()-start,
                        source_sha256={p.name:sha256(p) for p in snapshots.iterdir()})
        write_json(output/'experiment.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error));write_json(output/'experiment.json',manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--investigation',type=Path,default=Path('outputs/person-coverage-investigation'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.investigation,a.output)
