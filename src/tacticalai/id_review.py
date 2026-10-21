"""Clean ID-continuity review of updated tracking and earlier person linking."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

import cv2

from .challenge_presentation import CLIPS
from .court_gate import court_gate
from .coverage_presentation import web_frames
from .export import sha256,write_json
from .silhouette_presentation import render


def run(coverage,linked,output):
    import imageio_ffmpeg
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',clips={},automatic_only=True,roles_used=False,
        networks_rerun=False,tracking_reid_enabled=True,tracking_reid_model='yolo26n-cls.pt',
        new_tracking_gallery_identity_linking_rerun=False,
        defaults=dict(pending_detections=False,boundary_cases=False),
        id_sources=dict(updated='Updated TrackTrack + online ReID; T labels are raw tracker IDs',
                        linked='Earlier automatic CLIP-ReIdent temporal/gallery hypotheses; P labels use the older detection/tracking run'),
        accuracy_claim=None)
    write_json(output/'experiment.json',report)
    try:
        for clip in CLIPS:
            old=json.loads((linked/clip/'review-data.json').read_text())
            old_meta=json.loads((linked/clip/'metadata.json').read_text())
            current_meta=json.loads((coverage/clip/'metadata.json').read_text())
            if sha256(linked/clip/'review-data.json')!=old_meta['artifacts_sha256']['review-data.json']:
                raise ValueError('Earlier identity cache changed')
            if sha256(coverage/clip/'after-frames.jsonl')!=current_meta['artifacts_sha256']['after-frames.jsonl']:
                raise ValueError('Updated tracking cache changed')
            frames={
                'updated':[json.loads(l) for l in (coverage/clip/'after-frames.jsonl').read_text().splitlines()],
                'linked':copy.deepcopy(old['frames'])}
            info=old['video']
            for mode,values in frames.items():
                if len(values)!=info['frames']:raise ValueError('Frame count mismatch')
                for frame in values:
                    frame.setdefault('outside_removed',0)
                    for row in frame['tracks']:
                        row['role']=None
                        if mode=='linked':
                            row.update(court_gate(row['bbox_xyxy'],frame['image_to_court_h'],
                                frame['calibration'],(info['width'],info['height'])))
                            row['display_caption']=f"{row['display_label']} / T{row['track_id']}"
            target=output/clip;target.mkdir()
            payload=dict(video=info,clip=clip,frames=web_frames(frames),court_lines=old['court_lines'])
            (target/'review-data.json').write_text(json.dumps(payload,allow_nan=False,separators=(',',':')),encoding='utf8')
            shutil.copyfile(coverage/clip/'clean.mp4',target/'clean.mp4')
            video=Path('data/raw')/(clip+'.mp4')
            if sha256(video)!=info['sha256']:raise ValueError('Video mismatch')
            cap=cv2.VideoCapture(str(video))
            processes={mode:subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y',
                '-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(info['fps']),'-i','pipe:0',
                '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target/f'{mode}-clean.mp4')],
                stdin=subprocess.PIPE) for mode in frames}
            try:
                for index in range(info['frames']):
                    ok,image=cap.read()
                    if not ok:raise ValueError('Source decode incomplete')
                    for mode,values in frames.items():
                        frame=values[index]
                        rows=[r for r in frame['tracks'] if r['track_id'] is not None and r['court_gate']=='inside']
                        ids=[r['display_label'] for r in rows if r['display_label']!='?']
                        if len(ids)!=len(set(ids)):raise ValueError('A displayed ID is assigned to multiple observations in one frame')
                        rendered=render(image,dict(frame,tracks=rows),old['court_lines'])
                        processes[mode].stdin.write(rendered.tobytes())
                        if index in (0,55,104,131):cv2.imwrite(str(target/f'{mode}-{index:03d}.jpg'),rendered)
                for process in processes.values():
                    process.stdin.close()
                    if process.wait()!=0:raise ValueError('Encode failed')
            finally:
                cap.release()
                for process in processes.values():
                    if process.poll() is None:process.kill();process.wait()
            for mode in frames:
                check=cv2.VideoCapture(str(target/f'{mode}-clean.mp4'));decoded=0
                while check.read()[0]:decoded+=1
                check.release()
                if decoded!=info['frames']:raise ValueError('Clean MP4 frame count mismatch')
            meta=dict(status='complete',frames=info['frames'],roles_used=False,networks_rerun=False,
                sources_sha256=dict(updated_frames=sha256(coverage/clip/'after-frames.jsonl'),
                    linked_frames=sha256(linked/clip/'review-data.json'),video=sha256(video)),
                tracking_reid_enabled=current_meta['source_tracking_metadata']['after']['tracker_config']['with_reid'],
                artifacts_sha256={p.name:sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json',meta);report['clips'][clip]=meta
            print(clip,'clean ID views verified',flush=True)
        shutil.copyfile(Path(__file__).with_name('id_review.html'),output/'index.html')
        snapshots=output/'run_source';snapshots.mkdir()
        for name in ('id_review.py','id_review.html','court_gate.py','coverage_presentation.py','silhouette_presentation.py'):
            shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
        report.update(status='complete',source_sha256={p.name:sha256(p) for p in snapshots.iterdir()})
        write_json(output/'experiment.json',report)
    except BaseException as error:
        report.update(status='failed',error=repr(error));write_json(output/'experiment.json',report);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage',type=Path,default=Path('outputs/person-coverage-review-v1'))
    p.add_argument('--linked',type=Path,default=Path('outputs/automatic-silhouettes-v1'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.coverage,a.linked,a.output)
