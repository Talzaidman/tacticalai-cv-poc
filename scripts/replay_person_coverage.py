"""Replay identical cached detections through controlled TrackTrack settings."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track_tracker import TRACKTRACK
from ultralytics.trackers.utils.reid import ReID

from tacticalai.court_gate import court_gate
from tacticalai.export import records, sha256, write_json


class FrameEncoder:
    """Reuse exact crop embeddings across variants, without changing association."""
    def __init__(self, encoder):
        self.encoder=encoder
        self.cache={}

    def __call__(self,image,boxes):
        keys=[tuple(b[:4].tolist()) for b in boxes]
        missing=[i for i,k in enumerate(keys) if k not in self.cache]
        if missing:
            for i,feature in zip(missing,self.encoder(image,boxes[missing])):
                self.cache[keys[i]]=feature
        return [self.cache[k] for k in keys]


def run(source,cameras,output):
    torch.set_num_threads(4)
    output.mkdir(parents=True,exist_ok=False)
    encoder=ReID('models/yolo26n-cls.pt',device='cpu')
    shared=FrameEncoder(encoder)
    variants={
        'control':{},
        'lower_gates':dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25),
        'court':dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25),
        'court_confirm1':dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,min_track_len=1),
        'court_overlap':dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,min_track_len=1,tai_thr=.8),
    }
    report=dict(status='running',primary_detector_rerun=False,reid_rerun=True,
        reid_model='yolo26n-cls.pt',reid_model_sha256=sha256(Path(encoder.model.ckpt_path)),
        detector_comparison='identical saved detections for every tracker variant',
        recovered_pre_nms_boxes_used=False,
        replay_limit='Saved post-NMS detections do not include TrackTrack loose-NMS recovery extras. Control is a local replay, not an exact reproduction of the original GPU run.',
        court_margin_m=3.,court_minimum_box_overlap=.1,
        versions={p:importlib.metadata.version(p) for p in ('ultralytics','torch','opencv-python','numpy')},clips={})
    write_json(output/'experiment.json',report)
    started=time.perf_counter()
    try:
        for clip in ('attack_clip','attack_clip2'):
            src=source/clip/'tracktrack'
            meta=json.loads((src/'metadata.json').read_text())
            path=src/'frames.jsonl'
            frames=[json.loads(l) for l in path.read_text().splitlines()]
            camera_path=cameras/clip/'calibration.jsonl'
            camera_meta=json.loads((cameras/clip/'metadata.json').read_text())
            if sha256(camera_path)!=camera_meta['calibration_sha256'] or sha256(path)!=camera_meta['tracking_frames_sha256']:
                raise ValueError('Automatic camera/tracking provenance mismatch')
            calibration=[json.loads(l) for l in camera_path.read_text().splitlines()]
            video=Path('data/raw')/(clip+'.mp4')
            if sha256(video)!=meta['video_sha256']:raise ValueError('Video mismatch')
            trackers={}
            stats={}
            streams={}
            for name,changes in variants.items():
                settings=dict(meta['tracker_config'],**changes,device='cpu')
                tracker=TRACKTRACK(SimpleNamespace(**dict(settings,with_reid=False)))
                tracker.encoder=shared
                tracker.args.with_reid=True
                tracker.reset()
                trackers[name]=tracker
                folder=output/clip/name;folder.mkdir(parents=True)
                write_json(folder/'metadata.json',dict(status='running',tracker_config=settings,
                    detector_config=meta['config'],source_frames_sha256=sha256(path),
                    camera_sha256=sha256(camera_path),court_gate_enabled=name.startswith('court')))
                streams[name]=(folder/'frames.jsonl').open('w',encoding='utf8')
                stats[name]=dict(detections=0,accepted_detections=0,confirmed_tracks=0,unassigned_detections=0,rejected_outside=0,frames=0)
            cap=cv2.VideoCapture(str(video))
            try:
                for index,frame in enumerate(frames):
                    ok,image=cap.read()
                    if not ok:raise ValueError('Source decode failed')
                    shared.cache={}
                    camera=calibration[index]
                    rows=[]
                    for r in frame['detections']:
                        gate=court_gate(r['bbox_xyxy'],camera['refined_image_to_court_h'],camera['temporal_quality'],
                                        (meta['source']['width'],meta['source']['height']))
                        rows.append(dict(r,**gate,role=None))
                    for name,tracker in trackers.items():
                        accepted=[r for r in rows if r['keep'] or not name.startswith('court')]
                        array=np.array([r['bbox_xyxy']+[r['det_confidence'],0.] for r in accepted],np.float32).reshape(-1,6)
                        result=tracker.update(Boxes(array,image.shape[:2]),image)
                        tracks=records(Boxes(result[:,:7] if len(result) else np.empty((0,7),np.float32),image.shape[:2]),index,meta['source']['fps'])
                        association={int(t[-1]):int(t[4]) for t in result}
                        detections=[dict(r,track_id=association.get(i),
                            association_status='confirmed' if i in association else 'unassigned_detection') for i,r in enumerate(accepted)]
                        item=dict(frame_idx=index,timestamp_s=frame['timestamp_s'],detections=detections,tracks=tracks,
                                  rejected_detections=[r for r in rows if not r['keep']] if name.startswith('court') else [],
                                  calibration=camera['temporal_quality'],image_to_court_h=camera['refined_image_to_court_h'])
                        streams[name].write(json.dumps(item,allow_nan=False)+'\n')
                        counts=stats[name]
                        for key,value in dict(detections=len(rows),accepted_detections=len(accepted),confirmed_tracks=len(tracks),
                            unassigned_detections=len(accepted)-len(association),rejected_outside=len(item['rejected_detections']),frames=1).items():
                            counts[key]+=value
                    if index%25==0:print(clip,index,'/',len(frames),flush=True)
            finally:
                cap.release()
                for stream in streams.values():stream.close()
            for name in variants:
                folder=output/clip/name
                metadata=json.loads((folder/'metadata.json').read_text())
                metadata.update(status='complete',summary=stats[name],frames_sha256=sha256(folder/'frames.jsonl'))
                write_json(folder/'metadata.json',metadata)
            report['clips'][clip]=stats
            print(clip,json.dumps(stats),flush=True)
        report.update(status='complete',seconds=time.perf_counter()-started)
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:write_json(output/'experiment.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/occlusion-20261015T204150Z-029a7a35/reid'))
    parser.add_argument('--cameras',type=Path,default=Path('outputs/challenge-player-smoothed-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.cameras,args.output)
