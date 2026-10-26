"""Cache locally computed pose keypoints near the exported ball, preserving player tracks."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .ball import iou
from .ball_cli import load_source
from .ball_control import contact_geometry
from .ball_fusion_cli import eligible_people,load_ball
from .calibration_cli import video_info
from .export import sha256,write_json


def match_poses(rows,boxes,keypoints,minimum_iou=.35):
    if not rows or not len(boxes):return []
    overlap=np.array([[iou(row['bbox_xyxy'],box) for box in boxes] for row in rows])
    valid=overlap>=minimum_iou
    rr,cc=linear_sum_assignment(-(valid*(min(overlap.shape)+1)+overlap))
    return [dict(player_id=rows[r]['player_id'],identity_segment_id=rows[r].get('identity_segment_id'),
                 track_id=rows[r]['track_id'],pose_box_iou=float(overlap[r,c]),
                 bbox_xyxy=list(map(float,boxes[c])),keypoints_xy_conf=np.asarray(keypoints[c]).tolist())
            for r,c in zip(rr,cc) if valid[r,c]]


def run(args):
    import torch
    from ultralytics import YOLO
    if not args.model.is_file():raise ValueError('Download the official pose model explicitly first')
    torch.set_num_threads(args.threads)
    model=YOLO(str(args.model))
    if model.task!='pose':raise ValueError('Expected a pose model')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',device='cpu',model=dict(path=str(args.model.resolve()),sha256=sha256(args.model)),
                imgsz=args.imgsz,threads=args.threads,minimum_pose_box_iou=.35,
                roi_policy='union of automatically classified players within .65 body heights of the ball silhouette distance',
                keypoint_order='COCO 17: left/right wrist indices 9/10; confidence is model visibility score',
                versions={p:importlib.metadata.version(p) for p in ('torch','ultralytics','numpy','opencv-python','scipy')},clips={})
    write_json(args.output/'experiment.json',report)
    started=time.perf_counter()
    for clip in args.clips:
        tick=time.perf_counter();video=args.videos/(clip+'.mp4');info=video_info(video)
        people,source=load_source(args.people_results,clip,info)
        balls=load_ball(args.ball_results/f'RF-DETR-tiles-Kalman_{clip}.json',
                        args.ball_results/f'RF-DETR-tiles-Kalman_{clip}.mp4',info)
        target=args.output/clip;target.mkdir()
        cap=cv2.VideoCapture(str(video));frames=[];inference_s=0.;roi_count=0
        try:
            for index,frame in enumerate(people):
                ok,image=cap.read()
                if not ok:raise ValueError('Video ended early')
                ball=balls['tracked'][index];rows=[];poses=[];roi=None
                if ball['state']!='none':
                    point=ball['measurement_xy'] if ball['state']=='detected' else [ball['x'],ball['y']]
                    rows=[row for row in eligible_people(frame['tracks'])
                          if contact_geometry(point,row)['distance_heights']<=.65]
                if rows:
                    boxes=np.asarray([r['bbox_xyxy'] for r in rows])
                    pad=.15*np.max(boxes[:,3]-boxes[:,1])
                    x1,y1=np.maximum(0,np.floor(boxes[:,:2].min(axis=0)-pad)).astype(int)
                    x2,y2=np.minimum([info['width'],info['height']],np.ceil(boxes[:,2:].max(axis=0)+pad)).astype(int)
                    roi=[int(x1),int(y1),int(x2),int(y2)]
                    t=time.perf_counter()
                    result=model.predict(image[y1:y2,x1:x2],device='cpu',imgsz=args.imgsz,
                                         conf=.2,iou=.7,verbose=False,save=False)[0]
                    inference_s+=time.perf_counter()-t;roi_count+=1
                    if result.keypoints is not None and result.boxes is not None:
                        pose_boxes=result.boxes.xyxy.cpu().numpy()+[x1,y1,x1,y1]
                        keypoints=result.keypoints.data.cpu().numpy().copy()
                        keypoints[:,:,:2]+=[x1,y1]
                        poses=match_poses(rows,pose_boxes,keypoints)
                frames.append(dict(frame_idx=index,timestamp_s=index/info['fps'],roi_xyxy=roi,poses=poses))
                if index%25==0:print(clip,index,'/',info['frames'],'pose matches',len(poses),flush=True)
        finally:cap.release()
        with (target/'poses.jsonl').open('w') as stream:
            for frame in frames:stream.write(json.dumps(frame,allow_nan=False)+'\n')
        metadata=dict(status='complete',video=info,source_people=source,
                      ball_json_sha256=sha256(args.ball_results/f'RF-DETR-tiles-Kalman_{clip}.json'),
                      frames=len(frames),roi_frames=roi_count,matched_people=sum(len(f['poses']) for f in frames),
                      inference_s=inference_s,processing_s=time.perf_counter()-tick,
                      poses_sha256=sha256(target/'poses.jsonl'))
        write_json(target/'metadata.json',metadata);report['clips'][clip]=metadata
        write_json(args.output/'experiment.json',report)
    report.update(status='complete',processing_s=time.perf_counter()-started,source_sha256=sha256(Path(__file__)))
    write_json(args.output/'experiment.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('models/yolo26n-pose.pt'))
    parser.add_argument('--people-results',type=Path,required=True)
    parser.add_argument('--ball-results',type=Path,required=True)
    parser.add_argument('--videos',type=Path,default=Path('data/raw'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--clips',nargs='+',default=['attack_clip','attack_clip2'])
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--threads',type=int,default=4)
    run(parser.parse_args())


if __name__=='__main__':main()
