"""Small reproducible detector ablation, separate from tracking thresholds."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

from tacticalai.export import records, sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='models/yolo26s.pt')
    parser.add_argument('--sizes',type=int,nargs='+',default=[1280,1920])
    parser.add_argument('--nms-free',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    model=YOLO(args.model)
    meta=dict(status='running',model=args.model,model_sha256=sha256(Path(model.ckpt_path)),
        conf=.03,iou=.7,nms=False if args.nms_free else None,max_det=300,device='cpu',classes=[0],sizes=args.sizes,
        versions={p:importlib.metadata.version(p) for p in ('ultralytics','torch','opencv-python')},
        source_sha256=sha256(Path(__file__)),clips={})
    write_json(args.output/'metadata.json',meta)
    try:
        for clip,indices in [('attack_clip',[0,55,104,131]),('attack_clip2',[0,55,104,149])]:
            video=Path('data/raw')/(clip+'.mp4')
            cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
            meta['clips'][clip]=dict(video_sha256=sha256(video),frames=indices)
            try:
                for size in args.sizes:
                    rows=[];started=time.perf_counter()
                    for index in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
                        if not ok:raise ValueError('Decode failed')
                        result=model.predict(image,classes=[0],device='cpu',imgsz=size,
                            conf=.03,iou=.7,max_det=300,nms=False if args.nms_free else None,verbose=False)[0]
                        rows.append(dict(frame_idx=index,timestamp_s=index/fps,
                            detections=records(result.boxes.cpu().numpy(),index,fps)))
                    write_json(args.output/f'{clip}-{size}.json',dict(frames=rows,seconds=time.perf_counter()-started))
                    print(args.model,clip,size,[len(r['detections']) for r in rows],flush=True)
            finally:cap.release()
        meta['status']='complete'
    except BaseException as error:
        meta.update(status='failed',error=repr(error));raise
    finally:write_json(args.output/'metadata.json',meta)


if __name__=='__main__':main()
