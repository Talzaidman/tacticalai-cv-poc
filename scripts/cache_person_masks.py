"""Cache every person mask, independently of tracker confirmation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tacticalai.export import sha256,write_json


def run(output,imgsz):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    model=YOLO('models/yolo26s-seg.pt')
    meta=dict(status='running',model='yolo26s-seg.pt',model_sha256=sha256(Path(model.ckpt_path)),
        imgsz=imgsz,conf=.1,iou=.7,classes=[0],retina_masks=True,device='cpu',clips={},
        versions={p:importlib.metadata.version(p) for p in ('ultralytics','torch','opencv-python','numpy')})
    write_json(output/'metadata.json',meta)
    try:
        for clip in ('attack_clip','attack_clip2'):
            video=Path('data/raw')/(clip+'.mp4')
            cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
            count=0;model_s=0.;start=time.perf_counter()
            folder=output/clip;folder.mkdir()
            try:
                with (folder/'masks.jsonl').open('w',encoding='utf8') as stream:
                    while True:
                        ok,image=cap.read()
                        if not ok:break
                        t=time.perf_counter()
                        result=model.predict(image,classes=[0],conf=.1,iou=.7,device='cpu',
                            imgsz=imgsz,retina_masks=True,verbose=False)[0]
                        polygons=result.masks.xy if result.masks is not None else []
                        model_s+=time.perf_counter()-t
                        record=dict(frame_idx=count,timestamp_s=count/fps,
                            boxes=result.boxes.xyxy.cpu().numpy().tolist() if polygons else [],
                            confidences=result.boxes.conf.cpu().numpy().tolist() if polygons else [],
                            polygons=[np.round(p,2).tolist() for p in polygons])
                        stream.write(json.dumps(record,allow_nan=False)+'\n');count+=1
                        if count%25==0:print(clip,count,flush=True)
            finally:cap.release()
            meta['clips'][clip]=dict(video_sha256=sha256(video),frames=count,model_s=model_s,
                model_fps=count/model_s,processing_s=time.perf_counter()-start,
                masks_sha256=sha256(folder/'masks.jsonl'))
            print(clip,meta['clips'][clip],flush=True)
        meta['status']='complete'
    except BaseException as error:
        meta.update(status='failed',error=repr(error));raise
    finally:write_json(output/'metadata.json',meta)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--imgsz',type=int,default=1280)
    a=p.parse_args();run(a.output,a.imgsz)
