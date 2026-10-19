"""Cache time-indexed appearance observations for temporal identity experiments."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .clip_reident import ClipReIdent
from .export import sha256, write_json


def cached_features(root, clip):
    meta=json.loads((root/clip/'metadata.json').read_text())
    segments=json.loads((root/clip/'tracklets.json').read_text())
    result={}
    with np.load(root/clip/'embeddings.npz',allow_pickle=False) as values:
        for s in segments:
            indices=np.flatnonzero(values['segment_ids']==s['segment_id'])
            if len(indices)!=len(s['sample_frames']):
                raise ValueError('Cached sample/embedding mismatch')
            for index,f in zip(indices,s['sample_frames']):
                result[f,s['source_track_id']]=values['embeddings'][index]
    return meta,result


def run(original,enriched,output,step=5,device='cpu'):
    import torch
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    encoder=None
    for clip in ('attack_clip','attack_clip2'):
        started=time.perf_counter()
        meta,old=cached_features(original,clip)
        other,extra=cached_features(enriched,clip)
        if any(meta[k]!=other[k] for k in ('checkpoint_sha256','source_frames_sha256','video_sha256')):
            raise ValueError('Embedding caches differ in model or source')
        old.update(extra)
        source=Path(meta['config']['run'])/'frames.jsonl'
        video=Path(meta['video'])
        if sha256(source)!=meta['source_frames_sha256'] or sha256(video)!=meta['video_sha256']:
            raise ValueError('Source data changed')
        frames=[json.loads(line) for line in source.read_text().splitlines()]
        by_id={}
        for frame in frames:
            for row in frame['tracks']:
                by_id.setdefault(row['track_id'],[]).append((frame['frame_idx'],row))
        keys=set(old)
        # Time sampling, not gap splitting; retain clear appearances across gaps.
        for tid,observations in by_id.items():
            previous=-step
            for f,row in observations:
                if f-previous>=step:
                    keys.add((f,tid));previous=f
            keys.add((observations[-1][0],tid))
            # Adjacent observations of large image-space changes refine possible
            # boundaries. These are candidates, not declarations of an ID switch.
            for (fa,a),(fb,b) in zip(observations,observations[1:]):
                ba,bb=np.array(a['bbox_xyxy']),np.array(b['bbox_xyxy'])
                jump=np.linalg.norm((ba[:2]+ba[2:]-bb[:2]-bb[2:])/2)/max(1,ba[3]-ba[1])
                if jump>0.3:
                    keys.update((f,tid) for f,r in observations if abs(f-fb)<=5 or abs(f-fa)<=5)
        folder=output/clip
        (folder/'crops').mkdir(parents=True)
        observations,new_crops,new_indices=[],[],[]
        vectors=[]
        cap=cv2.VideoCapture(str(video))
        for frame in frames:
            ok,image=cap.read()
            if not ok:raise RuntimeError('Incomplete source video')
            for row in frame['tracks']:
                f,tid=frame['frame_idx'],row['track_id']
                if (f,tid) not in keys:continue
                box=np.array(row['bbox_xyxy'])
                x1,y1=np.maximum(0,np.floor(box[:2]).astype(int))
                x2,y2=np.minimum([image.shape[1],image.shape[0]],np.ceil(box[2:]).astype(int))
                if x2<=x1 or y2<=y1:continue
                crop=image[y1:y2,x1:x2].copy()
                area=max(1,(box[2]-box[0])*(box[3]-box[1]))
                overlap=max((max(0,min(box[2],r['bbox_xyxy'][2])-max(box[0],r['bbox_xyxy'][0]))*
                             max(0,min(box[3],r['bbox_xyxy'][3])-max(box[1],r['bbox_xyxy'][1]))/area
                             for r in frame['tracks'] if r['track_id']!=tid),default=0)
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                # Generic foreground-color fractions aid abrupt uniform/role
                # changes; not a learned team classifier or a jersey label.
                saturated=hsv[:,:,1]>70
                yellow=float(np.mean(saturated&(hsv[:,:,0]>=18)&(hsv[:,:,0]<=42)))
                red=float(np.mean(saturated&((hsv[:,:,0]<12)|(hsv[:,:,0]>168))))
                sharpness=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
                edge=bool(min(box[0],box[1],image.shape[1]-box[2],image.shape[0]-box[3])<2)
                quality=float(row['det_confidence']*(1-overlap)*min(1,sharpness/100))
                index=len(observations)
                observations.append(dict(frame=f,track_id=tid,bbox_xyxy=row['bbox_xyxy'],
                                         confidence=row['det_confidence'],overlap=overlap,
                                         sharpness=sharpness,edge=edge,quality=quality,
                                         yellow_fraction=yellow,red_fraction=red,
                                         crop=f'crops/t{tid}-f{f}.jpg'))
                cv2.imwrite(str(folder/observations[-1]['crop']),crop)
                if (f,tid) in old:vectors.append(old[f,tid])
                else:
                    vectors.append(None);new_crops.append(crop);new_indices.append(index)
        cap.release()
        run_meta=dict(status='running',source=meta,source_frames_sha256=meta['source_frames_sha256'],
                      checkpoint_sha256=meta['checkpoint_sha256'],video_sha256=meta['video_sha256'],
                      step=step,observations=len(observations),new_crops=len(new_crops),
                      reused_crops=len(observations)-len(new_crops),device=device,
                      python_libraries={p:importlib.metadata.version(p) for p in ('torch','open-clip-torch','numpy','opencv-python')},
                      script_sha256=sha256(Path(__file__)),
                      note='No review labels, known switch IDs, or boundaries enter feature sampling.')
        write_json(folder/'metadata.json',run_meta)
        print(f'{clip}: {len(observations)} samples; {len(new_crops)} new model crops on {device}',flush=True)
        try:
            if new_crops:
                if encoder is None:encoder=ClipReIdent(meta['checkpoint'],device,4)
                if encoder.checkpoint_sha256!=meta['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
                features=encoder.encode(new_crops)
                for index,feature in zip(new_indices,features):vectors[index]=feature
            np.savez_compressed(folder/'features.npz',features=np.stack(vectors))
            write_json(folder/'observations.json',observations)
            run_meta.update(status='complete',seconds=time.perf_counter()-started)
        except BaseException as error:
            run_meta.update(status='failed',error=repr(error));raise
        finally:write_json(folder/'metadata.json',run_meta)
        print(f'{clip}: feature cache complete in {run_meta["seconds"]:.1f}s',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,default=Path('outputs/clip-reident-paper-v1'))
    p.add_argument('--enriched',type=Path,default=Path('outputs/gallery-enriched-crops-v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=int,default=5)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.step<1:p.error('step must be positive')
    run(a.original,a.enriched,a.output,a.step,a.device)


if __name__=='__main__':main()
