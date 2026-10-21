"""Read-only return/duplicate-ID audit; supplied IDs are evaluation targets only."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import cv2
from PIL import Image, ImageDraw

from tacticalai.export import sha256,write_json


def stats(frames,tid):
    indices=[f['frame_idx'] for f in frames if any(r['track_id']==tid for r in f['tracks'])]
    return dict(first=indices[0] if indices else None,last=indices[-1] if indices else None,
                observations=len(indices),gaps=[[a,b] for a,b in zip(indices,indices[1:]) if b>a+1],frames=indices)


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection,1e-9))


def crop_sheet(output,clip,frames,ids,indices):
    """Render saved tracker crops for inspection; do not change associations."""
    cell_w,cell_h=280,220
    sheet=Image.new('RGB',(cell_w*len(ids),cell_h*len(indices)), '#111827')
    draw=ImageDraw.Draw(sheet)
    video=cv2.VideoCapture(str(Path('data/raw')/f'{clip}.mp4'))
    try:
        for row,idx in enumerate(indices):
            video.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,frame=video.read()
            if not ok:raise ValueError(f'Cannot read source frame {idx}')
            height,width=frame.shape[:2]
            for column,tid in enumerate(ids):
                left,top=column*cell_w,row*cell_h
                draw.text((left+10,top+8),f'Frame {idx} | T{tid}',fill='white')
                track=next((r for r in frames[idx]['tracks'] if r['track_id']==tid),None)
                if track is None:
                    draw.text((left+10,top+42),'No exported observation',fill='#9ca3af')
                    continue
                x1,y1,x2,y2=track['bbox_xyxy']
                x1,y1=max(0,int(x1)-8),max(0,int(y1)-8)
                x2,y2=min(width,int(x2)+9),min(height,int(y2)+9)
                crop=Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
                crop.thumbnail((cell_w-20,cell_h-42))
                sheet.paste(crop,(left+(cell_w-crop.width)//2,top+34))
    finally:
        video.release()
    sheet.save(output,quality=92)


def run(output,clip,first_id,return_id,reported_id,reference_id):
    output.mkdir(parents=True,exist_ok=True)
    old_root=Path('outputs/occlusion-20261015T204150Z-029a7a35/reid')/clip/'tracktrack'
    old_path=old_root/'frames.jsonl'
    old=[json.loads(l) for l in old_path.read_text().splitlines()]
    root=Path('outputs/person-coverage-investigation/tracker-replay-safe')/clip
    current_path=root/'court/frames.jsonl'
    current=[json.loads(l) for l in current_path.read_text().splitlines()]
    m=json.loads((root/'court/metadata.json').read_text())
    if sha256(current_path)!=m['frames_sha256']:raise ValueError('Current frames changed')
    result=dict(status='complete',clip=clip,fps=25,first_id=first_id,return_id=return_id,
        reported_id=reported_id,reference_id=reference_id,evaluation_only=True,inference_modified=False,
        original_reference=stats(old,reference_id),current_first=stats(current,first_id),
        current_return=stats(current,return_id),current_reported=stats(current,reported_id),
        source_sha256=dict(original=sha256(old_path),current=sha256(current_path)),variant_correspondences={})
    result['coexisting_frames']=sorted(set(result['current_first']['frames'])&set(result['current_return']['frames']))
    for variant in ('control','lower_gates','court','court_confirm1','court_overlap'):
        path=root/variant/'frames.jsonl'
        metadata=json.loads(path.with_name('metadata.json').read_text())
        if sha256(path)!=metadata['frames_sha256']:raise ValueError('Variant frames changed')
        frames=[json.loads(l) for l in path.read_text().splitlines()]
        matches={}
        for f,before in zip(frames,old):
            target=next((r for r in before['tracks'] if r['track_id']==reference_id),None)
            if target:
                for row in f['tracks']:
                    if iou(row['bbox_xyxy'],target['bbox_xyxy'])>.7:
                        matches.setdefault(str(row['track_id']),[]).append(f['frame_idx'])
        result['variant_correspondences'][variant]=dict(
            method='Same-frame bbox IoU > 0.7 to earlier automatic reference track; not physical identity ground truth',
            tracks={tid:dict(observations=len(fs),first=fs[0],last=fs[-1]) for tid,fs in matches.items()},
            source_sha256=sha256(path))
    start=result['current_return']['first']
    at_start=current[start]
    rows={str(tid):next(r for r in at_start['tracks'] if r['track_id']==tid) for tid in (first_id,return_id)}
    result['return_start']=dict(frame=start,rows=rows,
        overlap_iou=iou(rows[str(first_id)]['bbox_xyxy'],rows[str(return_id)]['bbox_xyxy']),
        tracker_config=m['tracker_config'])
    write_json(output/'audit.json',result)
    source=Path('outputs/id-continuity-review-v1')/clip
    video_meta=json.loads((source/'metadata.json').read_text())
    for origin,dest in [('linked-clean.mp4','earlier.mp4'),('updated-clean.mp4','updated.mp4')]:
        if sha256(source/origin)!=video_meta['artifacts_sha256'][origin]:raise ValueError('Review video changed')
        shutil.copyfile(source/origin,output/dest)
    # The explanatory page is specific to the inspected case. Other evaluation
    # IDs still produce the machine-readable audit without reusing those claims.
    if (clip,first_id,return_id,reported_id,reference_id)==('attack_clip2',38,292,32,215):
        crop_sheet(output/'T38-T292-crops.jpg',clip,current,[38,292],[0,70,74,75,81,84,87,96,100])
        crop_sheet(output/'T32-T292-crops.jpg',clip,current,[32,292],[0,74,75,100,149,174])
        shutil.copyfile(Path(__file__).with_name('reid_return_audit.html'),output/'index.html')
    print(json.dumps({k:result[k] for k in ('coexisting_frames','return_start')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/reid-return-audit-v1'))
    p.add_argument('--clip',default='attack_clip2')
    p.add_argument('--first-id',type=int,default=38)
    p.add_argument('--return-id',type=int,default=292)
    p.add_argument('--reported-id',type=int,default=32)
    p.add_argument('--reference-id',type=int,default=215)
    a=p.parse_args();run(a.output,a.clip,a.first_id,a.return_id,a.reported_id,a.reference_id)
