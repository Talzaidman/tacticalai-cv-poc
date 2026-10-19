"""Select temporally separated, clear crops from original tracks and encode them."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .clip_reident import ClipReIdent
from .export import sha256, write_json


def select_samples(pool, maximum=8, spacing=5):
    """Quality-ranked selection per temporal bin; no identity labels are used."""
    if not pool:
        return []
    ordered = sorted(pool, key=lambda s: s['frame'])
    selected = []
    for group in np.array_split(np.arange(len(ordered)), min(maximum, len(ordered))):
        for index in sorted(group, key=lambda i: -ordered[i]['quality']):
            item = ordered[index]
            if all(abs(item['frame'] - old['frame']) >= spacing for old in selected):
                selected.append(item)
                break
    return sorted(selected, key=lambda s: s['frame'])


def run(cache_root, output, device, batch_size=4):
    import torch
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    encoder = None
    for clip in ('attack_clip', 'attack_clip2'):
        start = time.perf_counter()
        meta = json.loads((cache_root / clip / 'metadata.json').read_text())
        source = Path(meta['config']['run']) / 'frames.jsonl'
        video = Path(meta['video'])
        if meta['status'] != 'complete' or sha256(source) != meta['source_frames_sha256'] or sha256(video) != meta['video_sha256']:
            raise ValueError('Original inference/source hashes differ')
        frames = [json.loads(line) for line in source.read_text().splitlines()]
        pools = {}
        cap = cv2.VideoCapture(str(video))
        for record in frames:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Incomplete original video')
            for row in record['tracks']:
                tid = row['track_id']
                pools.setdefault(tid, [])
                x1, y1, x2, y2 = row['bbox_xyxy']
                # Hard reject image-edge truncation and extremely small crops.
                if min(x1, y1, frame.shape[1]-x2, frame.shape[0]-y2) < 2 or x2-x1 < 12 or y2-y1 < 35:
                    continue
                area = (x2-x1)*(y2-y1)
                overlap = max((max(0,min(x2,b['bbox_xyxy'][2])-max(x1,b['bbox_xyxy'][0])) *
                               max(0,min(y2,b['bbox_xyxy'][3])-max(y1,b['bbox_xyxy'][1])) / area
                               for b in record['tracks'] if b['track_id'] != tid), default=0)
                if overlap > 0.6:
                    continue
                box = [int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))]
                crop = frame[box[1]:box[3],box[0]:box[2]]
                sharpness = float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
                quality = row['det_confidence'] * np.sqrt(area) * (1-overlap) * min(1,sharpness/100)
                pools[tid].append(dict(frame=record['frame_idx'],bbox=box,quality=float(quality),
                                       overlap=overlap,sharpness=sharpness))
        cap.release()
        folder = output / clip
        (folder / 'crops').mkdir(parents=True)
        segments, requests = [], {}
        for sid, tid in enumerate(sorted(pools)):
            selected = select_samples(pools[tid])
            segments.append(dict(segment_id=sid, source_track_id=tid,
                                 sample_frames=[s['frame'] for s in selected]))
            for sample in selected:
                requests.setdefault(sample['frame'], []).append((sid, sample))
        crops, segment_ids = [], []
        cap = cv2.VideoCapture(str(video))
        for f in range(len(frames)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Incomplete original video')
            for sid, sample in requests.get(f, []):
                x1,y1,x2,y2=sample['bbox']
                crop=frame[y1:y2,x1:x2].copy()
                if not cv2.imwrite(str(folder/'crops'/f's{sid}-f{f}.jpg'),crop):
                    raise RuntimeError('Could not save selected crop')
                crops.append(crop)
                segment_ids.append(sid)
        cap.release()
        print(f'{clip}: encoding {len(crops)} selected crops from {len(pools)} original IDs on {device}',flush=True)
        run_meta = dict(meta, status='running', config=dict(meta['config'],device=device),
                        selection=dict(maximum=8,min_spacing=5,edge_margin_px=2,
                                       min_width=12,min_height=35,max_overlap=0.6),
                        purpose='Full-video gallery enrichment; no identity-review labels used',
                        versions={p:importlib.metadata.version(p) for p in ('torch','open-clip-torch','numpy','opencv-python')},
                        code_sha256=sha256(Path(__file__)))
        write_json(folder/'metadata.json',run_meta)
        try:
            if encoder is None:
                encoder=ClipReIdent(meta['checkpoint'],device,batch_size)
            if encoder.checkpoint_sha256 != meta['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            embeddings=encoder.encode(crops)
        except BaseException as error:
            run_meta.update(status='failed',error=repr(error))
            write_json(folder/'metadata.json',run_meta)
            raise
        np.savez_compressed(folder/'embeddings.npz',embeddings=embeddings,segment_ids=np.array(segment_ids))
        write_json(folder/'tracklets.json',segments)
        write_json(folder/'sample-selection.json',{str(tid):select_samples(pool) for tid,pool in pools.items()})
        run_meta.update(status='complete',crops=len(crops),total_seconds=time.perf_counter()-start)
        write_json(folder/'metadata.json',run_meta)
        print(f'{clip}: complete in {run_meta["total_seconds"]:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,default=Path('outputs/clip-reident-paper-v1'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    run(args.cache_root,args.output,args.device)


if __name__=='__main__':
    main()
