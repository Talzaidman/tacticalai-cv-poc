"""WASB inference sweep; embedded in the standalone Colab notebook by its builder.

No labels or manually selected ball locations enter inference. All variants use
the same checkpoint, consecutive three-frame windows and 512x288 network input.
"""
import math
import time

import cv2
import numpy as np


WASB_VARIANTS = {
    'WASB-base': dict(mode='full', threshold=0.50),
    'WASB-full-035': dict(mode='full', threshold=0.35),
    'WASB-full-020': dict(mode='full', threshold=0.20),
    'WASB-full-010': dict(mode='full', threshold=0.10),
    'WASB-tiles-035': dict(mode='full+tiles', threshold=0.35),
    'WASB-tiles-020': dict(mode='full+tiles', threshold=0.20),
}

WASB_LOWRES_VARIANTS = {
    'WASB-tiles-020': dict(mode='full+tiles', threshold=0.20, crop_fraction=0.6),
    'WASB-tight-020': dict(mode='full+tiles', threshold=0.20, crop_fraction=0.4),
    'WASB-tight-010': dict(mode='full+tiles', threshold=0.10, crop_fraction=0.4),
    'WASB-tight-cubic': dict(mode='full+tiles', threshold=0.20, crop_fraction=0.4,
                            interpolation='cubic'),
    'WASB-tight-clahe': dict(mode='full+tiles', threshold=0.20, crop_fraction=0.4,
                            preprocessing='clahe'),
}


def wasb_views(width, height, tiled=False, crop_fraction=0.6):
    if not 0 < crop_fraction <= 1:
        raise ValueError('crop_fraction must be in (0, 1]')
    views = [(0, 0, width, height)]
    if tiled:
        cw, ch = max(1, round(width * crop_fraction)), max(1, round(height * crop_fraction))
        # At least 20% overlap. 60% crops give 2x2; 40% crops give 3x3.
        nx = math.ceil((width-cw)/max(1, cw*0.8))+1
        ny = math.ceil((height-ch)/max(1, ch*0.8))+1
        xs = np.linspace(0, width-cw, nx).round().astype(int)
        ys = np.linspace(0, height-ch, ny).round().astype(int)
        views += [(int(x), int(y), cw, ch) for y in ys for x in xs]
    return list(dict.fromkeys(views))


def wasb_preprocess(image, method='none'):
    if method == 'none':
        return image
    if method != 'clahe':
        raise ValueError(f'Unknown preprocessing: {method}')
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def wasb_plan(width, height, variants):
    """Share inference only when view, preprocessing and interpolation match."""
    plan = {}
    for name, config in variants.items():
        views = wasb_views(width, height, config['mode'] == 'full+tiles',
                           config.get('crop_fraction', 0.6))
        for view in views:
            key = (view, config.get('preprocessing', 'none'), config.get('interpolation', 'linear'))
            plan.setdefault(key, []).append(name)
    return plan


def wasb_components(hm, threshold, view):
    """Map connected heatmap components back to source-image coordinates."""
    ox, oy, width, height = view
    scale = min(512 / width, 288 / height)
    tx, ty = (512-scale*width)/2, (288-scale*height)/2
    count, labels = cv2.connectedComponents((hm > threshold).astype(np.uint8))
    found = []
    for lab in range(1, count):
        ys, xs = np.nonzero(labels == lab)
        weights = hm[ys, xs]
        x = ((xs*weights).sum()/weights.sum()-tx)/scale
        y = ((ys*weights).sum()/weights.sum()-ty)/scale
        if 0 <= x < width and 0 <= y < height:
            found.append(dict(x=float(x+ox), y=float(y+oy),
                              r=float(max(np.ptp(xs)+1, np.ptp(ys)+1)/scale/2),
                              conf=float(weights.max())))
    return found


def wasb_merge(detections, radius):
    kept = []
    for d in sorted(detections, key=lambda d: -d['conf']):
        if all(math.hypot(d['x']-k['x'], d['y']-k['y']) > radius for k in kept):
            kept.append(d)
    return kept


def wasb_sweep(model, frames, batch=4, variants=None):
    """Run each view once, reuse its heatmaps across threshold variants.

    Report inference/preprocessing/transfer time shared by threshold variants,
    plus each variant's extraction and merge time. Excludes decode/tracking/export.
    Small batches limit Colab T4 memory; no full-clip heatmap cache is retained.
    """
    import torch
    variants = WASB_VARIANTS if variants is None else variants
    if len(frames) < 3:
        raise ValueError('WASB needs at least three frames')
    height, width = frames[0].shape[:2]
    found = {name: [[] for _ in frames] for name in variants}
    extraction = {name: 0.0 for name in variants}
    inference_seconds = {name: 0.0 for name in variants}
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    plan = wasb_plan(width, height, variants)
    for vi, ((view, preprocessing, interpolation), names) in enumerate(plan.items(), 1):
        print(f'View {vi}/{len(plan)}: {view}, {preprocessing}, {interpolation}', flush=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ox, oy, cw, ch = view
        scale = min(512/cw, 288/ch)
        matrix = np.float32([[scale, 0, (512-scale*cw)/2], [0, scale, (288-scale*ch)/2]])
        inputs = []
        for f in frames:
            image = wasb_preprocess(f[oy:oy+ch, ox:ox+cw], preprocessing)
            image = cv2.warpAffine(image, matrix, (512, 288), flags={
                'linear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC}[interpolation])
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
            inputs.append(torch.from_numpy(((rgb-mean)/std).transpose(2, 0, 1)))
        elapsed = time.perf_counter()-t0
        active = {name: variants[name] for name in names}
        with torch.inference_mode():
            for start in range(0, len(frames)-2, batch):
                indices = list(range(start, min(start+batch, len(frames)-2)))
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                inp = torch.stack([torch.cat(inputs[i:i+3]) for i in indices]).cuda()
                maps = model(inp)[0].sigmoid().cpu().numpy()
                torch.cuda.synchronize()
                elapsed += time.perf_counter()-t0
                for name, config in active.items():
                    t0 = time.perf_counter()
                    for i, heatmaps in zip(indices, maps):
                        for k, hm in enumerate(heatmaps):
                            found[name][i+k].extend(wasb_components(hm, config['threshold'], view))
                    extraction[name] += time.perf_counter()-t0
        for name in names:
            inference_seconds[name] += elapsed
        del inputs
    results = {}
    for name, config in variants.items():
        t0 = time.perf_counter()
        # Preserve original baseline merge radius; use crop scale for tiled variants.
        scale = min(512/width, 288/height)
        radius = 4/scale * (config.get('crop_fraction', 0.6) if config['mode'] == 'full+tiles' else 1)
        detections = [wasb_merge(d, radius) for d in found[name]]
        extraction[name] += time.perf_counter()-t0
        results[name] = (detections, inference_seconds[name]+extraction[name])
    return results
