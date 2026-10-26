"""Local Roboflow basketball inference, embedded into the standalone notebook."""
import json
import math
import os
import sys
import time
from importlib import metadata

import cv2


BALL_CLASSES = {'ball', 'ball-in-basket'}


def model_runtime(model):
    """Validate the real detector, including Roboflow's generic adapter."""
    backend = getattr(model, '_model', model)
    architecture = type(backend).__name__
    if 'rfdetr' not in architecture.lower().replace('_', '').replace('-', ''):
        raise RuntimeError(f'Expected RF-DETR backend; got {architecture}. Check RF_MODEL_ID.')
    session = getattr(backend, '_session', None)
    if session is None:
        session = getattr(backend, 'session', None)
    providers = list(session.get_providers()) if session is not None else []
    device = str(getattr(backend, '_device', getattr(backend, 'device', 'unknown')))
    if session is not None:
        if 'CUDAExecutionProvider' not in providers:
            raise RuntimeError(f'RF-DETR fell back to CPU: {providers}. Repair the isolated CUDA environment.')
    elif not device.startswith('cuda'):
        raise RuntimeError(f'Cannot verify RF-DETR GPU execution (device={device}).')
    return dict(architecture_handler=type(model).__name__, backend_class=architecture,
                backend_module=type(backend).__module__, execution_providers=providers,
                device=device)


def ball_candidates(predictions, views):
    if len(predictions) != len(views):
        raise ValueError('Inference returned a different number of views')
    detections, names = [], set()
    for response, (ox, oy, _, _) in zip(predictions, views):
        for d in response.predictions:
            name = d.class_name.strip().lower()
            names.add(name)
            if name in BALL_CLASSES:
                detections.append(dict(x=float(d.x+ox), y=float(d.y+oy),
                    r=float(max(d.width, d.height)/2), conf=float(d.confidence), cls=name))
    return detections, names


def merge(detections):
    kept = []
    for d in sorted(detections, key=lambda d: -d['conf']):
        if all(math.hypot(d['x']-k['x'], d['y']-k['y']) > max(4, d['r']) for k in kept):
            kept.append(d)
    return kept


# One change at a time around the existing full-frame/2x2 baselines.
RF_VARIANTS = {
    'RF-DETR-full': dict(mode='full', candidate_confidence=0.10, tracker_min_conf=0.25),
    'RF-DETR-tiles': dict(mode='tiles', candidate_confidence=0.10, tracker_min_conf=0.25),
    'RF-DETR-tight': dict(mode='tight', candidate_confidence=0.10, tracker_min_conf=0.25),
    'RF-DETR-tight-low': dict(mode='tight', candidate_confidence=0.05, tracker_min_conf=0.25),
    'RF-DETR-tight-track010': dict(mode='tight', candidate_confidence=0.05, tracker_min_conf=0.10),
    'RF-DETR-tiles-track040': dict(mode='tiles', candidate_confidence=0.10, tracker_min_conf=0.40),
}


def crop_views(width, height, mode):
    if mode == 'full':
        return [(0, 0, width, height)]
    grid, fraction = (2, 0.6) if mode == 'tiles' else (3, 0.4)
    cw, ch = max(1, round(width*fraction)), max(1, round(height*fraction))
    return [(round(x*(width-cw)/(grid-1)), round(y*(height-ch)/(grid-1)), cw, ch)
            for y in range(grid) for x in range(grid)]


def detect_clip(model, frames, sweep=False, variant_labels=None):
    height, width = frames[0].shape[:2]
    variants = RF_VARIANTS if sweep else {k: RF_VARIANTS[k] for k in ('RF-DETR-full', 'RF-DETR-tiles')}
    if variant_labels is not None:
        if not variant_labels or any(label not in RF_VARIANTS for label in variant_labels):
            raise ValueError('Select at least one known RF-DETR variant.')
        variants = {label: RF_VARIANTS[label] for label in variant_labels}
    results = {label: [] for label in variants}
    seconds = {label: 0.0 for label in variants}
    names = set()
    full = crop_views(width, height, 'full')
    tiles = crop_views(width, height, 'tiles')
    tight = crop_views(width, height, 'tight')
    for i, image in enumerate(frames):
        cache = {}

        def infer_views(key, views, threshold):
            t0 = time.perf_counter()
            crops = [image[y:y+h, x:x+w] for x, y, w, h in views]
            # Keep crop batches small enough for a T4.
            candidates = []
            for start in range(0, len(crops), 4):
                stop = start+4
                ds, found_names = ball_candidates(
                    model.infer(crops[start:stop], confidence=threshold), views[start:stop])
                candidates.extend(ds)
                names.update(found_names)
            cache[key] = (candidates, time.perf_counter()-t0)

        modes = {config['mode'] for config in variants.values()}
        if modes & {'full', 'tiles'}:
            infer_views('full', full, 0.10)
        if 'tiles' in modes:
            infer_views('tiles', tiles, 0.10)
        if 'tight' in modes:
            infer_views('full-low', full, 0.05)
            infer_views('tight', tight, 0.05)
        for label, config in variants.items():
            keys = {'full': ['full'], 'tiles': ['full', 'tiles'],
                    'tight': ['full-low', 'tight']}[config['mode']]
            t0 = time.perf_counter()
            ds = [d for key in keys for d in cache[key][0]
                  if d['conf'] >= config['candidate_confidence']]
            # Same merge order as the original tiled baseline.
            if config['mode'] != 'full':
                ds = merge([d for d in cache[keys[0]][0]
                            if d['conf'] >= config['candidate_confidence']]) + [
                    d for d in cache[keys[1]][0] if d['conf'] >= config['candidate_confidence']]
            results[label].append(merge(ds))
            seconds[label] += sum(cache[key][1] for key in keys) + time.perf_counter()-t0
        if i % 25 == 0:
            print(f'RF-DETR: {i+1}/{len(frames)} frames', flush=True)
    return dict(variants={label: dict(detections=results[label], seconds=seconds[label],
                                    config=dict(config,
                                        inference_confidence=0.05 if config['mode']=='tight' else 0.10,
                                        crop_fraction={'full': 1.0, 'tiles': 0.6, 'tight': 0.4}[config['mode']],
                                        tile_grid={'full': [1, 1], 'tiles': [2, 2], 'tight': [3, 3]}[config['mode']]))
                          for label, config in variants.items()},
                class_names=sorted(names))


def main():
    # Load CUDA 12/cuDNN from the isolated environment before creating any session.
    import torch
    import onnxruntime as ort
    if not torch.cuda.is_available():
        raise RuntimeError('RF-DETR environment cannot access the Colab GPU.')
    ort.preload_dlls(directory='')
    from inference import get_model
    model_id = os.environ['RF_MODEL_ID']
    sweep = os.environ.get('RF_SWEEP', 'false').lower() == 'true'
    selection = os.environ.get('RF_VARIANTS', '').strip()
    variant_labels = selection.split(',') if selection else None
    model = get_model(model_id=model_id, api_key=os.environ['ROBOFLOW_API_KEY'])
    runtime = model_runtime(model)
    print('Requested model:', model_id, '| Runtime:', runtime, flush=True)
    versions = {}
    for package in ('inference-gpu', 'inference-models', 'torch', 'torchvision', 'numpy',
                    'opencv-python', 'onnxruntime-gpu', 'onnxruntime', 'nvidia-cublas-cu12',
                    'nvidia-cudnn-cu12', 'nvidia-cuda-runtime-cu12'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    for video, out in zip(sys.argv[1::2], sys.argv[2::2]):
        # Exercise full-frame AND tiled inference before decoding/processing the full clip.
        cap = cv2.VideoCapture(video)
        smoke_frames = []
        for _ in range(3):
            ok, image = cap.read()
            if not ok:
                break
            smoke_frames.append(image)
        cap.release()
        if not smoke_frames:
            raise ValueError(f'Could not decode {video}')
        smoke = detect_clip(model, smoke_frames, sweep=sweep, variant_labels=variant_labels)
        runtime = model_runtime(model)
        smoke['provenance'] = dict(model_id=model_id, **runtime, packages=versions,
                                   smoke_frames=len(smoke_frames), cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0))
        with open(out + '.smoke.json', 'w') as handle:
            json.dump(smoke, handle)
        print('Three-frame full/tiled smoke test passed; starting full clip.', flush=True)
        cap = cv2.VideoCapture(video)
        frames = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frames.append(image)
        cap.release()
        if not frames:
            raise ValueError(f'Could not decode {video}')
        data = detect_clip(model, frames, sweep=sweep, variant_labels=variant_labels)
        data['provenance'] = dict(model_id=model_id, **runtime,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), smoke_frames=len(smoke_frames),
            packages=versions, sweep=sweep, selected_variants=variant_labels,
            full_frame=True, local_inference=True,
            timing_shared_views=True,
            timing='preprocessing/inference/extraction; excludes decode/model load/tracking/export')
        with open(out, 'w') as handle:
            json.dump(data, handle)
        print('Observed classes:', data['class_names'], flush=True)


if __name__ == '__main__':
    main()
