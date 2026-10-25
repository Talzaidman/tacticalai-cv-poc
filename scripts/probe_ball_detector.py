"""Development-only detector probe. Does not load reference coordinates."""
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import torch
from ultralytics import YOLO

from tacticalai.ball import BallConfig, PossessionEstimator, ball_candidates
from tacticalai.ball_cli import detect_ball, load_source
from tacticalai.calibration_cli import video_info
from tacticalai.export import sha256, write_json


SAMPLES = dict(attack_clip=[0, 20, 40, 60, 80, 110, 120, 140],
               attack_clip2=[0, 24, 48, 72, 96, 120, 144, 168])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('outputs/ball-baseline-v2'))
    parser.add_argument('--videos', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--ball-class', type=int, default=0)
    parser.add_argument('--conf', type=float, default=.15)
    parser.add_argument('--tile-fraction', type=float, help='Tile size as a fraction of source image height')
    parser.add_argument('--person-crops', action='store_true')
    parser.add_argument('--no-tiles', action='store_true')
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError('Detector probe requires a local checkpoint; downloading is not permitted')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', automatic_only=True, manual_labels_used=False,
                  scope='development detector probe, NOT complete clip tracking',
                  configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  clips={})
    write_json(args.output/'experiment.json', report)
    try:
        torch.set_num_threads(4)
        model = YOLO(str(args.model))
        if model.names.get(args.ball_class, '').lower().replace('_', ' ') not in ('basketball', 'ball', 'sports ball'):
            raise ValueError('Probe expects a ball class')
        if args.tile_fraction is not None and not 0 < args.tile_fraction <= 2:
            raise ValueError('Tile fraction must be in (0, 2]')
        report['model_sha256'] = sha256(args.model)
        started = time.perf_counter()
        for clip, indices in SAMPLES.items():
            video = args.videos/(clip+'.mp4')
            info = video_info(video)
            cached, provenance = load_source(args.source, clip, info)
            folder = args.output/clip
            folder.mkdir()
            cap = cv2.VideoCapture(str(video))
            frames = []
            try:
                for index in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Probe decode failed')
                    size = (round(info['height']*args.tile_fraction) if args.tile_fraction is not None
                            else 480 if info['width'] < 1000 else 960)
                    options = SimpleNamespace(tiles=not args.no_tiles, tile_size=size,
                        person_crops=args.person_crops,
                        ball_class=args.ball_class, device='cpu', imgsz=args.imgsz, conf=args.conf)
                    raw = detect_ball(model, image, options, cached[index]['tracks'])
                    candidates = ball_candidates(raw, (info['width'], info['height']),
                                                 BallConfig(detection_confidence=args.conf))
                    frame = copy.deepcopy(cached[index])
                    ball = dict(status='unknown', reason='sampled_probe_not_tracking', candidates=candidates)
                    frame.update(ball=ball, ball_raw_detections=raw,
                                 possession=PossessionEstimator(info['fps']).update(ball, []))
                    frames.append(frame)
                    print(clip, index, len(raw), 'raw detections', flush=True)
            finally:
                cap.release()
            path = folder/'frames.jsonl'
            path.write_text(''.join(json.dumps(f, allow_nan=False)+'\n' for f in frames), encoding='utf8')
            meta = dict(status='complete', video=info, source=provenance,
                        artifacts_sha256={'frames.jsonl': sha256(path)})
            write_json(folder/'metadata.json', meta)
            report['clips'][clip] = meta
        report.update(status='complete', processing_s=time.perf_counter()-started)
        write_json(args.output/'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        write_json(args.output/'experiment.json', report)
        raise


if __name__ == '__main__':
    main()
