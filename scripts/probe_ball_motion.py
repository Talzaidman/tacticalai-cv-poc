"""Offline background rejection or motion proposals using local source evidence."""
import argparse
import copy
import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import cv2

from tacticalai.ball_motion import MotionConfig, analyze_motion
from tacticalai.ball import BallConfig, BallTracker, PossessionEstimator
from tacticalai.ball_cli import CLIPS, encode, flat_ball, load_source
from tacticalai.calibration_cli import video_info
from tacticalai.export import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/ball-basketball-v1'))
    parser.add_argument('--videos', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('filter', 'proposals'), default='filter')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = MotionConfig()
    report = dict(status='running', automatic_only=True, manual_labels_used=False,
                  config=asdict(config), method='experimental_color_camera_motion_proposals',
                  mode=args.mode, offline_future_frames_used=True,
                  threshold_selection='development-only visual reference feedback; no labels read during inference',
                  source_manifest_sha256=sha256(args.source/'experiment.json'),
                  limitations=['May reject a held/stationary ball; camera residual is not a ball classifier.',
                               'Cannot recover detector misses. No measured ball floor position.',
                               'Same-video development tuning, not independent-video validation.'],
                  versions={name: importlib.metadata.version(name) for name in ('numpy', 'opencv-python')},
                  clips={})
    report['method'] = ('camera_compensated_detector_background_rejection' if args.mode == 'filter'
                        else 'experimental_color_camera_motion_proposals')
    write_json(args.output/'experiment.json', report)
    started = time.perf_counter()
    try:
        source_manifest = json.loads((args.source/'experiment.json').read_text(encoding='utf8'))
        ball_config = BallConfig(**source_manifest['ball_configuration'])
        report['ball_configuration'] = asdict(ball_config)
        report['detector'] = source_manifest['model']
        report['detector_configuration'] = source_manifest['configuration']
        snapshots = args.output/'run_source'
        snapshots.mkdir()
        files = [Path(__file__)] + [
            Path(__file__).resolve().parents[1]/'src'/'tacticalai'/name
            for name in ('ball_motion.py', 'ball.py', 'ball_cli.py', 'silhouette_presentation.py')]
        for path in files:
            shutil.copyfile(path, snapshots/path.name)
        report['source_sha256'] = {p.name: sha256(p) for p in snapshots.iterdir()}
        for clip in CLIPS:
            video = args.videos/(clip+'.mp4')
            info = video_info(video)
            frames, provenance = load_source(args.source, clip, info)
            images = []
            cuts = [f['ball_camera_cut_reset'] for f in frames]
            cap = cv2.VideoCapture(str(video))
            try:
                for _ in frames:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Motion source decode failed')
                    images.append(cv2.resize(image, (config.working_width,
                        round(info['height']*config.working_width/info['width']))))
                if cap.read()[0]:
                    raise ValueError('Extra source video frames')
            finally:
                cap.release()
            evidence = analyze_motion(images, cuts, (info['width'], info['height']), config,
                                      [f['ball_raw_detections'] for f in frames])
            enriched = copy.deepcopy(frames)
            tracker = BallTracker((info['width'], info['height']), info['fps'], ball_config)
            controller = PossessionEstimator(info['fps'], ball_config)
            for index, (frame, motion) in enumerate(zip(enriched, evidence)):
                if args.mode == 'filter':
                    frame['ball_motion_accepted_detections'] = motion['accepted_detections']
                    frame['ball'] = tracker.update(index, motion['accepted_detections'], cut=cuts[index])
                    frame['possession'] = controller.update(frame['ball'], frame['tracks'], cut=cuts[index])
                else:
                    frame['ball_raw_detections'] = motion['candidates']
                    frame['ball'] = dict(status='unknown', reason='motion_proposals_not_tracking',
                                         candidates=motion['candidates'])
                    frame['possession'] = controller.update(frame['ball'], [], cut=cuts[index])
                frame['ball_motion'] = motion
            target = args.output/clip
            target.mkdir()
            path = target/'frames.jsonl'
            path.write_text(''.join(json.dumps(f, allow_nan=False)+'\n' for f in enriched), encoding='utf8')
            decoded = None
            if args.mode == 'filter':
                flat = [flat_ball(clip, frame) for frame in enriched]
                with (target/'ball.csv').open('w', newline='', encoding='utf8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
                    writer.writeheader()
                    writer.writerows(flat)
                decoded = encode(video, enriched, target, info)
            parent_meta = json.loads((args.source/clip/'metadata.json').read_text(encoding='utf8'))
            meta = dict(status='complete', video=info, source=provenance,
                        person_source='unchanged_automatic_source',
                        person_reid=parent_meta['person_reid'], person_gmc=parent_meta['person_gmc'],
                        annotated_frames_decoded=decoded,
                        counts=dict(raw_detections=sum(len(f['ball_raw_detections']) for f in enriched),
                                    motion_supported=sum(len(m['accepted_detections']) for m in evidence),
                                    observed_frames=sum(f['ball']['status'] == 'observed' for f in enriched),
                                    controller_frames=sum(f['possession']['controller_person_id'] is not None
                                                          for f in enriched)),
                        artifacts_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json', meta)
            report['clips'][clip] = meta
            print(clip, meta['counts'], flush=True)
        if args.mode == 'filter':
            (args.output/'index.html').write_text(
                '<!doctype html><html lang="en"><meta charset="utf-8"><title>Ball motion filter</title>'
                '<body><p>Experimental background rejection, not reliable ball tracking. '
                'Orange: motion-supported detector candidates, possibly false. '
                'Gray X: prediction only. Map ring: estimated controller floor position, not ball XY. '
                'Offline future-frame processing; same-video development feedback used.</p>'+
                ''.join(f'<h2>{clip}</h2><video controls preload="metadata" style="width:min(100%,1100px)" '
                        f'src="{clip}/annotated.mp4"></video>' for clip in CLIPS)+'</body></html>', encoding='utf8')
        report.update(status='complete', processing_s=time.perf_counter()-started)
        write_json(args.output/'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        write_json(args.output/'experiment.json', report)
        raise


if __name__ == '__main__':
    main()
