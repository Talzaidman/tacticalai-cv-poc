"""CPU/Colab ball baseline, optional automatic person-cache integration."""
import argparse
from collections import Counter
import copy
import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np

from .ball import BallConfig, BallTracker, PossessionEstimator, tile_windows
from .calibration import project_track
from .calibration_cli import video_info
from .court import floor_lines
from .export import sha256, write_json
from .silhouette_presentation import render


COURT = dict(length_m=28., width_m=15., key_depth_m=5.8, key_width_m=4.9)
CLIPS = ('attack_clip', 'attack_clip2')


def load_source(source, clip, info):
    """Only consume explicit automatic results; never fall back to manual caches."""
    manifest = json.loads((source/'experiment.json').read_text())
    automatic = (manifest.get('manual_labels_used') is False or
                 manifest.get('automatic_only') is True or
                 manifest.get('automatic_flow', {}).get('manual_review_labels_used') is False)
    manual = (manifest.get('manual_labels_used') is True or
              manifest.get('automatic_flow', {}).get('manual_review_labels_used') is True)
    if manifest['status'] != 'complete' or not automatic or manual:
        raise ValueError('Source must be complete and explicitly automatic')
    folder = source/clip
    meta = json.loads((folder/'metadata.json').read_text())
    path = folder/'frames.jsonl'
    if (meta['status'] != 'complete' or meta['video'] != info or
            sha256(path) != meta['artifacts_sha256']['frames.jsonl']):
        raise ValueError('Source video, completion status or frame hash differs')
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    if len(frames) != info['frames']:
        raise ValueError('Source frame count differs')
    for index, frame in enumerate(frames):
        if frame['frame_idx'] != index or abs(frame['timestamp_s']-index/info['fps']) > 1e-6:
            raise ValueError('Source frame/timestamp mismatch')
        if ('manual' in frame['calibration']['method'] or
                'interpolated_anchors' in frame['calibration']['method']):
            raise ValueError('Manual calibration is not permitted')
        for row in frame['tracks']:
            if 'manual' in str(row.get('role_source', '')) or 'manual' in str(row.get('classification_source', '')):
                raise ValueError('Manual classifications are not permitted')
    return frames, dict(path=str(path.resolve()), sha256=sha256(path),
                        metadata_sha256=sha256(folder/'metadata.json'),
                        experiment_sha256=sha256(source/'experiment.json'))


def prepare_court(folder):
    from .auto_calibration_cli import KALI_REVISION
    from .kalicalib_reference import ReferenceDetector
    assets = {}
    for name, remote in (('model_resnet.py', 'kalicalib/model_resnet.py'),
                         ('model_test.pth', 'models/model_test.pth'),
                         ('model_challenge.pth', 'models/model_challenge.pth'),
                         ('LICENSE.txt', 'LICENSE.txt')):
        path = folder/name
        url = f'https://raw.githubusercontent.com/CEA-LIST/KaliCalib/{KALI_REVISION}/{remote}'
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix+'.download')
            print(f'Downloading {name}', flush=True)
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        assets[name] = dict(url=url, sha256=sha256(path))
    return ReferenceDetector(folder, 'model_challenge.pth'), assets


def detect_ball(model, image, args, rows=()):
    height, width = image.shape[:2]
    windows = [(0, 0, width, height)]
    if args.tiles and (width > args.tile_size or height > args.tile_size):
        windows.extend(tile_windows(width, height, args.tile_size))
    sources = ['full_frame'] + ['tile']*(len(windows)-1)
    if getattr(args, 'person_crops', False):
        for row in rows:
            if row.get('court_region') != 'inside':
                continue
            box = np.asarray(row['bbox_xyxy'], float)
            if (box.shape != (4,) or not np.isfinite(box).all() or
                    np.any(box[:2] < 0) or np.any(box[2:] > [width, height]) or
                    np.any(box[2:] <= box[:2])):
                raise ValueError('Invalid person crop box')
            x1, y1, x2, y2 = box
            h = y2-y1
            if h <= 0:
                raise ValueError('Invalid person crop box')
            cx, cy = (x1+x2)/2, (y1+y2)/2
            crop = (max(0, round(cx-.85*h)), max(0, round(cy-.85*h)),
                    min(width, round(cx+.85*h)), min(height, round(cy+.85*h)))
            if crop not in windows:
                windows.append(crop)
                sources.append('person_context_crop')
    detections = []
    for index, (x1, y1, x2, y2) in enumerate(windows):
        result = model.predict(image[y1:y2, x1:x2], classes=[args.ball_class], device=args.device,
                               imgsz=args.imgsz, conf=args.conf, iou=.7,
                               verbose=False, save=False)[0]
        for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            detections.append(dict(bbox_xyxy=[box[0]+x1, box[1]+y1, box[2]+x1, box[3]+y1],
                det_confidence=float(confidence), detection_source=sources[index],
                tile_xyxy=[x1, y1, x2, y2]))
    return detections


def fresh_people(model, image, tracker_path, clip, shot, device):
    result = model.track(image, persist=True, tracker=str(tracker_path.resolve()),
                         classes=[0], device=device, imgsz=640,
                         conf=.25, iou=.7, verbose=False, save=False)[0]
    rows = []
    if result.boxes.is_track:
        for box, confidence, track_id in zip(result.boxes.xyxy.cpu().tolist(),
                                            result.boxes.conf.cpu().tolist(), result.boxes.id.cpu().tolist()):
            track_id = int(track_id)
            label = f'S{shot}-T{track_id}'
            rows.append(dict(track_id=track_id, player_id=f'{clip}:{label}',
                identity_segment_id=f'{clip}:{label}', identity_status='fresh_botsort_hypothesis',
                bbox_xyxy=box, det_confidence=float(confidence), role=None, team_id=None,
                role_source='unknown', silhouette_xy=None, court_xy_m=None, court_region='unknown',
                display_label=label, display_caption=label+' / role unknown',
                draw_detection_box_fallback=True))
    return rows


def camera_frame(image, rows, detector, flow, index, interval, cut):
    if detector is None:
        return None, dict(method='unavailable', reason='no_automatic_camera_source')
    from .kalicalib_reference import estimate, extract_centroids
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = [r['bbox_xyxy'] for r in rows]
    if cut or index % interval == 0 or flow.matrix is None:
        evidence = estimate(extract_centroids(detector.heatmaps(image)), (image.shape[1], image.shape[0]))
        value = evidence['refined_image_to_court_h']
        matrix = np.asarray(value) if value is not None else None
        flow.reset(gray, matrix, boxes, COURT)
        quality = dict(method='automatic_kalicalib_unverified' if flow.matrix is not None else 'unavailable',
                       evidence=evidence, accuracy_metrics=None)
        return flow.matrix, quality
    return flow.advance(gray, boxes)


def flat_ball(clip, frame):
    ball, possession = frame['ball'], frame['possession']
    center = ball['center_image_xy'] or [None, None]
    predicted = ball['predicted_center_image_xy'] or [None, None]
    xy = possession['minimap_xy_m'] or [None, None]
    return dict(clip=clip, frame_idx=frame['frame_idx'], timestamp_s=frame['timestamp_s'],
        ball_status=ball['status'], ball_reason=ball['reason'], ball_track_id=ball['track_id'],
        ball_x_px=center[0], ball_y_px=center[1], predicted_x_px=predicted[0], predicted_y_px=predicted[1],
        detector_confidence=ball['det_confidence'], ball_court_x_m=None, ball_court_y_m=None,
        possession_status=possession['status'], controller_person_id=possession['controller_person_id'],
        controller_track_id=possession['controller_track_id'],
        controller_identity_segment_id=possession['controller_identity_segment_id'],
        controller_role=possession['controller_role'], association_score=possession['score'],
        possession_reason=possession['reason'], marker_x_m=xy[0], marker_y_m=xy[1],
        marker_position_source=possession['minimap_position_source'])


def encode(video, frames, target, info):
    import imageio_ffmpeg
    cap = cv2.VideoCapture(str(video))
    process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', str(info['fps']),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(target/'annotated.mp4')], stdin=subprocess.PIPE)
    lines = floor_lines(COURT)
    snapshots = set(np.linspace(0, len(frames)-1, 6, dtype=int).tolist())
    try:
        for frame in frames:
            ok, image = cap.read()
            if not ok:
                raise ValueError('Rendering source decode incomplete')
            out = render(image, frame, lines, color_mode='team')
            process.stdin.write(out.tobytes())
            if frame['frame_idx'] in snapshots and not cv2.imwrite(str(target/f"frame-{frame['frame_idx']:04d}.jpg"), out):
                raise ValueError('Snapshot write failed')
        if cap.read()[0]:
            raise ValueError('Rendering source has extra frames')
        process.stdin.close()
        if process.wait() != 0:
            raise ValueError('Encoding failed')
    finally:
        cap.release()
        if process.poll() is None:
            process.kill()
            process.wait()
    check = cv2.VideoCapture(str(target/'annotated.mp4'))
    decoded = 0
    try:
        while check.read()[0]:
            decoded += 1
    finally:
        check.release()
    if decoded != len(frames):
        raise ValueError('Annotated video frame count differs')
    return decoded


def rerender(args):
    if args.source is None:
        raise ValueError('--render-only requires --source')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', automatic_only=True, manual_labels_used=False,
                  source=str(args.source.resolve()), inference_rerun=False,
                  display_policy='Raw detections orange, or motion-supported detections for filtered sources; '
                                 'predictions gray X; controller unchanged',
                  source_manifest_sha256=sha256(args.source/'experiment.json'), clips={})
    write_json(args.output/'experiment.json', report)
    try:
        for clip in CLIPS:
            video = args.videos/(clip+'.mp4')
            info = video_info(video)
            frames, provenance = load_source(args.source, clip, info)
            if any('ball' not in frame or 'possession' not in frame for frame in frames):
                raise ValueError('Re-render source is missing ball/possession results')
            target = args.output/clip
            target.mkdir()
            shutil.copyfile(args.source/clip/'frames.jsonl', target/'frames.jsonl')
            meta = json.loads((args.source/clip/'metadata.json').read_text(encoding='utf8'))
            csv_path = args.source/clip/'ball.csv'
            if sha256(csv_path) != meta['artifacts_sha256']['ball.csv']:
                raise ValueError('Source ball CSV hash differs')
            shutil.copyfile(csv_path, target/'ball.csv')
            decoded = encode(video, frames, target, info)
            result = dict(status='complete', video=info, source=provenance,
                counts=meta['counts'], annotated_frames_decoded=decoded,
                artifacts_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json', result)
            report['clips'][clip] = result
        (args.output/'index.html').write_text(
            '<!doctype html><html><meta charset="utf-8"><title>Basketball YOLO26m - orange detections</title>'
            '<body style="background:#111;color:white;font:16px Arial">'
            '<p>ALL raw ball detections: large orange rings (may include false detections). '
            'Gray X: prediction only. Minimap ring: estimated controller, NOT measured ball floor position.</p>'+
            ''.join(f'<h2>{clip}</h2><video controls preload="metadata" style="width:min(100%,1100px)" '
                    f'src="{clip}/annotated.mp4"></video>' for clip in CLIPS)+'</body></html>', encoding='utf8')
        snapshots = args.output/'run_source'
        snapshots.mkdir()
        for name in ('ball_cli.py', 'silhouette_presentation.py'):
            shutil.copyfile(Path(__file__).with_name(name), snapshots/name)
        report.update(status='complete', source_sha256={p.name: sha256(p) for p in snapshots.iterdir()})
        write_json(args.output/'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        write_json(args.output/'experiment.json', report)
        raise


def run(args):
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from .auto_calibration import FloorTracker

    if args.source is not None and args.automatic_court:
        raise ValueError('Use cached source calibration OR fresh automatic court, not both')
    if min(args.imgsz, args.tile_size, args.court_interval, args.threads) <= 0:
        raise ValueError('Sizes, interval and threads must be positive')
    if ultralytics.__version__ != '8.4.158':
        raise RuntimeError('Install the pinned requirements.txt')
    config = BallConfig(detection_confidence=args.conf)
    videos = {clip: args.videos/(clip+'.mp4') for clip in CLIPS}
    infos = {clip: video_info(video) for clip, video in videos.items()}
    for info in infos.values():
        if min(info['width'], info['height'], info['frames']) <= 0 or not np.isfinite(info['fps']) or info['fps'] <= 0:
            raise ValueError('Invalid video dimensions, frame count or FPS')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='running', automatic_only=True, manual_labels_used=False,
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        ball_configuration=asdict(config), clips={}, accuracy_metrics=None,
        versions={p: importlib.metadata.version(p) for p in
                  ('torch', 'torchvision', 'ultralytics', 'opencv-python', 'numpy', 'scipy', 'imageio-ffmpeg')},
        limitations=[
            'Detector weights/classes are recorded; basketball performance on these clips is unvalidated.',
            'Controller is repeated body proximity, not verified hand contact, dribble or tactical possession.',
            'A nearby ball in flight may still create a false control estimate. Scores are not probabilities.',
            'Unknown ball/control is retained; predictions never establish possession.',
            'Orange map ring is the estimated controller floor position, never airborne ball XY.',
            'Fresh tracks, when used, are new run-local person hypotheses with unknown roles/teams.',
            'Automatic calibration and bbox footpoints are unverified; assumed court is 28 x 15 m.',
            'Offline processing; CPU/GPU timings are measured, not playback FPS.'], court_assumption=COURT)
    write_json(args.output/'experiment.json', report)
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        model = YOLO(args.model)
        ball_name = model.names.get(args.ball_class, '').lower().replace('_', ' ')
        if model.task != 'detect' or ball_name not in ('sports ball', 'basketball', 'ball'):
            raise ValueError('Selected detector class must be sports ball, basketball or ball')
        report['model'] = dict(path=str(Path(model.ckpt_path).resolve()), sha256=sha256(model.ckpt_path),
            class_id=args.ball_class, class_name=model.names[args.ball_class],
            license='Verify checkpoint/dataset license separately; Ultralytics AGPL-3.0 / enterprise also applies')
        if args.model_provenance:
            report['model']['provenance'] = json.loads(args.model_provenance.read_text(encoding='utf8'))
            expected = report['model']['provenance'].get('weights_sha256')
            if expected is not None and expected != report['model']['sha256']:
                raise ValueError('Checkpoint does not match its provenance SHA-256')
        detector, assets = prepare_court(args.court_models) if args.automatic_court else (None, None)
        report['court_assets'] = assets
        snapshots = args.output/'run_source'
        snapshots.mkdir()
        for name in ('ball.py', 'ball_cli.py', 'silhouette_presentation.py', 'calibration.py',
                     'auto_calibration.py', 'kalicalib_reference.py', 'court.py', 'export.py'):
            shutil.copyfile(Path(__file__).with_name(name), snapshots/name)
        report['source_sha256'] = {p.name: sha256(p) for p in snapshots.iterdir()}
        (args.output/'environment.txt').write_text(subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf8')
        for clip, video in videos.items():
            tick = time.perf_counter()
            target = args.output/clip
            target.mkdir()
            info = infos[clip]
            cached, provenance = load_source(args.source, clip, info) if args.source else (None, None)
            people = None
            tracker_config = None
            if cached is None:
                people = YOLO(args.person_model)
                if people.task != 'detect' or people.names.get(0) != 'person':
                    raise ValueError('Fresh person tracking requires a COCO person detector')
                report['person_model'] = dict(path=str(Path(people.ckpt_path).resolve()),
                                              sha256=sha256(people.ckpt_path))
                tracker_config = yaml.safe_load((Path(ultralytics.__file__).parent/'cfg'/'trackers'/'botsort.yaml').read_text(encoding='utf8'))
                tracker_config.update(with_reid=False, gmc_method='sparseOptFlow')
                tracker_path = target/'person_tracker.yaml'
                tracker_path.write_text(yaml.safe_dump(tracker_config))
            tracker = BallTracker((info['width'], info['height']), info['fps'], config)
            possession = PossessionEstimator(info['fps'], config)
            flow = FloorTracker(args.court_interval-1)
            cap = cv2.VideoCapture(str(video))
            frames, previous, previous_shot, shot = [], None, None, 0
            timings = Counter()
            try:
                for index in range(info['frames']):
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Video ended before declared frame count')
                    thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 36)).astype(float)
                    cut = index == 0 or (previous is not None and np.abs(thumb-previous).mean() > 45)
                    previous = thumb
                    start = time.perf_counter()
                    if cached is not None:
                        frame = copy.deepcopy(cached[index])
                        source_shot = frame['calibration'].get('shot')
                        cut = cut or (index > 0 and source_shot != previous_shot)
                        previous_shot = source_shot
                    else:
                        if cut:
                            shot += 1
                            if index:
                                for person_tracker in people.predictor.trackers:
                                    person_tracker.reset()
                        rows = fresh_people(people, image, tracker_path, clip, shot, args.device)
                        timings['person_inference_s'] += time.perf_counter()-start
                        start = time.perf_counter()
                        matrix, quality = camera_frame(image, rows, detector, flow, index, args.court_interval, cut)
                        quality['shot'] = shot
                        frame = dict(frame_idx=index, timestamp_s=index/info['fps'], calibration=quality,
                            image_to_court_h=matrix.tolist() if matrix is not None else None,
                            tracks=[project_track(r, matrix, quality, COURT, (info['width'], info['height'])) for r in rows])
                        timings['camera_s'] += time.perf_counter()-start
                    start = time.perf_counter()
                    raw = detect_ball(model, image, args, frame['tracks'])
                    timings['ball_detection_s'] += time.perf_counter()-start
                    start = time.perf_counter()
                    ball = tracker.update(index, raw, cut=cut)
                    frame.update(ball=ball, possession=possession.update(ball, frame['tracks'], cut=cut),
                                 ball_raw_detections=raw, ball_camera_cut_reset=bool(cut))
                    timings['ball_association_s'] += time.perf_counter()-start
                    frames.append(frame)
                    if index % 25 == 0:
                        print(f"{clip} {index}/{info['frames']}: {ball['status']}, controller={frame['possession']['controller_person_id']}", flush=True)
                if cap.read()[0]:
                    raise ValueError('Source contains extra frames')
            finally:
                cap.release()
            with (target/'frames.jsonl').open('w', encoding='utf8') as stream:
                for frame in frames:
                    stream.write(json.dumps(frame, allow_nan=False)+'\n')
            flat = [flat_ball(clip, f) for f in frames]
            with (target/'ball.csv').open('w', newline='', encoding='utf8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
            if cached is not None:
                for before, after in zip(cached, frames):
                    if any(after[k] != v for k, v in before.items() if k not in
                           ('ball', 'possession', 'ball_raw_detections', 'ball_camera_cut_reset')):
                        raise ValueError('Original person/camera observation changed')
            start = time.perf_counter()
            decoded = encode(video, frames, target, info)
            timings['render_encode_s'] = time.perf_counter()-start
            counts = dict(ball_status=dict(Counter(f['ball']['status'] for f in frames)),
                          possession_status=dict(Counter(f['possession']['status'] for f in frames)),
                          minimap_marker_frames=sum(f['possession']['minimap_xy_m'] is not None for f in frames),
                          calibrated_frames=sum(f['image_to_court_h'] is not None for f in frames))
            meta = dict(status='complete', video=info, source=provenance,
                person_source='existing_automatic_cache' if cached is not None else 'fresh_botsort_no_reid',
                person_tracker=tracker_config, person_reid=False if cached is None else 'inherited_from_source',
                person_gmc='sparseOptFlow' if cached is None else 'inherited_from_source',
                counts=counts, timings_s=dict(timings), total_processing_s=time.perf_counter()-tick,
                annotated_frames_decoded=decoded, accuracy_metrics=None,
                artifacts_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json', meta)
            report['clips'][clip] = meta
            write_json(args.output/'experiment.json', report)
            print(clip, counts, flush=True)
        (args.output/'index.html').write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Ball baseline</title>'
            '<body style="background:#111;color:white;font:16px Arial">'
            '<p>Experimental ball tracking. Orange ring = estimated controller, NOT ball floor XY. '
            'Body proximity is not verified possession. Offline processing; unknowns retained.</p>'+
            ''.join(f'<h2>{clip}</h2><video controls preload="metadata" style="width:min(100%,1100px)" '
                    f'src="{clip}/annotated.mp4"></video>' for clip in CLIPS)+'</body></html>', encoding='utf8')
        report.update(status='complete', total_processing_s=time.perf_counter()-started)
        write_json(args.output/'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        write_json(args.output/'experiment.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, default=Path('data/raw'))
    parser.add_argument('--source', type=Path, help='Completed automatic person/court presentation root')
    parser.add_argument('--render-only', action='store_true', help='Re-render cached ball results without inference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='yolo26s.pt')
    parser.add_argument('--ball-class', type=int, default=32)
    parser.add_argument('--model-provenance', type=Path, help='Model URL/revision/license record, saved in manifest')
    parser.add_argument('--person-model', default='yolo26s.pt', help='COCO detector for fresh person tracking only')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--conf', type=float, default=.15)
    parser.add_argument('--tiles', action='store_true', help='Full-frame plus overlapping 960px crops')
    parser.add_argument('--tile-size', type=int, default=960)
    parser.add_argument('--person-crops', action='store_true',
                        help='Experimental extra context crops around automatically projected inside-court people')
    parser.add_argument('--automatic-court', action='store_true', help='Fresh KaliCalib + floor flow; downloads pinned assets')
    parser.add_argument('--court-models', type=Path, default=Path('models/kalicalib'))
    parser.add_argument('--court-interval', type=int, default=5)
    args = parser.parse_args()
    if args.render_only:
        rerender(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
