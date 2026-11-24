"""Compare raw person detections with confirmed tracks on an automatic court source.

This diagnostic intentionally stops before segmentation, roles, teams, or ball logic.
It records detector evidence that Ultralytics tracking output normally discards, so a
missing person can be attributed to detection, court gating, or track confirmation.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import cv2
import numpy as np
import yaml

from .ball_cli import load_source
from .calibration_cli import video_info
from .court_gate import court_gate
from .export import json_default, sha256, write_json
from .unlinked_people import box_matches
from .video_set import resolve_clips


def configured_tracker(defaults, *, high, low, new, minimum_length, buffer_frames,
                       with_reid, reid_model):
    """Return validated TrackTrack settings without mutating package defaults."""
    values = dict(defaults)
    if not 0 <= low <= high <= 1 or not 0 <= new <= 1:
        raise ValueError('Tracker thresholds must satisfy 0 <= low <= high <= 1 and 0 <= new <= 1')
    if minimum_length < 1 or buffer_frames < 1:
        raise ValueError('Tracker history and buffer lengths must be positive')
    values.update(track_high_thresh=float(high), track_low_thresh=float(low),
                  new_track_thresh=float(new), min_track_len=int(minimum_length),
                  track_buffer=int(buffer_frames), with_reid=bool(with_reid),
                  gmc_method='sparseOptFlow')
    if with_reid:
        values['model'] = str(reid_model)
    return values


def result_rows(result, tracked=False):
    boxes = result.boxes
    if boxes is None or (tracked and not boxes.is_track):
        return []
    ids = boxes.id.cpu().tolist() if tracked else [None] * len(boxes)
    return [dict(bbox_xyxy=[float(x) for x in box], det_confidence=float(conf),
                 track_id=int(track_id) if track_id is not None else None)
            for box, conf, track_id in zip(boxes.xyxy.cpu().tolist(),
                                           boxes.conf.cpu().tolist(), ids)]


def analyze_frame(raw_rows, track_rows, source_frame, image_size, match_iou=.3):
    """Attach automatic court evidence and identify detections withheld by tracking."""
    matrix = source_frame.get('image_to_court_h')
    quality = source_frame['calibration']
    def enrich(row):
        gate = court_gate(row['bbox_xyxy'], matrix, quality, image_size,
                          margin_m=1.5, minimum_overlap=.05)
        floor = gate.get('gate_floor_xy_m')
        # For this diagnostic, body overlap with the broadcast-view court is too
        # permissive: spectators immediately behind the baseline overlap it. Use
        # the automatic bbox-bottom floor estimate and retain only a small margin.
        diagnostic_keep = (floor is None or
            (-1.5 <= floor[0] <= 29.5 and -1.5 <= floor[1] <= 16.5))
        return dict(row, **gate, diagnostic_court_keep=bool(diagnostic_keep))
    detections = []
    for row in raw_rows:
        detections.append(enrich(row))
    tracks = [enrich(row) for row in track_rows]
    on_court_indices = [i for i, row in enumerate(detections) if row['diagnostic_court_keep']]
    on_court = [detections[i] for i in on_court_indices]
    on_court_tracks = [row for row in tracks if row['diagnostic_court_keep']]
    matches = box_matches([row['bbox_xyxy'] for row in on_court_tracks],
                          [row['bbox_xyxy'] for row in on_court], match_iou)
    matched_local = {j for _, j, _ in matches}
    for local_index, detection in enumerate(on_court):
        detection['track_match_status'] = ('confirmed_track' if local_index in matched_local
                                           else 'detected_not_confirmed')
    for row in detections:
        row.setdefault('track_match_status', 'outside_court_gate')
    return detections, tracks, dict(
        raw_detections=len(detections),
        on_court_detections=len(on_court),
        raw_confirmed_tracks=len(tracks),
        confirmed_tracks=len(on_court_tracks),
        on_court_detections_matched_to_track=len(matched_local),
        on_court_detections_not_confirmed=len(on_court)-len(matched_local),
        outside_court_gate=len(detections)-len(on_court),
    )


def draw_label(image, label, point, color, scale=.45):
    x, y = point
    cv2.putText(image, label, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def render(image, record):
    out = cv2.resize(image, (1280, 720))
    sx, sy = 1280/image.shape[1], 720/image.shape[0]
    for row in record['raw_detections']:
        if not row['diagnostic_court_keep']:
            continue
        unmatched = row['track_match_status'] == 'detected_not_confirmed'
        color = (0, 165, 255) if unmatched else (70, 210, 80)
        x1, y1, x2, y2 = row['bbox_xyxy']
        p1, p2 = (round(x1*sx), round(y1*sy)), (round(x2*sx), round(y2*sy))
        cv2.rectangle(out, p1, p2, color, 2 if unmatched else 1, cv2.LINE_AA)
        if unmatched:
            draw_label(out, f"det {row['det_confidence']:.2f} / no track",
                       (p1[0], max(18, p1[1]-4)), color)
    for row in record['tracks']:
        if not row['diagnostic_court_keep']:
            continue
        x1, y1, x2, y2 = row['bbox_xyxy']
        p1, p2 = (round(x1*sx), round(y1*sy)), (round(x2*sx), round(y2*sy))
        cv2.rectangle(out, p1, p2, (245, 210, 60), 2, cv2.LINE_AA)
        draw_label(out, f"T{row['track_id']} {row['det_confidence']:.2f}",
                   (p1[0], max(18, p1[1]-4)), (245, 210, 60))
    counts = record['counts']
    cv2.rectangle(out, (0, 670), (1280, 720), (20, 20, 20), -1)
    draw_label(out, 'green = raw on-court detection | orange = detected but no confirmed track | cyan = track',
               (16, 691), (230, 230, 230), .48)
    draw_label(out, (f"on-court detections {counts['on_court_detections']} | "
                     f"confirmed tracks {counts['confirmed_tracks']} | "
                     f"withheld {counts['on_court_detections_not_confirmed']}"),
               (16, 713), (230, 230, 230), .48)
    return out


def run(args):
    import imageio_ffmpeg
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    report = dict(status='running', automatic_only=True, manual_labels_used=False,
        purpose='person detection versus track-confirmation diagnostic',
        source=str(args.source.resolve()), model=str(args.model), model_sha256=sha256(args.model),
        settings=dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            track_high_thresh=args.track_high_thresh, track_low_thresh=args.track_low_thresh,
            new_track_thresh=args.new_track_thresh, min_track_len=args.min_track_len,
            track_buffer=args.track_buffer, with_reid=args.with_reid,
            reid_model=str(args.reid_model) if args.with_reid else None,
            court_gate_margin_m=1.5, court_gate_minimum_overlap=.05, match_iou=.3),
        versions={name: importlib.metadata.version(name) for name in
                  ('ultralytics', 'torch', 'opencv-python', 'numpy', 'imageio-ffmpeg')},
        platform=platform.platform(), python=platform.python_version(), clips={},
        accuracy_metrics=None,
        limitations=[
            'Counts are detector/tracker coverage diagnostics, not accuracy metrics.',
            'Automatic homography and bbox-bottom court gating can be wrong.',
            'No manual roles, identities, player count, or corrections are used.',
            'Raw detection and tracking calls run separately with identical detector settings.'
        ])
    write_json(args.output/'experiment.json', report)
    try:
        for clip, video in resolve_clips(args.videos, args.clips):
            started = time.perf_counter()
            info = video_info(video)
            source_frames, source_provenance = load_source(args.source, clip, info)
            detector, tracker_model = YOLO(str(args.model)), YOLO(str(args.model))
            if detector.task != 'detect' or detector.names.get(0) != 'person':
                raise ValueError('Person diagnostic requires a COCO person detector')
            target = args.output/clip
            target.mkdir()
            defaults = yaml.safe_load((Path(ultralytics.__file__).parent/'cfg'/'trackers'/'tracktrack.yaml').read_text())
            tracker_settings = configured_tracker(defaults, high=args.track_high_thresh,
                low=args.track_low_thresh, new=args.new_track_thresh,
                minimum_length=args.min_track_len, buffer_frames=args.track_buffer,
                with_reid=args.with_reid, reid_model=args.reid_model)
            tracker_path = target/'person_tracker.yaml'
            tracker_path.write_text(yaml.safe_dump(tracker_settings), encoding='utf8')
            cap = cv2.VideoCapture(str(video))
            process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', str(info['fps']),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(target/'annotated.mp4')], stdin=subprocess.PIPE)
            records, totals = [], Counter()
            detector_s = tracker_s = 0.
            previous = None
            try:
                for index, source_frame in enumerate(source_frames):
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Video decode ended early')
                    thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 36)).astype(float)
                    cut = index == 0 or (previous is not None and np.abs(thumb-previous).mean() > 45)
                    previous = thumb
                    if cut and index and getattr(tracker_model, 'predictor', None) is not None:
                        for active_tracker in tracker_model.predictor.trackers:
                            active_tracker.reset()
                    tick = time.perf_counter()
                    detected = detector.predict(image, classes=[0], device=args.device,
                        imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False, save=False)[0]
                    detector_s += time.perf_counter()-tick
                    tick = time.perf_counter()
                    tracked = tracker_model.track(image, persist=True, tracker=str(tracker_path.resolve()),
                        classes=[0], device=args.device, imgsz=args.imgsz, conf=args.conf,
                        iou=args.iou, verbose=False, save=False)[0]
                    tracker_s += time.perf_counter()-tick
                    raw_rows, track_rows = result_rows(detected), result_rows(tracked, tracked=True)
                    detections, track_rows, counts = analyze_frame(
                        raw_rows, track_rows, source_frame, (info['width'], info['height']))
                    totals.update(counts)
                    record = dict(frame_idx=index, timestamp_s=index/info['fps'], camera_cut_reset=cut,
                                  raw_detections=detections, tracks=track_rows, counts=counts)
                    records.append(record)
                    process.stdin.write(render(image, record).tobytes())
                    if index in (0, info['frames']//4, info['frames']//2,
                                 3*info['frames']//4, info['frames']-1):
                        cv2.imwrite(str(target/f'frame-{index:04d}.jpg'), render(image, record))
                    if index % 25 == 0:
                        print(clip, index, '/', info['frames'], counts, flush=True)
                if cap.read()[0]:
                    raise ValueError('Video has extra frames')
                process.stdin.close()
                if process.wait() != 0:
                    raise ValueError('Diagnostic video encoding failed')
            finally:
                cap.release()
                if process.poll() is None:
                    process.kill(); process.wait()
            with (target/'frames.jsonl').open('w', encoding='utf8') as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False, default=json_default)+'\n')
            per_frame = [record['counts'] for record in records]
            summary = dict(frames=len(records), totals=dict(totals),
                mean_on_court_detections=float(np.mean([x['on_court_detections'] for x in per_frame])),
                mean_confirmed_tracks=float(np.mean([x['confirmed_tracks'] for x in per_frame])),
                mean_detected_not_confirmed=float(np.mean([x['on_court_detections_not_confirmed'] for x in per_frame])),
                detector_inference_s=detector_s, tracking_call_s=tracker_s,
                processing_s=time.perf_counter()-started,
                source=source_provenance, counts_are_not_accuracy=True)
            metadata = dict(status='complete', video=info, settings=report['settings'], summary=summary,
                tracker_config=tracker_settings,
                artifacts_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json', metadata)
            report['clips'][clip] = dict(summary=summary, metadata_sha256=sha256(target/'metadata.json'))
            write_json(args.output/'experiment.json', report)
        snapshots = args.output/'run_source'
        snapshots.mkdir()
        shutil.copyfile(Path(__file__), snapshots/Path(__file__).name)
        report.update(status='complete', source_sha256={p.name: sha256(p) for p in snapshots.iterdir()})
        write_json(args.output/'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        write_json(args.output/'experiment.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, default=Path('data/raw'))
    parser.add_argument('--clips', nargs='+', required=True)
    parser.add_argument('--source', type=Path, required=True,
                        help='Completed automatic run supplying per-frame homographies')
    parser.add_argument('--model', type=Path, default=Path('models/yolo26s.pt'))
    parser.add_argument('--reid-model', type=Path, default=Path('models/yolo26n-cls.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.7)
    parser.add_argument('--track-high-thresh', type=float, default=.6)
    parser.add_argument('--track-low-thresh', type=float, default=.25)
    parser.add_argument('--new-track-thresh', type=float, default=.7)
    parser.add_argument('--min-track-len', type=int, default=3)
    parser.add_argument('--track-buffer', type=int, default=30)
    parser.add_argument('--with-reid', action=argparse.BooleanOptionalAction, default=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
