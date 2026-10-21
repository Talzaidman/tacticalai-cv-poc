"""Add display-only segmentation to the saved identity/court review."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .export import sha256, write_json
from .identity_presentation import person_color, text
from .challenge_presentation import CLIPS


def associate_masks(rows, boxes, polygons, confidences, minimum_iou=.3):
    """One-to-one geometric association; missing masks remain explicitly null."""
    for row in rows:
        row.update(silhouette_xy=None, silhouette_confidence=None, silhouette_box_iou=None)
    if not rows or not len(boxes):
        return
    a = np.asarray([r['bbox_xyxy'] for r in rows], dtype=float)
    b = np.asarray(boxes, dtype=float)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    areas_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    areas_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    iou = intersection / np.maximum(areas_a[:, None] + areas_b - intersection, 1e-9)
    # Prefer the largest number of valid matches, then their geometric overlap.
    valid = iou >= minimum_iou
    rr, cc = linear_sum_assignment(-(valid * (min(iou.shape) + 1) + iou))
    for r, c in zip(rr, cc):
        if valid[r, c] and len(polygons[c]) >= 3:
            rows[r].update(silhouette_xy=np.round(polygons[c], 2).tolist(),
                           silhouette_confidence=float(confidences[c]),
                           silhouette_box_iou=float(iou[r, c]))


TEAM_COLORS = {'A': (230, 155, 65), 'B': (70, 175, 245)}


def observation_color(row, mode='identity'):
    if mode == 'identity':
        return person_color(row['player_id'])
    if row.get('role') == 'player' and row.get('team_id') in TEAM_COLORS:
        return TEAM_COLORS[row['team_id']]
    if row.get('role') == 'referee':
        return (230, 120, 230)
    return (170, 170, 170)


def render(image, frame, lines, color_mode='identity'):
    out = cv2.resize(image, (1280, 720))
    sx, sy = 1280 / image.shape[1], 720 / image.shape[0]
    rows = [r for r in frame['tracks'] if r.get('display_on_video', True)]
    for row in rows:
        color = observation_color(row, color_mode)
        polygon = row['silhouette_xy']
        if polygon is not None:
            points = np.rint(np.asarray(polygon) * [sx, sy]).astype(np.int32)
            mask = np.zeros(out.shape[:2], np.uint8)
            cv2.fillPoly(mask, [points], 1)
            selected = mask.astype(bool)
            alpha = row.get('silhouette_alpha', .4)
            out[selected] = ((1-alpha) * out[selected] + alpha * np.asarray(color)).astype(np.uint8)
            cv2.polylines(out, [points], True, color, 1, cv2.LINE_AA)
    for row in rows:
        x, y = row['bbox_xyxy'][:2]
        if row.get('draw_detection_box_fallback') and row.get('silhouette_xy') is None:
            x1, y1, x2, y2 = row['bbox_xyxy']
            cv2.rectangle(out, (round(x1*sx), round(y1*sy)), (round(x2*sx), round(y2*sy)),
                          observation_color(row, color_mode), 1, cv2.LINE_AA)
        text(out, row.get('display_caption') or f"{row['display_label']} / {row['track_id']}",
             (max(0, round(x*sx)), max(20, round(y*sy)-6)), row.get('display_label_scale', .46), observation_color(row, color_mode))
    panel = np.full((190, 304, 3), (30, 40, 16), np.uint8)
    scale, origin = 9.6, np.array([17, 29])
    for line in lines:
        cv2.polylines(panel, [np.rint(np.asarray(line)*scale+origin).astype(np.int32)],
                      False, (170, 185, 180), 1, cv2.LINE_AA)
    for row in rows:
        if row['court_xy_m'] is None or row['court_region'] != 'inside':
            continue
        x, y = np.rint(np.asarray(row['court_xy_m'])*scale+origin).astype(int)
        color = observation_color(row, color_mode)
        cv2.circle(panel, (x, y), 3, color, -1, cv2.LINE_AA)
        text(panel, row['display_label'], (max(0, min(x+4, 260)), max(20, y-4)), .3, color)
    method = frame['calibration']['method']
    status = 'estimated' if 'gap_estimate' in method else 'unavailable' if method == 'unavailable' else 'smoothed'
    text(panel, 'COURT | '+status, (8, 16), .38)
    if color_mode == 'team':
        for x, label, color in ((14, 'A', TEAM_COLORS['A']), (72, 'B', TEAM_COLORS['B']),
                                 (130, 'Ref', (230,120,230)), (218, '?', (170,170,170))):
            cv2.circle(panel, (x, 181), 3, color, -1, cv2.LINE_AA)
            text(panel, label, (x+7, 185), .32, color)
    else:
        text(panel, 'Approximate person floor positions', (8, 183), .32)
    out[10:200, 10:314] = panel
    return out


def run(source, output, model_path, device='cpu', imgsz=640):
    from ultralytics import YOLO
    import torch
    import imageio_ffmpeg

    if output.exists():
        raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4)
    model = YOLO(str(model_path))
    if model.task != 'segment':
        raise ValueError('A segmentation checkpoint is required')
    manifest = json.loads((source/'experiment.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Source presentation is incomplete')
    shutil.copytree(source, output)
    manifest.update(status='running', source_presentation=str(source.resolve()),
                    source_experiment_sha256=sha256(source/'experiment.json'))
    settings = dict(model=str(model_path), model_sha256=sha256(Path(model.ckpt_path)),
                    device=device, imgsz=imgsz, conf=.2, iou=.7, classes=[0],
                    retina_masks=True, association='one-to-one maximum box IoU',
                    minimum_association_iou=.3, alpha=.4, missing_mask_policy='null; label only',
                    role_source='existing manual review', tracking_identity_projection_rerun=False,
                    accuracy_metrics=None, versions={p: importlib.metadata.version(p) for p in
                        ('ultralytics', 'torch', 'opencv-python', 'numpy', 'scipy')})
    manifest['silhouettes'] = settings
    write_json(output/'experiment.json', manifest)
    started = time.perf_counter()
    try:
        for clip in CLIPS:
            target = output/clip
            data = json.loads((source/clip/'review-data.json').read_text())
            video = Path('data/raw')/(clip+'.mp4')
            if sha256(video) != data['video']['sha256']:
                raise ValueError('Source video hash mismatch')
            cap = cv2.VideoCapture(str(video))
            process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', str(data['video']['fps']),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(target/'tracking-minimap.mp4')], stdin=subprocess.PIPE)
            inference_s, matched, total = 0., 0, 0
            clip_started = time.perf_counter()
            try:
                for frame in data['frames']:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Source decode incomplete')
                    original = copy.deepcopy(frame['tracks'])
                    t = time.perf_counter()
                    result = model.predict(image, device=device, imgsz=imgsz, conf=.2, iou=.7,
                                           classes=[0], retina_masks=True, verbose=False)[0]
                    inference_s += time.perf_counter()-t
                    polygons = result.masks.xy if result.masks is not None else []
                    associate_masks(frame['tracks'], result.boxes.xyxy.cpu().numpy() if polygons else [],
                                    polygons, result.boxes.conf.cpu().numpy())
                    for before, after in zip(original, frame['tracks']):
                        assert all(after[k] == v for k, v in before.items())
                    matched += sum(r['silhouette_xy'] is not None for r in frame['tracks'])
                    total += len(frame['tracks'])
                    rendered = render(image, frame, data['court_lines'])
                    process.stdin.write(rendered.tobytes())
                    if frame['frame_idx'] in (0, 55, 131):
                        cv2.imwrite(str(target/f"frame-{frame['frame_idx']:04d}.jpg"), rendered)
                    if frame['frame_idx'] % 25 == 0:
                        print(clip, frame['frame_idx'], '/', len(data['frames']), flush=True)
                if cap.read()[0]:
                    raise ValueError('Unexpected extra source frames')
                process.stdin.close()
                if process.wait() != 0:
                    raise ValueError('Encode failed')
            finally:
                cap.release()
                if process.poll() is None:
                    process.kill(); process.wait()
            check = cv2.VideoCapture(str(target/'tracking-minimap.mp4'))
            decoded = 0
            while check.read()[0]:
                decoded += 1
            check.release()
            if decoded != len(data['frames']):
                raise ValueError('Output decode frame count mismatch')
            stats = dict(matched_masks=matched, observed_tracks=total, missing_masks=total-matched,
                         decoded_frames=decoded, segmentation_inference_s=inference_s,
                         segmentation_inference_fps=decoded/inference_s,
                         processing_s=time.perf_counter()-clip_started,
                         coverage_is_not_accuracy=True)
            data['silhouettes'] = dict(settings, **stats)
            write_json(target/'review-data.json', data)
            with (target/'frames.jsonl').open('w', encoding='utf8') as stream:
                for frame in data['frames']:
                    stream.write(json.dumps(frame, allow_nan=False)+'\n')
            meta = json.loads((source/clip/'metadata.json').read_text())
            meta.update(silhouettes=dict(settings, **stats),
                        source_metadata_sha256=sha256(source/clip/'metadata.json'))
            meta['artifacts_sha256'] = {n: sha256(target/n) for n in
                ('frames.jsonl', 'positions.csv', 'tracking-minimap.mp4', 'review-data.json')}
            write_json(target/'metadata.json', meta)
            manifest['clips'][clip]['silhouettes'] = stats
            print(clip, stats, flush=True)
        shutil.copyfile(Path(__file__).with_name('identity_presentation.html'), output/'index.html')
        shutil.copyfile(Path(__file__), output/'run_source'/Path(__file__).name)
        shutil.copyfile(Path(__file__).with_name('identity_presentation.html'), output/'run_source'/'identity_presentation.html')
        manifest['source_sha256'] = {p.name: sha256(p) for p in (output/'run_source').iterdir()}
        manifest.update(status='complete', silhouette_processing_s=time.perf_counter()-started)
        write_json(output/'experiment.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        write_json(output/'experiment.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/identity-court-complete-final'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/yolo26s-seg.pt'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    # The intermediate cache preserves historical source provenance, but the
    # delivered presentation is rebuilt from automatic identities and cameras.
    # Its renderer never uses the source's manually assigned roles.
    cache = args.output.with_name(args.output.name + '-mask-cache')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    run(args.source, cache, args.model, args.device, args.imgsz)
    from .automatic_presentation import run as render_automatic
    render_automatic(cache, Path('outputs/temporal-reid-final'),
                     Path('outputs/challenge-player-smoothed-v1'), args.output)


if __name__ == '__main__':
    main()
