"""Evaluate ball localization against draft review centers; never feed inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from tacticalai.export import sha256, write_json


def score_samples(frames, samples, image_size, coordinate_size, radius, mode):
    if mode not in ('raw', 'accepted', 'observed'):
        raise ValueError('Unknown evaluation mode')
    for size in (image_size, coordinate_size):
        if np.shape(size) != (2,) or not np.isfinite(size).all() or np.any(np.asarray(size) <= 0):
            raise ValueError('Invalid evaluation image size')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid evaluation radius')
    lookup = {f['frame_idx']: f for f in frames}
    if len(lookup) != len(frames) or len({s['frame_idx'] for s in samples}) != len(samples):
        raise ValueError('Duplicate evaluation frame indices')
    totals = dict(samples=0, excluded_unknown=0, true_positives=0, false_positives=0,
                  false_negatives=0, predictions=0)
    details, errors = [], []
    for sample in samples:
        if sample['frame_idx'] not in lookup:
            raise ValueError(f"Missing evaluation frame {sample['frame_idx']}")
        frame = lookup[sample['frame_idx']]
        if sample['visibility'] == 'unknown':
            totals['excluded_unknown'] += 1
            continue
        if sample['visibility'] not in ('visible', 'absent'):
            raise ValueError('Invalid reference visibility')
        if mode == 'raw':
            points = [(np.asarray(r['bbox_xyxy'][:2])+r['bbox_xyxy'][2:])/2
                      for r in frame['ball_raw_detections']]
        elif mode == 'accepted':
            points = [r['center_image_xy'] for r in frame['ball']['candidates']]
        elif mode == 'observed':
            points = [frame['ball']['center_image_xy']] if frame['ball']['status'] == 'observed' else []
        else:
            raise ValueError('Unknown evaluation mode')
        if any(np.shape(p) != (2,) or not np.isfinite(p).all() for p in points):
            raise ValueError('Invalid predicted ball center')
        points = [np.asarray(p)*np.asarray(coordinate_size)/image_size for p in points]
        center = sample['center_xy']
        if sample['visibility'] == 'visible' and center is None:
            raise ValueError('Visible reference has no center')
        if center is not None and (np.shape(center) != (2,) or not np.isfinite(center).all()):
            raise ValueError('Invalid reference center')
        if sample['visibility'] == 'absent' and center is not None:
            raise ValueError('Absent reference has a center')
        distances = [float(np.linalg.norm(p-center)) for p in points] if center is not None else []
        matched = bool(distances and min(distances) <= radius)
        tp = int(matched)
        fp, fn = len(points)-tp, int(center is not None and not matched)
        totals['samples'] += 1
        totals['predictions'] += len(points)
        totals['true_positives'] += tp
        totals['false_positives'] += fp
        totals['false_negatives'] += fn
        if matched:
            errors.append(min(distances))
        details.append(dict(frame_idx=sample['frame_idx'], matched=matched,
                            false_positives=fp, false_negatives=fn,
                            closest_error_review_px=min(distances) if distances else None))
    tp, fp, fn = (totals[k] for k in ('true_positives', 'false_positives', 'false_negatives'))
    return dict(totals, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                matched_mean_error_review_px=float(np.mean(errors)) if errors else None, details=details)


def evaluate(source, reference, split):
    if split not in ('development', 'evaluation'):
        raise ValueError('Unknown reference split')
    labels = json.loads(reference.read_text(encoding='utf8'))
    manifest = json.loads((source/'experiment.json').read_text(encoding='utf8'))
    if manifest['status'] != 'complete':
        raise ValueError('Cannot evaluate an incomplete experiment')
    report = dict(reference_sha256=sha256(reference), source=str(source.resolve()),
                  split=split, annotation_quality=labels['quality'],
                  coordinate_size=labels['coordinate_size'], match_radius_px=labels['match_radius_px'], clips={})
    for clip, document in labels['clips'].items():
        folder = source/clip
        meta = json.loads((folder/'metadata.json').read_text(encoding='utf8'))
        path = folder/'frames.jsonl'
        if (meta['status'] != 'complete' or meta['video']['sha256'] != document['video_sha256'] or
                sha256(path) != meta['artifacts_sha256']['frames.jsonl']):
            raise ValueError('Reference video or result hash differs')
        frames = [json.loads(line) for line in path.read_text().splitlines()]
        samples = [s for s in document['samples'] if s['split'] == split]
        report['clips'][clip] = {mode: score_samples(frames, samples,
            [meta['video']['width'], meta['video']['height']], labels['coordinate_size'],
            labels['match_radius_px'], mode) for mode in ('raw', 'accepted', 'observed')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('configs/evaluation/ball_reference.json'))
    parser.add_argument('--split', choices=('development', 'evaluation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.source, args.reference, args.split)
    write_json(args.output, report)
    print(json.dumps({clip: {mode: {k: v for k, v in row.items() if k != 'details'}
                            for mode, row in rows.items()} for clip, rows in report['clips'].items()}, indent=2))


if __name__ == '__main__':
    main()
