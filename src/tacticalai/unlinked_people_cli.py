"""Render a basic video with confirmed tracks plus unlinked YOLO people.

Consumes immutable detection/mask caches. No inference, association tracker,
role/team model or identity linking is rerun. No investigation UI is generated.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import time

from .export import sha256, write_json
from .role_team_cli import CLIPS, FIELDS, flat_record, render_video, verify
from .unlinked_people import add_unlinked


EXTRA_FIELDS = ['observation_kind', 'observation_id', 'detector_source', 'detection_index',
                'segmentation_source', 'segmentation_detection_index']


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    source_manifest = json.loads((args.source/'experiment.json').read_text())
    mask_meta = json.loads((args.masks/'metadata.json').read_text())
    if source_manifest['status'] != 'complete' or mask_meta['status'] != 'complete':
        raise ValueError('Incomplete automatic inputs')
    started = time.perf_counter()
    manifest = dict(status='running', automatic_only=True, inference_rerun=False,
        tracker_identity_role_team_changes=False, manual_labels_used=False,
        color_mode=args.color_mode, team_colors_rgb={'A': '#419be6', 'B': '#f5af46', 'referee': '#e678e6', 'unknown': '#aaaaaa'},
        configuration=config, configuration_sha256=sha256(args.config),
        source=str(args.source.resolve()), source_manifest_sha256=sha256(args.source/'experiment.json'),
        segmentation_cache_metadata=mask_meta, masks_metadata_sha256=sha256(args.masks/'metadata.json'),
        label_policy='Unlinked = current YOLO person observation without assigned identity, role or team.',
        map_policy='Unlinked observations are image measurements only; no new court positions.',
        court_policy='Conservative automatic court gate for additions only; clipped/uncertain boxes remain. No role filter.',
        limitations=[
            'Occluded people still require actual detector/segmenter evidence; no hidden people or masks are invented.',
            'Geometric box matching and mask association can fail around overlapping people or duplicate boxes.',
            'Unlinked people may include spectators/staff; no player role is assigned.',
            'The court filter can exclude real boundary people when calibration is wrong.',
            'No new independently validated accuracy claim. Upstream classification and geometry are development results.'
        ], clips={})
    write_json(args.output/'experiment.json', manifest)
    combined = []
    try:
        for clip in CLIPS:
            start = time.perf_counter()
            source = args.source/clip
            meta = json.loads((source/'metadata.json').read_text())
            if meta['status'] != 'complete':
                raise ValueError('Incomplete clip source')
            source_frames = verify(source, 'frames.jsonl', meta)
            frames = [json.loads(l) for l in source_frames.read_text().splitlines()]
            info = meta['video']
            video = args.videos/(clip+'.mp4')
            masks_path = args.masks/clip/'masks.jsonl'
            if sha256(video) != info['sha256'] or info['sha256'] != mask_meta['clips'][clip]['video_sha256']:
                raise ValueError('Source video differs')
            if sha256(masks_path) != mask_meta['clips'][clip]['masks_sha256']:
                raise ValueError('Segmentation cache changed')
            masks = [json.loads(l) for l in masks_path.read_text().splitlines()]
            if not len(frames) == len(masks) == info['frames']:
                raise ValueError('Frame count mismatch')
            review = json.loads(verify(source, 'review-data.json', meta).read_text())
            target = args.output/clip; target.mkdir()
            enriched = [add_unlinked(f, m, (info['width'], info['height']), config) for f, m in zip(frames, masks)]
            stats = Counter()
            for frame in enriched:
                for k, v in frame['unlinked_display'].items():
                    stats[k] += len(v) if isinstance(v, list) else v
            with (target/'frames.jsonl').open('w', encoding='utf8') as stream:
                for frame in enriched:
                    stream.write(json.dumps(frame, allow_nan=False)+'\n')
            flat = [dict(flat_record(clip, f, r), **{k: r.get(k) for k in EXTRA_FIELDS})
                    for f in enriched for r in f['tracks']]
            combined.extend(flat)
            with (target/'observations.csv').open('w', newline='', encoding='utf8') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS+EXTRA_FIELDS)
                writer.writeheader(); writer.writerows(flat)
            decoded = render_video(video, enriched, review['court_lines'], target, info['fps'], color_mode=args.color_mode)
            # Preserve source identity and geometry. Only missing masks may be filled.
            for original, updated in zip(frames, enriched):
                n = len(original['tracks'])
                for before, after in zip(original['tracks'], updated['tracks'][:n]):
                    for k, v in before.items():
                        if k == 'display_caption' and before.get('player_id') is None:
                            continue
                        if k in ('silhouette_xy', 'silhouette_confidence', 'silhouette_box_iou') and before.get('silhouette_xy') is None:
                            continue
                        if after[k] != v:
                            raise ValueError(f'Changed original observation: {clip}/{k}')
                for row in updated['tracks'][n:]:
                    if any(row[k] is not None for k in ('track_id', 'player_id', 'role', 'team_id', 'court_xy_m')):
                        raise ValueError('Unlinked observation acquired an invented identity/classification/position')
            report = dict(status='complete', video=info, source_frames_sha256=sha256(source_frames),
                source_metadata_sha256=sha256(source/'metadata.json'), masks_sha256=sha256(masks_path),
                counts=dict(stats), output_observations=len(flat), annotated_frames_decoded=decoded,
                packaging_processing_s=time.perf_counter()-start,
                identity_role_team_position_measurements_preserved=True,
                artifacts_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json', report)
            manifest['clips'][clip] = report
            print(clip, dict(stats), flush=True)
        with (args.output/'observations.csv').open('w', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS+EXTRA_FIELDS)
            writer.writeheader(); writer.writerows(combined)
        # Basic native video controls only. Detailed notes belong in the chat.
        (args.output/'index.html').write_text('''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>סרטוני זיהוי אנשים</title>
<body style="margin:16px;background:#111;color:white;font:16px Arial"><p>קליפ 1 — Unlinked = אדם שזוהה ללא שיוך לזהות</p>
<video controls playsinline preload="metadata" style="width:min(100%,1100px)" src="attack_clip/annotated.mp4"></video>
<p>קליפ 2</p><video controls playsinline preload="metadata" style="width:min(100%,1100px)" src="attack_clip2/annotated.mp4"></video>
</body></html>''', encoding='utf8')
        snapshots = args.output/'run_source'; snapshots.mkdir()
        for name in ('unlinked_people.py', 'unlinked_people_cli.py', 'silhouette_presentation.py', 'role_team_cli.py', 'court_gate.py'):
            shutil.copyfile(Path(__file__).with_name(name), snapshots/name)
        shutil.copyfile(args.config, args.output/'configuration.json')
        manifest.update(status='complete', processing_s=time.perf_counter()-started,
            source_sha256={p.name: sha256(p) for p in snapshots.iterdir()})
        write_json(args.output/'experiment.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        write_json(args.output/'experiment.json', manifest)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/roles-teams-final-v1'))
    p.add_argument('--masks', type=Path, default=Path('outputs/person-coverage-investigation/masks-all'))
    p.add_argument('--videos', type=Path, default=Path('data/raw'))
    p.add_argument('--config', type=Path, default=Path('configs/unlinked_people.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--color-mode', choices=['team', 'identity'], default='team')
    run(p.parse_args())


if __name__ == '__main__':
    main()
