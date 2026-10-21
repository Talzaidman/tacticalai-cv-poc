"""Re-render immutable observations with team colors and a floor-context filter."""
import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import shutil
import time

import cv2

from .export import sha256, write_json
from .floor_context import apply_display_filter
from .role_team_cli import CLIPS, render_video, verify


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    source_manifest = json.loads((args.source/'experiment.json').read_text())
    if source_manifest['status'] != 'complete':
        raise ValueError('Incomplete observation source')
    report = dict(status='running', source=str(args.source.resolve()),
        source_manifest_sha256=sha256(args.source/'experiment.json'), color_mode='team',
        colors_rgb=dict(A='#419be6', B='#f5af46', referee='#e678e6', unknown='#aaaaaa'),
        inference_rerun=False, identities_roles_teams_positions_changed=False,
        manual_labels_used=False, offline_processing=True, floor_context=config,
        notes=[
            'Team colors use existing automatic predictions; unknown group remains gray.',
            'Display filter fits floor appearance per frame from the calibrated court interior, excluding people.',
            'Low local floor support hides unlinked/nonplayer rows only. Existing predicted players are retained.',
            'This is an experimental background-context heuristic, not verified court segmentation or proof a person is outside.',
            'Insufficient/occluded floor context is retained. Every hidden observation stays in JSONL/CSV with its reason.',
            'Upstream automatic predictions/calibration, offline smoothing and development-clip assumptions are inherited.'
        ], clips={})
    write_json(args.output/'experiment.json', report)
    started = time.perf_counter()
    try:
        for clip in CLIPS:
            t = time.perf_counter()
            source = args.source/clip
            meta = json.loads((source/'metadata.json').read_text())
            path = verify(source, 'frames.jsonl', meta)
            original = [json.loads(l) for l in path.read_text().splitlines()]
            frames = copy.deepcopy(original)
            info = meta['video']
            video = args.videos/(clip+'.mp4')
            if sha256(video) != info['sha256']:
                raise ValueError('Video source differs')
            # Court template is a cache from the immutable upstream role baseline.
            baseline = Path(source_manifest['source'])/clip
            baseline_meta = json.loads((baseline/'metadata.json').read_text())
            review = json.loads(verify(baseline, 'review-data.json', baseline_meta).read_text())
            cap = cv2.VideoCapture(str(video)); counts = Counter()
            try:
                for f in frames:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Incomplete source video')
                    counts.update(apply_display_filter(image, f, config))
                if cap.read()[0]:
                    raise ValueError('Extra source frames')
            finally:
                cap.release()
            target = args.output/clip; target.mkdir()
            with (target/'frames.jsonl').open('w', encoding='utf8') as stream:
                for before, after in zip(original, frames):
                    for a, b in zip(before['tracks'], after['tracks']):
                        if any(b[k] != v for k, v in a.items()):
                            raise ValueError('Original observation changed')
                    stream.write(json.dumps(after, allow_nan=False)+'\n')
            with verify(source, 'observations.csv', meta).open(newline='', encoding='utf8') as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames + ['display_on_video', 'display_filter_reason', 'floor_context_fraction']
                flat = list(reader)
            rows = [r for f in frames for r in f['tracks']]
            if len(flat) != len(rows):
                raise ValueError('CSV/JSON row count mismatch')
            with (target/'observations.csv').open('w', newline='', encoding='utf8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
                for record, row in zip(flat, rows):
                    if record['observation_id'] != row['observation_id']:
                        raise ValueError('CSV alignment differs')
                    record.update(display_on_video=row['display_on_video'],
                        display_filter_reason=row['display_filter_reason'],
                        floor_context_fraction=row['floor_context_evidence']['floor_fraction'])
                    writer.writerow(record)
            decoded = render_video(video, frames, review['court_lines'], target, info['fps'], color_mode='team')
            result = dict(status='complete', video=info, source_frames_sha256=sha256(path),
                counts=dict(counts), annotated_frames_decoded=decoded,
                original_observations_preserved=True, processing_s=time.perf_counter()-t,
                artifacts_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json', result)
            report['clips'][clip] = result
            print(clip, dict(counts), flush=True)
        snapshots = args.output/'run_source'; snapshots.mkdir()
        for name in ('team_video_cli.py', 'floor_context.py', 'silhouette_presentation.py', 'role_team_cli.py'):
            shutil.copyfile(Path(__file__).with_name(name), snapshots/name)
        shutil.copyfile(args.config, args.output/'configuration.json')
        (args.output/'index.html').write_text('''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>וידאו בצבעי קבוצה</title><body style="margin:16px;background:#111;color:white;font:16px Arial"><p>קליפ 1</p><video controls playsinline preload="metadata" style="width:min(100%,1100px)" src="attack_clip/annotated.mp4"></video><p>קליפ 2</p><video controls playsinline preload="metadata" style="width:min(100%,1100px)" src="attack_clip2/annotated.mp4"></video></body></html>''', encoding='utf8')
        report.update(status='complete', processing_s=time.perf_counter()-started,
            source_sha256={p.name: sha256(p) for p in snapshots.iterdir()})
        write_json(args.output/'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error)); write_json(args.output/'experiment.json', report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/unlinked-people-v2'))
    p.add_argument('--videos', type=Path, default=Path('data/raw'))
    p.add_argument('--config', type=Path, default=Path('configs/floor_context.json'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
