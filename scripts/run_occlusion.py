"""Four retention/ReID variants, both clips and trackers; no inferred hidden boxes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

try:
    from .review_runs import review
except ImportError:  # Direct script execution from the notebook.
    from review_runs import review
from tacticalai.preview import comparison_grid


VARIANTS = {
    'baseline': dict(track_buffer=30, with_reid=False),
    'buffer60': dict(track_buffer=60, with_reid=False),
    'reid': dict(track_buffer=30, with_reid=True),
    'buffer60_reid': dict(track_buffer=60, with_reid=True),
}
LABELS = ['Baseline: buffer 30, ReID off', 'Buffer 60, ReID off',
          'Buffer 30, ReID on', 'Buffer 60, ReID on']
CLIPS = ('attack_clip', 'attack_clip2')
TRACKERS = ('botsort', 'tracktrack')


def summarize(root):
    issues, rows, metas = [], [], {}
    for variant, settings in VARIANTS.items():
        variant_root = root / variant
        variant_root.mkdir(exist_ok=True)
        report = review(variant_root, package=False)
        issues.extend(f'{variant}: {issue}' for issue in report['issues'])
        rows.extend(dict(variant=variant, **row) for row in report['runs'])
        for clip in CLIPS:
            for tracker in TRACKERS:
                label = f'{variant}/{clip}/{tracker}'
                try:
                    meta = json.loads((variant_root/clip/tracker/'metadata.json').read_text())
                    metas[label] = meta
                    for key, value in settings.items():
                        if meta['tracker_config'][key] != value:
                            issues.append(f'{label}: unexpected {key}')
                    if meta.get('effective_track_buffer_frames') != settings['track_buffer']:
                        issues.append(f'{label}: effective retention differs from requested value')
                    if settings['with_reid'] and not meta.get('reid_model_sha256'):
                        issues.append(f'{label}: missing appearance encoder hash')
                except (OSError, ValueError, KeyError) as error:
                    issues.append(f'{label}: {error}')
    # Across variants, only retention/ReID settings may change. Compare actual
    # detector outputs as well as common config, weights and execution hardware.
    detector_fields = ('model', 'conf', 'iou', 'imgsz', 'max_det', 'device', 'half', 'seed', 'gmc_method')
    for clip in CLIPS:
        reference = metas.get(f'baseline/{clip}/botsort')
        if reference is None:
            continue
        for label, meta in metas.items():
            if label.split('/')[1] != clip:
                continue
            for key in ('video_sha256', 'model_sha256', 'detections_sha256', 'versions', 'gpu', 'source_sha256', 'source'):
                if meta.get(key) != reference.get(key):
                    issues.append(f'{label}: comparison mismatch {key}')
            if any(meta['config'].get(k) != reference['config'].get(k) for k in detector_fields):
                issues.append(f'{label}: detector/GMC configuration differs')
    hashes = {m.get('reid_model_sha256') for m in metas.values() if m['config']['with_reid']}
    if len(hashes) > 1:
        issues.append('Appearance encoder differs between ReID runs')
    summary = dict(artifact_checks_passed=not issues, runs=rows, issues=issues,
                   accuracy_metrics=None, note='Unique IDs and gaps are diagnostics, not measured ID switches. '
                   'No hidden-player positions are exported. Appearance encoder is a generic classification model, '
                   'not a basketball identity model. Review occlusions manually.')
    (root/'suite-review.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Occlusion experiment review', '', summary['note'], '',
             '| Variant | Run | Frames | Observations | IDs | FPS |', '|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['variant']} | {r['run']} | {r['frames']} | {r['tracks']} | {r['unique_ids']} | {r['pipeline_fps']:.2f} |" for r in rows]
    lines += ['', 'Artifact checks: ' + ('passed' if not issues else 'FAILED'), '']
    lines += [f'- {issue}' for issue in issues]
    (root/'suite-review.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return summary


def finish(root):
    report = summarize(root)
    previews, preview_errors = [], []
    for clip in CLIPS:
        for tracker in TRACKERS:
            paths = [root/variant/clip/tracker/'annotated.mp4' for variant in VARIANTS]
            try:
                previews.append(comparison_grid(paths, LABELS, root/'comparisons'/f'{clip}-{tracker}.mp4'))
            except Exception as error:
                preview_errors.append(f'{clip}/{tracker}: {error}')
    (root/'previews.json').write_text(json.dumps(dict(previews=previews, errors=preview_errors), indent=2))
    # One archive; no source footage, model weights or recursively nested ZIPs.
    with zipfile.ZipFile(root/'occlusion-results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv', '.yaml', '.txt', '.md', '.mp4', '.jpg'}:
                archive.write(path, path.relative_to(root).as_posix())
    if not report['artifact_checks_passed'] or preview_errors:
        raise RuntimeError('Some checks/previews failed; partial outputs are preserved. Read suite-review.json and previews.json.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.review_only:
        base = json.loads(Path('configs/baseline.json').read_text())
        manifest = dict(variants=VARIANTS, clips=CLIPS, trackers=TRACKERS,
                        reid_model='yolo26n-cls.pt', expected_full_runs=16)
        (root/'experiment-plan.json').write_text(json.dumps(manifest, indent=2))
        errors = []
        for variant, settings in VARIANTS.items():
            variant_root = root/variant
            variant_root.mkdir(exist_ok=True)
            config = dict(base, **settings, reid_model='yolo26n-cls.pt')
            config_path = variant_root/'config.json'
            config_path.write_text(json.dumps(config, indent=2))
            for clip in CLIPS:
                try:
                    subprocess.run([sys.executable, '-m', 'tacticalai.cli', '--video',
                                    str(args.video_root/f'{clip}.mp4'), '--output', str(variant_root/clip),
                                    '--config', str(config_path)], check=True)
                except subprocess.CalledProcessError as error:
                    errors.append(f'{variant}/{clip}: {error}')
                (root/'execution-errors.json').write_text(json.dumps(errors, indent=2))
    finish(root)


if __name__ == '__main__':
    main()
