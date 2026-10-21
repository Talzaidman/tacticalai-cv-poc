"""Package the verified earlier automatic identity results as the active review."""

import argparse
import json
from pathlib import Path
import shutil

import cv2

from tacticalai.export import sha256, write_json


def verified_copy(root, filename, metadata, destination):
    source = root / filename
    if sha256(source) != metadata['artifacts_sha256'][filename]:
        raise ValueError(f'Saved artifact changed: {source}')
    shutil.copyfile(source, destination)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    profile = Path('configs/reid/identity_checkpoint_20261019.json')
    config = json.loads(profile.read_text())
    active = json.loads(Path('configs/reid/classifier-n.json').read_text())
    if active != config:
        raise ValueError('Active ReID profile does not match the restored checkpoint')
    report = dict(status='running', active_baseline=True, automatic_only=True,
        roles_used=False, manual_identity_corrections=False, networks_rerun=False,
        tracking_profile=config, tracking_profile_sha256=sha256(profile),
        identity_stage='Automatic CLIP-ReIdent temporal splitting and gallery linking',
        defaults=dict(pending_detections=False, boundary_cases=False, id_source='linked'),
        offline_future_frames_used=True, thresholds_developed_on_these_clips=True,
        independent_accuracy_metrics=None, clips={})
    write_json(output / 'experiment.json', report)
    try:
        for clip, expected in [('attack_clip', 150), ('attack_clip2', 175)]:
            source = Path('outputs/id-continuity-review-v1') / clip
            automatic = Path('outputs/automatic-silhouettes-v1') / clip
            metadata = json.loads((source / 'metadata.json').read_text())
            automatic_metadata = json.loads((automatic / 'metadata.json').read_text())
            target = output / clip
            target.mkdir()
            data_path = source / 'review-data.json'
            if sha256(data_path) != metadata['artifacts_sha256']['review-data.json']:
                raise ValueError('Saved review data changed')
            payload = json.loads(data_path.read_text())
            payload['frames'] = {'linked': payload['frames']['linked']}
            frames = payload['frames']['linked']
            assert len(frames) == expected
            for frame in frames:
                ids = [r['display_label'] for r in frame['tracks'] if r['display_label'] != '?']
                assert len(ids) == len(set(ids)), 'Simultaneous duplicate display identity'
                assert all(r['role'] is None for r in frame['tracks'])
            write_json(target / 'review-data.json', payload)
            for filename in ('clean.mp4', 'linked-clean.mp4'):
                verified_copy(source, filename, metadata, target / filename)
            for filename in ('frames.jsonl', 'positions.csv', 'identity-evidence.json'):
                verified_copy(automatic, filename, automatic_metadata, target / filename)
            video = cv2.VideoCapture(str(target / 'linked-clean.mp4'))
            decoded = 0
            while video.read()[0]:
                decoded += 1
            video.release()
            assert decoded == expected, 'Incomplete annotated video'
            original = Path('outputs/occlusion-20261015T204150Z-029a7a35/reid') / clip / 'tracktrack/metadata.json'
            tracker_metadata = json.loads(original.read_text())
            meta = dict(status='complete', frames=expected, annotated_frames_decoded=decoded,
                roles_used=False, networks_rerun=False,
                tracker_config=tracker_metadata['tracker_config'],
                source_tracking_metadata=tracker_metadata,
                source_automatic_metadata=automatic_metadata,
                source_review_metadata_sha256=sha256(source / 'metadata.json'),
                source_tracking_metadata_sha256=sha256(original),
                artifacts_sha256={p.name: sha256(p) for p in target.iterdir()})
            write_json(target / 'metadata.json', meta)
            report['clips'][clip] = dict(frames=expected, annotated_frames_decoded=decoded,
                metadata_sha256=sha256(target / 'metadata.json'))
            print(f'{clip}: verified {decoded} frames; earlier identities only', flush=True)
        template = Path('src/tacticalai/id_review.html')
        html = template.read_text(encoding='utf8').replace('<body>', '<body data-baseline="true">', 1)
        (output / 'index.html').write_text(html, encoding='utf8')
        shutil.copyfile(profile, output / 'tracking-profile.json')
        snapshots = output / 'run_source'
        snapshots.mkdir()
        for source in (Path(__file__), template, Path('scripts/serve_review.py')):
            shutil.copyfile(source, snapshots / source.name)
        report.update(status='complete', source_sha256={p.name: sha256(p) for p in snapshots.iterdir()})
        write_json(output / 'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        write_json(output / 'experiment.json', report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/restored-identity-baseline-v1'))
    run(parser.parse_args().output)
