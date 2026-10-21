"""Verify consolidated artifacts against the unchanged automatic baseline."""
import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

from tacticalai.export import sha256, write_json


def validate(source, output, url=None):
    manifest = json.loads((output/'experiment.json').read_text())
    assert manifest['status'] == 'complete'
    for name, digest in manifest['source_sha256'].items():
        assert sha256(output/'run_source'/name) == digest
    checked = {}
    for clip in ('attack_clip', 'attack_clip2'):
        target = output/clip
        meta = json.loads((target/'metadata.json').read_text())
        assert sha256(target/'metadata.json') == manifest['clips'][clip]['metadata_sha256']
        for name, digest in meta['artifacts_sha256'].items():
            assert sha256(target/name) == digest, name
        baseline = [json.loads(l) for l in (source/clip/'frames.jsonl').read_text().splitlines()]
        frames = [json.loads(l) for l in (target/'frames.jsonl').read_text().splitlines()]
        assert len(frames) == len(baseline) == meta['video']['frames']
        assert sha256(source/clip/'frames.jsonl') == meta['source_frames_sha256']
        evidence = json.loads((target/'classification-evidence.json').read_text())
        assert evidence['configuration'] == manifest['configuration']
        assert evidence['manual_labels_used'] is False
        for values in evidence['samples'].values():
            for sample in values:
                assert sha256(target/sample['crop']) == sample['crop_sha256']
        flat = list(csv.DictReader((target/'observations.csv').open(newline='', encoding='utf8')))
        observations = []
        for original, enriched in zip(baseline, frames):
            assert original['frame_idx'] == enriched['frame_idx']
            assert original['timestamp_s'] == enriched['timestamp_s']
            assert len(original['tracks']) == len(enriched['tracks'])
            assert original['image_to_court_h'] == enriched['image_to_court_h']
            for before, after in zip(original['tracks'], enriched['tracks']):
                for k, v in before.items():
                    if k not in ('role', 'team_id', 'jersey_number'):
                        assert after[k] == v, (clip, original['frame_idx'], k)
                if after['player_id'] is None:
                    assert after['role'] is None and after['team_id'] is None
                if after['role'] != 'player':
                    assert after['team_id'] is None
                assert after['jersey_number'] is None
                observations.append((enriched['frame_idx'], after))
        assert len(observations) == len(flat) == meta['counts']['observations']
        for (index, row), record in zip(observations, flat):
            assert int(record['frame_idx']) == index
            assert int(record['track_id']) == row['track_id']
            assert record['person_id'] == (row['player_id'] or '')
            assert record['role'] == (row['role'] or '')
            assert record['team_id'] == (row['team_id'] or '')
            assert record['jersey_number'] == ''
            if row['court_xy_m'] is None:
                assert record['court_x_m'] == record['court_y_m'] == ''
            else:
                assert [float(record['court_x_m']), float(record['court_y_m'])] == row['court_xy_m']
        checked[clip] = dict(frames=len(frames), observations=len(observations),
                             output_hashes_verified=len(meta['artifacts_sha256']))
    combined = list(csv.DictReader((output/'observations.csv').open(newline='', encoding='utf8')))
    assert len(combined) == sum(v['observations'] for v in checked.values())
    evaluation = json.loads((output/'evaluation.json').read_text())
    assert evaluation['labels_used_for_inference'] is False
    for clip, digest in evaluation['prediction_files_sha256'].items():
        assert sha256(output/clip/'frames.jsonl') == digest
    http = []
    if url:
        files = ['index.html', 'schema.json', 'experiment.json', 'configuration.json',
                 'evaluation.json', 'observations.csv']
        for clip in checked:
            files += [f'{clip}/{n}' for n in ('annotated.mp4', 'clean.mp4', 'observations.csv',
                      'frames.jsonl', 'persons.json', 'classification-evidence.json', 'metadata.json', 'review-data.json')]
        for name in files:
            with urlopen(Request(url.rstrip('/')+'/'+name, method='HEAD')) as response:
                assert response.status == 200
                http.append(name)
        with urlopen(Request(url.rstrip('/')+'/attack_clip2/annotated.mp4', headers={'Range': 'bytes=0-127'})) as response:
            assert response.status == 206 and len(response.read()) == 128
    report = dict(status='complete', clips=checked, combined_csv_rows=len(combined),
                  source_measurements_preserved=True, unknowns_verified=True,
                  artifact_hashes_verified=True, evaluation_matches_predictions=True,
                  http_resources_verified=http, accuracy_claim=None)
    write_json(output/'verification.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/restored-identity-baseline-v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url')
    a = p.parse_args()
    validate(a.source, a.output, a.url)
