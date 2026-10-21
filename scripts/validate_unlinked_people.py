"""Check that extra visible rows are real detections with no invented IDs."""
import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

from tacticalai.export import sha256, write_json


def run(output, url=None):
    manifest = json.loads((output/'experiment.json').read_text())
    assert manifest['status'] == 'complete' and not manifest['inference_rerun']
    for name, digest in manifest['source_sha256'].items():
        assert sha256(output/'run_source'/name) == digest
    summary = {}
    for clip, meta in manifest['clips'].items():
        target = output/clip
        for name, digest in meta['artifacts_sha256'].items():
            assert sha256(target/name) == digest
        frames = [json.loads(l) for l in (target/'frames.jsonl').read_text().splitlines()]
        rows = [row for f in frames for row in f['tracks']]
        assert len(frames) == meta['video']['frames'] == meta['annotated_frames_decoded']
        assert len(rows) == meta['output_observations']
        table = list(csv.DictReader((target/'observations.csv').open(newline='', encoding='utf8')))
        assert len(rows) == len(table)
        for frame in frames:
            for row in frame['tracks']:
                if row['observation_kind'] != 'unlinked_person_detection':
                    continue
                for key in ('track_id', 'player_id', 'identity_segment_id', 'role', 'team_id', 'jersey_number', 'court_xy_m'):
                    assert row[key] is None, key
                assert row['detector_source'] == 'yolo26s_detection'
                source = frame['detections'][row['detection_index']]
                assert row['bbox_xyxy'] == source['bbox_xyxy']
                assert row['det_confidence'] == source['det_confidence']
                assert row['silhouette_xy'] is not None or row['draw_detection_box_fallback']
        for row, record in zip(rows, table):
            if row['observation_kind'] == 'unlinked_person_detection':
                assert all(record[k] == '' for k in ('track_id', 'person_id', 'role', 'team_id', 'jersey_number', 'court_x_m', 'court_y_m'))
            assert record['observation_id'] == row['observation_id']
        if url:
            with urlopen(Request(url.rstrip('/')+f'/{clip}/annotated.mp4', headers={'Range': 'bytes=0-127'})) as response:
                assert response.status == 206 and len(response.read()) == 128
        summary[clip] = dict(frames=len(frames), rows=len(rows), added_unlinked=meta['counts']['added_unlinked'])
    report = dict(status='complete', clips=summary, real_detection_sources_verified=True,
                  null_identity_role_team_position_verified=True, csv_nulls_verified=True,
                  hashes_verified=True, software_checks_are_not_accuracy=True)
    write_json(output/'verification.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url')
    a = p.parse_args()
    run(a.output, a.url)
