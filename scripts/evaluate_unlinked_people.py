"""Evaluation-only head-point coverage before/after adding unlinked people."""
import argparse
from collections import Counter
import json
from pathlib import Path

from tacticalai.coverage_evaluation import match_visible_points
from tacticalai.export import sha256, write_json


def run(output, annotations):
    labels = json.loads(annotations.read_text())
    manifest = json.loads((output/'experiment.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete output')
    data, sizes = {}, {}
    for clip in ('attack_clip', 'attack_clip2'):
        metadata = manifest['clips'][clip]
        if metadata['video']['sha256'] != labels['video_sha256'][clip]:
            raise ValueError('Annotation video differs')
        data[clip] = [json.loads(l) for l in (output/clip/'frames.jsonl').read_text().splitlines()]
        sizes[clip] = metadata['video']['width'], metadata['video']['height']
    counts, evidence = Counter(), []
    for sample in labels['frames']:
        points = sample['points']
        frame = data[sample['clip']][sample['frame_idx']]
        counts.update(total=len(points), partially_visible=sum(p['partially_visible'] for p in points))
        record = dict(clip=sample['clip'], frame_idx=sample['frame_idx'], split=sample['split'])
        for mode, rows in [('before', [r for r in frame['tracks'] if r['observation_kind'] == 'original_track']),
                           ('after', frame['tracks'])]:
            pairs = match_visible_points(points, rows, sizes[sample['clip']])
            counts.update({mode: len(pairs), mode+'_partial': sum(points[i]['partially_visible'] for i, _ in pairs)})
            record[mode] = dict(covered=len(pairs), matches=[dict(point_index=i,
                observation_id=rows[j]['observation_id']) for i, j in pairs])
        evidence.append(record)
    report = dict(counts=dict(counts), frames=len(evidence), annotations_sha256=sha256(annotations),
        script_sha256=sha256(Path(__file__)), annotations_used_for_inference=False,
        note='Inspected head-point coverage on previously used development clips. Not box AP, precision, role/team/identity accuracy or held-out validation.',
        frames_evidence=evidence)
    write_json(output/'coverage-evaluation.json', report)
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--annotations', type=Path, default=Path('configs/evaluation/person_coverage_points_v1.json'))
    args = p.parse_args()
    run(args.output, args.annotations)
