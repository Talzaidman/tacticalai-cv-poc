"""Report display-filter coverage changes using evaluation-only head points."""
import argparse
from collections import Counter
import json
from pathlib import Path

from tacticalai.coverage_evaluation import match_visible_points
from tacticalai.export import sha256, write_json


def run(output, annotations):
    labels = json.loads(annotations.read_text())
    manifest = json.loads((output/'experiment.json').read_text())
    counts = Counter(); changes = []
    for clip, metadata in manifest['clips'].items():
        if metadata['video']['sha256'] != labels['video_sha256'][clip]:
            raise ValueError('Annotation/video mismatch')
        info = metadata['video']; size = info['width'], info['height']
        frames = [json.loads(l) for l in (output/clip/'frames.jsonl').read_text().splitlines()]
        for sample in labels['frames']:
            if sample['clip'] != clip:
                continue
            rows = frames[sample['frame_idx']]['tracks']; points = sample['points']
            before = match_visible_points(points, rows, size)
            after = match_visible_points(points, [r for r in rows if r['display_on_video']], size)
            counts.update(total=len(points), before=len(before), after=len(after),
                partially_visible=sum(p['partially_visible'] for p in points),
                before_partial=sum(points[i]['partially_visible'] for i,j in before),
                after_partial=sum(points[i]['partially_visible'] for i,j in after))
            retained = {i for i,j in after}
            for i,j in before:
                if i not in retained:
                    changes.append(dict(clip=clip, frame_idx=sample['frame_idx'], point_index=i,
                        point_xy=points[i]['point_xy_1280x720'],
                        observation_id=rows[j]['observation_id'], predicted_role=rows[j].get('role'),
                        display_filter_reason=rows[j]['display_filter_reason']))
    result = dict(counts=dict(counts), removed_coverage_points=changes,
        annotations_sha256=sha256(annotations), script_sha256=sha256(Path(__file__)),
        annotations_used_for_filter=False,
        note='Development-clip point coverage only, including sideline people. Does not label who should be filtered, measure filter precision or establish held-out accuracy.')
    write_json(output/'display-evaluation.json', result)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--annotations', type=Path, default=Path('configs/evaluation/person_coverage_points_v1.json'))
    a = p.parse_args(); run(a.output, a.annotations)
