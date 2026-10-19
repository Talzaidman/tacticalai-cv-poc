"""Evaluate immutable temporal predictions; annotate the page with review facts."""
import argparse
import json
from pathlib import Path

from tacticalai.export import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--pairs', type=Path, default=Path('configs/reid_identity_reviews.json'))
    p.add_argument('--temporal-checks', type=Path, default=Path('configs/reid_temporal_checks.json'))
    p.add_argument('--switch-reviews', type=Path, default=Path('configs/reid_identity_switch_reviews.json'))
    a = p.parse_args()
    labels = json.loads(a.pairs.read_text())
    checks = json.loads(a.temporal_checks.read_text())
    switches = json.loads(a.switch_reviews.read_text())
    reports = []
    for clip in ('attack_clip','attack_clip2'):
        folder = a.results / clip
        data = json.loads((folder/'review-data.json').read_text())
        meta = json.loads((folder/'metadata.json').read_text())
        source = Path(labels['source_runs'][clip])/'frames.jsonl'
        if sha256(source) != meta['source_frames_sha256']:
            raise ValueError('Review labels refer to a different source')
        mapping = data['mapping']
        pairs = []
        for pair in labels['pairs']:
            if pair['clip'] != clip:
                continue
            x,y = (f'{tid}.0' for tid in pair['track_ids'])
            predicted = x in mapping and y in mapping and mapping[x] == mapping[y]
            pairs.append(dict(pair, early_phase_same_identity=predicted, agrees_with_review=predicted==pair['same_person']))
        chains, separated, boundaries = [], [], []
        reported = []
        if clip == checks['clip']:
            if checks['source_frames_sha256'] != meta['source_frames_sha256'] or switches['source_frames_sha256'] != meta['source_frames_sha256']:
                raise ValueError('Temporal checks refer to a different source')
            for chain in checks['chains']:
                x,y = chain['before'],chain['after']
                chains.append(dict(chain, recovered=x in mapping and y in mapping and mapping[x]==mapping[y]))
            for tid in checks['must_separate']:
                x,y = f'{tid}.0',f'{tid}.1'
                separated.append(dict(track_id=tid,separate=x in mapping and y in mapping and mapping[x]!=mapping[y]))
            for expected in checks['verified_boundaries']:
                event = next((e for e in data['events'] if e['track_id']==expected['track_id']),None)
                boundaries.append(dict(expected, matches=event is not None and all(event[k]==v for k,v in expected.items())))
            reported = [e['reused_track_id'] for e in switches['events']]
        report = dict(clip=clip, positive_total=sum(p['same_person'] for p in pairs),
                      positive_recovered=sum(p['same_person'] and p['early_phase_same_identity'] for p in pairs),
                      negative_total=sum(not p['same_person'] for p in pairs),
                      negative_merged=sum(not p['same_person'] and p['early_phase_same_identity'] for p in pairs),
                      chains_total=len(chains),chains_recovered=sum(c['recovered'] for c in chains),
                      pairs=pairs, chains=chains, split_checks=separated, boundary_checks=boundaries,
                      reported_switch_ids=reported,
                      unreviewed_switch_ids=[e['track_id'] for e in data['events'] if e['track_id'] not in reported],
                      note='Development feedback; no frame-level or held-out identity accuracy. Negative 85/106 check refers to the original early player, since 85 is mixed.')
        data.update(development_review=report,reported_switch_ids=reported)
        write_json(folder/'review-data.json',data)
        reports.append(report)
    write_json(a.results/'development-review.json',dict(split='development',clips=reports,
               prediction_files_unchanged=True,review_files_sha256={p.name:sha256(p) for p in (a.pairs,a.temporal_checks,a.switch_reviews)}))
    print(json.dumps(reports,indent=2))


if __name__ == '__main__':
    main()
