"""Compare raw detections and tracks against evaluation-only visible points."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from tacticalai.court_gate import court_gate
from tacticalai.coverage_evaluation import match_visible_points
from tacticalai.export import sha256,write_json


def run(labels,investigation):
    annotations=json.loads(labels.read_text())
    summary=defaultdict(lambda:dict(targets=0,covered=0,partial_targets=0,partial_covered=0,rows=0))
    detail=[]
    for sample in annotations['frames']:
        clip,index=sample['clip'],sample['frame_idx']
        original=Path('outputs/occlusion-20261015T204150Z-029a7a35/reid')/clip/'tracktrack'
        meta=json.loads((original/'metadata.json').read_text());info=meta['source'];size=(info['width'],info['height'])
        baseline=[json.loads(l) for l in (original/'frames.jsonl').read_text().splitlines()][index]
        camera=[json.loads(l) for l in (Path('outputs/challenge-player-smoothed-v1')/clip/'calibration.jsonl').read_text().splitlines()][index]
        variants=dict(original_tracks=baseline['tracks'],original_detections=baseline['detections'])
        for name in ('control','lower_gates','court','court_confirm1','court_overlap'):
            frames=[json.loads(l) for l in (investigation/'tracker-replay'/clip/name/'frames.jsonl').read_text().splitlines()]
            variants[name]=frames[index]['tracks']
            if name=='court_overlap':variants['court_all_observations']=frames[index]['detections']
        safe=investigation/'tracker-replay-safe'
        if (safe/'experiment.json').exists() and json.loads((safe/'experiment.json').read_text())['status']=='complete':
            for name in ('court','court_confirm1','court_overlap'):
                frames=[json.loads(l) for l in (safe/clip/name/'frames.jsonl').read_text().splitlines()]
                variants['safe_'+name]=frames[index]['tracks']
                if name=='court':variants['safe_court_all_observations']=frames[index]['detections']
        for model in ('small','medium','small-e2e'):
            folder=investigation/f'detector-{model}'
            if not (folder/'metadata.json').exists():continue
            if json.loads((folder/'metadata.json').read_text())['status']!='complete':continue
            for resolution in (1280,1920):
                path=folder/f'{clip}-{resolution}.json'
                if not path.exists():continue
                frames=json.loads(path.read_text())['frames']
                detections=next(f['detections'] for f in frames if f['frame_idx']==index)
                for confidence in (.03,.05,.1):
                    rows=[r for r in detections if r['det_confidence']>=confidence]
                    variants[f'{model}_{resolution}_{confidence}']=rows
                    variants[f'{model}_{resolution}_{confidence}_court']=[r for r in rows if court_gate(r['bbox_xyxy'],
                        camera['refined_image_to_court_h'],camera['temporal_quality'],size)['keep']]
        for name,rows in variants.items():
            pairs=match_visible_points(sample['points'],rows,size)
            matched={p for p,r in pairs}
            partial=sum(p['partially_visible'] for p in sample['points'])
            partial_covered=sum(p['partially_visible'] for i,p in enumerate(sample['points']) if i in matched)
            for split in ('all',sample['split']):
                counts=summary[f'{name}/{split}']
                for key,value in dict(targets=len(sample['points']),covered=len(matched),partial_targets=partial,
                                      partial_covered=partial_covered,rows=len(rows)).items():counts[key]+=value
            detail.append(dict(clip=clip,frame_idx=index,variant=name,targets=len(sample['points']),covered=len(matched),
                partial_targets=partial,partial_covered=partial_covered,rows=len(rows),
                missing_point_indices=[i for i in range(len(sample['points'])) if i not in matched]))
    write_json(investigation/'evaluation.json',dict(method='One-to-one inspected visible-head point coverage; not box AP or precision',
        labels_sha256=sha256(labels),labels_used_for_inference=False,summary=dict(summary),frames=detail))
    for key,counts in summary.items():
        if key.endswith('/all'):print(key,counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,default=Path('configs/evaluation/person_coverage_points_v1.json'))
    parser.add_argument('--investigation',type=Path,default=Path('outputs/person-coverage-investigation'))
    args=parser.parse_args();run(args.labels,args.investigation)
