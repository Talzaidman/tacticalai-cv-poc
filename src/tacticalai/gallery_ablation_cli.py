"""Evaluate gallery changes against reviewed pairs without manual ID overrides."""
import argparse
import html
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .export import sha256, write_json
from .gallery_reid_cli import build_galleries
from .gallery_linking import associate_galleries

VARIANTS = [
    ('baseline', 'nearest', 0, 1.0, False, 0.003),
    ('one_to_one_only', 'one_to_one', 0, 1.0, False, 0.003),
    ('spacing_only', 'nearest', 5, 1.0, False, 0.003),
    ('spacing_one_to_one', 'one_to_one', 5, 1.0, False, 0.003),
    ('strict_overlap', 'nearest', 5, 0.15, False, 0.003),
    ('strict_overlap_one_to_one', 'one_to_one', 5, 0.15, False, 0.003),
    ('enriched_nearest', 'nearest', 5, 1.0, True, 0.003),
    ('enriched_one_to_one', 'one_to_one', 5, 1.0, True, 0.003),
    ('enriched_one_to_one_calibrated', 'one_to_one', 5, 1.0, True, 0.001),
]


def run(original, enriched, labels_path, output):
    output.mkdir(parents=True, exist_ok=False)
    labels = json.loads(labels_path.read_text())
    report = dict(split='development', labels_sha256=sha256(labels_path),
                  code_sha256=sha256(Path(__file__)),
                  versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','opencv-python')},
                  threshold=0.98, default_margin=0.003, variants=[],
                  calibration_note='After the first eight variants, enriched one-to-one margins 0.0005, 0.001, 0.0015, 0.002 and 0.003 were inspected on these development labels. 0.001 is a development choice, not held-out validation.',
                  note='Reviewed development pairs only. Selection and comparison rules use no ID-specific labels. '
                       'No retraining, new tracker inference or manual identity overrides. Unreviewed merges are unknown.')
    for name, comparison, spacing, overlap, fresh, margin in VARIANTS:
        if fresh and enriched is None:
            continue
        cache_root = enriched if fresh else original
        record = dict(name=name, comparison=comparison, min_spacing=spacing, max_overlap=overlap,
                      enriched=fresh, margin=margin, clips=[], pairs=[],
                      counts=dict(correct_merge=0,false_merge=0,missed_connection=0,correct_rejection=0),
                      unreviewed_merges=0)
        for clip in ('attack_clip','attack_clip2'):
            cache=cache_root/clip
            meta=json.loads((cache/'metadata.json').read_text())
            source=Path(meta['config']['run'])/'frames.jsonl'
            if meta['status']!='complete' or sha256(source)!=meta['source_frames_sha256']:
                raise ValueError('Incomplete or incompatible embedding cache')
            frames=[json.loads(line) for line in source.read_text().splitlines()]
            segments=json.loads((cache/'tracklets.json').read_text())
            folder=output/name/clip
            (folder/'gallery').mkdir(parents=True)
            with np.load(cache/'embeddings.npz',allow_pickle=False) as values:
                tracks,galleries,crops=build_galleries(frames,segments,values['embeddings'],values['segment_ids'],cache,folder,
                                                      min_spacing=spacing,max_overlap=overlap)
            mapping,candidates=associate_galleries(tracks,galleries,comparison=comparison,margin=margin)
            assert set(mapping.values()).issubset(tracks)
            assert len(set(mapping.values()))<=len(tracks)
            write_json(folder/'mapping.json',mapping)
            write_json(folder/'candidates.json',candidates)
            write_json(folder/'galleries.json',crops)
            record.setdefault('diagnostic_galleries',{})[clip]={
                str(tid):[dict(sample,image=f'{name}/{clip}/{sample["image"]}') for sample in crops[tid]]
                for tid in tracks if tid in ({85,106} if clip=='attack_clip' else {212,219,228})}
            reviewed={tuple(sorted(p['track_ids'])) for p in labels['pairs'] if p['clip']==clip}
            accepted=[c for c in candidates if c['accepted']]
            record['unreviewed_merges']+=sum(tuple(sorted([c['before'],c['after']])) not in reviewed for c in accepted)
            record['clips'].append(dict(clip=clip,before=len(tracks),after=len(set(mapping.values())),
                                        accepted=accepted,embeddings_sha256=sha256(cache/'embeddings.npz'),
                                        checkpoint_sha256=meta['checkpoint_sha256'],source_sha256=sha256(source)))
            for pair in labels['pairs']:
                if pair['clip']!=clip:
                    continue
                a,b=pair['track_ids']
                predicted=mapping[a]==mapping[b]
                actual=pair['same_person']
                outcome=('correct_merge' if actual else 'false_merge') if predicted else (
                    'missed_connection' if actual else 'correct_rejection')
                c=next((c for c in candidates if {c['before'],c['after']}=={a,b}),None)
                record['counts'][outcome]+=1
                record['pairs'].append(dict(pair,predicted_same=predicted,outcome=outcome,
                                            score=c['score'] if c else None,reason=c['reason'] if c else 'temporal_overlap',
                                            gallery_sizes=[len(crops[a]),len(crops[b])]))
        report['variants'].append(record)
    write_json(output/'ablation.json',report)
    (output/'index.html').write_text(render(report),encoding='utf-8')
    print(json.dumps([{k:v[k] for k in ('name','counts','unreviewed_merges','clips')} for v in report['variants']],indent=2))


def render(report):
    positive_count=sum(p['same_person'] for p in report['variants'][0]['pairs'])
    pair_count=len(report['variants'][0]['pairs'])
    rows=''
    for v in report['variants']:
        c=v['counts']
        totals=' / '.join(str(x['after']) for x in v['clips'])
        rows+=f'<tr><td>{html.escape(v["name"])}</td><td>{v["margin"]}</td><td>{c["correct_merge"]}/{positive_count}</td><td>{c["false_merge"]}</td><td>{c["missed_connection"]}</td><td>{v["unreviewed_merges"]}</td><td>{totals}</td></tr>'
    headers=''.join(f'<th>{html.escape(v["name"])}</th>' for v in report['variants'])
    comparisons=''
    for index,pair in enumerate(report['variants'][0]['pairs']):
        a,b=pair['track_ids']
        comparisons+=f'<tr><td>{pair["clip"]}<br>{a} ↔ {b}<br>User: {"same" if pair["same_person"] else "different"}</td>'
        for v in report['variants']:
            p=v['pairs'][index]
            good=p['outcome'] in ('correct_merge','correct_rejection')
            comparisons+=f'<td class="{"good" if good else "bad"}">{p["outcome"].replace("_"," ")}<br><small>{p["reason"].replace("_"," ")}</small></td>'
        comparisons+='</tr>'
    diagnostics=''
    for v in report['variants']:
        if v['name'] not in {'baseline','spacing_one_to_one','enriched_one_to_one'}:
            continue
        diagnostics+=f'<h3>{html.escape(v["name"])}</h3>'
        for clip,ids in v['diagnostic_galleries'].items():
            for tid,crops in ids.items():
                images=''.join(f'<figure><img loading="lazy" src="{s["image"]}" alt="ID {tid}, frame {s["frame"]}">'
                               f'<figcaption>Frame {s["frame"]}</figcaption></figure>' for s in crops)
                diagnostics+=f'<p>{clip} · ID {tid} · {len(crops)} selected crop(s)</p><div class="crops">{images}</div>'
    return ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Gallery matching experiments</title><style>body{font:16px system-ui;background:#111a27;color:#e9f1fb;margin:28px}'
            'table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #40516c;padding:12px;text-align:left}'
            '.good{background:#183b31}.bad{background:#492c33}small{color:#bcc9db}a{color:#87cfff}.scroll{overflow:auto}'
            '.crops{display:flex;gap:8px;overflow:auto}figure{margin:0}img{height:150px;max-width:130px;object-fit:contain}figcaption{font-size:12px;color:#b9c7d9}</style>'
            f'<h1>Did the changes improve matching?</h1><p>Same original 18 IDs per clip. {pair_count} user-reviewed development pairs: {positive_count} same-person pairs and {pair_count-positive_count} different-person pairs. '
            'Threshold 0.98 remains fixed. The first eight variants use margin 0.003; the final variant calibrates it to 0.001 on these same development labels. A smaller ID count can still contain wrong merges.</p>'
            '<p><a href="candidate/index.html">Open the selected candidate: videos and gallery evidence</a></p>'
            '<div class="scroll"><table><tr><th>Change</th><th>Margin</th><th>Known correct recovered</th><th>Known false merges</th><th>Known missed</th><th>Unreviewed merges</th><th>Total IDs, clips 1 / 2</th></tr>'
            +rows+'</table></div><h2>Every reviewed pair</h2><div class="scroll"><table><tr><th>Pair</th>'+headers+'</tr>'+comparisons+'</table></div>'
            '<details><summary>Inspect the changed reference crops for the two difficult cases</summary>'+diagnostics+'</details>'
            '<p>Green: agrees with your review. Red: disagrees. A correct rejection may be an abstention due to insufficient evidence, not recognition of a different person. '
            'These are development results, not held-out accuracy. No labels are used to directly force a merge or split.</p>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,default=Path('outputs/clip-reident-paper-v1'))
    parser.add_argument('--enriched',type=Path)
    parser.add_argument('--labels',type=Path,default=Path('configs/reid_identity_reviews.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.original,args.enriched,args.labels,args.output)


if __name__=='__main__':
    main()
