"""Evaluate saved identity mappings against explicit user-reviewed development pairs."""
import argparse
import html
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('outputs/gallery-reid-final'))
    parser.add_argument('--labels', type=Path, default=Path('configs/reid_identity_reviews.json'))
    parser.add_argument('--switches', type=Path, default=Path('configs/reid_identity_switch_reviews.json'))
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())
    rows, counts = [], dict(correct_merge=0, false_merge=0, missed_connection=0, correct_rejection=0)
    for pair in labels['pairs']:
        clip = pair['clip']
        mapping = json.loads((args.results / clip / 'mapping.json').read_text())
        a, b = pair['track_ids']
        predicted = mapping[str(a)] == mapping[str(b)]
        actual = pair['same_person']
        outcome = ('correct_merge' if actual else 'false_merge') if predicted else (
            'missed_connection' if actual else 'correct_rejection')
        counts[outcome] += 1
        rows.append(dict(pair, model_same_identity=predicted, outcome=outcome))
    unreviewed = []
    for clip in {p['clip'] for p in labels['pairs']}:
        known = {tuple(sorted(p['track_ids'])) for p in labels['pairs'] if p['clip'] == clip}
        candidates = json.loads((args.results / clip / 'candidates.json').read_text())
        unreviewed.extend(dict(clip=clip, track_ids=[c['before'],c['after']], score=c['score'])
                          for c in candidates if c['accepted'] and tuple(sorted([c['before'],c['after']])) not in known)
    switch_review = None
    if args.switches.is_file():
        proposed_switch_review = json.loads(args.switches.read_text())
        result_meta_path = args.results / proposed_switch_review['clip'] / 'metadata.json'
        if result_meta_path.is_file():
            result_meta = json.loads(result_meta_path.read_text())
            if result_meta.get('source_frames_sha256') == proposed_switch_review['source_frames_sha256']:
                switch_review = proposed_switch_review
                (args.results / 'identity-switch-events.json').write_text(json.dumps(switch_review, indent=2))
    report = dict(split=labels['split'], reviewed_pairs=len(rows), counts=counts, pairs=rows,
                  unreviewed_merges=unreviewed,
                  reported_switch_events=len(switch_review['events']) if switch_review else 0,
                  note='User-reviewed development subset, not held-out whole-video accuracy. Model mappings unchanged.')
    (args.results / 'user-review.json').write_text(json.dumps(report, indent=2))
    # Add review annotations, leaving all model decisions/scores unchanged.
    for clip in {r['clip'] for r in rows}:
        path = args.results / clip / 'review-data.json'
        data = json.loads(path.read_text())
        lookup = {tuple(sorted(r['track_ids'])): r['same_person'] for r in rows if r['clip'] == clip}
        for c in data['candidates']:
            c['user_review_same_person'] = lookup.get(tuple(sorted([c['before'], c['after']])))
        path.write_text(json.dumps(data, indent=2))
    table = ''.join(f'<tr><td>{html.escape(r["clip"])}</td><td>{r["track_ids"][0]} ↔ {r["track_ids"][1]}</td>'
                    f'<td>{"Same person" if r["same_person"] else "Different people"}</td>'
                    f'<td>{html.escape(r["outcome"].replace("_", " "))}</td></tr>' for r in rows)
    proposed = counts['correct_merge'] + counts['false_merge']
    switch_block = ''
    if switch_review:
        event_rows = ''
        for event in switch_review['events']:
            first = event['first_verified_new_person_frame']
            when = f'Frame {first}' if first is not None else 'Switch frame not yet verified'
            chains = event.get('physical_person_chains', [event.get('physical_person_chain', [])])
            buttons = ''
            for chain in chains:
                if len(chain) == 2 and all(p['track_id'] is not None for p in chain):
                    a,b = (p['track_id'] for p in chain)
                    frame = first-1 if first is not None else 0
                    buttons += f'<button onclick="inspectOriginalIds({a},{b},{frame})">Inspect {a}/{b}</button> '
            event_rows += (f'<tr><td>ID {event["reused_track_id"]}</td>'
                           f'<td>{html.escape(event["old_person"])}</td>'
                           f'<td>{html.escape(event["new_person"])}</td><td>{when}<br>{buttons}</td></tr>')
        switch_block = ('<section style="border-left:4px solid #ffb36b;padding:14px;background:#392b26">'
                        f'<h2>{len(switch_review["events"])} reported ID-switch cases remain in clip 1</h2>'
                        '<p>Confirmed pair merges do not fix a label that changes physical person. '
                        'These cases are recorded but have not been repaired in the model output. '
                        'A track must be separated at a verified switch before its parts can be linked safely.</p>'
                        '<table style="width:100%;text-align:left"><tr><th>Reused label</th><th>Earlier person</th>'
                        '<th>Later person</th><th>Boundary</th></tr>'+event_rows+'</table>'
                        '<p>Whole-ID merges of 87/96, 89/91, 91/105 or 85/102 would mix different people. '
                        'The physical-person chains are in <a href="identity-switch-events.json">the review record</a>.</p></section>')
    block = (f'<!-- USER_REVIEW_START --><section class="panel"><h2>Your review: {counts["correct_merge"]} of {proposed} reviewed proposals confirmed</h2>'
             f'<p>{counts["false_merge"]} false merge(s) and {counts["missed_connection"]} missed connection(s). '
             f'There are also {len(unreviewed)} unreviewed proposed merge(s). '
             f'These are {len(rows)} reviewed development pairs, not full-video accuracy. '
             'Counts below still show the model output; review labels have not silently changed it.</p>'
             f'<details><summary>All {len(rows)} judgments</summary><table style="width:100%;text-align:left">'
             '<tr><th>Video</th><th>IDs</th><th>Your judgment</th><th>Model result</th></tr>'
             + table + '</table></details></section>'+switch_block+'<!-- USER_REVIEW_END -->')
    page_path = args.results / 'index.html'
    page = page_path.read_text(encoding='utf-8')
    page = re.sub(r'<!-- USER_REVIEW_START -->[\s\S]*?<!-- USER_REVIEW_END -->', '', page)
    page = page.replace('<div id="verdict">', block + '<div id="verdict">', 1)
    page_path.write_text(page, encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
