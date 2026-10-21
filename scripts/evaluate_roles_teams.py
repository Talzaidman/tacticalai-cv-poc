"""Evaluation-only role/team labels never enter role_team_cli.

Prepare unlabeled contact sheets from a fixed sample before reading predictions,
then score optional human labels. No label or annotation changes predictions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from tacticalai.export import sha256, write_json


def prepare(source, target):
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    for clip in ('attack_clip', 'attack_clip2'):
        frames = [json.loads(l) for l in (source/clip/'frames.jsonl').read_text().splitlines()]
        video = cv2.VideoCapture(str(Path('data/raw')/(clip+'.mp4')))
        for index in (37, 112):
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = video.read()
            if not ok:
                raise ValueError('Evaluation frame unavailable')
            rows = frames[index]['tracks']
            cells = []
            for row in rows:
                x1, y1, x2, y2 = np.rint(row['bbox_xyxy']).astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                crop = image[y1:y2, x1:x2]
                scale = min(150/crop.shape[1], 185/crop.shape[0])
                resized = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
                cell = np.full((230, 180, 3), (30, 30, 30), np.uint8)
                h, w = resized.shape[:2]; cell[27:27+h, (180-w)//2:(180-w)//2+w] = resized
                cv2.putText(cell, f"f{index} T{row['track_id']}", (7, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
                cells.append(cell)
                entries.append(dict(clip=clip, frame_idx=index, track_id=row['track_id'], role=None, uniform=None))
            cols = 6
            while len(cells) % cols:
                cells.append(np.zeros_like(cells[0]))
            sheet = np.vstack([np.hstack(cells[i:i+cols]) for i in range(0, len(cells), cols)])
            cv2.imwrite(str(target/f'{clip}-{index}.jpg'), sheet)
        video.release()
    write_json(target/'labels-template.json', dict(
        scope='Small development-clip spot check; not held-out-video validation.',
        role_values=['player', 'referee', 'other_person', 'uncertain'],
        uniform_values=['dark', 'light', 'uncertain', None], observations=entries))


def evaluate(source, labels_path):
    labels = json.loads(labels_path.read_text())
    rows = {}
    for clip in ('attack_clip', 'attack_clip2'):
        for line in (source/clip/'frames.jsonl').read_text().splitlines():
            frame = json.loads(line)
            for r in frame['tracks']:
                rows[clip, frame['frame_idx'], r['track_id']] = r
    confusion = Counter(); valid = []; team_rows = []
    for label in labels['observations']:
        prediction = rows[label['clip'], label['frame_idx'], label['track_id']]
        if label['role'] in ('player', 'referee', 'other_person'):
            valid.append((label, prediction))
            confusion[label['role'], prediction['role'] or 'unknown'] += 1
        if label['role'] == 'player' and label.get('uniform') in ('dark', 'light'):
            team_rows.append((label, prediction))
    correct = sum(l['role'] == p['role'] for l, p in valid)
    classified = sum(p['role'] is not None for _, p in valid)
    team_correct = team_assigned = 0; mappings = {}
    for clip in ('attack_clip', 'attack_clip2'):
        subset = [(l, p) for l, p in team_rows if l['clip'] == clip]
        options = [dict(A='dark', B='light'), dict(A='light', B='dark')]
        mapping = max(options, key=lambda m: sum(m.get(p['team_id']) == l['uniform'] for l, p in subset))
        mappings[clip] = mapping
        team_correct += sum(mapping.get(p['team_id']) == l['uniform'] for l, p in subset)
        team_assigned += sum(p['team_id'] is not None for _, p in subset)
    report = dict(scope=labels['scope'], labels_sha256=sha256(labels_path),
        evaluator_sha256=sha256(Path(__file__)),
        prediction_files_sha256={c: sha256(source/c/'frames.jsonl') for c in ('attack_clip', 'attack_clip2')},
        labels_used_for_inference=False, uncertain_labels_excluded=sum(l['role'] not in ('player', 'referee', 'other_person') for l in labels['observations']),
        role=dict(labeled_observations=len(valid), correct=correct, classified=classified,
                  accuracy_including_abstentions=correct/len(valid) if valid else None,
                  accuracy_when_classified=correct/classified if classified else None,
                  confusion=[dict(truth=a, predicted=b, count=n) for (a, b), n in confusion.items()]),
        team=dict(labeled_player_observations=len(team_rows), correct=team_correct, assigned=team_assigned,
                  accuracy_including_abstentions=team_correct/len(team_rows) if team_rows else None,
                  accuracy_when_assigned=team_correct/team_assigned if team_assigned else None,
                  evaluation_only_permutation=mappings),
        summary=f'Development-clip visual spot check: roles {correct}/{len(valid)} correct including unknowns '
                f'({classified} assigned); teams {team_correct}/{len(team_rows)} correct including unknowns '
                f'({team_assigned} assigned). Anonymous team labels are matched by permutation for scoring only. '
                'Small, correlated samples from previously used clips; no held-out-video accuracy claim.')
    write_json(source/'evaluation.json', report)
    print(report['summary'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/restored-identity-baseline-v1'))
    p.add_argument('--prepare', type=Path)
    p.add_argument('--labels', type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.source, a.prepare)
    elif a.labels:
        evaluate(a.source, a.labels)
    else:
        p.error('Specify --prepare or --labels')
