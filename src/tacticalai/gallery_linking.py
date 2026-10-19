"""Whole-track gallery association: retain every original identity by default."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def gallery_similarity(a, b, comparison='nearest'):
    """Symmetric median of nearest-neighbor cosine similarities.

    Nearest-neighbor gallery distance follows the DeepSORT pattern; the
    symmetric median over multiple queries is our explicit track-level extension.
    Inputs must be nonzero normalized appearance embeddings.
    """
    matrix = np.clip(np.asarray(a) @ np.asarray(b).T, -1, 1)
    if comparison == 'one_to_one':
        rows, cols = linear_sum_assignment(-matrix)
        return float(np.median(matrix[rows, cols]))
    if comparison != 'nearest':
        raise ValueError(f'Unsupported gallery comparison: {comparison}')
    return float((np.median(matrix.max(axis=1)) + np.median(matrix.max(axis=0))) / 2)


def associate_galleries(tracks, galleries, threshold=0.98, margin=0.003, max_gap=150, comparison='nearest', cannot_link=()):
    """One-to-one assignment with unmatched options and temporal exclusion.

    No splitting or new identity allocation. Canonical IDs are existing IDs.
    Tracks are {id: {start, end}}. Scores and margins are not probabilities.
    """
    if not -1 <= threshold < 1 or not 0 <= margin <= 2 or max_gap < 0:
        raise ValueError('Invalid matching thresholds')
    ids = sorted(tracks)
    forbidden = {frozenset(pair) for pair in cannot_link}
    candidates, outgoing, incoming = [], {}, {}
    for a in ids:
        for b in ids:
            if tracks[a]['end'] >= tracks[b]['start']:
                continue
            gap = tracks[b]['start'] - tracks[a]['end'] - 1
            c = dict(before=a, after=b, gap_frames=gap, score=None, accepted=False)
            candidates.append(c)
            if frozenset((a, b)) in forbidden:
                c['reason'] = 'incompatible_segments'
            elif gap > max_gap:
                c['reason'] = 'gap_too_long'
            elif a not in galleries or b not in galleries:
                c['reason'] = 'insufficient_gallery'
            else:
                c['score'] = gallery_similarity(galleries[a], galleries[b], comparison)
                outgoing.setdefault(a, []).append(c)
                incoming.setdefault(b, []).append(c)
    for group in list(outgoing.values()) + list(incoming.values()):
        group.sort(key=lambda c: -c['score'])
    costs = np.full((len(ids), 2 * len(ids)), 1e6)
    costs[:, len(ids):] = 0  # unmatched is always allowed
    positions = {tid: i for i, tid in enumerate(ids)}
    eligible = {}
    for c in candidates:
        if c['score'] is None:
            continue
        out, inc = outgoing[c['before']], incoming[c['after']]
        other_out = [v['score'] for v in out if v is not c]
        other_in = [v['score'] for v in inc if v is not c]
        c['outgoing_margin'] = c['score'] - max(other_out) if other_out else None
        c['incoming_margin'] = c['score'] - max(other_in) if other_in else None
        if c['score'] < threshold:
            c['reason'] = 'below_threshold'
        elif any(m is not None and m < margin for m in (c['outgoing_margin'], c['incoming_margin'])):
            c['reason'] = 'ambiguous'
        else:
            c['reason'] = 'not_selected'
            i, j = positions[c['before']], positions[c['after']]
            costs[i, j] = -(c['score'] - threshold + 1e-9)
            eligible[i, j] = c
    parent = {tid: tid for tid in ids}

    def root(tid):
        while parent[tid] != tid:
            tid = parent[tid]
        return tid

    if ids:
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            if (i, j) in eligible:
                c = eligible[i, j]
                a, b = root(c['before']), root(c['after'])
                members_a = [tid for tid in ids if root(tid) == a]
                members_b = [tid for tid in ids if root(tid) == b]
                if any(frozenset((x, y)) in forbidden for x in members_a for y in members_b):
                    c['reason'] = 'incompatible_component'
                    continue
                parent[b] = a
                c.update(accepted=True, reason='gallery_temporal_hypothesis')
    return {tid: root(tid) for tid in ids}, candidates
