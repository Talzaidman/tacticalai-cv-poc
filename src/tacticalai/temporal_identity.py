"""Experimental, offline appearance change detection; never edits tracker IDs.

Thresholds are development parameters, not calibrated probabilities. A missing
observation alone is not evidence of a person change. No review labels are read.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalConfig:
    min_samples: int = 3
    spacing: int = 3
    window: int = 5
    min_span: int = 10
    max_overlap: float = 0.25
    min_gain: float = 0.009
    max_within: float = 0.022
    assignment_margin: float = 0.002
    min_similarity: float = 0.97
    assignment_max_overlap: float = 0.60
    shared_color_min: float = 0.18
    shared_color_ratio: float = 0.75
    boundary_context: int = 5
    min_missing_frames: int = 5
    min_center_jump: float = 0.30


def dispersion(x):
    if len(x) < 2:
        return 0.0
    return float(np.median((1 - x @ x.T)[np.triu_indices(len(x), 1)]))


def spaced_indices(observations, indices, spacing):
    """Keep temporally diverse observations; don't count adjacent crops as votes."""
    selected = []
    for i in indices:
        if not selected or observations[i]['frame'] - observations[selected[-1]]['frame'] >= spacing:
            selected.append(i)
    return selected


def detect_change(observations, features, config=TemporalConfig()):
    """Find one persistent appearance-change candidate in an original track.

    Both local and whole-prefix/suffix gains must support the change. The
    uncertain transition interval is left unassigned. One change per track is a
    deliberate first-experiment limitation, recorded in exported metadata.
    """
    cfg = config
    if len(observations) != len(features):
        raise ValueError('Observation/feature row mismatch')
    if not np.isfinite(features).all() or (len(features) and not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4)):
        raise ValueError('Features must be finite normalized vectors')
    if any(a['frame'] >= b['frame'] for a, b in zip(observations, observations[1:])):
        raise ValueError('Observations must have increasing frames')
    clear = spaced_indices(observations, [i for i, o in enumerate(observations)
                                         if o['overlap'] <= cfg.max_overlap and not o['edge']], cfg.spacing)
    candidates = []
    for cut in range(cfg.min_samples, len(clear) - cfg.min_samples + 1):
        left, right = clear[:cut], clear[cut:]
        a, b = left[-cfg.window:], right[:cfg.window]
        if any(observations[s[-1]]['frame'] - observations[s[0]]['frame'] < cfg.min_span for s in (a, b)):
            continue
        wa, wb = dispersion(features[a]), dispersion(features[b])
        between = float(np.median(1 - features[a] @ features[b].T))
        gain = between - (wa + wb) / 2
        global_gain = float(np.median(1 - features[left] @ features[right].T)) - (dispersion(features[left]) + dispersion(features[right])) / 2
        supported = min(gain, global_gain) >= cfg.min_gain and max(wa, wb) <= cfg.max_within
        candidates.append(dict(left_frame=observations[a[-1]]['frame'], right_frame=observations[b[0]]['frame'],
                               local_gain=gain, global_gain=global_gain, within_left=wa, within_right=wb,
                               supported=supported, left_indices=left, right_indices=right))
    eligible = [c for c in candidates if c['supported']]
    if not eligible:
        return None, candidates
    best = max(eligible, key=lambda c: min(c['local_gain'], c['global_gain']))
    # Use broad phase references to reduce sensitivity to an individual pose.
    left, right = best['left_indices'], best['right_indices']
    scores = np.stack([np.median(features @ features[side].T, axis=1) for side in (left, right)], axis=1)
    # Only a veto: if BOTH reference phases have substantial yellow/red pixels,
    # don't use a crop dominated by another color to establish the boundary.
    # Fractions are generic image statistics, not team or identity predictions.
    shared_colors = {}
    for color in ('yellow_fraction', 'red_fraction'):
        minimum = min(float(np.median([observations[i].get(color, 0) for i in side])) for side in (left, right))
        if minimum >= cfg.shared_color_min:
            shared_colors[color] = minimum * cfg.shared_color_ratio
    labels = []
    for o, (a, b) in zip(observations, scores):
        label = None
        color_compatible = all(o.get(color, 0) >= minimum for color, minimum in shared_colors.items())
        if o['overlap'] <= cfg.assignment_max_overlap and not o['edge'] and color_compatible and max(a, b) >= cfg.min_similarity:
            if a - b >= cfg.assignment_margin:
                label = 0
            elif b - a >= cfg.assignment_margin:
                label = 1
        labels.append(label)
    old = [i for i in left if labels[i] == 0]
    new = [i for i in right if labels[i] == 1]
    if len(old) < cfg.min_samples or len(new) < cfg.min_samples:
        return None, candidates
    # Never bridge conflicting phase evidence. Exclude the complete transition
    # envelope from person assignment, including any intervening observations.
    first_new = min([i for i in range(len(observations)) if labels[i] == 1], default=new[0])
    last_old = max([i for i in range(len(observations)) if labels[i] == 0], default=old[-1])
    if first_new <= last_old:
        old_end = max((i for i in old if i < first_new), default=-1)
        new_start = min((i for i in new if i > last_old), default=len(observations))
    else:
        old_end, new_start = last_old, first_new
    if old_end < 0 or new_start >= len(observations):
        return None, candidates
    if sum(i <= old_end for i in old) < cfg.min_samples or sum(i >= new_start for i in new) < cfg.min_samples:
        return None, candidates
    event = dict(last_old_frame=observations[old_end]['frame'], first_new_frame=observations[new_start]['frame'],
                 local_gain=best['local_gain'], global_gain=best['global_gain'],
                 sample_labels=[dict(frame=o['frame'], phase=label, old_similarity=float(s[0]), new_similarity=float(s[1]))
                                for o, label, s in zip(observations, labels, scores)],
                 old_reference_frames=[observations[i]['frame'] for i in left],
                 new_reference_frames=[observations[i]['frame'] for i in right],
                 shared_color_veto=shared_colors,
                 status='automatic_hypothesis')
    return event, candidates


def split_tracks(frames, observations, features, config=TemporalConfig()):
    """Export person-consistent segment hypotheses and null transition rows."""
    tracks, source_rows = {}, {}
    for frame in frames:
        for row in frame['tracks']:
            tracks.setdefault(row['track_id'], []).append(frame['frame_idx'])
            source_rows[frame['frame_idx'], row['track_id']] = row
    segments, events, diagnostics, lookup = {}, [], {}, {}
    for tid, source_frames in sorted(tracks.items()):
        indices = [i for i, o in enumerate(observations) if o['track_id'] == tid]
        indices.sort(key=lambda i: observations[i]['frame'])
        obs = [observations[i] for i in indices]
        change, candidates = detect_change(obs, features[indices], config)
        # Appearance can change persistently when a person simply turns around.
        # Require a nearby source-observation break or an abrupt image-space
        # jump before applying a candidate. Use ALL source rows, not the sampled
        # feature cadence, so sampling itself cannot manufacture a gap.
        if change:
            evidence = []
            lo = change['last_old_frame'] - config.boundary_context
            hi = change['first_new_frame'] + config.boundary_context
            for fa, fb in zip(source_frames, source_frames[1:]):
                if fb < lo or fa > hi:
                    continue
                a = np.asarray(source_rows[fa, tid]['bbox_xyxy'])
                b = np.asarray(source_rows[fb, tid]['bbox_xyxy'])
                jump = float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2) / max(1,a[3]-a[1]))
                missing = fb-fa-1
                if missing >= config.min_missing_frames or jump >= config.min_center_jump:
                    evidence.append(dict(before_frame=fa, after_frame=fb, missing_frames=missing, center_jump_heights=jump))
            if evidence:
                change['tracking_evidence'] = evidence
            else:
                for candidate in candidates:
                    if candidate['supported']:
                        candidate['abstention'] = 'no_tracking_discontinuity'
                change = None
        diagnostics[tid] = candidates
        bounds = [(source_frames[0], source_frames[-1])] if change is None else [
            (source_frames[0], change['last_old_frame']), (change['first_new_frame'], source_frames[-1])]
        if change:
            events.append(dict(track_id=tid, **change))
        for phase, (start, end) in enumerate(bounds):
            sid = f'{tid}.{phase}'
            present = [f for f in source_frames if start <= f <= end]
            segments[sid] = dict(source_track_id=tid, phase=phase, start=start, end=end, frames=present,
                                 feature_indices=[i for i in indices if start <= observations[i]['frame'] <= end])
            for f in present:
                lookup[f, tid] = sid
    return segments, events, diagnostics, lookup


def config_dict(config=TemporalConfig()):
    return asdict(config)
