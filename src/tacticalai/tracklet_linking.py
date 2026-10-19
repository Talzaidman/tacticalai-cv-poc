"""Offline appearance linking with explicit evidence and unchanged source IDs."""
import numpy as np


def split_tracklets(frames):
    """Split at every missing frame; a repeated numeric ID is not ground truth."""
    segments, latest = [], {}
    for frame in frames:
        for row in frame["tracks"]:
            tid = row["track_id"]
            if tid is None:
                continue
            idx = frame["frame_idx"]
            segment = latest.get(tid)
            if segment is None or segment["end"] != idx - 1:
                segment = dict(segment_id=len(segments), source_track_id=tid,
                               start=idx, end=idx, observations=[])
                segments.append(segment)
                latest[tid] = segment
            segment["end"] = idx
            segment["observations"].append(row)
    return segments


def link_tracklets(segments, features, threshold=0.85, margin=0.03, max_gap=150):
    """Mutual-best temporal links with two-sided ambiguity rejection.

    Thresholds are development hypotheses, not calibrated probabilities.
    Features map segment IDs to normalized descriptors of sampled observations.
    Missing/short segments remain unlinked. Components must be time-disjoint.
    """
    if not -1 <= threshold <= 1 or not 0 <= margin <= 2 or max_gap < 1:
        raise ValueError("Invalid linking thresholds")
    candidates = []
    for a in segments:
        for b in segments:
            gap = b["start"] - a["end"] - 1
            if a["segment_id"] not in features or b["segment_id"] not in features or not 0 < gap <= max_gap:
                continue
            score = float(np.clip(np.dot(features[a["segment_id"]], features[b["segment_id"]]), -1, 1))
            candidates.append(dict(before=a["segment_id"], after=b["segment_id"],
                                   gap_frames=gap, cosine_similarity=score, accepted=False))
    outgoing, incoming = {}, {}
    for candidate in candidates:
        outgoing.setdefault(candidate["before"], []).append(candidate)
        incoming.setdefault(candidate["after"], []).append(candidate)
    for group in list(outgoing.values()) + list(incoming.values()):
        group.sort(key=lambda c: (-c["cosine_similarity"], c["before"], c["after"]))
    parent = {s["segment_id"]: s["segment_id"] for s in segments}
    intervals = {s["segment_id"]: [(s["start"], s["end"])] for s in segments}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for c in sorted(candidates, key=lambda c: -c["cosine_similarity"]):
        out, inc = outgoing[c["before"]], incoming[c["after"]]
        c["outgoing_margin"] = c["cosine_similarity"] - out[1]["cosine_similarity"] if len(out) > 1 else None
        c["incoming_margin"] = c["cosine_similarity"] - inc[1]["cosine_similarity"] if len(inc) > 1 else None
        if c["cosine_similarity"] < threshold:
            c["reason"] = "below_threshold"
        elif c is not out[0] or c is not inc[0]:
            c["reason"] = "not_mutual_best"
        elif any(m is not None and m < margin for m in (c["outgoing_margin"], c["incoming_margin"])):
            c["reason"] = "ambiguous"
        else:
            a, b = root(c["before"]), root(c["after"])
            if a == b or any(max(x1, y1) <= min(x2, y2)
                             for x1, x2 in intervals[a] for y1, y2 in intervals[b]):
                c["reason"] = "overlapping_identity_segments"
            else:
                parent[b] = a
                intervals[a].extend(intervals[b])
                c.update(accepted=True, reason="appearance_temporal_hypothesis")
    return {i: root(i) + 1 for i in parent}, candidates
