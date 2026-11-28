"""Associate measured SAM 3 hardware detections in source-image coordinates."""

import json
import math

import numpy as np

from .ball import iou
from .export import sha256


KINDS = {"basketball hoop": "hoop", "basketball backboard": "backboard"}
MAX_MISSED_FRAMES = 3
NMS_IOU = 0.65
MIN_MATCH_IOU = 0.15
MAX_CENTER_DISTANCE_IN_BOX_DIAGONALS = 0.60


def track_detections(records, width, height):
    """Match same-kind boxes one-to-one, retaining only measured positions."""
    active = {}
    next_id = 1
    for record in records:
        index = record["frame_idx"]
        active = {key: value for key, value in active.items()
                  if index - value[0] <= MAX_MISSED_FRAMES + 1}
        candidates = []
        for row in sorted(record["detections"], key=lambda row: -row["confidence"]):
            kind = KINDS[row["class_name"]]
            box = np.asarray(row["bbox_xyxy"], dtype=float)
            score = float(row["confidence"])
            if box.shape != (4,) or not np.isfinite(box).all() or not 0 <= score <= 1:
                raise ValueError("Invalid SAM 3 hardware detection")
            box = np.clip(box, 0, [width, height, width, height])
            if np.any(box[2:] <= box[:2]):
                continue
            if any(kind == other_kind and iou(box, other_box) >= NMS_IOU
                   for other_kind, other_box, _ in candidates):
                continue
            candidates.append((kind, box, score))

        matches = []
        for candidate_id, (kind, box, _) in enumerate(candidates):
            for track_id, (_, previous_kind, previous) in active.items():
                if previous_kind != kind:
                    continue
                overlap = iou(previous, box)
                distance = np.linalg.norm((box[:2] + box[2:] - previous[:2] - previous[2:]) / 2)
                scale = min(np.linalg.norm(box[2:] - box[:2]),
                            np.linalg.norm(previous[2:] - previous[:2]))
                relative_distance = distance / scale
                if overlap >= MIN_MATCH_IOU or relative_distance <= MAX_CENTER_DISTANCE_IN_BOX_DIAGONALS:
                    matches.append((-overlap, relative_distance, track_id, candidate_id))
        assignments = {}
        assigned_tracks = set()
        for _, _, track_id, candidate_id in sorted(matches):
            if track_id not in assigned_tracks and candidate_id not in assignments:
                assignments[candidate_id] = track_id
                assigned_tracks.add(track_id)

        tracks = []
        for candidate_id, (kind, box, score) in enumerate(candidates):
            track_id = assignments.get(candidate_id)
            if track_id is None:
                track_id = next_id
                next_id += 1
            active[track_id] = (index, kind, box)
            tracks.append(dict(kind=kind, track_id=track_id, confidence=score,
                               bbox_xyxy=box.tolist(), position_source="sam3_image_detection"))
        yield tracks


def load_hoop_tracks(folder, video, info, ball_path=None):
    """Validate the source identity and exact alignment before attaching tracks."""
    manifest = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
    evidence = folder / "detections.jsonl"
    adaptive = manifest.get("mode") == "adaptive_boxes"
    if (manifest["status"] != "complete" or (not adaptive and manifest["sample_stride"] != 1)
            or set(manifest["prompts"]) != KINDS.keys()):
        raise ValueError("Hoop stage must complete both prompts using full-rate or adaptive detection")
    if adaptive and (ball_path is None or manifest["ball_sha256"] != sha256(ball_path)):
        raise ValueError("Adaptive hoop stage used a different ball cache")
    if (manifest["source_sha256"] != sha256(video)
            or manifest["artifacts_sha256"][evidence.name] != sha256(evidence)):
        raise ValueError("Hoop detections do not match the source video or evidence hash")
    records = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
    indices = [row["frame_idx"] for row in records]
    stride = manifest["sample_stride"]
    invalid_timing = any(
        not 0 <= row["frame_idx"] < info["frames"]
        or not math.isclose(row["timestamp_s"], row["frame_idx"] / info["fps"], abs_tol=1e-6)
        for row in records
    )
    if (stride < 1 or indices != sorted(set(indices)) or invalid_timing
            or not set(range(0, info["frames"], stride)).issubset(indices)):
        raise ValueError("Hoop detections do not align with the source timeline")
    provenance = dict(manifest, association=dict(
        nms_iou=NMS_IOU, min_match_iou=MIN_MATCH_IOU, max_missed_frames=MAX_MISSED_FRAMES,
        max_center_distance_in_box_diagonals=MAX_CENTER_DISTANCE_IN_BOX_DIAGONALS,
        coordinates="source_image_pixels", position_interpolation=False))
    tracks = [None] * info["frames"]
    for index, matched in zip(indices, track_detections(records, info["width"], info["height"]), strict=True):
        tracks[index] = matched
    return tracks, provenance
