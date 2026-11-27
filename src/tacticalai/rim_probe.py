"""Utilities for a sampled open-vocabulary basketball-rim detector probe."""

from __future__ import annotations

from collections import Counter
from typing import Any


def detections_from_result(result: Any, class_names: dict[int, str]) -> list[dict]:
    """Convert one Ultralytics result to JSON-serializable detection records."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidence = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    return [
        {
            "class_id": int(class_id),
            "class_name": class_names.get(int(class_id), str(int(class_id))),
            "confidence": float(conf),
            "bbox_xyxy": [float(value) for value in bbox],
        }
        for bbox, conf, class_id in zip(xyxy, confidence, class_ids)
    ]


def summarize_frames(frames: list[dict]) -> dict:
    """Summarize observed detections without implying detector accuracy."""
    counts = Counter(
        detection["class_name"]
        for frame in frames
        for detection in frame["detections"]
    )
    detected_frames = sum(bool(frame["detections"]) for frame in frames)
    return {
        "sampled_frames": len(frames),
        "frames_with_any_detection": detected_frames,
        "detection_counts_by_prompt": dict(sorted(counts.items())),
        "note": "Detection frequency is not precision, recall, or verified rim accuracy.",
    }
