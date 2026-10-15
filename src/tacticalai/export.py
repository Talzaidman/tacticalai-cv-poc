"""Plain, CPU-compatible output schema helpers."""
import csv
import hashlib
import json


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(boxes, frame_idx, fps):
    """Serialize Ultralytics Boxes without mistaking detections for tracks."""
    if boxes is None:
        return []
    xyxy = boxes.xyxy.tolist()
    scores, classes = boxes.conf.tolist(), boxes.cls.tolist()
    ids = boxes.id.tolist() if boxes.is_track else [None] * len(xyxy)
    return [dict(frame_idx=frame_idx, timestamp_s=frame_idx / fps,
                 bbox_xyxy=[float(v) for v in box], det_confidence=float(score),
                 class_id=int(cls), class_name="person", role="unknown",
                 track_id=int(tid) if tid is not None else None,
                 team_id=None, jersey_number=None, court_xy_m=None,
                 position_source="image_detection", court_projection_quality=None)
            for box, score, cls, tid in zip(xyxy, scores, classes, ids)]


FIELDS = ["frame_idx", "timestamp_s", "x1", "y1", "x2", "y2",
          "det_confidence", "class_id", "class_name", "role", "track_id"]


def csv_writer(stream):
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    return writer


def write_rows(writer, rows):
    for row in rows:
        flat = {key: row[key] for key in FIELDS if key in row}
        flat.update(zip(("x1", "y1", "x2", "y2"), row["bbox_xyxy"]))
        writer.writerow(flat)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
