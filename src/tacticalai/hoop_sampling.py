"""Select full-rate basket measurements around cached ball observations."""

import json
import math
import time

import cv2

from .rim_probe import summarize_frames


MARGIN_DIAGONAL = 0.05
LEAD_S = 0.30
TAIL_S = 0.20


def refinement_frames(coarse, ball_points, stride, fps, width, height):
    """Use nearby coarse boxes only to schedule inference, never as output measurements."""
    margin = MARGIN_DIAGONAL * math.hypot(width, height)
    lead = math.ceil(LEAD_S * fps)
    tail = math.ceil(TAIL_S * fps)
    selected = set()
    for index, point in enumerate(ball_points):
        if point["state"] != "detected":
            continue
        x, y = point["measurement_xy"]
        anchor = index - index % stride
        boxes = [
            detection["bbox_xyxy"]
            for frame in range(max(0, anchor - stride), anchor + 2 * stride + 1, stride)
            for detection in coarse.get(frame, {}).get("detections", [])
        ]
        # Scan conservatively when the sampled detector supplied no local reference.
        if not boxes or any(
            x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin
            for x1, y1, x2, y2 in boxes
        ):
            selected.update(range(max(0, index - lead), min(len(ball_points), index + tail + 1)))
    return selected


def scan_boxes(source, output, info, stride, predict, ball_points=None):
    """Scan the original timeline and journal measured frames for restart."""
    fps, count = info["fps"], info["frames"]
    journal = output / "measured-checkpoint.jsonl"
    records = {}
    with journal.open("a+b") as stream:
        stream.seek(0)
        while line := stream.readline():
            if not line.endswith(b"\n"):
                stream.seek(-len(line), 1)
                stream.truncate()
                break
            row = json.loads(line)
            index = row["frame_idx"]
            if index in records or not 0 <= index < count or row["timestamp_s"] != index / fps:
                raise ValueError("Invalid hoop checkpoint frame")
            records[index] = row

        def measure(indices, phase):
            pending = indices.difference(records)
            print(f"{phase}: {len(pending)}/{len(indices)} frames to measure", flush=True)
            if not pending:
                return
            capture = cv2.VideoCapture(str(source))
            try:
                last = max(pending)
                for index in range(last + 1):
                    if index not in pending:
                        if not capture.grab():
                            raise ValueError(f"Could not advance to source frame {index}")
                        continue
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Could not decode source frame {index}")
                    started = time.perf_counter()
                    detections = [detection for detection, _ in predict(frame)]
                    row = dict(frame_idx=index, timestamp_s=index / fps, detections=detections,
                               inference_s=time.perf_counter() - started)
                    stream.write((json.dumps(row, allow_nan=False) + "\n").encode("utf-8"))
                    stream.flush()
                    records[index] = row
                    if len(records) % 25 == 0 or index == last:
                        print(f"{phase}: {len(records)} total measured (source frame {index})", flush=True)
            finally:
                capture.release()

        coarse = set(range(0, count, stride))
        measure(coarse, "Coarse scan")
        if ball_points is not None:
            windows = refinement_frames({i: records[i] for i in coarse}, ball_points,
                                        stride, fps, info["width"], info["height"])
            measure(windows - coarse, "Shot refinement")

    ordered = [records[index] for index in sorted(records)]
    with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return dict(summarize_frames(ordered), video=info, coarse_frames=len(coarse),
                refinement_frames=len(records) - len(coarse), unmeasured_frames=count - len(records),
                inference_s=sum(row["inference_s"] for row in ordered))
