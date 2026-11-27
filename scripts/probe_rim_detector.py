"""Run a reproducible sampled YOLO-World rim/hoop/backboard probe on a video."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import time

import cv2
from ultralytics import YOLOWorld

from tacticalai.export import sha256, write_json
from tacticalai.rim_probe import detections_from_result, summarize_frames


DEFAULT_PROMPTS = ("basketball rim", "basketball hoop", "basketball backboard")
COLORS = ((0, 140, 255), (0, 220, 0), (255, 170, 0))


def video_info(capture: cv2.VideoCapture) -> dict:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0 or frames <= 0:
        raise ValueError("Invalid or unreadable video metadata")
    return {"fps": fps, "width": width, "height": height, "frames": frames}


def annotate(image, record: dict, prompts: tuple[str, ...]):
    canvas = image.copy()
    for detection in record["detections"]:
        x1, y1, x2, y2 = (round(value) for value in detection["bbox_xyxy"])
        class_id = detection["class_id"]
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
    caption = (f"source frame {record['frame_idx']} | t={record['timestamp_s']:.2f}s | "
               f"prompts: {', '.join(prompts)}")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def package_versions() -> dict:
    versions = {}
    for package in ("ultralytics", "clip", "torch", "torchvision", "opencv-python", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if not 0 < args.conf <= 1:
        raise ValueError("--conf must be in (0, 1]")

    args.output.mkdir(parents=True)
    capture = cv2.VideoCapture(str(args.source))
    info = video_info(capture)
    prompts = tuple(args.prompts)
    configuration = {
        "model": args.model,
        "prompts": prompts,
        "sample_stride": args.sample_stride,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": args.device,
    }
    experiment = {
        "status": "running",
        "scope": "sampled open-vocabulary detector probe; not validated rim tracking",
        "automatic_only": True,
        "manual_labels_used": False,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "video": info,
        "configuration": configuration,
        "packages": package_versions(),
        "python": platform.python_version(),
    }
    write_json(args.output / "experiment.json", experiment)

    started = time.perf_counter()
    writer = None
    records = []
    try:
        model = YOLOWorld(args.model)
        model.set_classes(list(prompts))
        checkpoint = Path(args.model)
        if checkpoint.is_file():
            experiment["model_checkpoint"] = str(checkpoint.resolve())
            experiment["model_sha256"] = sha256(checkpoint)
            write_json(args.output / "experiment.json", experiment)
        class_names = {int(key): value for key, value in model.names.items()}
        sampled_fps = max(1.0, info["fps"] / args.sample_stride)
        writer = cv2.VideoWriter(
            str(args.output / "annotated_samples.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            sampled_fps,
            (info["width"], info["height"]),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create annotated MP4")

        frame_idx = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame_idx % args.sample_stride == 0:
                result = model.predict(
                    image, imgsz=args.imgsz, conf=args.conf, device=args.device,
                    verbose=False,
                )[0]
                record = {
                    "frame_idx": frame_idx,
                    "timestamp_s": frame_idx / info["fps"],
                    "detections": detections_from_result(result, class_names),
                }
                records.append(record)
                writer.write(annotate(image, record, prompts))
                print(f"frame {frame_idx}/{info['frames'] - 1}: "
                      f"{len(record['detections'])} detections", flush=True)
            frame_idx += 1
    except BaseException as error:
        experiment.update(status="failed", error=repr(error), elapsed_s=time.perf_counter()-started)
        write_json(args.output / "experiment.json", experiment)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    detections_path = args.output / "detections.jsonl"
    detections_path.write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = summarize_frames(records)
    write_json(args.output / "summary.json", summary)
    experiment.update(
        status="complete",
        elapsed_s=time.perf_counter() - started,
        summary=summary,
        artifacts_sha256={
            name: sha256(args.output / name)
            for name in ("annotated_samples.mp4", "detections.jsonl", "summary.json")
        },
    )
    write_json(args.output / "experiment.json", experiment)


if __name__ == "__main__":
    main()
