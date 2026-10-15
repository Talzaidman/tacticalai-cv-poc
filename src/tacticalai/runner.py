"""Pinned Ultralytics tracking API; exports raw detections before association."""
import importlib.metadata
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .export import csv_writer, records, sha256, write_json, write_rows


def run(video, output, config, tracker_name, max_frames=None):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.158":
        raise RuntimeError("Install requirements.txt: tracker/export API is pinned to 8.4.158")
    if config["device"] != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select a Colab GPU or explicitly use --device cpu")
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                config=config, tracker=tracker_name, video=str(video), video_sha256=sha256(video),
                python=sys.version, platform=platform.platform(),
                versions={p: importlib.metadata.version(p) for p in
                          ("ultralytics", "torch", "torchvision", "opencv-python", "numpy", "lap", "PyYAML")},
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                torch_cuda_build=torch.version.cuda,
                source_sha256={str(p.name): sha256(p) for p in Path(__file__).parent.glob("*.py")},
                accuracy_metrics=None, timestamp_policy="zero-based frame_idx / source FPS; CFR input")
    write_json(output / "metadata.json", meta)
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    cap, writer = None, None
    started = time.perf_counter()
    try:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        model = YOLO(config["model"])
        weights = Path(model.ckpt_path)
        meta["model_sha256"] = sha256(weights)
        meta["model_path"] = str(weights.resolve())
        tracker_path = None
        if tracker_name != "none":
            source = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{tracker_name}.yaml"
            tracker_config = yaml.safe_load(source.read_text(encoding="utf-8"))
            tracker_config.update(with_reid=config.get("with_reid", False),
                                  gmc_method=config["gmc_method"])
            if "track_buffer" in config:
                tracker_config["track_buffer"] = config["track_buffer"]
            if tracker_config["with_reid"]:
                tracker_config["model"] = config.get("reid_model", "yolo26n-cls.pt")
            meta["tracker_config"] = tracker_config
            tracker_path = output / "tracker.yaml"
            tracker_path.write_text(yaml.safe_dump(tracker_config), encoding="utf-8")
        else:
            meta["tracker_config"] = None

        # Registered BEFORE model.track installs association callbacks. Copy here:
        # Ultralytics subsequently replaces/prunes boxes with confirmed tracks.
        raw = []
        def capture(predictor):
            raw[:] = [predictor.results[0].boxes.cpu().numpy()]
        model.add_callback("on_predict_postprocess_end", capture)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError("Invalid video dimensions or FPS")
        meta["source"] = dict(width=width, height=height, fps=fps, declared_frames=declared,
                              duration_s=declared/fps)
        writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("MP4 writer unavailable (mp4v codec)")
        count, model_seconds, export_seconds = 0, 0.0, 0.0
        ids, detection_count, track_count = set(), 0, 0
        loop_start = time.perf_counter()
        with (output / "frames.jsonl").open("w", encoding="utf-8") as frames, \
             (output / "detections.csv").open("w", newline="", encoding="utf-8") as det_file, \
             (output / "tracks.csv").open("w", newline="", encoding="utf-8") as track_file:
            import json
            det_csv, track_csv = csv_writer(det_file), csv_writer(track_file)
            while max_frames is None or count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                kwargs = {k: config[k] for k in ("conf", "iou", "imgsz", "max_det", "device", "half")}
                kwargs.update(classes=[0], verbose=False, save=False, batch=1, augment=False)
                if config["device"] != "cpu":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                if tracker_path:
                    result = model.track(frame, persist=True, tracker=str(tracker_path.resolve()), **kwargs)[0]
                else:
                    result = model.predict(frame, **kwargs)[0]
                if count == 0 and tracker_path:
                    tracker = model.predictor.trackers[0]
                    meta["effective_track_buffer_frames"] = getattr(
                        tracker, "max_frames_lost", getattr(tracker, "max_time_lost", None))
                    meta["effective_tracker_config"] = vars(tracker.args).copy()
                    encoder = getattr(tracker, "encoder", None)
                    meta["reid_encoder_type"] = type(encoder).__name__ if encoder is not None else None
                    reid_weights = getattr(getattr(encoder, "model", None), "ckpt_path", None)
                    if reid_weights:
                        meta["reid_model_path"] = str(Path(reid_weights).resolve())
                        meta["reid_model_sha256"] = sha256(Path(reid_weights))
                    # torch.device is not JSON serializable; preserve its textual value.
                    if "device" in meta["effective_tracker_config"]:
                        meta["effective_tracker_config"]["device"] = str(meta["effective_tracker_config"]["device"])
                if config["device"] != "cpu":
                    torch.cuda.synchronize()
                model_seconds += time.perf_counter() - t
                t = time.perf_counter()
                detections = records(raw[0], count, fps)
                tracks = records(result.boxes.cpu().numpy(), count, fps) if result.boxes.is_track else []
                frames.write(json.dumps(dict(frame_idx=count, timestamp_s=count/fps,
                                             detections=detections, tracks=tracks)) + "\n")
                write_rows(det_csv, detections)
                write_rows(track_csv, tracks)
                ids.update(r["track_id"] for r in tracks)
                detection_count += len(detections)
                track_count += len(tracks)
                writer.write(result.plot())
                export_seconds += time.perf_counter() - t
                count += 1
        loop_seconds = time.perf_counter() - loop_start
        if count == 0 or (max_frames is None and declared > 0 and count != declared):
            raise RuntimeError(f"Decoded {count} frames, expected {declared}; output is incomplete")
        writer.release()
        writer = None
        check = cv2.VideoCapture(str(output / "annotated.mp4"))
        decoded = 0
        while check.read()[0]:
            decoded += 1
        check.release()
        if decoded != count:
            raise RuntimeError(f"Output MP4 has {decoded} frames; expected {count}")
        meta.update(status="complete", frames_processed=count, processed_duration_s=count/fps,
                    truncated=max_frames is not None and count < declared,
                    detection_rows=detection_count, track_rows=track_count, unique_track_ids=len(ids),
                    detections_sha256=sha256(output / "detections.csv"),
                    timing=dict(setup_and_processing_s=time.perf_counter()-started,
                                frame_loop_s=loop_seconds, model_calls_including_warmup_s=model_seconds,
                                export_and_overlay_s=export_seconds, pipeline_fps=count/loop_seconds,
                                model_call_fps=count/model_seconds),
                    timing_notes="Model calls include preprocessing, association and first-call warmup. "
                                 "Loop includes decode, export and encoding; excludes setup and final video verification. "
                                 "Unique IDs are diagnostics, not identity accuracy.")
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        write_json(output / "metadata.json", meta)
    print(f"Saved {output}: {count} frames, {meta['timing']['pipeline_fps']:.2f} pipeline FPS")
