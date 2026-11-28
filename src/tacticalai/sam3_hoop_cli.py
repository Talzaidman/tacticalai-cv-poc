"""Probe SAM 3 text-prompted hoop, rim and backboard detection on one clip."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np

from .export import sha256, write_json
from .calibration_cli import video_info
from .hoop_sampling import LEAD_S, MARGIN_DIAGONAL, TAIL_S, scan_boxes
from .rim_probe import summarize_frames


MODEL_ID = "facebook/sam3"
MODEL_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
PROMPTS = ("basketball hoop", "basketball backboard")
COLORS = ((255, 210, 40), (0, 140, 255), (110, 255, 105))  # BGR


def box_predictions(output, size, threshold):
    """Match Transformers 5.24.1 instance scores and boxes without mask postprocessing."""
    import torch

    height, width = size
    scores = output.pred_logits.sigmoid()
    if output.presence_logits is not None:
        scores = scores * output.presence_logits.sigmoid()
    keep = scores[0] > threshold
    scale = torch.tensor([width, height, width, height], device=output.pred_boxes.device)
    boxes = output.pred_boxes[0] * scale
    return boxes[keep].float().cpu().tolist(), scores[0][keep].float().cpu().tolist()


def predict_frame(frame, model, processor, text_inputs, threshold, dtype, boxes_only=False):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image, return_tensors="pt")
    sizes = inputs["original_sizes"].tolist()
    pixels = inputs["pixel_values"].to(device=model.device, dtype=dtype)
    vision = model.get_vision_features(pixel_values=pixels)
    for prompt_id, prompt in enumerate(text_inputs):
        output = model(vision_embeds=vision, **text_inputs[prompt])
        if boxes_only:
            boxes, scores = box_predictions(output, sizes[0], threshold)
            for box, score in zip(boxes, scores, strict=True):
                yield dict(class_id=prompt_id, class_name=prompt, confidence=score, bbox_xyxy=box), None
            continue
        result = processor.post_process_instance_segmentation(
            output, threshold=threshold, mask_threshold=0.5, target_sizes=sizes,
        )[0]
        boxes = result["boxes"].float().cpu().tolist()
        scores = result["scores"].float().cpu().tolist()
        masks = result["masks"].bool().cpu().numpy()
        for box, score, mask in zip(boxes, scores, masks, strict=True):
            yield dict(class_id=prompt_id, class_name=prompt, confidence=score,
                       bbox_xyxy=box, mask_area_px=int(mask.sum())), mask


def annotate(frame, candidates, frame_idx, timestamp):
    canvas = frame.copy()
    for detection, mask in candidates:
        color = COLORS[detection["class_id"] % len(COLORS)]
        canvas[mask] = 0.65 * canvas[mask] + 0.35 * np.asarray(color)
    label_positions = []
    for detection, _ in candidates:
        color = COLORS[detection["class_id"] % len(COLORS)]
        x1, y1, x2, y2 = (round(value) for value in detection["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        (width, height), _ = cv2.getTextSize(label, font, 0.62, 1)
        x = max(0, min(x1, canvas.shape[1] - width - 5))
        y = max(60, min(y1 - 6, canvas.shape[0] - 8))
        while any(x < right and x + width > left and abs(y - row) < 23
                  for left, right, row in label_positions) and y + 23 < canvas.shape[0] - 5:
            y += 23
        label_positions.append((x, x + width, y))
        cv2.rectangle(canvas, (x, y - height - 4), (x + width + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x + 2, y), font, 0.62, color, 1, cv2.LINE_AA)
    caption = f"SAM 3 | frame {frame_idx} | {timestamp:.2f}s | {len(candidates)} candidates"
    cv2.rectangle(canvas, (0, 0), (520, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def probe_video(source, output, stride, prompts, predict):
    capture = cv2.VideoCapture(str(source))
    encoder = None
    video_path = output / "annotated_samples.mp4"
    records = []
    inference_s = 0.0
    frame_idx = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or not math.isfinite(fps) or min(fps, width, height, expected) <= 0:
            raise ValueError(f"Unreadable source video: {source}")
        encoder = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps / stride),
            "-i", "pipe:0", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(video_path),
        ], stdin=subprocess.PIPE)
        with encoder.stdin, (output / "detections.jsonl").open("x", encoding="utf-8") as evidence:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    started = time.perf_counter()
                    candidates = list(predict(frame))
                    inference_s += time.perf_counter() - started
                    timestamp = frame_idx / fps
                    record = dict(frame_idx=frame_idx, timestamp_s=timestamp,
                                  detections=[detection for detection, _ in candidates])
                    evidence.write(json.dumps(record, allow_nan=False) + "\n")
                    records.append(record)
                    encoder.stdin.write(annotate(frame, candidates, frame_idx, timestamp).tobytes())
                    print(f"Frame {frame_idx}/{expected - 1}: {len(candidates)} candidates", flush=True)
                frame_idx += 1
        if encoder.wait() != 0:
            raise RuntimeError("Annotated video encoding failed")
        if frame_idx != expected:
            raise ValueError(f"Decoded {frame_idx} of {expected} source frames")
    finally:
        capture.release()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()

    frames_by_prompt = Counter(
        prompt for record in records for prompt in {d["class_name"] for d in record["detections"]}
    )
    summary = summarize_frames(records)
    counts = summary["detection_counts_by_prompt"]
    summary.update(
        video=dict(width=width, height=height, fps=fps, frames=frame_idx, preview_fps=fps / stride),
        detection_counts_by_prompt={prompt: counts.get(prompt, 0) for prompt in prompts},
        frames_with_detection_by_prompt={prompt: frames_by_prompt[prompt] for prompt in prompts},
        inference_s=inference_s,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--boxes-only", action="store_true")
    parser.add_argument("--ball-results", type=Path, help="SAM 3 ball cache for full-rate shot refinement")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Source video does not exist: {args.source}")
    if args.sample_stride < 1 or not 0 < args.conf <= 1:
        parser.error("Sample stride must be positive and confidence must be in (0, 1]")
    if len(set(args.prompts)) != len(args.prompts) or any(not prompt.strip() for prompt in args.prompts):
        parser.error("Prompts must be unique and nonempty")
    if args.ball_results and (not args.boxes_only or args.sample_stride < 2):
        parser.error("Shot refinement requires --boxes-only and a stride greater than one")

    import torch
    from transformers import Sam3Model, Sam3Processor

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before running the SAM 3 probe")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN to a Hugging Face read token with access to facebook/sam3")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    source_hash = sha256(args.source)
    ball_points = None
    mode = "boxes" if args.boxes_only else "preview"
    report = dict(
        status="running", model=MODEL_ID, model_revision=MODEL_REVISION,
        source=args.source.name, source_sha256=source_hash, prompts=list(args.prompts),
        sample_stride=args.sample_stride, confidence_threshold=args.conf,
        mode=mode, mask_threshold=None if args.boxes_only else 0.5,
        device=torch.cuda.get_device_name(0), dtype=str(dtype), automatic_only=True,
        offline_processing=True, manual_labels_used=False, tracking=False,
        packages={name: version(name) for name in ("torch", "transformers", "opencv-python", "imageio-ffmpeg")},
    )
    if args.boxes_only:
        info = video_info(args.source)
        report["implementation_sha256"] = {name: sha256(Path(__file__).with_name(name))
                                           for name in ("sam3_hoop_cli.py", "hoop_sampling.py")}
    if args.ball_results:
        from .ball_fusion_cli import ball_result_paths, load_ball

        clip = args.source.stem
        ball_manifest = json.loads((args.ball_results / "experiment.json").read_text())
        if (ball_manifest["status"] != "complete"
                or ball_manifest["clips"][clip]["source_video_sha256"] != source_hash):
            raise ValueError("Ball cache does not match the source video")
        ball_json, ball_video = ball_result_paths(args.ball_results, "SAM3-Native", clip)
        ball_points = load_ball(ball_json, ball_video, info)["tracked"]
        report.update(mode="adaptive_boxes", ball_sha256=sha256(ball_json),
                      refinement=dict(margin_diagonal=MARGIN_DIAGONAL, lead_s=LEAD_S, tail_s=TAIL_S))
    manifest = args.output / "experiment.json"
    if args.boxes_only and manifest.is_file():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in report.items() if key not in ("status", "device")):
            raise ValueError("Hoop checkpoint configuration differs. Select a new output folder.")
        if previous["status"] == "complete":
            return
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    write_json(manifest, report)
    started = time.perf_counter()
    try:
        model = Sam3Model.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token,
                                          dtype=dtype).to("cuda").eval()
        processor = Sam3Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token)
        text_inputs = {prompt: processor(text=prompt, return_tensors="pt").to(model.device)
                       for prompt in args.prompts}
        with torch.inference_mode():
            def predict(frame):
                return predict_frame(frame, model, processor, text_inputs, args.conf, dtype, args.boxes_only)

            if args.boxes_only:
                summary = scan_boxes(args.source, args.output, info, args.sample_stride, predict, ball_points)
            else:
                summary = probe_video(args.source, args.output, args.sample_stride, args.prompts, predict)
        write_json(args.output / "summary.json", summary)
        artifacts = ["detections.jsonl", "summary.json"]
        if not args.boxes_only:
            artifacts.append("annotated_samples.mp4")
        report.update(status="complete", processing_s=time.perf_counter() - started,
                      artifacts_sha256={name: sha256(args.output / name) for name in artifacts})
    except Exception as error:
        message = str(error).replace(token, "[REDACTED]")
        report.update(status="failed", error=f"{type(error).__name__}: {message}")
        raise RuntimeError(report["error"]) from None
    finally:
        write_json(manifest, report)


if __name__ == "__main__":
    main()
