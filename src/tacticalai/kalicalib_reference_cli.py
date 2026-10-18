"""Compare published KaliCalib processing/checkpoints on fixed review frames."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import urllib.request

import cv2
import numpy as np

from .auto_calibration_cli import KALI_REVISION, manual_agreement
from .calibration import Calibration, transform
from .calibration_cli import read_frame, video_info, polylines, label
from .court import floor_lines
from .export import sha256, write_json
from .kalicalib_reference import ReferenceDetector, extract_centroids, estimate

SAMPLES = dict(attack_clip=[0, 25, 50, 75, 100, 125, 149],
               attack_clip2=[0, 25, 50, 75, 100, 125, 150, 174])


def overlay(frame, court, matrix, title, subtitle):
    image = cv2.resize(frame, (1280, 720))
    if matrix is not None:
        scale = np.diag([1280/frame.shape[1], 720/frame.shape[0], 1.])
        polylines(image, floor_lines(court), scale @ np.linalg.inv(matrix), (0, 230, 255))
    label(image, title, 24)
    label(image, subtitle, 47)
    if matrix is None:
        label(image, "NO SUPPORTED CALIBRATION", 76, (0, 150, 255))
    return image


def write_review(output, manifest):
    template = Path(__file__).with_name("kalicalib_reference_review.html").read_text(encoding="utf8")
    payload = json.dumps(manifest, allow_nan=False).replace("<", "\\u003c")
    (output/"review.html").write_text(template.replace("__REPORT_JSON__", payload), encoding="utf8")
    write_json(output/"review_metadata.json", dict(template_sha256=sha256(Path(__file__).with_name("kalicalib_reference_review.html")),
               generator_sha256=sha256(Path(__file__)), page_sha256=sha256(output/"review.html")))


def summarize(manifest):
    summary = {}
    for clip in SAMPLES:
        summary[clip] = {}
        for model in ("test", "challenge"):
            rows = [r for r in manifest["results"] if r["clip"] == clip and r["checkpoint"] == model]
            stages = {}
            for stage in ("raw", "refined"):
                checks = [r[stage+"_manual_agreement"]["mean_error_px"] for r in rows if r.get(stage+"_manual_agreement")]
                stages[stage] = dict(samples=len(rows), available=sum(r[stage+"_image_to_court_h"] is not None for r in rows),
                                     checked_frames=len(checks), mean_checkpoint_discrepancy_native_px=float(np.mean(checks)) if checks else None,
                                     median_checkpoint_discrepancy_native_px=float(np.median(checks)) if checks else None)
            summary[clip][model] = stages
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models/kalicalib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/homography-kalicalib-reference-v1"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/homography-auto-v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = args.output/"run_source"
    snapshots.mkdir()
    for name in ("kalicalib_reference.py", "kalicalib_reference_cli.py", "kalicalib_reference_review.html", "auto_calibration.py", "auto_calibration_cli.py", "calibration.py", "court.py"):
        shutil.copyfile(Path(__file__).with_name(name), snapshots/name)
    assets = {}
    for name, remote in (("model_test.pth", "models/model_test.pth"), ("model_challenge.pth", "models/model_challenge.pth"),
                         ("model_resnet.py", "kalicalib/model_resnet.py"), ("LICENSE.txt", "LICENSE.txt")):
        url = f"https://raw.githubusercontent.com/CEA-LIST/KaliCalib/{KALI_REVISION}/{remote}"
        destination = args.models/name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, destination)
        assets[name] = dict(url=url, sha256=sha256(destination))
    manifest = dict(status="running", frames=SAMPLES, results=[], assets=assets,
                    source_sha256={p.name: sha256(p) for p in snapshots.iterdir()},
                    versions={n: importlib.metadata.version(n) for n in ("torch", "opencv-python", "numpy")},
                    device="cpu", threads=4, input_size=[960, 540], manual_used_for_inference=False,
                    extraction="centroids of all positive winning pixels; no confidence/area threshold",
                    ransac=dict(threshold_working_px=35, max_iterations=2000, seed=0),
                    refinement="OpenCV pinhole, aspect ratio fixed, zero tangential/fixed radial distortion; upstream 100..1800cm floor-span check",
                    adaptations=["No fixed mean-camera fallback; failure is null", "Refine/check in 960x540 then scale matrix to source size"],
                    validation="Draft manual clicks with best axis reflection per checkpoint, evaluation only; not ground truth", accuracy_metrics=None)
    write_json(args.output/"experiment.json", manifest)
    references, previous = {}, {}
    for clip, indices in SAMPLES.items():
        folder = args.output/clip
        folder.mkdir()
        manual_path = Path("configs/calibration")/(clip+(".sequence.manual.json" if clip == "attack_clip" else ".manual.json"))
        doc = json.loads(manual_path.read_text())
        video = Path("data/raw")/(clip+".mp4")
        if video_info(video) != doc["video"]:
            raise ValueError("Reference/video provenance mismatch")
        references[clip] = Calibration(doc)
        shutil.copyfile(manual_path, folder/"manual_reference.json")
        previous_path = args.previous/clip/"detections.jsonl"
        previous[clip] = {r["frame_idx"]: r for r in map(json.loads, previous_path.read_text().splitlines())}
        old_meta = json.loads((previous_path.parent/"metadata.json").read_text())
        if old_meta["video"] != doc["video"]:
            raise ValueError("Previous experiment/video mismatch")
        manifest.setdefault("inputs", {})[clip] = dict(video=doc["video"], manual_sha256=sha256(manual_path),
                                                        previous_sha256=sha256(previous_path))
        for index in indices:
            frame = read_frame(video, index)
            matrix, quality = references[clip].at(index)
            image = overlay(frame, doc["court"], matrix, f"MANUAL REFERENCE | frame {index}", quality["method"]+" | draft, not ground truth")
            key = next((k for k in references[clip].keys if k["frame_idx"] == index), None)
            if key:
                for anchor in key["anchors"].values():
                    p = tuple(np.rint(np.array(anchor["image_xy"])*[1280/frame.shape[1], 720/frame.shape[0]]).astype(int))
                    cv2.circle(image, p, 6, (255, 220, 0), 2)
            cv2.imwrite(str(folder/f"manual-{index:04d}.jpg"), image)
            old = previous[clip][index]
            matrix = np.array(old["automatic_h"]) if old["automatic_h"] is not None else None
            image = overlay(frame, doc["court"], matrix, f"PREVIOUS CUSTOM BASELINE | frame {index}", old["automatic_quality"]["method"])
            cv2.imwrite(str(folder/f"previous-{index:04d}.jpg"), image)
    started = time.perf_counter()
    for checkpoint in ("model_test.pth", "model_challenge.pth"):
        model_name = checkpoint.removeprefix("model_").removesuffix(".pth")
        detector = ReferenceDetector(args.models, checkpoint)
        for clip, indices in SAMPLES.items():
            reference = references[clip]
            for index in indices:
                frame = read_frame(Path("data/raw")/(clip+".mp4"), index)
                tick = time.perf_counter()
                heatmaps = detector.heatmaps(frame)
                inference_s = time.perf_counter()-tick
                points = extract_centroids(heatmaps)
                tick = time.perf_counter()
                result = estimate(points, (frame.shape[1], frame.shape[0]))
                geometry_s = time.perf_counter()-tick
                result.update(clip=clip, frame_idx=index, checkpoint=model_name, points=points,
                              runtime_s=dict(inference=inference_s, geometry=geometry_s))
                key = next((k for k in reference.keys if k["frame_idx"] == index), None)
                for stage, field in (("raw", "raw_image_to_court_h"), ("refined", "refined_image_to_court_h")):
                    matrix = np.array(result[field]) if result[field] is not None else None
                    result[stage+"_manual_agreement"] = manual_agreement(matrix, key, reference.court)
                    image = overlay(frame, reference.court, matrix, f"{model_name.upper()} CHECKPOINT | {stage.upper()} | frame {index}",
                                    f"{result['inliers']}/{len(points)} inliers | unverified candidate" if matrix is not None else str(result.get("camera", {}).get("reason", "no fit")))
                    cv2.imwrite(str(args.output/clip/f"{model_name}-{stage}-{index:04d}.jpg"), image)
                np.savez_compressed(args.output/clip/f"{model_name}-heatmaps-{index:04d}.npz", heatmaps=heatmaps)
                manifest["results"].append(result)
                write_json(args.output/"experiment.json", manifest)
                print(f"{model_name} {clip} frame {index}: {result['inliers']}/{len(points)} inliers; camera={result['refined_image_to_court_h'] is not None}", flush=True)
        del detector
    manifest.update(status="complete", elapsed_s=time.perf_counter()-started)
    manifest["summary"] = summarize(manifest)
    write_json(args.output/"experiment.json", manifest)
    write_review(args.output, manifest)
    print(f"Browser comparison: {args.output/'review.html'}", flush=True)


if __name__ == "__main__":
    main()
