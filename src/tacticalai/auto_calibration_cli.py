"""Reproducible CPU benchmark and local browser review of automatic calibration.

Run prepare once to download pinned, licensed model assets, then run benchmark.
Source footage is never uploaded. Existing tracking inference is reused.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

import cv2
import numpy as np

from .auto_calibration import KaliDetector, FloorTracker, fit_candidate
from .calibration import Calibration, project_track, transform
from .calibration_cli import video_info, polylines, label
from .court import floor_lines, landmarks
from .export import sha256, write_json
from .minimap import compose, apply_roles

KALI_REVISION = "3b3f52b59e8e38e155ba7e9c681281d44748c173"
YOLO_REVISION = "6cb899251439982067a81baa225b33f45f335981"


def prepare(folder):
    assets = {
        "kalicalib/model_resnet.py": f"https://raw.githubusercontent.com/CEA-LIST/KaliCalib/{KALI_REVISION}/kalicalib/model_resnet.py",
        "kalicalib/model_test.pth": f"https://raw.githubusercontent.com/CEA-LIST/KaliCalib/{KALI_REVISION}/models/model_test.pth",
        "kalicalib/model_challenge.pth": f"https://raw.githubusercontent.com/CEA-LIST/KaliCalib/{KALI_REVISION}/models/model_challenge.pth",
        "kalicalib/LICENSE.txt": f"https://raw.githubusercontent.com/CEA-LIST/KaliCalib/{KALI_REVISION}/LICENSE.txt",
        "court-keypoints/best.pt": f"https://huggingface.co/koppolusameer/yolo11n-basketball-court-keypoints/resolve/{YOLO_REVISION}/best.pt",
    }
    provenance = {}
    for name, url in assets.items():
        destination = folder/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix+".download")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(destination)
        provenance[name] = dict(url=url, sha256=sha256(destination))
    write_json(folder/"court_models.json", dict(assets=provenance, licenses=dict(
        kalicalib="CeCILL-2.1; see kalicalib/LICENSE.txt", yolo_court="AGPL-3.0 (model card)",
        yolo_training_dataset="CC BY 4.0 (dataset export)")))
    print("Prepared pinned court model assets", flush=True)


def manual_agreement(matrix, key, court):
    """Evaluation only, never feeds detection/gating/flow. Court ends ambiguous."""
    if matrix is None or key is None:
        return None
    names = list(key["anchors"])
    world = np.array([landmarks(court)[n] for n in names])
    clicked = np.array([key["anchors"][n]["image_xy"] for n in names])
    choices = []
    for flip_x in (False, True):
        for flip_y in (False, True):
            oriented = world.copy()
            if flip_x:
                oriented[:, 0] = 28-oriented[:, 0]
            if flip_y:
                oriented[:, 1] = 15-oriented[:, 1]
            residual = np.linalg.norm(transform(oriented, np.linalg.inv(matrix))-clicked, axis=1)
            if np.isfinite(residual).all():
                choices.append(dict(flip_x=flip_x, flip_y=flip_y, mean_error_px=float(residual.mean()),
                                    max_error_px=float(residual.max()), points=len(names)))
    return min(choices, key=lambda c: c["mean_error_px"]) if choices else None


def run_clip(args, clip, kali, yolo):
    directory = args.output/clip
    directory.mkdir(exist_ok=False)
    video = Path("data/raw")/(clip+".mp4")
    info = video_info(video)
    tracks_path = args.tracking_root/clip/"tracktrack/frames.jsonl"
    tracks_meta = json.loads(tracks_path.with_name("metadata.json").read_text())
    if tracks_meta["status"] != "complete" or tracks_meta["video_sha256"] != info["sha256"]:
        raise ValueError("Tracking/video provenance mismatch")
    tracks = [json.loads(line) for line in tracks_path.read_text().splitlines()]
    if len(tracks) != info["frames"] or any(r["frame_idx"] != i or abs(r["timestamp_s"]-i/info["fps"]) > 1e-6 for i, r in enumerate(tracks)):
        raise ValueError("Tracking length/timestamps mismatch")
    manual_path = Path("configs/calibration")/(clip+(".sequence.manual.json" if clip == "attack_clip" else ".manual.json"))
    manual_doc = json.loads(manual_path.read_text())
    if manual_doc["video"] != info:
        raise ValueError("Manual reference/video provenance mismatch")
    manual = Calibration(manual_doc)
    roles_path = Path("configs/calibration")/(clip+".roles.manual.json")
    roles = json.loads(roles_path.read_text()) if roles_path.exists() else {}
    if roles and roles["tracking_frames_sha256"] != sha256(tracks_path):
        raise ValueError("Role provenance mismatch")
    shutil.copyfile(manual_path, directory/"manual_reference.json")
    shutil.copyfile(tracks_path.with_name("metadata.json"), directory/"tracking_metadata.json")
    if roles:
        shutil.copyfile(roles_path, directory/"roles.json")
    cap = cv2.VideoCapture(str(video))
    tracker = FloorTracker(args.interval-1)
    report = dict(clip=clip, video=info, frames=[], elapsed_s=None, status="running",
                  manual_checkpoints=[k["frame_idx"] for k in manual.keys],
                  tracking_frames_sha256=sha256(tracks_path), manual_sha256=sha256(manual_path),
                  roles_sha256=sha256(roles_path) if roles else None, accuracy_metrics=None,
                  notes="Manual clicks are draft references, not ground truth. Symmetry alignment is evaluation only. Geometric pass is not verified accuracy. No manual initialization or manual fallback.")
    write_json(directory/"metadata.json", report)
    started = time.perf_counter()
    with (directory/"detections.jsonl").open("x", encoding="utf8") as output:
        for index in range(info["frames"]):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Source video ended early")
            boxes = [r["bbox_xyxy"] for r in tracks[index]["tracks"]]
            tick = time.perf_counter()
            points = kali.detect(frame, args.probability)
            kali_time = time.perf_counter()-tick
            matrix, quality = fit_candidate(points, (info["width"], info["height"]))
            tick = time.perf_counter()
            result = yolo.predict(frame, device="cpu", imgsz=640, conf=.25, verbose=False)[0]
            yolo_time = time.perf_counter()-tick
            raw_yolo = []
            if len(result.boxes):
                best = int(result.boxes.conf.argmax())
                raw_yolo = result.keypoints.data[best].cpu().tolist()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tick = time.perf_counter()
            if index % args.interval == 0 or tracker.matrix is None:
                hybrid, hybrid_quality = matrix, dict(quality)
                tracker.reset(gray, hybrid, boxes, manual.court)
                if hybrid is not None:
                    hybrid_quality["method"] = "automatic_detector_reset_unverified"
            else:
                hybrid, hybrid_quality = tracker.advance(gray, boxes)
                if hybrid is None:
                    # The per-frame detector is cached as a diagnostic baseline;
                    # use it for immediate recovery when floor flow is lost.
                    hybrid, hybrid_quality = matrix, dict(quality)
                    tracker.reset(gray, hybrid, boxes, manual.court)
            temporal_time = time.perf_counter()-tick
            key = next((k for k in manual.keys if k["frame_idx"] == index), None)
            data = dict(frame_idx=index, timestamp_s=index/info["fps"], kali_points=points,
                        yolo_keypoints=raw_yolo, yolo_mapping_status="unverified_not_used_for_homography",
                        automatic_h=matrix.tolist() if matrix is not None else None, automatic_quality=quality,
                        hybrid_h=hybrid.tolist() if hybrid is not None else None, hybrid_quality=hybrid_quality,
                        manual_agreement=manual_agreement(matrix, key, manual.court),
                        runtime_s=dict(kalicalib=kali_time, yolo=yolo_time, temporal_update=temporal_time))
            output.write(json.dumps(data, allow_nan=False)+"\n")
            report["frames"].append({k: v for k, v in data.items() if k not in ("kali_points", "yolo_keypoints", "automatic_h", "hybrid_h")})
            if index % 25 == 0:
                print(f"{clip}: {index+1}/{info['frames']} | {quality['inliers']} inliers | {quality['method']}", flush=True)
    cap.release()
    report.update(status="complete", elapsed_s=time.perf_counter()-started)
    report["counts"] = dict(frames=info["frames"],
        automatic_geometric_pass=sum(f["automatic_quality"]["method"] != "unavailable" for f in report["frames"]),
        hybrid_available=sum(f["hybrid_quality"]["method"] != "unavailable" for f in report["frames"]),
        flow_frames=sum(f["hybrid_quality"]["method"] == "automatic_floor_flow_unverified" for f in report["frames"]))
    report["timing"] = {name: float(np.mean([f["runtime_s"][name] for f in report["frames"]])) for name in ("kalicalib", "yolo", "temporal_update")}
    write_json(directory/"metadata.json", report)
    return report


def encode(path, fps):
    import imageio_ffmpeg
    temporary = path.with_suffix(".h264.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-n", "-loglevel", "error", "-i", str(path),
                    "-an", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)], check=True)
    temporary.replace(path)


def render_clip(args, clip):
    directory = args.output/clip
    report = json.loads((directory/"metadata.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Incomplete detection run")
    if any((directory/(mode+".mp4")).exists() for mode in ("detector", "hybrid", "minimap", "yolo", "manual")) or (directory/"positions.jsonl").exists():
        raise FileExistsError("Render products already exist; preserve this run and use a new output directory")
    if video_info(Path("data/raw")/(clip+".mp4")) != report["video"]:
        raise ValueError("Source video changed since inference")
    data = [json.loads(line) for line in (directory/"detections.jsonl").read_text().splitlines()]
    reference = Calibration(json.loads((directory/"manual_reference.json").read_text()))
    rows_path = args.tracking_root/clip/"tracktrack/frames.jsonl"
    if sha256(rows_path) != report["tracking_frames_sha256"]:
        raise ValueError("Tracking export changed")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    roles = json.loads((directory/"roles.json").read_text()) if (directory/"roles.json").exists() else {}
    cap = cv2.VideoCapture(str(Path("data/raw")/(clip+".mp4")))
    modes = ["detector", "hybrid", "minimap", "yolo", "manual"]
    writers = {mode: cv2.VideoWriter(str(directory/(mode+".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), report["video"]["fps"], (1280, 720)) for mode in modes}
    if not all(w.isOpened() for w in writers.values()):
        raise RuntimeError("Cannot open output encoder")
    positions = (directory/"positions.jsonl").open("x", encoding="utf8")
    try:
        for index, item in enumerate(data):
            ok, original = cap.read()
            if not ok:
                raise ValueError("Video decode failed during rendering")
            sx = 1280/original.shape[1]
            scale = np.diag([sx, 720/original.shape[0], 1.])
            automatic = np.array(item["automatic_h"]) if item["automatic_h"] is not None else None
            hybrid = np.array(item["hybrid_h"]) if item["hybrid_h"] is not None else None
            tracks = [project_track(r, hybrid, item["hybrid_quality"], reference.court,
                                    (original.shape[1], original.shape[0])) for r in rows[index]["tracks"]]
            tracks = apply_roles(tracks, roles, index)
            positions.write(json.dumps(dict(frame_idx=index, timestamp_s=item["timestamp_s"],
                                           image_to_court_h=item["hybrid_h"], calibration=item["hybrid_quality"], tracks=tracks), allow_nan=False)+"\n")
            manual_matrix, manual_quality = reference.at(index)
            for mode, writer in writers.items():
                frame = cv2.resize(original, (1280, 720))
                matrix = automatic if mode == "detector" else hybrid
                quality = item["automatic_quality"] if mode == "detector" else item["hybrid_quality"]
                if mode == "manual":
                    matrix, quality = manual_matrix, manual_quality
                if mode in ("detector", "hybrid", "manual"):
                    if matrix is not None:
                        polylines(frame, floor_lines(reference.court), scale @ np.linalg.inv(matrix), (0, 230, 255))
                    if mode == "detector":
                        for point in item["kali_points"]:
                            p = tuple(np.rint(transform([point["image_xy"]], scale)[0]).astype(int))
                            good = point["index"] in quality.get("inlier_indices", [])
                            color = (30, 230, 80) if good else (90, 90, 240)
                            cv2.circle(frame, p, 3, color, -1)
                            cv2.putText(frame, str(point["index"]), (p[0]+3, p[1]-3), 0, .4, color, 1, cv2.LINE_AA)
                    if mode == "manual":
                        key = next((k for k in reference.keys if k["frame_idx"] == index), None)
                        if key:
                            for name, anchor in key["anchors"].items():
                                p = tuple(np.rint(transform([anchor["image_xy"]], scale)[0]).astype(int))
                                cv2.circle(frame, p, 6, (255, 220, 0), 2)
                                cv2.putText(frame, name.replace("left_", ""), p, 0, .45, (255, 220, 0), 1, cv2.LINE_AA)
                    label(frame, f"{mode.upper()} | {quality['method']} | frame {index}", 25)
                    label(frame, "UNVERIFIED court fit | manual clicks are draft references", 48)
                    if matrix is None:
                        label(frame, "NO SUPPORTED HOMOGRAPHY", 76, (0, 150, 255))
                elif mode == "minimap":
                    frame, shown = compose(original, tracks, reference.court, item["hybrid_quality"],
                                           item["timestamp_s"], players_only=bool(roles))
                    label(frame, "AUTOMATIC CALIBRATION CANDIDATE | orientation unverified", 700)
                else:
                    for i, (x, y, confidence) in enumerate(item["yolo_keypoints"]):
                        if confidence < .3:
                            continue
                        p = (round(x*sx), round(y*720/original.shape[0]))
                        cv2.circle(frame, p, 4, (20, 210, 255), -1)
                        cv2.putText(frame, f"{i}:{confidence:.2f}", p, 0, .4, (20, 210, 255), 1, cv2.LINE_AA)
                    label(frame, f"YOLO11n RAW LANDMARKS | frame {index}", 25)
                    label(frame, "Index-to-court mapping unverified; no homography or player projection", 48)
                writer.write(frame)
                if index in {0, len(data)//2, len(data)-1}:
                    cv2.imwrite(str(directory/f"{mode}-{index:04d}.jpg"), frame)
    finally:
        cap.release()
        positions.close()
        for writer in writers.values():
            writer.release()
    for mode in modes:
        path = directory/(mode+".mp4")
        encode(path, report["video"]["fps"])
        check = cv2.VideoCapture(str(path))
        decoded = 0
        while check.read()[0]:
            decoded += 1
        check.release()
        if decoded != len(data):
            raise ValueError("Rendered video frame count mismatch")
    report["rendering"] = dict(status="complete", frames=len(data), width=1280, height=720,
                                fps=report["video"]["fps"], codec="H264", audio=False,
                                modes=modes, verified_full_decode=True,
                                source_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                                               ("auto_calibration_cli.py", "auto_review.html", "calibration.py", "court.py", "minimap.py")})
    write_json(directory/"metadata.json", report)
    print(f"Rendered and decoded {clip}: {len(data)} frames in all {len(modes)} views", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "benchmark", "render"])
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("outputs/homography-auto-v1"))
    parser.add_argument("--tracking-root", type=Path, default=Path("outputs/occlusion-20261015T204150Z-029a7a35/reid"))
    parser.add_argument("--probability", type=float, default=.3)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.probability < 1 or args.interval < 2:
        parser.error("Probability must be between zero and one; interval >=2")
    if args.command == "prepare":
        prepare(args.models)
        return
    clips = ["attack_clip", "attack_clip2"]
    if args.command == "benchmark":
        from ultralytics import YOLO
        args.output.mkdir(parents=True, exist_ok=True)
        manifest = dict(status="running", parameters=vars(args).copy(),
            versions={name: importlib.metadata.version(name) for name in ["torch", "ultralytics", "opencv-python", "numpy"]},
            model_assets=json.loads((args.models/"court_models.json").read_text()),
            source_sha256={name: sha256(Path(__file__).with_name(name)) for name in ["auto_calibration.py", "auto_calibration_cli.py", "calibration.py", "court.py", "minimap.py"]},
            device="cpu", threads=4, manual_coordinates_used_for_inference=False,
            kalicalib_input=[960, 540], yolo_imgsz=640,
            fit_gates=dict(ransac_px_at_960=8, min_inliers=8, min_ratio=.4, min_support_fraction=.005),
            temporal=dict(interval=args.interval, max_flow_age=args.interval-1, forward_backward_px=1.5,
                          min_flow_inliers=12, min_flow_ratio=.6, cut_mean_absdiff=45),
            notes="Every-frame detector baseline plus hybrid selects periodic detections/flow. Runtime of the combined experiment is not deployed hybrid throughput. No retraining.")
        manifest["parameters"] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest["parameters"].items()}
        for name, asset in manifest["model_assets"]["assets"].items():
            if sha256(args.models/name) != asset["sha256"]:
                raise ValueError(f"Model asset changed since prepare: {name}")
        write_json(args.output/"experiment.json", manifest)
        kali = KaliDetector(args.models/"kalicalib")
        yolo = YOLO(args.models/"court-keypoints/best.pt")
        for clip in clips:
            run_clip(args, clip, kali, yolo)
        manifest["status"] = "complete"
        write_json(args.output/"experiment.json", manifest)
    else:
        reports = {}
        for clip in clips:
            reports[clip] = render_clip(args, clip)
        template = Path(__file__).with_name("auto_review.html").read_text(encoding="utf8")
        payload = json.dumps(reports, allow_nan=False).replace("<", "\\u003c")
        (args.output/"review.html").write_text(template.replace("__REPORT_JSON__", payload), encoding="utf8")
        print(f"Browser review: {args.output/'review.html'}", flush=True)


if __name__ == "__main__":
    main()
