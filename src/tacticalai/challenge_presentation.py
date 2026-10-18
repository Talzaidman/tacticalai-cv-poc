"""Full-clip challenge-checkpoint calibration and local tracking presentation.

Run infer, then render. Reuses saved TrackTrack + ReID exports; never reruns
player inference or uses manual court coordinates. Roles are explicit manual
run-specific annotations. Failed calibration remains null.
"""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from .auto_calibration_cli import encode, KALI_REVISION
from .calibration import project_track
from .calibration_cli import video_info
from .export import sha256, write_json
from .kalicalib_reference import ReferenceDetector, extract_centroids, estimate
from .minimap import apply_roles, compose, track_color

COURT = dict(length_m=28., width_m=15., key_depth_m=5.8, key_width_m=4.9,
             three_point_corner_offset_m=.9, assumption="Nominal FIBA dimensions, not venue survey")
CLIPS = ("attack_clip", "attack_clip2")


def tracking(args, clip, info):
    path = args.tracking_root/clip/"tracktrack/frames.jsonl"
    metadata = json.loads(path.with_name("metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["video_sha256"] != info["sha256"]:
        raise ValueError("Tracking/video mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != info["frames"] or any(r["frame_idx"] != i or abs(r["timestamp_s"]-i/info["fps"]) > 1e-6 for i, r in enumerate(rows)):
        raise ValueError("Tracking frames/timestamps mismatch")
    return path, metadata, rows


def infer(args):
    args.output.mkdir(parents=True, exist_ok=False)
    detector = ReferenceDetector(args.models, "model_challenge.pth")
    source = args.output/"run_source"
    source.mkdir()
    for name in ("challenge_presentation.py", "challenge_presentation.html", "kalicalib_reference.py", "auto_calibration.py", "calibration.py", "court.py", "minimap.py"):
        shutil.copyfile(Path(__file__).with_name(name), source/name)
    manifest = dict(status="running", checkpoint="model_challenge.pth", upstream_revision=KALI_REVISION,
                    model_sha256=sha256(args.models/"model_challenge.pth"), architecture_sha256=sha256(args.models/"model_resnet.py"),
                    model_license="CeCILL-2.1 (KaliCalib)",
                    source_sha256={p.name: sha256(p) for p in source.iterdir()},
                    versions={n: importlib.metadata.version(n) for n in ("torch", "opencv-python", "numpy")},
                    device="cpu", threads=4, input_size=[960, 540], ransac_threshold_working_px=35,
                    court=COURT, manual_court_anchors_used=False, temporal_propagation=False,
                    line_refinement=False, mean_camera_fallback=False, accuracy_metrics=None,
                    notes="Offline source-synchronous playback. Automatic per-frame refined camera. Missing measurements stay null. No identity repair or off-screen inference.", clips={})
    write_json(args.output/"experiment.json", manifest)
    for clip in CLIPS:
        folder = args.output/clip
        folder.mkdir()
        video = Path("data/raw")/(clip+".mp4")
        info = video_info(video)
        tracks_path, tracks_meta, _ = tracking(args, clip, info)
        shutil.copyfile(tracks_path.with_name("metadata.json"), folder/"tracking_metadata.json")
        clip_report = dict(status="running", video=info, tracking_frames_sha256=sha256(tracks_path),
                           tracking_metadata_sha256=sha256(tracks_path.with_name("metadata.json")),
                           frames=0, calibrated_frames=0, elapsed_s=None)
        write_json(folder/"metadata.json", clip_report)
        cap = cv2.VideoCapture(str(video))
        started = time.perf_counter()
        try:
            with (folder/"calibration.jsonl").open("x", encoding="utf8") as output:
                for index in range(info["frames"]):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError("Video decode failed")
                    tick = time.perf_counter()
                    points = extract_centroids(detector.heatmaps(frame))
                    result = estimate(points, (info["width"], info["height"]))
                    result.update(frame_idx=index, timestamp_s=index/info["fps"], elapsed_s=time.perf_counter()-tick)
                    output.write(json.dumps(result, allow_nan=False)+"\n")
                    clip_report["frames"] += 1
                    clip_report["calibrated_frames"] += result["refined_image_to_court_h"] is not None
                    if index % 25 == 0:
                        print(f"{clip}: {index+1}/{info['frames']} | camera available {clip_report['calibrated_frames']}", flush=True)
        finally:
            cap.release()
        clip_report.update(status="complete", elapsed_s=time.perf_counter()-started,
                           calibration_sha256=sha256(folder/"calibration.jsonl"))
        write_json(folder/"metadata.json", clip_report)
        manifest["clips"][clip] = clip_report
        write_json(args.output/"experiment.json", manifest)
    manifest["status"] = "complete"
    write_json(args.output/"experiment.json", manifest)


def render(args):
    manifest = json.loads((args.output/"experiment.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Calibration run incomplete")
    for clip in CLIPS:
        folder = args.output/clip
        if (folder/"projection.mp4").exists() or (folder/"frames.jsonl").exists():
            raise FileExistsError("Presentation products already exist")
        report = json.loads((folder/"metadata.json").read_text())
        video = Path("data/raw")/(clip+".mp4")
        info = video_info(video)
        if info != report["video"] or sha256(folder/"calibration.jsonl") != report["calibration_sha256"]:
            raise ValueError("Calibration inputs changed")
        tracks_path, tracks_meta, tracks = tracking(args, clip, info)
        if sha256(tracks_path) != report["tracking_frames_sha256"]:
            raise ValueError("Tracking export changed")
        roles_path = Path("configs/calibration")/(clip+".roles.manual.json")
        roles = json.loads(roles_path.read_text())
        if roles.get("schema_version") != 1 or roles.get("tracking_frames_sha256") != sha256(tracks_path):
            raise ValueError("Role annotation provenance mismatch")
        shutil.copyfile(roles_path, folder/"roles.json")
        calibrations = [json.loads(line) for line in (folder/"calibration.jsonl").read_text().splitlines()]
        prepared = None
        if report.get("prepared_frames_sha256"):
            if sha256(folder/"prepared_frames.jsonl") != report["prepared_frames_sha256"]:
                raise ValueError("Prepared projection frames changed")
            prepared = [json.loads(line) for line in (folder/"prepared_frames.jsonl").read_text().splitlines()]
            if len(prepared) != info["frames"]:
                raise ValueError("Prepared projection frame count mismatch")
        cap = cv2.VideoCapture(str(video))
        writer = cv2.VideoWriter(str(folder/"projection.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (1280, 720))
        if not writer.isOpened():
            raise RuntimeError("Cannot open presentation encoder")
        counts = dict(frames=0, tracked_player_observations=0, shown_player_observations=0, person_observations=0,
                      calibrated_frames=0)
        frame_stats = []
        fields = ["frame_idx", "timestamp_s", "track_id", "role", "court_x_m", "court_y_m", "position_source", "shown_on_minimap"]
        try:
            with (folder/"frames.jsonl").open("x", encoding="utf8") as exported, (folder/"positions.csv").open("x", newline="", encoding="utf8") as csv_file:
                table = csv.DictWriter(csv_file, fields)
                table.writeheader()
                for index, calibration in enumerate(calibrations):
                    ok, frame = cap.read()
                    if not ok or calibration["frame_idx"] != index:
                        raise ValueError("Calibration/frame mismatch")
                    matrix = np.array(calibration["refined_image_to_court_h"]) if calibration["refined_image_to_court_h"] is not None else None
                    quality = dict(method="challenge_camera_unverified" if matrix is not None else "unavailable",
                                   reason=calibration.get("camera", {}).get("reason", calibration.get("reason")), accuracy_metrics=None)
                    rows = apply_roles([project_track(r, matrix, quality, COURT, (info["width"], info["height"])) for r in tracks[index]["tracks"]], roles, index)
                    if prepared is not None:
                        item = prepared[index]
                        if item["frame_idx"] != index or [r["track_id"] for r in item["tracks"]] != [r["track_id"] for r in tracks[index]["tracks"]]:
                            raise ValueError("Prepared tracks differ from source observations")
                        rows, quality = item["tracks"], item["calibration"]
                    # Draw current player boxes even when court calibration is missing.
                    for row in rows:
                        if row["role"] != "player":
                            continue
                        x1, y1, x2, y2 = map(round, row["bbox_xyxy"])
                        color = track_color(row["track_id"])
                        thickness = max(1, round(info["width"]/640))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                        cv2.putText(frame, str(row["track_id"]), (x1, max(14, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                                    .45*info["width"]/640, color, thickness, cv2.LINE_AA)
                    combined, shown = compose(frame, rows, COURT, quality, index/info["fps"], players_only=True)
                    writer.write(combined)
                    frame_stats.append(dict(frame_idx=index, calibration_available=matrix is not None,
                                            calibration_method=quality["method"],
                                            tracked_players=sum(r["role"] == "player" for r in rows), projected_players=len(shown)))
                    payload = dict(frame_idx=index, timestamp_s=index/info["fps"], image_to_court_h=calibration["refined_image_to_court_h"],
                                   calibration=quality, minimap_track_ids=shown, tracks=rows)
                    exported.write(json.dumps(payload, allow_nan=False)+"\n")
                    for row in rows:
                        xy = row["court_xy_m"] or [None, None]
                        table.writerow(dict(frame_idx=index, timestamp_s=index/info["fps"], track_id=row["track_id"], role=row["role"],
                                            court_x_m=xy[0], court_y_m=xy[1], position_source=row["position_source"], shown_on_minimap=row["track_id"] in shown))
                    counts["frames"] += 1
                    counts["person_observations"] += len(rows)
                    counts["tracked_player_observations"] += sum(r["role"] == "player" for r in rows)
                    counts["shown_player_observations"] += len(shown)
                    counts["calibrated_frames"] += matrix is not None
                    if index in (0, info["frames"]//2, info["frames"]-1):
                        cv2.imwrite(str(folder/f"frame-{index:04d}.jpg"), combined)
        finally:
            cap.release()
            writer.release()
        encode(folder/"projection.mp4", info["fps"])
        check = cv2.VideoCapture(str(folder/"projection.mp4"))
        decoded = 0
        while check.read()[0]:
            decoded += 1
        check.release()
        if decoded != info["frames"] or counts["frames"] != info["frames"]:
            raise ValueError("Presentation decode length mismatch")
        report.update(rendering=dict(status="complete", codec="H264", width=1280, height=720, fps=info["fps"],
                                      audio=False, full_decode_verified=True,
                                      source_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                                                     ("challenge_presentation.py", "challenge_presentation.html", "auto_calibration_cli.py", "minimap.py", "calibration.py", "court.py")}),
                      roles_sha256=sha256(roles_path), counts=counts, frame_stats=frame_stats)
        write_json(folder/"metadata.json", report)
        manifest["clips"][clip] = report
        print(f"Rendered {clip}: {counts}", flush=True)
    manifest["presentation_status"] = "complete"
    write_json(args.output/"experiment.json", manifest)
    write_page(args.output, manifest)


def write_page(output, manifest):
    if manifest.get("presentation_status") != "complete":
        raise ValueError("Presentation videos are incomplete")
    if manifest.get("temporal_smoothing"):
        import imageio_ffmpeg
        import subprocess
        for clip in CLIPS:
            comparison = output/clip/"comparison.mp4"
            if comparison.exists():
                continue
            # A single encoded comparison guarantees identical frame/timestamp
            # on both sides, including slow motion and frame stepping.
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-n", "-loglevel", "error",
                "-i", str(output/clip/"projection.mp4"),
                "-i", str(Path(manifest["source_presentation"])/clip/"projection.mp4"),
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an",
                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(comparison)], check=True)
            check = cv2.VideoCapture(str(comparison))
            count = 0
            while check.read()[0]:
                count += 1
            check.release()
            if count != manifest["clips"][clip]["video"]["frames"]:
                raise ValueError("Comparison video length mismatch")
    template_path = Path(__file__).with_name("challenge_presentation.html")
    template = template_path.read_text(encoding="utf8")
    (output/"index.html").write_text(template.replace("__REPORT_JSON__", json.dumps(manifest, allow_nan=False).replace("<", "\\u003c")), encoding="utf8")
    write_json(output/"presentation_metadata.json", dict(page_sha256=sha256(output/"index.html"),
               template_sha256=sha256(template_path), generator_sha256=sha256(Path(__file__)),
               video_sha256={clip: sha256(output/clip/"projection.mp4") for clip in CLIPS}))
    print(f"Browser presentation: {output/'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["infer", "render", "page"])
    parser.add_argument("--output", type=Path, default=Path("outputs/challenge-player-presentation-v1"))
    parser.add_argument("--models", type=Path, default=Path("models/kalicalib"))
    parser.add_argument("--tracking-root", type=Path, default=Path("outputs/occlusion-20261015T204150Z-029a7a35/reid"))
    args = parser.parse_args()
    if args.command == "page":
        write_page(args.output, json.loads((args.output/"experiment.json").read_text()))
    else:
        (infer if args.command == "infer" else render)(args)


if __name__ == "__main__":
    main()
