"""Manual floor-anchor annotation and CPU projection of existing tracking exports.

Run with python -m tacticalai.calibration_cli --help. No detector is imported.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration, project_track, transform
from .court import floor_lines, landmarks
from .export import sha256, write_json


def video_info(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    info = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return dict(info, sha256=sha256(video))


def read_frame(video, index):
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {index}")
        return frame
    finally:
        cap.release()


def verify_video(document, video):
    actual = video_info(video)
    if any(actual[k] != document["video"][k] for k in actual):
        raise ValueError("Calibration video hash/size/FPS/length does not match")
    return actual


def polylines(image, lines, matrix, color):
    height, width = image.shape[:2]
    for line in lines:
        pts = transform(line, matrix)
        for a, b in zip(pts, pts[1:]):
            if not np.isfinite([*a, *b]).all() or np.max(np.abs([a, b])) > 1e7:
                continue
            ok, p, q = cv2.clipLine((0, 0, width, height), tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)))
            if ok:
                cv2.line(image, p, q, color, 1, cv2.LINE_AA)


def map_canvas(court, height=360, width=640):
    canvas = np.full((height, width, 3), (38, 46, 40), dtype=np.uint8)
    scale = min((width-50)/court["length_m"], (height-90)/court["width_m"])
    matrix = np.array([[scale, 0, 25], [0, scale, 55], [0, 0, 1]], dtype=float)
    polylines(canvas, floor_lines(court), matrix, (190, 190, 190))
    return canvas, matrix


def label(image, text, y, color=(255, 255, 255)):
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)


def annotate(args):
    """Native OpenCV click tool, intended to run locally from VS Code."""
    document = json.loads(args.calibration.read_text(encoding="utf-8"))
    info = verify_video(document, args.video)
    if not 0 <= args.frame < info["frames"]:
        raise ValueError("Frame index outside video")
    source = read_frame(args.video, args.frame)
    scale = min(1280/info["width"], 720/info["height"])
    width, height = round(info["width"]*scale), round(info["height"]*scale)
    points = landmarks(document["court"])
    names = list(points)
    key = next((k for k in document["keyframes"] if k["frame_idx"] == args.frame), None)
    anchors = json.loads(json.dumps(key["anchors"])) if key else {}
    state = dict(selected=0, mode="fit", preview=True)

    def click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        name = names[state["selected"]]
        if x < width and y < height:
            anchors[name] = dict(image_xy=[x/scale, y/scale], use=state["mode"])
        elif x >= width:
            candidates = transform(list(points.values()), map_matrix)
            distances = np.linalg.norm(candidates - [x-width, y], axis=1)
            if distances.min() < 20:
                state["selected"] = int(distances.argmin())

    window = "Manual court anchors"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            frame = cv2.resize(source, (width, height))
            panel, map_matrix = map_canvas(document["court"], height, 640)
            name = names[state["selected"]]
            if state["preview"]:
                from .calibration import fit
                try:
                    matrix, diagnostic = fit(anchors, document["court"])
                    scaled = np.diag([scale, scale, 1]) @ np.linalg.inv(matrix)
                    polylines(frame, floor_lines(document["court"]), scaled, (0, 255, 255))
                    label(panel, f"Fit RMSE {diagnostic['fit_rmse_px']:.2f}px (not accuracy)", 60)
                    if diagnostic["checks"]:
                        label(panel, f"Held-out max {max(c['error_px'] for c in diagnostic['checks']):.2f}px", 80)
                except ValueError:
                    pass
            for n, xy in points.items():
                p = tuple(np.rint(transform([xy], map_matrix)[0]).astype(int))
                color = (0, 255, 255) if n == name else ((0, 210, 0) if n in anchors else (180, 180, 180))
                cv2.circle(panel, p, 5, color, -1)
            for n, anchor in anchors.items():
                p = tuple(np.rint(np.array(anchor["image_xy"])*scale).astype(int))
                color = (255, 170, 0) if anchor.get("use") == "check" else (0, 220, 0)
                cv2.circle(frame, p, 4, color, -1)
                cv2.putText(frame, n, p, cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)
            label(panel, f"Frame {args.frame}: {name}", 20)
            label(panel, f"Mode: {state['mode']} | {len(anchors)} anchors", 40)
            label(panel, "Click map dot, then matching floor point in video", height-48)
            label(panel, "N/P select | F fit / C check | U remove selected", height-30)
            label(panel, "V preview | S save keyframe | Q/Esc cancel", height-12)
            cv2.imshow(window, np.hstack((frame, panel)))
            pressed = cv2.waitKey(30) & 0xff
            if pressed in (27, ord("q")):
                return
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return
            if pressed in (ord("n"), ord("p")):
                state["selected"] = (state["selected"] + (1 if pressed == ord("n") else -1)) % len(names)
            if pressed in (ord("f"), ord("c")):
                state["mode"] = "fit" if pressed == ord("f") else "check"
            if pressed == ord("u"):
                anchors.pop(name, None)
            if pressed == ord("v"):
                state["preview"] = not state["preview"]
            if pressed == ord("s"):
                from .calibration import fit
                try:
                    fit(anchors, document["court"])
                except ValueError as error:
                    print(f"Cannot save: {error}")
                    continue
                if key is None:
                    key = dict(frame_idx=args.frame, interpolate_to_next=False)
                    document["keyframes"].append(key)
                key["anchors"] = anchors
                document["keyframes"].sort(key=lambda k: k["frame_idx"])
                # Preserve prior annotations before an intentional interactive edit.
                backup = args.calibration.with_suffix(f".{time.time_ns()}.bak.json")
                backup.write_bytes(args.calibration.read_bytes())
                write_json(args.calibration, document)
                print(f"Saved {args.calibration}; backup {backup}")
                return
    finally:
        cv2.destroyWindow(window)


def render(args):
    started = time.perf_counter()
    document = json.loads(args.calibration.read_text(encoding="utf-8"))
    info = verify_video(document, args.video)
    calibration = Calibration(document)
    if args.frame is not None and not 0 <= args.frame < info["frames"]:
        raise ValueError("Frame index outside video")
    meta = json.loads((args.tracks.parent/"metadata.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete" or meta["video_sha256"] != info["sha256"]:
        raise ValueError("Tracking run is incomplete or belongs to another video")
    layout = getattr(args, "layout", "diagnostic")
    codec = getattr(args, "codec", "mp4v")
    roles_path = getattr(args, "roles", None)
    roles = json.loads(roles_path.read_text(encoding="utf-8")) if roles_path else {}
    if roles and (roles.get("schema_version") != 1 or roles.get("tracking_frames_sha256") != sha256(args.tracks)):
        raise ValueError("Role annotations do not match this tracking export")
    if getattr(args, "players_only", False) and not roles:
        raise ValueError("--players-only requires run-specific --roles annotations")
    for entry in roles.get("intervals", []):
        if entry["role"] not in ("player", "referee", "staff", "unknown") or not 0 <= entry["start_frame"] <= entry["end_frame"] < info["frames"]:
            raise ValueError("Invalid manual role interval")
    if codec == "h264":
        import imageio_ffmpeg
        encoder = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"calibration.json", document)
    write_json(args.output/"tracking_metadata.json", meta)
    if roles:
        write_json(args.output/"roles.json", roles)
    cap = cv2.VideoCapture(str(args.video))
    writer = None
    counts = dict(frames=0, calibrated_frames=0, track_observations=0, projected_observations=0)
    key_indices = {k["frame_idx"] for k in calibration.keys}
    report = dict(status="running", video=info, calibration_sha256=sha256(args.calibration),
                  tracking_frames_sha256=sha256(args.tracks), tracking_metadata_sha256=sha256(args.tracks.parent/"metadata.json"),
                  versions=dict(opencv=cv2.__version__, numpy=np.__version__),
                  source_sha256={n: sha256(Path(__file__).with_name(n)) for n in ("court.py", "calibration.py", "calibration_cli.py", "minimap.py")},
                  accuracy_metrics=None, annotations_status=document.get("status", "unreviewed"),
                  rendering=dict(layout=layout, codec=codec, players_only=getattr(args, "players_only", False),
                                 map_flip_x=getattr(args, "map_flip_x", False), audio=False),
                  roles_sha256=sha256(roles_path) if roles_path else None,
                  keyframe_diagnostics={str(index): diagnostic for index, (_, diagnostic) in calibration.fitted.items()},
                  notes="All confirmed person tracks retained; roles and ground contact unverified. No off-screen inference. Fit residuals are not player accuracy.")
    write_json(args.output/"metadata.json", report)
    fields = ["frame_idx", "timestamp_s", "track_id", "role", "foot_x_px", "foot_y_px", "court_x_m", "court_y_m", "court_region", "position_source", "calibration_method", "foot_contact_visibility"]
    try:
        with args.tracks.open(encoding="utf-8") as source, (args.output/"frames.jsonl").open("w", encoding="utf-8") as out, (args.output/"positions.csv").open("w", newline="", encoding="utf-8") as table:
            csv_out = csv.DictWriter(table, fieldnames=fields)
            csv_out.writeheader()
            for idx in range(info["frames"]):
                line = source.readline()
                if not line:
                    raise ValueError("Tracking export ended before the video")
                data = json.loads(line)
                if data["frame_idx"] != idx or abs(data["timestamp_s"]-idx/info["fps"]) > 1e-6:
                    raise ValueError("Tracking export frame/timestamp mismatch")
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Video ended early")
                if args.frame is not None and idx != args.frame:
                    continue
                matrix, quality = calibration.at(idx)
                rows = [project_track(row, matrix, quality, calibration.court, (info["width"], info["height"])) for row in data["tracks"]]
                from .minimap import apply_roles, compose
                rows = apply_roles(rows, roles, idx)
                combined, shown = None, None
                if layout == "minimap":
                    combined, shown = compose(frame, rows, calibration.court, quality, idx/info["fps"],
                                              players_only=getattr(args, "players_only", False),
                                              flip_x=getattr(args, "map_flip_x", False))
                counts["frames"] += 1
                counts["calibrated_frames"] += matrix is not None
                counts["track_observations"] += len(rows)
                counts["projected_observations"] += sum(r["court_xy_m"] is not None for r in rows)
                payload = dict(frame_idx=idx, timestamp_s=idx/info["fps"], calibration=quality,
                               image_to_court_h=matrix.tolist() if matrix is not None else None, tracks=rows,
                               minimap_track_ids=shown)
                out.write(json.dumps(payload, allow_nan=False)+"\n")
                for row in rows:
                    xy = row["court_xy_m"] or [None, None]
                    csv_out.writerow(dict(frame_idx=idx, timestamp_s=row["timestamp_s"], track_id=row["track_id"], role=row["role"],
                                          foot_x_px=row["footpoint_image_xy"][0], foot_y_px=row["footpoint_image_xy"][1],
                                          court_x_m=xy[0], court_y_m=xy[1], court_region=row["court_region"],
                                          position_source=row["position_source"], calibration_method=quality["method"], foot_contact_visibility="unknown"))
                if combined is None:
                    if matrix is not None:
                        polylines(frame, floor_lines(calibration.court), np.linalg.inv(matrix), (0, 255, 255))
                    keyframe = next((k for k in calibration.keys if k["frame_idx"] == idx), None)
                    if keyframe is not None:
                        for name, anchor in keyframe["anchors"].items():
                            p = tuple(np.rint(anchor["image_xy"]).astype(int))
                            color = (255, 180, 0) if anchor.get("use") == "check" else (0, 220, 0)
                            cv2.circle(frame, p, 3, color, -1)
                            cv2.putText(frame, name, p, cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
                    # A fixed-size preview keeps 1080p clip rendering inexpensive on CPU.
                    display = cv2.resize(frame, (640, 360))
                    panel, map_matrix = map_canvas(calibration.court)
                    label(display, f"Frame {idx} | {quality['method']}", 18)
                    label(display, f"Calibration status: {document.get('status', 'unreviewed')}", 36, (0, 200, 255))
                    label(panel, "PERSON tracks | role/contact unverified", 18)
                    label(panel, "Approximate floor projection; no hidden positions", 35)
                    for row in rows:
                        foot = tuple(np.rint(np.array(row["footpoint_image_xy"])*[640/info["width"], 360/info["height"]]).astype(int))
                        cv2.circle(display, foot, 3, (0, 170, 255), -1)
                        cv2.putText(display, str(row["track_id"]), foot, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                        if row["court_region"] == "inside":
                            p = tuple(np.rint(transform([row["court_xy_m"]], map_matrix)[0]).astype(int))
                            cv2.circle(panel, p, 4, (0, 170, 255), -1)
                            cv2.putText(panel, str(row["track_id"]), p, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                    label(panel, f"Calibration: {quality['method']}", 340)
                    combined = np.hstack((display, panel))
                if idx in key_indices or args.frame is not None:
                    cv2.imwrite(str(args.output/f"frame-{idx:04d}.png"), combined)
                    cv2.imwrite(str(args.output/f"overlay-{idx:04d}.png"), frame)
                if args.frame is None:
                    if writer is None:
                        writer = cv2.VideoWriter(str(args.output/"projection.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (combined.shape[1], combined.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("Cannot open MP4 encoder")
                    writer.write(combined)
            if source.readline():
                raise ValueError("Tracking export has more frames than video")
        if writer is not None:
            writer.release()
            writer = None
            if codec == "h264":
                import subprocess
                converted = args.output/"projection.h264.mp4"
                subprocess.run([encoder, "-n", "-loglevel", "error", "-i", str(args.output/"projection.mp4"),
                                "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(converted)], check=True)
                converted.replace(args.output/"projection.mp4")
            check = cv2.VideoCapture(str(args.output/"projection.mp4"))
            decoded = 0
            try:
                while check.read()[0]:
                    decoded += 1
            finally:
                check.release()
            if decoded != counts["frames"]:
                raise ValueError("Rendered MP4 did not decode to the expected frame count")
        report.update(status="complete", counts=counts, elapsed_s=time.perf_counter()-started)
        write_json(args.output/"metadata.json", report)
        print(json.dumps(counts))
    except Exception as error:
        report.update(status="failed", error=str(error), counts=counts)
        write_json(args.output/"metadata.json", report)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "annotate", "render"):
        sub = commands.add_parser(name)
        sub.add_argument("--video", required=True, type=Path)
        sub.add_argument("--calibration", required=True, type=Path)
        if name == "annotate":
            sub.add_argument("--frame", required=True, type=int)
        if name == "render":
            sub.add_argument("--layout", choices=["diagnostic", "minimap"], default="diagnostic")
            sub.add_argument("--codec", choices=["mp4v", "h264"], default="mp4v", help="H.264 requires imageio-ffmpeg")
            sub.add_argument("--roles", type=Path, help="Optional run-specific manual role intervals")
            sub.add_argument("--players-only", action="store_true", help="Minimap shows only manually labelled players")
            sub.add_argument("--map-flip-x", action="store_true", help="Mirror diagram x for display only")
            sub.add_argument("--frame", type=int, help="Render one frame only; otherwise render full clip")
            sub.add_argument("--tracks", required=True, type=Path, help="Existing frames.jsonl; sibling metadata.json required")
            sub.add_argument("--output", required=True, type=Path, help="New directory")
    args = parser.parse_args()
    if args.command == "init":
        document = dict(schema_version=1, status="draft", video=video_info(args.video),
                        court=dict(length_m=28.0, width_m=15.0, key_depth_m=5.8, key_width_m=4.9, three_point_corner_offset_m=.9,
                                   assumption="FIBA-style nominal dimensions; verify venue. Line-centre approximation.",
                                   reference="https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-v10a.pdf"),
                        orientation="x along court, y far to near; diagram left/right unrelated to image left/right",
                        ransac_threshold_px=4.0, max_interpolation_gap_frames=0, keyframes=[])
        args.calibration.parent.mkdir(parents=True, exist_ok=True)
        with args.calibration.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
    elif args.command == "annotate":
        annotate(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
