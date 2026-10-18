"""Refine cached KaliCalib review cameras against yellow court markings."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .auto_calibration_cli import manual_agreement
from .calibration_cli import read_frame, video_info
from .court_line_refinement import refine
from .export import sha256, write_json
from .kalicalib_reference_cli import overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("outputs/homography-kalicalib-reference-v1"))
    parser.add_argument("--tracking-root", type=Path, default=Path("outputs/occlusion-20261015T204150Z-029a7a35/reid"))
    args = parser.parse_args()
    root = args.reference
    if (root/"line-experiment.json").exists():
        raise FileExistsError("Line experiment already exists")
    report = json.loads((root/"experiment.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Incomplete reference experiment")
    evidence = {}
    for clip in report["frames"]:
        video = Path("data/raw")/(clip+".mp4")
        if video_info(video) != report["inputs"][clip]["video"]:
            raise ValueError("Video changed since reference experiment")
        rows = args.tracking_root/clip/"tracktrack/frames.jsonl"
        metadata = json.loads(rows.with_name("metadata.json").read_text())
        if metadata["status"] != "complete" or metadata["video_sha256"] != report["inputs"][clip]["video"]["sha256"]:
            raise ValueError("Person masks belong to different footage")
        evidence[clip] = dict(tracks=[json.loads(l) for l in rows.read_text().splitlines()],
                              manual=json.loads((root/clip/"manual_reference.json").read_text()))
    report["line_refinement"] = dict(status="running", scipy=importlib.metadata.version("scipy"),
        initialization="accepted automatic camera only", manual_used_for_optimization=False,
        settings=dict(working_size=[960, 540], hsv_low=[18, 25, 100], hsv_high=[45, 255, 255],
                      opening_kernel=9, person_margin_working_px=7, score_exclusion_fraction=.15,
                      max_control_shift_working_px=20, distance_cap_px=25, regularization=.018,
                      minimum_cost_improvement=.05, maximum_evaluations=80, maximum_initial_line_distance_px=15),
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                       ("court_line_refinement.py", "court_line_refinement_cli.py", "kalicalib_reference_review.html")})
    for name in ("court_line_refinement.py", "court_line_refinement_cli.py"):
        (root/"run_source"/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    started = time.perf_counter()
    for row in report["results"]:
        clip, index = row["clip"], row["frame_idx"]
        frame = read_frame(Path("data/raw")/(clip+".mp4"), index)
        manual = evidence[clip]["manual"]
        boxes = [r["bbox_xyxy"] for r in evidence[clip]["tracks"][index]["tracks"]]
        initial = np.array(row["refined_image_to_court_h"]) if row["refined_image_to_court_h"] is not None else None
        tick = time.perf_counter()
        matrix, diagnostic, mask = refine(frame, initial, manual["court"], boxes)
        diagnostic["elapsed_s"] = time.perf_counter()-tick
        row.update(line_image_to_court_h=matrix.tolist() if matrix is not None else None, line_diagnostic=diagnostic)
        key = next((k for k in manual["keyframes"] if k["frame_idx"] == index), None)
        row["lines_manual_agreement"] = manual_agreement(matrix, key, manual["court"])
        name = row["checkpoint"]
        image = overlay(frame, manual["court"], matrix, f"{name.upper()} | LINE REFINEMENT | frame {index}",
                        "Experimental local line alignment; no manual initialization" if matrix is not None else diagnostic["reason"])
        cv2.imwrite(str(root/clip/f"{name}-lines-{index:04d}.jpg"), image)
        if mask is not None:
            cv2.imwrite(str(root/clip/f"{name}-line-mask-{index:04d}.png"), mask)
        write_json(root/"line-experiment.json", report)
        print(f"{name} {clip} {index}: {diagnostic['reason'] or 'line candidate'}; agreement={row['lines_manual_agreement']}", flush=True)
    report["line_refinement"].update(status="complete", elapsed_s=time.perf_counter()-started)
    write_json(root/"line-experiment.json", report)
    template = Path(__file__).with_name("kalicalib_reference_review.html").read_text(encoding="utf8")
    template = template.replace('href="experiment.json"', 'href="line-experiment.json"')
    (root/"review-lines.html").write_text(template.replace("__REPORT_JSON__", json.dumps(report, allow_nan=False).replace("<", "\\u003c")), encoding="utf8")
    print(f"Line comparison: {root/'review-lines.html'}", flush=True)


if __name__ == '__main__':
    main()
