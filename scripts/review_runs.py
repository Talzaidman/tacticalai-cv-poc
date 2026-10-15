"""Validate baseline exports and package existing results, without inference."""
import csv
import json
from pathlib import Path
import zipfile


def review(root, package=True):
    root = Path(root)
    rows, issues, metadata = [], [], {}
    for clip, expected in (("attack_clip", 150), ("attack_clip2", 175)):
        for tracker in ("botsort", "tracktrack"):
            folder = root / clip / tracker
            label = f"{clip}/{tracker}"
            try:
                meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
                metadata[label] = meta
                if meta.get("status") != "complete":
                    raise ValueError(f"status={meta.get('status')}: {meta.get('error', '')}")
                if meta.get("frames_processed") != expected or meta.get("truncated"):
                    raise ValueError("Unexpected frame count or truncated run")
                counts = {"detections": 0, "tracks": 0}
                ids, empty = set(), 0
                count = 0
                fps = meta["source"]["fps"]
                for index, line in enumerate((folder / "frames.jsonl").read_text(encoding="utf-8").splitlines()):
                    frame = json.loads(line)
                    if frame["frame_idx"] != index or abs(frame["timestamp_s"] - index/fps) > 1e-6:
                        raise ValueError(f"Frame index/timestamp mismatch at {index}")
                    for kind in counts:
                        counts[kind] += len(frame[kind])
                    ids.update(obj["track_id"] for obj in frame["tracks"])
                    empty += not frame["tracks"]
                    count += 1
                if count != expected or None in ids:
                    raise ValueError("Incomplete JSONL or null ID in confirmed tracks")
                for kind in counts:
                    with (folder / f"{kind}.csv").open(newline="", encoding="utf-8") as stream:
                        csv_count = sum(1 for _ in csv.DictReader(stream))
                    if csv_count != counts[kind] or counts[kind] != meta[f"{kind[:-1] if kind == 'tracks' else 'detection'}_rows"]:
                        raise ValueError(f"{kind} counts disagree between CSV, JSONL and metadata")
                if not (folder / "annotated.mp4").is_file() or (folder / "annotated.mp4").stat().st_size == 0:
                    raise ValueError("Missing/empty annotated video")
                settings = meta["tracker_config"]
                rows.append(dict(run=label, frames=count, **counts, unique_ids=len(ids),
                                 frames_without_tracks=empty,
                                 processing_s=meta["timing"]["frame_loop_s"],
                                 pipeline_fps=meta["timing"]["pipeline_fps"],
                                 with_reid=settings["with_reid"], gmc=settings["gmc_method"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.append(f"{label}: {error}")
        pair = [metadata.get(f"{clip}/{name}") for name in ("botsort", "tracktrack")]
        if all(pair):
            for field in ("video_sha256", "model_sha256", "config", "versions", "gpu", "source"):
                if field not in pair[0] or field not in pair[1] or pair[0][field] != pair[1][field]:
                    issues.append(f"{clip}: comparison mismatch/missing {field}")
    report = dict(artifact_checks_passed=not issues, runs=rows, issues=issues,
                  accuracy_metrics=None,
                  note="Counts are diagnostics, not player accuracy or ID switches. "
                       "MP4 existence checked here; runner verifies decoded frame count. "
                       "Visual review and labeled ground truth still required.")
    (root / "review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Baseline artifact review", "", report["note"], "",
             "| Run | Frames | Detections | Track observations | Unique IDs | Loop seconds | Pipeline FPS |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['frames']} | {row['detections']} | {row['tracks']} | "
                     f"{row['unique_ids']} | {row['processing_s']:.3f} | {row['pipeline_fps']:.2f} |")
    lines += ["", "Artifact checks: " + ("passed" if not issues else "FAILED"), ""]
    lines.extend(f"- {issue}" for issue in issues)
    (root / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # ZIP contains existing experiment products, no source footage or model weights.
    if not package:
        print("\n".join(lines))
        return report
    archive_path = root / "baseline-results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for clip in ("attack_clip", "attack_clip2", "smoke"):
            for path in sorted((root / clip).rglob("*")):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".yaml", ".txt", ".mp4"}:
                    archive.write(path, path.relative_to(root).as_posix())
        for name in ("review.json", "review.md"):
            archive.write(root / name, name)
    print("\n".join(lines))
    print(f"Results archive: {archive_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    result = review(arguments.root)
    raise SystemExit(0 if result["artifact_checks_passed"] else 1)
