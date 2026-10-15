"""CLI for one video, with fresh model/tracker state for each experiment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; existing runs are never overwritten")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--tracker", choices=["botsort", "tracktrack", "deepocsort", "none"],
                        help="Override comparison list; none runs detection only")
    parser.add_argument("--device", help="0 for Colab CUDA; cpu for a short local smoke run")
    parser.add_argument("--max-frames", type=int, help="Optional short smoke run")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not args.video.is_file():
        parser.error(f"Missing video: {args.video}")
    if not isinstance(config.get("with_reid", False), bool):
        parser.error("with_reid must be true or false")
    if "track_buffer" in config and (type(config["track_buffer"]) is not int or config["track_buffer"] < 1):
        parser.error("track_buffer must be a positive integer")
    for name in ("conf", "iou"):
        if not 0 < config[name] <= 1:
            parser.error(f"{name} must be in (0, 1]")
    if not isinstance(config["imgsz"], int) or config["imgsz"] <= 0:
        parser.error("imgsz must be a positive integer")
    trackers = [args.tracker] if args.tracker else config["trackers"]
    if not trackers or any(t not in {"botsort", "tracktrack", "deepocsort", "none"} for t in trackers):
        parser.error("Unsupported or empty trackers list")
    from .runner import run
    args.output.mkdir(parents=True, exist_ok=False)
    for tracker in trackers:
        run(args.video.resolve(), args.output / tracker, config, tracker, args.max_frames)


if __name__ == "__main__":
    main()
