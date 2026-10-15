"""CPU-only source video metadata and evenly spaced contact sheets."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def inspect(path, output):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    samples, count = [], 0
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = {round(i * (declared - 1) / 5) for i in range(6)}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count in selected:
                tile = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tile.thumbnail((640, 360))
                samples.append((count, tile))
            count += 1
    finally:
        cap.release()
    sheet = Image.new("RGB", (1280, 3 * 390), "black")
    draw = ImageDraw.Draw(sheet)
    for i, (idx, tile) in enumerate(samples):
        x, y = (i % 2) * 640, (i // 2) * 390
        sheet.paste(tile, (x, y))
        draw.text((x + 5, y + 365), f"frame {idx}, {idx/fps:.2f}s", fill="white")
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output / f"{path.stem}.jpg")
    return dict(path=str(path), width=width, height=height, fps=fps,
                declared_frames=declared, decoded_frames=count, duration_s=count/fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/inspection"))
    args = parser.parse_args()
    report = [inspect(p, args.output) for p in args.videos]
    (args.output / "videos.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
