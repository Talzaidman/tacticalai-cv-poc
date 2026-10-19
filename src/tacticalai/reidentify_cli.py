"""Run CLIP-ReIdent on saved tracklets and review offline identity hypotheses."""
import argparse
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

from .clip_reident import ClipReIdent, UPSTREAM_COMMIT
from .export import sha256, write_json
from .tracklet_linking import link_tracklets, split_tracklets


def run(args):
    import torch
    if args.samples < 1 or args.min_observations < 1 or args.batch_size < 1:
        raise ValueError("Sample, observation and batch counts must be positive")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu or a Colab GPU")
    source_meta = json.loads((args.run / "metadata.json").read_text())
    if source_meta["status"] != "complete" or sha256(args.video) != source_meta["video_sha256"]:
        raise ValueError("Requires a complete source run and its exact original video")
    frames_path = args.run / "frames.jsonl"
    frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
    if [f["frame_idx"] for f in frames] != list(range(len(frames))):
        raise ValueError("Source frames must be contiguous and zero based")
    segments = split_tracklets(frames)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "crops").mkdir()
    started = time.perf_counter()
    meta = dict(status="running", method="CLIP-ReIdent Paper ViT-L-14 + offline mutual-best links",
                upstream_commit=UPSTREAM_COMMIT, video=str(args.video.resolve()),
                video_sha256=source_meta["video_sha256"], source_frames_sha256=sha256(frames_path),
                source_metadata=source_meta, config={k: str(v) if isinstance(v, Path) else v
                                                   for k, v in vars(args).items()},
                python=sys.version, platform=platform.platform(),
                gpu=torch.cuda.get_device_name(0) if args.device != "cpu" else None,
                versions={p: importlib.metadata.version(p) for p in
                          ("torch", "torchvision", "open-clip-torch", "numpy", "opencv-python")},
                source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                accuracy_metrics=None, identity_status="unvalidated hypotheses",
                note="Reuses saved tracks; source buffer is recorded, not changed by postprocessing. "
                     "No hidden detections or ID-steal repair. No challenge reranking/ensemble.")
    write_json(args.output / "metadata.json", meta)
    try:
        torch.set_num_threads(args.cpu_threads)
        requests = {}
        for segment in segments:
            obs = segment["observations"]
            segment["sample_frames"] = []
            if len(obs) < args.min_observations:
                continue
            indices = np.unique(np.linspace(0, len(obs) - 1, min(args.samples, len(obs))).round().astype(int))
            for i in indices:
                row = obs[i]
                requests.setdefault(row["frame_idx"], []).append((segment, row))
        cap = cv2.VideoCapture(str(args.video))
        crops, crop_segments = [], []
        for frame_idx in range(len(frames)):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError("Source video shorter than exported frames")
            for segment, row in requests.get(frame_idx, []):
                x1, y1, x2, y2 = row["bbox_xyxy"]
                x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
                x2, y2 = min(frame.shape[1], int(np.ceil(x2))), min(frame.shape[0], int(np.ceil(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2].copy()
                sid = segment["segment_id"]
                if not cv2.imwrite(str(args.output / "crops" / f"s{sid}-f{frame_idx}.jpg"), crop):
                    raise RuntimeError("Could not save review crop")
                crops.append(crop)
                crop_segments.append(sid)
                segment["sample_frames"].append(frame_idx)
        cap.release()
        if not crops:
            raise ValueError("No valid tracklet crops")
        print(f"Preparing {len(crops)} real crops from {len(segments)} tracklets on {args.device}", flush=True)
        reuse = getattr(args, "reuse_embeddings", None)
        checkpoint_hash = sha256(args.checkpoint)
        meta.update(checkpoint_sha256=checkpoint_hash, checkpoint=str(args.checkpoint.resolve()),
                    preprocessing="RGB, aspect-pad black 224, INTER_LINEAR_EXACT, CLIP mean/std",
                    activation="QuickGELU", remove_projection=True, precision="float32")
        embedding_started = time.perf_counter()
        if reuse is None:
            encoder = ClipReIdent(args.checkpoint, args.device, args.batch_size)
            embeddings = encoder.encode(crops)
        else:
            cached_meta = json.loads((reuse / 'metadata.json').read_text())
            if cached_meta['status'] != 'complete' or any(
                cached_meta[k] != meta[k] for k in ('source_frames_sha256', 'video_sha256', 'checkpoint_sha256')):
                raise ValueError('Cached embeddings have different source data or checkpoint')
            if any(cached_meta['config'][k] != getattr(args, k) for k in ('samples', 'min_observations')):
                raise ValueError('Cached embeddings use different crop sampling')
            with np.load(reuse / 'embeddings.npz', allow_pickle=False) as cached:
                if not np.array_equal(cached['segment_ids'], crop_segments):
                    raise ValueError('Cached embedding segment order differs')
                embeddings = cached['embeddings']
            if len(embeddings) != len(crops) or not np.isfinite(embeddings).all():
                raise ValueError('Invalid cached embeddings')
            meta['embeddings_reused_from'] = str(reuse.resolve())
            meta['cached_embedding_seconds'] = cached_meta['embedding_seconds']
        meta["embedding_seconds"] = time.perf_counter() - embedding_started
        np.savez_compressed(args.output / "embeddings.npz", embeddings=embeddings,
                            segment_ids=np.array(crop_segments))
        features = {}
        for sid in sorted(set(crop_segments)):
            mean = embeddings[np.array(crop_segments) == sid].mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 1e-8:
                features[sid] = mean / norm
        identities, candidates = link_tracklets(segments, features, args.threshold, args.margin, args.max_gap)
        write_json(args.output / "links.json", candidates)
        write_json(args.output / "tracklets.json", [dict(
            segment_id=s["segment_id"], source_track_id=s["source_track_id"],
            start=s["start"], end=s["end"], observations=len(s["observations"]),
            sample_frames=s["sample_frames"], player_id=identities[s["segment_id"]],
            identity_status="hypothesis") for s in segments])
        lookup = {(r["frame_idx"], s["source_track_id"]): s["segment_id"]
                  for s in segments for r in s["observations"]}
        with (args.output / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for frame in frames:
                tracks = []
                for row in frame["tracks"]:
                    sid = lookup[(frame["frame_idx"], row["track_id"])]
                    tracks.append(dict(row, segment_id=sid, player_id=identities[sid],
                                       identity_status="hypothesis"))
                stream.write(json.dumps(dict(frame, tracks=tracks)) + "\n")
        _review(args.output, segments, identities, candidates)
        _video(args.video, args.output, frames, lookup, identities, source_meta["source"]["fps"])
        meta.update(status="complete", frames_processed=len(frames), crops=len(crops),
                    tracklets=len(segments), accepted_links=sum(c["accepted"] for c in candidates),
                    candidate_links=len(candidates), hypothesized_identities=len(set(identities.values())),
                    total_seconds=time.perf_counter() - started)
        (args.output / "environment.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        print(json.dumps({k: meta[k] for k in ("status", "crops", "tracklets", "accepted_links", "total_seconds")}), flush=True)
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(args.output / "metadata.json", meta)


def _review(output, segments, identities, candidates):
    def segment_html(sid):
        s = segments[sid]
        images = ''.join(f'<img src="crops/s{sid}-f{f}.jpg" title="frame {f}">' for f in s["sample_frames"])
        return f's{sid}; raw ID {s["source_track_id"]}; frames {s["start"]}–{s["end"]}; P{identities[sid]}<br>{images}'
    ordered = sorted(candidates, key=lambda c: (not c["accepted"], -c["cosine_similarity"]))
    rows = ''.join(f'<tr><td>{segment_html(c["before"])}</td><td>{segment_html(c["after"])}</td>'
                   f'<td>{c["cosine_similarity"]:.4f}<br>gap {c["gap_frames"]}<br>{html.escape(c["reason"])}</td></tr>'
                   for c in ordered[:100])
    (output / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>CLIP-ReIdent review</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#151b24;color:#eee}'
        'td{padding:12px;border:1px solid #526070}img{height:140px;max-width:140px;object-fit:contain;margin:3px}'
        'video{max-width:100%}</style><h1>CLIP-ReIdent: disappearance / return</h1>'
        '<p>Unvalidated identity hypotheses. Similarity is not a probability. Raw IDs are preserved. '
        'No hidden observations are filled. Top 100 candidate pairs; accepted links first.</p>'
        '<video src="annotated.mp4" controls></video><table><tr><th>Before</th><th>After</th><th>Evidence</th></tr>'
        + rows + '</table>', encoding="utf-8")


def _video(video, output, frames, lookup, identities, fps):
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        if not writer.isOpened():
            raise RuntimeError("MP4 writer unavailable")
        for frame in frames:
            ok, image = cap.read()
            if not ok:
                raise RuntimeError("Video decode failed during overlay")
            for row in frame["tracks"]:
                sid = lookup[(frame["frame_idx"], row["track_id"])]
                pid = identities[sid]
                color = tuple(int(x) for x in np.random.default_rng(pid).integers(80, 255, size=3))
                x1, y1, x2, y2 = map(int, row["bbox_xyxy"])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f'P{pid}? / T{row["track_id"]}', (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(image, 'P = identity hypothesis; T = source track', (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
        cap.release()
    check = cv2.VideoCapture(str(output / "annotated.mp4"))
    count = 0
    while check.read()[0]:
        count += 1
    check.release()
    if count != len(frames):
        raise RuntimeError("Incomplete annotated video")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing complete tracker output directory")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/clip-reident/model/ViT-L-14_openai/Paper/weights_e4.pth"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--max-gap", type=int, default=150, help="Missing frames for offline linking; separate from track buffer")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--reuse-embeddings", type=Path, help="Completed matching run with identical crops/model; no new inference")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
