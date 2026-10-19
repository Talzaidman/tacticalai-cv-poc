"""Compare original track-ID galleries using cached CLIP-ReIdent crop features."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from .export import sha256, write_json
from .gallery_linking import associate_galleries


def build_galleries(frames, segments, embeddings, segment_ids, cache, output, maximum=8,
                    min_spacing=0, max_overlap=1.0):
    tracks, observations = {}, {}
    for frame in frames:
        for row in frame['tracks']:
            tid, f = row['track_id'], frame['frame_idx']
            tracks.setdefault(tid, dict(start=f, end=f, frames=[]))
            tracks[tid]['end'] = f
            tracks[tid]['frames'].append(f)
            observations[f, tid] = row
    samples = {}
    for s in segments:
        indices = np.flatnonzero(segment_ids == s['segment_id'])
        if len(indices) != len(s['sample_frames']):
            raise ValueError('Cached sample/embedding mismatch')
        for index, f in zip(indices, s['sample_frames']):
            tid = s['source_track_id']
            crop_path = cache / 'crops' / f"s{s['segment_id']}-f{f}.jpg"
            crop = cv2.imread(str(crop_path))
            if crop is None:
                raise ValueError(f'Missing cached crop: {crop_path}')
            row = observations[f, tid]
            box = row['bbox_xyxy']
            area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            cover = 0.0
            for other in frames[f]['tracks']:
                if other['track_id'] == tid:
                    continue
                b = other['bbox_xyxy']
                intersection = max(0, min(box[2], b[2]) - max(box[0], b[0])) * max(0, min(box[3], b[3]) - max(box[1], b[1]))
                cover = max(cover, intersection / area)
            sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            quality = row['det_confidence'] * np.sqrt(area) * (1 - cover) * min(1, sharpness / 100)
            samples.setdefault(tid, []).append(dict(frame=f, index=int(index), path=crop_path,
                                                    quality=float(quality), overlap=float(cover), sharpness=sharpness))
    galleries, records = {}, {}
    for tid in tracks:
        pool = sorted(samples.get(tid, []), key=lambda s: s['frame'])
        selected = [max([pool[i] for i in indices], key=lambda s: s['quality'])
                    for indices in np.array_split(np.arange(len(pool)), min(maximum, len(pool)))] if pool else []
        diverse = []
        for sample in sorted(selected, key=lambda s: -s['quality']):
            if sample['overlap'] <= max_overlap and all(abs(sample['frame']-old['frame']) >= min_spacing for old in diverse):
                diverse.append(sample)
        selected = sorted(diverse, key=lambda s: s['frame'])
        records[tid] = []
        for sample in selected:
            name = f'gallery/{tid}-f{sample["frame"]}.jpg'
            shutil.copyfile(sample['path'], output / name)
            records[tid].append(dict(frame=sample['frame'], image=name, quality=sample['quality'],
                                     overlap=sample['overlap'], sharpness=sample['sharpness']))
        if len(selected) >= 3:
            galleries[tid] = embeddings[[s['index'] for s in selected]]
    return tracks, galleries, records


def run(cache_root, output, threshold=0.98, margin=0.003, comparison='nearest', min_spacing=0, max_overlap=1.0):
    import imageio_ffmpeg
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(clips=[], accuracy_metrics=None, threshold=threshold, margin=margin,
                  method=f'CLIP-ReIdent galleries; {comparison} cosine; linear assignment',
                  min_spacing=min_spacing, max_overlap=max_overlap,
                  note='Original IDs preserved unless merged. Cached embeddings; no new model inference. '
                       'Whole-track lifetime overlap forbids merging. No ID-steal repair.')
    for clip in ('attack_clip', 'attack_clip2'):
        cache = cache_root / clip
        meta = json.loads((cache / 'metadata.json').read_text())
        if meta['status'] != 'complete':
            raise ValueError('Incomplete cached inference')
        source = Path(meta['config']['run'])
        source_frames = source / 'frames.jsonl'
        video = Path(meta['video'])
        if sha256(source_frames) != meta['source_frames_sha256'] or sha256(video) != meta['video_sha256']:
            raise ValueError('Cached inference source changed')
        frames = [json.loads(line) for line in source_frames.read_text().splitlines()]
        segments = json.loads((cache / 'tracklets.json').read_text())
        folder = output / clip
        (folder / 'gallery').mkdir(parents=True)
        with np.load(cache / 'embeddings.npz', allow_pickle=False) as values:
            embeddings, segment_ids = values['embeddings'], values['segment_ids']
        if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
            raise ValueError('Invalid normalized embeddings')
        tracks, galleries, gallery_records = build_galleries(frames, segments, embeddings, segment_ids, cache, folder,
                                                            min_spacing=min_spacing, max_overlap=max_overlap)
        mapping, candidates = associate_galleries(tracks, galleries, threshold, margin, comparison=comparison)
        linked_frames, chart, seen_raw, seen_linked = [], [], set(), set()
        for frame in frames:
            rows = [dict(row, player_id=mapping[row['track_id']], identity_status='hypothesis') for row in frame['tracks']]
            linked_frames.append(dict(frame, tracks=rows))
            seen_raw.update(r['track_id'] for r in rows)
            seen_linked.update(r['player_id'] for r in rows)
            chart.append(dict(frame=frame['frame_idx'], before=len(seen_raw), after=len(seen_linked)))
        assert len(seen_linked) <= len(seen_raw)
        assert all(mapping[t] in tracks for t in tracks)
        for i, a in enumerate(tracks):
            for b in list(tracks)[i + 1:]:
                if mapping[a] == mapping[b]:
                    assert tracks[a]['end'] < tracks[b]['start'] or tracks[b]['end'] < tracks[a]['start']
        with (folder / 'frames.jsonl').open('w', encoding='utf-8') as stream:
            for frame in linked_frames:
                stream.write(json.dumps(frame) + '\n')
        write_json(folder / 'mapping.json', mapping)
        write_json(folder / 'candidates.json', candidates)
        write_json(folder / 'galleries.json', gallery_records)
        # Original footage with no baked labels: the browser draws one selected
        # identity, or short IDs on demand, on a synchronized canvas.
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(video),
                        '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(folder / 'clean.mp4')], check=True)
        cap = cv2.VideoCapture(str(folder / 'clean.mp4'))
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        cap.release()
        if decoded != len(frames):
            raise RuntimeError('Incomplete clean review video')
        payload = dict(clip=clip, fps=meta['source_metadata']['source']['fps'],
                       comparison=comparison, min_spacing=min_spacing, max_overlap=max_overlap,
                       before=len(seen_raw), after=len(seen_linked), frames=linked_frames,
                       mapping=mapping, tracks=tracks, galleries=gallery_records,
                       candidates=candidates, chart=chart)
        write_json(folder / 'review-data.json', payload)
        provenance = dict(source_metadata=meta, source_frames_sha256=sha256(source_frames),
                          embeddings_sha256=sha256(cache / 'embeddings.npz'),
                          scripts_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('gallery*.py')},
                          review_template_sha256=sha256(Path(__file__).with_name('gallery_review.html')),
                          versions={p: importlib.metadata.version(p) for p in
                                    ('numpy', 'scipy', 'opencv-python', 'imageio-ffmpeg')},
                          threshold=threshold, margin=margin, max_gap=150, gallery_size=8, min_gallery=3,
                          comparison=comparison, min_spacing=min_spacing, max_overlap=max_overlap,
                          source_buffer=meta['source_metadata']['tracker_config']['track_buffer'],
                          status='complete', source_ids=len(seen_raw), resulting_ids=len(seen_linked),
                          proposed_merges=sum(c['accepted'] for c in candidates),
                          inference_rerun=False, accuracy_metrics=None)
        write_json(folder / 'metadata.json', provenance)
        report['clips'].append(dict(clip=clip, before=len(seen_raw), after=len(seen_linked),
                                   merges=[c for c in candidates if c['accepted']]))
    shutil.copyfile(Path(__file__).with_name('gallery_review.html'), output / 'index.html')
    report['seconds'] = time.perf_counter() - started
    write_json(output / 'summary.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, default=Path('outputs/clip-reident-paper-v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.98)
    parser.add_argument('--margin', type=float, default=0.003)
    parser.add_argument('--comparison', choices=['nearest','one_to_one'], default='nearest')
    parser.add_argument('--min-spacing', type=int, default=0)
    parser.add_argument('--max-overlap', type=float, default=1.0)
    args = parser.parse_args()
    if args.min_spacing < 0 or not 0 <= args.max_overlap <= 1:
        parser.error('min-spacing must be nonnegative and max-overlap in [0,1]')
    run(args.cache_root, args.output, args.threshold, args.margin,
        args.comparison, args.min_spacing, args.max_overlap)


if __name__ == '__main__':
    main()
