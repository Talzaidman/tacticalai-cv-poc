"""Split appearance-inconsistent tracks, then associate person segment galleries."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from .enrich_galleries_cli import select_samples
from .export import sha256, write_json
from .gallery_linking import associate_galleries
from .temporal_identity import TemporalConfig, config_dict, split_tracks


def segment_galleries(segments, observations, features, maximum=8, spacing=5):
    galleries, records = {}, {}
    for sid, segment in segments.items():
        pool = []
        for i in segment['feature_indices']:
            o = observations[i]
            b = o['bbox_xyxy']
            if o['edge'] or o['overlap'] > .6 or b[2]-b[0] < 12 or b[3]-b[1] < 35:
                continue
            pool.append(dict(o, index=i, quality=o['quality'] * np.sqrt((b[2]-b[0])*(b[3]-b[1]))))
        selected = select_samples(pool, maximum, spacing)
        records[sid] = selected
        if len(selected) >= 3:
            galleries[sid] = features[[s['index'] for s in selected]]
    return galleries, records


def analyze(frames, observations, features, config=TemporalConfig(), threshold=.98, margin=.001):
    segments, events, diagnostics, lookup = split_tracks(frames, observations, features, config)
    galleries, samples = segment_galleries(segments, observations, features)
    cannot_link = [(a, b) for a in segments for b in segments if a < b and
                   segments[a]['source_track_id'] == segments[b]['source_track_id']]
    mapping, candidates = associate_galleries(segments, galleries, threshold, margin,
                                              comparison='one_to_one', cannot_link=cannot_link)
    # Stable display names follow the earliest source label. Suffixes distinguish
    # people separated within a raw label; they are not new tracker observations.
    for frame in frames:
        for row in frame['tracks']:
            sid = lookup.get((frame['frame_idx'], row['track_id']))
            row.update(identity_segment_id=sid, player_id=mapping.get(sid),
                       identity_status='automatic_hypothesis' if sid else 'unknown_transition')
    return dict(frames=frames, segments=segments, events=events, diagnostics=diagnostics,
                mapping=mapping, candidates=candidates, galleries=samples)


def run(cache_root, output, clips, threshold=.98, margin=.001):
    import imageio_ffmpeg
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    config = TemporalConfig()
    report = dict(status='running', clips=[], accuracy_metrics=None, development_experiment=True,
                  note='Automatic hypotheses; no user review labels read by inference. One change per raw ID maximum.')
    write_json(output / 'summary.json', report)
    for clip in clips:
        cache = cache_root / clip
        meta = json.loads((cache / 'metadata.json').read_text())
        if meta['status'] != 'complete':
            raise ValueError('Incomplete feature cache')
        source = Path(meta['source']['config']['run']) / 'frames.jsonl'
        video = Path(meta['source']['video'])
        if sha256(source) != meta['source_frames_sha256'] or sha256(video) != meta['video_sha256']:
            raise ValueError('Source hash mismatch')
        frames = [json.loads(line) for line in source.read_text().splitlines()]
        observations = json.loads((cache / 'observations.json').read_text())
        with np.load(cache / 'features.npz', allow_pickle=False) as data:
            features = data['features']
        result = analyze(frames, observations, features, config, threshold, margin)
        folder = output / clip
        folder.mkdir()
        # Original crop names are stable and give frame-level evidence.
        shutil.copytree(cache / 'crops', folder / 'crops')
        for name in ('segments', 'events', 'diagnostics', 'mapping', 'candidates', 'galleries'):
            write_json(folder / f'{name}.json', result[name])
        with (folder / 'frames.jsonl').open('w', encoding='utf-8') as stream:
            for frame in frames:
                stream.write(json.dumps(frame) + '\n')
        raw_ids = {r['track_id'] for f in frames for r in f['tracks']}
        player_ids = {r['player_id'] for f in frames for r in f['tracks'] if r['player_id'] is not None}
        unknown = sum(r['player_id'] is None for f in frames for r in f['tracks'])
        rows = sum(len(f['tracks']) for f in frames)
        for frame in frames:
            active = [r['player_id'] for r in frame['tracks'] if r['player_id'] is not None]
            if len(active) != len(set(active)):
                raise ValueError('Two simultaneous observations share an identity')
        # Verify every original observation and all original fields are preserved.
        original = [json.loads(line) for line in source.read_text().splitlines()]
        for before, after in zip(original, frames):
            assert len(before['tracks']) == len(after['tracks'])
            for a, b in zip(before['tracks'], after['tracks']):
                assert all(b[k] == v for k, v in a.items())
            assert all(after[k] == v for k, v in before.items() if k != 'tracks')
        clean = Path('outputs/gallery-improvements-v2/candidate') / clip / 'clean.mp4'
        clean_meta = clean.with_name('metadata.json')
        compatible = clean_meta.is_file() and json.loads(clean_meta.read_text()).get('source_metadata',{}).get('video_sha256') == meta['video_sha256']
        if clean.exists() and compatible:
            probe = cv2.VideoCapture(str(clean))
            reusable = probe.get(cv2.CAP_PROP_FRAME_COUNT) == len(frames)
            probe.release()
        else:
            reusable = False
        if reusable:
            shutil.copyfile(clean, folder / 'clean.mp4')
        else:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(video), '-an', '-c:v',
                            'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(folder / 'clean.mp4')], check=True)
        cap = cv2.VideoCapture(str(folder / 'clean.mp4')); decoded = 0
        while cap.read()[0]:
            decoded += 1
        cap.release()
        if decoded != len(frames):
            raise ValueError('Incomplete review video')
        summary = dict(clip=clip, raw_ids=len(raw_ids), segments=len(result['segments']), player_ids=len(player_ids),
                       unknown_rows=unknown, total_rows=rows, switches=len(result['events']),
                       links=[c for c in result['candidates'] if c['accepted']])
        report['clips'].append(summary)
        write_json(folder / 'review-data.json', dict(clip=clip, fps=meta['source']['source_metadata']['source']['fps'],
                   width=meta['source']['source_metadata']['source']['width'], height=meta['source']['source_metadata']['source']['height'],
                   **{k: result[k] for k in ('frames','segments','events','mapping','candidates','galleries')}, summary=summary))
        write_json(folder / 'metadata.json', dict(status='complete', source=meta, temporal_config=config_dict(config),
                   threshold=threshold, margin=margin, comparison='one_to_one', min_gallery=3, spacing=5, gallery_size=8,
                   source_frames_sha256=sha256(source), features_sha256=sha256(cache/'features.npz'),
                   observations_sha256=sha256(cache/'observations.json'), source_fields_preserved=True,
                   source_buffer=meta['source']['source_metadata']['tracker_config']['track_buffer'],
                   inference_review_labels_used=False, detector_tracker_rerun=False, accuracy_metrics=None,
                   scripts_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('temporal_identity.py'),Path(__file__).with_name('gallery_linking.py'),Path(__file__).with_name('enrich_galleries_cli.py')]},
                   template_sha256=sha256(Path(__file__).with_name('temporal_review.html')),
                   artifacts_sha256={name:sha256(folder/name) for name in ('frames.jsonl','mapping.json','clean.mp4')},
                   versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','opencv-python','imageio-ffmpeg')}))
        print(json.dumps(summary), flush=True)
    shutil.copyfile(Path(__file__).with_name('temporal_review.html'), output / 'index.html')
    report.update(status='complete',seconds=time.perf_counter() - started)
    write_json(output / 'summary.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', type=Path, default=Path('outputs/temporal-features-v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clips', nargs='+', default=['attack_clip','attack_clip2'], choices=['attack_clip','attack_clip2'])
    p.add_argument('--threshold', type=float, default=.98)
    p.add_argument('--margin', type=float, default=.001)
    a = p.parse_args()
    try:
        run(a.cache_root, a.output, a.clips, a.threshold, a.margin)
    except BaseException as error:
        summary = a.output / 'summary.json'
        if summary.is_file():
            report = json.loads(summary.read_text())
            if report.get('status') == 'running':
                report.update(status='failed',error=repr(error))
                write_json(summary,report)
        raise


if __name__ == '__main__':
    main()
