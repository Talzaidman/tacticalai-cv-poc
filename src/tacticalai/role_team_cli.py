"""Enrich saved automatic person observations; retain geometry and identity."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import cv2
import numpy as np

from .export import sha256, write_json
from .role_team import ClipRoles, classify_identities, identity_key, select_samples, torso_color
from .silhouette_presentation import render


CLIPS = ('attack_clip', 'attack_clip2')
FIELDS = ['clip', 'frame_idx', 'timestamp_s', 'track_id', 'person_id', 'identity_segment_id',
          'identity_status', 'x1', 'y1', 'x2', 'y2', 'det_confidence', 'role', 'role_reason',
          'role_margin', 'role_agreement', 'role_samples', 'team_id', 'team_reason',
          'team_margin_lab', 'team_agreement', 'team_samples', 'jersey_number',
          'court_x_m', 'court_y_m', 'camera_smoothed_x_m', 'camera_smoothed_y_m',
          'foot_x_px', 'foot_y_px', 'foot_contact_visibility', 'position_source',
          'court_region', 'calibration_method', 'projection_quality_json',
          'role_scores_json', 'team_distances_lab_json', 'classification_evidence_id']


def verify(source, name, metadata):
    path = source / name
    if sha256(path) != metadata['artifacts_sha256'][name]:
        raise ValueError(f'Source changed: {path}')
    return path


def sample_clip(video, frames, config, model, target, info):
    selected = select_samples(frames, config, info['width'], info['height'])
    scheduled = defaultdict(list)
    for key, items in selected.items():
        for frame_idx, row_idx in items:
            scheduled[frame_idx].append((key, row_idx))
    samples = defaultdict(list)
    batch, pending = [], []
    inference_s = 0.
    def flush():
        nonlocal inference_s
        if not batch:
            return
        start = time.perf_counter()
        scores = model.score(batch)
        inference_s += time.perf_counter()-start
        for (key, record), score in zip(pending, scores):
            record['role_scores'] = score
            samples[key].append(record)
        batch.clear(); pending.clear()
    cap = cv2.VideoCapture(str(video))
    count = 0
    try:
        for frame in frames:
            ok, image = cap.read()
            if not ok:
                raise ValueError('Source decode incomplete')
            count += 1
            for key, row_idx in scheduled[frame['frame_idx']]:
                row = frame['tracks'][row_idx]
                x1, y1, x2, y2 = np.rint(row['bbox_xyxy']).astype(int)
                crop = image[y1:y2, x1:x2].copy()
                name = f"samples/{key}-{frame['frame_idx']:04d}.jpg"
                if not cv2.imwrite(str(target/name), crop):
                    raise ValueError('Could not save evidence crop')
                record = dict(frame_idx=frame['frame_idx'], track_id=row['track_id'],
                    bbox_xyxy=row['bbox_xyxy'], crop=name, crop_sha256=sha256(target/name),
                    torso_lab=torso_color(image, row, config))
                batch.append(crop); pending.append((key, record))
                if len(batch) >= 8:
                    flush()
            if frame['frame_idx'] % 50 == 0:
                print(f"{video.stem}: sampling frame {frame['frame_idx']}", flush=True)
        flush()
        if cap.read()[0]:
            raise ValueError('Source has extra frames')
    finally:
        cap.release()
    return dict(samples), dict(decoded_frames=count, sampled_crops=sum(map(len, samples.values())),
                               role_encoding_s=inference_s)


def enrich(frames, decisions):
    result = copy.deepcopy(frames)
    for frame in result:
        # Detections remain anonymous and are not assigned a role from proximity.
        for row in frame.get('detections', []):
            row.update(role=None, team_id=None)
        for row in frame['tracks']:
            key = identity_key(row)
            decision = decisions.get(key)
            if decision is None:
                decision = dict(role=None, team_id=None, role_reason='unknown_identity_or_no_usable_samples',
                    team_reason='role_not_player', role_scores=None, role_margin=None,
                    role_agreement=None, role_samples=0, team_samples=0,
                    team_distances_lab=None, team_margin_lab=None, team_agreement=None, torso_lab=None)
            row.update(copy.deepcopy(decision), classification_evidence_id=key,
                       classification_source='automatic_clip_roles_and_torso_color',
                       classification_temporal_scope='whole_clip_person_hypothesis',
                       jersey_number=None)
            row['display_caption'] = f"{row['display_label']} / T{row['track_id']} | {row['role'] or 'unknown'}"
            if row['team_id']:
                row['display_caption'] += f" {row['team_id']}"
    return result


def flat_record(clip, frame, row):
    record = {k: row.get(k) for k in FIELDS}
    record.update(clip=clip, frame_idx=frame['frame_idx'], timestamp_s=frame['timestamp_s'],
        person_id=row.get('player_id'), calibration_method=frame['calibration']['method'],
        role_scores_json=json.dumps(row.get('role_scores')), team_distances_lab_json=json.dumps(row.get('team_distances_lab')),
        projection_quality_json=json.dumps(row.get('court_projection_quality')))
    record.update(zip(('x1', 'y1', 'x2', 'y2'), row['bbox_xyxy']))
    record.update(zip(('court_x_m', 'court_y_m'), row.get('court_xy_m') or [None, None]))
    record.update(zip(('camera_smoothed_x_m', 'camera_smoothed_y_m'), row.get('court_xy_camera_smoothed_m') or [None, None]))
    record.update(zip(('foot_x_px', 'foot_y_px'), row.get('footpoint_image_xy') or [None, None]))
    return record


def person_summaries(frames, decisions, fps):
    grouped = defaultdict(list)
    for frame in frames:
        for row in frame['tracks']:
            key = identity_key(row)
            if key is not None:
                grouped[key].append((frame['frame_idx'], row))
    results = []
    for key, observations in grouped.items():
        indices = [i for i, _ in observations]
        gaps = [dict(first_missing_frame=a+1, last_missing_frame=b-1,
                     missing_frames=b-a-1, reason='unobserved; cause_unknown')
                for a, b in zip(indices, indices[1:]) if b-a > 1]
        results.append(dict(person_id=key, **{k: decisions.get(key, {}).get(k) for k in
            ('role', 'team_id', 'role_reason', 'team_reason', 'role_margin', 'role_agreement')},
            first_frame=indices[0], last_frame=indices[-1], first_timestamp_s=indices[0]/fps,
            last_timestamp_s=indices[-1]/fps, observed_frames=len(indices),
            observed_duration_s=len(indices)/fps, gaps=gaps,
            track_ids=sorted({r['track_id'] for _, r in observations}),
            projected_observations=sum(r['court_xy_m'] is not None for _, r in observations)))
    return results


def render_video(video, frames, lines, target, fps, color_mode='identity'):
    import imageio_ffmpeg
    # Reuse the identity-color renderer; captions carry the automatic roles/teams.
    cap = cv2.VideoCapture(str(video))
    process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', str(fps), '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(target/'annotated.mp4')], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            ok, image = cap.read()
            if not ok:
                raise ValueError('Incomplete rendering source')
            rendered = render(image, frame, lines, color_mode=color_mode)
            cv2.rectangle(rendered, (0, 681), (1280, 720), (14, 24, 32), -1)
            cv2.putText(rendered, 'Automatic role/team hypotheses | offline | unknown labels retained',
                        (18, 706), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1, cv2.LINE_AA)
            process.stdin.write(rendered.tobytes())
            if frame['frame_idx'] in (0, 55, 112):
                cv2.imwrite(str(target/f"frame-{frame['frame_idx']:04d}.jpg"), rendered)
        process.stdin.close()
        if process.wait() != 0:
            raise ValueError('Annotated-video encoding failed')
    finally:
        cap.release()
        if process.poll() is None:
            process.kill(); process.wait()
    check = cv2.VideoCapture(str(target/'annotated.mp4')); count = 0
    while check.read()[0]:
        count += 1
    check.release()
    if count != len(frames):
        raise ValueError('Annotated-video frame count mismatch')
    return count


def run(args):
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    if json.loads((args.source/'experiment.json').read_text())['status'] != 'complete':
        raise ValueError('Source baseline is incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', automatic_only=True, manual_labels_used=False,
        source=str(args.source.resolve()), source_manifest_sha256=sha256(args.source/'experiment.json'),
        configuration=config, configuration_sha256=sha256(args.config), device=args.device,
        cpu_threads=args.threads, platform=platform.platform(), python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in
                  ('torch', 'open-clip-torch', 'numpy', 'opencv-python', 'scipy', 'imageio-ffmpeg')},
        detector_tracker_identity_camera_segmentation_rerun=False,
        role_network_executed=True, offline_future_frames_used=True,
        full_pipeline_benchmark_s=None, independent_accuracy_metrics=None, clips={},
        limitations=[
            'Experimental generic CLIP roles; scores and margins are not probabilities.',
            'Role model, prompts and full-body preprocessing were revised after observing ambiguity on these clips; evaluation is a development spot check.',
            'Two uniform groups assumed. A/B are anonymous and local to each clip; no roster or fixed player count.',
            'Roles/teams aggregate the whole person hypothesis and inherit upstream identity mistakes.',
            'Known clip feedback was used for upstream identity and smoothing development.',
            'No manual role, team or identity corrections enter classification.',
            'Court dimensions assume 28 x 15 m. Bbox bottom center is an unverified floor point.',
            'Camera and position smoothing use future frames; this is offline processing.',
            'Ball, jersey identity, off-screen position and tactics remain unknown.',
            'Historical stage times overlap and use different hardware/caches; they must not be summed.',
            'No held-out-video accuracy or generalization claim.'
        ])
    write_json(args.output/'experiment.json', manifest)
    try:
        t = time.perf_counter()
        model = None
        if args.classification_cache:
            cache_manifest = json.loads((args.classification_cache/'experiment.json').read_text())
            if cache_manifest['status'] != 'complete' or cache_manifest['configuration'] != config:
                raise ValueError('Classification cache incomplete or configuration differs')
            checkpoint_sha256 = cache_manifest['role_model_sha256']
            manifest.update(role_network_executed=False,
                role_inference_source=str(args.classification_cache.resolve()),
                role_inference_source_manifest_sha256=sha256(args.classification_cache/'experiment.json'))
        else:
            model = ClipRoles(config, args.model_cache, args.device, args.threads)
            checkpoint_sha256 = model.checkpoint_sha256
        manifest.update(role_model_sha256=checkpoint_sha256,
                        role_model_setup_s=time.perf_counter()-t)
        all_flat = []
        for clip in CLIPS:
            start = time.perf_counter()
            source = args.source/clip
            source_meta = json.loads((source/'metadata.json').read_text())
            if source_meta['status'] != 'complete':
                raise ValueError('Incomplete source metadata')
            frames_path = verify(source, 'frames.jsonl', source_meta)
            frames = [json.loads(l) for l in frames_path.read_text().splitlines()]
            review = json.loads(verify(source, 'review-data.json', source_meta).read_text())
            info = review['video']
            video = args.videos/(clip+'.mp4')
            if sha256(video) != info['sha256'] or len(frames) != info['frames']:
                raise ValueError('Video or frame count differs from baseline')
            for i, f in enumerate(frames):
                if f['frame_idx'] != i or abs(f['timestamp_s']-i/info['fps']) > 1e-6:
                    raise ValueError('Unexpected frame/timestamp alignment')
            target = args.output/clip; target.mkdir(); (target/'samples').mkdir()
            if args.classification_cache:
                cache = args.classification_cache/clip
                cache_meta = json.loads((cache/'metadata.json').read_text())
                if cache_meta['source_frames_sha256'] != sha256(frames_path):
                    raise ValueError('Cached classification uses different source observations')
                cached_evidence = json.loads(verify(cache, 'classification-evidence.json', cache_meta).read_text())
                if cached_evidence['configuration'] != config or cached_evidence['role_model_sha256'] != checkpoint_sha256:
                    raise ValueError('Cached model or prompts differ')
                samples = cached_evidence['samples']
                expected = select_samples(frames, config, info['width'], info['height'])
                if set(samples) != set(expected):
                    raise ValueError('Cached identity samples differ')
                for key, values in samples.items():
                    if len(values) != len(expected[key]):
                        raise ValueError('Cached sample count differs')
                    for sample, (frame_idx, row_idx) in zip(values, expected[key]):
                        row = frames[frame_idx]['tracks'][row_idx]
                        if sample['frame_idx'] != frame_idx or sample['bbox_xyxy'] != row['bbox_xyxy']:
                            raise ValueError('Cached sample geometry differs')
                        source_crop = cache/sample['crop']
                        if sha256(source_crop) != sample['crop_sha256']:
                            raise ValueError('Cached evidence crop changed')
                        shutil.copyfile(source_crop, target/sample['crop'])
                timings = dict(decoded_frames=0, sampled_crops=sum(map(len, samples.values())),
                    role_encoding_s=0., cached_role_encoding_s=cache_meta['timings']['role_encoding_s'],
                    role_encoding_reused=True)
            else:
                samples, timings = sample_clip(video, frames, config, model, target, info)
            decisions, centers = classify_identities(samples, config)
            enriched = enrich(frames, decisions)
            evidence = dict(roles=decisions, samples=samples, team_centers_lab=centers,
                role_model_sha256=checkpoint_sha256, configuration=config,
                manual_labels_used=False, evidence_scores_are_probabilities=False)
            write_json(target/'classification-evidence.json', evidence)
            with (target/'frames.jsonl').open('w', encoding='utf8') as stream:
                for frame in enriched:
                    stream.write(json.dumps(frame, allow_nan=False)+'\n')
            flat = [flat_record(clip, f, r) for f in enriched for r in f['tracks']]
            all_flat.extend(flat)
            with (target/'observations.csv').open('w', newline='', encoding='utf8') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(flat)
            people = person_summaries(enriched, decisions, info['fps'])
            write_json(target/'persons.json', people)
            shutil.copyfile(verify(source, 'clean.mp4', source_meta), target/'clean.mp4')
            decoded = render_video(video, enriched, review['court_lines'], target, info['fps'])
            timings.update(new_clip_processing_s=time.perf_counter()-start, annotated_frames_decoded=decoded)
            counts = dict(observations=len(flat), person_hypotheses=len(people),
                unknown_identity_observations=sum(r['person_id'] is None for r in flat),
                role_observations=dict(Counter(r['role'] or 'unknown' for r in flat)),
                team_observations=dict(Counter(r['team_id'] or 'unknown' for r in flat)),
                projected_observations=sum(r['court_x_m'] is not None for r in flat),
                counts_are_predictions_not_ground_truth=True)
            historical = dict(
                tracking=dict(hardware=source_meta['source_tracking_metadata'].get('gpu'),
                    timing=source_meta['source_tracking_metadata']['timing'],
                    scope=source_meta['source_tracking_metadata']['timing_notes'],
                    tracker_config=source_meta['tracker_config'],
                    detector_config=source_meta['source_tracking_metadata']['config'],
                    versions=source_meta['source_tracking_metadata']['versions']),
                segmentation=dict(inference_s=source_meta['source_automatic_metadata']['silhouettes']['segmentation_inference_s'],
                                  hardware='CPU', scope='historical segmentation inference only; reused in this run'),
                other_stages=dict(benchmark_s=None, reason='No unified end-to-end execution in this run'))
            provenance = source_meta['source_automatic_metadata']['source_automatic_metadata']
            for stage in ('identities', 'cameras'):
                stage_path = Path(provenance[stage])
                if stage_path.exists():
                    stage_meta = json.loads(stage_path.read_text())
                    if stage == 'identities':
                        feature = stage_meta.get('source', {})
                        historical['identity_feature_extraction'] = dict(
                            metadata_sha256=sha256(stage_path),
                            seconds=feature.get('seconds'), device=feature.get('device'),
                            new_crops=feature.get('new_crops'), reused_crops=feature.get('reused_crops'),
                            scope='Historical temporal appearance extraction, including cache reuse; not complete identity pipeline')
                    else:
                        historical['camera_presentation_stage'] = dict(
                            metadata_sha256=sha256(stage_path), seconds=stage_meta.get('elapsed_s'),
                            scope='Historical camera/presentation stage elapsed field; includes its recorded scope, not an isolated calibration benchmark')
            web_frames = []
            for frame in enriched:
                web_rows = []
                for row in frame['tracks']:
                    web_row = dict(row)
                    if row.get('silhouette_xy'):
                        polygon = np.asarray(row['silhouette_xy'], dtype=np.float32)
                        web_row['silhouette_xy'] = cv2.approxPolyDP(polygon, .5, True).reshape(-1, 2).tolist()
                    web_rows.append(web_row)
                web_frames.append(dict(frame_idx=frame['frame_idx'], timestamp_s=frame['timestamp_s'],
                    calibration=frame['calibration'], tracks=web_rows))
            write_json(target/'review-data.json', dict(clip=clip, video=info, frames=web_frames,
                persons=people, counts=counts, court_lines=review['court_lines'], team_centers_lab=centers))
            shutil.copyfile(source/'metadata.json', target/'baseline-provenance.json')
            # Verify preserved measurements before declaring success.
            allowed = {'role', 'team_id', 'jersey_number'}
            for original, current in zip(frames, enriched):
                for before, after in zip(original['tracks'], current['tracks']):
                    if any(after[k] != v for k, v in before.items() if k not in allowed):
                        raise ValueError('Original identity or measurement changed')
            meta = dict(status='complete', video=info, source_frames_sha256=sha256(frames_path),
                source_metadata_sha256=sha256(source/'metadata.json'), timings=timings,
                historical_stages=historical, counts=counts, source_measurements_preserved=True,
                artifacts_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json', meta)
            manifest['clips'][clip] = dict(counts=counts, timings=timings, historical_stages=historical,
                metadata_sha256=sha256(target/'metadata.json'))
            write_json(args.output/'experiment.json', manifest)
            print(clip, counts, timings, flush=True)
        with (args.output/'observations.csv').open('w', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(all_flat)
        shutil.copyfile(args.config, args.output/'configuration.json')
        shutil.copyfile(Path(__file__).with_name('role_team_schema.json'), args.output/'schema.json')
        shutil.copyfile(Path(__file__).with_name('role_team_review.html'), args.output/'index.html')
        snapshots = args.output/'run_source'; snapshots.mkdir()
        for name in ('role_team.py', 'role_team_cli.py', 'role_team_review.html', 'role_team_schema.json',
                     'silhouette_presentation.py', 'identity_presentation.py', 'export.py'):
            shutil.copyfile(Path(__file__).with_name(name), snapshots/name)
        manifest.update(status='complete', new_run_total_s=time.perf_counter()-started,
            source_sha256={p.name: sha256(p) for p in snapshots.iterdir()})
        write_json(args.output/'experiment.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        write_json(args.output/'experiment.json', manifest)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/restored-identity-baseline-v1'))
    p.add_argument('--videos', type=Path, default=Path('data/raw'))
    p.add_argument('--config', type=Path, default=Path('configs/role_team.json'))
    p.add_argument('--model-cache', type=Path, default=Path('models/role-clip'))
    p.add_argument('--classification-cache', type=Path,
                   help='Reuse hash-verified sampled network scores with identical configuration; no role inference rerun')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    run(p.parse_args())


if __name__ == '__main__':
    main()
