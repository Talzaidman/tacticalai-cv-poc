"""Render automatic results only, using immutable identity/camera/mask caches.

Manual role files, fused role-dependent positions and human review judgments
are never read. The source silhouette cache supplies mask geometry only.
"""
import argparse
import copy
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from .calibration import project_track
from .challenge_presentation import CLIPS, COURT
from .court import floor_lines
from .export import sha256, write_json
from .identity_presentation import person_color, person_label
from .silhouette_presentation import render
from .temporal_projection import smooth_players


MASK_FIELDS = ('silhouette_xy', 'silhouette_confidence', 'silhouette_box_iou')


def automatic_frames(identities, cameras, masks, info):
    if not len(identities) == len(cameras) == len(masks) == info['frames']:
        raise ValueError('Frame count mismatch')
    frames, cuts, previous_shot = [], [], None
    for index, (identity, camera, mask_frame) in enumerate(zip(identities, cameras, masks)):
        for record in (identity, camera, mask_frame):
            if record['frame_idx'] != index or abs(record['timestamp_s']-index/info['fps']) > 1e-6:
                raise ValueError('Frame/timestamp mismatch')
        frame = copy.deepcopy(identity)
        quality = copy.deepcopy(camera['temporal_quality'])
        matrix = camera['refined_image_to_court_h']
        matrix = np.asarray(matrix) if matrix is not None else None
        lookup = {r['track_id']: r for r in mask_frame['tracks']}
        if set(lookup) != {r['track_id'] for r in identity['tracks']}:
            raise ValueError('Mask/identity observations differ')
        rows = []
        for source in frame['tracks']:
            row = dict(source, role=None)  # No role classifier is implemented.
            cached = lookup[row['track_id']]
            if cached['bbox_xyxy'] != row['bbox_xyxy']:
                raise ValueError('Mask refers to different track geometry')
            row.update({k: cached.get(k) for k in MASK_FIELDS})
            row = project_track(row, matrix, quality, COURT, (info['width'], info['height']))
            row['court_xy_camera_smoothed_m'] = row['court_xy_m']
            if row['court_xy_m'] is not None:
                row['position_source'] = ('estimated_calibration_floor_projection'
                    if 'gap_estimate' in quality['method'] else 'temporally_smoothed_floor_projection')
            row['display_label'] = person_label(row['player_id'])
            row['display_color_hex'] = '#' + ''.join(f'{c:02x}' for c in person_color(row['player_id'])[::-1])
            rows.append(row)
        frame.update(tracks=rows, calibration=quality,
                     image_to_court_h=matrix.tolist() if matrix is not None else None)
        shot = quality.get('shot')
        cuts.append(index == 0 or shot != previous_shot)
        previous_shot = shot
        frames.append(frame)
    # Apply the same automatic motion filter to every observed person, without
    # role boundaries. Unknown identities and missing observations reset it.
    smooth_players(frames, info['fps'], np.asarray(cuts),
                   identity_field='identity_segment_id', role_filter=None)
    for frame in frames:
        for row in frame['tracks']:
            row['shown_on_minimap'] = row['court_xy_m'] is not None and row['court_region'] == 'inside'
    return frames


def run(masks_root, identities_root, cameras_root, output):
    import imageio_ffmpeg
    output.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((masks_root/'experiment.json').read_text())
    if source_manifest['status'] != 'complete':
        raise ValueError('Incomplete mask cache')
    settings = dict(roles_used=False, manual_review_labels_used=False,
        map_filter='all observed persons with valid inside-court projections',
        position_smoothing='all persons; identity/gap/cut boundaries; no role filtering',
        court_assumption=COURT, offline_future_frames_used=True,
        identity_thresholds_developed_on_these_clips=True,
        independent_accuracy_metrics=None)
    manifest = dict(status='running', clips={}, automatic_flow=settings,
                    versions={p: importlib.metadata.version(p) for p in ('numpy','scipy','opencv-python','imageio-ffmpeg')})
    write_json(output/'experiment.json', manifest)
    started = time.perf_counter()
    lines = floor_lines(COURT)
    try:
        for clip in CLIPS:
            cache = masks_root/clip
            cached = json.loads((cache/'review-data.json').read_text())
            cache_meta = json.loads((cache/'metadata.json').read_text())
            if sha256(cache/'review-data.json') != cache_meta['artifacts_sha256']['review-data.json']:
                raise ValueError('Mask cache changed')
            identities_path = identities_root/clip/'frames.jsonl'
            camera_path = cameras_root/clip/'calibration.jsonl'
            identity_meta = json.loads(identities_path.with_name('metadata.json').read_text())
            camera_meta = json.loads(camera_path.with_name('metadata.json').read_text())
            if identity_meta['status'] != 'complete' or camera_meta['status'] != 'complete':
                raise ValueError('Incomplete automatic inputs')
            if sha256(identities_path) != identity_meta['artifacts_sha256']['frames.jsonl']:
                raise ValueError('Identity input changed')
            if sha256(camera_path) != camera_meta['calibration_sha256']:
                raise ValueError('Camera input changed')
            if identity_meta['source_frames_sha256'] != camera_meta['tracking_frames_sha256']:
                raise ValueError('Identity/camera tracking source mismatch')
            video = Path('data/raw')/(clip+'.mp4')
            info = cached['video']
            video_hash = sha256(video)
            if not video_hash == info['sha256'] == camera_meta['video']['sha256'] == identity_meta['source']['video_sha256']:
                raise ValueError('Video mismatch')
            frames = automatic_frames([json.loads(l) for l in identities_path.read_text().splitlines()],
                [json.loads(l) for l in camera_path.read_text().splitlines()], cached['frames'], info)
            target = output/clip; target.mkdir()
            shutil.copyfile(cache/'clean.mp4', target/'clean.mp4')
            cap = cv2.VideoCapture(str(video))
            process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-v','error','-y',
                '-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(info['fps']),'-i','pipe:0',
                '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
                str(target/'tracking-minimap.mp4')], stdin=subprocess.PIPE)
            try:
                for frame in frames:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Incomplete source decode')
                    rendered = render(image, frame, lines)
                    process.stdin.write(rendered.tobytes())
                    if frame['frame_idx'] in (0,55,131):
                        cv2.imwrite(str(target/f"frame-{frame['frame_idx']:04d}.jpg"), rendered)
                if cap.read()[0]:
                    raise ValueError('Unexpected source frames')
                process.stdin.close()
                if process.wait() != 0:
                    raise ValueError('Encode failed')
            finally:
                cap.release()
                if process.poll() is None:
                    process.kill(); process.wait()
            check = cv2.VideoCapture(str(target/'tracking-minimap.mp4')); decoded = 0
            while check.read()[0]:
                decoded += 1
            check.release()
            if decoded != len(frames):
                raise ValueError('Output decode mismatch')
            summary = dict(raw_ids=len({r['track_id'] for f in frames for r in f['tracks']}),
                person_hypotheses=len({r['player_id'] for f in frames for r in f['tracks'] if r['player_id'] is not None}),
                unknown_identity_rows=sum(r['player_id'] is None for f in frames for r in f['tracks']),
                source_rows=sum(len(f['tracks']) for f in frames), rendered_frames=decoded)
            segmentation = {k:v for k,v in cached['silhouettes'].items() if k != 'role_source'}
            segmentation.update(role_source=None, tracking_identity_projection_rerun=None,
                                mask_inference_rerun=False, positions_recomputed_without_roles=True)
            # Retain automatic matching evidence, without human evaluation labels.
            evidence = {name:json.loads((identities_root/clip/f'{name}.json').read_text())
                        for name in ('events','candidates','mapping')}
            write_json(target/'identity-evidence.json', evidence)
            write_json(target/'review-data.json', dict(clip=clip, video=info, frames=frames,
                summary=summary, court_lines=[l.tolist() for l in lines], silhouettes=segmentation,
                automatic_flow=settings))
            with (target/'frames.jsonl').open('w', encoding='utf8') as stream:
                for frame in frames:
                    stream.write(json.dumps(frame, allow_nan=False)+'\n')
            columns = ['frame_idx','timestamp_s','track_id','player_id','identity_status',
                       'court_x_m','court_y_m','position_source','shown_on_minimap']
            with (target/'positions.csv').open('w',newline='',encoding='utf8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
                for frame in frames:
                    for row in frame['tracks']:
                        xy = row['court_xy_m'] or [None,None]
                        record = {k:row.get(k) for k in columns}
                        record.update(frame_idx=frame['frame_idx'], timestamp_s=frame['timestamp_s'],
                                      court_x_m=xy[0], court_y_m=xy[1])
                        writer.writerow(record)
            write_json(target/'metadata.json', dict(status='complete', automatic_flow=settings,
                source_inputs_sha256=dict(identities=sha256(identities_path), cameras=sha256(camera_path),
                                          mask_cache=sha256(cache/'review-data.json'), video=video_hash),
                source_automatic_metadata=dict(identities=str(identities_path.with_name('metadata.json').resolve()),
                                               cameras=str(camera_path.with_name('metadata.json').resolve())),
                silhouettes=segmentation, summary=summary,
                artifacts_sha256={n:sha256(target/n) for n in ('frames.jsonl','positions.csv',
                    'review-data.json','tracking-minimap.mp4','clean.mp4','identity-evidence.json')}))
            manifest['clips'][clip] = dict(summary=summary)
            print(clip, summary, flush=True)
        snapshots = output/'run_source'; snapshots.mkdir()
        for name in ('automatic_presentation.py','automatic_presentation.html','silhouette_presentation.py',
                     'temporal_projection.py','calibration.py','court.py'):
            shutil.copyfile(Path(__file__).with_name(name), snapshots/name)
        shutil.copyfile(Path(__file__).with_name('automatic_presentation.html'), output/'index.html')
        manifest.update(status='complete', processing_s=time.perf_counter()-started,
                        source_sha256={p.name:sha256(p) for p in snapshots.iterdir()})
        write_json(output/'experiment.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error)); write_json(output/'experiment.json',manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masks',type=Path,default=Path('outputs/identity-court-silhouettes-v1'))
    parser.add_argument('--identities',type=Path,default=Path('outputs/temporal-reid-final'))
    parser.add_argument('--cameras',type=Path,default=Path('outputs/challenge-player-smoothed-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.masks,args.identities,args.cameras,args.output)


if __name__ == '__main__':
    main()
