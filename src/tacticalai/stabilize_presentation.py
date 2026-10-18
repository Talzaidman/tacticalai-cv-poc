"""Stabilize cached court estimates and player projections; no neural inference."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .calibration import project_track, transform
from .challenge_presentation import CLIPS, COURT
from .export import sha256, write_json
from .temporal_projection import smooth_cameras, smooth_players, roughness, image_probes


def scene_cuts(video):
    cap = cv2.VideoCapture(str(video))
    cuts, previous, previous_histogram = [], None, None
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            image = cv2.resize(image, (160, 90))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
            cv2.normalize(histogram, histogram)
            cut = previous is None or (cv2.compareHist(histogram, previous_histogram, cv2.HISTCMP_BHATTACHARYYA) > .55
                                       and np.mean(cv2.absdiff(image, previous)) > 35)
            cuts.append(bool(cut))
            previous, previous_histogram = image, histogram
    finally:
        cap.release()
    return np.array(cuts)


def player_roughness(before, after):
    lookup = [{r['track_id']: r for r in f['tracks'] if r['role'] == 'player'} for f in before]
    lookup_after = [{r['track_id']: r for r in f['tracks'] if r['role'] == 'player'} for f in after]
    original, smoothed = [], []
    for index in range(1, len(before)-1):
        ids = set(lookup[index-1]) & set(lookup[index]) & set(lookup[index+1])
        for track_id in ids:
            a = [lookup[i][track_id]['court_xy_m'] for i in range(index-1, index+2)]
            b = [lookup_after[i][track_id]['court_xy_m'] for i in range(index-1, index+2)]
            if all(p is not None for p in a+b):
                original.append(float(np.linalg.norm(np.array(a[2])-2*np.array(a[1])+a[0])))
                smoothed.append(float(np.linalg.norm(np.array(b[2])-2*np.array(b[1])+b[0])))
    def stats(values):
        return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), samples=len(values)) if values else None
    return dict(raw=stats(original), smoothed=stats(smoothed), units='m per frame second difference; same observed triples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/challenge-player-presentation-v1'))
    parser.add_argument('--output', type=Path, default=Path('outputs/challenge-player-smoothed-v1'))
    args = parser.parse_args()
    source_manifest = json.loads((args.source/'experiment.json').read_text())
    if source_manifest.get('presentation_status') != 'complete':
        raise ValueError('Original presentation is incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(source_manifest)
    manifest.pop('presentation_status', None)
    manifest['source_presentation'] = str(args.source.resolve())
    manifest['source_presentation_sha256'] = sha256(args.source/'experiment.json')
    manifest['original_clips'] = copy.deepcopy(source_manifest['clips'])
    import os
    manifest['comparison_relative_path'] = os.path.relpath(args.source, args.output).replace('\\', '/')
    manifest['temporal_smoothing'] = dict(method='robust constant-velocity Kalman + offline RTS',
        camera_measurement_sigma_m=.25, camera_acceleration_sigma_m_s2=6.,
        player_measurement_sigma_m=.14, player_acceleration_sigma_m_s2=12.,
        camera_max_estimation_gap_frames=5, outlier_local_radius_frames=4, outlier_floor_m=.8,
        outlier_scatter_multiplier=4, player_reset_step_m='.6 + 12/fps',
        camera_representation='court positions of nine fixed lower-image probes; refit homography',
        camera_probe_x_fraction=[.2,.5,.8], camera_probe_y_fraction=[.62,.75,.88],
        cuts='HSV Bhattacharyya > .55 AND thumbnail mean absolute difference >35',
        offline_future_frames_used=True, no_missing_player_inference=True)
    manifest['temporal_propagation'] = True
    source_folder = args.output/'run_source'
    source_folder.mkdir()
    for name in ('temporal_projection.py', 'stabilize_presentation.py', 'challenge_presentation.py', 'calibration.py', 'minimap.py', 'court.py'):
        shutil.copyfile(Path(__file__).with_name(name), source_folder/name)
    manifest['smoothing_source_sha256'] = {p.name:sha256(p) for p in source_folder.iterdir()}
    manifest['smoothing_metrics'] = {}
    for clip in CLIPS:
        original = args.source/clip
        target = args.output/clip
        target.mkdir()
        report = json.loads((original/'metadata.json').read_text())
        calibrations = [json.loads(l) for l in (original/'calibration.jsonl').read_text().splitlines()]
        raw_frames = [json.loads(l) for l in (original/'frames.jsonl').read_text().splitlines()]
        cuts = scene_cuts(Path('data/raw')/(clip+'.mp4'))
        if len(cuts) != len(calibrations):
            raise ValueError('Scene-cut/video length mismatch')
        matrices = [np.array(r['refined_image_to_court_h']) if r['refined_image_to_court_h'] is not None else None for r in calibrations]
        size = (report['video']['width'], report['video']['height'])
        smoothed, qualities, measurements, rejected = smooth_cameras(matrices, size, report['video']['fps'], cuts)
        frames = []
        for index, (raw, matrix, quality) in enumerate(zip(raw_frames, smoothed, qualities)):
            rows = []
            for source_row in raw['tracks']:
                row = project_track(source_row, matrix, quality, COURT, size)
                row['court_xy_raw_m'] = source_row['court_xy_m']
                row['court_xy_camera_smoothed_m'] = row['court_xy_m']
                row['camera_temporal_method'] = quality['method']
                if row['court_xy_m'] is not None:
                    row['position_source'] = ('estimated_calibration_floor_projection' if quality['method']=='kalman_rts_camera_gap_estimate'
                                              else 'temporally_smoothed_floor_projection')
                rows.append(row)
            frames.append(dict(frame_idx=index, timestamp_s=raw['timestamp_s'], tracks=rows,
                               image_to_court_h=matrix.tolist() if matrix is not None else None, calibration=quality))
        smooth_players(frames, report['video']['fps'], cuts)
        with (target/'prepared_frames.jsonl').open('x', encoding='utf8') as output, (target/'calibration.jsonl').open('x', encoding='utf8') as calibration_output:
            for index, (frame, calibration) in enumerate(zip(frames, calibrations)):
                output.write(json.dumps(frame, allow_nan=False)+'\n')
                calibration = dict(calibration, unsmoothed_refined_image_to_court_h=calibration['refined_image_to_court_h'],
                                   refined_image_to_court_h=frame['image_to_court_h'], temporal_quality=frame['calibration'])
                calibration_output.write(json.dumps(calibration, allow_nan=False)+'\n')
        for name in ('tracking_metadata.json', 'roles.json'):
            shutil.copyfile(original/name, target/name)
        shutil.copyfile(original/'calibration.jsonl', target/'source_calibration.jsonl')
        report.pop('rendering', None)
        report.pop('counts', None)
        report.pop('frame_stats', None)
        report.update(calibration_sha256=sha256(target/'calibration.jsonl'), prepared_frames_sha256=sha256(target/'prepared_frames.jsonl'),
                      source_calibration_sha256=sha256(original/'calibration.jsonl'),
                      calibrated_frames=sum(m is not None for m in smoothed), source_calibrated_frames=sum(m is not None for m in matrices))
        write_json(target/'metadata.json', report)
        after = np.array([transform(image_probes(size), m).ravel() if m is not None else np.full(18, np.nan) for m in smoothed])
        # Equal camera evidence support: do not count filled gaps as improved measurements.
        before = measurements.copy()
        support = np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
        before[~support] = np.nan
        after[~support] = np.nan
        metrics = dict(camera_raw=roughness(before), camera_smoothed=roughness(after),
                       camera_outliers_rejected=np.flatnonzero(rejected).tolist(),
                       estimated_camera_frames=[i for i, q in enumerate(qualities) if q['method']=='kalman_rts_camera_gap_estimate'],
                       unavailable_camera_frames=[i for i, m in enumerate(smoothed) if m is None],
                       cut_frames=np.flatnonzero(cuts).tolist(), player_roughness=player_roughness(raw_frames, frames),
                       note='Second-difference roughness is a jitter proxy, not measured accuracy or true player acceleration.')
        manifest['smoothing_metrics'][clip] = metrics
        manifest['clips'][clip] = report
        print(clip, json.dumps(metrics), flush=True)
    write_json(args.output/'experiment.json', manifest)


if __name__ == '__main__':
    main()
