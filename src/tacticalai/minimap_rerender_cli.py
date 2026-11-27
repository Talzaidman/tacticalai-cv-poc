"""Re-render a completed automatic fusion cache with stable team-colored map dots.

This is a display-only offline pass. It preserves source detections, identities,
roles, teams, ball states and raw court coordinates; no neural inference runs.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

from .ball_cli import minimap_smoothing_summary, stabilize_minimap
from .ball_fusion_cli import encode, flat_frame
from .calibration_cli import video_info
from .export import sha256, write_json
from .pass_events import PassConfig, infer_pass_events
from .video_set import resolve_clips


def run(source, videos, clips, output, source_url=None):
    manifest_path = source/'experiment.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if (manifest.get('status') != 'complete' or
            manifest.get('automatic_only') is not True or
            manifest.get('manual_labels_used') is not False):
        raise ValueError('Source must be a completed explicitly automatic fusion run')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', automatic_only=True, manual_labels_used=False,
        neural_inference_rerun=False, source=str(source.resolve()), source_url=source_url,
        source_experiment_sha256=sha256(manifest_path), clips={},
        presentation=dict(player_colors='anonymous automatic team groups',
            team_palette_bgr={'A':[230,155,65], 'B':[70,175,245]},
            minimap_smoothing='robust Kalman/RTS over observed identity segments',
            raw_court_xy_preserved=True, gaps_filled=False,
            offline_future_frames_used=True),
        limitations=[
            'Smoothness is not court-position accuracy; calibration may be smoothly wrong.',
            'Team A/B are automatic anonymous clip-local groups, not roster identities.',
            'Missing player observations remain missing; no off-screen position is invented.',
            'Ball/controller evidence is inherited unchanged from the completed source run.',
            'Pass paths are endpoint interpolations, not reconstructed airborne trajectories.'])
    write_json(output/'experiment.json', report)
    started = time.perf_counter()
    try:
        for clip, video in resolve_clips(videos, clips):
            info = video_info(video)
            source_folder = source/clip
            source_meta_path = source_folder/'metadata.json'
            source_frames_path = source_folder/'frames.jsonl'
            metadata = json.loads(source_meta_path.read_text(encoding='utf8'))
            if metadata.get('status') != 'complete' or metadata.get('video') != info:
                raise ValueError('Source metadata does not match the local video')
            expected = metadata.get('artifacts_sha256', {}).get('frames.jsonl')
            if expected is None or sha256(source_frames_path) != expected:
                raise ValueError('Source frame cache hash differs from completed metadata')
            frames = [json.loads(line) for line in source_frames_path.read_text(encoding='utf8').splitlines()]
            if len(frames) != info['frames']:
                raise ValueError('Source frame count differs from the video')
            for index, frame in enumerate(frames):
                if frame['frame_idx'] != index or abs(frame['timestamp_s']-index/info['fps']) > 1e-6:
                    raise ValueError('Source frame timeline differs from the video')
            stabilize_minimap(frames, info['fps'])
            pass_audit = infer_pass_events(frames, info['fps'], PassConfig())
            diagnostics = minimap_smoothing_summary(frames)
            target = output/clip
            target.mkdir()
            with (target/'frames.jsonl').open('w', encoding='utf8') as stream:
                for frame in frames:
                    stream.write(json.dumps(frame, allow_nan=False)+'\n')
            write_json(target/'pass_events.json', pass_audit)
            flat = [flat_frame(frame) for frame in frames]
            with (target/'ball_possession.csv').open('w', newline='', encoding='utf8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
                writer.writeheader(); writer.writerows(flat)
            decoded = encode(video, frames, target, info, player_presentation='silhouettes')
            result = dict(status='complete', video=info,
                source=dict(frames=str(source_frames_path.resolve()),
                    frames_sha256=expected, metadata_sha256=sha256(source_meta_path)),
                minimap_smoothing=dict(method='robust Kalman/RTS', display_only=True,
                    raw_court_xy_preserved=True, gaps_filled=False,
                    offline_future_frames_used=True, diagnostics=diagnostics),
                ball_minimap_projection=dict(enabled=False,
                    reason='airborne ball depth is not observable from one broadcast view'),
                pass_visualization=dict(method='offline controller-change endpoint interpolation',
                    accepted_events=pass_audit['accepted_events'],
                    conservative_events=pass_audit['conservative_events'],
                    tentative_events=pass_audit['tentative_events'],
                    tentative_policy=pass_audit['tentative_policy'],
                    styles=dict(conservative='solid orange',tentative='dashed yellow'),
                    true_airborne_trajectory_known=False,
                    same_automatic_team_required=True,
                    audit_artifact='pass_events.json'),
                inherited_counts=metadata.get('counts'), annotated_frames_decoded=decoded,
                accuracy_metrics=None,
                artifacts_sha256={p.name:sha256(p) for p in target.iterdir() if p.is_file()})
            write_json(target/'metadata.json', result)
            report['clips'][clip] = result
            write_json(output/'experiment.json', report)
        snapshots = output/'run_source'; snapshots.mkdir()
        for name in ('minimap_rerender_cli.py', 'ball_fusion_cli.py', 'pass_events.py', 'ball_cli.py',
                     'silhouette_presentation.py', 'temporal_projection.py'):
            shutil.copyfile(Path(__file__).with_name(name), snapshots/name)
        (output/'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Stable team minimap</title>'
            '<body style="margin:20px;background:#111;color:#eee;font:16px Arial">'
            '<p>Automatic anonymous teams; offline display-only minimap smoothing and inferred passes. '
            'Pass lines connect controller endpoints and are not measured airborne trajectories.</p>'+''.join(
                f'<h2>{clip}</h2><video controls preload="metadata" style="width:min(100%,1100px)" '
                f'src="{clip}/annotated.mp4"></video>' for clip, _ in resolve_clips(videos, clips))+
            '</body>', encoding='utf8')
        report.update(status='complete', processing_s=time.perf_counter()-started,
                      source_sha256={p.name:sha256(p) for p in snapshots.iterdir()})
        write_json(output/'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        write_json(output/'experiment.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Completed automatic fusion result root')
    parser.add_argument('--videos', type=Path, default=Path('data/raw'))
    parser.add_argument('--clips', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-url', help='Optional Drive/source provenance URL')
    args = parser.parse_args()
    run(args.source, args.videos, args.clips, args.output, args.source_url)


if __name__ == '__main__':
    main()
