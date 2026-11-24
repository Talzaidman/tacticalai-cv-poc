"""Run gated SAM 3 basketball detection and export the common ball schema."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .calibration_cli import video_info
from .export import sha256, write_json
from .sam3_ball import Sam3SelectionConfig, native_track_timeline, select_dynamic_track
from .video_set import resolve_clips


PREFIX = 'SAM3-Native'


def read_frame_chunks(path, chunk_frames):
    if chunk_frames < 1:
        raise ValueError('SAM 3 chunk size must be positive')
    cap = cv2.VideoCapture(str(path)); frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frames.append(frame)
            if len(frames) == chunk_frames:
                yield frames
                frames = []
        if frames:
            yield frames
    finally:
        cap.release()


def infer(model, processor, frames, prompt, dtype, torch):
    options = dict(video=np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]),
                   inference_device='cuda', processing_device='cpu', video_storage_device='cpu')
    if 'dtype' in inspect.signature(processor.init_video_session).parameters:
        options['dtype'] = dtype
    session = processor.init_video_session(**options)
    session = processor.add_text_prompt(inference_session=session, text=prompt)
    detections = [[] for _ in frames]
    torch.cuda.synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        for output in model.propagate_in_video_iterator(inference_session=session):
            result = processor.postprocess_outputs(session, output)
            def array(value):
                return value.detach().float().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value, float)
            ids = array(result['object_ids']).reshape(-1)
            scores = array(result['scores']).reshape(-1)
            boxes = array(result['boxes']).reshape(-1, 4)
            for object_id, confidence, (x1, y1, x2, y2) in zip(ids, scores, boxes):
                detections[output.frame_idx].append(dict(x=float((x1+x2)/2), y=float((y1+y2)/2),
                    r=float(max(x2-x1, y2-y1)/2), conf=float(confidence), sam3_id=int(object_id)))
    torch.cuda.synchronize()
    return detections, time.perf_counter()-started


def render(video, path, tracked, info):
    import imageio_ffmpeg
    cap = cv2.VideoCapture(str(video))
    process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f"{info['width']}x{info['height']}",
        '-r', str(info['fps']), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    trail = []
    try:
        for index, point in enumerate(tracked):
            ok, frame = cap.read()
            if not ok: raise ValueError('Source video ended early while rendering SAM 3')
            if point['state'] == 'detected':
                center = (round(point['x']), round(point['y'])); trail.append(center); trail = trail[-25:]
                if len(trail) > 1: cv2.polylines(frame, [np.asarray(trail, np.int32)], False, (0, 220, 0), 2)
                cv2.circle(frame, center, 14, (0, 230, 0), 3)
            elif point['state'] == 'predicted':
                x, y = round(point['x']), round(point['y'])
                cv2.line(frame, (x-8, y-8), (x+8, y+8), (180, 180, 180), 2)
                cv2.line(frame, (x-8, y+8), (x+8, y-8), (180, 180, 180), 2)
            else:
                trail = []
            cv2.putText(frame, f'SAM3 ball | frame {index} | {point["state"]}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        if cap.read()[0]: raise ValueError('Source video has extra frames')
        process.stdin.close()
        if process.wait() != 0: raise RuntimeError('SAM 3 video encoding failed')
    finally:
        cap.release()
        if process.poll() is None: process.kill(); process.wait()


def run(args):
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    if not torch.cuda.is_available(): raise ValueError('SAM 3 requires a CUDA runtime for this pipeline')
    token = os.environ.get('HF_TOKEN')
    if not token: raise ValueError('HF_TOKEN is required; add it as a private Colab secret')
    clips = resolve_clips(args.videos, args.clips)
    args.output.mkdir(parents=True, exist_ok=False)
    selection = Sam3SelectionConfig(maximum_prediction_s=args.maximum_prediction_s)
    report = dict(status='running', automatic_only=True, manual_labels_used=False,
        model_id=args.model_id, prompt=args.prompt, prefix=PREFIX, offline_processing=True,
        selection=asdict(selection), clips={}, limitations=[
            'One small moving SAM native track is selected automatically; unusual camera or graphic motion can still mislead it.',
            'Long videos use bounded independent SAM 3 sessions; native identity and temporal prediction reset at every chunk boundary.',
            'Predicted gap positions are inferred and never establish holder evidence.',
            'Ball image coordinates are not projected through the floor homography.',
            'No independently annotated ball accuracy is available for this clip.'])
    write_json(args.output/'experiment.json', report)
    started = time.perf_counter()
    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        model = Sam3VideoModel.from_pretrained(args.model_id, token=token).to('cuda', dtype=dtype).eval()
        processor = Sam3VideoProcessor.from_pretrained(args.model_id, token=token)
        versions = {name: importlib.metadata.version(name) for name in
                    ('torch', 'transformers', 'numpy', 'opencv-python', 'imageio-ffmpeg')}
        for clip, video in clips:
            tick = time.perf_counter(); info = video_info(video)
            native, selected, tracked, chunk_diagnostics = [], [], [], []
            inference_s = 0.; frame_offset = 0; segment_offset = 0
            for chunk_index, frames in enumerate(read_frame_chunks(video, args.chunk_frames)):
                chunk_native, chunk_s = infer(model, processor, frames, args.prompt, dtype, torch)
                chunk_selected, diagnostics = select_dynamic_track(
                    chunk_native, info['width'], info['height'], selection)
                chunk_tracked = native_track_timeline(
                    chunk_selected, info['width'], info['height'], info['fps'], selection)
                for rows in chunk_native:
                    native.append([dict(row, sam3_chunk_index=chunk_index) for row in rows])
                selected.extend([[dict(row, sam3_chunk_index=chunk_index) for row in rows]
                                 for rows in chunk_selected])
                local_segments = [point['segment_id'] for point in chunk_tracked
                                  if point['segment_id'] is not None]
                for point in chunk_tracked:
                    point['frame_idx'] += frame_offset
                    point['timestamp_s'] = point['frame_idx']/info['fps']
                    if point['segment_id'] is not None:
                        point['segment_id'] += segment_offset
                    tracked.append(point)
                chunk_diagnostics.append(dict(chunk_index=chunk_index, first_frame=frame_offset,
                    frames=len(frames), **diagnostics))
                frame_offset += len(frames)
                segment_offset += max(local_segments, default=0)
                inference_s += chunk_s
                print(clip, 'SAM3 chunk', chunk_index+1, 'frames', frame_offset, '/', info['frames'], flush=True)
                del frames, chunk_native, chunk_selected, chunk_tracked
                gc.collect(); torch.cuda.empty_cache()
            if frame_offset != info['frames']:
                raise ValueError('SAM 3 decode count differs')
            diagnostics = dict(selected_native_track_id=None,
                native_track_diagnostics=None, chunked=True, chunk_frames=args.chunk_frames,
                chunks=len(chunk_diagnostics), chunk_diagnostics=chunk_diagnostics,
                selection='one small moving native track selected independently per bounded chunk')
            config = dict(model_id=args.model_id, prompt=args.prompt, tracker='sam3_native_video_track',
                reid=False, gmc=False, offline=True, selection_config=asdict(selection), **diagnostics)
            stats = dict(frames=info['frames'], detector_seconds=inference_s,
                fps_model=info['frames']/inference_s, frames_with_any_detection=sum(bool(x) for x in selected),
                states=dict(Counter(point['state'] for point in tracked)))
            output_json = args.output/f'{PREFIX}_{clip}.json'
            output_video = args.output/f'{PREFIX}_{clip}.mp4'
            output_json.write_text(json.dumps(dict(config=config, stats=stats, detections=selected,
                tracked=tracked, native_detections=native), allow_nan=False))
            render(video, output_video, tracked, info)
            meta = dict(status='complete', video=info, source_video_sha256=sha256(video),
                json=str(output_json), video_output=str(output_video), stats=stats, config=config,
                processing_s=time.perf_counter()-tick)
            report['clips'][clip] = meta; write_json(args.output/'experiment.json', report)
            del native, selected, tracked
        report.update(status='complete', processing_s=time.perf_counter()-started,
                      versions=versions, source_sha256=sha256(Path(__file__)))
        write_json(args.output/'experiment.json', report)
    except BaseException as error:
        message = str(error).replace(token, '[REDACTED]') if token else str(error)
        report.update(status='failed', error=f'{type(error).__name__}: {message}')
        write_json(args.output/'experiment.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, default=Path('data/raw'))
    parser.add_argument('--clips', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-id', default='facebook/sam3')
    parser.add_argument('--prompt', default='basketball')
    parser.add_argument('--maximum-prediction-s', type=float, default=.20)
    parser.add_argument('--chunk-frames', type=int, default=100,
                        help='Maximum frames loaded into one SAM 3 session (default: 100)')
    run(parser.parse_args())


if __name__ == '__main__': main()
