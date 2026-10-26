"""Regenerate the standalone Colab ball notebook (no repo upload needed). Runs nothing itself."""
import json
import textwrap
from pathlib import Path


def cell(kind, source):
    value = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


cells = [
    cell("markdown", r"""
# Ball detection: filtered RF-DETR + Kalman on both clips
Standalone notebook: reads `MyDrive/TacticalAI/data/raw/attack_clip.mp4` and
`attack_clip2.mp4`, runs the selected variants, tracks the ball, and **plays every result inline**.

| # | Model | What it is | Needs |
|---|---|---|---|
| 1 | **WASB-SBDT** (BMVC 2023) | Sports-ball heatmap network, sees 3 consecutive frames; official **basketball** weights (NBA broadcast) | nothing |
| 2 | **RF-DETR basketball** (Roboflow `basketball-player-detection-3-ycjdo/13`) | Requested RF-DETR medium checkpoint; architecture checked at runtime; full frame and 2x2 tiles | secret `ROBOFLOW_API_KEY` |
| 3 | **SAM 3** video, prompt `"basketball"` | Meta's text-prompted detector + video tracker | secret `HF_TOKEN`, after accepting terms at huggingface.co/facebook/sam3 |

**How to run in VS Code:** select your **Colab GPU kernel**, then **Run All**.
In browser Colab: Runtime -> Change runtime type -> **GPU**, then **Run all**.
For the default RF-DETR experiment, add your **private Roboflow API key** to
**Colab Secrets** as `ROBOFLOW_API_KEY` and enable notebook access. Get it at
https://app.roboflow.com/settings/api . Do not paste it into notebook source or chat.
If Secrets are unavailable in VS Code, setup asks for the key in a hidden prompt;
that value is kept only in the current runtime session.
RF-DETR uses local Colab inference; model authorization/download requires the key.
Re-run from setup so the new variant definitions and output directory are used.

**Current experiment:** `RF-DETR-tiles` baseline versus **Kalman + motion gates + short-gap interpolation**
on **both clips**, using saved detections from each clip. Only Kalman-filtered videos
are displayed by default; baseline metrics/videos are still saved.
This is the second individual video from the previous sweep, selected from visual
feedback, not established as the most accurate model. Settings are unchanged:
checkpoint `/13`, full frame + four overlapping 60% crops, detector cutoff 0.10,
legacy tracker cutoff 0.25, six-frame gap, and 0.6 reinitialization confidence.
The Kalman comparison requires three coherent frames to start, rejects out-of-gate
measurements regardless of confidence, predicts for at most 0.20 s, and interpolates
only gaps up to 0.16 s between accepted measurements in the same segment.
Its gates use image pixels, not physical court speed. They can reject real shots,
bounces, or camera motion; missing detections and incorrect initial locks remain possible.
`RF_REUSE_DIRS` points to each saved run; missing caches trigger inference only for
the missing clip. Set it to `{}` for fresh inference on both clips.
Cached detection time is historical, not time spent in this CPU postprocessing run.
WASB, SAM 3, and the other RF-DETR variants are skipped. No manual correction or
checkpoint location enters inference. A new output folder preserves previous runs.
To restore the clip-1 sweep, set `RF_CLIP_NAMES = ('attack_clip',)`,
`RF_SELECTED_VARIANTS = None`, and `RF_SWEEP = True`.

Set `EXPERIMENT = 'lowres'` to repeat the five low-resolution WASB variants,
using the previous final
`WASB-tiles-020` configuration as the reference. This reference choice assumes
"last test" meant that variant; visual feedback is not verified accuracy.
Set `EXPERIMENT = 'original'` in setup to restore the previous six-variant sweep
on both clips, including the unchanged clip 2 configuration. New run directories
preserve previous Drive artifacts. SAM 3 is skipped for this sweep (`RUN_OPTIONAL_MODELS = False`). Enable it separately
after model access is approved. RF-DETR remains enabled for the default experiment.
Each variant produces a full-length video, JSON evidence, and summary row.
The final cells show synchronized comparisons of the models run in this session.

**In the videos:** raw/rejected candidate rings are hidden (`SHOW_RAW_CANDIDATES = False`);
green ring = accepted measurement / filtered position; cyan ring labelled inferred =
offline interpolation; gray X = bounded Kalman extrapolation. Inferred points have
null measurement/confidence and are never counted as observed hits. Green trails
break at gaps and new segments. No interpolation across resets or long occlusions. Accepted false detections can
still appear; filtering is not verified ground truth. Raw/rejected evidence stays in JSON.
Outputs (videos, detections JSON, summary) are saved to `MyDrive/TacticalAI/outputs/ball-notebook-<time>/`.
"""),
    cell("code", r"""
# Setup (installs first, before anything is imported)
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers', 'accelerate',
                'omegaconf', 'gdown', 'uv'], check=True)

import json, math, os, time
import hashlib, importlib.metadata, platform
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from IPython.display import Video, display, Markdown
from google.colab import drive, userdata

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/TacticalAI')
EXPERIMENT = 'rfdet'  # 'rfdet': RF-DETR (optional WASB reference); 'lowres': WASB tweaks; 'original': both clips
RF_MODEL_ID = 'basketball-player-detection-3-ycjdo/13'
RF_SWEEP = False  # Used when RF_SELECTED_VARIANTS is None.
RF_SELECTED_VARIANTS = ('RF-DETR-tiles',)  # Second individual video from the prior sweep.
RF_CLIP_NAMES = ('attack_clip', 'attack_clip2')
RF_REUSE_DIRS = {  # Set to {} to force fresh inference on both clips.
    'attack_clip': ROOT / 'outputs' / 'ball-notebook-20261026-205848',
    'attack_clip2': ROOT / 'outputs' / 'ball-notebook-20261026-210537',
}
SHOW_RAW_CANDIDATES = False  # Rejected/raw candidates remain in JSON evidence.
SHOW_BASELINE = False  # Compute baseline metrics, display only Kalman videos.
COMPARE_KALMAN = True
KALMAN_SETTINGS = dict(confirmation_frames=3, max_gap_s=0.20, interpolation_gap_s=0.16,
                       max_speed_diagonal_s=1.0, max_innovation_diagonal=0.04,
                       measurement_std_diagonal=0.005, acceleration_std_diagonal_s2=2.0,
                       mahalanobis_squared_gate=9.21)
assert EXPERIMENT in ('rfdet', 'lowres', 'original')
CLIP_NAMES = (RF_CLIP_NAMES if EXPERIMENT == 'rfdet' else
              ('attack_clip', 'attack_clip2') if EXPERIMENT == 'original' else ('attack_clip',))
CLIPS = {name: ROOT / 'data/raw' / f'{name}.mp4' for name in CLIP_NAMES}
for path in CLIPS.values():
    assert path.is_file(), f'Missing video: {path}'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'outputs' / ('ball-notebook-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
OUT.mkdir(parents=True)

RF_CACHE_SOURCES = {clip: RF_REUSE_DIRS[clip] / f'rf_raw_{clip}.json'
                    for clip in CLIPS if clip in RF_REUSE_DIRS
                    and (RF_REUSE_DIRS[clip] / f'rf_raw_{clip}.json').is_file()}
RF_MISSING_CLIPS = {clip: path for clip, path in CLIPS.items() if clip not in RF_CACHE_SOURCES}
RF_CACHE_AVAILABLE = not RF_MISSING_CLIPS
assert RF_CACHE_AVAILABLE or torch.cuda.is_available(), 'Select a GPU runtime for fresh inference'
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, '| torch', torch.__version__)
print('Output folder:', OUT)

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

ROBOFLOW_API_KEY, HF_TOKEN = secret('ROBOFLOW_API_KEY'), secret('HF_TOKEN')
print('ROBOFLOW_API_KEY:', 'found' if ROBOFLOW_API_KEY else 'not loaded (not needed for cached postprocessing)')
print('HF_TOKEN:', 'found' if HF_TOKEN else 'not loaded (only needed if SAM 3 is enabled)')
if EXPERIMENT == 'rfdet' and not ROBOFLOW_API_KEY and not RF_CACHE_AVAILABLE:
    from getpass import getpass
    ROBOFLOW_API_KEY = getpass('Private Roboflow API key (hidden, session only): ').strip()
    if not ROBOFLOW_API_KEY:
        raise RuntimeError('A private Roboflow key is needed. Add it in Colab Secrets or the hidden setup prompt.')

# Minimum confidence for the tracker; WASB variant values are set in its experiment cell.
MIN_CONF = {'WASB': 0.5, 'RF-DETR': 0.25, 'SAM3': 0.3}
RUN_WASB = False  # Already checked; skip downloads, setup and inference.
RUN_OPTIONAL_MODELS = False  # Focus this run on RF-DETR; SAM 3 access was pending.
RUN_CONFIGS = {}
versions = {}
for package in ('torch', 'numpy', 'opencv-python', 'omegaconf', 'transformers', 'gdown'):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
(OUT / 'environment.json').write_text(json.dumps(dict(
    python=platform.python_version(), packages=versions, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, offline=True, reid=False, gmc=False,
    experiment=EXPERIMENT, clips=CLIP_NAMES, run_wasb=RUN_WASB,
    run_optional_models=RUN_OPTIONAL_MODELS, rf_sweep=RF_SWEEP,
    rf_selected_variants=RF_SELECTED_VARIANTS, compare_kalman=COMPARE_KALMAN,
    kalman_settings=KALMAN_SETTINGS,
    rf_cache_sources={clip: str(path) for clip, path in RF_CACHE_SOURCES.items()},
    show_raw_candidates=SHOW_RAW_CANDIDATES, show_baseline=SHOW_BASELINE,
    assumptions=['one ball', 'constant-velocity association'],
    evaluation='19 draft development checkpoints; not held-out accuracy'), indent=2))
"""),
    cell("code", r"""
# Shared helpers: video IO, ball trackers, drawing, scoring.
# __BALL_TEMPORAL_SOURCE__
def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    return frames, fps

def write_video(path, frames, fps, frame_shape=None):
    h, w = frames[0].shape[:2] if frame_shape is None else frame_shape
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                             '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                             '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
                            stdin=subprocess.PIPE)
    for img in frames:
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, f'ffmpeg failed for {path}'

def show(path, width=960):
    display(Video(str(path), embed=True, width=width, html_attributes='controls loop'))

def merge(dets, radius):
    # Drop duplicates (same ball found twice) by center distance; keep the most confident.
    kept = []
    for d in sorted(dets, key=lambda d: -d['conf']):
        if all(math.hypot(d['x'] - k['x'], d['y'] - k['y']) > radius for k in kept):
            kept.append(d)
    return kept

def track(dets_per_frame, width, height, fps, min_conf, max_gap=6):
    # One ball, constant-velocity prediction, gated nearest/most-confident association.
    diag = math.hypot(width, height)
    pos = vel = None
    last = None
    segment = 0
    out = []
    for i, dets in enumerate(dets_per_frame):
        cands = [d for d in dets if d['conf'] >= min_conf]
        if pos is not None and i - last > max_gap:
            pos = None
        chosen, reinit = None, False
        if pos is not None:
            gap = i - last
            pred = pos + vel * gap
            gate = diag * (0.03 + 0.04 * gap)
            scored = []
            for d in cands:
                dist = math.hypot(d['x'] - pred[0], d['y'] - pred[1])
                if dist <= gate:
                    scored.append((d['conf'] - 0.5 * dist / gate, d))
            if scored:
                chosen = max(scored, key=lambda s: s[0])[1]
            elif cands and max(c['conf'] for c in cands) >= 0.6:
                chosen, reinit = max(cands, key=lambda d: d['conf']), True
        elif cands:
            chosen, reinit = max(cands, key=lambda d: d['conf']), True
        if chosen is not None:
            if reinit:
                segment += 1
            p = np.array([chosen['x'], chosen['y']])
            vel = np.zeros(2) if reinit else 0.6 * vel + 0.4 * (p - pos) / (i - last)
            pos, last = p, i
            out.append(dict(state='detected', x=float(p[0]), y=float(p[1]), conf=chosen['conf'], segment_id=segment))
        elif pos is not None:
            pred = pos + vel * (i - last)
            if 0 <= pred[0] < width and 0 <= pred[1] < height:
                out.append(dict(state='predicted', x=float(pred[0]), y=float(pred[1]), conf=None, segment_id=segment))
            else:
                pos = None
                out.append(dict(state='none'))
        else:
            out.append(dict(state='none'))
    return out

def draw(img, dets, tracked, index, label, scale=1.0):
    img = img.copy() if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale)
    u = max(1, round(img.shape[0] / 360))
    if SHOW_RAW_CANDIDATES:
        for d in dets[index]:
            r = max(5 * u, int(d.get('r', 0) * scale) + 3 * u)
            cv2.circle(img, (int(d['x'] * scale), int(d['y'] * scale)), r, (0, 140, 255), u)
    for segment_points in trail_segments(tracked, index):
        points = np.array([(int(x*scale), int(y*scale)) for x, y in segment_points], np.int32)
        cv2.polylines(img, [points], False, (0, 200, 0), u, cv2.LINE_AA)
    t = tracked[index]
    if t['state'] == 'detected':
        c = (int(t['x'] * scale), int(t['y'] * scale))
        cv2.circle(img, c, 11 * u, (0, 230, 0), 2 * u)
        cv2.putText(img, f"{t['conf']:.2f}", (c[0] + 12 * u, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * u,
                    (0, 230, 0), u, cv2.LINE_AA)
    elif t['state'] == 'interpolated':
        c = (int(t['x']*scale), int(t['y']*scale))
        cv2.circle(img, c, 9*u, (255, 210, 0), u)
        cv2.putText(img, 'inferred', (c[0]+10*u, c[1]), cv2.FONT_HERSHEY_SIMPLEX, .35*u, (255, 210, 0), u)
    elif t['state'] == 'predicted':
        x, y, s = int(t['x'] * scale), int(t['y'] * scale), 6 * u
        cv2.line(img, (x - s, y - s), (x + s, y + s), (170, 170, 170), u)
        cv2.line(img, (x - s, y + s), (x + s, y - s), (170, 170, 170), u)
    cv2.rectangle(img, (0, 0), (img.shape[1], 16 * u), (0, 0, 0), -1)
    cv2.putText(img, f"{label} | frame {index} | {t['state']}", (4 * u, 12 * u), cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * u, (255, 255, 255), u, cv2.LINE_AA)
    return img

# Draft check points: ball centers marked by eye on 19 frames (1280x720 coordinates).
# Not certified ground truth; ambiguous frames were left out.
REFERENCE = {
    'attack_clip': {0: (184, 384), 30: (166, 410), 40: (130, 426), 50: (174, 312), 80: (330, 358),
                    90: (348, 358), 100: (364, 218), 120: (685, 95)},
    'attack_clip2': {0: (251, 476), 24: (337, 474), 36: (440, 495), 48: (483, 471), 60: (542, 483),
                     96: (783, 572), 108: (744, 460), 120: (650, 392), 144: (695, 407), 156: (730, 337),
                     168: (765, 229)},
}
MATCH_RADIUS = 14  # pixels in 1280x720 space
(OUT / 'evaluation_draft.json').write_text(json.dumps(dict(
    reference=REFERENCE, coordinate_size=[1280, 720], match_radius=MATCH_RADIUS,
    status='draft development checkpoints, not held-out ground truth'), indent=2))

def score(clip, dets, tracked, width, height):
    sx, sy = 1280 / width, 720 / height
    raw_hits = tracked_hits = tracked_wrong = false_candidates = candidates = 0
    for frame, (rx, ry) in REFERENCE[clip].items():
        near = lambda d: math.hypot(d['x'] * sx - rx, d['y'] * sy - ry) <= MATCH_RADIUS
        raw_hits += any(near(d) for d in dets[frame])
        candidates += len(dets[frame])
        false_candidates += sum(not near(d) for d in dets[frame])
        t = tracked[frame]
        if t['state'] == 'detected':
            if near(t):
                tracked_hits += 1
            else:
                tracked_wrong += 1
    return dict(check_frames=len(REFERENCE[clip]), raw_found=raw_hits, tracked_correct=tracked_hits,
                tracked_wrong_place=tracked_wrong, check_candidates=candidates,
                check_off_target_candidates=false_candidates)

RESULTS = {}

def run_model(label, detect, tracking=None):
    # detect(clip_name, frames) -> (list of per-frame detections [{x, y, r, conf}], seconds)
    RESULTS[label] = {}
    for clip, path in CLIPS.items():
        try:
            frames, fps = read_video(path)
            h, w = frames[0].shape[:2]
            dets, seconds = detect(clip, frames)
            assert len(dets) == len(frames), 'detector returned wrong frame count'
            tracking_started = time.perf_counter()
            tracked = (tracking or track)(dets, w, h, fps, MIN_CONF[label])
            tracking_seconds = time.perf_counter()-tracking_started
            video = OUT / f'{label}_{clip}.mp4'
            write_video(video, (draw(f, dets, tracked, i, label) for i, f in enumerate(frames)),
                        fps, frame_shape=(h, w))
            states = [t['state'] for t in tracked]
            stats = dict(frames=len(frames), detector_seconds=round(seconds, 3),
                         fps_model=round(len(frames) / seconds, 1),
                         detector_time_origin='cached_original_run' if RUN_CONFIGS.get(label, {}).get('reused_detections') else 'this_run',
                         frames_with_any_detection=sum(bool(d) for d in dets),
                         frames_ball_tracked=states.count('detected'), frames_predicted=states.count('predicted'),
                         frames_interpolated=states.count('interpolated'), tracker_seconds=round(tracking_seconds, 4),
                         motion_rejections=sum(len(t.get('rejected', [])) for t in tracked),
                         **score(clip, dets, tracked, w, h))
            RESULTS[label][clip] = dict(stats=stats, detections=dets, tracked=tracked)
            (OUT / f'{label}_{clip}.json').write_text(json.dumps(dict(stats=stats, detections=dets,
                tracked=tracked, config=RUN_CONFIGS.get(label, {}))))
            print(label, clip, stats)
            if SHOW_BASELINE or not COMPARE_KALMAN or label.endswith('-Kalman'):
                show(video)
        except Exception as error:
            RESULTS[label][clip] = dict(error=f'{type(error).__name__}: {error}')
            print(label, clip, 'FAILED:', RESULTS[label][clip]['error'])
            (OUT / f'{label}_{clip}_error.json').write_text(json.dumps(RESULTS[label][clip]))
"""),
    cell("markdown", r"""
## 1. WASB-SBDT: low-resolution sweep (same basketball weights)
Official repo + authors' basketball checkpoint (cached in `MyDrive/TacticalAI/models/`).
Each step feeds 3 consecutive frames at 512x288 and gets 3 heatmaps back, so every frame
is checked up to 3 times.

Default `lowres` profile, **clip 1 only**:

| Variant | Views | Threshold | Change from reference |
|---|---|---|---|
| WASB-tiles-020 | full + 2x2 crops (60% size) | 0.20 | Previous final configuration |
| WASB-tight-020 | full + 3x3 crops (40% size) | 0.20 | Tighter crops |
| WASB-tight-010 | full + 3x3 crops (40% size) | 0.10 | Tighter crops, lower threshold |
| WASB-tight-cubic | full + 3x3 crops (40% size) | 0.20 | Tighter crops, cubic resizing |
| WASB-tight-clahe | full + 3x3 crops (40% size) | 0.20 | Tighter crops, mild luminance CLAHE |

At 640x360, 40% crops are 256x144, resized to 512x288. The ball occupies
more network pixels, but resizing does not recover lost image detail. CLAHE
uses clipLimit=1.5, grid=8x8 on LAB luminance before resizing. This is an
experiment, not a proven improvement; contrast may also amplify compression noise.
All crops cover the whole image automatically; no hand-marked locations guide inference.
The full frame is retained for context and boundary recovery. Distinct views are shared
between thresholds but processed again for different interpolation/preprocessing.
This profile uses 34 distinct view/preprocessing combinations per temporal window,
versus five for the previous tiled detector, so expect substantially more GPU work.
Visible tiny-ball misses and actual occlusion are different failures: no hidden-ball
measurements are created. Gray crosses remain short motion predictions, then unknown.

`original` profile retains the previous settings on both clips:

| Variant | Views | Heatmap / tracker threshold |
|---|---|---|
| WASB-base | full frame | 0.50 / 0.50 (original baseline) |
| WASB-full-035 | full frame | 0.35 / 0.35 |
| WASB-full-020 | full frame | 0.20 / 0.20 |
| WASB-full-010 | full frame | 0.10 / 0.10 |
| WASB-tiles-035 | full + four overlapping 60% crops | 0.35 / 0.35 |
| WASB-tiles-020 | full + four overlapping 60% crops | 0.20 / 0.20 |

Only inference/postprocessing changes; no retraining. Crop coordinates are mapped back
automatically. Threshold variants reuse the same heatmaps. Reported detector time
includes preprocessing, GPU inference/transfers, and candidate extraction/merging;
shared inference time is charged to each applicable variant, not measured as isolated runs.
It excludes video decoding, tracking and export. Standard tiles cost five views per
window; tighter crops cost ten, before sharing identical views between variants.
The tracker retains the original 0.6 out-of-gate reinitialization rule and six-frame gap.
Lower thresholds can produce many false positives. Draft checkpoints are development
feedback used to motivate this sweep; they are not an independent evaluation set.
"""),
    cell("code", r"""
import importlib.util, gdown
from omegaconf import OmegaConf

WASB_DIR = Path('/content/WASB-SBDT')
if not WASB_DIR.exists():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/nttcom/WASB-SBDT.git', str(WASB_DIR)], check=True)
WASB_COMMIT = '923462cacdeb3353b84ddebdedb3f4b7a8553b0f'
subprocess.run(['git', '-C', str(WASB_DIR), 'checkout', '-q', WASB_COMMIT], check=True)
WASB_WEIGHTS = MODELS_DIR / 'wasb_basketball_best.pth.tar'
if not WASB_WEIGHTS.exists():
    gdown.download(id='1nfECuSyJvPUmz3njZCdFERSQQbERt8FU', output=str(WASB_WEIGHTS), quiet=False)

spec = importlib.util.spec_from_file_location('wasb_hrnet', WASB_DIR / 'src/models/hrnet.py')
hrnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hrnet)
wasb_cfg = OmegaConf.load(WASB_DIR / 'src/configs/model/wasb.yaml')
wasb = hrnet.HRNet(wasb_cfg)
state = torch.load(WASB_WEIGHTS, map_location='cpu', weights_only=False)
state = state.get('model_state_dict', state)
wasb.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
wasb = wasb.cuda().eval()
# __WASB_SWEEP_SOURCE__ (inserted by the builder; no repo upload needed)

ACTIVE_VARIANTS = ({'WASB-tiles-020': WASB_VARIANTS['WASB-tiles-020']}
                   if EXPERIMENT == 'rfdet' else
                   WASB_LOWRES_VARIANTS if EXPERIMENT == 'lowres' else WASB_VARIANTS)
weight_sha256 = hashlib.sha256(WASB_WEIGHTS.read_bytes()).hexdigest()
for label, raw_config in ACTIVE_VARIANTS.items():
    config = dict(raw_config)
    config.setdefault('crop_fraction', 0.6)
    config.setdefault('preprocessing', 'none')
    config.setdefault('interpolation', 'linear')
    MIN_CONF[label] = config['threshold']
    RUN_CONFIGS[label] = dict(**config, tracker_min_conf=config['threshold'],
        tracker_max_gap=6, tracker_reinitialize_conf=0.6, batch=4,
        checkpoint_sha256=weight_sha256, upstream_commit=WASB_COMMIT,
        input_size=[512, 288], frames_in=3, frames_out=3,
        clahe_clip_limit=1.5 if config['preprocessing'] == 'clahe' else None,
        clahe_grid=[8, 8] if config['preprocessing'] == 'clahe' else None,
        experiment=EXPERIMENT,
        reid=False, gmc=False, offline=True,
        timing='shared inference + variant extraction; excludes decode/tracking/export')
(OUT / 'wasb_configs.json').write_text(json.dumps(RUN_CONFIGS, indent=2))

# Cache candidate lists, not heatmaps. Each clip is inferred once per view.
sweep_cache = {}
sweep_errors = {}
def variant_detector(label):
    def detect(clip, frames):
        if clip in sweep_errors:
            raise RuntimeError(sweep_errors[clip])
        if clip not in sweep_cache:
            print('Inferring selected WASB views for', clip, EXPERIMENT, flush=True)
            try:
                sweep_cache[clip] = wasb_sweep(wasb, frames, batch=4, variants=ACTIVE_VARIANTS)
            except Exception as error:
                sweep_errors[clip] = str(error)
                raise
        return sweep_cache[clip][label]
    return detect

for label in ACTIVE_VARIANTS:
    print('\nVARIANT', label, RUN_CONFIGS[label]['mode'], 'threshold', MIN_CONF[label])
    run_model(label, variant_detector(label))
del wasb; torch.cuda.empty_cache()
"""),
    cell("markdown", r"""
## 2. RF-DETR basketball (Roboflow)
Runs in its own small Python environment (avoids package clashes with this notebook).
Weights download with your key; inference runs locally in Colab. The default run selects **RF-DETR-tiles on both clips**, with the Kalman overlay. The available variants are:

| Variant | Views | Candidate cutoff | Tracker cutoff |
|---|---|---|---|
| RF-DETR-full | full frame | 0.10 | 0.25 |
| RF-DETR-tiles | full + 2x2, 60% crops (baseline) | 0.10 | 0.25 |
| RF-DETR-tight | full + 3x3, 40% crops | 0.10 | 0.25 |
| RF-DETR-tight-low | same tighter crops | 0.05 | 0.25 |
| RF-DETR-tight-track010 | same tighter crops | 0.05 | 0.10 |
| RF-DETR-tiles-track040 | baseline views | 0.10 | 0.40 |

Tight variants share inference at 0.05, then filter candidates to their stated cutoff.
Baseline variants keep inference at 0.10. No checkpoint location enters detection or
tracking. Crop batches are limited to four views. Timings charge shared inference
to each applicable variant, not independently timed runs. Tracking gap (6 frames)
and reinitialization confidence (0.6) stay fixed. Raw checkpoint hits, wrong tracked
locations, and off-target candidates matter more than visual coverage.
These tests were selected after reviewing the previous run: only 3/8 draft hits,
with 9 off-target candidates for full frame and 24 for tiles. This is development
feedback, not an independent evaluation. Set `RF_SWEEP = False` for the two baselines.
The adapter's underlying architecture, active execution provider, and library versions are saved.
A three-frame full/tiled smoke test must pass on GPU before the full clip runs.
An isolated Python 3.11 / Torch CUDA 12.4 / ONNX Runtime CUDA 12 environment
avoids inheriting Colab's CUDA 13 libraries. Its first installation can take several minutes.
Model availability and access are checked when your key is supplied; this checkpoint has
not yet been downloaded or executed here. No cloud inference fallback is used.
"""),
    cell("code", r"""
RF_SCRIPT = Path('/content/rf_detect.py')
RF_SCRIPT.write_text(__RF_BALL_SOURCE__)
RF_LABELS = ['RF-DETR-full', 'RF-DETR-tiles']
if RF_SELECTED_VARIANTS is not None:
    RF_LABELS = list(RF_SELECTED_VARIANTS)
elif RF_SWEEP:
    RF_LABELS += ['RF-DETR-tight', 'RF-DETR-tight-low', 'RF-DETR-tight-track010', 'RF-DETR-tiles-track040']


def rf_detector(source_label, output_label, temporal=False):
    def detect(clip, frames):
        data = json.loads((OUT / f'rf_raw_{clip}.json').read_text())
        result = data['variants'][source_label]
        assert len(result['detections']) == len(frames), 'Cached detection frame count differs from video'
        RUN_CONFIGS[output_label] = dict(**data['provenance'], **result['config'],
            tracker='kalman_cv_gated_interpolation' if temporal else 'legacy_constant_velocity',
            reid=False, gmc=False, offline=True,
            temporal_config=asdict(TemporalConfig(min_confidence=MIN_CONF[output_label], **KALMAN_SETTINGS)) if temporal else None,
            source_detection_file=str(RF_CACHE_SOURCES[clip]) if clip in RF_CACHE_SOURCES else None,
            source_hash_verified=False if clip in RF_CACHE_SOURCES else None,
            reused_detections=clip in RF_CACHE_SOURCES, raw_candidates_visible=SHOW_RAW_CANDIDATES)
        return result['detections'], result['seconds']
    return detect


if (EXPERIMENT == 'rfdet' or RUN_OPTIONAL_MODELS) and (ROBOFLOW_API_KEY or RF_CACHE_AVAILABLE):
    try:
        for clip, source in RF_CACHE_SOURCES.items():
            print('Reusing saved detections:', source)
            data = json.loads(source.read_text())
            assert data['provenance']['model_id'] == RF_MODEL_ID, 'Cached checkpoint differs'
            assert all(label in data['variants'] for label in RF_LABELS), 'Selected variant absent from cache'
            (OUT / f'rf_raw_{clip}.json').write_text(json.dumps(data))
        if RF_CACHE_SOURCES:
            print('Cached model/variant/frame count checked; older caches have no source-video hash.')
        if RF_MISSING_CLIPS:
            venv = Path('/content/venv_rf_cuda12_v2')
            if not venv.exists():
                subprocess.run(['uv', 'venv', '-q', '--python', '3.11', str(venv)], check=True)
            subprocess.run(['uv', 'pip', 'install', '-q', '--python', str(venv / 'bin/python'),
                            '--index-url', 'https://download.pytorch.org/whl/cu124',
                            'torch==2.6.0', 'torchvision==0.21.0'], check=True)
            subprocess.run(['uv', 'pip', 'install', '-q', '--python', str(venv / 'bin/python'),
                            'inference-gpu==1.8.3', 'onnxruntime-gpu[cuda,cudnn]==1.21.1',
                            'torch==2.6.0', 'torchvision==0.21.0'], check=True)
            freeze = subprocess.run(['uv', 'pip', 'freeze', '--python', str(venv / 'bin/python')],
                                    check=True, capture_output=True, text=True)
            (OUT / 'rf_environment.txt').write_text(freeze.stdout)
            args = [str(venv / 'bin/python'), str(RF_SCRIPT)]
            for clip, path in RF_MISSING_CLIPS.items():
                args += [str(path), str(OUT / f'rf_raw_{clip}.json')]
            completed = subprocess.run(args, env=dict(os.environ, ROBOFLOW_API_KEY=ROBOFLOW_API_KEY,
                RF_MODEL_ID=RF_MODEL_ID, RF_SWEEP=str(RF_SWEEP).lower(),
                RF_VARIANTS=','.join(RF_SELECTED_VARIANTS or ()), ACTIVE_LEARNING_ENABLED='false'),
                capture_output=True, text=True)
            print((completed.stdout + completed.stderr).replace(ROBOFLOW_API_KEY, '[REDACTED]'))
            if completed.returncode:
                raise RuntimeError('RF-DETR inference failed; see redacted output above')
        raw = json.loads((OUT / f'rf_raw_{next(iter(CLIPS))}.json').read_text())
        for label in RF_LABELS:
            MIN_CONF[label] = raw['variants'][label]['config']['tracker_min_conf']
            run_model(label, rf_detector(label, label))
            if COMPARE_KALMAN:
                output_label = label+'-Kalman'
                MIN_CONF[output_label] = MIN_CONF[label]
                def kalman_tracking(detections, width, height, fps, confidence):
                    return track_temporal(detections, width, height, fps,
                                          TemporalConfig(min_confidence=confidence, **KALMAN_SETTINGS))
                run_model(output_label, rf_detector(label, output_label, temporal=True), tracking=kalman_tracking)
    except Exception as error:
        message = str(error).replace(ROBOFLOW_API_KEY, '[REDACTED]') if ROBOFLOW_API_KEY else str(error)
        for label in RF_LABELS:
            RESULTS.setdefault(label, {clip: dict(error=message) for clip in CLIPS})
        (OUT / 'rf_setup_error.json').write_text(json.dumps(dict(error=message)))
        print('RF-DETR setup failed:', message)
else:
    print('RF-DETR skipped (no key or cached detections).')
"""),
    cell("markdown", r"""
## 3. SAM 3 video, text prompt "basketball"
Meta's model finds every object matching the text and follows each one through the video
using memory (it reads the whole clip first, so this is offline processing).
"""),
    cell("code", r"""
SAM3_PROMPT = 'basketball'

if RUN_OPTIONAL_MODELS and not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass('Hugging Face read token (hidden, session only; Enter to skip): ').strip()

if RUN_OPTIONAL_MODELS and HF_TOKEN:
    try:
        import inspect
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        sam3 = Sam3VideoModel.from_pretrained('facebook/sam3', token=HF_TOKEN).to('cuda', dtype=dtype).eval()
        sam3_processor = Sam3VideoProcessor.from_pretrained('facebook/sam3', token=HF_TOKEN)

        def to_numpy(value):
            return value.detach().float().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value, float)

        def sam3_detect(clip, frames):
            options = dict(video=np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]),
                           inference_device='cuda', processing_device='cpu', video_storage_device='cpu')
            if 'dtype' in inspect.signature(sam3_processor.init_video_session).parameters:
                options['dtype'] = dtype
            session = sam3_processor.init_video_session(**options)
            sam3_processor.add_text_prompt(inference_session=session, text=SAM3_PROMPT)
            found = [[] for _ in frames]
            torch.cuda.synchronize(); t0 = time.time()
            with torch.inference_mode():
                for output in sam3.propagate_in_video_iterator(inference_session=session):
                    r = sam3_processor.postprocess_outputs(session, output)
                    ids = to_numpy(r['object_ids']).reshape(-1)
                    for oid, conf, (x0, y0, x1, y1) in zip(ids, to_numpy(r['scores']).reshape(-1),
                                                          to_numpy(r['boxes']).reshape(-1, 4)):
                        found[output.frame_idx].append(dict(x=float((x0 + x1) / 2), y=float((y0 + y1) / 2),
                                                            r=float(max(x1 - x0, y1 - y0) / 2),
                                                            conf=float(conf), sam3_id=int(oid)))
            torch.cuda.synchronize()
            return found, time.time() - t0

        run_model('SAM3', sam3_detect)
        del sam3; torch.cuda.empty_cache()
    except Exception as error:
        message = str(error).replace(HF_TOKEN, '[REDACTED]')
        for clip in CLIPS:
            RESULTS.setdefault('SAM3', {}).setdefault(clip, dict(error=message))
        (OUT / 'sam3_setup_error.json').write_text(json.dumps(dict(error=message)))
        print('SAM 3 unavailable; continuing to summary:', message)
    finally:
        if 'sam3' in globals():
            del sam3
        torch.cuda.empty_cache()
else:
    print('SAM3 skipped (optional models disabled or token missing).')
"""),
    cell("markdown", r"""
## Side-by-side comparison and summary
Every successful variant appears in synchronized comparison videos. Each page has
the original and up to three variants so the ball stays visible. Individual full-size
videos above remain available. All rows (including failures) are saved in summary.csv.
"""),
    cell("code", r"""
import pandas as pd

for clip, path in CLIPS.items():
    frames, fps = read_video(path)
    scale = 640 / frames[0].shape[1]
    labels = [label for label in RESULTS if 'tracked' in RESULTS[label].get(clip, {})
              and (SHOW_BASELINE or not COMPARE_KALMAN or label.endswith('-Kalman'))]
    for page, start in enumerate(range(0, len(labels), 3), 1):
        selected = labels[start:start+3]
        def comparison_frames():
            for i, f in enumerate(frames):
                panels = [cv2.resize(f, None, fx=scale, fy=scale)]
                cv2.putText(panels[0], 'Original', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                for label in selected:
                    r = RESULTS[label][clip]
                    panels.append(draw(f, r['detections'], r['tracked'], i, label, scale))
                while len(panels) < 4:
                    panels.append(np.zeros_like(panels[0]))
                yield np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        video = OUT / f'comparison_{clip}_page{page}.mp4'
        # Stream composition instead of retaining a full 1280x720 grid in memory.
        write_video(video, comparison_frames(), fps, frame_shape=(720, 1280))
        print(clip, 'comparison', page, selected)
        show(video, width=1100)

rows = [dict(model=label, clip=clip, **r.get('stats', {'error': r.get('error')}))
        for label, clips in RESULTS.items() for clip, r in clips.items()]
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'summary.csv', index=False)
display(summary)
display(Markdown('''
- **raw_found**: of the hand-marked check frames (8 in clip 1, 11 in clip 2), how many had *some* detection on the ball.
- **tracked_correct / tracked_wrong_place**: the tracked ball was on the real ball / somewhere else.
- **check_off_target_candidates**: candidates outside the draft reference radius on checked frames (duplicates on-target are not penalized). Not a complete precision metric.
- Interpolated/predicted positions are inferred, excluded from observed-hit counts. Tracker time is separate from detector time, which is historical when cached.
- The 19 draft checkpoints are development feedback, not held-out ground truth. Coverage and smooth overlays are not accuracy; evaluate separately before selecting a winner.
'''))
print('Everything saved in', OUT)
"""),
]

temporal_source = (Path(__file__).parent / 'ball_temporal.py').read_text(encoding='utf-8-sig')
module_source = (Path(__file__).parent / 'wasb_sweep.py').read_text(encoding='utf-8')
rf_source = (Path(__file__).parent / 'rf_ball_detect.py').read_text(encoding='utf-8')
for item in cells:
    source = ''.join(item['source'])
    if '# __BALL_TEMPORAL_SOURCE__' in source:
        source = source.replace('# __BALL_TEMPORAL_SOURCE__', temporal_source)
        item['source'] = source.splitlines(keepends=True)
    if '# __WASB_SWEEP_SOURCE__' in source:
        source = source.replace('# __WASB_SWEEP_SOURCE__ (inserted by the builder; no repo upload needed)', module_source)
        source = ('if RUN_WASB:\n' + textwrap.indent(source, '    ')
                  + "\nelse:\n    print('WASB skipped (RUN_WASB = False; previously checked).')\n")
        item['source'] = source.splitlines(keepends=True)
    if '__RF_BALL_SOURCE__' in source:
        item['source'] = source.replace('__RF_BALL_SOURCE__', repr(rf_source)).splitlines(keepends=True)

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                        language_info=dict(name="python", version="3.12"),
                                        accelerator="GPU", colab=dict(gpuType="L4", provenance=[])),
                nbformat=4, nbformat_minor=5)
for index, item in enumerate(cells):
    item["id"] = f"ball-{index:02d}"
target = Path(__file__).resolve().parents[1] / "notebooks/colab_ball_benchmark.ipynb"
if target.exists():
    # Keep the user's editor/kernel metadata; clear obsolete execution outputs.
    previous = json.loads(target.read_text(encoding='utf-8'))
    for key in ('kernelspec', 'language_info'):
        if key in previous.get('metadata', {}):
            notebook['metadata'][key] = previous['metadata'][key]
target.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {target}")
