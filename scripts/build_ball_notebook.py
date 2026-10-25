"""Regenerate the standalone Colab ball notebook (no repo upload needed). Runs nothing itself."""
import json
from pathlib import Path


def cell(kind, source):
    value = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


cells = [
    cell("markdown", r"""
# Ball detection + tracking: top 3 models on clip 1 and clip 2
Standalone notebook: reads `MyDrive/TacticalAI/data/raw/attack_clip.mp4` and
`attack_clip2.mp4`, runs each model, tracks the ball, and **plays the result inline**.

| # | Model | What it is | Needs |
|---|---|---|---|
| 1 | **WASB-SBDT** (BMVC 2023) | Sports-ball heatmap network, sees 3 consecutive frames; official **basketball** weights (NBA broadcast) | nothing |
| 2 | **RF-DETR basketball** (Roboflow `basketball-player-detection-3-ycjdo/4`) | Basketball detector with `ball` class; run on full frame + 2x2 zoomed tiles for small balls | secret `ROBOFLOW_API_KEY` (free account) |
| 3 | **SAM 3** video, prompt `"basketball"` | Meta's text-prompted detector + video tracker | secret `HF_TOKEN`, after accepting terms at huggingface.co/facebook/sam3 |

**How to run:** Runtime -> Change runtime type -> **GPU** (L4/A100 best, T4 ok) -> add the
secrets in the key icon (enable notebook access) -> Runtime -> **Run all**.
A model whose secret is missing is skipped; the others still run.

**In the videos:** thin orange ring = any raw detection; **green ring + trail** = tracked ball;
gray X = short motion guess while the ball is not detected (not a measurement).
Outputs (videos, detections JSON, summary) are saved to `MyDrive/TacticalAI/outputs/ball-notebook-<time>/`.
"""),
    cell("code", r"""
# Setup (installs first, before anything is imported)
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers', 'accelerate',
                'omegaconf', 'gdown', 'uv'], check=True)

import json, math, os, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from IPython.display import Video, display, Markdown
from google.colab import drive, userdata

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/TacticalAI')
CLIPS = {name: ROOT / 'data/raw' / f'{name}.mp4' for name in ('attack_clip', 'attack_clip2')}
for path in CLIPS.values():
    assert path.is_file(), f'Missing video: {path}'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'outputs' / ('ball-notebook-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
OUT.mkdir(parents=True)

assert torch.cuda.is_available(), 'Select a GPU runtime first'
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)
print('Output folder:', OUT)

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

ROBOFLOW_API_KEY, HF_TOKEN = secret('ROBOFLOW_API_KEY'), secret('HF_TOKEN')
print('ROBOFLOW_API_KEY:', 'found' if ROBOFLOW_API_KEY else 'missing -> RF-DETR will be skipped')
print('HF_TOKEN:', 'found' if HF_TOKEN else 'missing -> SAM 3 will be skipped')

# Minimum confidence for the tracker to use a detection (scores mean different things per model).
MIN_CONF = {'WASB': 0.5, 'RF-DETR': 0.25, 'SAM3': 0.3}
"""),
    cell("code", r"""
# Shared helpers: video IO, one shared ball tracker, drawing, scoring.
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

def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
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
            p = np.array([chosen['x'], chosen['y']])
            vel = np.zeros(2) if reinit else 0.6 * vel + 0.4 * (p - pos) / (i - last)
            pos, last = p, i
            out.append(dict(state='detected', x=float(p[0]), y=float(p[1]), conf=chosen['conf']))
        elif pos is not None:
            pred = pos + vel * (i - last)
            if 0 <= pred[0] < width and 0 <= pred[1] < height:
                out.append(dict(state='predicted', x=float(pred[0]), y=float(pred[1]), conf=None))
            else:
                pos = None
                out.append(dict(state='none'))
        else:
            out.append(dict(state='none'))
    return out

def draw(img, dets, tracked, index, label, scale=1.0):
    img = img.copy() if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale)
    u = max(1, round(img.shape[0] / 360))
    for d in dets[index]:
        r = max(5 * u, int(d.get('r', 0) * scale) + 3 * u)
        cv2.circle(img, (int(d['x'] * scale), int(d['y'] * scale)), r, (0, 140, 255), u)
    trail = [(int(t['x'] * scale), int(t['y'] * scale)) for t in tracked[max(0, index - 25):index + 1]
             if t['state'] == 'detected']
    if len(trail) > 1:
        cv2.polylines(img, [np.array(trail, np.int32)], False, (0, 200, 0), u, cv2.LINE_AA)
    t = tracked[index]
    if t['state'] == 'detected':
        c = (int(t['x'] * scale), int(t['y'] * scale))
        cv2.circle(img, c, 11 * u, (0, 230, 0), 2 * u)
        cv2.putText(img, f"{t['conf']:.2f}", (c[0] + 12 * u, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * u,
                    (0, 230, 0), u, cv2.LINE_AA)
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

def score(clip, dets, tracked, width, height):
    sx, sy = 1280 / width, 720 / height
    raw_hits = tracked_hits = tracked_wrong = 0
    for frame, (rx, ry) in REFERENCE[clip].items():
        near = lambda d: math.hypot(d['x'] * sx - rx, d['y'] * sy - ry) <= MATCH_RADIUS
        raw_hits += any(near(d) for d in dets[frame])
        t = tracked[frame]
        if t['state'] == 'detected':
            if near(t):
                tracked_hits += 1
            else:
                tracked_wrong += 1
    return dict(check_frames=len(REFERENCE[clip]), raw_found=raw_hits, tracked_correct=tracked_hits,
                tracked_wrong_place=tracked_wrong)

RESULTS = {}

def run_model(label, detect):
    # detect(clip_name, frames) -> (list of per-frame detections [{x, y, r, conf}], seconds)
    RESULTS[label] = {}
    for clip, path in CLIPS.items():
        try:
            frames, fps = read_video(path)
            h, w = frames[0].shape[:2]
            dets, seconds = detect(clip, frames)
            assert len(dets) == len(frames), 'detector returned wrong frame count'
            tracked = track(dets, w, h, fps, MIN_CONF[label])
            video = OUT / f'{label}_{clip}.mp4'
            write_video(video, [draw(f, dets, tracked, i, label) for i, f in enumerate(frames)], fps)
            states = [t['state'] for t in tracked]
            stats = dict(frames=len(frames), fps_model=round(len(frames) / seconds, 1),
                         frames_with_any_detection=sum(bool(d) for d in dets),
                         frames_ball_tracked=states.count('detected'), frames_predicted=states.count('predicted'),
                         **score(clip, dets, tracked, w, h))
            RESULTS[label][clip] = dict(stats=stats, detections=dets, tracked=tracked)
            (OUT / f'{label}_{clip}.json').write_text(json.dumps(dict(stats=stats, detections=dets,
                                                                      tracked=tracked)))
            print(label, clip, stats)
            show(video)
        except Exception as error:
            RESULTS[label][clip] = dict(error=f'{type(error).__name__}: {error}')
            print(label, clip, 'FAILED:', RESULTS[label][clip]['error'])
"""),
    cell("markdown", r"""
## 1. WASB-SBDT (basketball weights)
Official repo + authors' basketball checkpoint (cached in `MyDrive/TacticalAI/models/`).
Each step feeds 3 consecutive frames at 512x288 and gets 3 heatmaps back, so every frame
is checked up to 3 times. Heatmap peaks above 0.5 become detections.
"""),
    cell("code", r"""
import importlib.util, gdown
from omegaconf import OmegaConf

WASB_DIR = Path('/content/WASB-SBDT')
if not WASB_DIR.exists():
    subprocess.run(['git', 'clone', '-q', 'https://github.com/nttcom/WASB-SBDT.git', str(WASB_DIR)], check=True)
    subprocess.run(['git', '-C', str(WASB_DIR), 'checkout', '-q', '923462cacdeb3353b84ddebdedb3f4b7a8553b0f'], check=True)
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

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def wasb_detect(clip, frames, threshold=0.5, batch=16):
    h, w = frames[0].shape[:2]
    s = min(512 / w, 288 / h)
    tx, ty = (512 - s * w) / 2, (288 - s * h) / 2
    M = np.float32([[s, 0, tx], [0, s, ty]])
    x = [torch.from_numpy(((cv2.cvtColor(cv2.warpAffine(f, M, (512, 288)), cv2.COLOR_BGR2RGB) / 255.0
                            - MEAN) / STD).astype(np.float32).transpose(2, 0, 1)) for f in frames]
    starts = list(range(len(frames) - 2))
    found = [[] for _ in frames]
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(starts), batch):
            chunk = starts[b:b + batch]
            inp = torch.stack([torch.cat(x[i:i + 3]) for i in chunk]).cuda()
            heatmaps = wasb(inp)[0].sigmoid().cpu().numpy()
            for start, hms in zip(chunk, heatmaps):
                for k, hm in enumerate(hms):
                    n, labels = cv2.connectedComponents((hm > threshold).astype(np.uint8))
                    for lab in range(1, n):
                        ys, xs = np.nonzero(labels == lab)
                        wts = hm[ys, xs]
                        cx, cy = (xs * wts).sum() / wts.sum(), (ys * wts).sum() / wts.sum()
                        found[start + k].append(dict(x=float((cx - tx) / s), y=float((cy - ty) / s),
                                                     r=float(max(np.ptp(xs) + 1, np.ptp(ys) + 1) / s / 2),
                                                     conf=float(wts.max())))
    torch.cuda.synchronize()
    seconds = time.time() - t0
    return [merge(d, radius=4 / s) for d in found], seconds

run_model('WASB', wasb_detect)
del wasb; torch.cuda.empty_cache()
"""),
    cell("markdown", r"""
## 2. RF-DETR basketball (Roboflow)
Runs in its own small Python environment (avoids package clashes with this notebook).
Weights download with your key; inference runs on this Colab GPU. Each frame is checked
whole **and** as 4 overlapping zoomed tiles, so a small ball covers more model pixels.
"""),
    cell("code", r"""
RF_SCRIPT = Path('/content/rf_detect.py')
RF_SCRIPT.write_text('''
import json, math, os, sys, time
import cv2
from inference import get_model

model = get_model('basketball-player-detection-3-ycjdo/4', api_key=os.environ['ROBOFLOW_API_KEY'])
names = set()
for video, out in zip(sys.argv[1::2], sys.argv[2::2]):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    h, w = frames[0].shape[:2]
    tw, th = int(w * 0.6), int(h * 0.6)
    crops = [(0, 0, w, h)] + [(x, y, tw, th) for y in (0, h - th) for x in (0, w - tw)]
    result, t0 = [], time.time()
    for img in frames:
        images = [img[y:y + ch, x:x + cw] for x, y, cw, ch in crops]
        preds = model.infer(images, confidence=0.1)
        dets = []
        for (ox, oy, _, _), p in zip(crops, preds):
            for d in p.predictions:
                names.add(d.class_name)
                if d.class_name in ('ball', 'ball-in-basket'):
                    dets.append(dict(x=d.x + ox, y=d.y + oy, r=max(d.width, d.height) / 2,
                                     conf=float(d.confidence), cls=d.class_name))
        kept = []
        for d in sorted(dets, key=lambda d: -d['conf']):
            if all(math.hypot(d['x'] - k['x'], d['y'] - k['y']) > max(4, d['r']) for k in kept):
                kept.append(d)
        result.append(kept)
    json.dump(dict(detections=result, seconds=time.time() - t0, class_names=sorted(names)), open(out, 'w'))
    print(video, len(frames), 'frames', round(time.time() - t0, 1), 's')
''')

def rf_detect(clip, frames):
    data = json.loads((Path('/content') / f'rf_{clip}.json').read_text())
    return data['detections'], data['seconds']

if ROBOFLOW_API_KEY:
    venv = Path('/content/venv_rf')
    if not venv.exists():
        subprocess.run(['uv', 'venv', '-q', '--system-site-packages', '--python', sys.executable, str(venv)], check=True)
        if subprocess.run(['uv', 'pip', 'install', '-q', '--python', str(venv / 'bin/python'), 'inference-gpu']).returncode:
            print('inference-gpu failed to install; trying CPU package "inference" (slower)')
            subprocess.run(['uv', 'pip', 'install', '-q', '--python', str(venv / 'bin/python'), 'inference'], check=True)
    args = [str(venv / 'bin/python'), str(RF_SCRIPT)]
    for clip, path in CLIPS.items():
        args += [str(path), f'/content/rf_{clip}.json']
    code = subprocess.run(args, env=dict(os.environ, ROBOFLOW_API_KEY=ROBOFLOW_API_KEY)).returncode
    if code == 0:
        run_model('RF-DETR', rf_detect)
    else:
        print('RF-DETR detection failed, see the error above')
else:
    print('Skipped: add the ROBOFLOW_API_KEY secret (key icon on the left) and rerun this cell.')
"""),
    cell("markdown", r"""
## 3. SAM 3 video, text prompt "basketball"
Meta's model finds every object matching the text and follows each one through the video
using memory (it reads the whole clip first, so this is offline processing).
"""),
    cell("code", r"""
SAM3_PROMPT = 'basketball'

if HF_TOKEN:
    import inspect
    from huggingface_hub import login
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    login(token=HF_TOKEN)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    sam3 = Sam3VideoModel.from_pretrained('facebook/sam3').to('cuda', dtype=dtype).eval()
    sam3_processor = Sam3VideoProcessor.from_pretrained('facebook/sam3')

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
else:
    print('Skipped: accept terms at https://huggingface.co/facebook/sam3, add the HF_TOKEN secret, rerun this cell.')
"""),
    cell("markdown", r"""
## Side-by-side comparison and summary
Top-left original, then each model. Same frame, same moment.
"""),
    cell("code", r"""
import pandas as pd

for clip, path in CLIPS.items():
    frames, fps = read_video(path)
    scale = 640 / frames[0].shape[1]
    grid = []
    for i, f in enumerate(frames):
        panels = [cv2.resize(f, None, fx=scale, fy=scale)]
        for label in ('WASB', 'RF-DETR', 'SAM3'):
            r = RESULTS.get(label, {}).get(clip, {})
            if 'tracked' in r:
                panels.append(draw(f, r['detections'], r['tracked'], i, label, scale))
            else:
                blank = np.zeros_like(panels[0])
                cv2.putText(blank, f'{label}: not run', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                panels.append(blank)
        grid.append(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
    video = OUT / f'comparison_{clip}.mp4'
    write_video(video, grid, fps)
    print(clip)
    show(video, width=1100)

rows = [dict(model=label, clip=clip, **r.get('stats', {'error': r.get('error')}))
        for label, clips in RESULTS.items() for clip, r in clips.items()]
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'summary.csv', index=False)
display(summary)
display(Markdown('''
- **raw_found**: of the hand-marked check frames (8 in clip 1, 11 in clip 2), how many had *some* detection on the ball.
- **tracked_correct / tracked_wrong_place**: the tracked ball was on the real ball / somewhere else.
- The check points are a small draft set marked by eye; judge mainly by watching the videos.
'''))
print('Everything saved in', OUT)
"""),
]

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                        language_info=dict(name="python", version="3.12"),
                                        accelerator="GPU", colab=dict(gpuType="L4", provenance=[])),
                nbformat=4, nbformat_minor=5)
for index, item in enumerate(cells):
    item["id"] = f"ball-{index:02d}"
target = Path(__file__).resolve().parents[1] / "notebooks/colab_ball_benchmark.ipynb"
target.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {target}")
