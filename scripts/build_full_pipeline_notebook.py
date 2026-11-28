"""Build the single-video Colab driver for the automatic POC pipeline."""
import json
from pathlib import Path


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


cells = [
    cell("markdown", r"""
# TacticalAI unified automatic pipeline - configurable video

Set `CLIP_NAME` once, select a Colab GPU runtime, and run all cells. The notebook
never loads manual roles, calibration, identity corrections, or measurements.
Outputs persist under `MyDrive/TacticalAI/outputs/unified-<clip>-<timestamp>/`.

Stages: TrackTrack person tracking with ReID and sparse-optical-flow GMC, automatic
KaliCalib court estimation, visible-person court projection, segmentation silhouettes,
automatic role/team hypotheses, SAM 3 basketball detection/tracking, wrist-aware
holder estimation, sparse SAM 3 hoop and backboard detection with full-rate
refinement around possible shots, and one final
annotated MP4 with inferred passes and shots plus JSONL/CSV/runtime metadata.
`basket_hardware` in `frames.jsonl`
holds measured image-space boxes and clip-local, same-class track IDs.
`basket_hardware_sampled` indicates whether SAM 3 was run on a frame.
IDs can restart after missed frames.
Unknown measurements remain null.

Ball holder/controller is a temporal wrist/body-proximity estimate, not verified
possession. Only observed SAM 3 ball measurements can establish or change a holder.
Airborne ball image coordinates are never projected through the floor homography.
Track IDs/person hypotheses are not verified roster identities.
Dashed magenta lines connect an inferred shooter to the matched basket on the
mini-map. `shot_events.json` records possible hoop/backboard contacts, last-holder
evidence and unknowns. Contact and shooter are inferred offline, and outcome is unknown.
"""),
    cell("code", r"""
from google.colab import drive, userdata
from pathlib import Path
from datetime import datetime
import json, os, shutil, subprocess, sys, zipfile

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/TacticalAI')
VIDEO_ROOT = ROOT / 'data/raw'
CLIP_NAME = 'attack_clip2'  # Validate on this 7-second clip, then switch to 'galil_elion_offense_30s'.
SHOT_MAX_HOLDER_AGE_S = 1.25  # Seconds from last measured control to possible basket contact.
HOOP_STRIDE = 5  # Use 1 for the full-rate boxes-only reference.
VIDEO = VIDEO_ROOT / f'{CLIP_NAME}.mp4'
ARCHIVE = ROOT / 'code/tacticalai-source-adaptive-hoop.zip'
MODEL_ROOT = ROOT / 'models'
assert VIDEO.is_file(), VIDEO

required_source = {f'src/tacticalai/{name}'
                   for name in ('sam3_hoop_cli.py', 'hoop_sampling.py', 'hoop_tracking.py', 'shot_events.py')}
archive_candidates = (ARCHIVE,)

def current_archive():
    for path in archive_candidates:
        try:
            with zipfile.ZipFile(path) as archive:
                if required_source.issubset(archive.namelist()):
                    return path
        except (OSError, zipfile.BadZipFile):
            continue
    return None

source_archive = current_archive()
if source_archive is None:
    print('Source archive missing or outdated in the current Drive mount. Refreshing the mount.')
    drive.mount('/content/drive', force_remount=True)
    source_archive = current_archive()
if source_archive is None:
    raise RuntimeError(f'Could not find the adaptive hoop source ZIP at {ARCHIVE}. '
                       'Check that Colab mounted the Google account containing TacticalAI.')
print('Using source archive:', source_archive.name)

WORK = Path('/content/tacticalai')
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir()
subprocess.run(['unzip', '-q', '-o', str(source_archive), '-d', str(WORK)], check=True)
os.chdir(WORK)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation', '-e', '.[roles]'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.24.1', 'accelerate'], check=True)

import torch, ultralytics
from ultralytics import YOLO
assert torch.cuda.is_available(), 'Select a Colab GPU runtime before Run all'
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| ultralytics', ultralytics.__version__)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
for name in ('yolo26s.pt','yolo26n-cls.pt','yolo26s-seg.pt','yolo26n-pose.pt'):
    destination = MODEL_ROOT/name
    if not destination.is_file():
        print('Downloading official Ultralytics weight:', name)
        loaded = YOLO(name)
        source = Path(loaded.ckpt_path)
        shutil.copyfile(source, destination)
    assert destination.is_file() and destination.stat().st_size > 100_000, destination

def stage_complete(path):
    manifest = path / 'experiment.json'
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text()).get('status') == 'complete'
    except (OSError, ValueError):
        return False

# Reuse completed person/ball stages and resume a matching hardware checkpoint.
HOOP_STAGE = f'sam3-hoop-boxes-stride-{HOOP_STRIDE}'
FINAL_STAGE = f'06-final-shots-boxes-stride-{HOOP_STRIDE}'
if SHOT_MAX_HOLDER_AGE_S != 1.25:
    FINAL_STAGE += f'-holder-{SHOT_MAX_HOLDER_AGE_S}s'
output_root = ROOT / 'outputs'
output_root.mkdir(parents=True, exist_ok=True)
candidates = [path for path in sorted(output_root.glob(f'unified-{CLIP_NAME}-*'), reverse=True)
              if stage_complete(path/'01-foundation') and stage_complete(path/'02-segmented')]
RUN = next((path for path in candidates if (path/HOOP_STAGE/CLIP_NAME/'experiment.json').is_file()),
           candidates[0] if candidates else None)
if RUN is None:
    RUN = output_root / (f'unified-{CLIP_NAME}-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    RUN.mkdir()
    print('Starting persistent output:', RUN)
else:
    print('Resuming persistent output:', RUN)

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass('Hugging Face read token for facebook/sam3 (hidden): ').strip()
if not HF_TOKEN:
    raise RuntimeError('HF_TOKEN is required for SAM 3')
os.environ['HF_TOKEN'] = HF_TOKEN
"""),
    cell("code", r"""
# Stage 1: fresh automatic tracks, camera/court, and projection.
foundation = RUN / '01-foundation'
if stage_complete(foundation):
    print('Reusing completed foundation stage')
else:
    cmd = [sys.executable, '-m', 'tacticalai.ball_cli',
           '--videos', str(VIDEO_ROOT), '--clips', CLIP_NAME, '--output', str(foundation),
           '--skip-ball',
           '--person-model', str(MODEL_ROOT/'yolo26s.pt'), '--person-tracker', 'tracktrack',
           '--person-reid', '--reid-model', str(MODEL_ROOT/'yolo26n-cls.pt'),
           '--person-imgsz', '640', '--person-conf', '0.20', '--person-iou', '0.70',
           '--person-track-high-thresh', '0.35', '--person-track-low-thresh', '0.15',
           '--person-new-track-thresh', '0.35', '--person-min-track-len', '2',
           '--person-track-buffer', '50',
           '--device', '0', '--imgsz', '1280', '--tiles', '--person-crops',
           '--automatic-court', '--court-models', str(MODEL_ROOT/'kalicalib')]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Foundation stage failed with exit code {result.returncode}; see output above')
print(json.loads((foundation/'experiment.json').read_text())['status'])
"""),
    cell("code", r"""
# Stage 2: fresh person segmentation; one-to-one mask/track association.
segmented = RUN / '02-segmented'
if stage_complete(segmented):
    print('Reusing completed segmentation stage')
else:
    result = subprocess.run([sys.executable, '-m', 'tacticalai.silhouette_presentation',
        '--source', str(foundation), '--output', str(segmented),
        '--model', str(MODEL_ROOT/'yolo26s-seg.pt'), '--device', '0',
        '--videos', str(VIDEO_ROOT), '--clips', CLIP_NAME, '--cache-only'],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Segmentation stage failed with exit code {result.returncode}; see output above')
print(json.loads((segmented/'experiment.json').read_text())['status'])
"""),
    cell("code", r"""
# Stage 3: automatic generic-CLIP role hypotheses and anonymous team colors.
roles = RUN / '03-roles-teams'
if not stage_complete(roles):
    if roles.exists():
        shutil.rmtree(roles)
    result = subprocess.run([sys.executable, '-m', 'tacticalai.role_team_cli',
        '--source', str(segmented), '--videos', str(VIDEO_ROOT), '--clips', CLIP_NAME,
        '--config', 'configs/role_team.json', '--model-cache', str(MODEL_ROOT/'role-clip'),
        '--device', 'cuda:0', '--output', str(roles)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Role/team stage failed with exit code {result.returncode}; see output above')
else:
    print('Reusing completed role/team stage')
print(json.loads((roles/'experiment.json').read_text())['status'])
"""),
    cell("code", r"""
# Stage 4: SAM 3 basketball detection and native video tracking.
sam3_ball = RUN / '04-sam3-ball'
if not stage_complete(sam3_ball):
    if sam3_ball.exists():
        shutil.rmtree(sam3_ball)
    result = subprocess.run([sys.executable, '-m', 'tacticalai.sam3_ball_cli',
        '--videos', str(VIDEO_ROOT), '--clips', CLIP_NAME, '--output', str(sam3_ball),
        '--prompt', 'basketball'],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(f'SAM 3 stage failed with exit code {result.returncode}; see output above')
else:
    print('Reusing completed SAM 3 stage')
print(json.loads((sam3_ball/'experiment.json').read_text())['status'])
"""),
    cell("code", r"""
# Stage 5: pose keypoints near the ball for conservative wrist/body association.
poses = RUN / '05-ball-poses'
if not stage_complete(poses):
    if poses.exists():
        shutil.rmtree(poses)
    result = subprocess.run([sys.executable, '-m', 'tacticalai.pose_cache_cli',
        '--people-results', str(roles), '--ball-results', str(sam3_ball),
        '--ball-prefix', 'SAM3-Native', '--videos', str(VIDEO_ROOT), '--clips', CLIP_NAME,
        '--model', str(MODEL_ROOT/'yolo26n-pose.pt'), '--device', '0', '--output', str(poses)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Pose stage failed with exit code {result.returncode}; see output above')
else:
    print('Reusing completed pose stage')
print(json.loads((poses/'experiment.json').read_text())['status'])
"""),
    cell("code", r"""
# Detect coarse basket boxes and fill in frames around possible approaches.
hoop_root = RUN / HOOP_STAGE
hoop = hoop_root / CLIP_NAME
if not stage_complete(hoop):
    cmd = [sys.executable, '-u', '-m', 'tacticalai.sam3_hoop_cli',
           '--source', str(VIDEO), '--output', str(hoop),
           '--sample-stride', str(HOOP_STRIDE), '--boxes-only',
           '--prompts', 'basketball hoop', 'basketball backboard']
    if HOOP_STRIDE > 1:
        cmd.extend(['--ball-results', str(sam3_ball)])
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace') as process:
        try:
            for line in process.stdout:
                print(line.replace(HF_TOKEN, '[REDACTED]'), end='', flush=True)
        except KeyboardInterrupt:
            process.terminate()
            raise
        if process.wait():
            raise RuntimeError('SAM 3 hardware detection failed. See output above.')
else:
    print('Reusing completed SAM 3 hardware detection stage')
print((hoop/'summary.json').read_text())
"""),
    cell("code", r"""
# Stage 6: combine people, ball, holder estimates and hardware tracks, and infer passes/shots.
final = RUN / FINAL_STAGE
if not stage_complete(final):
    if final.exists():
        shutil.rmtree(final)
    result = subprocess.run([sys.executable, '-m', 'tacticalai.ball_fusion_cli',
        '--people-results', str(roles), '--ball-results', str(sam3_ball),
        '--ball-prefix', 'SAM3-Native', '--pose-results', str(poses),
        '--hoop-results', str(hoop_root),
        '--shot-max-holder-age-s', str(SHOT_MAX_HOLDER_AGE_S),
        '--videos', str(VIDEO_ROOT), '--clips', CLIP_NAME, '--output', str(final),
        '--association', 'temporal', '--player-presentation', 'silhouettes'],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Fusion stage failed with exit code {result.returncode}; see output above')
else:
    print('Reusing completed fusion stage')
print(json.loads((final/'experiment.json').read_text())['status'])
"""),
    cell("code", r"""
# Validate required artifacts and display the final native-controls video.
from IPython.display import Video, display
clip_output = final/CLIP_NAME
required = [final/'experiment.json', clip_output/'metadata.json', clip_output/'frames.jsonl',
            clip_output/'ball_possession.csv', clip_output/'annotated.mp4', clip_output/'shot_events.json']
for path in required:
    assert path.is_file() and path.stat().st_size > 0, path
manifest = json.loads((final/'experiment.json').read_text())
metadata = json.loads((clip_output/'metadata.json').read_text())
assert manifest['status'] == metadata['status'] == 'complete'
assert metadata['annotated_frames_decoded'] == metadata['video']['frames']
summary = dict(run=str(RUN), video=metadata['video'], counts=metadata['counts'],
               processing_s=metadata['processing_s'], limitations=manifest['limitations'])
(RUN/'run-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
display(Video(str(clip_output/'annotated.mp4'), embed=True, width=960, html_attributes='controls'))
""")
]

notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "accelerator": "GPU"}, "cells": cells}
Path('notebooks/colab_full_pipeline.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf8')
