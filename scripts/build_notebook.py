"""Regenerate the small Colab driver; no experiments run by this script."""
import json
from pathlib import Path


def cell(kind, source):
    value = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


cells = [
    cell("markdown", """
# TacticalAI: occlusion / ID continuity experiments
Select a **Colab GPU kernel** in VS Code or a GPU runtime in browser Colab.
Source code stays in Python modules; this notebook only stages and runs it.
Local source files do not automatically appear in Colab.

Before running: follow README's PowerShell archive command, privately upload the
source ZIP to `MyDrive/TacticalAI/code/` and originals to `MyDrive/TacticalAI/data/raw/`.
Run the staging cell again after source edits (it creates a fresh checkout).
Runs four variants on BOTH trackers and BOTH clips (16 full runs): baseline
(buffer 30, ReID off), buffer 60, ReID, and buffer 60 + ReID.
YOLO26s detector settings and sparseOptFlow GMC remain fixed. ReID uses explicit
yolo26n-cls.pt appearance embeddings; it is experimental, not verified identity.
Hidden positions remain unmeasured. Outputs and H.264 comparison grids go to Drive.
"""),
    cell("code", """
from google.colab import drive
drive.mount('/content/drive')
"""),
    cell("code", """
from pathlib import Path
from datetime import datetime, timezone
import uuid
import os
import zipfile

DRIVE_ROOT = Path('/content/drive/MyDrive/TacticalAI')
SOURCE_ZIP = DRIVE_ROOT / 'code/tacticalai-source.zip'
VIDEO_ROOT = DRIVE_ROOT / 'data/raw'
assert SOURCE_ZIP.is_file(), f'Upload code archive first: {SOURCE_ZIP}'
VIDEOS = [VIDEO_ROOT / name for name in ('attack_clip.mp4', 'attack_clip2.mp4')]
for video in VIDEOS:
    assert video.is_file(), f'Upload private footage first: {video}'

token = 'occlusion-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
REPO = Path('/content') / ('tacticalai-' + token)
REPO.mkdir()
with zipfile.ZipFile(SOURCE_ZIP) as archive:
    # Accept both Windows and POSIX archive separators.
    for member in archive.infolist():
        name = member.filename.replace('\\\\', '/')
        target = (REPO / name).resolve()
        assert target.is_relative_to(REPO.resolve()), 'Archive path outside checkout'
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
assert (REPO / 'pyproject.toml').is_file(), 'ZIP must contain src/ and pyproject.toml at root'
os.chdir(REPO)
OUTPUT_ROOT = DRIVE_ROOT / 'outputs' / token
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
probe = OUTPUT_ROOT / '.write-check'
probe.write_text('ok')
probe.unlink()
print('Remote checkout:', REPO)
print('Persistent output:', OUTPUT_ROOT)
"""),
    cell("code", """
import subprocess
import sys

def command(*args):
    subprocess.run([str(arg) for arg in args], check=True)

command(sys.executable, '-m', 'pip', 'install', 'setuptools==75.8.0')
command(sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt')
command(sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', '.')
command(sys.executable, '-m', 'pip', 'install', 'jedi>=0.16,<0.20', 'imageio-ffmpeg==0.6.0')
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
print(check.stdout, check.stderr)
(OUTPUT_ROOT / 'pip-check.txt').write_text(check.stdout + check.stderr)
if check.returncode:
    print('Shared Colab environment has dependency conflicts; inspect the report above.')
# Focused project checks and the real GPU smoke runs below are hard failure gates.
command(sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v')
"""),
    cell("markdown", """
If package installation requests a kernel restart, restart and rerun the staging
cell before continuing. The pip-check report is saved to Drive; focused tests
and both real ReID GPU smoke runs must pass before the full suite can start.
If Drive authorization is unavailable in the VS Code kernel, open this notebook in
browser Colab, mount Drive there and use the same workflow.
"""),
    cell("code", """
import torch
import ultralytics
import json
import shutil

assert torch.cuda.is_available(), 'Select a Colab GPU runtime before inference'
print('GPU:', torch.cuda.get_device_name(0))
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Ultralytics:', ultralytics.__version__)
assert ultralytics.__version__ == '8.4.158'
print((REPO / 'configs/baseline.json').read_text())
# Freeze the staged source archive beside outputs, before any experiment.
shutil.copy2(SOURCE_ZIP, OUTPUT_ROOT / 'source.zip')
"""),
    cell("code", """
# Real GPU ReID smoke tests for BOTH trackers before launching the full suite.
assert (REPO / 'scripts/run_occlusion.py').is_file(), 'Upload the updated source archive first'
smoke_config = json.loads((REPO / 'configs/baseline.json').read_text())
smoke_config.update(with_reid=True, track_buffer=60, reid_model='yolo26n-cls.pt')
smoke_path = OUTPUT_ROOT / 'smoke-config.json'
smoke_path.write_text(json.dumps(smoke_config, indent=2))
command(sys.executable, '-m', 'tacticalai.cli', '--video', VIDEOS[0],
        '--output', OUTPUT_ROOT / 'smoke', '--config', smoke_path, '--max-frames', '3')
for tracker in ('botsort', 'tracktrack'):
    smoke = json.loads((OUTPUT_ROOT / 'smoke' / tracker / 'metadata.json').read_text())
    assert smoke['status'] == 'complete' and smoke['frames_processed'] == 3
    assert smoke['tracker_config']['with_reid'] and smoke['reid_model_sha256']
    print(tracker, 'ReID smoke passed:', smoke['effective_track_buffer_frames'], 'retention frames')
"""),
    cell("code", """
# All 16 full runs, artifact validation, H.264 grids, and a single results ZIP.
# Do not rerun this cell in the same OUTPUT_ROOT: runs never overwrite outputs.
command(sys.executable, 'scripts/run_occlusion.py', '--video-root', VIDEO_ROOT,
        '--output', OUTPUT_ROOT)
"""),
    cell("code", """
from IPython.display import Video, display
print((OUTPUT_ROOT / 'suite-review.md').read_text())
for path in sorted((OUTPUT_ROOT / 'comparisons').glob('*.mp4')):
    print(path.name)
    display(Video(str(path), embed=True, width=960))
print('Persistent outputs:', OUTPUT_ROOT)
print('Drive review/download bundle:', OUTPUT_ROOT / 'occlusion-results.zip')
"""),
]
cells.extend([
    cell("markdown", """
## Review results
Each comparison is synchronized: top left baseline; top right buffer 60;
bottom left ReID; bottom right buffer 60 + ReID. IDs are local to each run.
Compare the same player before and after an occlusion, not numeric IDs across panels.
Counts do not establish fewer ID switches; manual review/ground truth is required.

The suite writes suite-review.json, suite-review.md, previews.json,
comparisons/*.mp4 and occlusion-results.zip directly into Drive.
Tell Codex the output folder/token; it can retrieve these with the Drive MCP
and show videos in chat. No public sharing is needed.

To rebuild reports/previews after inference without rerunning the models, run:
`command(sys.executable, 'scripts/run_occlusion.py', '--video-root', VIDEO_ROOT,
         '--output', OUTPUT_ROOT, '--review-only')`
"""),
])
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                        language_info=dict(name="python", version="3.12")),
                nbformat=4, nbformat_minor=5)
for index, item in enumerate(cells):
    item["id"] = f"baseline-{index:02d}"
target = Path("notebooks/colab_baseline.ipynb")
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
