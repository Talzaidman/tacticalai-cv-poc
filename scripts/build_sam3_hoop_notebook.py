"""Build the portable Colab driver from the reviewable Python source."""

from base64 import b64encode
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    value = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def main():
    bundle = BytesIO()
    with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
        for name in ("__init__.py", "export.py", "rim_probe.py", "sam3_hoop_cli.py"):
            path = ROOT / "src/tacticalai" / name
            archive.writestr(ZipInfo(f"tacticalai/{name}"), path.read_bytes(), ZIP_DEFLATED)
    encoded = b64encode(bundle.getvalue()).decode("ascii")
    cells = [
        cell("markdown", """
# SAM 3 basketball hoop, rim and backboard

Use a Colab GPU runtime and Run all. The default is `attack_clip2.mp4`, sampled
every five frames. SAM 3 tests three independent text prompts at confidence 0.30.
The annotated MP4 shows scored boxes and mask overlays. Review the footage to
verify what each prompt found. The counts describe model candidates.

Use the existing private Drive copy of the clip or set `USE_DRIVE = False`
in browser Colab to upload the single MP4 for this session. Set a private
`HF_TOKEN` Colab secret with approved access to [Meta SAM 3](https://huggingface.co/facebook/sam3).
In VS Code Colab, the cell can also prompt for the token with hidden input.
The source bundle below is generated from `src/tacticalai/sam3_hoop_cli.py`
and its small local dependencies.
"""),
        cell("code", """
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
import os
import subprocess
import sys

import torch
from google.colab import drive, files, userdata

CLIP_NAME = 'attack_clip2.mp4'
USE_DRIVE = True
SAMPLE_STRIDE = 5
CONFIDENCE = 0.30
PROMPTS = ['basketball hoop', 'metal basketball rim', 'basketball backboard']

if not torch.cuda.is_available():
    raise RuntimeError('Connect to a Colab GPU runtime before running this notebook.')
print('GPU:', torch.cuda.get_device_name(0))

try:
    token = userdata.get('HF_TOKEN')
except Exception:
    token = None
if not token:
    token = getpass('Hugging Face read token for facebook/sam3 (hidden): ').strip()
if not token:
    raise RuntimeError('A Hugging Face read token with access to facebook/sam3 is required.')

if USE_DRIVE:
    drive.mount('/content/drive')
    project = Path('/content/drive/MyDrive/TacticalAI')
    VIDEO = project / 'data/raw' / CLIP_NAME
    output_root = project / 'outputs'
else:
    VIDEO = Path('/content') / CLIP_NAME
    if not VIDEO.is_file():
        files.upload(target_dir=str(VIDEO.parent))
    output_root = Path('/content/sam3-hoop-outputs')
if not VIDEO.is_file():
    raise FileNotFoundError(f'Expected clip at {VIDEO}. Check CLIP_NAME and USE_DRIVE.')

stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
OUT = output_root / f'sam3-hoop-{VIDEO.stem}-{stamp}'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers==5.24.1', 'opencv-python==4.12.0.88',
                'imageio-ffmpeg==0.6.0'], check=True)
"""),
        cell("code", f"""
# Generated source bundle. Rebuild with scripts/build_sam3_hoop_notebook.py.
from base64 import b64decode
from io import BytesIO
from zipfile import ZipFile

WORK = Path('/content/tacticalai-sam3-hoop')
with ZipFile(BytesIO(b64decode({encoded!r}))) as bundle:
    bundle.extractall(WORK)
"""),
        cell("code", """
command = [sys.executable, '-u', '-m', 'tacticalai.sam3_hoop_cli',
           '--source', str(VIDEO), '--output', str(OUT),
           '--sample-stride', str(SAMPLE_STRIDE), '--conf', str(CONFIDENCE),
           '--prompts', *PROMPTS]
with subprocess.Popen(command, cwd=WORK, env=dict(os.environ, HF_TOKEN=token),
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, encoding='utf-8', errors='replace') as process:
    for line in process.stdout:
        print(line.replace(token, '[REDACTED]'), end='', flush=True)
    exit_code = process.wait()
if exit_code:
    raise RuntimeError(f'SAM 3 exited with code {exit_code}. See the error above.')
"""),
        cell("code", """
from IPython.display import Video, display

print((OUT / 'summary.json').read_text(encoding='utf-8'))
print('Artifacts:', OUT)
display(Video(str(OUT / 'annotated_samples.mp4'), embed=True, width=960))

# With USE_DRIVE=False, download the output bundle before disconnecting.
if not USE_DRIVE:
    import shutil
    archive = shutil.make_archive(str(OUT), 'zip', root_dir=OUT)
    files.download(archive)
"""),
    ]
    cells[2]["metadata"] = {"cellView": "form"}
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=4,
                    metadata=dict(accelerator="GPU", colab=dict(provenance=[]),
                                  kernelspec=dict(name="python3", display_name="Python 3"),
                                  language_info=dict(name="python")))
    path = ROOT / "notebooks/colab_sam3_hoop.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
