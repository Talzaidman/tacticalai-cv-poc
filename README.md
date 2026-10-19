# TacticalAI CV POC — YOLO26 baseline

Combined tracking + identity + court-map checkpoint:
[complete presentation](docs/IDENTITY_COURT_PRESENTATION.md),
`outputs/identity-court-complete-final/index.html`. It shows all current player
tracks by default, with all-person and original/person-ID display options,
annotated MP4 downloads, positions and embedded matching evidence. Automatic
ID-renaming checks pass on both clips. Manual role labels remain display inputs.

Latest identity development trial: [temporal identity consistency](docs/TEMPORAL_IDENTITY_20261019.md).
The new layer splits mixed tracker IDs and reconnects person-consistent portions.
It recovers the four reported person chains and retains all six confirmed pair
connections, while keeping both reviewed negative original-player pairs separate.
Full-clip counts are **18→16 / 18→14**, with 30 / 0 unknown identity observations.
Review `outputs/temporal-reid-final/index.html` through a local HTTP server.
ID 89 has an additional proposed change that needs review. These are development
results, not held-out or frame-level accuracy. Original tracking is preserved.

Earlier [gallery improvements and ablations](docs/GALLERY_IMPROVEMENTS_20261019.md)
remain available under `outputs/gallery-improvements-v2/candidate/index.html`.

Earlier identity experiment: [whole-ID gallery matching](docs/GALLERY_REID.md),
preserving the original 18 IDs and proposing 18→16 / 18→15 identities.
Open `outputs/gallery-reid-final/index.html` through a local HTTP server for the
clearer before/after review. Accuracy remains unverified.

Player ReID work on branch `player-reid`: [model candidates, ready configs and
occlusion/matching experiments](docs/PLAYER_REID.md). Dedicated encoders have
not yet been benchmarked on our footage.

Basketball-trained CLIP-ReIdent now has a separate [offline reconnect experiment](docs/CLIP_REIDENT.md)
using saved tracklets, review crops and provisional persistent identity labels.
Both clips have been processed locally. Open `outputs/clip-reident-paper-v2/index.html`
to review the first results; correct-recovery accuracy has not been measured.

Latest preview adds robust Kalman/RTS camera and player-position smoothing:
`outputs/challenge-player-smoothed-v1/index.html`, with original/smoothed and
frame-synchronized comparison views. See [smoothing method and results](docs/PROJECTION_SMOOTHING.md).

The current two-clip browser presentation uses the KaliCalib challenge checkpoint
with existing TrackTrack + ReID observations: player boxes and a top-left
projected mini-map. Open `outputs/challenge-player-presentation-v1/index.html`.
See [presentation details and commands](docs/CHALLENGE_PRESENTATION.md).

Manual court calibration is now available on branch `homography`:
[workflow and commands](docs/HOMOGRAPHY.md). Clip 1 now has a full 6-second
top-left mini-map preview from seven draft manual keyframes with interpolated
anchors. Clip 2 retains its single-frame draft. Measured accuracy is pending.
This workflow reuses saved TrackTrack + ReID observations and runs on CPU.

The first automatic court baseline has also run on both full clips:
[results, limitations and reproduction](docs/AUTOMATIC_HOMOGRAPHY_20261018.md).
Open `outputs/homography-auto-v1/review.html` for browser comparison of automatic
anchors, floor-feature tracking, mini-map and the manual reference. Automatic
coverage is incomplete; it does not yet replace manual calibration.

Current work and deferred improvements are tracked in
[docs/PROJECT_TASKS.md](docs/PROJECT_TASKS.md), including court boundary control,
jersey/role validation and the confirmed ID 87-to-96 failure. TrackTrack + ReID
with buffer 30 is the current development candidate; accuracy remains unverified.

## Occlusion experiment (current notebook)

`notebooks/colab_baseline.ipynb` now drives a 16-run retention/ReID experiment:
both original clips, BoT-SORT and TrackTrack, and four variants (buffer 30/ReID
off, buffer 60/off, buffer 30/on, buffer 60/on). Detector settings stay fixed.
The default CLI config remains the original baseline. ReID is now explicitly
supported with `with_reid`, `track_buffer`, and `reid_model` in a JSON config.
The notebook uses `yolo26n-cls.pt` as a generic appearance encoder, not a
basketball identity model. Original experiments and footage are preserved.

Regenerate the source archive with `scripts/prepare_colab.ps1` after edits.
Stage the updated archive in Drive before running the notebook from the top.
It runs real three-frame ReID smoke checks on both trackers before full runs.
Source, configs, encoder hashes, effective retention, versions, exports and
videos are saved under `MyDrive/TacticalAI/outputs/occlusion-<token>/`.
`suite-review.json` validates all 16 runs and compares detector-output hashes;
it does not measure ID switches. `comparisons/` contains four synchronized H.264
grids (baseline top left, buffer 60 top right, ReID bottom left, both bottom
right). `occlusion-results.zip` packages the outputs for Drive MCP retrieval.
The final notebook cell displays the videos. `--review-only` on
`scripts/run_occlusion.py` rebuilds reports/videos without inference.

Hidden players have no measured positions. Evaluate the same manually reviewed
occlusions before claiming improved identity continuity. ReID may incorrectly
match teammates, and longer retention may reconnect an old ID to another person.

Modular person detection and tracking with YOLO26s, BoT-SORT and TrackTrack. Develop in VS Code and execute on a Colab GPU. The first baseline inference has completed; see `docs/BASELINE_RESULTS_20261015.md`. Accuracy evaluation remains pending.

## How to use

1. Open this extracted folder in VS Code.
2. Open the Codex extension's chat for this workspace.
3. Paste the text under `START_HERE.md` as the first prompt, or simply ask Codex: **"Read START_HERE.md, AGENTS.md, and docs/PROJECT_CONTEXT.md, inspect the original POC brief and both videos, then implement the first baseline."**
4. Develop in VS Code; run GPU jobs in Colab. **Do not assume local files automatically appear in Colab**. Sync code via Git / the VS Code Colab extension or upload, and stage videos separately.
5. Keep original footage private. This package contains the original two video files and the PDF.

## Files

- `AGENTS.md` — persistent, concise Codex repository instructions.
- `START_HERE.md` — first Codex prompt.
- `docs/PROJECT_CONTEXT.md` — detailed summary and suggested plan.
- `docs/POC_CV.pdf` — founders' original POC document.
- `data/raw/*.mp4` — supplied basketball footage.

The original handoff contained no inference results. Subsequent baseline runs are documented separately; no model training or tactical scoring has been performed.

## Execute the baseline

Open [notebooks/colab_baseline.ipynb](notebooks/colab_baseline.ipynb). Cells explicitly stage private code, mount Drive, check CUDA, install dependencies and run both trackers on both clips into persistent storage. Configure paths before inference.

This folder is not currently a Git repository. The prepared upload folder is
`outputs/colab-upload/TacticalAI/`. Upload that **TacticalAI folder** into Google
Drive's My Drive; it contains the source ZIP and verified copies of both videos
at the paths expected by the notebook. Keep it private.

After editing source, regenerate the archive and upload folder locally:

```powershell
./scripts/prepare_colab.ps1
```

Upload it privately to `MyDrive/TacticalAI/code/tacticalai-source.zip`. Upload the original videos separately to `MyDrive/TacticalAI/data/raw/`. Recreate/upload the archive after source changes and restage into a new runtime directory. Alternatively use a private Git remote and explicitly pull changes in Colab; keep credentials in runtime secrets. Never publish source footage.

With Google's Colab VS Code extension, open the notebook and select a Colab GPU kernel. Cells run remotely; local `.py` edits do **not** automatically synchronize. Use the same staging cells or pull the private repository remotely. Browser Colab uses the same notebook. See [Google's extension announcement](https://developers.googleblog.com/en/google-colab-is-coming-to-vs-code/).

From a staged checkout with Python 3.10+ (3.12 recommended):

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m unittest discover -s tests -v
python -m tacticalai.cli --video data/raw/attack_clip.mp4 --output outputs/clip1-first
python -m tacticalai.cli --video data/raw/attack_clip2.mp4 --output outputs/clip2-first
```

In Colab use an absolute Drive output path as shown in the notebook. Output directories must be new. Each comparison gets fresh model/tracker state. Use `--tracker botsort`, `--tracker tracktrack`, optional `--tracker deepocsort`, or `--tracker none` for detection only. A local CPU smoke run uses `--device cpu --max-frames 3`; first use downloads weights. Direct dependencies are pinned; Colab's CUDA-compatible Torch/Torchvision are retained and their exact versions recorded with `pip freeze`. This is not a fully locked Colab image.

## Artifacts and meaning

- `annotated.mp4`: original resolution/FPS/frame count, no audio. Uses mp4v; some browser players require H.264 conversion. Runner decodes it after writing to verify frame count.
- `frames.jsonl`: one JSON object per frame, even empty frames, containing raw `detections` and confirmed `tracks`. Timestamps are zero-based frame index / source FPS (CFR clips).
- `detections.csv`: primary detector person boxes before association, empty IDs. `tracks.csv`: confirmed tracker observations with IDs. Coordinates are original-image pixel XYXY. These populations differ.
- `metadata.json`: status, source/model SHA256, settings, package versions, hardware, full tracker config, processed frames and timings. Use only runs with `status=complete`; failed runs preserve partial artifacts.
- `tracker.yaml`: exact resolved tracker settings; the original baseline has ReID false and GMC sparseOptFlow. The occlusion suite explicitly varies ReID and retention. `environment.txt`: full resolved pip freeze.

COCO `classes=[0]` detects **people**, including referees, staff and spectators. Player-only filtering is deferred; role/team/jersey/court position remain unknown. Track IDs are run-local labels, not player identities. No off-screen predictions are exported.

[configs/baseline.json](configs/baseline.json) sets YOLO26s, confidence 0.10, IoU 0.70, imgsz 1280, FP32 and seed 0. Low confidence admits low-score recovery candidates but may add crowd detections. Association thresholds retain the pinned upstream defaults and differ between algorithms. Inference resize/letterbox does not change saved video resolution.

Both trackers receive identical input, weights and inference settings. This compares repeated inference, not a frozen-detection association ablation. TrackTrack can recover secondary candidates internally; raw exports contain primary boxes before association/recovery, so recovered candidates may appear only in tracks. A strict association study needs a shared detection cache and a defined recovery policy. GPU nondeterminism may remain.

Timing includes first-call warmup. Pipeline FPS includes decode, inference/association, overlays, exports and encoding. Model-call FPS includes preprocessing and tracking, not just the forward pass. Setup/download and final MP4 verification are excluded from loop FPS and included in setup-and-processing seconds. Drive I/O affects throughput; compare on identical hardware/storage. Unique ID count is not an ID-switch count; accuracy stays null until annotated.

See [evaluation checklist](docs/EVALUATION.md) and [asset inspection](docs/ASSET_INSPECTION.md). Ultralytics 8.4.158's downloaded wheel was inspected for tracker files and callback API; see [official tracking docs](https://docs.ultralytics.com/modes/track/). Review [Ultralytics licensing](https://www.ultralytics.com/license) and model/data licenses before commercial use. Court mapping, ball tracking, identity verification and tactics remain later work.

## Verification performed

After a Colab experiment, run the final **Validate and package the completed experiments**
notebook cell. It checks all four expected runs, frame/timestamp and export row counts,
and matching detector/environment settings. It writes `review.md`, `review.json` and
`baseline-results.zip` in that run's Drive output directory. The cell is self-contained
and does not require re-uploading source or rerunning completed inference. Download the
ZIP into local `outputs/`, extract into a new directory, and inspect it locally. The
same validation is available as `python scripts/review_runs.py <extracted-run-directory>`.
Artifact checks do not establish player accuracy or identity stability.

Five CPU tests passed: schema/timestamps/null IDs, CSV export, file hashing, real MP4 encode/decode, and runner completion/failure with simulated inference (including empty frames and detections removed by association). Notebook JSON and Python cells parse. Both originals decoded completely. Actual YOLO/Ultralytics tracker inference and the remote Colab workflow remain unexecuted; the notebook's three-frame GPU smoke cell is the next validation gate before full runs. No model-performance numbers are reported.

Two additional CPU checks passed for result review: missing-run reporting/packaging,
and complete exports versus a mismatched comparison configuration. Remote results
have not yet been retrieved for local review.
