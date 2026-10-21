# Restored automatic identity baseline — 21 October 2026

The user accepted returning to the earlier setup after the T38/T292 continuity
regression. `configs/reid/classifier-n.json` now exactly matches the executed
`configs/reid/identity_checkpoint_20261019.json` profile. This replaces its later,
unexecuted buffer-45 setting. The unrelated original detector/tracker comparison
CLI default (`configs/baseline.json`) remains its historical baseline.

Current presentation: http://127.0.0.1:8887/ —
`outputs/restored-identity-baseline-v1/index.html`.
Both full clips use the earlier automatic person hypotheses (P IDs), with raw
tracker IDs in captions, colored silhouettes and a top-left court minimap.
Pending detections and boundary/uncertain cases default off. Their controls
affect display only; there is no new court prefilter in the restored tracker.
ID highlighting, observation timelines, first/last appearance, frame stepping,
slow playback and clean annotated MP4 downloads remain available.

The complete selected flow is YOLO26s (conf .10, IoU .70, imgsz 1280), TrackTrack
(high/low/new .60/.25/.70), online yolo26n-cls ReID, buffer 30, sparseOptFlow GMC,
then automatic CLIP-ReIdent temporal splitting/gallery matching. Automatic
YOLO26s-seg masks and KaliCalib camera/position smoothing are retained.
The presentation reuses the saved executed results. No inference was rerun.
Source artifact hashes and complete tracker/version metadata accompany each clip.

No manual role labels or identity corrections are used. Role/team/jersey/ball/
tactical classifications remain unknown. Identity and smoothing stages are
offline; thresholds were developed using feedback on these clips. Nominal court
dimensions and bbox floor-point approximations remain assumptions. This selection
does not establish independently measured identity or calibration accuracy.

Packaging verified source hashes, null role fields, per-frame identity uniqueness,
and full annotated-video decoding (150/175 frames). The lower-threshold coverage
experiment and prior outputs remain preserved separately.
The active effective tracker config was checked against both original executed
run metadata files and matched exactly. Browser checks covered both clips,
default-off filters, P87/P215 highlighting, last-observation jumps (f149/f174),
backward frame stepping and 0.5× playback without console errors.

To reproduce the presentation, use a fresh output folder:

```powershell
.venv/Scripts/python.exe scripts/restore_identity_presentation.py --output outputs/restored-identity-baseline-fresh
.venv/Scripts/python.exe scripts/serve_review.py outputs/restored-identity-baseline-fresh --port 8887
```

For future tracking runs, select the restored profile explicitly, for example:

```powershell
.venv/Scripts/python.exe -m tacticalai.cli --video data/raw/attack_clip.mp4 --output outputs/restored-tracking-fresh/attack_clip --config configs/reid/classifier-n.json --device cpu
```

Tracking exports alone do not run the offline identity stage. The existing
temporal feature and temporal ReID CLIs in [the identity documentation](TEMPORAL_IDENTITY_20261019.md)
provide that stage. The presentation packaging command consumes the already
verified saved complete pipeline results, not arbitrary new tracker outputs.
