# KaliCalib reference comparison — 2026-10-18

Executed locally on branch `homography`. Browser results are at
`outputs/homography-kalicalib-reference-v1/review.html`. This is a fixed-frame
comparison, not a new full-clip mini-map. Previous artifacts remain unchanged.

## What changed

Compared both supplied KaliCalib checkpoints (`model_test.pth` and
`model_challenge.pth`) on exactly the same 15 review frames:

- Clip 1: 0, 25, 50, 75, 100, 125, 149.
- Clip 2: 0, 25, 50, 75, 100, 125, 150, 174.

The adapted implementation follows the published winning-channel whole-mask
centroids, 35 px RANSAC threshold at 960×540, constrained OpenCV camera
refinement, and 100–1800 cm floor-span check. The original 91-point floor grid
is used; two elevated basket channels and background are excluded. RGB/ImageNet
normalization and both checkpoint architectures are identical to the first
experiment. There is no retraining, manual initialization, player inference,
temporal propagation, or tuning against manual clicks in this comparison.

Two explicit departures from upstream: a failed estimate stays null rather than
using the fixed mean-camera fallback; refinement and its probe coordinates use
960×540 consistently before scaling the matrix back to source pixels. The
published demonstration mixes working-resolution points and original image
dimensions for some inputs. Keeping a consistent coordinate system is necessary
to compare our 360p and 1080p sources. This is an adaptation of the published
pipeline, not a claim of bit-for-bit execution of its full dependency stack or
reproduction of its paper's benchmark results.

Upstream source:
[CEA-LIST/KaliCalib](https://github.com/CEA-LIST/KaliCalib), revision
`3b3f52b59e8e38e155ba7e9c681281d44748c173`, particularly
`kalicalib/estimateHomography.py`, `modeling/example_camera_model.py`, and
`data/datasets/viewds.py`. Declared license: CeCILL-2.1; the original license
is retained with the downloaded model source.

## Actual results

| Clip / metric | Test checkpoint | Challenge checkpoint |
|---|---:|---:|
| Clip 1: raw homography available | 7/7 samples | 7/7 samples |
| Clip 1: camera estimate accepted by upstream span check | 7/7 samples | 7/7 samples |
| Clip 1: mean checkpoint discrepancy after camera refinement | 29.41 px | 9.83 px |
| Clip 1: median checkpoint discrepancy after camera refinement | 19.87 px | 7.66 px |
| Clip 2: raw homography available | 8/8 samples | 8/8 samples |
| Clip 2: camera estimate accepted by upstream span check | 7/8 samples | 7/8 samples |
| Clip 2: discrepancy at the sole manual checkpoint, frame 0 | 112.70 px | 52.49 px |

Errors are in **native source pixels**: 640×360 for clip 1, 1920×1080 for clip 2.
They should not be compared directly across clips. Clip 1 uses the mean of seven
per-checkpoint means, against four draft points each and five at frame 149.
Court-axis reflections are selected separately at each checkpoint for evaluation
only; the reported agreement is optimistic and does not establish orientation.
The draft manual anchors are not surveyed ground truth or player-position labels.
No thresholds were selected using these errors.

Both checkpoints rejected clip 2 frame 50 after camera refinement. The raw
RANSAC fit is still displayed as a separate diagnostic there; it is not silently
substituted for an accepted camera. Available or accepted estimates do not imply
correct line alignment. The test checkpoint's clip 1 frame 25 is a conspicuous
failure despite passing the upstream span check (102.18 px mean discrepancy).

The challenge checkpoint is a more promising candidate on these samples. It
still visibly displaces court lines; its camera fit is not a validated automatic
solution. The refinement stage alone is a modest improvement for the challenge
model: clip 1 raw mean discrepancy 10.24 px versus refined 9.83 px, and clip 2
frame 0 raw 54.67 px versus refined 52.49 px. Merely increasing availability or
switching to the upstream threshold would not have solved the original problem.

All 30 model/frame evaluations completed in 38.92 seconds, including inference,
geometry, image/heatmap exports and incremental result writes, excluding initial
asset/reference preparation. This is CPU offline execution, not a live FPS
measurement. Four PyTorch threads, PyTorch 2.12.1, OpenCV 4.12.0, NumPy 2.2.6.

## Artifacts and verification

- `experiment.json`: all correspondence points, raw/refined matrices or null,
  camera diagnostics, per-stage timings, draft-click agreement, input/model/code
  hashes and aggregate summary.
- `run_source/`: exact source snapshots captured when the comparison began.
- Per clip: raw/refined overlays for each checkpoint, manual reference overlays,
  previous custom baseline overlays, original manual JSON and compressed model
  heatmaps for further geometry experiments without rerunning the network.
- `review.html`: clip/frame/stage selectors, both model overlays, manual and
  previous reference, and click-to-enlarge images.

All 24 CPU tests passed. Added tests check winning-region centroid parity,
exclusion of elevated basket channels, known-camera geometry, source-resolution
scaling, and null results instead of default-camera fallback. All 90 review JPEGs
decoded at 1280×720. These software/artifact checks do not establish model
accuracy.

Reproduce in a new directory from the repository root:

```powershell
.venv/Scripts/python.exe -m tacticalai.auto_calibration_cli prepare
.venv/Scripts/python.exe -m tacticalai.kalicalib_reference_cli --output outputs/my-kalicalib-reference
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The reference command fetches the pinned challenge checkpoint if it is absent.
It requires the prior baseline exports at `outputs/homography-auto-v1` for the
comparison panel. Original footage stays local and is never uploaded.

The next geometry experiment should refine the challenge candidate against
visible court-line evidence, with masks for moving people and score graphics,
and evaluate failures against additional independent manual landmarks. No new
player minimap should be presented as an accuracy improvement until alignment
and temporal stability have been checked.

## Follow-up: local painted-line refinement (executed)

This next step was also attempted on the same 30 cached checkpoint/frame
estimates, without rerunning either network. Latest browser artifact:
`outputs/homography-kalicalib-lines-v2/review-lines.html`. The default view
retains the camera estimates; the stage selector exposes the line experiment.
All previous results and the original reference page are preserved.

The venue-specific prototype extracts thin yellow paint using HSV bounds
`[18,25,100]..[45,255,255]`, subtracts a 9-pixel morphological opening to suppress
filled logos, masks existing person boxes and the bottom 15% score region, and
locally optimizes projected template points against the line distance transform.
Only template samples initially within 15 working pixels of detected evidence
participate. Image-corner control shifts are bounded to ±20 pixels at 960×540,
with regularization and a minimum 5% line-cost reduction. No manual anchor enters
initialization, segmentation, optimization or rejection.

The first attempt used a higher saturation cutoff and all visible template
samples: 0/30 accepted corrections. Inspection showed sparse paint evidence.
The second attempt lowered the saturation cutoff and excluded unsupported
template sections. This was a development iteration on these same clips, not a
held-out evaluation. It accepted only **2/30 corrections**: challenge clip 1
frame 0, and test clip 2 frame 25. The remaining cases were rejected at correction
bounds or for weak improvement (26), or lacked an initial camera (2). This is
not a broadly successful refinement method.

At challenge clip 1 frame 0, draft-click mean discrepancy changed from 6.53 px
to 4.69 px; max discrepancy from 10.01 px to 6.69 px. The other accepted case has
no manual checkpoint, so improvement there is unknown. No aggregate accuracy
improvement or new full-sequence coverage is claimed. Rejected refinements are
null in the line stage, with no hidden fallback to a camera estimate.

`line-experiment.json` records all corrections, rejection reasons, settings,
timings and source hashes. Line masks and overlay images are saved for every
supported initialization. SciPy 1.18.1 is provided by the optional `geometry`
extra. All **27** CPU tests passed, including a synthetic painted-court offset
recovery test and masks/missing-camera rejection. These tests validate the
implementation, not the real-footage model accuracy.

Run against a fresh copy of the reference results (the command refuses to
overwrite an existing line experiment):

```powershell
.venv/Scripts/python.exe -m pip install -e ".[geometry]"
.venv/Scripts/python.exe -m tacticalai.court_line_refinement_cli --reference outputs/my-kalicalib-reference
```

The measured outcome supports keeping the challenge checkpoint for further
research, but this simple colour/line optimizer is not ready to replace the
manual workflow. Better semantic court-line evidence and more independent
validation landmarks remain necessary.
