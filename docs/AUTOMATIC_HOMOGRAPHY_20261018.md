# Automatic court-calibration experiment — 2026-10-18

Executed locally on branch `homography`, with no retraining and no footage
upload. This is an initial automatic baseline, not a replacement for the manual
calibration. The original videos and previous results are preserved.

Open `outputs/homography-auto-v1/review.html` in Chrome for the full-clip review.
It includes frame stepping, manual checkpoint buttons, synchronized manual
comparison, fullscreen playback, detected correspondences, temporal court fit,
an automatic mini-map, and the raw YOLO diagnostic. All ten H.264 videos were
fully decoded after rendering: five views of each original clip, at 25 FPS,
150 / 175 frames, 1280 × 720, without audio.

## Actual results

| Measurement | Clip 1 | Clip 2 |
|---|---:|---:|
| Source frames | 150 | 175 |
| Per-frame detector fits passing geometry gates | 26 (17.3%) | 22 (12.6%) |
| Hybrid frames with candidate calibration | 53 (35.3%) | 45 (25.7%) |
| Frames using floor-feature flow | 37 | 32 |
| Rejected detector fits: weak inlier support | 117 | 138 |
| Rejected detector fits: projective horizon crosses court | 7 | 15 |
| Mean KaliCalib CPU inference time | 426.8 ms/frame | 408.1 ms/frame |
| Mean YOLO court CPU inference time | 98.8 ms/frame | 74.4 ms/frame |
| Combined inference/fit experiment elapsed | 81.35 s | 90.89 s |

These counts measure availability under the experiment's gates, **not correct
calibration coverage**. Sampled automatic overlays are imperfect, including
line displacement. Candidate calibration was left null on unsupported frames.
The mini-map consequently has gaps; it does not silently use the manual
calibration or freeze previous positions. The manual process remains the
working reference.

Only one of clip 1's seven existing manual checkpoints had an available direct
automatic fit: frame 0. At its four draft clicks, the mean image discrepancy
was 10.36 px, maximum 15.53 px, after choosing the best court-axis reflection.
Clip 2's sole manual checkpoint had no accepted automatic fit. These references
are draft visual estimates, not annotated ground truth. Reflections are chosen
separately at each checkpoint for this optimistic diagnostic; they do not
resolve orientation for the automatic sequence or feed inference. This is not
a measured player-position error. No ground-truth accuracy or temporal jitter
metric has been established.

## Models and processing

- [YOLO11n basketball court keypoints](https://huggingface.co/koppolusameer/yolo11n-basketball-court-keypoints),
  revision `6cb899251439982067a81baa225b33f45f335981`, `best.pt`, 640 input,
  court detection confidence 0.25. The 48 raw keypoint slots are exported, with
  confidence >=0.3 displayed. First-frame predictions were visibly misplaced.
  Its published card requires a landmark-index mapping but does not supply a
  complete metric map. No guessed mapping is used to project players.
- [KaliCalib](https://github.com/CEA-LIST/KaliCalib), revision
  `3b3f52b59e8e38e155ba7e9c681281d44748c173`, `model_test.pth` and original
  `model_resnet.py`. RGB input 960 × 540, ImageNet normalization, CPU, four
  PyTorch threads. Floor-grid channel coordinates follow upstream
  `getFieldPoints()`, in metres. Channels 91/92 represent elevated baskets and
  are excluded; channel 93 is background. A grid correspondence need not be a
  visible physical line intersection.
- Experiment-specific heatmap extraction selects the largest connected region
  per floor channel with channel probability >=0.3, at least three output
  pixels, and winner across channels. These confidence values are not
  calibrated probabilities of correct registration.
- Experiment-specific RANSAC threshold: 8 px at 960-wide input, scaled to source
  pixels. Require >=8 inliers, >=0.4 inlier ratio, >=0.005 image support fraction,
  finite invertible fit, and no court-crossing projective horizon. This is
  **stricter and different postprocessing** than upstream's 35 px homography
  threshold and camera-model refinement. These results do not reproduce or
  refute KaliCalib's published benchmark performance.
- Hybrid: reset from automatic detection every five frames or after loss;
  propagate at most four frames using Lucas–Kanade floor features,
  forward/backward error <1.5 source pixels, RANSAC with >=12 flow inliers and
  >=0.6 ratio. Mask person boxes plus a five-pixel margin and the bottom 15%
  score-graphic region. A coarse mean-absolute-difference cut gate of 45 is
  experimental and cannot guarantee cut detection. No smoothing is used to
  disguise fit failures.

The combined experiment deliberately runs both detectors on every frame so
that the direct and periodic/flow strategies can be compared from the same
evidence. Its wall time is not deployed hybrid throughput. Rendering and model
loading are excluded from the per-frame detector timing. This was CPU offline
processing, not live inference at 25 FPS.

Exact installed versions: PyTorch 2.12.1, Ultralytics 8.4.158, OpenCV 4.12.0,
NumPy 2.2.6. The experiment manifest records weight/source hashes, parameters,
upstream URLs and licenses. Rendering records its own source hashes because
the viewer/rendering implementation was completed after inference began.

## Player observations and limits

No player inference was rerun. Reused YOLO26s / TrackTrack + ReID
(`yolo26n-cls.pt`), buffer 30, GMC `sparseOptFlow`, with exact source metadata
copied into each clip directory. Existing tracking errors remain, including
the confirmed clip 1 87-to-96 wrong-person association.

Clip 1 mini-map filters with the existing run-specific manual role labels.
Clip 2 shows person tracks with unverified roles. All person rows remain in
`positions.jsonl`, with null court coordinates when calibration is unavailable
or boxes are clipped. Ground contact is unverified; bbox bottom centre is the
footpoint approximation. Colors are tracker IDs, not teams. Court axes are the
model's convention and may differ from the manual map. No missing players,
hidden trajectories, or airborne ball locations are inferred.

## Reproduce

Run from the repository root in the existing environment (the `preview` extra
provides H.264 encoding). All processing is local; only public model assets
are downloaded. `models/` and outputs are ignored by Git.

```powershell
.venv/Scripts/python.exe -m tacticalai.auto_calibration_cli prepare
.venv/Scripts/python.exe -m tacticalai.auto_calibration_cli benchmark --output outputs/my-auto-court-run
.venv/Scripts/python.exe -m tacticalai.auto_calibration_cli render --output outputs/my-auto-court-run
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Use a new output directory; prior clip detections and rendered products are
refused. All 21 repository CPU tests passed, including synthetic projective
geometry with outliers, rejection/null projection, translated floor-feature
tracking, bounded flow lifetime and masking moving-person regions.

License metadata: KaliCalib CeCILL-2.1 (download includes original license),
YOLO model AGPL-3.0, training dataset export CC BY 4.0. This records declared
licenses; it is not commercial clearance.

## Recommended next experiment

Keep these failures as the first automatic baseline. Compare the upstream
KaliCalib extraction/refinement recipe and its alternate pretrained checkpoint
on fixed validation frames; then consider court-line refinement around a
candidate. Review additional independent landmarks across both clips before
choosing thresholds or claiming improvement. A verified YOLO skeleton/metric
mapping is needed before evaluating that model's homography. Retraining is
deferred until these pretrained options have been measured.
