# Person coverage investigation — 21 October 2026

**Subsequent identity regression:** the lower-threshold candidate fragments and
duplicates the inspected clip-2 player as T38/T292, while the earlier online
tracker keeps T215. See [return audit and checkpoint recommendation](REID_REGRESSION_20261021.md).
The coverage gains below remain valid for their stated metric; they do not
justify replacing the earlier identity baseline.

Local review: `outputs/person-coverage-review-v1/index.html`.
Serve with `python -m http.server 8884 --bind 127.0.0.1 --directory outputs/person-coverage-review-v1`.

## Diagnosis

The old display showed confirmed tracks only. The detector threshold was 0.10,
but TrackTrack used high=0.60, low=0.25, new=0.70 and confirmation length 3.
People detected at moderate confidence could therefore disappear from the
display, especially during occlusion or while entering view. This is supported
by the saved raw measurements, not inferred from missing silhouettes.

An eight-frame inspection set contains 101 visible-person head points, including
25 marked partially visible. Labels are in
`configs/evaluation/person_coverage_points_v1.json`; they are evaluation-only and
never enter inference, filters, identity matching, rendering or map projection.
One-to-one box/point containment prevents a merged box from covering two people.
This is inspected point coverage, not bounding-box AP, precision, identity
accuracy or independent validation. Four frames are marked development and four
check; all are from these previously used clips. Annotation uncertainty remains.

| Stage | All inspected points | Partially visible points |
|---|---:|---:|
| Original tracks | 72 / 101 | 8 / 25 |
| Original raw detections | 99 / 101 | 24 / 25 |
| Lower tracker gates | 95 / 101 | 20 / 25 |
| Lower gates + conservative court filter | 95 / 101 | 20 / 25 |
| Retained detections, including those without IDs | 99 / 101 | 24 / 25 |

Tracker gates are high=0.25, low=0.10, new=0.25. Confirmation length 3,
TAI overlap 0.55, ReID yolo26n-cls, GMC sparseOptFlow and buffer 30 are unchanged.
Confirmation length 1 and TAI 0.80 added observations but no inspected-point
coverage, so those changes were not selected. Increased observations or unique
IDs do not establish improved precision or identity stability.

## Detector comparisons

YOLO26s and YOLO26m were tested on the same eight frames at 1280 and 1920,
confidence 0.03/0.05/0.10. An additional YOLO26s NMS-free 1280 probe was run.
Each probe records model hashes, versions, settings and runtime. At 0.10,
YOLO26m and higher resolution did not improve coverage over YOLO26s/1280.
At 0.03, YOLO26s covered 100 points rather than 99, but generated 559 raw boxes
instead of 246 across the sample (392 versus 209 after the conservative gate).
Partially visible coverage remained 24/25. Those extra boxes have not been
validated as on-court people. The default retains 0.10 rather than claiming
that flooding the display with weak boxes is improved person detection.

## Court filtering and limits

The filter runs before tracking, using saved automatic, smoothed KaliCalib
homographies. A box is removed only when its bbox-bottom floor point is outside
a generous court margin AND less than 10% of its body box overlaps the projected
court. Invalid/missing cameras and clipped boxes remain uncertain and are kept.
Estimated cameras get an additional 0.5 m margin.

A 1.5 m margin excluded three inspected boundary persons. The selected 3 m
margin retains all 99 points represented by the raw detector and removes 37
other boxes across the eight inspected frames. That is coverage evidence, not
proof those 37 boxes were all false positives. Across the complete clips it
removes 551 and 691 detections. Calibration and hidden feet can still misclassify
the foreground crowd, sideline staff and people near boundaries. There is no
verified floor segmentation mask or automatic role classifier; several real
spectators/staff remain. Geometry filtering is conservative, not perfect.

The court assumes 28 × 15 m. Projection uses bbox bottom center, with contact
visibility unknown. Clipped boxes are not projected. Camera smoothing uses
future frames. No hidden or off-screen detections are created.

## Executed full-clip outputs

All five tracker settings were replayed against identical saved detections on
each complete clip. A local control reproduces the original observation counts
(1,358 / 1,894). The replay does not restore pre-NMS recovery extras, so it is
not claimed to reproduce the original GPU run bit for bit. Crop embeddings are
shared across variants per frame; tracker and GMC states remain separate.

| Output diagnostic | Clip 1 | Clip 2 |
|---|---:|---:|
| Source / rendered frames per variant | 150 | 175 |
| Old confirmed track observations | 1,358 | 1,894 |
| New confirmed track observations | 2,529 | 2,987 |
| Additional detections awaiting IDs | 776 | 2,509 |
| Filtered outside observations | 551 | 691 |

These counts include duplicates and people outside the actual playing area;
they are not true-positive counts. New track IDs are local tracker labels,
not the old gallery-linked person hypotheses. Stable identity accuracy is not
established by this experiment. Lower gates can increase ID switches, short
tracks and false positives. The user can hide pending detections in the page.

YOLO26s-seg was rerun on all 325 frames at 1280, confidence 0.10, independently
of tracker confirmation. Missing masks keep labels. Tracking/re-identification
were rerun from saved detections; primary detector inference was rerun only on
the sampled detector probes. Full before/after H.264 videos, CSV, JSONL, mask
coverage, source hashes, configurations and measured times accompany the page.
The original videos and earlier experiments are preserved.

All four before/after MP4s were fully decoded at the expected frame counts.
Every retained output box/track assignment was compared with the replay input;
all artifact and source snapshot hashes were verified. All 67 CPU tests pass,
including null-ID retention, one-to-one coverage matching, conservative court
gating and exclusion of role labels. Real-browser checks covered both clips,
second-clip playback, frame stepping, synchronized comparison and the pending-ID
and boundary toggles. No browser console errors were observed. Browser payloads
simplify mask contours by at most 0.5 source pixels for load size; full mask
measurements remain in JSONL, and downloadable MP4s use the full contours.

## Reproduce

```powershell
.venv/Scripts/python.exe scripts/probe_detection_coverage.py --output outputs/person-coverage-investigation/detector-small
.venv/Scripts/python.exe scripts/probe_detection_coverage.py --model models/yolo26m.pt --output outputs/person-coverage-investigation/detector-medium
.venv/Scripts/python.exe scripts/probe_detection_coverage.py --sizes 1280 --nms-free --output outputs/person-coverage-investigation/detector-small-e2e
.venv/Scripts/python.exe scripts/replay_person_coverage.py --output outputs/person-coverage-investigation/tracker-replay-safe
.venv/Scripts/python.exe scripts/evaluate_person_coverage.py
.venv/Scripts/python.exe scripts/cache_person_masks.py --output outputs/person-coverage-investigation/masks-all
.venv/Scripts/python.exe -m tacticalai.coverage_presentation --output outputs/person-coverage-review-v1
```

Use fresh directories. Evaluation also reads the historical 1.5 m replay under
`tracker-replay`; that comparison is preserved from this investigation.
`configs/person_coverage.json` supplies the selected tracker settings for a
fresh `tacticalai-track` run, including Colab. The baseline runner itself does
not apply court filtering or retain unassigned detections in its MP4; use the
coverage replay/presentation pipeline for those behaviors. Videos, weights and
outputs remain ignored. Ultralytics package licensing remains AGPL-3.0;
model/dataset rights require review before commercial use.

Official tracker parameter semantics:
https://docs.ultralytics.com/modes/track/.
The executed behavior was checked against the locally pinned 8.4.158 source.
