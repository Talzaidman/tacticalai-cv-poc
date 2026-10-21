# Colored silhouette review

Executed locally on 21 October 2026. Both output MP4s were fully decoded:
150 and 175 frames. Associated-mask coverage was 1,305/1,358 and 1,839/1,894
observations (53 and 55 missing masks). Segmentation model calls, including
first-call warmup, took 43.71 s and 48.13 s on CPU, about 3.43 and 3.64 FPS.
All 61 CPU tests passed. Source fields, artifact hashes and snapshots were
verified. Browser checks covered both clip selections, second-clip playback,
and switching between silhouettes and boxes; representative renders were
visually inspected. These checks do not establish identity or mask accuracy.

The local presentation is `outputs/identity-court-silhouettes-v1/index.html`.
It extends `identity-court-complete-final` with display-only YOLO26s-seg masks.
Tracking, identity hypotheses, manually reviewed player/referee/staff roles,
camera estimates and floor projections are preserved. No team, jersey, ball
or tactical classification is added.

Both complete clips offer translucent silhouettes without bounding boxes by
default, person and raw tracker IDs, role labels, a top-left player minimap,
a complete current-person roster, frame stepping, playback speed controls,
H.264 downloads, JSONL/CSV exports and the existing identity evidence.
Choose bounding boxes or labels only in the overlay selector to change style.
The download uses silhouettes and all tracked people; the live page can filter
to players. Only player floor positions appear in the downloaded minimap.

Segmentation runs separately on each original frame, class 0 only, confidence
0.2, image size 640, NMS IoU 0.7 and full-resolution masks. One-to-one assignment
matches segmentation boxes to the saved track boxes with minimum IoU 0.3.
Colors follow existing person hypotheses; gray means unknown identity.
Unmatched observations keep their label with a null mask. There is no rectangle,
interpolated body or mask propagation fallback. Crowded overlaps can still
associate incorrectly; mask coverage does not measure segmentation accuracy.
Rendering uses mask polygons with alpha 0.4. Floor positions still use the
original bbox bottom centers, not mask-derived contact points.

Every output frame retains all original track fields and adds `silhouette_xy`,
`silhouette_confidence` and `silhouette_box_iou`. Metadata records weights hash,
library versions, settings, measured segmentation runtime, mask coverage, full
MP4 decode counts and hashes. Playback FPS and segmentation inference FPS are
different measurements. The original tracker pipeline is not benchmarked again.

Ultralytics 8.4.158 package metadata declares AGPL-3.0. Weights are downloaded
from the official Ultralytics assets release. Model/dataset licensing requires
review before commercial use; this local research output grants no new rights.

Reproduce from the repository root (use a fresh output directory):

```powershell
.venv/Scripts/python.exe -m tacticalai.silhouette_presentation --output outputs/identity-court-silhouettes-v1
.venv/Scripts/python.exe -m http.server 8882 --bind 127.0.0.1 --directory outputs/identity-court-silhouettes-v1
```

Open http://127.0.0.1:8882/. The model defaults to `models/yolo26s-seg.pt` and
CPU execution. Use `--device 0` on a CUDA host. Videos, models and output caches
remain Git-ignored. The tests check one-to-one association, null handling and
source-field preservation; they do not establish visual accuracy.
