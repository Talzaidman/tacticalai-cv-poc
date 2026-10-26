# Local players, ball and estimated control — 26 October 2026

## Current combined presentation (v6)

`outputs/players-ball-control-v6/` combines the automatic team-colored silhouettes,
person captions and mini-map with v5's solid assigned-player box and orange ball
marker/short trail. Filled positions keep the same styling. Only the assigned
player receives a video box; other missing silhouettes do not add fallback boxes.
The map ring uses the assigned player's supported floor position, never the ball's
image position. No position is invented on the map when that player's box is missing.
The mini-map remains on top of image-space ball trails to avoid confusing the two
coordinate spaces. All model data, assignments and inference flags remain unchanged.

`--player-presentation silhouettes` is the new CLI default; `--player-presentation clean`
restores the v5 display. Both clips remain on a basic native-controls playback page.
This is a rendering change only; no model/association execution settings changed.
Both clips are open together at <http://127.0.0.1:8897/>. All 150/175 output frames
decoded successfully; combined JSONL data are byte-identical to v5 and artifact
hashes were checked. Sampled silhouette/map/ball frames were visually verified.

## Current presentation: assigned box and ball trail (v5)

At the user's request, `outputs/players-ball-control-v5/` uses the original footage
with only the assigned player's solid green box and an orange ball marker/trail.
Filled assignments use the same solid box. Interpolated/extrapolated ball positions
share the marker/trail style; measurement/inference distinctions remain unchanged
in JSON/CSV and are disclosed in the basic playback page. Other player overlays,
the mini-map, dashed boxes, crosses and diagnostic captions are hidden.

The trailing path covers 0.8 s (`--ball-trail-s`), breaks at unknown ball frames,
new ball segments, camera shots and large inter-frame jumps (>0.06 image diagonals).
There is no new inference or association change. The page shows both clips side by
side with native video controls. Previous presentations and source evidence remain
preserved. Uniform styling is a presentation preference, not a claim that inferred
positions are detector measurements or verified ground truth.
Both clips are visible together at <http://127.0.0.1:8896/>. Both videos fully
decoded (150/175 frames); all pre-existing data/assignments were verified identical
to v4, with only derived display trails added. Artifact hashes, trail length/jump
limits and sampled frames were checked. No association or detector tests were
rerun for this presentation-only change.

## Current result: bounded dribble continuity (v4)

`outputs/players-ball-control-v4/` ([basic playback](http://127.0.0.1:8895/)) adds offline assignment smoothing to the
unchanged wrist-based v3 association. This is categorical controller continuity;
it does not smooth or replace ball measurements. The wrist/ball/player models
were not rerun. All existing source data and v3 results are preserved.

Unknown gaps may be filled only between measured assignments to the same
person/identity segment/track, with at most 0.60 s between supporting anchors.
Every intervening ball point must remain in the same ball segment and camera
shot, close to that player's body/dribble envelope. Strong rival wrist evidence,
repeated rival contact, a departing ball, different endpoint handlers, long gaps,
or a track/camera reset prevent bridging. The method cannot extend indefinitely
from previously inferred intervals; measured anchors are required on both sides.
This uses future frames and is explicitly offline, not zero-latency control.

A missing player box may be linearly interpolated for at most 0.12 s between
observations of the same identity, solely for continuity geometry/rendering.
It is saved as `inferred_controller_bbox_xyxy`, never inserted into `tracks`,
and receives no inferred floor/map coordinate. The highlight is dashed cyan and
labelled inferred continuity; green solid boxes retain current measured support.
The unsmoothed assignment remains in `possession_raw`, inferred scores are null,
and `control_smoothing.json` records decisions, rejection reasons and anchor frames.

On the reviewed clip-2 dribble, P221's five assignment runs over frames 23–70
become one continuous run: 21 previously unknown frames are bridged (32–43, 48,
53–56 and 60–63). The later handler changes are not bridged. Clip 1 is unchanged.
These are continuity/coverage diagnostics, **not new possession accuracy results**.
The settings were developed using these clips and user feedback; a quick pass
back to the same player or wrong endpoint assignments can still mislead this
heuristic. Independent annotated possession evaluation remains absent.

Eight smoothing tests cover dribble gaps, handoffs, rival support, departures,
camera/track changes, long/missing-player gaps, and prevention of inferred-anchor
chaining. Ten wrist-controller and six fusion tests also pass. Both videos were
regenerated and fully decoded (150/175 frames). Verification confirms unchanged
player/ball observations, preservation of raw assignments, null scores for newly
filled control frames, measured same-owner anchors, and artifact hashes. Sampled
gap/missing-box frames were visually reviewed; clip 2 was opened in Chrome.
Reproduce with:

```powershell
./.venv/Scripts/python.exe -m tacticalai.ball_fusion_cli `
  --people-results outputs/team-colors-floor-v1 `
  --ball-results outputs/colab-ball-20261026-212056 `
  --pose-results outputs/ball-contact-pose-v1 `
  --support-window-s .4 --control-margin .1 `
  --dribble-gap-s .6 --missing-player-gap-s .12 `
  --output outputs/players-ball-control-v5
```

Use `--no-smooth-control` to disable the offline pass and recover the underlying
association behavior. Keep the same inputs/settings when comparing variants.

## Latest result: wrist-aware association (v3)

The user's clip-2 review exposed whole-box proximity errors and fragmented
confirmation. A silhouette-only trial in `players-ball-control-v2` was not
promoted: it still confused overlapping bodies. The current output is
`outputs/players-ball-control-v3/`, with browser playback at
<http://127.0.0.1:8894/>. The preceding v1 description below is historical.

With the user's authorization to download/run models on this PC, the official
7,878,574-byte `yolo26n-pose.pt` checkpoint was downloaded from Ultralytics assets
release `v8.4.0`. Its SHA-256/URL/license note is in
`models/yolo26n-pose.provenance.json`. See
[official pose documentation](https://docs.ultralytics.com/tasks/pose).
The model ran on CPU, caching pose predictions near the exported ball in
`outputs/ball-contact-pose-v1/`. One-to-one IoU matching attaches each skeleton
to at most one existing player hypothesis; original tracks are unchanged.

`ball_control.py` now supports wrist-to-ball distance (COCO wrists 9/10,
keypoint confidence >=0.25, distance <=0.25 body heights), silhouette-distance
checks, relative-motion rejection, and three supporting measurements in a
0.40-second window. Reliable wrist evidence is mandatory for establishing or
changing an owner in the v3 run. Missing poses/wrists produce abstention rather
than falling back to a broad box. A confirmed owner may be retained briefly
through a dribble (<=0.28 s) or an inferred ball gap (<=0.16 s), only with body
support and no rival wrist evidence. These assignments are explicitly labelled
`estimated_control_continuity` / `recent_wrist_measurement_memory`; no new owner
can be created from an inferred point. Ambiguity, cuts, new ball segments and
contradictory motion interrupt continuity.

Clip 2 now has 42 current-measurement/wrist-supported assignment frames,
16 labelled continuity frames, and 117 unknown frames. Clip 1 has 4/11/135,
respectively. These are coverage counts, **not improved-accuracy measurements**.
The overlap at clip-2 frame 111 changes from P211 to P221 in the new output;
sampled frames were reviewed and the updated clip was opened in Chrome.
Some accepted ball measurements are still wrong: at frame 96 the ball differs
from the draft reference by 404.5 source pixels. Association alone cannot repair
that detector error. Occluded/incorrect wrist predictions, pose-to-track matching,
ball flight and inherited identity errors remain limitations. No possession
ground truth is available, and these clips supplied development feedback.

Local pose extraction took 21.70 s for both clips; fusion/rendering took 38.24 s.
Both output videos decoded completely, player/source fields remained unchanged,
all strong assignments had reliable wrist support, and artifact hashes matched.
36 relevant CPU tests pass. JSONL/CSV retain pose evidence, contact distances,
support counts, continuity sources and unknown reasons. Raw candidate rings remain
hidden. Ball court XY stays null; the map ring represents the controller's floor
position. Ball and player detectors/trackers were not rerun.

```powershell
# Only needed again if pose inputs or model settings change:
./.venv/Scripts/python.exe -m tacticalai.pose_cache_cli `
  --people-results outputs/team-colors-floor-v1 `
  --ball-results outputs/colab-ball-20261026-212056 `
  --output outputs/ball-contact-pose-v2

# Repeat association experiments cheaply from the existing pose cache:
./.venv/Scripts/python.exe -m tacticalai.ball_fusion_cli `
  --people-results outputs/team-colors-floor-v1 `
  --ball-results outputs/colab-ball-20261026-212056 `
  --pose-results outputs/ball-contact-pose-v1 `
  --support-window-s .4 --control-margin .1 `
  --output outputs/players-ball-control-v4
```

`--association legacy` remains available for the original box-proximity method.
Without `--pose-results`, temporal association uses the experimental silhouette
fallback rather than wrist evidence; that fallback is not the promoted v3 run.

All 15 files from the latest completed Colab solution were downloaded from
[ball-notebook-20261026-212056](https://drive.google.com/drive/folders/1H_J4QwqigQnqirAR18y3jMz9C38gezSe)
to `outputs/colab-ball-20261026-212056/`. This includes raw RF-DETR candidates,
baseline/Kalman JSON and MP4s for both clips, comparison videos, environment,
draft evaluation points and summary. `download_manifest.json` records Drive IDs,
byte counts and local SHA-256 values; all 15 downloaded files were verified.
No remote sharing settings were changed. Source videos and results remain ignored
by Git. Colab is now only needed for future GPU inference experiments.

## Local execution

The new CPU-only CLI consumes the exported Kalman solution and the latest
automatic team-color/floor-context player cache. It imports no Torch or detector.

```powershell
./.venv/Scripts/python.exe -m tacticalai.ball_fusion_cli `
  --ball-results outputs/colab-ball-20261026-212056 `
  --people-results outputs/team-colors-floor-v1 `
  --videos data/raw `
  --output outputs/players-ball-control-v2
```

The completed first run is `outputs/players-ball-control-v1/`. Use a new output
directory for every iteration. The basic `index.html` contains native video
controls; direct `attack_clip/annotated.mp4` and `attack_clip2/annotated.mp4` files
are also available. No backend, GPU, keys, Colab runtime or model download is needed.
The local environment needs NumPy, OpenCV, SciPy and imageio-ffmpeg.

## Assignment and rendering

The unchanged `PossessionEstimator` from `tacticalai.ball` ranks automatically
classified, visible player tracks by ball-to-body proximity, including a small
lateral/low-dribble allowance. It requires 0.12 s of repeated support (three
consecutive measurements at 25 FPS), score at least 0.55 and a 0.12 margin over
the second candidate. CLI options `--confirmation-s`, `--min-control-score` and
`--control-margin` expose these thresholds for local experiments. Scores are
heuristic compatibility values, not probabilities.

Only accepted detector measurements establish control. The original measurement
coordinates provide association evidence; the Kalman-filtered coordinates are
displayed. Missing, interpolated, extrapolated and ambiguous ball evidence produces
unknown control immediately. New ball/person identity segments and camera shots
require fresh confirmation. Unknown/referee/staff roles and hidden/unlinked tracks
are not assigned as players; no manual role labels or identity corrections are used.

The green box and `WITH BALL (estimated)` label mark the supported player. Green
ball rings show filtered accepted measurements; cyan rings show interpolation and
gray crosses show extrapolation. Raw/rejected candidates and long trails are hidden
but retained in the JSON. The mini-map ring uses the estimated controller's floor
position, **never a homography of an airborne basketball**. Role/team/identity/court
errors in the inherited automatic source remain possible.

## Executed result and checks

| Clip | Frames | Accepted ball frames | Interpolated / predicted | Estimated control / unknown |
|---|---:|---:|---:|---:|
| attack_clip | 150 | 21 | 2 / 15 | 13 / 137 |
| attack_clip2 | 175 | 107 | 15 / 20 | 16 / 159 |

The local fusion/rendering run completed in 39.68 s for both clips; this excludes
historical player/ball neural inference and Drive transfer. Controller hypotheses
were P95 on clip 1, and P216/P211/P224 on clip 2. These are clip-local automatic
person hypotheses, not jersey/roster identities. Frame counts and assignments are
software diagnostics, **not possession accuracy**. No possession ground truth exists.
Clip 1 remains especially sparse because the detector frequently misses the ball.
Overlapping bodies, ball flight and wrong accepted detections can still produce
incorrect controller assignments; sampled rendered frames show why these must
remain labelled estimates.

Both MP4s decoded completely (150/175 frames). All original player/camera fields
were verified unchanged, every assigned controller had an accepted ball measurement
and an automatic player track, all ball court coordinates stayed null, and artifact
hashes matched. Six fusion tests plus 18 existing ball tests passed. Sampled annotated
frames were visually inspected. Native Media Player was opened but did not display
playback; an automated local browser preview was blocked by the browser URL policy.
Full motion review remains a user-facing next check via the direct MP4s.

Each clip exports combined `frames.jsonl`, `ball_possession.csv`, `metadata.json`,
snapshots and `annotated.mp4`. The top-level experiment records source hashes,
versions, timings, limitations and inherited tracker/identity lineage. Active player
tracking settings are inherited from the restored automatic baseline's recorded
profile: YOLO26s, TrackTrack, ReID enabled (`yolo26n-cls.pt`), sparseOptFlow GMC,
followed by automatic CLIP-ReIdent splitting/gallery linking. Ball ReID/GMC remain
disabled. No inference was rerun here. Historical ball exports lack a source-video
hash; exported MP4 dimensions, FPS, frame counts, timestamps and raw-to-accepted
measurement links were checked against the local sources. This limitation is recorded.
