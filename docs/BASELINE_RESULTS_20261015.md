# First Colab baseline results

Inspected the Drive report and clip 1 tracker metadata on 2026-10-15.
Run: `20261015T194640Z-1087597a`.
Drive folder: https://drive.google.com/drive/folders/1IIjc-tIDxU7g4j7KlX49JdP0Lqb8QEzM

## Reported results

The saved `review.json` reports `artifact_checks_passed: true`, no issues,
and all four expected runs with complete frame counts. These are remote
report results, not independently repeated local validation.

| Clip | Tracker | Frames | Detector boxes | Track observations | Unique IDs | Loop time (s) | Pipeline FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| attack_clip | BoT-SORT | 150 | 3856 | 2399 | 37 | 9.397 | 15.96 |
| attack_clip | TrackTrack | 150 | 3856 | 1267 | 26 | 8.010 | 18.73 |
| attack_clip2 | BoT-SORT | 175 | 6187 | 2751 | 64 | 22.972 | 7.62 |
| attack_clip2 | TrackTrack | 175 | 6187 | 1848 | 27 | 20.317 | 8.61 |

Every run reports zero frames without any tracks. This does not establish that
every player was tracked.

Clip 1 metadata confirms complete, non-truncated processing with YOLO26s,
confidence 0.10, IoU 0.70, imgsz 1280, FP32, seed 0 and Tesla T4. Versions:
Ultralytics 8.4.158, Torch 2.11.0+cu130, torchvision 0.26.0+cu130,
OpenCV 4.12.0.88, NumPy 2.2.6, Python 3.13.15. ReID is disabled and GMC is
`sparseOptFlow` for both trackers in all four report rows.

Both clip 1 tracker metadata files have matching source-video, model, source-code
and detector-output SHA256 values. Detector output SHA256:
`763cd969c00277c294a095e89fe6e6ec8a76c9d57d3849c367076ba5c1836000`.
The remote review reports matching comparison settings for clip 2; clip 2
metadata and detection hashes have not been independently inspected here.

## Interpretation

TrackTrack's observed pipeline FPS was about 17% higher on clip 1 and 13% higher
on clip 2 in these single runs. Timings include warmup, decode, tracking,
overlays, exports and encoding; they are not isolated tracker benchmarks.
Neither run achieved the source rate of 25 FPS.

Fewer unique IDs cannot be interpreted as fewer ID switches. TrackTrack also
exports fewer track observations. Its default association thresholds differ:
on clip 1, BoT-SORT high/new thresholds are 0.25/0.25, versus 0.60/0.70 for
TrackTrack. TrackTrack also uses min_track_len 3. Thus fewer IDs can reflect
different track initiation and retention as well as identity behavior.

The detector produces about 25.7 person boxes per frame on clip 1 and 35.4 on
clip 2. The current class is COCO person; role filtering is absent. Those
counts are not player counts, and may include referees, spectators, staff,
duplicates or false positives. Their contribution requires visual inspection.

Accuracy metrics remain null. No manual ID-switch counts, precision/recall,
IDF1 or HOTA have been measured.

## Review limitation and next step

The in-app browser's ZIP download failed with an HTTP error. The clip 1
BoT-SORT video preview reported "Unable to load video. Try again."
No annotated video frames were visually inspected, and the results bundle has
not been retrieved locally. The preview error alone does not establish that
the MP4 is corrupt; the runner uses mp4v, which may need H.264 conversion for
browser playback.

Retrieve `baseline-results.zip` into local `outputs/`, extract to a fresh
directory, validate with `scripts/review_runs.py`, decode all four annotated
videos, and visually compare matched frames before choosing a tracker or
tuning player filtering. Annotate representative player trajectories to
measure missed players, false positives and ID switches.
