# Temporal identity consistency — development experiment, 19 October 2026

The new offline layer recovers all four user-reported physical-person chains in
clip 1 while separating the different people that shared a tracker label. It
preserves the original detector outputs, boxes, timestamps and `track_id` fields.
No detection or tracking was rerun. Identity accuracy remains unmeasured beyond
the limited development checks described below.

Serve `outputs/temporal-reid-final/` locally and open `index.html`. The page has
synchronized before/after overlays, focused person selection, optional zoom,
frame stepping, slow playback, gallery crops and cumulative unique-ID curves.
The `.0` / `.1` suffixes distinguish early/late segments of a reused tracker ID.
The displayed person name follows the earliest segment in a proposed chain.

## Results on the complete available clips

| Diagnostic | Clip 1, 150 frames / 6 s | Clip 2, 175 frames / 7 s |
|---|---:|---:|
| Original tracker IDs | 18 | 18 |
| Segment hypotheses after splitting | 22 | 18 |
| Accepted segment connections | 6 | 4 |
| Resulting person hypotheses | **16** | **14** |
| Unknown identity observations | 30 / 1,358 (2.21%) | 0 / 1,894 |

All four newly reported chains are recovered automatically:

| Earlier portion | Later portion | One-to-one cosine score |
|---|---|---:|
| Original player 85 (`85.0`) | ID 102 (`102.0`) | .985782 |
| Original player 87 (`87.0`) | Player portion of ID 96 (`96.1`) | .986036 |
| Original player 89 (`89.0`) | Late portion of ID 91 (`91.1`) | .987446 |
| Original player 91 (`91.0`) | ID 105 (`105.0`) | .985630 |

Scores are similarity statistics, not probabilities. The earlier confirmed
88/101, 100/103, 214/229, 218/225, 219/228 and 222/227 connections are retained.
The negative original-player 85/106 and 212/228 checks remain separate. ID 106
still lacks three time-separated samples; abstention does not prove that the
encoder distinguishes every crop of the two people.

Automatic change anchors in clip 1:

| Raw ID | Last old-person anchor | First new-person anchor | Unassigned observed rows |
|---|---:|---:|---:|
| 85 | 40 | 79 | 29 |
| 89 | 72 | 83 | 1 |
| 91 | 93 | 112 | 0 |
| 96 | 130 | 131 | 0 |

For 85, intervening crops include overlapping people; the layer abstains instead
of inventing an exact transfer time. ID 91 has no exported observations in the
intervening gap, so there are no rows to mark unknown. The ID-96 boundary matches
the previously verified frame 130/131 boundary. The other boundaries are automatic
proposals, not frame-level ground truth. The additional ID-89 change was found
during this experiment; sampled crops show different jersey appearances, but its
boundary and full identity chain still need review.

The 16 / 14 counts are diagnostics. Splitting one mixed label increases the
person count; joining duplicate segments reduces it. We did not force a target
count of 13–14. Very short tracks, remaining duplicates, staff/referees and any
undetected switches can still affect the totals.

## Method and controls

1. Reuse the exact original 18-ID TrackTrack runs and actual Paper CLIP-ReIdent
   checkpoint from the earlier experiment. Preserve video/frame/checkpoint hashes.
2. Extract a time-indexed appearance cache, normally every five frames, plus
   existing cached crops and denser neighborhoods around large position changes.
   Sampling uses no review IDs, labels or switch times.
3. Propose one persistent appearance change per raw ID at most. Both local
   five-sample windows and whole-prefix/suffix comparisons must have cosine
   distance separation above within-phase dispersion by at least .009. Require
   three clear, spaced samples per side, at least ten frames of local support,
   and within-window dispersion no greater than .022.
4. Refine old/new anchors using median similarity to phase references, a .002
   margin, .97 minimum similarity and overlap limits. A generic yellow/red pixel
   fraction check can veto a boundary crop when both phases share substantial
   color content and that crop lacks it. This is not a team or jersey classifier.
5. Require supporting tracking evidence within five frames of the transition
   envelope: at least five missing source frames or a center jump of at least
   .30 previous-box heights. This uses **all original observations**, so feature
   sampling cannot manufacture a gap. A gap by itself never triggers a split.
6. Build up to eight quality-selected gallery crops per segment, five frames
   apart, with at least three samples for association. Use median cosine from a
   Hungarian one-to-one crop assignment, threshold .98 and two-sided margin
   .001. Exclude overlapping lifetimes. Opposite phases of a detected change
   cannot reconnect directly or through another segment.

The change detector, color veto, tracking corroboration and temporal aggregation
are our experimental heuristics; they are not claimed as the challenge-winning
method. The basketball appearance encoder is the released CLIP-ReIdent model.

An appearance-only intermediate run (`outputs/temporal-reid-v2/`) incorrectly
split referee 226 when he turned from side to front, producing 15 hypotheses in
clip 2. Visual crop inspection identified that error. Requiring tracking
corroboration suppresses that split without suppressing the four clip-1 changes.
The failed early partial export is under `outputs/temporal-reid-v1/`; use
`outputs/temporal-reid-final/`, which repeats v3 predictions and adds artifact hashes.

These parameters were developed using these clips and user feedback. Neither
the six pair checks nor the four phase-scoped chain checks are held-out results.
No user labels are read by inference; the separate evaluation script only adds
review annotations to the web page, leaving exported predictions unchanged.

## Execution and provenance

Real CPU inference processed **266 / 364 new crops**, reusing 217 / 274 cached
crops. Complete caches contain 483 / 638 observations. Measured extraction,
model loading where applicable and encoding took 317.3 / 474.4 seconds with FP32,
four CPU threads and batch size four. Cached postprocessing/export took about
four seconds across both clips. This is not a GPU throughput benchmark.

Source tracker: YOLO26s, confidence .10, IoU .70, imgsz 1280, TrackTrack, original
buffer **30**, ReID enabled with `yolo26n-cls.pt`, GMC `sparseOptFlow`.
The selected buffer **45** remains configured for future tracker runs; it does
not retroactively change these saved observations. Model hashes, source settings,
feature and script hashes, library versions and all thresholds accompany each run.

Per-frame rows add `identity_segment_id`, `player_id` and `identity_status`.
Unknown transitions have both identity fields `null`; raw labels/boxes remain.
Missing detections and off-screen paths are not filled in. The export checks
that every original field survives and that no two simultaneous observations
receive the same non-null person identity.

All **54 CPU tests passed**, including stable appearances across missing frames,
transient occluders, persistent changes, unknown transitions, smooth pose changes,
source preservation and direct/indirect forbidden merges. Both review videos were
fully decoded at their expected frame counts. Interactive page controls and
overlay text were exercised with a simulated DOM; no real-browser visual QA is
claimed. Crop contact sheets were visually inspected.

## Reproduction

Starting from the existing original and enriched CLIP-ReIdent caches:

```powershell
.venv/Scripts/python.exe -m tacticalai.temporal_features_cli --output outputs/temporal-features-fresh
.venv/Scripts/python.exe -m tacticalai.temporal_reid_cli --cache-root outputs/temporal-features-fresh --output outputs/temporal-review-fresh
.venv/Scripts/python.exe scripts/review_temporal_identities.py --results outputs/temporal-review-fresh
.venv/Scripts/python.exe -m http.server 8879 --bind 127.0.0.1 --directory outputs/temporal-review-fresh
```

Open `http://127.0.0.1:8879/`. Use fresh output directories; originals are unchanged
and nothing is uploaded. Source code is modular and CPU-compatible. The Colab
notebook also includes an optional temporal feature/export stage using the GPU.
Untouched clips or additional frame-scoped annotations are needed to measure
generalization, false splits, missed matches and full-video identity accuracy.
