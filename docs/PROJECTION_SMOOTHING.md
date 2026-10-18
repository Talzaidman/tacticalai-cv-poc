# Temporal stabilization of court/player projections

Implemented and executed on branch `homography`, 2026-10-18. Latest browser
artifact: `outputs/challenge-player-smoothed-v1/index.html`. The page offers
Smoothed, Original and a frame-synchronized side-by-side comparison, for both
complete clips. The original presentation and source observations are preserved.

## Method

The working assumption is smooth camera motion **within a continuous shot**.
Independent per-frame neural calibration was visibly noisy. Instead of averaging
homography coefficients, map nine fixed lower-image probe points through each
image-to-court homography and filter those 18 court-coordinate components. Fixed
image probes have useful scale during a pan even when the opposite court end
is far off-screen; projecting full-court corners into the image was extremely
ill-conditioned for these partial views.

Camera states are constant-velocity Kalman states with process noise allowing
acceleration. A Rauch–Tung–Striebel backward pass uses future observations for
offline smoothing, avoiding the ordinary forward low-pass phase delay. Refit
a projective transform from the nine smoothed correspondences; reject nonfinite,
rank-deficient or >0.25 m RMS correspondence fits. Matrix coefficients are never
filtered directly.

Before filtering, detect abrupt calibration outliers using leave-one-out robust
local linear prediction over ±4 frames. Require coordinate RMS discrepancy
above both 0.8 m and four times local scatter. Do not reject a sustained linear
pan merely because its velocity is high. These gates/variances are engineering
defaults, not learned noise statistics or ground-truth confidence intervals.

Short camera gaps are smoothed only between supported observations, up to five
missing frames (200 ms at 25 FPS). There is no extrapolation before/after support
or bridging a longer gap. Estimated camera frames explicitly use
`kalman_rts_camera_gap_estimate`; dependent projected rows use
`estimated_calibration_floor_projection`. No default mean camera is inserted.

Reset across scene cuts. The current cut detector uses HSV histogram
Bhattacharyya distance >0.55 **and** mean absolute thumbnail difference >35,
at 160×90. It found only the initial boundary in each clip, consistent with prior
visual review. This is a heuristic, not validated broadcast-shot detection.

Then lightly smooth player XY within each contiguous observed track-ID/role
segment, using a higher acceleration allowance than for the camera. Split at
missing measurements, camera cuts, role changes, or displacement >`0.6+12/fps`
metres per frame. Never join IDs, repair identities, synthesize a missing player
row, or show a previously observed player after disappearance. Raw player boxes
and footpoints remain unchanged.

Default parameters:

| State | Measurement sigma | Acceleration sigma |
|---|---:|---:|
| Camera probe coordinates | 0.25 m | 6 m/s² |
| Observed player coordinates | 0.14 m | 12 m/s² |

Image probe coordinates: x fractions 0.2,0.5,0.8 and y fractions 0.62,0.75,0.88.
This representation is appropriate to the floor region in these clips; other
camera angles need validation. RTS uses future frames, so this is an offline
presentation improvement, not a claim of causal real-time operation.

## Executed results

No detector, tracker or calibration network was rerun. CPU postprocessing reused
the complete saved challenge-checkpoint estimates and TrackTrack/ReID observations.

| Measurement | Clip 1 | Clip 2 |
|---|---:|---:|
| Source / rendered frames | 150 | 175 |
| Original model camera estimates | 150 | 168 |
| Smoothed supported camera estimates | 150 | 175 |
| Abrupt camera measurements rejected | 0 | 8 |
| Camera frames reconstructed/estimated | 0 | 15 |
| Median camera probe roughness, original → smoothed | 0.31253 → 0.00225 m | 0.48796 → 0.00314 m |
| Median observed-player roughness, original → smoothed | 0.32101 → 0.00626 m | 0.42406 → 0.00717 m |
| Matched player triples for roughness | 880 | 1,254 |

Roughness is Euclidean second-difference displacement, compared on the same
supported frame triples. The median player proxy falls approximately 98.1% /
98.3%. **This measures smoothness, not correct positions or true acceleration.**
Real movement can also be attenuated. Camera estimates can be smoothly wrong.

Clip 2 rejected original measurements at frames 5,12,21,46,51,62,123,145.
Estimated frames are 5,12,21,43,44,46,47,50,51,52,55,56,62,123,145; the additional
seven were missing original cameras. The browser status explicitly shows
Estimated at these frames. There are no unsupported remaining camera intervals
in these two clips under the configured bound.

Agreement with existing draft manual anchors (evaluation only, best court-axis
reflection per checkpoint): clip 1 mean 9.83 → 8.70 native pixels across seven
checkpoints; clip 2 frame 0 52.49 → 55.89 native pixels. The latter worsens slightly,
illustrating why reduced jitter must not be presented as verified accuracy.
These clicks are not surveyed ground truth and were not inputs to smoothing.

All 1,358 / 1,894 original person observations, IDs, roles and bbox coordinates
were retained. Every row records `court_xy_raw_m`; player rows also retain
`court_xy_camera_smoothed_m` before their own temporal filter. Final `court_xy_m`
is the displayed estimate. Unknown/clipped footpoints remain null. No referee
or staff dots are displayed. Existing ID switches/duplicate player tracks remain.

Both smoothed MP4s fully decoded at 25 FPS, 1280×720 and 150/175 frames. Comparison
MP4s combine smoothed/original frames into one 2560×720 stream so seeking, slow
motion and playback are synchronized exactly. All 31 repository CPU tests passed,
including constant-velocity reconstruction without systematic phase shift,
camera outlier rejection, bounded gaps, cut resets, no endpoint extrapolation,
and no player insertion or ID/role blending.

## Reproduce

```powershell
.venv/Scripts/python.exe -m tacticalai.stabilize_presentation --source outputs/challenge-player-presentation-v1 --output outputs/my-smoothed-presentation
.venv/Scripts/python.exe -m tacticalai.challenge_presentation render --output outputs/my-smoothed-presentation
```

Use a new output directory. `prepared_frames.jsonl`, `calibration.jsonl`,
`source_calibration.jsonl`, `frames.jsonl`, CSV, metadata and source snapshots
preserve raw inputs, filtered estimates and processing parameters. Rendering uses
the prepared positions and checks their source ID order/provenance. To rebuild
only the page: `python -m tacticalai.challenge_presentation page --output ...`.

The working candidate still uses nominal court dimensions, bbox bottom-centre
ground-position approximation, manual role labels, and unverified physical player
identities. Filtering improves visual stability; it does not resolve those limits.
