# Two-clip challenge-checkpoint presentation

The current browser presentation is
`outputs/challenge-player-presentation-v1/index.html`, created on branch
`homography` on 2026-10-18. It contains both full clips with player track boxes,
matching colored IDs/foot markers, and a top-left court mini-map. Clip selectors,
continuous two-clip playback, 0.5× / 0.25× playback, frame stepping, fullscreen,
MP4 downloads and position exports are available. Add `?clip=2` to start on clip 2.

## Selected pipeline

Court calibration: KaliCalib `model_challenge.pth`, upstream revision
`3b3f52b59e8e38e155ba7e9c681281d44748c173`, positive winning-channel centroids,
35 px RANSAC at 960×540 and constrained camera refinement. Every source frame
was actually processed locally on CPU (four PyTorch threads). No manual court
anchor, colour-line refinement, temporal propagation or fixed-camera fallback
is used. Failed estimates remain null. Nominal FIBA dimensions are assumed.

Tracking: existing YOLO26s / TrackTrack + ReID (`yolo26n-cls.pt`), buffer 30,
GMC `sparseOptFlow`, from the completed 2026-10-15 experiment. No person
detection/tracking inference was rerun. Source hashes and tracking metadata
are checked and preserved.

Both clips use explicit manual role annotations bound to their exact tracking
exports. Clip 2's new annotations were reviewed at frames
0,25,50,75,100,125,150,174, with targeted crops for short-lived IDs. These are
sampled role labels, not exhaustive identity ground truth. Referees/staff are
excluded from boxes and the mini-map; every original person row is retained
in the machine-readable output. Fragmented/duplicate tracks and the previously
confirmed clip 1 87-to-96 wrong-person association are not repaired.

## Executed outputs

| Measurement | Clip 1 | Clip 2 |
|---|---:|---:|
| Full source/rendered frames | 150 | 175 |
| Camera estimate available | 150 | 168 |
| Preserved person-track observations | 1,358 | 1,894 |
| Role-labelled player-track observations | 969 | 1,484 |
| Displayed projected player-track observations | 955 | 1,417 |
| Per-frame displayed dot range | 4–9 | 0–10 |
| Calibration processing time | 78.52 s | 89.64 s |

Clip 2 calibration is unavailable at frames 43,44,47,50,52,55,56. Player boxes
continue to display current tracker observations at these frames, while the
mini-map has no projected dots. Image-clipped footpoints and points projected
outside the nominal court are also excluded from the map. No stale dots,
off-screen trajectories or unseen players are added.

Both MP4s are H.264, 1280×720, 25 FPS, exactly 6 / 7 seconds, without audio.
The 360p clip is upscaled for overlay legibility; no new image detail is added.
Both MP4s were fully decoded to verify length. All 27 repository CPU tests pass.
Export audits verified source observation counts, exact player-only displayed
populations, null projection during missing calibration, and no referee dots.
Browser screenshots verified the layout and media loading for both clips.

These are availability/export measurements, not ground-truth accuracy. A camera
estimate passing the upstream check may still be incorrect. Bbox bottom centre
approximates a floor-contact point, whose visibility is unknown. Colors encode
tracker IDs, not teams, jersey identities or stable physical people. Playback is
synchronized to original timestamps; this is offline rendering, not measured
25 FPS live inference.

## Reproduce / inspect

```powershell
.venv/Scripts/python.exe -m tacticalai.auto_calibration_cli prepare
.venv/Scripts/python.exe -m tacticalai.challenge_presentation infer --output outputs/my-challenge-presentation
.venv/Scripts/python.exe -m tacticalai.challenge_presentation render --output outputs/my-challenge-presentation
```

The `preview` package extra provides `imageio-ffmpeg` for H.264 encoding. Use a
new output directory; inference and render products are protected from overwrite.
To regenerate only the HTML after changing its template:

```powershell
.venv/Scripts/python.exe -m tacticalai.challenge_presentation page --output outputs/my-challenge-presentation
```

Each clip directory includes `projection.mp4`, preview JPEGs, raw/refined
per-frame `calibration.jsonl`, `frames.jsonl`, `positions.csv`, role annotations,
tracking metadata and counts. `experiment.json`, `presentation_metadata.json`
and `run_source/` preserve model/input/code hashes, versions and source snapshots.
The files remain local and ignored by Git; original videos are unchanged.
