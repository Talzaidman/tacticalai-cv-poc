# Manual court calibration prototype

An initial automatic baseline has now been executed on both clips. See
[the automatic experiment results](AUTOMATIC_HOMOGRAPHY_20261018.md) and
`outputs/homography-auto-v1/review.html` for synchronized browser review.
Automatic coverage is incomplete and alignment remains unverified; the manual
calibration below remains the working reference.

The follow-up [KaliCalib reference comparison](KALICALIB_REFERENCE_20261018.md)
tests the original extraction/camera-refinement method with both supplied
checkpoints on 15 fixed frames. The challenge checkpoint improves draft-click
agreement but still needs line alignment refinement. Open
`outputs/homography-kalicalib-reference-v1/review.html` for the comparison.

Work is on branch `homography`. This is CPU postprocessing of the existing
YOLO26s / TrackTrack / ReID (`yolo26n-cls.pt`) runs, buffer 30,
GMC `sparseOptFlow`. No detection or tracking inference was rerun.

## Full clip 1 mini-map preview

The latest artifact is `outputs/homography-minimap-v1/attack_clip/projection.mp4`:
the complete 6-second clip at 25 FPS, H.264, 1280 x 720, with an inset in the
top-left and colored track-ID foot markers. The source 640 x 360 footage is
upscaled for overlay readability; this adds no image detail. `preview.gif`
provides a smaller 25 FPS looping preview. Audio is omitted.

`configs/calibration/attack_clip.sequence.manual.json` contains seven draft
manual keyframes at 0, 25, 50, 75, 100, 125 and 149. Each fits the four corners
of the key. Some partially occluded intersections were estimated from visible
adjoining line segments. The renderer refits linearly interpolated image
anchors between these keyframes. Overlays at the keyframes and intermediate
frames 12, 37, 62, 87, 112 and 137 were visually inspected. This supports a
preview throughout all 150 frames, not a measured full-sequence accuracy claim.
Intermediate matrices retain their `interpolated_anchors_unvalidated` label in
the exports. Clip 2 has not yet been extended through camera movement.

The optional run-specific file `configs/calibration/attack_clip.roles.manual.json`
contains **sampled manual role annotations**, not a trained role classifier or
exhaustive identity ground truth. Referees/staff are omitted from the mini-map
with `--players-only`; all tracks remain in the data. Track 96 is labelled staff
through frame 130 and player from frame 131, reflecting the previously confirmed
wrong-person switch without changing or linking any IDs. Colors encode track
IDs, not teams. Missing detections disappear; no off-screen positions are filled.

To reproduce into a new directory (H.264 additionally needs
`imageio-ffmpeg==0.6.0`, available via the `preview` package extra):

```powershell
$trackingRun = 'outputs/occlusion-20261015T204150Z-029a7a35/reid/attack_clip/tracktrack'
.venv/Scripts/python.exe -m tacticalai.calibration_cli render --video data/raw/attack_clip.mp4 --calibration configs/calibration/attack_clip.sequence.manual.json --tracks "$trackingRun/frames.jsonl" --roles configs/calibration/attack_clip.roles.manual.json --players-only --layout minimap --map-flip-x --codec h264 --output outputs/my-full-minimap
```

`--map-flip-x` changes display orientation to put the active basket at image
right, matching this source clip; it does not change exported metric XY. The
default `diagnostic` layout remains available for court-line inspection.
Role annotations must match the tracking export SHA256. Exact displayed IDs
are recorded in each JSONL frame as `minimap_track_ids`.

The completed run preserved all 1,358 tracker observations, produced approximate
floor projections for 1,287 of them (including non-player tracks in data), and
left 71 image-clipped observations unknown. The mini-map displays 955 player
observations in total, between 4 and 9 per frame; this inherits incomplete
tracker coverage and is not a claim that all ten players are recovered. The MP4 decoded to exactly 150
frames at 25 FPS and 1280 x 720. All 17 repository CPU tests passed, including
manual role interval boundaries and exclusion of missing/outside/referee points
from the mini-map. Playback is synchronized to the source; this is offline
rendering, not a measured live-inference throughput claim.

## Original single-frame checkpoint (preserved)

Two draft manual calibrations are provided:

- `configs/calibration/attack_clip.manual.json`: frame 149 (5.96 s).
- `configs/calibration/attack_clip2.manual.json`: frame 0 (0.00 s).

Each has four fitting landmarks and one held-out landmark. These are initial
visual click estimates by Codex, not reviewed human ground truth. Four points
can fit almost exactly even when the selected correspondences are wrong.
Review the entire court-line overlay and independent landmarks, not just fit
RMSE. Do not interpret the held-out landmark residual as measured player accuracy.

**Only these individual frames are calibrated.** Both cameras move. All other
frames remain unknown until more anchors / supported temporal intervals are
provided. There is no static-matrix fallback, extrapolation in time, optical
flow propagation, or automatic landmark detector in this prototype.

Court geometry uses nominal FIBA-style dimensions with line-centre
approximations; the venue dimensions have not been surveyed. The reference is
[FIBA Official Basketball Rules 2024, Article 2](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-v10a.pdf).
The template draws full free-throw circles for diagnostic alignment, including
the semicircle that may be dashed or hidden in the footage. It omits the
no-charge arc, basket/backboard and small rebound markings.

## Review or add a keyframe locally

From the repository root in VS Code's PowerShell terminal:

```powershell
.venv/Scripts/python.exe -m tacticalai.calibration_cli annotate --video data/raw/attack_clip.mp4 --calibration configs/calibration/attack_clip.manual.json --frame 149
```

The OpenCV window shows the source frame and a court diagram. Click a diagram
dot to select its landmark, then the matching **floor** point on the video.
`N` / `P` cycle landmark names. `F` assigns fitting points; `C` assigns held-out
checks. `U` removes the selected point. `V` toggles the fitting overlay. `S`
saves (requiring at least four usable fitting points), and `Q` / Escape cancels.
Saving preserves a timestamped JSON backup beside the calibration file.
This GUI requires local OpenCV with desktop support; it does not open in Colab.
Colab can consume the same saved JSON through the rendering command.

For an additional frame, change `--frame`. Inspect frames throughout each clip,
starting with 10-25 frame spacing and adding more where the camera changes
quickly. Do not click an elevated rim, an ad board, a hidden intersection, or an
arbitrary point on a curve. Use visible, identifiable floor intersections spread
across the image. Prefer more than four fitting points where visibility allows,
and keep additional independent checks.

`left` / `right` mean the court diagram's ends, **not image left/right**. The
origin is diagram-left baseline / far sideline; x increases along the court,
y across it, all in metres. In the draft clip 1 calibration the diagram-left
basket is at image right. In clip 2 it is at image left. Keep this convention
fixed through each clip, including when the camera pans.

For a new video, create an empty document first (existing files are refused):

```powershell
.venv/Scripts/python.exe -m tacticalai.calibration_cli init --video data/raw/attack_clip.mp4 --calibration outputs/my-calibration.json
```

Video hash, original image dimensions, FPS and frame count are stored. Anchor
coordinates are always original pixels, even when the click window is scaled.

## Render from existing tracks

```powershell
$trackingRun = 'outputs/occlusion-20261015T204150Z-029a7a35/reid/attack_clip/tracktrack'
.venv/Scripts/python.exe -m tacticalai.calibration_cli render --video data/raw/attack_clip.mp4 --calibration configs/calibration/attack_clip.manual.json --tracks "$trackingRun/frames.jsonl" --frame 149 --output outputs/my-calibration-review
```

Use a new output directory each time. For clip 2, change both clip names and use
`--frame 0`. Omit `--frame` to render the full video. With the supplied draft
annotations, only 1/150 frames in clip 1 and 1/175 in clip 2 will be calibrated.
This is expected, not evidence of full-sequence court tracking.

Local execution on 2026-10-18 completed under
`outputs/homography-manual-v1/attack_clip` and `attack_clip2`. Both full MP4s
decoded to their expected lengths (150 and 175). The confirmed person-track
observation totals were preserved: 1,358 and 1,894. Nine observations were
projected at clip 1 frame 149 (one image-clipped box was left unknown), and seven
at clip 2 frame 0. The remaining frames have null court positions. The single
held-out manual-click residual was 1.75 px for clip 1 and 11.35 px for clip 2,
at their respective original resolutions. These tiny samples do not establish
accuracy; clip 2's overlay still needs refinement and both need more keyframes.
All 15 repository CPU tests passed. The interactive click GUI has not been
manually exercised in this session; the fitting/rendering/export paths were
executed and their images visually inspected.

Outputs:

- `frame-*.png`: court overlay plus top-down person tracks at annotated frames.
- `overlay-*.png`: source-resolution court/anchor overlay; fit anchors green,
  held-out check anchors cyan. Use this for detailed inspection.
- `projection.mp4`: 1280 x 360 diagnostic video, original FPS, no audio; only
  generated for full-clip rendering. Uses mp4v and is decoded after writing to
  verify frame count. Some browser players need an H.264 conversion.
- `frames.jsonl`: one record per processed frame, the image-to-court matrix (or
  null), fitting/check diagnostics, and observed confirmed person tracks.
- `positions.csv`: corresponding track rows; unavailable metric positions are
  empty, and empty frames remain present in JSONL.
- `metadata.json`: completion/failure, hashes of video/calibration/track exports,
  tracking metadata and geometry code, library versions, coverage counts,
  keyframe diagnostics and elapsed CPU postprocessing time. No accuracy score.
- `calibration.json` and `tracking_metadata.json`: exact input copies.

The renderer requires a completed tracking run from the identical video. It
checks export frame indices/timestamps and video provenance. Neither detector
nor tracker is imported or run. Originals and previous experiment outputs are
preserved.

## Temporal calibration

Interpolation is off by default. After manually inspecting an interval with no
cut and enough common visible landmarks, explicitly set the left keyframe's
`interpolate_to_next` to `true`, and set `max_interpolation_gap_frames` to an
appropriate positive maximum. At least four shared fitting landmarks are
required. The code linearly interpolates their **image points** and refits the
homography; it does not interpolate matrix coefficients.

This is an approximation to camera motion. Intervening frames are labelled
`interpolated_anchors_unvalidated`, with no synthetic residual presented as a
measurement. Inspect overlays / held-out manual checks between keyframes before
using these projected positions for analysis. Long gaps, cuts (disable the
interval), insufficient common landmarks and times outside annotated intervals
remain unknown. Automatic anchor detection can later supply the same named
correspondences to the fitting module.

## Position semantics and limits

All confirmed **person** tracks are retained; team/role validation is pending.
The mini-map draws positions inside the nominal court, but this does not prove
that a dot is a player. Outside projections are retained and marked outside in
the exports; no filtering is fed back into association.

Bounding-box bottom centre is an approximate footpoint. Contact/visibility
remains `unknown`; jumping, occlusion and loose boxes can make it wrong. Boxes
touching the image boundary get null court positions. Points outside the fitting
anchor hull are marked spatially extrapolated. No positions are created for
absent tracks, no IDs are repaired, and an airborne ball is never projected.

Tests cover known projective geometry, independent checks, degenerate anchors,
bounded and explicitly enabled interpolation, missing correspondences,
null/clipped positions, provenance, existing-output protection, and an actual
MP4 encode/decode roundtrip. Run:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```
