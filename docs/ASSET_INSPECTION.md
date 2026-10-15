# Source inspection, 2026-10-15

Read START_HERE.md, AGENTS.md and PROJECT_CONTEXT.md; extracted and visually reviewed both rendered pages of the original Hebrew brief. The full brief requests court/ball/player localization, team/referee distinction, specific player identity, continuity through occlusion/view changes, unseen-position prediction and a court map. Tactics are optional. This baseline is an incremental step, not fulfillment of that entire brief.

Both source videos were fully decoded using CPU OpenCV 4.12.0.88:

| Source | Dimensions | FPS | Declared / decoded frames | Duration |
|---|---|---|---|---|
| attack_clip.mp4 | 640 x 360 | 25 | 150 / 150 | 6.0 s |
| attack_clip2.mp4 | 1920 x 1080 | 25 | 175 / 175 | 7.0 s |

Visually inspected six evenly spaced frames per clip, including first/last, in ignored `outputs/inspection/` contact sheets. This is sampling plus full decode, not frame-by-frame annotation.

Clip 1 shows yellow/dark uniforms, half-court action, overlaps, orange/black referees, spectators, lower-image broadcast graphics and camera reframing. Clip 2 pans from one basket across the court into the opposite half; players enter/exit the image and overlap near the basket. Generic-person detections can include spectators/staff/referees. Neither clip justifies a static homography assumption.

No model inference or quantitative accuracy evaluation was performed. Original assets were preserved. Recreate contact sheets with `python scripts/inspect_assets.py data/raw/attack_clip.mp4 data/raw/attack_clip2.mp4` (requires OpenCV and Pillow; optional Pillow pin: 11.3.0).
