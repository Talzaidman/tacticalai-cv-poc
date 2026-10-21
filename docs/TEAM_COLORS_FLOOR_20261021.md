# Team-colored video and experimental sideline filtering - 21 October 2026

Current basic-player videos:

- http://127.0.0.1:8890/attack_clip/annotated.mp4
- http://127.0.0.1:8890/attack_clip2/annotated.mp4

Output: `outputs/team-colors-floor-v1/`. No investigation UI was added.

## Team colors

The silhouette, fallback box, caption and minimap dot use the same automatic
role/team lookup: Team A blue `#419be6`, Team B amber `#f5af46`, referee purple
`#e678e6`, unknown/other gray `#aaaaaa`. Colors represent existing automatic
predictions, not manual team labels. A/B remain anonymous and clip-local.
Player/track labels remain available to distinguish teammates. The minimap has
a compact palette legend. New unlinked-video runs default to team coloring;
`--color-mode identity` restores the alternative.

## Experimental display filter

Input is the immutable `outputs/unlinked-people-v2/` observation cache. No YOLO,
CLIP, identity linking, tracking, court calibration or segmentation inference was
rerun. All source observations, identities, role/team predictions and positions
remain unchanged in JSONL/CSV. Each row adds `display_on_video`, a filter reason
and local floor-context evidence. Hidden rows are excluded from video and map
only, not deleted from data.

Bbox-bottom projection alone retained foreground spectators because hidden feet
and calibration error can put head boxes inside the nominal court. The additional
filter learns a per-frame floor-color model from the projected court interior,
inset 2 m and excluding detected-person masks. It fits three CIELAB clusters at
640 x 360, discards small clusters, and checks visible context near each bbox's
bottom. Low floor support hides unlinked/nonplayer observations. Existing
automatically classified players are protected; insufficient or occluded context
is retained. This is not verified court segmentation or a proof of off-court
position. It can hide referees near sidelines and retain some spectators.

`configs/floor_context.json` records the appearance/context thresholds, including
minimum training lightness 35, CIELAB distance 18, minimum context 35 pixels and
minimum floor-support fraction .12. These are development assumptions on these
clips, not calibrated probabilities or demonstrated generalization. They never
use manual person roles, team labels or evaluation points.

Across the full clips, the filter hides 1,181 / 2,598 repeated observations and
shows 2,073 / 2,851. These are prediction/display counts, not unique persons,
correctly removed spectators or measured filter precision.

The existing eight-frame development inspection set changes from 99/101 covered
head points to 93/101 displayed; partially visible coverage stays 24/25. Six
previously covered people near sidelines are hidden, including referee
observations. The inspection set includes sideline people and does not define
which people should be filtered. This is not independent player recall or
filter-accuracy validation. Details are in `display-evaluation.json`.

## Verification and reproduction

All 89 CPU tests pass. New checks cover identical raw pixel colors in silhouettes
and map dots, role/unknown colors, preservation of measurements, exclusion of
person pixels from floor fitting, missing cameras and occluded-context retention.
Both full H.264 videos were decoded at 150/175 frames. CPU floor filtering,
exports, rendering and verification took 17.42 / 30.35 s (47.91 s total), excluding
historical cached inference. Native Chrome video playback was visually checked.

```powershell
.venv/Scripts/python.exe -m tacticalai.team_video_cli --output outputs/team-colors-floor-fresh
.venv/Scripts/python.exe scripts/evaluate_floor_display.py --output outputs/team-colors-floor-fresh
.venv/Scripts/python.exe scripts/serve_review.py outputs/team-colors-floor-fresh --port 8890
```

Use a fresh output directory. The prior unfiltered basic videos remain at port
8889 for comparison. Source footage and all original measurements are preserved.
