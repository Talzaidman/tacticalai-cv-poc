# Unlinked person observations - 21 October 2026

Current development playback uses the native video player, with notes in chat.
No detailed investigation UI is generated. Source baseline and earlier pages
remain preserved.

- Clip 1: http://127.0.0.1:8889/attack_clip/annotated.mp4
- Clip 2: http://127.0.0.1:8889/attack_clip2/annotated.mp4
- Output: `outputs/unlinked-people-v2/`.

The role/team presentation previously drew only confirmed tracks. The saved
YOLO26s detection stream includes people without those tracks. The new compositor
preserves existing identities, role/team decisions, boxes and court positions,
and adds otherwise unrepresented detections as gray `Unlinked` observations.
Their track/person/role/team/jersey/court-position fields remain null. They are
not new identities and are not projected onto the minimap.

Existing tracks reserve mask matches first; unmatched people receive remaining
automatic YOLO26s-seg masks by one-to-one geometric matching. Missing original
masks can be recovered from the existing higher-resolution mask cache. A box
and label remain when no mask matches. No complete hidden body/mask is invented.
No detector, segmenter, tracker, ReID or CLIP network was rerun.

Configuration is in `configs/unlinked_people.json`: one-to-one detection/track
IoU >= .5; mask IoU >= .3; conservative automatic court gate with a 3 m margin
and 10% body-box overlap support. Uncertain/clipped observations remain eligible.
Near-identical-head overlapping maskless duplicates are suppressed only when
IoU >= .5 and normalized head differences <= .1. Overlap alone does not suppress
another person with a distinct head or its own matched mask.

The first candidate also added segmentation-only detections. It added crowd
clutter without improving the existing inspected-person coverage, so that option
is off in the selected configuration. These are development-driven choices,
not independent validation. Original cached measurements remain available.

| Per-frame observation diagnostic | Clip 1 | Clip 2 |
|---|---:|---:|
| Original tracks preserved | 1,358 | 1,894 |
| Added unlinked detector observations | 1,896 | 3,555 |
| Added observations with masks | 1,555 | 2,754 |
| Recovered masks on original tracks | 52 | 55 |
| Rows using a box fallback | 342 | 801 |
| Suppressed near-identical maskless duplicates | 51 | 47 |

These are repeated per-frame predictions, not unique people or true positives.
The automatic court gate still retains some spectators/staff, and calibration
errors may exclude a real boundary person. No role is invented for these rows.
Geometric matching can still confuse overlapping or duplicate boxes.

On the existing eight-frame development inspection set, head-point coverage
increased **72/101 -> 99/101**, and partially visible coverage increased
**8/25 -> 24/25**. This is one-to-one point containment, not box AP, precision,
identity accuracy or held-out validation. Annotation uncertainty remains. Labels
are evaluation-only and never enter composition, mask matching or rendering.
People missed by both source models still have no automatic measurement.

Full frame JSONL, CSV, null semantics, source/cache/configuration hashes,
suppression reasons and runtime metadata accompany the MP4s. New packaging took
18.79 / 32.99 s, excluding the historical inference already cached. It is not a
full-pipeline runtime benchmark. Both new MP4s fully decoded at 150/175 frames.
All 85 CPU tests passed; artifact checks verify source-detection equality,
unknown IDs/roles/teams/positions and CSV nulls.

```powershell
.venv/Scripts/python.exe -m tacticalai.unlinked_people_cli --output outputs/unlinked-people-fresh
.venv/Scripts/python.exe scripts/evaluate_unlinked_people.py --output outputs/unlinked-people-fresh
.venv/Scripts/python.exe scripts/serve_review.py outputs/unlinked-people-fresh --port 8889
.venv/Scripts/python.exe scripts/validate_unlinked_people.py --output outputs/unlinked-people-fresh --url http://127.0.0.1:8889
```

Use a fresh directory. The minimal index contains only two native video controls;
the direct MP4 links open the browser's basic player. Source footage and previous
outputs are unchanged. Source videos, model assets and results remain ignored.
