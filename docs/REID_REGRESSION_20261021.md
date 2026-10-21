# Return / duplicate-ID regression — 21 October 2026

The user subsequently accepted restoration. The active ReID profile and
[current presentation](RESTORED_IDENTITY_BASELINE.md) now use the earlier setup.
The audit below preserves the evidence and recommendation before that action.

The user reported T32 → T292 in the current clip-2 review. Inspection of the
actual crops finds the likely intended pair is **T38 → T292**. In this saved
run T32 remains on a different orange-shirted person throughout all 175 frames.
That observation is diagnostic inspection, not a new classifier or role tag.

## Verified findings

- Current T38: 84 observations, frames 0–96; gaps 75→81 and 87→96.
- Current T292: 98 observations, frames 75–174; gap 84→87.
- The IDs coexist at frames 75, 81, 82, 83, 84, 87 and 96. Representative crops
  show two boxes on the same black-uniformed person. The T38 crop at frame 96
  is mixed/ambiguous, so this is not a clean handoff with a single missing gap.
- The earlier online tracker retains raw ID 215 for the inspected player,
  with 170 observations across frames 0–174 and short gaps 19→21, 72→75 and
  160→163. The earlier linked display remains P215; this example did not need
  the offline linking stage to repair a raw tracker-ID change.
- The local conservative control also matches all 170 original ID-215 boxes
  with a single control track, using same-frame IoU > 0.7.
- Lower thresholds without court filtering already split the corresponding
  observations across IDs 14 and 291. Therefore court filtering is not required
  for this regression. Numerical IDs differ across variants and are not global
  physical-person identities.

The changed high/low/new confidence gates were 0.60/0.25/0.70 → 0.25/0.10/0.25.
At the first exported T292 frame (75), T38 is a partial box and T292 is a larger
box on the same person. Their tracked-box IoU is about 0.497, below TAI's 0.55
duplicate-suppression threshold. T292's current score is about 0.436. This is
consistent with more permissive initialization allowing a second track around
an existing person. Frame 75 is the first confirmed export, not a measured
creation time: unconfirmed internal states and the original birth confidence
were not logged. No exact association-cost explanation is claimed.

## Recommendation

Use the earlier full identity checkpoint as the stable POC baseline again:
original TrackTrack confidence gates, online ReID, buffer 30, GMC sparseOptFlow,
and its automatic CLIP temporal/gallery identity layer. Keep the silhouette
renderer and automatic display filters. The high-coverage threshold experiment
is useful but has a demonstrated identity regression and should not replace
the identity checkpoint on the strength of detection coverage alone.

`configs/reid/identity_checkpoint_20261019.json` is generated from the original
run metadata and exactly matches its effective tracker settings. It does not
use the later buffer-45 configuration, which was not the executed checkpoint.
No previous run or inference result was modified. No specific ID pair is joined
or corrected in the algorithm. This recommendation is based on this inspected
case and previous development checks, not comprehensive identity accuracy.

Before promoting new tracker thresholds, compare missed-person coverage and
identity continuity separately, including duplicate simultaneous tracks, false
joins, fragmentation and return matching on the same annotated cases.

## Evidence

Local page: `outputs/reid-return-audit-v1/index.html` (port 8886), with both
full clean videos, frame jumps, crop sheets and `audit.json`. Evaluation IDs
parameterize the audit only; they are not consumed by inference.

```powershell
.venv/Scripts/python.exe scripts/audit_reid_return.py
.venv/Scripts/python.exe scripts/serve_review.py outputs/reid-return-audit-v1 --port 8886
```

The range-aware static server supports native MP4 seeking. Bounded, open-ended,
suffix and unsatisfiable byte-range responses were checked; the browser loaded
both videos at frame 75 (3.00004 seconds) with seekable ranges 0–7 seconds.
