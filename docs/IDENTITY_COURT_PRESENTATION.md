# Combined tracking, identity and court-map checkpoint

Delivered locally on 19 October 2026 under
`outputs/identity-court-complete-final/index.html`. It combines the actual temporal
identity results with the previously accepted smoothed KaliCalib camera estimates.
Both complete clips have synchronized tracking, a separate unobscured court map,
all-player/all-person controls, person/raw-ID labels, frame stepping, slow playback,
continuous two-clip playback, exports and embedded matching evidence.

## Identity logic versus review tags

No reviewed ID pair, known switch frame or manual identity correction enters
identity inference. The algorithm detects persistent appearance differences,
requires corroborating tracking discontinuity, splits inconsistent raw tracks,
abstains in unclear transitions and reconnects compatible segment galleries.
See [method and parameters](TEMPORAL_IDENTITY_20261019.md).

The actual audit adds **5000 to every raw ID** in both clips, keeps every appearance
feature/box/frame unchanged, and reruns the complete temporal identity analysis.
Detected changes, old/new anchors and accepted segment connections remain identical
after translating the labels back. Both checks pass; per-clip
`id-independence-check.json` records the decisions. This verifies numeric ID labels
do not drive the repairs. It is not a generalization or accuracy test.

User judgments are development evaluation data. Thresholds were developed using
these clips and feedback, so the judgments influenced parameter selection. They
are not an independent held-out benchmark. The additional ID-89 change and most
proposed boundaries still need review.

**Separate manual input:** player/referee/staff roles use existing run-specific
manual annotations, including the coach/player role transition for 96. Roles are
applied after identity inference for display and projection filtering. They do not
force identity splits or gallery joins. Court calibration is cached automatic
KaliCalib inference with offline camera smoothing, not manual court clicks.

## Fusion and executed artifacts

No detector, tracker, calibration network or ReID network was rerun for this
presentation. The rename audit reruns cached-feature identity postprocessing.
Video hashes, original tracking hashes, identity export hashes, calibration hashes
and role-annotation hashes must agree before fusion.

Court positions are recomputed from the saved smoothed camera matrices and original
bbox bottom centers. Player-position smoothing now includes `identity_segment_id`
in its state key. It resets at person changes, unknown identity, missing
observations, missing positions, camera cuts and large jumps. It does not bridge
raw IDs or invent missing players. Unknown identity can still have a current,
anonymous floor-position observation, shown gray with `?`.

| Diagnostic | Clip 1 | Clip 2 |
|---|---:|---:|
| Original / rendered frames | 150 / 150 | 175 / 175 |
| Raw tracker IDs | 18 | 18 |
| Person hypotheses, all roles | 16 | 14 |
| Player hypotheses, manual role filter | 12 | 10 |
| Preserved person observations | 1,358 | 1,894 |
| Player observations | 969 | 1,484 |
| Displayed current player map positions | 955 | 1,484 |
| Unknown identity observations | 30 | 0 |
| Estimated camera frames | 0 | 15 |

The MP4s are H.264, 1600×720, 25 FPS, 6 / 7 seconds, without audio. A 1280×720
footage panel retains the complete source aspect ratio; the court/status panel is
separate. The 360p source is upscaled for legibility; no new image detail is added.
Both MP4s were fully decoded. MP4s use player-only display; the interactive page
can additionally show all current tracked people. CSV and JSONL retain every row.

All **59 CPU tests passed**, including preservation and person-boundary smoothing.
Representative annotated frames from both clips were visually inspected. A
simulated-DOM check exercised clip changes, raw/person labels, player/all-person
filtering, estimated camera status and the evidence panel. No real-browser visual
QA is claimed. These checks do not establish identity or projection accuracy.

Each clip exports `tracking-minimap.mp4`, `clean.mp4`, `frames.jsonl`,
`positions.csv`, `review-data.json`, `calibration.jsonl`, `roles.json`, provenance
and the rename audit. `run_source/` snapshots the fusion/rendering source. The
dedicated identity review is copied into `identity-review/`, so the complete page
and evidence are served from one local folder.

## Remaining work

- Review the additional ID-89 change, verify the proposed 85/89/91 boundaries,
  and resolve ambiguous crops. ID 85 has 29 unassigned observed rows between
  anchors 40/79; ID 89 has one. The known 96 boundary is recovered at 130/131.
- Handle short tracks such as 106 without unsafe single-crop joins. Investigate
  remaining duplicate person hypotheses and original detection/tracking gaps.
  The algorithm currently allows at most one persistent change per raw ID and
  can miss smooth transfers lacking tracking discontinuity.
- Measure false joins, false splits, missed reconnects and full-frame identity
  consistency on untouched longer footage. A count near 10 players or 13–14
  people is a useful diagnostic, not proof of correct identities.
- Run the configured buffer 45 tracker experiment on Colab. This presentation
  still uses the original buffer 30, ReID `yolo26n-cls.pt`, GMC sparseOptFlow
  observations. Benchmark the whole pipeline and compare on identical inputs.
- Replace manual role filtering with an evaluated automatic method; assess
  camera accuracy, actual court dimensions and foot-contact visibility. Smoothed
  coordinates can still be wrong. Team/jersey recognition, ball tracking and
  inferred off-screen motion remain separate experimental workstreams.

## Reproduce

```powershell
.venv/Scripts/python.exe -m tacticalai.identity_presentation --output outputs/identity-court-fresh
.venv/Scripts/python.exe -m http.server 8881 --bind 127.0.0.1 --directory outputs/identity-court-fresh
```

Open `http://127.0.0.1:8881/`. Default inputs are
`outputs/temporal-reid-final`, `outputs/temporal-features-v1`, and
`outputs/challenge-player-smoothed-v1`; override them with `--identities`,
`--features`, and `--cameras`. Use a fresh output directory. Everything stays local;
the source footage and prior experiments are preserved.
