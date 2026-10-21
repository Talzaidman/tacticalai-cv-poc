# Automatic-only presentation

The 21 October 2026 revision is `outputs/automatic-silhouettes-v1/index.html`.
Open http://127.0.0.1:8883/ while the local server runs.

The previous presentation contained manually assigned player/referee/staff tags.
Those also affected player filtering, player counts, map eligibility and position
smoothing segments. Human judgments were shown as development evaluation claims
and in the embedded review page. These are excluded from this new presentation.
It does not load role annotation files, fused positions or human pair judgments.

Inputs retained from automatic stages:

- YOLO26s person detections and TrackTrack tracking, ReID yolo26n-cls,
  GMC sparseOptFlow, buffer 30.
- Automatic temporal identity splitting and gallery matching. The legacy field
  `player_id` denotes a person hypothesis, not a player-role classification.
- YOLO26s-seg person masks and one-to-one geometric mask association.
- KaliCalib camera estimates and automatic camera smoothing.

Positions are recomputed from automatic identity rows and saved camera matrices.
Position smoothing applies equally to every observed person, resetting at gaps,
cuts and unknown/changed identity segments. The map includes every valid person
projection inside the assumed court; there is no automatic player-role classifier.
Rows retain `role: null`; role columns and role-dependent counts are absent.
Downloads include automatic identity events, candidate scores and mapping only.

Remaining assumptions and development dependencies:

- Identity thresholds were developed using feedback on these clips. Inference
  does not read human identity-pair labels, but performance is not independently
  validated. No new identity accuracy claim is made.
- Court size assumes nominal FIBA 28 × 15 m, not a venue measurement.
- Camera/player smoothing is offline and uses future frames. Supported camera
  gaps remain explicitly marked estimated. Missing people are not invented.
- Bounding-box bottom center approximates a floor point; actual foot contact,
  jumping and visibility are unverified. Segmentation does not change footpoints.
- Segmentation confidence, mask association threshold, identity thresholds and
  smoothing parameters are configured engineering choices.
- Role, team, jersey, ball and tactical classification are unimplemented.

Both full annotated MP4s were rendered and fully decoded (150/175 frames).
All 63 CPU tests passed, including a test that changes manual role labels and
cached fused positions without changing any automatic output. Networks were
not rerun; cached automatic measurements were reused. Source hashes, masks,
original IDs and appearance hypotheses are checked. Prior experiments remain
unchanged for historical provenance and are not linked as current results.

Reproduce from the saved automatic caches:

```powershell
.venv/Scripts/python.exe -m tacticalai.automatic_presentation --output outputs/automatic-silhouettes-fresh
.venv/Scripts/python.exe -m http.server 8883 --bind 127.0.0.1 --directory outputs/automatic-silhouettes-fresh
```

For new masks, `tacticalai.silhouette_presentation` now finishes by building the
automatic-only presentation. Its intermediate `-mask-cache` folder preserves
source provenance and is not the delivered review page.
