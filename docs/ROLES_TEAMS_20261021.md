# Automatic teams, roles and consolidated exports - 21 October 2026

Review: http://127.0.0.1:8888/ while the local server runs.
Current output: `outputs/roles-teams-final-v1/`.
The restored identity baseline remains unchanged and is the input to this work.

## Implemented requirements

1. A combined observation CSV plus per-clip CSV, full frame JSONL, person summaries,
   export data dictionary, configurations, source/model hashes, library versions,
   processing-time scopes and classification evidence. Unknowns are JSON `null`
   or empty CSV cells. The legacy JSON `player_id` is a person hypothesis, exported
   as `person_id` in CSV; it does not establish player role or roster identity.
2. Automatic anonymous Team A/B classification from masked torso colors of
   automatically classified players. Two uniform groups are assumed; jersey
   colors and player counts are not configured. Groups are local to each clip.
3. Automatic player/referee/other-person classification using generic pretrained
   CLIP ViT-L/14 (OpenAI weights) with descriptive visual prompts and abstention.
   No role, team or identity annotations are read by inference.

Both full annotated MP4s retain the source lengths, 150 and 175 frames at 25 FPS.
The interactive page has silhouettes/boxes, identity or team colors, role/team
filters, frame stepping, playback speed, person evidence crops and downloads.
Counts and filters use predictions only. MP4s show all saved track observations;
interactive filters do not modify the exports, IDs, court positions or MP4s.

## Method

For each non-null automatic linked-person hypothesis, select up to eight
approximately uniform, untruncated crops at least five frames apart. Minimum
crop dimensions are 12 x 40 px. Missing identities are not borrowed from adjacent
raw tracker observations. Gray square padding preserves the entire body before
CLIP preprocessing. Compare normalized image embeddings against the normalized
mean of four text embeddings per role. Aggregate cosine scores by median.

Require three appearances, median winning similarity >= .20, role margin >= .012
and sample agreement >= .625. These are engineering thresholds, not calibrated
probability thresholds. Classification is assigned across the whole person
hypothesis, so future appearances affect earlier labels. It inherits upstream
identity mistakes and does not handle role changes within a mixed identity.

Use the median CIELAB color of the central torso (25-75% box width, 20-50% height),
restricted to the automatic segmentation polygon. Missing masks and tiny torso
regions provide no color measurement. Fit deterministic two-means to per-person
color medians from accepted player hypotheses, giving each person equal weight.
Require at least two hypotheses per group and centroid separation >= 20 CIELAB
units. Assign A/B in centroid lightness order. Require at least three color
samples, median winning distance <= 40, distance margin >= 8 and agreement >= .625.
Otherwise team stays unknown. This is not a trained basketball team classifier.

No new detection, tracking, identity linking, segmentation or camera inference was
performed. Original boxes, timestamps, raw/linked IDs and floor projections are
preserved. Identity linking and camera/player smoothing already use future frames.
Court dimensions remain nominal 28 x 15 m; actual foot contact is unverified.

## Executed results and evaluation

The initial ViT-B/32 default-crop model was too ambiguous: it assigned only 5 of
41 checked observations, all correct, and no teams. That experiment is preserved
under `outputs/roles-teams-v1/`. The second run changed model size, full-body crop
preprocessing and visual prompts together; it is not an isolated model ablation.
Its complete executed inference is under `outputs/roles-teams-v2/`.

The final package recomputes aggregation/exports/rendering from the second run's
hash-verified scores and crops, with the identical classification configuration.
It explicitly reports cached role inference rather than claiming a network rerun.

| Full-clip prediction coverage | Clip 1 | Clip 2 |
|---|---:|---:|
| Saved person observations | 1,358 | 1,894 |
| Predicted player | 765 | 1,484 |
| Predicted referee | 288 | 410 |
| Predicted other person | 101 | 0 |
| Unknown role | 204 | 0 |
| Assigned anonymous team | 765 | 1,484 |

These are repeated per-frame predictions, not verified person counts, precision,
recall, roster sizes or identity accuracy. Detection coverage is inherited.

Evaluation-only labels were recorded by visual inspection of unlabelled crops at
frames 37 and 112 of both clips, before inspecting classification score evidence.
They are in `configs/evaluation/role_team_spotcheck_v1.json`. The small sample has
41 detected-person observations: 31 players, 8 referees and 2 other-person
observations. There are repeated people, and only two development clips.

- Roles: **39/41 correct including abstentions**; 2 unknown, 0 wrong assigned labels.
- Teams: **29/31 correct including abstentions**; 2 unknown, 0 wrong assigned labels.
- A/B is matched to human dark/light labels by a per-clip permutation for scoring
  only. This mapping never enters classification or changes output team IDs.

The role model/preprocessing/prompt choice was revised using ambiguity on these
clips. This is a development spot check, not held-out-video validation, and is
insufficient to claim generalization or comprehensive role/team accuracy. It also
does not measure undetected people, false detections, tracking or court accuracy.
No annotations overwrite predictions or appear as model role labels in the viewer.

## Timing

Executed ViT-L/14 CPU role encoding: **143.08 / 136.45 s** for **117 / 112 crops**.
Shared model loading/text encoding took 15.01 s. That run took 327.83 s in total,
including sampling, exports, rendering and verification, while reusing all
upstream stages. Four CPU threads, FP32, batch size eight; software and platform
versions are in the manifest. This is not a full-pipeline benchmark.

Final cached packaging took 32.30 s across both clips. Historical T4 tracking,
CPU segmentation, temporal appearance extraction and camera-presentation timing
fields are consolidated with their scopes and hashes. Different hardware, cache
reuse and overlapping stages mean these numbers must not be summed. A complete
end-to-end processing benchmark remains unknown. Playback at 25 FPS is separate.

## Reproduce

Install the optional roles dependencies with `pip install -e ".[roles]"` in the
existing environment. Keep all cached input baseline artifacts intact.

```powershell
# Fresh CPU inference; downloads official CLIP weights if not cached.
.venv/Scripts/python.exe -m tacticalai.role_team_cli --output outputs/roles-teams-fresh

# Or reproduce the final package from the executed classification cache.
.venv/Scripts/python.exe -m tacticalai.role_team_cli --output outputs/roles-teams-repack --classification-cache outputs/roles-teams-v2

# Evaluation is a separate command; inference has no label-file argument.
.venv/Scripts/python.exe scripts/evaluate_roles_teams.py --source outputs/roles-teams-repack --labels configs/evaluation/role_team_spotcheck_v1.json
.venv/Scripts/python.exe scripts/serve_review.py outputs/roles-teams-repack --port 8888
.venv/Scripts/python.exe -m unittest discover -s tests
```

Use fresh output directories. Full JSONL preserves masks; browser contours are
simplified by at most 0.5 source pixels. Unknown identity rows remain unclassified.
Ball, jersey/roster identity, off-screen prediction and tactics remain unimplemented.

## Provenance and licensing

OpenCLIP version 2.0.2 loads the official pretrained weights; the downloaded
ViT-L/14 SHA256 is `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
Model and code sources: [OpenCLIP](https://github.com/mlfoundations/open_clip),
[CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md),
[CLIP MIT license](https://github.com/openai/CLIP/blob/main/LICENSE).
CLIP-ReIdent remains a separate upstream identity encoder, not this role model.
Existing detector/segmentation model and footage rights still require review
before commercial use. No source videos, crops, weights or outputs were published.

## Verification

All 76 CPU tests passed, including new role/team abstention, two-team support,
deterministic grouping, mask sampling, full-body padding, unknown-identity
isolation, manual-label independence and preservation of measured positions.
Both annotated MP4s were fully decoded at 150/175 frames. Packaging verifies
baseline artifact hashes, classification cache/config/source/crop agreement and
unchanged original per-observation measurements. These software checks do not
establish CV accuracy.

`scripts/validate_roles_teams.py` verified all 3,252 combined CSV rows against the
frame JSONL and baseline measurements, output/source/crop hashes, null semantics,
evaluation-to-prediction hashes, 22 HTTP resources and MP4 byte-range seeking.
Its report is saved as `verification.json`. Real Chrome checks covered both clips,
playback, frame stepping/scrubbing, role/team filters and per-person evidence;
no browser console errors were observed.
