# Gallery improvements — 2026-10-19

**Subsequent user review:** 100/103 is confirmed as the same person. All six
candidate whole-ID proposals are now user-confirmed, on eight reviewed pairs
(six positive, two negative). The original seven-pair ablation below remains
historical. The user also reported three within-ID switches in clip 1, recorded
in `configs/reid_identity_switch_reviews.json`. These remain **unrepaired**;
pair-level agreement does not establish whole-video identity correctness.
The candidate page includes focused inspection controls for these events.

The original 18 TrackTrack IDs per clip remain the baseline. This experiment
tests gallery selection and crop comparison; it does not rerun detection or
tracking, split source IDs, train a new model, or force a target identity count.
The saved source tracker used buffer 30. The user-selected 45 remains the
setting for future tracking runs.

## Development evidence

Seven explicit user judgments are retained in
`configs/reid_identity_reviews.json`: five same-person pairs and two
different-person pairs. These are **development labels**, used to choose
settings. They are not a held-out test set or exhaustive identity annotation.

| Variant | Known positive pairs recovered / 5 | Known false merges | Known positive pairs missed | Additional unreviewed merges | Total IDs, clips 1 / 2 |
|---|---:|---:|---:|---:|---|
| Original gallery baseline | 4 | 1 | 1 | 0 | 16 / 15 |
| One-to-one crop matching only | 4 | 2 | 1 | 0 | 16 / 14 |
| Five-frame spacing only | 4 | 0 | 1 | 1 | 16 / 15 |
| Spacing + one-to-one | 4 | 0 | 1 | 0 | 17 / 15 |
| Strict overlap rejection | 1 | 0 | 4 | 1 | 17 / 17 |
| Strict overlap + one-to-one | 1 | 0 | 4 | 1 | 17 / 17 |
| New full-video crops + nearest-neighbor | 4 | 0 | 1 | 1 | 16 / 15 |
| New crops + one-to-one, margin .003 | 4 | 0 | 1 | 1 | 16 / 15 |
| New crops + one-to-one, development margin .001 | **5** | **0** | **0** | **1** | **16 / 14** |

Threshold .98 was held fixed. The first eight variants use margin .003.
After inspecting those results, margins .0005, .001, .0015, .002 and .003
were examined for enriched one-to-one matching on the same development labels.
Margins .0005–.0015 had the same accepted pairs; .002–.003 retained the missed
219/228 connection. The selected .001 is a development calibration, not a
generalization claim. No direct user-label override enters the matcher.

**Lower counts alone can mislead:** one-to-one matching without better gallery
support reaches 14 IDs in clip 2 but incorrectly joins 212/228. That variant is
not selected.

## What changed

1. **Time-separated support.** At least five source frames separate gallery
   samples, rather than counting consecutive frames as separate supporting
   observations. Three samples are still required to attempt a connection.
   This does not guarantee different viewpoints or statistical independence.
2. **Full-video crop selection.** Select up to eight crops per original ID,
   across time bins, favoring confidence, size, sharpness and limited overlap.
   Reject boxes touching the image boundary within two pixels, crops smaller
   than 12×35 pixels, and overlap above .6. These are development heuristics,
   not measured visibility classifiers. The stricter .15 overlap ablation
   discarded too much evidence and is not selected.
3. **One-to-one crop assignment.** Match the two galleries with linear
   assignment on their cosine matrix, then take the median selected similarity.
   A reference crop cannot support multiple matches in that assignment. With
   unequal gallery sizes, unmatched excess crops do not contribute to the score.
4. **Whole-ID association.** Existing temporal exclusion, unmatched choices,
   per-end ambiguity rejection and one-to-one track association remain. Only
   the enriched development candidate changes the ambiguity margin to .001.

The actual Paper CLIP-ReIdent checkpoint is unchanged. New inference ran on
**117 / 130 selected crops** for clips 1 / 2 on CPU, FP32, four threads and
batch size four. Measured selection, loading and inference time was approximately
144.5 / 155.4 seconds. New embedding caches and their provenance are under
`outputs/gallery-enriched-crops-v1/`. Original experiments are preserved.

## The two errors

- **85/106:** the 106 track has only frames 147–149. It now supplies one
  time-separated sample, below the three-sample requirement. The candidate
  abstains instead of merging it. This is not proof the appearance encoder can
  distinguish the two people. The 85 gallery also includes overlapping people;
  pre-existing source-ID contamination remains a limitation.
- **219/228 versus 212/228:** these are referee identities in similar uniforms.
  One original 219 crop was cut off at the frame edge. Full-video selection
  supplies eight separated 219 crops and five 228 crops. Under enriched
  one-to-one matching, scores are .983050 for 219/228 and .981523 for 212/228,
  a separation of about .001527. The .001 development margin accepts the
  correct pair and rejects the competing one. Team/role labels alone would
  not separate the two referees.

The selected candidate proposes all five user-confirmed same-person joins:
88/101, 214/229, 218/225, 219/228 and 222/227. It also proposes **100/103 in
clip 1**, which was unreviewed at experiment time and is now user-confirmed. Resulting
counts 16 / 14 include that extra proposal; no whole-video accuracy is claimed.

## Review and reproduction

Serve `outputs/gallery-improvements-v2/` locally. `candidate/index.html` has
the synchronized videos and crop evidence. Root `index.html` compares all nine
variants, the seven reviewed decisions, and gallery changes. `ablation.json`
retains parameters, source/model/embedding hashes, scores, mappings and outcomes.
The candidate's `user-review.json` distinguishes reviewed results from unknown
new proposals.

```bash
python -m tacticalai.enrich_galleries_cli --output outputs/gallery-enriched-crops-v1
python -m tacticalai.gallery_ablation_cli --enriched outputs/gallery-enriched-crops-v1 --output outputs/gallery-improvements-v2
python -m tacticalai.gallery_reid_cli --cache-root outputs/gallery-enriched-crops-v1 --output outputs/gallery-improvements-v2/candidate --comparison one_to_one --min-spacing 5 --margin 0.001
python scripts/review_identity_pairs.py --results outputs/gallery-improvements-v2/candidate
```

Use fresh output directories. All inputs remain private/local. Locate the
reported switch boundaries, then collect untouched disappearance/reappearance
pairs to assess whether the chosen settings improve accuracy beyond these
development examples. ID 96's coach-to-player boundary was verified previously
at frame 131; the 85/91 boundaries remain null until checked. Whole-ID merges
must not stand in for time-scoped physical-person chains.
