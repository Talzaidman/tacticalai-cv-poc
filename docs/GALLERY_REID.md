# Whole-ID gallery matching — 2026-10-19

The initial results below are preserved as the baseline. See
[subsequent gallery improvements](GALLERY_IMPROVEMENTS_20261019.md) for
user-reviewed errors, controlled ablations, and the newer candidate.

This corrects the integration in the earlier segment-linking experiment.
The **original TrackTrack output has 18 IDs per clip**. Those IDs are retained
through all gaps by default. The new stage can merge existing identities; it
cannot allocate additional identities or split a source ID. Detection and tracking
were not rerun. The saved source used buffer 30; the new tracker configurations
remain at the user-selected buffer 45 for subsequent inference runs.

## Executed result

| Clip | Original IDs | Resulting identity hypotheses | Proposed merges |
|---|---:|---:|---|
| attack_clip | 18 | 16 | 88→101; 85→106 |
| attack_clip2 | 18 | 15 | 214→229; 218→225; 222→227 |

Arrows denote earlier/later original IDs; the resulting identity uses the
earlier ID. Original `track_id` fields remain unchanged; `player_id` is a
separate hypothesis. Counts include **all tracked people**, not just players.
The user's rough 13–14-person target is a useful diagnostic, not a forced cluster
count or annotated population. Accuracy metrics remain null.

Open `outputs/gallery-reid-final/index.html`, served locally, for the review.
The page shows whole-video before/after counts, cumulative distinct-ID curves,
synchronized views, selected-pair-only highlighting, frame stepping, slow motion,
reference crops and an optional all-IDs mode. It defaults to the highest-scoring
proposed connection. Video annotations are drawn interactively over clean H.264
footage, without crowded baked boxes and compound labels.

Sampled visual review shows plausible same-player evidence for black jersey 12
under IDs 88/101 and yellow jersey 10 under IDs 214/229. This is not exhaustive
identity ground truth. Existing mixed-person source tracks can contaminate a
gallery and are not repaired here. The previous segment-level 89→91 proposal is
not eligible as a whole-ID merge because their original lifetimes overlap.

## Method and provenance

- Reuse the actual Paper CLIP-ReIdent embeddings from
  `outputs/clip-reident-paper-v1/`, verified against source video/frame hashes.
  No second encoder inference or training was performed.
- Group sampled crop embeddings by the **original track ID**, across its whole
  lifetime. IDs with fewer than three cached crops are retained but not matched.
- Retain up to eight crops per ID. Split the available samples into temporal
  bins, choosing within each bin using detection confidence, crop area,
  overlap with other tracked boxes and Laplacian sharpness. These are heuristic
  quality indicators, not learned visibility probabilities. Existing cached
  samples bound coverage; we have not re-encoded every video frame.
- Compute nearest-neighbor cosine similarity from each gallery into the other;
  average the two directional medians. Nearest-neighbor gallery comparison is
  an established [DeepSORT pattern](https://github.com/nwojke/deep_sort/blob/master/deep_sort/nn_matching.py).
  Symmetric median aggregation is our explicit whole-track extension.
- Allow only chronologically disjoint track lifetimes, up to 150 missing frames.
  Overlapping lifetimes block whole-ID merging, even when some later segments
  look similar. This intentionally leaves ID stealing out of scope.
- Require gallery score ≥0.98 and a ≥0.003 margin over eligible alternatives
  at both ends. These development thresholds are not calibrated probabilities.
- Use SciPy linear assignment with explicit unmatched choices, one predecessor
  and successor per ID. This is **DeepSORT-style matching machinery**, not an
  execution or accuracy reproduction of the original DeepSORT tracker, and not
  a newly trained association network.

The earlier 33/55 segment labels are superseded as the whole-video identity
output. Historical results remain available for comparison; the learned encoder
itself has not changed. The new matcher processed both clips using cached features
in about eight seconds locally, including preparation of browser videos; this
excludes the previous CPU encoder inference time and is not an inference FPS result.

## Reproduction

From the repository, install the optional dependencies without changing the
existing Colab CUDA-compatible torch build:

```bash
pip install -e '.[clip-reident,geometry,preview]'
python -m tacticalai.gallery_reid_cli --cache-root outputs/clip-reident-paper-v1 --output outputs/gallery-reid-final
```

Use a fresh output path. `summary.json`, per-clip `mapping.json`,
`candidates.json`, `galleries.json`, `frames.jsonl` and `metadata.json` retain
the decisions, actual selected crops, hashes, model provenance and configuration.
`review-data.json` includes the cumulative-count series and original observations.
The checkpoint and footage remain local and Git-ignored.

Next evaluation: review every proposed pair and annotate actual people over the
two full clips. Report correct/incorrect connections and remaining fragmentation
alongside total identity counts. A count near 13–14 alone does not rule out
incorrect merges or identity switches.
