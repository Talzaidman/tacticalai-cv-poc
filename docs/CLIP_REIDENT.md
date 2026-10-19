# CLIP-ReIdent disappearance / return experiment

This experiment runs the released basketball-trained **Paper ViT-L/14**
checkpoint on saved tracking crops, then proposes links between separate visible
tracklets. Source track IDs, boxes, detections and missing frames are preserved.
`player_id` is a separate, provisional identity hypothesis; it is not a jersey
number or verified player identity. ID stealing inside a tracklet is not repaired.

## Model provenance and reference behavior

- [Official winning code](https://github.com/DeepSportradar/2022-winners-player-reidentification-challenge),
  commit `e93af99ac3f7924a4b06bae4753b2f69408feea1`.
- [Released checkpoint archive](https://drive.google.com/file/d/1Gm5J19okhLdnZTQLUsjfYoI0rwrLQ09i/view),
  member `model/ViT-L-14_openai/Paper/weights_e4.pth`.
- Checkpoint SHA256: `4dcd441f9553ad6f11b2908134e3c538f24f4318b72b68baf267c15cc567ef4e`.
- Archive SHA256: `503e13aa796814099a4cf22f0a39fe0244abcbcd139b16b13ad291d287ea6267`.
- Its `result_server.txt` reports challenge mAP `0.9844143738769614`,
  rank-1 `0.9914529914529915`. This is historical **2022 crop retrieval**,
  not a 98% probability of reconnecting a player in our videos.

The adapter uses OpenCLIP 2.0.2, QuickGELU, the 224-pixel ViT-L/14 visual
encoder with projection removed, and 1024-dimensional L2-normalized features.
It loads every retained parameter strictly from the fine-tuned checkpoint;
there is no generic CLIP fallback or extra base-weight download.
Preprocessing follows the original validation path: BGR to RGB, preserve aspect
ratio with centered black padding, OpenCV `INTER_LINEAR_EXACT`, CLIP mean/std.
The upstream MIT notice is retained in [licenses/CLIP_REIDENT_MIT.txt](licenses/CLIP_REIDENT_MIT.txt).
Checkpoint/training-data commercial terms have not independently been established.

We are **not reproducing the challenge pipeline or its metric**: this first
matching experiment uses single-checkpoint features and mean tracklet descriptors,
without challenge re-ranking or an ensemble. Challenge re-ranking uses the
whole query/gallery set; it is not automatically an online reconnect method.

## Setup and commands

Keep Colab's compatible CUDA torch/torchvision. From the staged repository:

```bash
pip install -r requirements.txt
pip install -e '.[clip-reident]'
python scripts/prepare_clip_reident.py
```

The optional setup downloads the official approximately 3.8 GB archive and
extracts only the selected Paper checkpoint and its reference records. The
checkpoint alone is approximately 1.2 GB. Assets stay under ignored `models/`.
The preparation script writes hashes and provenance. Preserve models and outputs
in persistent storage if using an ephemeral Colab runtime.

Run against a complete saved tracker output and its exact original video:

```bash
python -m tacticalai.reidentify_cli --video data/raw/attack_clip.mp4 --run outputs/occlusion-20261015T204150Z-029a7a35/reid/attack_clip/tracktrack --output outputs/clip-reident-paper-v1/attack_clip --device cuda:0
python -m tacticalai.reidentify_cli --video data/raw/attack_clip2.mp4 --run outputs/occlusion-20261015T204150Z-029a7a35/reid/attack_clip2/tracktrack --output outputs/clip-reident-paper-v1/attack_clip2 --device cuda:0
```

Use `--device cpu --batch-size 4 --cpu-threads 4` locally. CPU is much slower;
inference is on selected crops, not every detection. Existing output directories
are rejected. Runtime, environment, source hashes, source-run metadata, checkpoint
hash and all matching parameters are saved.

`notebooks/colab_clip_reident.ipynb` is a small Colab driver with persistent
Drive model/output paths and configurable saved-run location. Refresh the source
archive with `scripts/prepare_colab.ps1` and stage it privately before use.

The source run is not rerun: these historical TrackTrack+classifier outputs used
**buffer 30**. Our new tracking configs remain **buffer 45**. Postprocessing can
link across either buffer, and `--max-gap` is its own independent time limit.
Use future buffer-45 runs as `--run` without changing the linking code.

## First linking baseline

1. Split source tracks at every missing observation, including gaps where the
   same numeric track ID later returns. Numeric continuity is not treated as GT.
2. Sample up to three evenly spaced observed crops per segment, requiring at
   least three observations. Record sampled frames and actual image crops.
3. Average normalized crop embeddings and normalize the segment descriptor.
4. Compare temporally compatible segments with 1–150 missing frames. Require
   cosine similarity >= 0.85 and a >= 0.03 margin over competing candidates
   in both directions. Accept mutual-best links only; prohibit temporal overlap
   inside a resulting identity component.
5. Leave rejected/short/ambiguous segments separate. Do not fill hidden frames.

The 0.85 threshold and 0.03 margin are **uncalibrated development choices**,
not confidence probabilities. This is a conservative first baseline, not an
optimal assignment solver. Team/role, court motion, clean-crop selection,
multiple-view galleries and confirmed jersey evidence are not part of this run.
Same-team appearance ambiguity and already mixed source tracks remain limitations.

## Review and evaluation

Open `review.html` inside each output folder for up to 100 candidate pairs,
before/after crops, temporal gaps, raw IDs and scores. `annotated.mp4` displays
`P<n>?` for hypothesized identity and `T<n>` for the preserved source track ID.
The MP4 uses the existing runner's `mp4v` codec; browser support varies, and
the crop table works independently of playback.

- `links.json`: all considered pairs, accepted/rejected status and reason.
- `tracklets.json`: source IDs, spans, sampled frames and identity hypotheses.
- `embeddings.npz`: sampled-crop embeddings and segment IDs, for CPU analysis.
- `frames.jsonl`: original observations plus identity/segment fields.
- `metadata.json` / `environment.txt`: complete provenance and measured timing.

Accepted-link counts are algorithm outputs, **not successful-recovery metrics**.
Separate links between different raw IDs from links between fragments of the
same raw ID. Manually annotate held-out disappearance/reappearance events and
same-team alternatives, then report correct recovery, false joins and abstentions.
No HOTA/IDF1 or whole-clip accuracy claim is justified without appropriate GT.

## Executed local trials — 2026-10-18

Both full clips were processed using the actual released Paper checkpoint on
CPU (torch 2.12.1+cpu), FP32, batch 4 and four torch threads per process.
The two jobs ran concurrently; these are not isolated hardware benchmarks.

| Clip | Source frames | Sampled crops | Tracklets | First run total seconds |
|---|---:|---:|---:|---:|
| attack_clip | 150 | 117 | 43 | 207.38 |
| attack_clip2 | 175 | 153 | 60 | 256.90 |

The initial trial, `outputs/clip-reident-paper-v1/`, used threshold 0.85 and
margin 0.03 and accepted **zero** links on either clip. Inspection showed tightly
clustered cosine scores near 1.0: an absolute 0.03 margin was too large for this
model's observed scale. This is not evidence that the model found no similar pairs.

After inspecting clip 1 candidate crops, a **development** trial used threshold
0.98 and margin 0.003. It reused exactly the same saved embeddings (no second
model inference). Both trials are preserved; this tuning is not held-out validation.

| Clip | Proposed links | Between different raw IDs | Between fragments of same raw ID |
|---|---:|---:|---:|
| attack_clip | 10 | 1 | 9 |
| attack_clip2 | 5 | 0 | 5 |

One promising clip 1 link is segment 7 / raw ID 89, frames 16–72, to segment 35 /
raw ID 91, frames 112–149: 39 missing frames (1.56 s), cosine 0.994244,
two-sided margins 0.010339 / 0.009856. Sampled crop inspection shows black
jersey **9** on both sides. Both segments display provisional `P8` in the
second trial. This is sampled visual evidence, not a whole-clip identity annotation.
Other highly ranked candidates include black jersey 12 under raw IDs 88/101,
but the conservative assignment has not accepted that connection.

Open `outputs/clip-reident-paper-v2/index.html` for both annotated videos and
candidate review tables. Separate H.264 browser copies are included. Artifact
checks verify original boxes, detections, source IDs, observation counts, frame
counts and missing frames are preserved. These checks do not establish link accuracy.

To reproduce the second trial without repeating inference:

```bash
python -m tacticalai.reidentify_cli --video data/raw/attack_clip.mp4 --run outputs/occlusion-20261015T204150Z-029a7a35/reid/attack_clip/tracktrack --output outputs/clip-reident-paper-v2/attack_clip --threshold 0.98 --margin 0.003 --reuse-embeddings outputs/clip-reident-paper-v1/attack_clip
```

The cache is accepted only when the checkpoint hash, source video/frames,
sampling parameters and segment order match. Use a fresh output path for a repeat.
Full per-crop features and rejected pairs remain available for evaluation.
