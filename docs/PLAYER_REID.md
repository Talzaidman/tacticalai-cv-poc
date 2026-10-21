# Player ReID experiments

Started 2026-10-18 on branch `player-reid`, from the accepted `homography`
checkpoint. These are experiment candidates, not measured winning settings.
The dedicated YOLO ReID configurations below have not been benchmarked.
CLIP-ReIdent has now run on both clips; see [executed trials](CLIP_REIDENT.md).
Independent identity accuracy evaluation remains pending.

## What is already known about occlusion

The existing [occlusion experiment](OCCLUSION_RESULTS_20261015.md) compared
30/60-frame buffers on identical detector outputs. With ReID enabled, all
four clip/tracker combinations were identical between those two buffers.
At the source 25 FPS, 30 frames is about 1.2 s and 60 about 2.4 s; the runner
records the effective buffer from the instantiated tracker.

Clip 1's confirmed 87-to-96 failure is a wrong-person reassignment, not simply
a newly allocated ID: ID 96 already belonged to a sideline person. The gap
from the last player-87 observation to the bad match was 27 frame intervals,
inside the 30-frame retention setting. Other events still need diagnosis.

A new ID after an overlap can result from a missed/merged detection, motion
prediction drift, rejected association, an appearance descriptor contaminated
by the occluder, or track removal. Increasing retention only addresses the last
cause and leaves more stale tracks competing for matches.

## Model shortlist

The current appearance encoder is `yolo26n-cls.pt`, a generic classifier.
The pinned local Ultralytics 8.4.158 encoder also supports dedicated
`yolo26{n,s,m,l,x}-reid.onnx` assets. The [official tracking documentation](https://docs.ultralytics.com/modes/track/)
lists them as ready-to-use tracking encoders. Availability and size alone
do not establish basketball accuracy.

| Candidate | Purpose | Integration status |
|---|---|---|
| ReID off | Motion/association control | Ready |
| `yolo26n-cls.pt` | Reproduce the current appearance baseline | Ready |
| `yolo26n-reid.onnx` | Small dedicated encoder | Config ready; inference pending |
| `yolo26s-reid.onnx` | First dedicated candidate to inspect | Config ready; inference pending |
| `yolo26m-reid.onnx` | Test whether extra model capacity helps | Config ready; inference pending |
| CLIP-ReIdent, basketball-trained ViT-L/14 | Domain-specific quality candidate | Adapter and offline linking: [experiment](CLIP_REIDENT.md) |
| Deep-EIoU's sports OSNet checkpoint | Sports appearance candidate | Needs adapter and checkpoint/preprocessing verification |
| PRTreID | Part visibility, team and role representations | Separate research candidate; soccer-to-basketball transfer unverified |

The [basketball challenge organizers](https://github.com/DeepSportradar/player-reidentification-challenge)
report 98.44% mAP for CLIP ViT-L/14 on the **2022 challenge set**. This is
crop retrieval on that split, not our MOT accuracy or a current universal
leaderboard claim. The 2023 challenge set was made harder.
The [winning CLIP-ReIdent repository](https://github.com/DeepSportradar/2022-winners-player-reidentification-challenge)
links checkpoints and its inference/evaluation code. Use the trained checkpoint,
not an unadapted CLIP embedding, for that comparison.

The [official Deep-EIoU repository](https://github.com/hsiangwei0903/Deep-EIoU)
provides `sports_model.pth.tar-60` as an OSNet ReID checkpoint and uses expanded
IoU association for sports motion. [PRTreID](https://github.com/VlSomers/prtreid)
jointly addresses ReID, team affiliation and role classification; its SoccerNet
results do not establish basketball performance. Verify code, weight and
training-data licenses before commercial adoption.

Recommendation: start with dedicated **s**, compare **n/m** and the existing
classifier with identical association settings, then integrate basketball-trained
CLIP-ReIdent as the domain-specific challenger. No model is designated best
until measured on our footage. Larger l/x encoders can follow if n/s/m results
justify their runtime cost.

## Ready configurations and execution

**21 October restoration:** `classifier-n.json` now uses the earlier executed
30-frame identity checkpoint with explicit original association settings. The
45-frame profiles below remain unbenchmarked alternatives; they are no longer
identical to the active classifier profile. See [restored baseline](RESTORED_IDENTITY_BASELINE.md).

`configs/reid/{off,classifier-n,dedicated-n,dedicated-s,dedicated-m}.json`
keep detector weights `yolo26s.pt`, confidence 0.1, NMS IoU 0.7, image size
1280, seed 0, FP32, TrackTrack, buffer **45** (1.8 s at 25 FPS) and GMC
`sparseOptFlow` fixed. The 45-frame buffer was selected by the user;
the historical 30/60-frame results above are unchanged.
Association remains the pinned package defaults for this first comparison.
These settings apply to **all detected people**; the detector's person class
does not isolate players from referees, coaches or spectators.

Run in the staged Colab GPU environment after installing `requirements.txt`
and this package with `pip install -e . --no-deps`. Use a new output directory
per invocation and copy outputs to persistent storage before runtime deletion:

```bash
python -m tacticalai.cli --video data/raw/attack_clip.mp4 --output outputs/reid-s-clip1 --config configs/reid/dedicated-s.json
python -m tacticalai.cli --video data/raw/attack_clip2.mp4 --output outputs/reid-s-clip2 --config configs/reid/dedicated-s.json
```

Repeat for all five configurations. The existing `run_occlusion.py` and baseline
notebook still run their original classifier/buffer matrix; they do not select
this new matrix automatically. `--tracker botsort` allows a separate tracker
comparison; hold its own association settings fixed across encoders.

The ONNX backend requires `onnx` and a compatible `onnxruntime-gpu` in Colab
(`onnxruntime` for CPU). Ultralytics may install these on first use. The runner
records the environment again after encoder initialization, the exact ONNX
checkpoint path/hash, effective crop size and runtime providers. Check for
`CUDAExecutionProvider`; a CUDA detector can coexist with a CPU ReID backend.
Archive `environment.txt` to reproduce the resolved optional dependencies.
Run short smoke checks first; these candidates have not yet been executed here.

## Matching controls: inspect the pinned code

The details below were checked against the installed **8.4.158** source,
especially `trackers/track_tracker.py`, `bot_sort.py`, and `utils/reid.py`.
Use that implementation when online generic tuning advice differs.

- **TrackTrack:** defaults include high/low/new detection thresholds
  0.6/0.25/0.7, `match_thresh=0.7`, `lost_match_thr=0`, appearance/motion
  weights 0.5/0.5, confidence weight 0.1 and angle weight 0.05.
  Its cost matrix forces pairs with predicted-box IoU **<= 0.10** to cost 1.
  A better embedding cannot bypass that gate at normal match thresholds.
  `appearance_thresh` and `proximity_thresh` are not read by this tracker;
  adding those BoT-SORT knobs would silently do nothing upstream.
- **BoT-SORT:** defaults are `proximity_thresh=0.5`,
  `appearance_thresh=0.8`, `match_thresh=0.8`. Higher proximity requires
  more spatial overlap. The pinned code computes embedding distance as
  cosine distance / 2 and compares it to `1 - appearance_thresh`.
  Thus 0.8 is not literally a raw cosine-similarity cutoff of 0.8.
  Motion-only associations can still win after the appearance branch is
  rejected, so this is not a hard identity-consistency veto.

Tracker-specific association knobs can now be supplied in a config, e.g.:

```json
"tracker_overrides": {
  "tracktrack": {"lost_match_thr": 0.85}
}
```

This is a **separate, unvalidated ablation**, after the encoder comparison.
The relaxed lost-track pass may recover missed links but also introduce wrong
matches; it does not remove the IoU gate. The runner rejects override keys
absent from that tracker's YAML, including those ineffective BoT-SORT knobs
on TrackTrack. Use top-level fields for model, buffer, ReID and GMC.

## Other techniques to test after the model comparison

1. Log each disappearance/reappearance: raw detection coverage, gap length,
   track state/removal reason, predicted IoU, appearance distance and final
   assignment cost. These internal diagnostics are proposed, not implemented.
2. Build each player's appearance gallery from several clear, separated views;
   suppress updates from heavily overlapping, tiny or blurred crops. A mixed
   crop can teach the track the wrong person's appearance.
3. Add confident team/role constraints before assignment. Team color can reject
   an opponent or sideline person but cannot distinguish teammates. Unknown
   role/team must remain unknown, with confidence recorded.
4. Compare expanded spatial search / uncertainty-aware motion for lost tracks,
   plus court-speed plausibility where homography is reliable. Do not force a
   match merely because a player is the nearest candidate.
5. For these offline clips, test one-to-one tracklet stitching using temporal
   compatibility, several appearance observations, motion and team evidence.
   Preserve source track IDs; export a separate hypothesized player identity,
   evidence and confidence, leaving ambiguous joins null.
6. Add jersey-number evidence only when readable and consistent across frames.
   Keep long-term player identity separate from short-term tracker IDs.

## Decide using measured recovery

Annotate real pre/post-occlusion identities on both clips, including same-team
crossings, partial overlaps, missed detections and the 87-to-96 regression.
Record actual frame indices and source boxes; do not treat the tracker output
as ground truth. Separate development events from held-out evaluation events.

For each candidate report correct reconnections / eligible annotated events,
wrong-person joins, fragmentation, unknown cases and coverage, plus runtime,
versions, checkpoint hashes, ReID/GMC settings and hardware. Compare actual
`detections_sha256` across runs; differing hashes invalidate an association-only
comparison. For a crop-level model comparison use the same query/gallery crops,
rank-1/mAP and same-team impostor errors; tune score thresholds per encoder on
development data. Different encoder score scales need not be interchangeable.
Report IDF1/HOTA only with suitable identity annotations and an evaluator.
Fewer IDs alone can mean more incorrect merges.
