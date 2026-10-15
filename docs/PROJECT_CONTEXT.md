# TacticalAI basketball video analysis — complete handoff context

_Compiled from a ChatGPT discussion on 2026-10-15 for continuation in Codex in VS Code._

## 1. Situation and product vision

A former army colleague, now a developer, teamed up with a product professional (both work at Deloitte) and approached the user as a potential third cofounder/technical lead in computer vision. They are exploring a startup called TacticalAI (name appears in their POC document, sometimes TacticAI). The long-term product: a coach uploads basketball video; the system analyzes player/ball movements, scores and possessions, attacks and defense, player/team statistics, and play types such as pick-and-roll, pick-and-pop, layups, isolation, spot-up, and post-up.

The user agreed only to an initial scoped proof of concept, to assess technical feasibility and personal/team fit; no commitment to joining or to a full product.

The user has experience with Python, CV, YOLO, ML/DL, PyTorch, optimization and algorithm integration, but has not previously done multi-agent sports video analytics or tactical group-action reasoning.

## 2. Original brief — what they explicitly requested

Read the supplied `docs/POC_CV.pdf` for the **authoritative wording**. Summary:

- Exactly **two real basketball clips**, each covering one offensive/defensive possession/sequence rather than a fixed length.
- Requested capabilities in their 'required POC' section: court line detection / camera-to-court homography; ball location in relevant frames; distinguish players, teams and referees; identify specific players (e.g., jersey number); maintain player ID through occlusion or camera changes; predict position when a player is out of view; top-down animated court map for each player and ball.
- Bonus, explicitly optional: tactical play/action detection such as pick-and-roll, pick-and-pop, layup, isolation, etc.
- Deliverables: annotated video; top-down mini-map with moving dots/trajectories; structured JSON/CSV positions by frame/time; processing time; limitations, approach/models, and what would be needed to improve precision and scale.
- Acceptance concepts: identification accuracy, ID stability, processing time, understandable outputs, transparency about failures. No numeric thresholds or labeled ground truth specified.
- They emphasize a **small exploratory POC rather than complete production implementation**.

**Engineering scope interpretation (not an explicit client-approved change):** There is a mismatch between the 'small POC' description and hard requirements around stable identity through occlusion, airborne ball placement on court map, and off-screen position prediction. Recommendation: separate core demonstrator from experimental features and discuss expectations with the founders.

## 3. Supplied video assets — verified metadata

| File | Resolution | Duration | FPS | Frames |
|---|---:|---:|---:|---:|
| `data/raw/attack_clip.mp4` | 640x360 | 6.0 s | 25 | 150 |
| `data/raw/attack_clip2.mp4` | 1920x1080 | 7.0 s | 25 | 175 |

This metadata was confirmed with `ffprobe`; these are source files, not synthetic samples. Prior visual inspection found both to be basketball *broadcast-style* footage, not static top-down tactical footage.

- **Clip 1**: lower resolution, roughly half-court offensive sequence; main risks include tiny ball and unreadable jersey numbers. Camera movement appears limited compared to clip 2; do **not** assume static homography without measurement.
- **Clip 2**: higher resolution but dynamic camera pan following play / transition; homography must adapt over time; tracking across entering/exiting FOV and crowded overlaps is a challenge.
- These qualitative descriptions come from visual sampling, **not a measured CV baseline or manual annotation**. Inspect the real files again in Codex.

## 4. Important technical distinctions / constraints

**Detection != MOT != identity.** `track_id=7` is not necessarily jersey/player 7. Generic motion trackers can switch IDs when players are occluded, leave the FOV or cross. ReID appearance embeddings are imperfect because teammates often have indistinguishable jerseys. Jersey OCR is likely unreliable at 360p.

**Court projection:** use pixel foot-contact point (e.g., bottom center of bbox as initial baseline; better via segmentation/pose if warranted) and map to physical court XY through an estimated homography `H_t`. A camera pan/zoom means `H_t` varies by frame. In partially visible shots, calibration can be underconstrained; report unknown / quality.

**Ball:** a basketball is small, blurry, and occluded; YOLO general-class confidence may be poor. The ground-plane homography does not provide the correct physical court location of a ball in flight. Need separate semantics (image position vs hypothesized ground position vs unknown); do not pretend exact 3D localization from a single perspective.

**Off-frame:** can extrapolate only with growing uncertainty, cannot observe true motion while unseen. Keep measured and inferred positions distinct.

**Tactics:** pick-and-roll requires multi-player temporal relationships, possession / screen / roll behavior, not merely bounding boxes. Explicitly out of scope for phase 1.

**Evaluation:** qualitative annotated video alone is insufficient. Ideally annotate a representative sample for player detection and some identities/footpoints, count ID switches; evaluate detection precision/recall and, if sufficient GT, IDF1/HOTA, and metric court projection error. Make uncertainty and missingness explicit.

## 5. Tools and infrastructure discussion

- User has no capable local GPU and wants to use **Google Colab**.
- Preferred developer UX: **write/review source in VS Code**, connect a Jupyter notebook to a **Colab GPU runtime** via the official Colab VS Code extension, and manage modular `.py` code in a Git repository. The runtime does not automatically have local source files or videos; use explicit synchronization/upload or `git clone/pull` and separately stage video assets in Colab or Drive.
- Colab is ephemeral; avoid assumptions of persistent installed packages, files, checkpoints or environment. Preserve reproducible configs, dependencies and run outputs; sync artifacts back to persistent storage.
- CPU/local is suitable for geometry, lightweight transforms, visualization and evaluation. GPU preferred for PyTorch inference.
- No database, API, frontend or MLOps platform required for the initial two clips.

## 6. Tracker selection and current experiment plan

Reference: https://docs.ultralytics.com/modes/track/ (checked 2026-10-15; current docs can change)

Ultralytics documentation currently lists six built-in trackers:

- `bytetrack.yaml` — fast/simple, no ReID or camera motion compensation.
- `botsort.yaml` — linear motion, global camera motion compensation and optional ReID; **recommended first baseline** for moving broadcast camera.
- `ocsort.yaml` — observation-centric motion for abrupt/nonlinear changes, no ReID.
- `deepocsort.yaml` — OC-SORT + optional ReID and GMC.
- `fasttrack.yaml` — occlusion-aware variant, no ReID/GMC.
- `tracktrack.yaml` — default as of checked documentation, multi-cue association and optional ReID and GMC.

**First practical comparison:** use `yolo26s.pt`, same frame preprocessing / resolution, `classes=[0]` (COCO person), same detector confidence, and run separately with `botsort.yaml` and `tracktrack.yaml`, then `deepocsort.yaml` if practical. Track ReID and GMC config: defaults vary, and ReID may be off. Hold the detector output fixed for any fair comparison of association strategies, or at least keep all inference settings identical. Try `imgsz=1280` for the high-resolution clip, but profile performance and memory usage; not guaranteed to help.

**No experiments have been run yet as part of this handoff.** We have not measured HOTA, IDF1, ball recall, homography errors or GPU speed. No accuracy claims.

For user-friendly reproducibility, design commands/notebook cells that can run on Colab with `pip install` and a staged MP4. Do not assume Colab currently provides specific GPU hardware or exact library versions. Verify with `torch.cuda.is_available()` and inspect `ultralytics.__version__`.

## 7. Proposed incremental POC roadmap

1. Inspect raw assets and create a reproducible scaffold with pinned dependencies, a CLI/script, notebook and result directory. Use privacy-aware local storage and `.gitignore`.
2. YOLO26 person detection only; export per-frame `frame_idx`, `timestamp`, `xyxy`, `confidence`, `class`, plus source-model metadata; save annotated MP4.
3. Compare BoT-SORT and TrackTrack on identical inputs, measure runtime and count ID fragmentation/switches on representative sequences; do not equate transient tracker IDs with known players.
4. Team classification / referee differentiation (jersey colors plus heuristics/classifier), assess failures where colors overlap with background and spectators.
5. Court calibration and dynamic homography; project only observed ground-contact points; top-down mini-map and time-stamped structured JSON/CSV.
6. Experimental ball detection/tracking; jersey identification / ReID; estimation under occlusion with explicit uncertainty and nulls. Only after core evaluation.
7. Present outputs and failure analysis to founders. Confirm which experimental features to pursue; tactical recognition is bonus/future work.

## 8. Suggested structured output schema

This is an **illustrative proposal**, not a label dataset. One record per detected object/frame:

```json
{
  "frame_idx": 94,
  "timestamp_s": 3.76,
  "class": "player",
  "bbox_xyxy": [205, 83, 250, 183],
  "det_confidence": 0.92,
  "track_id": 7,
  "team_id": null,
  "jersey_number": null,
  "court_xy_m": null,
  "position_source": "unavailable",
  "court_projection_quality": null
}
```

If mapping a basketball, distinguish image coordinates, assumed court ground intercept, and true court position (unknown if airborne). Do not make up physics outputs.

## 9. Intellectual property / commercial caveats

POC is for a possible third cofounder contribution. No final partnership or paid work is assumed. Define scope and code/IP ownership before extensive work. Do not publish source footage without permission. Review Ultralytics AGPL/Enterprise license options and all model/dataset weights before product commercialization.

## 10. Suggested first Codex assignment

See `START_HERE.md` in repository root. The first task should **scaffold a reproducible baseline**, not implement the whole business product or falsely claim success. If Codex has no GPU, prepare the notebook/scripts and CPU smoke tests; run GPU evaluation only after Colab resources are connected.
