# TacticalAI basketball video analysis — complete handoff context

**Latest combined presentation:** `outputs/players-ball-control-v6/` restores the
automatic team-colored silhouettes/person captions and mini-map, while retaining
v5's solid assigned-player box and orange ball marker/trail for both clips. Filled
assignments share the same style, per user preference, with evidence flags preserved.
Only the assigned player has a bbox overlay. The map ring is the controller's floor
position; airborne ball coordinates are never projected. Source detections, poses,
association and smoothing remain unchanged. The older display preferences below
describe earlier versions; use `--player-presentation clean` to restore v5.

**Current display preference:** `outputs/players-ball-control-v5/` shows only a
solid assigned-player box plus ball marker and short trail on original footage,
for both clips. User explicitly requested identical styling for filled assignments
and ball positions. No dashed/cyan/debug status overlays or other player/map
overlays are shown. Inference flags and raw/smoothed assignments remain preserved
in JSON/CSV; uniform styling is not verified measurement accuracy. The v4
association and continuity logic is unchanged.

**Dribble continuity update:** Current local output is
`outputs/players-ball-control-v4/`. A bounded offline pass fills only unknown gaps
bracketed by measured support for the same controller, with nearby ball evidence,
unchanged ball/person/camera segments and no strong rival evidence. Gaps are capped
at 0.60 s; missing player boxes may be interpolated for at most 0.12 s and are
explicitly inferred rather than new detections. Dashed cyan boxes distinguish
continuity from current support; raw assignments remain in `possession_raw`.
Clip 2's reviewed P221 dribble changes from five runs to one by filling 21 frames;
this is a continuity metric, not accuracy. All detector/pose/player caches remain
unchanged. See [current method](LOCAL_BALL_FUSION.md).

**Wrist association update (2026-10-26):** User authorized local model downloads
and computation ("on this machine you can do whatever") for this CV work; the
older local-assets-only download restriction is superseded. The latest local
output is `outputs/players-ball-control-v3/` (<http://127.0.0.1:8894/>).
Official YOLO26n-pose ran on CPU and cached near-ball wrists under
`outputs/ball-contact-pose-v1/`. Wrist/silhouette geometry and temporal measured
support replace broad-box proximity; brief continuity is explicitly inferred.
Ball/player source measurements stay unchanged. Clip-2 assignment coverage is
42 measured-support + 16 continuity / 175 frames, not possession accuracy.
Some accepted ball detections remain wrong, including frame 96. Both clips were
rendered and fully decoded; 36 relevant tests pass. See
[current method and local reproduction](LOCAL_BALL_FUSION.md).

**Local continuation (2026-10-26):** All 15 files from the latest Colab
`ball-notebook-20261026-212056` result are now archived locally under
`outputs/colab-ball-20261026-212056/`, with transfer hashes. Continue postprocessing
on this PC using `python -m tacticalai.ball_fusion_cli`; GPU inference stays optional
in Colab. Completed `outputs/players-ball-control-v1/` combines the exported RF-DETR
tiles/Kalman ball with the unchanged automatic `team-colors-floor-v1` player cache.
Green player boxes mark experimental repeated-body-proximity control; unknowns
remain explicit and inferred ball points cannot establish control. No model
inference was rerun. Both full clips decode correctly; control is estimated on
13/150 and 16/175 frames, not verified possession accuracy. Ball court XY remains
null; only the controller's floor point receives a map ring. See
[local method, outputs, checks and reproduction](LOCAL_BALL_FUSION.md).

**Ball experiment added (2026-10-25):** Independent ball tracking and experimental
body-proximity controller estimation now have a reproducible CPU/Colab CLI.
The minimap ring uses the estimated controller's floor position, never an
airborne-ball homography. Unknowns remain explicit. This checkout lacked prior
automatic output caches, so a new BoT-SORT/no-ReID + automatic KaliCalib baseline
was used, not the earlier accepted player identity setup. See
[method and reproduction](BALL_TRACKING.md). Earlier statements that ball
tracking is unimplemented are historical; accurate possession, hand contact,
3D ball position, jersey/roster identity and tactics remain unverified or absent.

**Ball quality correction:** User review exposed crowd false positives in the
basketball-specific YOLO26m run. Its 106/150 and 132/175 observed-frame counts
were not true-ball detection rates. A local-only camera-compensated background
filter now has separate full-clip output in `outputs/ball-motion-filter-v1`,
with basic playback at <http://127.0.0.1:8892/>. On eight draft reserved visible
reference frames, accepted false candidates fell 42 -> 4 but recall remained
1/8; clip 1 localized 0/4. Do not present this as solved ball tracking or
possession. No labels enter inference; draft labels only measure/tune
development diagnostics. The user declined resuming another model download;
use local assets only unless permission changes. Source people/cameras/raw
detections remain unchanged, ReID off and GMC sparseOptFlow inherited from the
fresh baseline. See [results and reproduction](BALL_TRACKING.md).

**Latest rendering update (2026-10-21):** `outputs/team-colors-floor-v1/` uses
consistent team colors for silhouettes/captions/minimap (A blue, B amber,
referees purple, unknown gray). Basic MP4 playback is at port 8890. An automatic
floor-context display filter reduces crowd/sideline overlays without changing
or deleting measurements. Existing predicted players and uncertain/occluded
context are retained; some sideline referees can be hidden. Development head-point
coverage changes 99/101 -> 93/101, partial remains 24/25; not filter accuracy.
89 CPU tests pass. See [method and reproduction](TEAM_COLORS_FLOOR_20261021.md).

**Latest display update (2026-10-21):** `outputs/unlinked-people-v2/` adds actual
unlinked YOLO26s detections to the role/team baseline, with cached segmentation
or a box fallback. Original identity/role/team/position decisions are preserved.
Unlinked identities/classes/court positions stay null. Development review now
uses direct MP4s in a basic player at port 8889 with notes in chat; avoid building
new elaborate investigation pages unless requested. Existing point coverage is
72/101 -> 99/101 (partial 8/25 -> 24/25), a development diagnostic, not accuracy.
See [method, limitations and reproduction](UNLINKED_PEOPLE_20261021.md).

**Latest feature checkpoint (2026-10-21):** Automatic role/team classification
and consolidated exports now build on the unchanged restored identity baseline.
Review http://127.0.0.1:8888/ (`outputs/roles-teams-final-v1/`). Generic CLIP ViT-L/14
roles plus masked torso-color grouping produce anonymous clip-local Team A/B;
no manual labels enter inference. The small development spot check has 39/41
correct roles and 29/31 correct teams including abstentions, with two unknowns
in each task; this is not held-out-video accuracy. Both full clips, CSV/JSONL,
evidence, scoped historical/current timings and configurations are available.
76 CPU tests pass. See [execution, limitations and reproduction](ROLES_TEAMS_20261021.md).
The older checkpoints below remain historical; ball, jersey/roster identity,
off-screen prediction and tactics are still unimplemented.

_Compiled from a ChatGPT discussion on 2026-10-15 for continuation in Codex in VS Code._

**Current continuation status (2026-10-18):** The homography/projection benchmark
on branch `homography` is concluded at the user's request. The user accepted
the smoothed two-clip presentation as the current visual POC checkpoint. This
is visual acceptance, not verified calibration/player accuracy or completion
of every requirement in the original brief. Preserve the delivered artifacts
and configurations; further experiments belong to the future-work backlog.
The original handoff in sections 1–10 below remains historical.

## 0. Completed checkpoint and future research

### Current deliverable and working pipeline

- Open `outputs/challenge-player-smoothed-v1/index.html` locally for both full
  clips, player track boxes, colored track IDs and the top-left projected
  mini-map. It includes original/smoothed switching, frame-synchronized
  side-by-side comparison, slow motion and fullscreen playback.
- Tracking remains the existing YOLO26s / TrackTrack + ReID
  (`yolo26n-cls.pt`), buffer 30, GMC `sparseOptFlow`. No player inference was
  rerun during geometry/smoothing work. This is the development candidate,
  not an accuracy winner. Known wrong-person matches and duplicate/fragmented
  IDs remain, including clip 1's player 87 taking sideline-person ID 96.
- Automatic calibration uses KaliCalib's **challenge checkpoint**, with the
  published winning-channel centroid extraction, 35 px RANSAC at 960×540,
  and camera refinement. The challenge checkpoint was better on the sampled
  review frames than the test checkpoint. Failed estimates use null, not the
  upstream fixed mean-camera fallback. The first YOLO court model and custom
  strict calibration baseline were insufficient; simple yellow-line refinement
  was also unreliable (only 2/30 corrections accepted in its second iteration).
- Stabilization uses robust constant-velocity Kalman filtering plus an offline
  RTS backward pass. It smooths court positions of nine fixed image probes,
  refits homographies, and then lightly filters contiguous observed player
  track segments. It never averages homography coefficients, joins IDs, or
  fills missing player detections. Short supported camera gaps, up to five
  frames, are explicitly labelled estimated. Camera-cut boundaries reset filters.
- Both clips retain all 150/175 frames at 25 FPS. The raw challenge model
  supplied camera estimates on 150/168 frames. The smoothed presentation has
  supported camera estimates on 150/175 frames, including 15 explicitly estimated
  clip 2 frames (eight rejected outliers plus seven original gaps). All
  1,358/1,894 original person-track observations are retained. Role filtering is
  sampled **manual annotation**, not an automatic player/referee classifier.
- The median projected-player second-difference roughness fell approximately
  98% on matched observed triples. This is a smoothness proxy, not accuracy.
  Draft-click agreement improved slightly on clip 1 but worsened slightly at
  clip 2's only checked frame; a smooth projection can still be wrong.
- All 31 CPU tests passed; full MP4 decode/frame counts, displayed populations,
  unchanged IDs/roles/boxes, raw coordinate preservation and estimated-position
  labels were checked. These are software/artifact checks, not ground-truth
  validation. Playback at 25 FPS is not processing at 25 FPS.

Detailed records and reproduction:
[manual geometry](HOMOGRAPHY.md),
[automatic baseline](AUTOMATIC_HOMOGRAPHY_20261018.md),
[checkpoint/refinement comparison](KALICALIB_REFERENCE_20261018.md),
[full challenge presentation](CHALLENGE_PRESENTATION.md), and
[temporal smoothing](PROJECTION_SMOOTHING.md).
The unsmoothed reference remains at
`outputs/challenge-player-presentation-v1/index.html`. Source footage and prior
experiments remain unchanged; model assets and output videos stay Git-ignored.

### Prioritized improvements and necessary research

| Priority | Research / improvement | Next investigation and evidence required |
|---|---|---|
| First | Independent calibration and player-position evaluation | Annotate visible floor landmarks and player foot-contact points across both clips, including pans, occlusions and difficult frames. Keep development/tuning frames separate from evaluation frames. Report image reprojection error, court-coordinate error where reference is defensible, supported/estimated/unknown coverage, and sample sizes. Existing draft anchors and pleasing overlays are insufficient ground truth. |
| First | Smoothing versus real motion | Ablate raw projection, camera smoothing only, and camera-plus-player smoothing on the same observations. Measure jitter reduction alongside position bias, turning/acceleration attenuation, onset/stop timing, and endpoint behavior. Tune process/measurement noise against held-out evidence rather than the smoothness score alone. |
| High | Automatic calibration quality and uncertainty | Test whether inlier residuals, spatial support, camera plausibility and semantic line agreement can identify a confidently wrong homography. Validate uncertainty and rejection thresholds. Investigate camera-parameter/pose filtering versus the current probe representation and include lens distortion / actual court dimensions where needed. |
| High | Court-line / landmark evidence | Investigate a documented basketball landmark detector or semantic court-line segmentation followed by geometric template refinement. Handle logos, painted areas, partial views and occlusion. The simple venue-specific yellow-pixel optimizer did not generalize even across these samples; do not promote it without measured improvement. Compare pretrained methods before considering fine-tuning. |
| High | Camera motion, cuts and recovery | Test abrupt cuts, zooms, shake, replays, camera-side/orientation changes and longer unavailable intervals. Validate shot detection, initialization and re-acquisition. Keep off-support estimates unknown and distinguish measured model outputs from reconstructed camera estimates. The current smooth-motion assumption is supported only within the supplied continuous shots. |
| High | Player ground-contact estimation | Compare bbox bottom centre with ankle/pose or segmentation-based floor contact, especially for jumping, loose boxes and occluded legs. Evaluate visibility/confidence and resulting metric projection error; court smoothing cannot fix an incorrect footpoint. |
| High | Player roles, teams and stable identity | Replace sampled manual role labels with evaluated automatic classification; combine court boundaries and reliable role/team cues with tracking. Diagnose the confirmed 87-to-96 switch and duplicate tracks before further matching-threshold changes. Track IDs remain separate from player/jersey identity. See the tracking backlog for regression events and evaluation. |
| Medium | Causal/live filtering and runtime | Compare forward Kalman or fixed-lag filtering with offline RTS. Measure lag, recovery, error and end-to-end processing time on specified Colab hardware, including detector, ReID, calibration and exports. The current RTS result uses future frames and is not a live-inference demonstration. |
| Medium | Generalization, data and model provenance | Evaluate longer footage, different venues/marking colors, resolutions and camera styles. Check documented landmark schemas, training overlap, pretrained model/dataset suitability and licenses. If fine-tuning becomes necessary, collect diverse footage and independent validation; the two supplied clips alone are not a generalization dataset. |
| Deferred | Ball, jersey OCR, off-screen trajectories and tactics | Keep these as separate experimental workstreams. A floor homography cannot locate an airborne ball in 3D. Hidden positions must remain explicit uncertain inference rather than observed ground truth. Tactical recognition follows validated spatial/identity measurements. |

The next engineering cycle should start with the small independent evaluation
set and a smoothing/geometry ablation, not more cosmetic overlay tuning. See
[PROJECT_TASKS.md](PROJECT_TASKS.md) for actionable tasks and closure evidence.

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
