# Project tasks and deferred improvements

Updated: 2026-10-18.

## Closed homography/projection checkpoint

The user concluded the `homography` benchmark after accepting the smoothed
two-clip presentation visually. Latest artifact:
`outputs/challenge-player-smoothed-v1/index.html`. The working pipeline is
KaliCalib challenge-checkpoint camera refinement, robust Kalman/offline RTS
camera stabilization, and light per-observed-track projection smoothing.
Manual role labels exclude referees/staff; automatic role classification and
verified calibration/identity accuracy remain unfinished. Preserve the current
artifacts/configurations as the comparison checkpoint.

The updated [project summary](PROJECT_CONTEXT.md#0-completed-checkpoint-and-future-research)
records the delivered result and prioritized research. See
[smoothing details](PROJECTION_SMOOTHING.md) for parameters, metrics and limits.

## Geometry / projection backlog for a future cycle

| Priority | Task | Evidence needed to close it |
|---|---|---|
| First | Small independent landmark/footpoint evaluation set | Reviewed visibility/uncertainty labels, separate tuning/evaluation frames, reprojection/metric errors, coverage and sample counts. |
| First | Raw vs camera-only vs camera+player smoothing ablation | Matched-frame comparisons of jitter, positional bias, turn/acceleration preservation and timing; distinguish offline RTS from causal filtering. |
| High | Calibration quality gates and uncertainty | Detect wrong-but-consistent fits using spatial support, camera plausibility and semantic court evidence; evaluate false acceptance/rejection and recovery. |
| High | Semantic court lines / documented landmark models | Improve on the accepted challenge checkpoint on independent data. The simple yellow-paint refinement is an unsuccessful baseline, not the selected method. |
| High | Shot changes, zooms, orientation and re-acquisition | Test cut detection, state resets, short-gap estimates and long-gap unknowns on representative longer footage. |
| High | Better player floor-contact points | Quantify bbox vs pose/segmentation error during occlusion, jumps and clipped visibility. |
| Medium | Forward/fixed-lag filtering and throughput | Compare latency and accuracy with offline RTS; report hardware and full pipeline timings rather than video playback FPS. |
| Medium | Cross-venue/generalization study | Evaluate varied markings/cameras/resolutions and verify model/dataset provenance before fine-tuning. |

## Current working decision

Continue the POC using TrackTrack with ReID and a 30-frame retention buffer as
the working candidate. This is a development choice, not a verified accuracy
winner. The original baseline config and all experiments remain preserved;
this task list does not change executable defaults.

The first baseline and the 16-run occlusion comparison have completed. See
[baseline results](BASELINE_RESULTS_20261015.md) and
[occlusion results](OCCLUSION_RESULTS_20261015.md).
Buffer 60 added no tracking change with ReID on these clips. At least one
wrong-person match is confirmed: clip 1's player 87 takes existing sideline
person ID 96 at 5.24 s, after player 87's last observation at 4.16 s. This is
not explained by track expiry alone.

The tasks below are recorded for future work so that tracking tuning does not
consume the entire POC. No task is completed merely by being listed here.

## Tracking and player-filtering backlog

| Priority | Task | Work to perform | Evidence needed to close it |
|---|---|---|---|
| High | Court boundary control | Add a court polygon with a generous margin and inside/boundary/outside categories. Apply filtering before association. Update polygons for camera movement; start with manually selected keyframes before automating. Keep uncertain boundary detections rather than discarding them confidently. | Check the 87-to-96 failure and both full clips. Record staff/crowd tracks removed and actual players wrongly excluded, especially at sidelines, during jumps and camera pans. |
| High | Jersey and role validation | Estimate team uniform appearance, distinguish players/referees/staff, and use reliable team/role history to discourage incompatible matches. Use confidence and temporal aggregation; allow unknown when crops are obscured or mixed. | Inspect same-team crossings, staff in similar colors, referees, and occluded crops. Compare missed players and wrong-person matches with the unfiltered baseline. |
| High | Diagnose the confirmed ID switch | Examine raw boxes/confidence and log candidate associations, appearance costs, spatial/motion gates and track state around clip 1 frames 104–140. Determine why track 87 failed to reconnect and track 96 accepted the player. | A frame-level explanation grounded in detections and matching decisions; current exports alone do not establish each internal cost's contribution. |
| High | Small evaluation/regression set | Preserve this user-identified event and add a representative sample of clear and difficult player occlusions, including successful cases. Label physical player identity, visible boxes, visibility, and uncertainty. Never invent boxes during full occlusion. | Report correct reconnections, wrong-person matches, fragmentation, missed detections and sample size on the same events for every candidate. Unique-ID count is not the score. |
| Medium | Targeted association tuning | After diagnosis, vary matching thresholds and appearance/motion weights one at a time. Keep detector, footage and other settings fixed. Do not simply increase retention or loosen every gate. | Improvement on annotated events without additional wrong-person matches or reduced player coverage; retain configs and runtime records. |
| Medium | Appearance encoder comparison | Compare the current generic yolo26n-cls.pt encoder against a compatible person/sports ReID encoder. Assess same-uniform confusion, crop quality, latency and model license. | Repeat the same evaluation with model hashes/configs; better embeddings are a hypothesis, not an assumed upgrade. |
| Medium | Detector failure checks | Check whether occluded/returning players are missed, low confidence, duplicated or merged into a mixed-person box. Test confidence/resolution/model size only where detection evidence warrants it. | Compare player detection coverage and false positives on labeled frames. Do not interpret more detections or prettier overlays as improvement. |
| Later | Longer absence / persistent identity | Separate run-local track IDs from physical player identities. Consider linking tracklets after track expiry using multiple independent cues. Jersey-number OCR is a separate experimental feature, not the meaning of jersey validation above. | Explicit handling of uncertain links; evaluate off-camera returns separately. New track IDs after long absences remain an accepted baseline limitation, not fulfillment of the broader identity requirement. |
| Later | Runtime and output usability | Profile appearance encoding, transfers and exports before optimization. Keep H.264 comparison videos and readable labels available locally and through Drive. Test stable per-track overlay colors as a display-only aid. | Record comparable hardware/settings and verified playable outputs. Playback rate is not processing throughput. |

## Continue the scoped POC

- [x] Complete the initial court-calibration feasibility prototype: manual
  anchors, pretrained automatic comparison, full two-clip mini-map and temporal
  stabilization. User accepted the smoothed visual checkpoint on 2026-10-18;
  quantitative accuracy remains to be evaluated in the future backlog above.
- [ ] Preserve the distinction between generic person tracks and validated
  players until boundary/role tasks are implemented. A plotted person may
  still be staff or a referee; court projection does not correct identity.
- [ ] Record calibration quality and footpoint visibility. Bottom-center box
  points are approximations; jumps and occlusion require uncertainty/unknowns.
  Do not project airborne balls with the floor homography or fill hidden
  player positions as observed measurements.
- [ ] Keep manual annotation small initially; defer exhaustive metrics while
  exploring geometry. Before claiming accuracy or an improved tracker,
  complete the relevant evaluation tasks above.
- [ ] Leave ball tracking, jersey-number recognition, long-term identity,
  off-screen trajectory inference and tactical recognition experimental or
  deferred, consistent with the existing POC roadmap and original brief.

Boundary control and jersey validation are important additions, but neither
alone solves all identity errors. The missing supporting work is association
diagnosis, a small regression set, and checking detector/appearance failures.
