# TacticalAI CV POC — repository guidance for Codex

- This is an **early research/engineering proof of concept**, not a production application.
- User is a Python/Computer Vision developer working in **VS Code**, with **no local NVIDIA GPU**. Preferred GPU execution: **Google Colab**, ideally via its VS Code extension. Keep Python source modular and Git-friendly; use a notebook as an experiment driver, not as the only place code lives.
- **Authoritative scope:** `docs/POC_CV.pdf`. See `docs/PROJECT_CONTEXT.md` for the previous discussion, footage facts, engineering decisions, and known uncertainties. Read these files when planning POC work, not reflexively on every small edit.
- Videos: `data/raw/attack_clip.mp4` and `data/raw/attack_clip2.mp4`; preserve originals. Do not commit the videos publicly without permission.
- First objective: reproducible **player-only detection and tracking** experiments (YOLO26s baseline, BoT-SORT, TrackTrack; optionally Deep OC-SORT), track IDs, annotated MP4, machine-readable per-frame detections, and runtime metrics. Do not claim the experiment was run unless actually executed.
- Compare trackers on identical video input, detection model, confidence, and resolution settings. Store model/library versions and configs with each run. ReID and GMC settings must be reported explicitly.
- Then add court calibration / time-varying homography and court-coordinate projection of visible players. Distinguish detection, tracking, identification, and inferred off-screen positions.
- A planar court homography applies to **points on the floor**, not an airborne basketball. Foot-contact point is a player ground-position approximation, with confidence/visibility quality recorded.
- Treat ball tracking, jersey ID/ReID, off-frame trajectories, and tactical recognition as **experimental**. Mark missing measurements `null`/unknown rather than inventing confident ground truth.
- Prefer simple readable Python, small composable modules, parameterized CLI/scripts, and CPU-compatible postprocessing. Avoid web backends, databases, Kubernetes, and retraining before measuring pretrained baselines.
- Use `.gitignore` for source videos, model weights, cached data, runtime secrets, and outputs. Check model/dataset licenses before commercial use.
- Explain limitations and provide measurable outputs; don't equate visually pleasing overlays with verified accuracy. Confirm reported metrics against annotated ground truth.
- Present experiment results in a local web page with playable annotated videos and matching evidence, and give the user a direct link to review it.
