# Codex conversation kickoff — paste the following into Codex chat in VS Code

You are helping me develop a Computer Vision POC for TacticalAI, a possible basketball-video-analysis startup. This repository is the handoff from an earlier ChatGPT conversation; treat `docs/POC_CV.pdf` as the authoritative customer brief, `docs/PROJECT_CONTEXT.md` as the summarized technical discussion, and `AGENTS.md` as project instructions. Inspect the two videos at `data/raw/` and verify all assumptions instead of treating earlier visual impressions as measured results.

**My current goal is not the whole product.** I want to start with YOLO26s player detection and tracking on the two supplied clips, comparing `botsort.yaml` and `tracktrack.yaml` (optionally `deepocsort.yaml` next) under consistent parameters. I have no local GPU, so implement a maintainable Python repo that I can edit in VS Code and run on Google Colab GPU (including through Colab's VS Code extension).

Please:
1. Inspect the files, state concise assumptions and a minimal plan.
2. Set up modular Python scripts/config for running the detector/tracker experiment on a selected video, exporting annotated MP4 plus CSV/JSON of detections and track IDs and run metadata (duration, FPS, processing time, model versions/config, tracker config).
3. Provide a lightweight notebook for the Colab GPU execution and clear instructions for staging the code/videos in the remote runtime (VS Code Colab extension does not automatically synchronize local `.py` files).
4. Include simple CPU-only smoke tests where practical, and use pinned dependencies; do not overengineer.
5. Define a lightweight evaluation checklist for missed detections, identity switches, fragmentation, and computational cost. No invented performance numbers or results.
6. Defer court homography, ball tracking, ReID identity verification, and tactical understanding until the baseline is inspectable.

Before running GPU-heavy experiments, ensure paths are configured and outputs will survive Colab runtime deletion. Start by presenting the plan and then implement the minimal first baseline.
