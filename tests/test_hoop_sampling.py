import json

import pytest

from tacticalai.export import sha256
from tacticalai.hoop_sampling import refinement_frames
from tacticalai.hoop_tracking import KINDS, load_hoop_tracks


def test_refinement_uses_measured_ball_and_adds_approach_context():
    hoop = dict(detections=[dict(bbox_xyxy=[700, 150, 750, 180])])
    coarse = {index: hoop for index in range(0, 100, 5)}
    points = [dict(state="none") for _ in range(100)]
    points[10] = dict(state="predicted", measurement_xy=[720, 160])
    points[30] = dict(state="detected", measurement_xy=[300, 160])
    points[60] = dict(state="detected", measurement_xy=[780, 170])
    assert refinement_frames(coarse, points, 5, 50, 1000, 500) == set(range(45, 71))


def test_adaptive_loader_distinguishes_skipped_frames_and_checks_ball_cache(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    ball = tmp_path / "ball.json"
    ball.write_text("{}")
    evidence = tmp_path / "detections.jsonl"
    evidence.write_text("".join(json.dumps(dict(frame_idx=i, timestamp_s=i / 50, detections=[])) + "\n"
                                for i in (0, 4, 5)))
    manifest = dict(mode="adaptive_boxes", status="complete", sample_stride=5, prompts=list(KINDS),
                    source_sha256=sha256(video), ball_sha256=sha256(ball),
                    artifacts_sha256={evidence.name: sha256(evidence)})
    (tmp_path / "experiment.json").write_text(json.dumps(manifest))
    info = dict(frames=7, fps=50, width=1000, height=500)
    tracks, _ = load_hoop_tracks(tmp_path, video, info, ball)
    assert tracks == [[], None, None, None, [], [], None]
    ball.write_text("[]")
    with pytest.raises(ValueError, match="different ball cache"):
        load_hoop_tracks(tmp_path, video, info, ball)
