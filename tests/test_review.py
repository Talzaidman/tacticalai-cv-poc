import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.review_runs import review


class ReviewTests(unittest.TestCase):
    def test_missing_runs_report_failures_and_still_package(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = review(directory)
            self.assertFalse(report["artifact_checks_passed"])
            self.assertEqual(len(report["issues"]), 4)
            with zipfile.ZipFile(Path(directory) / "baseline-results.zip") as archive:
                self.assertIn("review.json", archive.namelist())

    def test_complete_exports_and_comparison_mismatch(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for clip, frames in (("attack_clip", 150), ("attack_clip2", 175)):
                for tracker in ("botsort", "tracktrack"):
                    folder = root / clip / tracker
                    folder.mkdir(parents=True)
                    meta = dict(status="complete", frames_processed=frames, truncated=False,
                                detection_rows=0, track_rows=0, video_sha256=clip, model_sha256="same",
                                config={}, versions={}, gpu="test", source={"fps": 25},
                                timing={"frame_loop_s": 1, "pipeline_fps": frames},
                                tracker_config={"with_reid": False, "gmc_method": "sparseOptFlow"})
                    (folder / "metadata.json").write_text(json.dumps(meta))
                    (folder / "frames.jsonl").write_text("\n".join(json.dumps(dict(
                        frame_idx=i, timestamp_s=i/25, detections=[], tracks=[])) for i in range(frames)))
                    for kind in ("detections", "tracks"):
                        (folder / f"{kind}.csv").write_text("frame_idx,track_id\n")
                    (folder / "annotated.mp4").write_bytes(b"existence-only fixture")
            self.assertTrue(review(root)["artifact_checks_passed"])
            path = root / "attack_clip/tracktrack/metadata.json"
            meta = json.loads(path.read_text())
            meta["config"] = {"conf": 0.9}
            path.write_text(json.dumps(meta))
            report = review(root)
            self.assertFalse(report["artifact_checks_passed"])
            self.assertIn("attack_clip: comparison mismatch/missing config", report["issues"])
