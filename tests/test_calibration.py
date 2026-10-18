import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from tacticalai.calibration import Calibration, fit, project_track, transform
from tacticalai.calibration_cli import render, video_info
from tacticalai.court import landmarks


COURT = dict(length_m=28., width_m=15., key_depth_m=5.8, key_width_m=4.9)
MATRIX = np.array([[15., 2., 30.], [1., 15., 30.], [.001, .002, 1.]])
NAMES = ["left_corner_far", "left_corner_near", "right_corner_far", "right_corner_near"]


def anchors(matrix=MATRIX):
    known = landmarks(COURT)
    return {n: dict(image_xy=transform([known[n]], matrix)[0].tolist(), use="fit") for n in NAMES}


def document():
    return dict(schema_version=1, court=COURT, video=dict(width=640, height=360, frames=5),
                max_interpolation_gap_frames=4,
                keyframes=[dict(frame_idx=0, anchors=anchors(), interpolate_to_next=True),
                           dict(frame_idx=4, anchors=anchors(MATRIX+np.array([[0, 0, 4], [0, 0, 0], [0, 0, 0]])))])


class CalibrationTests(unittest.TestCase):
    def test_known_projective_geometry_and_held_out_points(self):
        points = anchors()
        expected = transform([[14, 7.5]], MATRIX)[0]
        points["centre"] = dict(image_xy=(expected+[5, 0]).tolist(), use="check")
        inverse, diagnostic = fit(points, COURT)
        np.testing.assert_allclose(transform([expected], inverse), [[14, 7.5]], atol=1e-5)
        self.assertAlmostEqual(diagnostic["checks"][0]["error_px"], 5., places=4)
        self.assertIsNone(diagnostic["accuracy_metrics"])

    def test_degenerate_and_insufficient_correspondences(self):
        with self.assertRaises(ValueError):
            fit(dict(list(anchors().items())[:3]), COURT)
        bad = anchors()
        for i, a in enumerate(bad.values()):
            a["image_xy"] = [i, 10]
        with self.assertRaises(ValueError):
            fit(bad, COURT)

    def test_interpolation_is_bounded_explicit_and_refits_anchors(self):
        d = document()
        calibration = Calibration(d)
        matrix, quality = calibration.at(2)
        expected_pixels = (transform([[7, 8]], MATRIX)+transform([[7, 8]], MATRIX+np.array([[0, 0, 4], [0, 0, 0], [0, 0, 0]])))/2
        np.testing.assert_allclose(transform(expected_pixels, matrix), [[7, 8]], atol=1e-5)
        self.assertEqual(quality["method"], "interpolated_anchors_unvalidated")
        self.assertIsNone(quality["fit_rmse_px"])
        self.assertIsNone(calibration.at(5)[0])
        d["max_interpolation_gap_frames"] = 3
        self.assertIsNone(Calibration(d).at(2)[0])
        d["max_interpolation_gap_frames"] = 4
        d["keyframes"][0]["interpolate_to_next"] = False
        self.assertIsNone(Calibration(d).at(2)[0])

    def test_missing_or_changed_landmarks_do_not_enable_interpolation(self):
        d = document()
        # Both fits valid, but there are only three shared fit landmarks.
        d["keyframes"][1]["anchors"].pop(NAMES[0])
        d["keyframes"][1]["anchors"]["left_key_free_far"] = dict(image_xy=transform([[5.8, 5.05]], MATRIX)[0].tolist())
        self.assertIsNone(Calibration(d).at(2)[0])

    def test_duplicate_frames_bad_bounds_and_schema_are_rejected(self):
        for mutate in [lambda d: d["keyframes"].append(copy.deepcopy(d["keyframes"][0])),
                       lambda d: d["keyframes"][0].update(frame_idx=5),
                       lambda d: d.update(schema_version=2),
                       lambda d: d["keyframes"][0]["anchors"][NAMES[0]].update(image_xy=[-1, 10])]:
            d = document()
            mutate(d)
            with self.assertRaises(ValueError):
                Calibration(d)

    def test_projection_keeps_unknown_contact_and_never_infers_hidden_positions(self):
        matrix, quality = Calibration(document()).at(0)
        p = transform([[5, 6]], MATRIX)[0]
        row = dict(bbox_xyxy=[p[0]-10, p[1]-40, p[0]+10, p[1]], track_id=7, role="unknown")
        result = project_track(row, matrix, quality, COURT, (640, 360))
        np.testing.assert_allclose(result["court_xy_m"], [5, 6], atol=1e-5)
        self.assertEqual(result["role"], "unknown")
        self.assertEqual(result["foot_contact_visibility"], "unknown")
        self.assertNotIn("court_xy_m", row)
        self.assertIsNone(project_track(row, None, {"method": "unavailable"}, COURT, (640, 360))["court_xy_m"])
        row["bbox_xyxy"][3] = 359.8
        self.assertIsNone(project_track(row, matrix, quality, COURT, (640, 360))["court_xy_m"])

    def test_real_video_roundtrip_and_no_calibration_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root/"input.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25., (640, 360))
            for _ in range(5):
                writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
            writer.release()
            d = document()
            d["video"] = video_info(video)
            d["keyframes"] = d["keyframes"][:1]
            config = root/"calibration.json"
            config.write_text(json.dumps(d))
            frames = root/"frames.jsonl"
            row = dict(bbox_xyxy=[100, 80, 120, 160], track_id=7, role="unknown", timestamp_s=0)
            with frames.open("w") as stream:
                for i in range(5):
                    stream.write(json.dumps(dict(frame_idx=i, timestamp_s=i/25,
                                                 tracks=[dict(row, timestamp_s=i/25)] if i != 2 else []))+"\n")
            (root/"metadata.json").write_text(json.dumps(dict(status="complete", video_sha256=d["video"]["sha256"])))
            args = SimpleNamespace(video=video, calibration=config, tracks=frames, output=root/"out", frame=None)
            render(args)
            result = [json.loads(line) for line in (args.output/"frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(result), 5)
            self.assertEqual(result[2]["tracks"], [])
            self.assertIsNotNone(result[0]["tracks"][0]["court_xy_m"])
            self.assertIsNone(result[1]["tracks"][0]["court_xy_m"])
            self.assertEqual(json.loads((args.output/"metadata.json").read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                render(args)
            (root/"metadata.json").write_text(json.dumps(dict(status="complete", video_sha256="wrong")))
            args.output = root/"wrong"
            with self.assertRaises(ValueError):
                render(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
