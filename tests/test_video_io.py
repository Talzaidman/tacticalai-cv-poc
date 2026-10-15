"""Real CPU codec check; no model downloads or torch required."""
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV/numpy not installed")
class VideoTests(unittest.TestCase):
    def test_mp4_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(3):
                writer.write(np.full((48, 64, 3), i * 50, dtype=np.uint8))
            writer.release()
            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].shape, (48, 64, 3))
