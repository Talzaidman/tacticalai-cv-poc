"""Exercise runner lifecycle and real video/export I/O with simulated inference.

This deliberately does not assert that a pretrained model or real tracker ran.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import yaml
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV/numpy/PyYAML not installed")
class RunnerTests(unittest.TestCase):
    def test_raw_detections_empty_frames_and_confirmed_tracks(self):
        from tacticalai.runner import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            weights = root / "fake.pt"
            weights.write_bytes(b"simulated weights")
            cfg = root / "cfg/trackers"
            cfg.mkdir(parents=True)
            (cfg / "botsort.yaml").write_text("tracker_type: botsort\nmatch_thresh: 0.8\n")

            def boxes(count, tracked=False):
                value = SimpleNamespace(xyxy=np.array([[1, 2, 20, 30]] * count),
                                        conf=np.array([0.8] * count), cls=np.array([0] * count),
                                        is_track=tracked, id=np.array([7] * count) if tracked else None)
                value.cpu = lambda: value
                value.numpy = lambda: value
                return value

            class Model:
                ckpt_path = weights

                def __init__(self, *args):
                    self.callback = None
                    self.index = 0

                def add_callback(self, event, callback):
                    self.callback = callback

                def track(self, frame, **kwargs):
                    settings = yaml.safe_load(Path(kwargs['tracker']).read_text())
                    encoder = None
                    if str(settings.get('model', '')).endswith('.onnx'):
                        encoder = SimpleNamespace(imgsz=448, fp16=False, model=SimpleNamespace(
                            device='cpu', session=SimpleNamespace(
                                get_providers=lambda: ['CPUExecutionProvider'])))
                    self.predictor = SimpleNamespace(trackers=[SimpleNamespace(
                        args=SimpleNamespace(**settings),
                        max_frames_lost=settings.get('track_buffer', 30), encoder=encoder)])
                    if self.index == 2 and fail[0]:
                        raise RuntimeError("simulated inference failure")
                    raw = boxes(2 if self.index == 0 else 0)
                    result = SimpleNamespace(boxes=raw, plot=lambda: frame)
                    self.callback(SimpleNamespace(results=[result]))
                    # Emulate the callback replacing detections with confirmed tracks.
                    result.boxes = boxes(1, True) if self.index == 0 else boxes(0)
                    self.index += 1
                    return [result]

            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                         manual_seed=lambda seed: None,
                                         version=SimpleNamespace(cuda=None),
                                         backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True)))
            fake_ultra = SimpleNamespace(__version__="8.4.158", __file__=str(root / "__init__.py"), YOLO=Model)
            config = dict(model="fake.pt", device="cpu", seed=0, conf=0.1, iou=0.7,
                          imgsz=1280, max_det=100, half=False, gmc_method="sparseOptFlow")
            fail = [False]
            with patch.dict("sys.modules", {"torch": fake_torch, "ultralytics": fake_ultra}), \
                 patch("importlib.metadata.version", return_value="simulated"), \
                 patch("subprocess.check_output", return_value="simulated environment"):
                output = root / "success"
                run(video, output, config, "botsort")
                meta = json.loads((output / "metadata.json").read_text())
                frames = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
                self.assertEqual(meta["status"], "complete")
                self.assertEqual(meta["frames_processed"], 3)
                self.assertEqual(len(frames[0]["detections"]), 2)
                self.assertEqual(len(frames[0]["tracks"]), 1)
                self.assertIsNone(frames[0]["detections"][0]["track_id"])
                self.assertEqual(frames[1]["detections"], [])
                self.assertEqual(frames[1]["tracks"], [])
                reid_config = dict(config, with_reid=True, track_buffer=60,
                                   reid_model='yolo26n-cls.pt')
                run(video, root / 'reid-settings', reid_config, 'botsort')
                reid_meta = json.loads((root / 'reid-settings/metadata.json').read_text())
                self.assertTrue(reid_meta['tracker_config']['with_reid'])
                self.assertEqual(reid_meta['tracker_config']['model'], 'yolo26n-cls.pt')
                self.assertEqual(reid_meta['effective_track_buffer_frames'], 60)
                # Simulated ONNX download/backend: verify the exact supplied file,
                # providers and association overrides survive into run metadata.
                onnx_weights = root / 'downloaded-reid.onnx'
                onnx_weights.write_bytes(b'simulated ONNX weights')
                downloads = SimpleNamespace(attempt_download_asset=lambda name: str(onnx_weights))
                onnx_config = dict(config, with_reid=True, reid_model='yolo26s-reid.onnx',
                                   tracker_overrides={'botsort': {'match_thresh': 0.75}})
                with patch.dict('sys.modules', {'ultralytics.utils.downloads': downloads}):
                    run(video, root / 'onnx-settings', onnx_config, 'botsort')
                onnx_meta = json.loads((root / 'onnx-settings/metadata.json').read_text())
                from tacticalai.export import sha256
                self.assertEqual(onnx_meta['reid_model_sha256'], sha256(onnx_weights))
                self.assertEqual(onnx_meta['tracker_config']['model'], str(onnx_weights.resolve()))
                self.assertEqual(onnx_meta['effective_tracker_config']['match_thresh'], 0.75)
                self.assertEqual(onnx_meta['reid_runtime']['providers'], ['CPUExecutionProvider'])
                fail[0] = True
                with self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
                    run(video, root / "failed", config, "botsort")
                failed = json.loads((root / "failed/metadata.json").read_text())
                self.assertEqual(failed["status"], "failed")
