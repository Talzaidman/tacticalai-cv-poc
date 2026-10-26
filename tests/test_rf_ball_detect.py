"""Check RF-DETR view mapping and reusable outputs without model downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location('rf_ball_detect',
    Path(__file__).parents[1] / 'scripts/rf_ball_detect.py')
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)


class RoboflowTests(unittest.TestCase):
    def test_single_variant_selection_keeps_tiles_settings(self):
        class Model:
            def __init__(self):
                self.calls = []

            def infer(self, images, confidence):
                self.calls.append((len(images), confidence))
                return [SimpleNamespace(predictions=[SimpleNamespace(
                    class_name='ball', x=20, y=30, width=5, height=5, confidence=.8)])
                    for _ in images]

        frames = [np.zeros((720, 1280, 3), np.uint8)]
        baseline = rf.detect_clip(Model(), frames)['variants']['RF-DETR-tiles']
        model = Model()
        selected = rf.detect_clip(model, frames, variant_labels=['RF-DETR-tiles'])['variants']
        self.assertEqual(list(selected), ['RF-DETR-tiles'])
        self.assertEqual(selected['RF-DETR-tiles']['config'], baseline['config'])
        self.assertEqual(selected['RF-DETR-tiles']['detections'], baseline['detections'])
        self.assertEqual(model.calls, [(1, .1), (4, .1)])
        with self.assertRaises(ValueError):
            rf.detect_clip(model, frames, variant_labels=['unknown'])

    def test_sweep_changes_one_setting_and_reuses_views(self):
        class Model:
            def __init__(self):
                self.calls = []

            def infer(self, images, confidence):
                self.calls.append((len(images), confidence))
                return [SimpleNamespace(predictions=[SimpleNamespace(
                    class_name='ball', x=20, y=20, width=4, height=4, confidence=c)
                    for c in (0.08, 0.3) if c >= confidence]) for _ in images]

        model = Model()
        variants = rf.detect_clip(model, [np.zeros((720, 1280, 3), np.uint8)], sweep=True)['variants']
        self.assertEqual(set(variants), set(rf.RF_VARIANTS))
        self.assertEqual(model.calls, [(1, .1), (4, .1), (1, .05), (4, .05), (4, .05), (1, .05)])
        for result in variants.values():
            self.assertEqual(len(result['detections']), 1)
            self.assertTrue(all(d['conf'] >= result['config']['candidate_confidence']
                                for d in result['detections'][0]))
        self.assertEqual(variants['RF-DETR-tight-low']['detections'],
                         variants['RF-DETR-tight-track010']['detections'])
        self.assertEqual(variants['RF-DETR-tiles']['detections'],
                         variants['RF-DETR-tiles-track040']['detections'])
        self.assertEqual(variants['RF-DETR-tight-track010']['config']['tracker_min_conf'], .1)
        # Use spatially distinct weak detections to verify the lower raw threshold.
        class WeakModel:
            def infer(self, images, confidence):
                return [SimpleNamespace(predictions=[SimpleNamespace(
                    class_name='ball', x=20, y=20, width=4, height=4, confidence=.08)]
                    if confidence <= .08 else []) for _ in images]
        weak = rf.detect_clip(WeakModel(), [np.zeros((720, 1280, 3), np.uint8)], sweep=True)['variants']
        self.assertFalse(weak['RF-DETR-tight']['detections'][0])
        self.assertTrue(weak['RF-DETR-tight-low']['detections'][0])

    def test_tighter_crops_cover_all_edges(self):
        views = rf.crop_views(1280, 720, 'tight')
        self.assertEqual(len(views), 9)
        self.assertEqual(views[0], (0, 0, 512, 288))
        self.assertEqual(views[-1], (768, 432, 512, 288))
        for x in range(1280):
            self.assertTrue(any(ox <= x < ox+w for ox, _, w, _ in views))
        for y in range(720):
            self.assertTrue(any(oy <= y < oy+h for _, oy, _, h in views))

    def test_adapter_checks_backend_and_active_gpu_provider(self):
        backend = type('RFDetrForObjectDetectionONNX', (), {})()
        backend._device = 'cuda:0'
        backend._session = SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model = SimpleNamespace(_model=backend)
        runtime = rf.model_runtime(model)
        self.assertEqual(runtime['backend_class'], 'RFDetrForObjectDetectionONNX')
        backend._session = SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider'])
        with self.assertRaisesRegex(RuntimeError, 'fell back to CPU'):
            rf.model_runtime(model)

    def test_rejects_wrong_architecture_and_unverified_device(self):
        with self.assertRaisesRegex(RuntimeError, 'Expected RF-DETR backend'):
            rf.model_runtime(SimpleNamespace(_model=type('YOLO', (), {})()))
        backend = type('RFDetrForObjectDetectionTorch', (), {})()
        with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
            rf.model_runtime(SimpleNamespace(_model=backend))
        backend._device = 'cuda:0'
        self.assertEqual(rf.model_runtime(SimpleNamespace(_model=backend))['device'], 'cuda:0')

    def test_gated_sam_access_records_failure_without_stopping_summary(self):
        import json
        import tempfile
        import contextlib
        import io
        root = Path(__file__).parents[1]
        notebook = json.loads((root / 'notebooks/colab_ball_benchmark.ipynb').read_text())
        source = ''.join(notebook['cells'][8]['source'])
        token = 'dummy-test-credential'

        class GatedModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise OSError('403: awaiting review; ' + kwargs['token'])

        with tempfile.TemporaryDirectory() as tmp:
            scope = dict(RUN_OPTIONAL_MODELS=True, HF_TOKEN=token, json=json,
                         torch=SimpleNamespace(cuda=SimpleNamespace(
                             is_bf16_supported=lambda: False, empty_cache=lambda: None)),
                         RESULTS={'RF-DETR-full': {'clip': {'stats': {'frames': 3}}}},
                         CLIPS={'clip': None}, OUT=Path(tmp))
            scope['torch'].float32 = 'float32'
            output = io.StringIO()
            with patch.dict('sys.modules', {'transformers': SimpleNamespace(
                    Sam3VideoModel=GatedModel, Sam3VideoProcessor=GatedModel)}):
                with contextlib.redirect_stdout(output):
                    exec(compile(source, '<sam-access-check>', 'exec'), scope)
            self.assertIn('awaiting review', scope['RESULTS']['SAM3']['clip']['error'])
            self.assertIn('stats', scope['RESULTS']['RF-DETR-full']['clip'])
            self.assertNotIn(token, output.getvalue())
            self.assertNotIn(token, (Path(tmp) / 'sam3_setup_error.json').read_text())

    def test_class_filter_and_source_coordinate_mapping(self):
        def prediction(name):
            return SimpleNamespace(class_name=name, x=20, y=30, width=8, height=6, confidence=.8)
        response = SimpleNamespace(predictions=[prediction('ball'), prediction('player')])
        dets, names = rf.ball_candidates([response], [(100, 200, 300, 200)])
        self.assertEqual(len(dets), 1)
        self.assertEqual((dets[0]['x'], dets[0]['y']), (120, 230))
        self.assertEqual(names, {'ball', 'player'})
        with self.assertRaises(ValueError):
            rf.ball_candidates([], [(0, 0, 640, 360)])

    def test_full_frame_is_reused_and_both_outputs_cover_clip(self):
        class Model:
            calls = []

            def infer(self, images, confidence):
                self.calls.append(len(images))
                self.assert_confidence = confidence
                return [SimpleNamespace(predictions=[SimpleNamespace(
                    class_name='ball', x=40, y=40, width=8, height=8, confidence=.9)]) for _ in images]

        model = Model()
        result = rf.detect_clip(model, [np.zeros((360, 640, 3), np.uint8) for _ in range(2)])
        self.assertEqual(model.calls, [1, 4, 1, 4])
        self.assertEqual(model.assert_confidence, .1)
        full, tiled = result['variants']['RF-DETR-full'], result['variants']['RF-DETR-tiles']
        self.assertEqual(len(full['detections']), 2)
        self.assertEqual(len(tiled['detections']), 2)
        self.assertTrue(all(len(d) == 1 for d in full['detections']))
        self.assertTrue(all(len(d) == 4 for d in tiled['detections']))
        self.assertGreaterEqual(tiled['seconds'], full['seconds'])

    def test_embedded_child_script_is_current(self):
        import ast
        import json
        root = Path(__file__).parents[1]
        notebook = json.loads((root / 'notebooks/colab_ball_benchmark.ipynb').read_text())
        source = ''.join(notebook['cells'][6]['source'])
        tree = ast.parse(source)
        call = tree.body[1].value
        self.assertEqual(ast.literal_eval(call.args[0]), (root / 'scripts/rf_ball_detect.py').read_text())


if __name__ == '__main__':
    unittest.main()
