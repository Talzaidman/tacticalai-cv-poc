"""CPU checks for sweep geometry, temporal indexing, and threshold reuse."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch


spec = importlib.util.spec_from_file_location(
    'wasb_sweep', Path(__file__).parents[1] / 'scripts/wasb_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class SweepTests(unittest.TestCase):
    def test_tight_crops_cover_frame_and_preprocessing_stays_separate(self):
        views = sweep.wasb_views(640, 360, tiled=True, crop_fraction=0.4)
        self.assertEqual(len(views), 10)
        covered = np.zeros((360, 640), np.uint8)
        for x, y, width, height in views[1:]:
            self.assertEqual((width, height), (256, 144))
            covered[y:y+height, x:x+width] = 1
        self.assertTrue(covered.all())
        original = sweep.wasb_plan(640, 360, sweep.WASB_VARIANTS)
        focused = sweep.wasb_plan(640, 360, sweep.WASB_LOWRES_VARIANTS)
        self.assertEqual(len(original), 5)
        self.assertEqual(len(focused), 34)
        for (_, method, interpolation), names in focused.items():
            for name in names:
                config = sweep.WASB_LOWRES_VARIANTS[name]
                self.assertEqual(method, config.get('preprocessing', 'none'))
                self.assertEqual(interpolation, config.get('interpolation', 'linear'))
        reference_views = [key for key, names in focused.items() if 'WASB-tiles-020' in names]
        self.assertEqual(set(reference_views), set(original))

    def test_contrast_processing_preserves_image_and_source(self):
        rng = np.random.default_rng(24)
        image = rng.integers(50, 100, (144, 256, 3), dtype=np.uint8)
        original = image.copy()
        result = sweep.wasb_preprocess(image, 'clahe')
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.array_equal(image, result))

    def test_crop_coordinates_and_padding(self):
        heatmap = np.zeros((288, 512), np.float32)
        heatmap[144, 256] = 0.8
        detection = sweep.wasb_components(heatmap, 0.5, (400, 200, 600, 400))[0]
        self.assertAlmostEqual(detection['x'], 700)
        self.assertAlmostEqual(detection['y'], 400)
        heatmap[:] = 0
        heatmap[144, 0] = 0.8  # horizontal padding for this 3:2 crop
        self.assertEqual(sweep.wasb_components(heatmap, 0.5, (400, 200, 600, 400)), [])

    def test_temporal_edges_and_thresholds_share_inference(self):
        class FakeModel:
            calls = 0

            def __call__(self, batch):
                self.calls += 1
                assert batch.shape[1:] == (9, 288, 512)
                logits = torch.full((len(batch), 3, 288, 512), -20.)
                logits[:, :, 144, 256] = torch.logit(torch.tensor(0.4))
                return [logits]

        model = FakeModel()
        frames = [np.zeros((360, 640, 3), np.uint8) for _ in range(5)]
        with patch.object(torch.cuda, 'synchronize'), patch.object(torch.Tensor, 'cuda', lambda t: t):
            results = sweep.wasb_sweep(model, frames, batch=4)
        self.assertEqual(model.calls, 5)  # one batch per view, not one per threshold
        self.assertTrue(all(not d for d in results['WASB-base'][0]))
        for name in ('WASB-full-035', 'WASB-full-020', 'WASB-full-010'):
            detections, seconds = results[name]
            self.assertGreater(seconds, 0)
            self.assertEqual([len(d) for d in detections], [1]*5)
            self.assertTrue(all(abs(d[0]['x']-320) < 1e-5 for d in detections))
        self.assertTrue(all(len(d) == 5 for d in results['WASB-tiles-035'][0]))
        model.calls = 0
        with patch.object(torch.cuda, 'synchronize'), patch.object(torch.Tensor, 'cuda', lambda t: t):
            focused = sweep.wasb_sweep(model, frames, batch=4,
                                        variants=sweep.WASB_LOWRES_VARIANTS)
        self.assertEqual(model.calls, 34)
        self.assertEqual(focused['WASB-tiles-020'][0], results['WASB-tiles-020'][0])
        self.assertEqual(set(focused), set(sweep.WASB_LOWRES_VARIANTS))
        self.assertTrue(all(len(detections) == len(frames) for detections, _ in focused.values()))

    def test_notebook_compiles_and_tracker_accepts_lower_confidence(self):
        import ast
        import json
        notebook = json.loads((Path(__file__).parents[1] / 'notebooks/colab_ball_benchmark.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        helpers = ''.join(notebook['cells'][2]['source'])
        tree = ast.parse(helpers)
        tracker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'track')
        namespace = {'np': np, 'math': __import__('math')}
        exec(compile(ast.Module(body=[tracker], type_ignores=[]), '<tracker>', 'exec'), namespace)
        dets = [[dict(x=50., y=50., conf=0.3)]]
        self.assertEqual(namespace['track'](dets, 640, 360, 25, .5)[0]['state'], 'none')
        self.assertEqual(namespace['track'](dets, 640, 360, 25, .2)[0]['state'], 'detected')


if __name__ == '__main__':
    unittest.main()
