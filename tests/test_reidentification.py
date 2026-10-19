"""ReID preprocessing and linking checks; not pretrained accuracy measurements."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from tacticalai.clip_reident import MEAN, STD, preprocess
from tacticalai.tracklet_linking import link_tracklets, split_tracklets


class ReidentificationTests(unittest.TestCase):
    def test_rgb_conversion_black_padding_and_normalization(self):
        crop = np.zeros((20, 10, 3), dtype=np.uint8)
        crop[:] = [0, 127, 255]
        tensor = preprocess(crop)
        self.assertEqual(tensor.shape, (3, 224, 224))
        np.testing.assert_allclose(tensor[:, 100, 100], (np.array([255, 127, 0]) / 255 - MEAN) / STD, atol=1e-6)
        np.testing.assert_allclose(tensor[:, 100, 0], -MEAN / STD, atol=1e-6)
        with self.assertRaises(ValueError):
            preprocess(np.empty((0, 10, 3), dtype=np.uint8))

    def test_same_numeric_track_is_split_at_missing_observation(self):
        frames = [dict(frame_idx=i, tracks=[dict(track_id=7, frame_idx=i)] if i != 2 else []) for i in range(5)]
        segments = split_tracklets(frames)
        self.assertEqual([(s['start'], s['end']) for s in segments], [(0, 1), (3, 4)])

    def test_reconnect_after_gap_and_reject_overlapping_or_old_candidates(self):
        segments = [dict(segment_id=i, start=start, end=end)
                    for i, (start, end) in enumerate([(0, 5), (10, 15), (3, 7), (100, 105)])]
        features = {0: np.array([1., 0]), 1: np.array([1., 0]),
                    2: np.array([0., 1]), 3: np.array([1., 0])}
        identities, links = link_tracklets(segments, features, max_gap=20)
        self.assertEqual(identities[0], identities[1])
        self.assertNotEqual(identities[0], identities[2])
        self.assertNotEqual(identities[0], identities[3])
        self.assertFalse(any(c['before'] == 0 and c['after'] in {2, 3} for c in links))

    def test_ambiguous_reconnection_remains_separate(self):
        segments = [dict(segment_id=0, start=0, end=5), dict(segment_id=1, start=0, end=5),
                    dict(segment_id=2, start=10, end=15)]
        identities, links = link_tracklets(segments, {i: np.array([1., 0]) for i in range(3)})
        self.assertEqual(len(set(identities.values())), 3)
        self.assertFalse(any(c['accepted'] for c in links))
        self.assertTrue(any(c['reason'] == 'ambiguous' for c in links))

    def test_cli_preserves_observations_and_does_not_fill_hidden_frames(self):
        from tacticalai.reidentify_cli import run
        from tacticalai.export import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'source.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25, (96, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(np.full((64, 96, 3), 100, dtype=np.uint8))
            writer.release()
            source = root / 'run'
            source.mkdir()
            meta = dict(status='complete', video_sha256=sha256(video), source=dict(fps=25))
            (source / 'metadata.json').write_text(json.dumps(meta))
            frames = [dict(frame_idx=i, tracks=[dict(track_id=7 if i < 2 else 19,
                       frame_idx=i, bbox_xyxy=[20, 20, 40, 55])] if i < 2 or i > 5 else [],
                       detections=[]) for i in range(8)]
            (source / 'frames.jsonl').write_text('\n'.join(map(json.dumps, frames)))

            class FakeEncoder:
                checkpoint_sha256 = 'simulated'
                checkpoint = root / 'simulated.pth'

                def __init__(self, *args):
                    pass

                def encode(self, crops):
                    return np.tile([1., 0.], (len(crops), 1))

            args = SimpleNamespace(run=source, video=video, output=root / 'output', samples=2,
                                   min_observations=2, batch_size=1, device='cpu', cpu_threads=1,
                                   checkpoint=root / 'simulated.pth', threshold=0.85, margin=0.03,
                                   max_gap=45)
            args.checkpoint.write_bytes(b'simulated checkpoint')
            with patch('tacticalai.reidentify_cli.ClipReIdent', FakeEncoder), \
                 patch('tacticalai.reidentify_cli.importlib.metadata.version', return_value='simulated'), \
                 patch('tacticalai.reidentify_cli.subprocess.check_output', return_value='simulated'):
                run(args)
            result = [json.loads(line) for line in (args.output / 'frames.jsonl').read_text().splitlines()]
            self.assertEqual(result[0]['tracks'][0]['player_id'], result[7]['tracks'][0]['player_id'])
            for before, after in zip(frames, result):
                self.assertEqual(len(before['tracks']), len(after['tracks']))
                for original, linked in zip(before['tracks'], after['tracks']):
                    self.assertTrue(all(linked[k] == v for k, v in original.items()))
            self.assertEqual(json.loads((args.output / 'metadata.json').read_text())['status'], 'complete')
            cached = args.output
            args.output = root / 'reuse'
            args.reuse_embeddings = cached
            with patch('tacticalai.reidentify_cli.ClipReIdent', side_effect=AssertionError('must reuse')), \
                 patch('tacticalai.reidentify_cli.importlib.metadata.version', return_value='simulated'), \
                 patch('tacticalai.reidentify_cli.subprocess.check_output', return_value='simulated'):
                run(args)
            self.assertEqual(json.loads((args.output / 'metadata.json').read_text())['accepted_links'], 1)
            args.samples = 1
            args.output = root / 'mismatch'
            with patch('tacticalai.reidentify_cli.importlib.metadata.version', return_value='simulated'), \
                 self.assertRaisesRegex(ValueError, 'different crop sampling'):
                run(args)


if __name__ == '__main__':
    unittest.main()
