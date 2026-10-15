"""Suite validation uses synthetic exports; no pretrained inference is simulated as measured accuracy."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_occlusion import CLIPS, TRACKERS, VARIANTS, summarize


class OcclusionTests(unittest.TestCase):
    def test_expected_variants_and_detector_drift(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for variant, settings in VARIANTS.items():
                for clip, count in zip(CLIPS, (150, 175)):
                    for tracker in TRACKERS:
                        folder = root/variant/clip/tracker
                        folder.mkdir(parents=True)
                        meta = dict(status='complete', frames_processed=count, truncated=False,
                                    detection_rows=0, track_rows=0, config=dict(conf=.1, **settings),
                                    video_sha256=clip, model_sha256='weights', versions={}, gpu='fixture',
                                    source={'fps': 25}, source_sha256={}, detections_sha256='same',
                                    tracker_config=dict(gmc_method='sparseOptFlow', **settings),
                                    effective_track_buffer_frames=settings['track_buffer'],
                                    reid_model_sha256='encoder' if settings['with_reid'] else None,
                                    timing={'frame_loop_s': 1, 'pipeline_fps': count})
                        (folder/'metadata.json').write_text(json.dumps(meta))
                        (folder/'frames.jsonl').write_text('\n'.join(json.dumps(dict(
                            frame_idx=i, timestamp_s=i/25, detections=[], tracks=[])) for i in range(count)))
                        for kind in ('detections', 'tracks'):
                            (folder/f'{kind}.csv').write_text('frame_idx,track_id\n')
                        (folder/'annotated.mp4').write_bytes(b'existence-only test fixture')
            self.assertTrue(summarize(root)['artifact_checks_passed'])
            path = root/'buffer60/attack_clip/botsort/metadata.json'
            meta = json.loads(path.read_text())
            meta['detections_sha256'] = 'drifted detector outputs'
            path.write_text(json.dumps(meta))
            report = summarize(root)
            self.assertFalse(report['artifact_checks_passed'])
            self.assertIn('buffer60/attack_clip/botsort: comparison mismatch detections_sha256', report['issues'])
