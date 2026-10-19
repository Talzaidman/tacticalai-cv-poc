"""Require genuinely separated observations rather than adjacent-frame duplicates."""
import unittest
from tacticalai.enrich_galleries_cli import select_samples


class GallerySamplingTests(unittest.TestCase):
    def test_three_adjacent_frames_do_not_form_three_independent_views(self):
        samples=[dict(frame=f,quality=1) for f in (147,148,149)]
        self.assertEqual(len(select_samples(samples,spacing=5)),1)

    def test_sampling_retains_time_coverage_and_spacing(self):
        samples=[dict(frame=f,quality=1+f%7) for f in range(50)]
        chosen=select_samples(samples,maximum=8,spacing=5)
        self.assertLessEqual(len(chosen),8)
        self.assertGreaterEqual(len(chosen),3)
        self.assertTrue(all(b['frame']-a['frame']>=5 for a,b in zip(chosen,chosen[1:])))
        self.assertLess(chosen[0]['frame'],10)
        self.assertGreater(chosen[-1]['frame'],35)


if __name__=='__main__':
    unittest.main()
