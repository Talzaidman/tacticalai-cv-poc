"""Whole-ID association must preserve continuity and forbid count inflation."""
import unittest
import numpy as np
from tacticalai.gallery_linking import associate_galleries, gallery_similarity


class GalleryLinkingTests(unittest.TestCase):
    def test_one_to_one_does_not_reuse_the_same_supporting_crop(self):
        a = np.array([[1., 0], [1., 0], [1., 0]])
        b = np.array([[1., 0], [0., 1], [-1., 0]])
        self.assertGreater(gallery_similarity(a,b), gallery_similarity(a,b,'one_to_one'))
        self.assertAlmostEqual(gallery_similarity(a,b,'one_to_one'), gallery_similarity(b,a,'one_to_one'))
        with self.assertRaises(ValueError):
            gallery_similarity(a,b,'unsupported')

    def test_unmatched_and_missing_gallery_ids_are_preserved(self):
        tracks = {1: dict(start=0, end=4), 4: dict(start=8, end=12), 9: dict(start=0, end=20)}
        mapping, _ = associate_galleries(tracks, {})
        self.assertEqual(mapping, {1: 1, 4: 4, 9: 9})

    def test_matching_two_existing_ids_reduces_count_without_allocating_ids(self):
        tracks = {1: dict(start=0, end=4), 4: dict(start=8, end=12), 9: dict(start=0, end=20)}
        galleries = {1: np.array([[1., 0]] * 3), 4: np.array([[1., 0]] * 3), 9: np.array([[0., 1]] * 3)}
        mapping, links = associate_galleries(tracks, galleries)
        self.assertEqual(mapping, {1: 1, 4: 1, 9: 9})
        self.assertEqual(sum(c['accepted'] for c in links), 1)

    def test_overlapping_lifetimes_block_even_identical_appearance(self):
        tracks = {1: dict(start=0, end=10), 4: dict(start=8, end=12)}
        mapping, links = associate_galleries(tracks, {i: np.array([[1., 0]] * 3) for i in tracks})
        self.assertEqual(mapping, {1: 1, 4: 4})
        self.assertEqual(links, [])

    def test_competing_equal_candidates_are_not_forced(self):
        tracks = {1: dict(start=0, end=4), 2: dict(start=0, end=4), 4: dict(start=8, end=12)}
        mapping, links = associate_galleries(tracks, {i: np.array([[1., 0]] * 3) for i in tracks})
        self.assertEqual(len(set(mapping.values())), 3)
        self.assertFalse(any(c['accepted'] for c in links))

    def test_long_gap_abstains_and_gallery_score_is_symmetric(self):
        a = np.array([[1., 0], [0., 1]])
        b = np.array([[1., 0]])
        self.assertAlmostEqual(gallery_similarity(a, b), gallery_similarity(b, a))
        mapping, _ = associate_galleries({1: dict(start=0, end=1), 2: dict(start=200, end=202)}, {1:a, 2:a})
        self.assertEqual(mapping, {1: 1, 2: 2})

    def test_assignment_can_link_a_temporal_chain_without_extra_identity(self):
        tracks = {1: dict(start=0, end=4), 4: dict(start=8, end=12), 7: dict(start=15, end=20)}
        mapping, links = associate_galleries(tracks, {i: np.array([[1., 0]] * 3) for i in tracks}, margin=0)
        self.assertEqual(mapping, {1: 1, 4: 1, 7: 1})
        self.assertEqual(sum(c['accepted'] for c in links), 2)


if __name__ == '__main__':
    unittest.main()
