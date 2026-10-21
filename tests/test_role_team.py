"""Abstention and label isolation for automatic role/team enrichment."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from tacticalai.role_team import (classify_identities, decide_role, decide_team,
                                 fit_teams, select_samples, square_pad, torso_color)
from tacticalai.role_team_cli import enrich, flat_record, person_summaries


CONFIG = json.loads(Path('configs/role_team.json').read_text())


def samples(role='player', color=None, count=4):
    scores = {k: .22 for k in CONFIG['role_prompts']}
    scores[role] = .30
    return [dict(role_scores=scores.copy(), torso_lab=color) for _ in range(count)]


class RoleTeamTests(unittest.TestCase):
    def test_full_body_padding_preserves_all_original_pixels(self):
        crop = np.full((70, 20, 3), 240, np.uint8)
        padded = square_pad(crop)
        self.assertEqual(padded.shape, (70, 70, 3))
        np.testing.assert_array_equal(padded[:, 25:45], crop)
        self.assertTrue(np.all(padded[:, :25] == 127))

    def test_roles_abstain_on_sparse_ambiguous_and_inconsistent_evidence(self):
        self.assertIsNone(decide_role(samples(count=2), CONFIG)['role'])
        ambiguous = samples()
        for s in ambiguous:
            s['role_scores']['referee'] = .299
        self.assertIsNone(decide_role(ambiguous, CONFIG)['role'])
        inconsistent = samples(count=2) + samples('referee', count=2)
        self.assertIsNone(decide_role(inconsistent, CONFIG)['role'])
        self.assertEqual(decide_role(samples('referee'), CONFIG)['role'], 'referee')

    def test_no_forced_two_team_split_for_one_uniform_or_small_group(self):
        self.assertIsNone(fit_teams([[30, 0, 0]]*6, CONFIG))
        self.assertIsNone(fit_teams([[20, 0, 0], [80, 0, 0]], CONFIG))
        self.assertIsNone(fit_teams([[20, 0, 0]]*5 + [[80, 0, 0]], CONFIG))

    def test_two_teams_and_non_player_abstention(self):
        records = {str(i): samples(color=[20 if i < 3 else 80, 0, 0]) for i in range(6)}
        records['ref'] = samples('referee', [50, 50, 50])
        decisions, centers = classify_identities(records, CONFIG)
        self.assertEqual(decisions['0']['team_id'], 'A')
        self.assertEqual(decisions['5']['team_id'], 'B')
        self.assertIsNone(decisions['ref']['team_id'])
        self.assertEqual(centers, [[20., 0., 0.], [80., 0., 0.]])
        self.assertIsNone(decide_team(samples(color=[50, 0, 0]), 'player', centers, CONFIG)['team_id'])
        self.assertIsNone(decide_team(samples(color=[50, 100, 100]), 'player', centers, CONFIG)['team_id'])

    def test_team_permutation_is_deterministic(self):
        colors = [[20, 2, 3], [21, 2, 3], [80, 2, 3], [81, 2, 3]]
        np.testing.assert_allclose(fit_teams(colors, CONFIG), fit_teams(colors[::-1], CONFIG))

    def test_sampling_uses_no_labels_and_never_uses_unknown_identity(self):
        frames = [dict(frame_idx=i, tracks=[dict(player_id='x', track_id=1, role='referee',
                    team_id='manual', bbox_xyxy=[10, 10, 30, 70]),
                    dict(player_id=None, bbox_xyxy=[10, 10, 30, 70])]) for i in range(60)]
        before = select_samples(frames, CONFIG, 100, 100)
        for f in frames:
            f['tracks'][0].update(role='player', team_id='something_else')
        self.assertEqual(before, select_samples(frames, CONFIG, 100, 100))
        self.assertEqual(set(before), {'x'})
        self.assertLessEqual(len(before['x']), CONFIG['samples_per_identity'])

    def test_torso_uses_mask_and_missing_masks_stay_unknown(self):
        image = np.full((100, 100, 3), (0, 220, 220), np.uint8)
        row = dict(bbox_xyxy=[10, 10, 70, 90], silhouette_xy=None)
        self.assertIsNone(torso_color(image, row, CONFIG))
        row['silhouette_xy'] = [[10, 10], [70, 10], [70, 90], [10, 90]]
        self.assertEqual(len(torso_color(image, row, CONFIG)), 3)

    def test_enrichment_does_not_modify_measurements_or_leak_manual_tags(self):
        row = dict(player_id=None, track_id=1, display_label='?', bbox_xyxy=[1, 2, 3, 4],
                   role='referee', team_id='B', court_xy_m=[10, 12], jersey_number=23)
        frames = [dict(frame_idx=0, timestamp_s=0., tracks=[row], detections=[dict(role='player', team_id='A')],
                       calibration=dict(method='test'))]
        original = copy.deepcopy(frames)
        enriched = enrich(frames, {})
        self.assertEqual(frames, original)
        result = enriched[0]['tracks'][0]
        for k in ('role', 'team_id', 'jersey_number', 'classification_evidence_id'):
            self.assertIsNone(result[k])
        self.assertEqual(result['court_xy_m'], row['court_xy_m'])
        self.assertIsNone(enriched[0]['detections'][0]['role'])
        record = flat_record('clip', enriched[0], result)
        self.assertEqual((record['court_x_m'], record['court_y_m']), (10, 12))
        self.assertIsNone(record['person_id'])

    def test_gap_is_missing_observation_not_assumed_offscreen(self):
        row = dict(player_id='a', track_id=1, court_xy_m=None)
        frames = [dict(frame_idx=i, tracks=[row]) for i in (0, 1, 4)]
        summary = person_summaries(frames, {}, 25)[0]
        self.assertEqual(summary['gaps'][0]['reason'], 'unobserved; cause_unknown')
        self.assertEqual(summary['gaps'][0]['missing_frames'], 2)
        self.assertEqual(summary['projected_observations'], 0)


if __name__ == '__main__':
    unittest.main()
