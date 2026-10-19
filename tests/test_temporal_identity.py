"""Temporal identity hypotheses must abstain without corrupting source tracks."""
import copy
import unittest

import numpy as np

from tacticalai.gallery_linking import associate_galleries
from tacticalai.temporal_identity import detect_change, split_tracks
from tacticalai.temporal_reid_cli import analyze


def observations(frames):
    return [dict(frame=f, track_id=1, overlap=0., edge=False, quality=1., bbox_xyxy=[0,0,20,60]) for f in frames]


class TemporalIdentityTests(unittest.TestCase):
    def test_missing_frames_do_not_split_a_stable_person(self):
        obs = observations(list(range(0,21,3)) + list(range(70,91,3)))
        event, _ = detect_change(obs, np.tile([1.,0,0], (len(obs),1)))
        self.assertIsNone(event)

    def test_persistent_change_is_localized_without_a_review_label(self):
        obs = observations(range(0,45,3))
        x = np.array([[1.,0,0]]*7 + [[0.,1,0]]*8)
        event, _ = detect_change(obs, x)
        self.assertEqual((event['last_old_frame'],event['first_new_frame']), (18,21))

    def test_short_occluder_does_not_become_a_new_identity(self):
        obs = observations(range(0,63,3))
        x = np.array([[1.,0,0]]*9 + [[0.,1,0]]*3 + [[1.,0,0]]*9)
        event, _ = detect_change(obs, x)
        self.assertIsNone(event)

    def test_uncertain_transition_is_null_and_never_fills_missing_detections(self):
        obs = observations(range(0,48,3))
        x = np.array([[1.,0,0]]*7 + [[0.,0,1]] + [[0.,1,0]]*8)
        frames = [dict(frame_idx=o['frame'],tracks=[dict(track_id=1,bbox_xyxy=[0,0,20,60])]) for o in obs]
        for f in frames:
            if f['frame_idx'] >= 24:
                f['tracks'][0]['bbox_xyxy'] = [30,0,50,60]
        segments, events, _, lookup = split_tracks(frames, obs, x)
        self.assertEqual(len(segments),2)
        self.assertNotIn((21,1),lookup)
        self.assertNotIn((22,1),lookup)
        self.assertEqual((events[0]['last_old_frame'],events[0]['first_new_frame']),(18,24))
        before = copy.deepcopy(frames)
        result = analyze(frames,obs,x)
        self.assertEqual(len(result['frames']),len(before))
        for a,b in zip(before,result['frames']):
            self.assertEqual(a['tracks'][0]['track_id'],b['tracks'][0]['track_id'])
            self.assertEqual(a['tracks'][0]['bbox_xyxy'],b['tracks'][0]['bbox_xyxy'])
        self.assertIsNone(result['frames'][7]['tracks'][0]['player_id'])

    def test_split_phases_cannot_be_merged_directly_or_via_a_third_segment(self):
        tracks = {'a':dict(start=0,end=2),'b':dict(start=4,end=6),'c':dict(start=8,end=10)}
        galleries = {key:np.array([[1.,0]]*3) for key in tracks}
        mapping, candidates = associate_galleries(tracks,galleries,margin=0,cannot_link=[('a','c')])
        self.assertNotEqual(mapping['a'],mapping['c'])
        self.assertFalse(next(c for c in candidates if c['before']=='a' and c['after']=='c')['accepted'])

    def test_invalid_feature_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_change(observations([0]),np.array([[2.,0]]))

    def test_persistent_pose_change_with_smooth_tracking_does_not_split(self):
        obs = observations(range(0,45,3))
        x = np.array([[1.,0,0]]*7 + [[0.,1,0]]*8)
        frames = [dict(frame_idx=o['frame'],tracks=[dict(track_id=1,bbox_xyxy=[0,0,20,60])]) for o in obs]
        segments, events, _, _ = split_tracks(frames,obs,x)
        self.assertEqual(len(segments),1)
        self.assertEqual(events,[])


if __name__ == '__main__':
    unittest.main()
