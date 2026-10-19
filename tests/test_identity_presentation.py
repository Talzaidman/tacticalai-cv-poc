"""Projection must preserve identities and never smooth across person changes."""
import copy
import unittest

import numpy as np

from tacticalai.identity_presentation import fuse_frames, person_label, person_color
from tacticalai.temporal_projection import smooth_players


class IdentityPresentationTests(unittest.TestCase):
    def test_smoothing_resets_at_a_small_same_raw_id_person_change(self):
        frames=[dict(tracks=[dict(track_id=1,role='player',identity_segment_id='1.0' if i<6 else '1.1',
                                 court_xy_m=[3. if i<6 else 3.8,5.])]) for i in range(12)]
        smooth_players(frames,25,np.array([True]+[False]*11),identity_field='identity_segment_id')
        self.assertTrue(all(np.allclose(f['tracks'][0]['court_xy_m'],[3. if i<6 else 3.8,5.]) for i,f in enumerate(frames)))
        self.assertEqual(frames[0]['tracks'][0]['position_temporal_filter']['segment_observations'],6)
        self.assertEqual(frames[-1]['tracks'][0]['position_temporal_filter']['segment_observations'],6)

    def test_unknown_identity_does_not_enter_a_smoothing_segment(self):
        frames=[dict(tracks=[dict(track_id=1,role='player',identity_segment_id=None,court_xy_m=[3.,5.])])]
        before=copy.deepcopy(frames)
        smooth_players(frames,25,np.array([True]),identity_field='identity_segment_id')
        self.assertEqual(frames,before)

    def test_fusion_keeps_staff_boxes_and_unknown_identity_without_inventing_positions(self):
        row=dict(track_id=96,bbox_xyxy=[10,10,20,40],det_confidence=.9,player_id=None,
                 identity_segment_id=None,identity_status='unknown_transition')
        frames=[dict(frame_idx=0,timestamp_s=0.,tracks=[row])]
        cameras=[dict(frame_idx=0,timestamp_s=0.,refined_image_to_court_h=None,
                      temporal_quality=dict(method='unavailable',shot=1))]
        roles=dict(intervals=[dict(track_id=96,start_frame=0,end_frame=0,role='staff')])
        result=fuse_frames(frames,cameras,roles)
        actual=result[0]['tracks'][0]
        self.assertEqual(actual['track_id'],96)
        self.assertEqual(actual['bbox_xyxy'],row['bbox_xyxy'])
        self.assertEqual(actual['role'],'staff')
        self.assertIsNone(actual['player_id'])
        self.assertIsNone(actual['court_xy_m'])
        self.assertFalse(actual['shown_on_minimap'])
        self.assertEqual(actual['display_label'],'?')

    def test_camera_time_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_frames([dict(frame_idx=0,timestamp_s=0.,tracks=[])],
                        [dict(frame_idx=1,timestamp_s=.04)],{})

    def test_person_display_is_stable_across_raw_labels(self):
        self.assertEqual(person_label('87.0'),'P87')
        self.assertEqual(person_label('85.1'),'P85.1')
        self.assertEqual(person_color('87.0'),person_color('87.0'))
        self.assertEqual(person_label(None),'?')


if __name__=='__main__':
    unittest.main()
