"""Manual role labels and fused positions must not affect automatic output."""
import copy
import unittest

from tacticalai.automatic_presentation import automatic_frames


class AutomaticPresentationTests(unittest.TestCase):
    def test_manual_cache_labels_and_positions_cannot_influence_output(self):
        row = dict(track_id=1, bbox_xyxy=[10,10,20,40], player_id='1.0',
                   identity_segment_id='1.0', identity_status='automatic_hypothesis', role='unknown')
        identities = [dict(frame_idx=0,timestamp_s=0.,tracks=[row])]
        cameras = [dict(frame_idx=0,timestamp_s=0.,refined_image_to_court_h=[[.1,0,0],[0,.1,0],[0,0,1]],
                        temporal_quality=dict(method='kalman_rts_camera',shot=0))]
        cached_row = dict(row,role='referee',court_xy_m=[999,999],shown_on_minimap=False,
                          silhouette_xy=[[10,10],[20,10],[20,40]],silhouette_confidence=.9,silhouette_box_iou=.8)
        masks = [dict(frame_idx=0,timestamp_s=0.,tracks=[cached_row])]
        info = dict(width=100,height=100,fps=25,frames=1)
        original = copy.deepcopy(identities)
        a = automatic_frames(identities,cameras,masks,info)
        cached_row.update(role='player',court_xy_m=None,shown_on_minimap=True)
        b = automatic_frames(identities,cameras,masks,info)
        self.assertEqual(a,b)
        self.assertEqual(identities,original)
        result = a[0]['tracks'][0]
        self.assertIsNone(result['role'])
        self.assertEqual(result['court_xy_m'],[1.5,4.])
        self.assertTrue(result['shown_on_minimap'])
        self.assertEqual(result['track_id'],1)
        self.assertEqual(result['player_id'],'1.0')

    def test_missing_camera_does_not_inherit_manual_fused_position(self):
        row = dict(track_id=1,bbox_xyxy=[10,10,20,40],player_id=None,identity_segment_id=None,role='unknown')
        identities = [dict(frame_idx=0,timestamp_s=0.,tracks=[row])]
        cameras = [dict(frame_idx=0,timestamp_s=0.,refined_image_to_court_h=None,
                        temporal_quality=dict(method='unavailable',shot=0))]
        masks = [dict(frame_idx=0,timestamp_s=0.,tracks=[dict(row,court_xy_m=[1,2],role='player')])]
        result = automatic_frames(identities,cameras,masks,dict(width=100,height=100,fps=25,frames=1))
        self.assertIsNone(result[0]['tracks'][0]['court_xy_m'])
        self.assertFalse(result[0]['tracks'][0]['shown_on_minimap'])


if __name__ == '__main__':
    unittest.main()
