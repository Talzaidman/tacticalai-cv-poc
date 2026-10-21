import unittest

from tacticalai.court_gate import court_gate
from tacticalai.coverage_evaluation import match_visible_points
from tacticalai.coverage_presentation import observations


class CoverageTests(unittest.TestCase):
    def test_missing_camera_and_clipped_person_are_not_confidently_excluded(self):
        self.assertTrue(court_gate([10,10,20,30],None,{},(100,100))['keep'])
        result=court_gate([95,10,100,20],[[1,0,0],[0,1,0],[0,0,1]],{},(100,100))
        self.assertTrue(result['keep'])
        self.assertEqual(result['court_gate'],'uncertain')

    def test_court_gate_rejects_clear_outside_but_keeps_boundary_person(self):
        matrix=[[1,0,0],[0,1,0],[0,0,1]]
        self.assertFalse(court_gate([40,40,50,60],matrix,{},(100,100))['keep'])
        self.assertTrue(court_gate([27,3,32,10],matrix,{},(100,100))['keep'])
        self.assertTrue(court_gate([2,2,6,8],matrix,{},(100,100))['keep'])

    def test_merged_detection_cannot_count_as_two_visible_people(self):
        points=[dict(point_xy_1280x720=[20,20]),dict(point_xy_1280x720=[30,20])]
        rows=[dict(bbox_xyxy=[10,10,40,70])]
        self.assertEqual(len(match_visible_points(points,rows,(1280,720))),1)

    def test_detection_without_confirmed_track_is_retained_and_not_given_an_id(self):
        source=dict(frame_idx=0,timestamp_s=0.,image_to_court_h=None,
            calibration=dict(method='unavailable'),rejected_detections=[],
            detections=[dict(track_id=None,bbox_xyxy=[10,10,20,40],class_name='person',
                             det_confidence=.2,role='referee')],tracks=[])
        mask=dict(boxes=[],polygons=[],confidences=[])
        frame=observations(source,mask,dict(width=100,height=100),'after')
        self.assertEqual(len(frame['tracks']),1)
        row=frame['tracks'][0]
        self.assertIsNone(row['track_id'])
        self.assertIsNone(row['player_id'])
        self.assertIsNone(row['role'])
        self.assertIsNone(row['silhouette_xy'])
        self.assertIsNone(row['court_xy_m'])
        self.assertEqual(row['display_label'],'?')


if __name__=='__main__':unittest.main()
