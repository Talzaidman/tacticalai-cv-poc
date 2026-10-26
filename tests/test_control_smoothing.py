"""Offline controller continuity must not hide handoffs, cuts or evidence loss."""
import copy
import unittest
from tacticalai.control_smoothing import smooth_control_gaps,ControlSmoothingConfig


def row():
    return dict(player_id='p',identity_segment_id='p.0',track_id=1,role='player',display_on_video=True,
                bbox_xyxy=[0,0,100,200],silhouette_xy=[[0,0],[100,0],[100,200],[0,200]],
                display_label='P1',court_xy_m=[5,5],court_region='inside')


def frame(i,owner=None):
    return dict(frame_idx=i,timestamp_s=i/25,tracks=[row()],calibration=dict(shot=1),
                ball_temporal=dict(state='detected',segment_id=1,x=50.,y=150.,measurement_xy=[50.,150.]),
                possession=dict(status='estimated_control' if owner else 'unknown',
                                controller_person_id=owner,controller_identity_segment_id='p.0' if owner else None,
                                controller_track_id=1 if owner else None,controller_role='player' if owner else None,
                                candidates=[],minimap_xy_m=None,minimap_position_source=None,score=.8,
                                reason='wrist_and_temporal_support',support_measurements=3))


class ControlSmoothingTests(unittest.TestCase):
    def sequence(self,gap=8):
        return [frame(i,'p' if i in (0,gap+1) else None) for i in range(gap+2)]

    def test_dribble_gap_between_same_measured_owner_is_filled_and_audited(self):
        frames=self.sequence();original=copy.deepcopy(frames)
        output,report=smooth_control_gaps(frames,25)
        self.assertEqual(report['filled_frames'],8)
        for f in output[1:-1]:
            self.assertEqual(f['possession']['controller_person_id'],'p')
            self.assertEqual(f['possession']['status'],'estimated_control_continuity')
            self.assertEqual(f['possession']['evidence_source'],'offline_same_owner_gap_bridge')
            self.assertTrue(f['possession']['offline_future_frames_used'])
            self.assertIsNone(f['possession']['score'])
            self.assertIsNone(f['possession_raw']['controller_person_id'])
        self.assertEqual(frames,original)
        self.assertTrue(all(a['tracks']==b['tracks'] and a['ball_temporal']==b['ball_temporal']
                            for a,b in zip(frames,output)))

    def test_different_owner_at_end_is_a_handoff_not_a_bridge(self):
        frames=self.sequence();frames[-1]['possession']['controller_person_id']='q'
        output,report=smooth_control_gaps(frames,25)
        self.assertEqual(report['filled_frames'],0)
        self.assertTrue(all(f['possession']['controller_person_id'] is None for f in output[1:-1]))

    def test_rival_wrist_evidence_interrupts_continuity(self):
        for score,support in ((.8,0),(.4,2)):
            frames=self.sequence()
            frames[4]['possession']['candidates']=[dict(person_id='q',track_id=2,contact_score=score,support_measurements=support)]
            self.assertEqual(smooth_control_gaps(frames,25)[1]['filled_frames'],0)

    def test_ball_departure_cut_and_new_segment_prevent_bridge(self):
        for kind in ('departure','cut','segment','unknown'):
            frames=self.sequence()
            if kind=='departure':frames[4]['ball_temporal'].update(x=1000,measurement_xy=[1000,150])
            if kind=='cut':frames[4]['calibration']['shot']=2
            if kind=='segment':frames[4]['ball_temporal']['segment_id']=2
            if kind=='unknown':frames[4]['ball_temporal'].update(state='none',segment_id=None)
            self.assertEqual(smooth_control_gaps(frames,25)[1]['filled_frames'],0)

    def test_short_missing_player_box_is_separate_inference_not_new_detection(self):
        frames=self.sequence();frames[3]['tracks']=[];frames[4]['tracks']=[]
        output,report=smooth_control_gaps(frames,25)
        self.assertEqual(report['filled_frames'],8)
        for i in (3,4):
            self.assertEqual(output[i]['tracks'],[])
            self.assertFalse(output[i]['possession']['current_player_box_observed'])
            self.assertIsNotNone(output[i]['possession']['inferred_controller_bbox_xyxy'])
            self.assertIsNone(output[i]['possession']['minimap_xy_m'])

    def test_long_gaps_missing_player_and_unbracketed_control_stay_unknown(self):
        self.assertEqual(smooth_control_gaps(self.sequence(16),25)[1]['filled_frames'],0)
        frames=self.sequence()
        for f in frames[2:6]:f['tracks']=[]
        self.assertEqual(smooth_control_gaps(frames,25)[1]['filled_frames'],0)
        frames=self.sequence();frames[-1]['possession']['controller_person_id']=None
        self.assertEqual(smooth_control_gaps(frames,25)[1]['filled_frames'],0)

    def test_cannot_chain_inferred_endpoints_into_new_control(self):
        frames=self.sequence()
        for i in (0,len(frames)-1):frames[i]['possession']['status']='estimated_control_continuity'
        self.assertEqual(smooth_control_gaps(frames,25)[1]['filled_frames'],0)

    def test_invalid_settings(self):
        for kwargs in (dict(max_gap_s=0),dict(max_missing_player_s=-1),dict(strong_rival_contact=2)):
            with self.assertRaises(ValueError):ControlSmoothingConfig(**kwargs)
        with self.assertRaises(ValueError):smooth_control_gaps([],0)


if __name__=='__main__':unittest.main()
