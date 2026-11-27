"""CPU fusion regressions: measurement-only control, ambiguity and source integrity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tacticalai.ball_fusion_cli import (ball_result_paths,combine_frames,controller_ball,
                                        eligible_people,load_ball,player_camera_data_preserved,
                                        add_ball_floor_ray_projections,
                                        smooth_ball_floor_ray_projections)
from tacticalai.pass_events import PassConfig, infer_pass_events


def player(key='p', role='player', shown=True):
    return dict(player_id=key,track_id=1 if key=='p' else 2,identity_segment_id=key+'.0',
                role=role,display_on_video=shown,classification_source='automatic_clip_roles_and_torso_color',
                bbox_xyxy=[100,100,140,200],court_xy_m=[4,5],court_region='inside')


def ball(index, state='detected'):
    return dict(frame_idx=index,timestamp_s=index/25,state=state,x=120 if state!='none' else None,
                y=150 if state!='none' else None,conf=.8 if state=='detected' else None,
                measurement_xy=[120,150] if state=='detected' else None,segment_id=1,
                accepted_candidate_index=0,position_source=state)


def inputs(states, rows=None):
    people=[dict(frame_idx=i,timestamp_s=i/25,calibration=dict(method='automatic',shot=1),
                 tracks=copy.deepcopy(rows if rows is not None else [player()])) for i in range(len(states))]
    balls=dict(tracked=[ball(i,state) for i,state in enumerate(states)],
               detections=[[dict(x=120,y=150,conf=.8)] for _ in states])
    return people,balls


class FusionTests(unittest.TestCase):
    def test_same_team_controller_change_creates_endpoint_pass_not_ball_projection(self):
        frames=[]
        owners=['a']*5+[None]*5+['b']*5
        for index,owner in enumerate(owners):
            rows=[]
            if owner is not None:
                rows=[dict(player_id=owner,role='player',team_id='A')]
            point={'a':[2.,4.],'b':[8.,7.]}.get(owner)
            frames.append(dict(frame_idx=index,calibration=dict(shot=1),tracks=rows,
                ball_temporal=dict(state='detected'),
                possession=dict(controller_person_id=owner,minimap_xy_m=point)))
        audit=infer_pass_events(frames,25,PassConfig(endpoint_hold_s=0))
        self.assertEqual(audit['accepted_events'],1)
        event=audit['events'][0]
        self.assertEqual(event['confidence'],'conservative')
        self.assertEqual((event['start_xy_m'],event['end_xy_m']),([2.,4.],[8.,7.]))
        self.assertFalse(event['true_airborne_trajectory_known'])
        self.assertIsNone(frames[4]['pass_event'])
        self.assertEqual(frames[5]['pass_event']['progress'],0.)
        self.assertEqual(frames[10]['pass_event']['progress'],1.)

    def test_opponent_controller_change_is_not_rendered_as_pass(self):
        frames=[]
        owners=['a']*5+[None]*5+['b']*5
        for index,owner in enumerate(owners):
            team='A' if owner=='a' else 'B'
            rows=[] if owner is None else [dict(player_id=owner,role='player',team_id=team)]
            point={'a':[2.,4.],'b':[8.,7.]}.get(owner)
            frames.append(dict(frame_idx=index,calibration=dict(shot=1),tracks=rows,
                ball_temporal=dict(state='detected'),
                possession=dict(controller_person_id=owner,minimap_xy_m=point)))
        audit=infer_pass_events(frames,25)
        self.assertEqual(audit['accepted_events'],0)
        self.assertEqual(audit['decisions'][0]['reason'],'different_team')
        self.assertTrue(all(frame['pass_event'] is None for frame in frames))

    def test_unstable_or_unknown_team_candidates_are_tentative(self):
        for kind in ('unstable','unknown_team'):
            owners=['a']*(2 if kind=='unstable' else 5)+[None]*5+['b']*5
            frames=[]
            for index,owner in enumerate(owners):
                team=None if kind=='unknown_team' and owner=='b' else 'A'
                rows=[] if owner is None else [dict(player_id=owner,role='player',team_id=team)]
                point={'a':[2.,4.],'b':[8.,7.]}.get(owner)
                frames.append(dict(frame_idx=index,calibration=dict(shot=1),tracks=rows,
                    ball_temporal=dict(state='detected'),
                    possession=dict(controller_person_id=owner,minimap_xy_m=point)))
            audit=infer_pass_events(frames,25)
            self.assertEqual(audit['accepted_events'],1)
            self.assertEqual(audit['events'][0]['confidence'],'tentative')
            self.assertIn('unstable_control_endpoint' if kind=='unstable' else 'unknown_team',
                          audit['events'][0]['relaxed_reasons'])

    def test_missing_receiver_endpoint_uses_bounded_same_identity_interpolation(self):
        owners=['a']*5+[None]*5+['b']*5+[None]
        frames=[]
        for index,owner in enumerate(owners):
            rows=[]
            if owner=='a':
                rows=[dict(player_id='a',identity_segment_id='a.0',role='player',team_id='A',
                           minimap_xy_m=[2.,4.])]
            elif index in (9,15):
                rows=[dict(player_id='b',identity_segment_id='b.0',role='player',team_id='A',
                           minimap_xy_m=[8.,6.] if index==9 else [9.,7.])]
            elif owner=='b':
                rows=[dict(player_id='b',identity_segment_id='b.0',role='player',team_id='A',
                           minimap_xy_m=None)]
            possession=dict(controller_person_id=owner,minimap_xy_m=([2.,4.] if owner=='a' else None),
                            controller_identity_segment_id=(owner+'.0' if owner else None))
            frames.append(dict(frame_idx=index,calibration=dict(shot=1),tracks=rows,
                ball_temporal=dict(state='detected'),possession=possession))
        audit=infer_pass_events(frames,25)
        self.assertEqual(audit['accepted_events'],1)
        event=audit['events'][0]
        self.assertEqual(event['confidence'],'tentative')
        self.assertIn('interpolated_endpoint',event['relaxed_reasons'])
        self.assertEqual(event['target_position_quality']['source'],
                         'bounded_same_identity_interpolation')
        self.assertEqual(event['target_position_quality']['anchor_frames'],[9,15])
        self.assertAlmostEqual(event['end_xy_m'][0],8+1/6)
        self.assertTrue(event['target_position_quality']['offline_future_frames_used'])

    def test_only_measured_ball_gets_explicit_floor_ray_projection(self):
        people,balls=inputs(['detected','predicted','interpolated','none'])
        frames=combine_frames(people,balls,25)
        for frame in frames:
            frame['image_to_court_h']=[[.05,0,0],[0,.05,0],[0,0,1]]
        self.assertEqual(add_ball_floor_ray_projections(frames),1)
        self.assertEqual(frames[0]['ball_minimap_floor_ray_xy_m'],[6.,7.5])
        self.assertEqual(frames[0]['ball_minimap_projection_source'],
                         'measured_ball_floor_ray_intersection')
        self.assertIsNone(frames[0]['ball_true_court_xy_m'])
        for frame in frames[1:]:
            self.assertIsNone(frame['ball_minimap_floor_ray_xy_m'])
            self.assertEqual(frame['ball_minimap_projection_quality']['reason'],
                             'ball_not_measured_this_frame')

    def test_ball_floor_ray_smoothing_reduces_impulse_without_filling_gap(self):
        people,balls=inputs(['detected']*7)
        frames=combine_frames(people,balls,25)
        points=[[2,5],[2.2,5],[8,5],[2.6,5],None,[3,5],[3.2,5]]
        for index,(frame,point) in enumerate(zip(frames,points)):
            frame['ball_minimap_floor_ray_xy_m']=point
            frame['ball_temporal']['segment_id']=1
            frame['frame_idx']=index
        smooth_ball_floor_ray_projections(frames,25)
        self.assertIsNone(frames[4]['ball_minimap_floor_ray_xy_m'])
        self.assertEqual(frames[2]['ball_minimap_floor_ray_raw_xy_m'],[8,5])
        self.assertLess(abs(frames[2]['ball_minimap_floor_ray_xy_m'][0]-2.4),1.)
        self.assertGreater(frames[2]['ball_minimap_floor_ray_filter']['measurement_variance_scale'],10)

    def test_control_needs_repeated_measurements_and_preserves_sources(self):
        people,balls=inputs(['detected']*4)
        before=copy.deepcopy((people,balls))
        result=combine_frames(people,balls,25)
        self.assertEqual([r['possession']['status'] for r in result],
                         ['unknown','unknown','estimated_control','estimated_control'])
        self.assertEqual(result[-1]['possession']['controller_person_id'],'p')
        self.assertEqual(result[-1]['possession']['minimap_xy_m'],[4,5])
        self.assertIsNone(result[-1]['ball_court_xy_m'])
        self.assertEqual((people,balls),before)
        self.assertTrue(all(all(frame[k]==v for k,v in original.items())
                            for frame,original in zip(result,people)))

    def test_foundation_ball_placeholders_may_be_replaced_without_weakening_player_integrity(self):
        people,balls=inputs(['detected']*4)
        for frame in people:
            frame.update(
                ball=dict(status='unknown',reason='no_candidates'),
                possession=dict(status='unknown',reason='ball_not_observed'),
                ball_raw_detections=[],ball_camera_cut_reset=True,
                image_to_court_h=[[1,0,0],[0,1,0],[0,0,1]])
        before=copy.deepcopy(people)
        result=combine_frames(people,balls,25)
        self.assertEqual(result[-1]['possession']['status'],'estimated_control')
        self.assertTrue(all(player_camera_data_preserved(source,fused)
                            for source,fused in zip(people,result)))
        self.assertEqual(people,before)
        broken=copy.deepcopy(result[0]);broken['tracks'][0]['bbox_xyxy'][0]+=1
        self.assertFalse(player_camera_data_preserved(people[0],broken))

    def test_inference_never_establishes_or_retains_control(self):
        for state in ('predicted','interpolated','none'):
            people,balls=inputs(['detected']*3+[state]+['detected']*3)
            result=combine_frames(people,balls,25)
            self.assertIsNotNone(result[2]['possession']['controller_person_id'])
            self.assertTrue(all(result[i]['possession']['controller_person_id'] is None for i in (3,4,5)))
            self.assertIsNotNone(result[6]['possession']['controller_person_id'])
            self.assertEqual(controller_ball(ball(0,state))['status'],'unknown')

    def test_ambiguous_overlap_abstains(self):
        people,balls=inputs(['detected']*6,[player(),player('q')])
        result=combine_frames(people,balls,25)
        self.assertTrue(all(r['possession']['controller_person_id'] is None for r in result))
        self.assertEqual(result[-1]['possession']['reason'],'ambiguous_people')

    def test_unknown_referee_hidden_and_manual_rows_are_not_eligible_players(self):
        rows=[player(role=None),player(role='referee'),player(shown=False),
              dict(player(),classification_source='manual_review'),dict(player(),track_id=None)]
        self.assertEqual(eligible_people(rows),[])
        people,balls=inputs(['detected']*5,rows)
        self.assertTrue(all(r['possession']['controller_person_id'] is None
                            for r in combine_frames(people,balls,25)))

    def test_cut_or_new_ball_segment_requires_fresh_confirmation(self):
        for change in ('cut','segment'):
            people,balls=inputs(['detected']*5)
            if change=='cut':
                for frame in people[3:]:frame['calibration']['shot']=2
            else:
                for point in balls['tracked'][3:]:point['segment_id']=2
            result=combine_frames(people,balls,25)
            self.assertIsNotNone(result[2]['possession']['controller_person_id'])
            self.assertTrue(all(result[i]['possession']['controller_person_id'] is None for i in (3,4)))

    def test_ball_loader_rejects_timeline_dimensions_and_candidate_mismatch(self):
        info=dict(width=640,height=360,frames=1,fps=25,sha256='local')
        data=dict(config=dict(tracker='kalman_cv_gated_interpolation'),tracked=[ball(0)],
                  detections=[[dict(x=120,y=150,conf=.8)]])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ball.json';path.write_text(json.dumps(data))
            with patch('tacticalai.ball_fusion_cli.video_info',return_value=info):
                self.assertEqual(load_ball(path,Path('video.mp4'),info),data)
                for mutation in ('timestamp','measurement','confidence'):
                    broken=copy.deepcopy(data)
                    if mutation=='timestamp':broken['tracked'][0]['timestamp_s']=1
                    if mutation=='measurement':broken['tracked'][0]['measurement_xy']=[500,150]
                    if mutation=='confidence':broken['tracked'][0]['conf']=.2
                    path.write_text(json.dumps(broken))
                    with self.assertRaises(ValueError):load_ball(path,Path('video.mp4'),info)
            path.write_text(json.dumps(data))
            with patch('tacticalai.ball_fusion_cli.video_info',return_value=dict(info,width=1280)):
                with self.assertRaises(ValueError):load_ball(path,Path('video.mp4'),info)

    def test_sam3_native_bundle_and_configurable_prefix_are_supported(self):
        info=dict(width=640,height=360,frames=1,fps=25,sha256='local')
        data=dict(config=dict(tracker='sam3_native_video_track'),tracked=[ball(0)],
                  detections=[[dict(x=120,y=150,conf=.8)]])
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            path,video=ball_result_paths(folder,'SAM3-Native','clip')
            self.assertEqual(path.name,'SAM3-Native_clip.json')
            path.write_text(json.dumps(data))
            with patch('tacticalai.ball_fusion_cli.video_info',return_value=info):
                self.assertEqual(load_ball(path,video,info),data)


if __name__=='__main__':unittest.main()
