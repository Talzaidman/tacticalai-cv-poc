"""CPU fusion regressions: measurement-only control, ambiguity and source integrity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tacticalai.ball_fusion_cli import combine_frames, controller_ball, eligible_people, load_ball


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


if __name__=='__main__':unittest.main()
