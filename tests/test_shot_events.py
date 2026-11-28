import unittest

import numpy as np

from tacticalai.ball_fusion_cli import COURT, SHOT_COLOR, flat_frame, render_combined
from tacticalai.pass_events import infer_pass_events
from tacticalai.shot_events import ShotConfig, infer_shot_events


INFO = dict(fps=25, width=960, height=540)


def make_flight(side='left', kind='hoop'):
    # Synthetic camera places each hoop/backboard at a different, predictable pixel.
    offset, height = (157.5, 305.) if kind == 'hoop' else (120., 345.)
    x_m = offset if side == 'left' else 2800-offset
    x = 100+400*x_m/(2000+height)
    y = 80+400*750/(2000+height)
    hardware = dict(kind=kind, track_id=1, confidence=.8, bbox_xyxy=[x-7, y-4, x+7, y+4])
    camera = dict(intrinsic=[[400, 0, 100], [0, 400, 80], [0, 0, 1]],
                  rotation=np.eye(3).tolist(), translation_cm=[0, 0, 2000])
    matrix = np.linalg.inv([[20, 0, 100], [0, 20, 80], [0, 0, 1]]).tolist()
    frames = []
    for index in range(12):
        bx = x + (-50+3*index if index < 8 else 0.)
        owner = 'p1' if index <= 2 else 'p2' if index >= 9 else None
        point = {'p1': [7., 5.], 'p2': [11., 5.]}.get(owner)
        control = dict(controller_person_id=owner, controller_track_id=1 if owner else None,
                       status='estimated_control' if owner else 'unknown', score=.7 if owner else None,
                       reason='test', minimap_xy_m=point, candidates=[])
        frames.append(dict(frame_idx=index, timestamp_s=index/INFO['fps'], tracks=[],
                           calibration=dict(shot=1, method='automatic',
                                            evidence={'camera': camera} if index == 5 else {}),
                           image_to_court_h=matrix, ball_court_xy_m=None,
                           possession=control, basket_hardware=[dict(hardware)],
                           ball_temporal=dict(state='detected', measurement_xy=[bx, y], x=bx, y=y, conf=.9,
                                              segment_id=1, accepted_candidate_index=0, position_source='test'),
                           ball_raw_candidates=[{'r': 3.}]))
    return frames


class ShotEventsTests(unittest.TestCase):
    def test_measured_approach_draws_offline_shot_and_blocks_rebound_as_pass(self):
        frames = make_flight()
        audit = infer_shot_events(frames, INFO, COURT)
        self.assertEqual(audit['contact_candidates'], 1)
        self.assertEqual(audit['attributed_shots'], 1)
        event = audit['events'][0]
        self.assertEqual((event['release_frame'], event['contact_frame']), (3, 8))
        self.assertEqual(event['shooter_player_id'], 'p1')
        self.assertEqual(event['end_xy_m'], [1.575, 7.5])
        self.assertEqual(event['camera_pose_frame'], 5)
        self.assertIsNone(event['made'])
        self.assertTrue(all(frame['ball_court_xy_m'] is None for frame in frames))
        self.assertEqual(frames[3]['shot_event']['progress'], 0.)
        self.assertEqual(frames[8]['shot_event']['progress'], 1.)
        self.assertEqual(flat_frame(frames[8])['shot_shooter_player_id'], 'p1')
        rendered = render_combined(np.zeros((540, 960, 3), np.uint8), frames[8],
                                   player_presentation='silhouettes')
        self.assertTrue(np.any(np.all(rendered[10:200, 10:314] == SHOT_COLOR, axis=2)))
        for frame in frames:
            owner = frame['possession']['controller_person_id']
            if owner:
                frame['tracks'] = [dict(player_id=owner, role='player', team_id='A')]
        self.assertEqual(infer_pass_events(frames, INFO['fps'])['decisions'][0]['reason'],
                         'possible_basket_contact_in_pass_interval')

    def test_stale_holder_and_continuity_fill_leave_shooter_unknown(self):
        frames = make_flight('right', 'backboard')
        frames[7]['possession_raw'] = dict(frames[7]['possession'])
        frames[7]['possession'].update(status='estimated_control_continuity', controller_person_id='p1')
        event, = infer_shot_events(frames, INFO, COURT, ShotConfig(maximum_holder_age_s=.20))['events']
        self.assertEqual(event['attribution_reason'], 'stale_holder')
        self.assertEqual(event['contact_kind'], 'backboard')
        self.assertEqual(event['basket_end'], 'right')
        self.assertEqual(event['last_measured_holder_frame'], 2)
        self.assertIsNone(event['shooter_player_id'])
        self.assertIsNone(frames[7]['shot_event'])

    def test_competing_wrist_and_ball_segment_change_block_attribution(self):
        frames = make_flight()
        frames[4]['possession']['candidates'] = [dict(person_id='other',
            geometry_source='wrist_keypoints', contact_score=.7)]
        event, = infer_shot_events(frames, INFO, COURT)['events']
        self.assertEqual(event['attribution_reason'], 'competing_wrist_contact')
        self.assertIsNone(event['shooter_player_id'])
        for frame in frames[5:]:
            frame['ball_temporal']['segment_id'] = 2
        event, = infer_shot_events(frames, INFO, COURT)['events']
        self.assertEqual(event['attribution_reason'], 'no_measured_holder_in_ball_segment')

    def test_predicted_overlap_and_approach_across_camera_cut_do_not_count(self):
        for mode in ('predicted', 'cut'):
            with self.subTest(mode=mode):
                frames = make_flight()
                for frame in frames[8:]:
                    if mode == 'predicted':
                        frame['ball_temporal'].update(state='predicted', measurement_xy=None)
                    else:
                        frame['calibration']['shot'] = 2
                self.assertEqual(infer_shot_events(frames, INFO, COURT)['events'], [])

    def test_unavailable_calibration_preserves_contact_and_shooter(self):
        frames = make_flight()
        frames[8]['image_to_court_h'] = None
        event, = infer_shot_events(frames, INFO, COURT)['events']
        self.assertEqual(event['shooter_player_id'], 'p1')
        self.assertIsNone(event['end_xy_m'])
        self.assertEqual(event['basket_end_reason'], 'missing_camera_at_contact')
        self.assertIsNone(frames[7]['shot_event'])

    def test_new_holder_near_basket_does_not_reuse_precontact_approach(self):
        frames = make_flight()
        for frame in frames[10:]:
            frame['possession'].update(status='unknown', controller_person_id=None)
        audit = infer_shot_events(frames, INFO, COURT)
        self.assertEqual([event['contact_frame'] for event in audit['events']], [8])
