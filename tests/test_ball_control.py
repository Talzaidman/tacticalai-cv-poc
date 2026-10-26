"""Measured support, silhouette geometry and bounded continuity regression tests."""
import unittest

from tacticalai.ball_control import ControlConfig,TemporalControlEstimator,contact_geometry


def person(key='p',polygon=None):
    return dict(player_id=key,track_id=1 if key=='p' else 2,identity_segment_id=key,
                role='player',bbox_xyxy=[0,0,100,200],court_xy_m=[5,5],court_region='inside',
                silhouette_xy=polygon if polygon is not None else [[0,0],[100,0],[100,200],[0,200]])


def ball(x=50,state='observed',segment=1):
    return dict(status=state,center_image_xy=[x,100] if state=='observed' else None,
                inferred_center_image_xy=[x,100] if state=='inferred' else None,track_id=segment)


class TemporalControlTests(unittest.TestCase):
    def test_wrists_beat_overlapping_body_box(self):
        first,second=person(),person('q')
        for row,wrist_x in ((first,95),(second,50)):
            points=[[0.,0.,0.] for _ in range(17)]
            points[9]=[wrist_x,100.,.9]
            row['control_pose']=dict(keypoints_xy_conf=points)
        estimator=TemporalControlEstimator(25,ControlConfig(pose_required=True))
        for _ in range(3):result=estimator.update(ball(50),[first,second])
        self.assertEqual(result['controller_person_id'],'q')
        self.assertEqual(result['reason'],'wrist_and_temporal_support')
        self.assertEqual(result['candidates'][0]['geometry_source'],'wrist_keypoints')

    def test_missing_or_low_confidence_wrists_cannot_establish_control(self):
        for confidence in (0.,.1):
            row=person();points=[[50.,100.,confidence] for _ in range(17)]
            row['control_pose']=dict(keypoints_xy_conf=points)
            estimator=TemporalControlEstimator(25,ControlConfig(pose_required=True))
            for _ in range(10):
                self.assertIsNone(estimator.update(ball(),[row])['controller_person_id'])

    def test_dribble_continuity_requires_prior_wrist_measurements(self):
        row=person();points=[[0.,0.,0.] for _ in range(17)];points[9]=[50.,100.,.9]
        row['control_pose']=dict(keypoints_xy_conf=points)
        estimator=TemporalControlEstimator(25,ControlConfig(pose_required=True,support_window_s=.4))
        for _ in range(3):estimator.update(ball(),[row])
        low=ball();low['center_image_xy']=[50,150]
        result=estimator.update(low,[row])
        self.assertEqual(result['status'],'estimated_control_continuity')
        self.assertEqual(result['evidence_source'],'recent_wrist_measurement_memory')
        fresh=TemporalControlEstimator(25,ControlConfig(pose_required=True))
        for _ in range(5):self.assertIsNone(fresh.update(low,[row])['controller_person_id'])

    def test_silhouette_rejects_empty_box_area_as_equal_contact(self):
        decoy=person('p',[[0,0],[10,0],[10,200],[0,200]])
        actual=person('q',[[80,0],[100,0],[100,200],[80,200]])
        estimator=TemporalControlEstimator(25)
        for _ in range(3):result=estimator.update(ball(95),[decoy,actual])
        self.assertEqual(result['controller_person_id'],'q')
        self.assertEqual(result['candidates'][0]['geometry_source'],'visible_silhouette')
        self.assertGreater(contact_geometry([95,100],decoy)['distance_heights'],.4)

    def test_support_accumulates_across_short_measurement_gaps(self):
        estimator=TemporalControlEstimator(25)
        for state in ('observed','inferred','observed','inferred','observed'):
            result=estimator.update(ball(state=state),[person()])
            if state=='inferred':self.assertIsNone(result['controller_person_id'])
        self.assertEqual(result['controller_person_id'],'p')
        self.assertEqual(result['support_measurements'],3)
        self.assertEqual(result['evidence_source'],'current_measurement_window')

    def test_continuity_is_labelled_bounded_and_cannot_change_owner(self):
        estimator=TemporalControlEstimator(25)
        for _ in range(3):estimator.update(ball(),[person()])
        for i in range(1,6):
            result=estimator.update(ball(state='inferred'),[person()])
            if i<=4:
                self.assertEqual(result['status'],'estimated_control_continuity')
                self.assertEqual(result['evidence_source'],'recent_measurement_memory')
            else:self.assertIsNone(result['controller_person_id'])
        fresh=TemporalControlEstimator(25)
        for _ in range(10):
            self.assertIsNone(fresh.update(ball(state='inferred'),[person()])['controller_person_id'])

    def test_persistent_overlap_remains_unknown(self):
        estimator=TemporalControlEstimator(25)
        for _ in range(10):
            result=estimator.update(ball(),[person(),person('q')])
            self.assertIsNone(result['controller_person_id'])
        self.assertEqual(result['reason'],'ambiguous_people')

    def test_fast_relative_pass_does_not_keep_contact_or_carry_it(self):
        estimator=TemporalControlEstimator(25)
        for _ in range(3):result=estimator.update(ball(10),[person()])
        self.assertEqual(result['controller_person_id'],'p')
        result=estimator.update(ball(95),[person()])
        self.assertIsNone(result['controller_person_id'])
        self.assertGreater(result['candidates'][0]['relative_speed_heights_s'],8)
        self.assertIsNone(estimator.update(ball(95,'inferred'),[person()])['controller_person_id'])

    def test_scene_ball_segment_and_player_visibility_reset_continuity(self):
        for kind in ('cut','segment','hidden'):
            estimator=TemporalControlEstimator(25)
            for _ in range(3):estimator.update(ball(),[person()])
            rows=[] if kind=='hidden' else [person()]
            result=estimator.update(ball(state='inferred',segment=2 if kind=='segment' else 1),rows,cut=kind=='cut')
            self.assertIsNone(result['controller_person_id'])

    def test_fallback_and_bad_configuration_are_explicit(self):
        row=person();row['silhouette_xy']=None
        self.assertEqual(contact_geometry([50,100],row)['geometry_source'],'box_fallback')
        for kwargs in (dict(minimum_measurements=1),dict(continuity_s=1),dict(minimum_contact_score=2)):
            with self.assertRaises(ValueError):ControlConfig(**kwargs)
        with self.assertRaises(ValueError):TemporalControlEstimator(0)


if __name__=='__main__':unittest.main()
