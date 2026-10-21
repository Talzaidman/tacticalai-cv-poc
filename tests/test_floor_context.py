import copy
import unittest
from unittest.mock import patch

import numpy as np

from tacticalai.floor_context import DEFAULTS, apply_display_filter, context_evidence, floor_model
from tacticalai.silhouette_presentation import TEAM_COLORS, observation_color, render


class FloorDisplayTests(unittest.TestCase):
    def test_hidden_people_do_not_change_measurements_and_unknown_context_is_kept(self):
        rows = [dict(bbox_xyxy=[100,100,140,200], role=None, player_id=None, court_xy_m=None),
                dict(bbox_xyxy=[200,100,240,200], role='player', player_id='a', court_xy_m=[3,4])]
        before = copy.deepcopy(rows)
        frame = dict(image_to_court_h=None, tracks=rows)
        model = dict(floor_pixels=np.zeros((360,640),bool), occupied=np.zeros((360,640),bool),
                     scale=(1.,1.), centers_lab=[[50,0,0]])
        with patch('tacticalai.floor_context.floor_model', return_value=model):
            apply_display_filter(np.zeros((360,640,3),np.uint8), frame, DEFAULTS)
        self.assertFalse(rows[0]['display_on_video'])
        self.assertTrue(rows[1]['display_on_video'])
        for a,b in zip(before, rows):
            self.assertTrue(all(b[k] == v for k,v in a.items()))
        with patch('tacticalai.floor_context.floor_model', return_value=None):
            apply_display_filter(np.zeros((360,640,3),np.uint8), frame, DEFAULTS)
        self.assertTrue(all(r['display_on_video'] for r in rows))

    def test_occluded_context_is_unknown_not_outside(self):
        model = dict(floor_pixels=np.zeros((360,640),bool), occupied=np.ones((360,640),bool), scale=(1.,1.))
        evidence = context_evidence([100,100,140,200], model, DEFAULTS)
        self.assertIsNone(evidence['floor_fraction'])
        self.assertEqual(evidence['reason'], 'occluded_or_insufficient_context')

    def test_floor_learning_excludes_person_pixels_and_requires_camera(self):
        image = np.full((360,640,3), (180,150,160), np.uint8)
        image[100:250,200:250] = 0
        rows = [dict(bbox_xyxy=[200,100,250,250], silhouette_xy=[[200,100],[250,100],[250,250],[200,250]])]
        self.assertIsNone(floor_model(image, None, rows, DEFAULTS))
        model = floor_model(image, [[28/640,0,0],[0,15/360,0],[0,0,1]], rows, DEFAULTS)
        self.assertIsNotNone(model)
        self.assertFalse(model['floor_pixels'][150,220])
        self.assertTrue(model['floor_pixels'][150,300])
        self.assertTrue(model['occupied'][150,220])

    def test_team_color_is_shared_by_silhouette_and_minimap(self):
        rows = []
        for i, (team, x, court_x) in enumerate([('A',500,6),('A',700,8),('B',900,10)]):
            rows.append(dict(player_id=str(i), role='player', team_id=team,
                silhouette_xy=[[x,350],[x+50,350],[x+50,450],[x,450]], bbox_xyxy=[x,350,x+50,450],
                court_xy_m=[court_x,6], court_region='inside', display_label=str(i), track_id=i))
        out = render(np.zeros((720,1280,3),np.uint8), dict(tracks=rows, calibration=dict(method='smoothed')), [], color_mode='team')
        for row in rows:
            expected = np.asarray(TEAM_COLORS[row['team_id']])
            x = row['bbox_xyxy'][0]+25
            np.testing.assert_array_equal(out[400,x], (.4*expected).astype(np.uint8))
            mx,my = np.rint(np.asarray(row['court_xy_m'])*9.6+[17,29]).astype(int)+[10,10]
            np.testing.assert_array_equal(out[my,mx], expected)
        self.assertEqual(observation_color(dict(role=None,team_id=None,player_id='7'),'team'), (170,170,170))
        self.assertNotEqual(observation_color(dict(role='referee',team_id=None,player_id='7'),'team'), TEAM_COLORS['A'])


if __name__ == '__main__':
    unittest.main()
