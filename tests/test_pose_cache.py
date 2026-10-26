"""One-to-one pose association must not copy one skeleton to two player identities."""
import unittest
import numpy as np
from tacticalai.pose_cache_cli import match_poses


class PoseMatchingTests(unittest.TestCase):
    def test_one_pose_cannot_be_assigned_to_two_people(self):
        rows=[dict(player_id='a',track_id=1,bbox_xyxy=[0,0,100,200]),
              dict(player_id='b',track_id=2,bbox_xyxy=[10,0,110,200])]
        matches=match_poses(rows,[[0,0,100,200]],np.zeros((1,17,3)))
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['player_id'],'a')

    def test_pose_outside_people_is_not_force_matched(self):
        rows=[dict(player_id='a',track_id=1,bbox_xyxy=[0,0,100,200])]
        self.assertEqual(match_poses(rows,[[500,500,600,700]],np.zeros((1,17,3))),[])
        self.assertEqual(match_poses([],[],[]),[])


if __name__=='__main__':unittest.main()
