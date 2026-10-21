"""Evaluation-only visible-head point coverage; never imported by inference."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_visible_points(points, rows, image_size, reference_size=(1280,720)):
    """One-to-one containment, then distance to upper box centre.

    This measures coverage of inspected head points, not box AP, identity
    accuracy, or precision. One large merged box cannot cover two people.
    """
    if not points or not rows:
        return []
    boxes=np.asarray([r['bbox_xyxy'] for r in rows],float)
    boxes*=np.array([reference_size[0]/image_size[0],reference_size[1]/image_size[1]]*2)
    xy=np.asarray([p['point_xy_1280x720'] for p in points],float)
    inside=((xy[:,None,:]>=boxes[None,:,:2]-3)&(xy[:,None,:]<=boxes[None,:,2:]+3)).all(axis=2)
    head=np.column_stack(((boxes[:,0]+boxes[:,2])/2,boxes[:,1]+.12*(boxes[:,3]-boxes[:,1])))
    dimensions=np.maximum(boxes[:,2:]-boxes[:,:2],1)
    distance=np.linalg.norm((xy[:,None,:]-head[None,:,:])/dimensions[None,:,:],axis=2)
    score=inside*(min(inside.shape)+2)-np.minimum(distance,1)
    rr,cc=linear_sum_assignment(-score)
    return [(int(r),int(c)) for r,c in zip(rr,cc) if inside[r,c]]
