"""Conservative automatic court filtering; uncertain boundary boxes survive."""
import cv2
import numpy as np

from .calibration import transform


def court_gate(box, matrix, quality, image_size, margin_m=3., minimum_overlap=.1):
    """Reject only an outside floor point AND little overlap with the court.

    No role/jersey labels are used. A missing/invalid camera or clipped box is
    uncertain, not proof a person is outside. Bbox bottoms are approximate.
    """
    x1,y1,x2,y2 = map(float, box)
    unknown = dict(keep=True, court_gate='uncertain', court_gate_reason='no_valid_camera',
                   gate_floor_xy_m=None, court_box_overlap=None)
    if matrix is None:
        return unknown
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        return unknown
    foot = transform([[(x1+x2)/2,y2]], matrix)[0]
    if not np.isfinite(foot).all():
        return unknown
    extra = .5 if 'gap_estimate' in quality.get('method','') else 0.
    margin = margin_m + extra
    inside = 0 <= foot[0] <= 28 and 0 <= foot[1] <= 15
    near = -margin <= foot[0] <= 28+margin and -margin <= foot[1] <= 15+margin
    result = dict(keep=True, court_gate='inside' if inside else 'boundary',
                  court_gate_reason='floor_point_inside' if inside else 'generous_floor_margin',
                  gate_floor_xy_m=foot.tolist(), court_box_overlap=None)
    if near:
        return result
    width,height = image_size
    if x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1:
        return dict(result,court_gate='uncertain',court_gate_reason='clipped_box')
    polygon = transform([[0,0],[28,0],[28,15],[0,15]], np.linalg.inv(matrix))
    if not np.isfinite(polygon).all():
        return unknown
    polygon = cv2.convexHull(polygon.astype(np.float32))
    rect = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)
    overlap = cv2.intersectConvexConvex(polygon,rect)[0] / max((x2-x1)*(y2-y1),1)
    if overlap >= minimum_overlap:
        return dict(result,court_gate_reason='body_intersects_court',court_box_overlap=float(overlap))
    return dict(result, keep=False, court_gate='outside', court_gate_reason='outside_floor_and_body',
                court_box_overlap=float(overlap))
