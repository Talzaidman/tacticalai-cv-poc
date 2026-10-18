"""Local court-template alignment to thin yellow markings, experimental.

Colour settings are specific to this venue's yellow markings. The automatic
camera is the only initialization; no manual points enter the optimizer.
"""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from .calibration import transform
from .court import floor_lines


def correction_matrix(delta, width=960, height=540):
    source = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    target = source+np.asarray(delta).reshape(4, 2)
    rows, values = [], []
    for (x, y), (u, v) in zip(source, target):
        rows.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        values.extend([u, v])
    return np.append(np.linalg.solve(rows, values), 1).reshape(3, 3)


def refine(frame, image_to_court, court, boxes):
    if image_to_court is None:
        return None, dict(reason="no_camera_initialization"), None
    image = cv2.resize(frame, (960, 540))
    source_scale = np.diag([960/frame.shape[1], 540/frame.shape[0], 1.])
    seed = source_scale @ np.linalg.inv(image_to_court)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (18, 25, 100), (45, 255, 255))
    # Remove broad filled logos/paint; retain thin marking fragments.
    opened = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    thin = cv2.subtract(yellow, opened)
    allowed = np.full((540, 960), 255, np.uint8)
    allowed[459:] = 0
    for box in boxes:
        x1, y1, x2, y2 = np.array(box)*[960/frame.shape[1], 540/frame.shape[0], 960/frame.shape[1], 540/frame.shape[0]]
        cv2.rectangle(allowed, (int(x1)-7, int(y1)-7), (int(x2)+7, int(y2)+7), 0, -1)
    polygon = transform([[0, 0], [28, 0], [28, 15], [0, 15]], seed)
    floor_mask = np.zeros((540, 960), np.uint8)
    if not np.isfinite(polygon).all() or np.abs(polygon).max() > 1e6:
        return None, dict(reason="invalid_initial_court_polygon"), None
    cv2.fillConvexPoly(floor_mask, np.rint(polygon).astype(np.int32), 255)
    floor_mask = cv2.dilate(floor_mask, np.ones((25, 25), np.uint8))
    allowed &= floor_mask
    thin &= allowed
    evidence = int(np.count_nonzero(thin))
    if evidence < 100:
        return None, dict(reason="insufficient_line_pixels", line_pixels=evidence), thin
    distance = cv2.distanceTransform(255-thin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(float)
    samples = []
    for line in floor_lines(court):
        for a, b in zip(line, line[1:]):
            count = max(2, int(np.linalg.norm(b-a)/.08))
            samples.extend(np.linspace(a, b, count))
    pixels = transform(samples, seed)
    inside = (pixels[:, 0] >= 4) & (pixels[:, 0] < 956) & (pixels[:, 1] >= 4) & (pixels[:, 1] < 455)
    pixels = pixels[inside]
    xy = np.rint(pixels).astype(int)
    pixels = pixels[allowed[xy[:, 1], xy[:, 0]] > 0]
    # Exclude unsupported template sections (off-paint/dashed/hidden markings)
    # before local fitting, keeping the support fixed during optimization.
    nearby = map_coordinates(distance, [pixels[:, 1], pixels[:, 0]], order=1, mode="constant", cval=25) < 15
    pixels = pixels[nearby]
    if len(pixels) < 80:
        return None, dict(reason="insufficient_visible_template", line_pixels=evidence), thin

    def distances(delta):
        shifted = transform(pixels, correction_matrix(delta))
        return map_coordinates(distance, [shifted[:, 1], shifted[:, 0]], order=1, mode="constant", cval=25)

    # Bound local correction: cannot recover a different court half or camera.
    def residual(delta):
        return np.r_[np.minimum(distances(delta), 25)/np.sqrt(len(pixels)), .018*np.asarray(delta)]

    zero = np.zeros(8)
    before = distances(zero)
    optimized = least_squares(residual, zero, bounds=(-20, 20), max_nfev=80, loss="soft_l1", f_scale=.5)
    after = distances(optimized.x)
    before_score = float(np.mean(np.minimum(before, 25)))
    after_score = float(np.mean(np.minimum(after, 25)))
    diagnostic = dict(line_pixels=evidence, template_samples=len(pixels),
                       mean_capped_line_distance_before_960_px=before_score,
                       mean_capped_line_distance_after_960_px=after_score,
                       control_point_delta_960_px=optimized.x.reshape(4, 2).tolist(),
                       optimizer_success=bool(optimized.success), evaluations=int(optimized.nfev),
                       accuracy_metrics=None, reason=None)
    if after_score > before_score*.95 or np.abs(optimized.x).max() >= 19.9:
        return None, dict(diagnostic, reason="weak_improvement_or_correction_bound"), thin
    corrected = np.linalg.inv(source_scale) @ correction_matrix(optimized.x) @ seed
    return np.linalg.inv(corrected), diagnostic, thin
