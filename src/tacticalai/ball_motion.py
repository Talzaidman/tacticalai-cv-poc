"""Experimental camera-compensated image evidence; no annotations or court mask."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    working_width: int = 640
    lag: int = 2
    min_motion: float = 10.
    min_area: int = 5
    max_area: int = 180
    min_circularity: float = .3
    min_detection_motion: float = 25.

    def __post_init__(self):
        if (not all(math.isfinite(value) for value in vars(self).values()) or
                not isinstance(self.working_width, int) or not isinstance(self.lag, int) or
                self.working_width < 32 or self.lag < 1 or self.min_motion <= 0 or
                not 0 < self.min_detection_motion <= 255 or
                not 0 < self.min_area < self.max_area or not 0 < self.min_circularity <= 1):
            raise ValueError('Invalid motion configuration')


def align_background(neighbor, image):
    """Fit on scene texture; reject unsupported transforms rather than use identity."""
    a = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.full(a.shape, 255, np.uint8)
    mask[round(.65*a.shape[0]):] = 0
    points = cv2.goodFeaturesToTrack(a, 600, .02, 5, mask=mask)
    if points is None or len(points) < 15:
        return None, dict(reason='insufficient_features')
    target, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None)
    if target is None or status is None:
        return None, dict(reason='optical_flow_failed')
    good = status.ravel().astype(bool)
    if good.sum() < 15:
        return None, dict(reason='insufficient_flow')
    matrix, inliers = cv2.estimateAffinePartial2D(points[good], target[good],
        method=cv2.RANSAC, ransacReprojThreshold=2., maxIters=2000, confidence=.99)
    if (matrix is None or inliers is None or not np.isfinite(matrix).all() or
            inliers.sum() < 12 or inliers.mean() < .4):
        return None, dict(reason='unsupported_camera_motion')
    scale = float(np.hypot(*matrix[:, 0]))
    if not .85 <= scale <= 1.18:
        return None, dict(reason='implausible_camera_scale')
    warped = cv2.warpAffine(neighbor, matrix, (image.shape[1], image.shape[0]))
    valid = cv2.warpAffine(np.ones(a.shape, np.uint8), matrix, (image.shape[1], image.shape[0]),
                           flags=cv2.INTER_NEAREST).astype(bool)
    difference = np.max(np.abs(image.astype(float)-warped.astype(float)), axis=2)
    difference[~valid] = 0
    return difference, dict(reason=None, affine=matrix.tolist(),
                            inliers=int(inliers.sum()), inlier_fraction=float(inliers.mean()))


def motion_candidates(image, differences, source_size, config=MotionConfig()):
    if not differences:
        return [], dict(reason='no_supported_camera_neighbors')
    motion = np.minimum.reduce(differences)
    b, g, r = cv2.split(image.astype(float))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color = ((r-g >= 6) & (r >= b*.95) & (hsv[:, :, 1] >= 18) &
             (hsv[:, :, 1] <= 200) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 245))
    moving = cv2.dilate((motion >= config.min_motion).astype(np.uint8), np.ones((3, 3), np.uint8))
    mask = (color & moving.astype(bool)).astype(np.uint8)*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    factor = np.asarray(source_size)/[image.shape[1], image.shape[0]]
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        if (not config.min_area <= area <= config.max_area or
                min(width, height) < 3 or max(width, height) > 26 or not .45 <= width/height <= 2.2):
            continue
        circularity = 4*np.pi*area/max(perimeter**2, 1)
        if circularity < config.min_circularity:
            continue
        local_mask = np.zeros((height, width), np.uint8)
        cv2.drawContours(local_mask, [contour-[x, y]], -1, 1, -1)
        strength = float(np.mean(motion[y:y+height, x:x+width][local_mask.astype(bool)]))
        if strength < config.min_motion:
            continue
        score = min(.99, .4*circularity + .6*min(1., strength/60))
        box = (np.asarray([x, y, x+width, y+height])*np.tile(factor, 2)).tolist()
        candidates.append(dict(bbox_xyxy=box, center_image_xy=((np.asarray(box[:2])+box[2:])/2).tolist(),
            det_confidence=float(score), proposal_score=float(score),
            score_kind='uncalibrated_color_motion_shape', motion_strength=strength,
            circularity=float(circularity), detection_source='automatic_color_motion_blob'))
    return candidates, dict(reason=None, candidates=len(candidates))


def filter_detections(detections, differences, source_size, config=MotionConfig()):
    """A background-rejection diagnostic, not a probability or ball classifier."""
    accepted, evidence = [], []
    motion = np.minimum.reduce(differences) if differences else None
    size = np.asarray(source_size, float)
    for row in detections:
        box = np.asarray(row['bbox_xyxy'], float)
        if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[:2] < 0) or
                np.any(box[2:] > size) or np.any(box[2:] <= box[:2])):
            raise ValueError('Invalid motion-filter detection box')
        support = None
        reason = 'no_supported_camera_neighbors'
        if motion is not None:
            scale = np.asarray([motion.shape[1], motion.shape[0]])/size
            x1, y1 = np.floor(box[:2]*scale).astype(int)
            x2, y2 = np.ceil(box[2:]*scale).astype(int)
            support = float(np.median(motion[y1:y2, x1:x2]))
            reason = 'motion_supported' if support >= config.min_detection_motion else 'insufficient_residual_motion'
        keep = reason == 'motion_supported'
        evidence.append(dict(detection=row, accepted=keep, reason=reason,
                             median_residual_0_255=support))
        if keep:
            accepted.append(row)
    return accepted, evidence


def analyze_motion(images, cuts, source_size, config=MotionConfig(), detections=None):
    if len(images) != len(cuts):
        raise ValueError('Motion frame/cut lengths differ')
    if detections is not None and len(detections) != len(images):
        raise ValueError('Motion frame/detection lengths differ')
    results = []
    for index, image in enumerate(images):
        differences, cameras = [], []
        for other in (index-config.lag, index+config.lag):
            if other < 0 or other >= len(images):
                continue
            lower, upper = sorted((index, other))
            if any(cuts[lower+1:upper+1]):
                continue
            difference, quality = align_background(images[other], image)
            cameras.append(dict(neighbor_frame=other, **quality))
            if difference is not None:
                differences.append(difference)
        candidates, quality = motion_candidates(image, differences, source_size, config)
        result = dict(candidates=candidates, quality=quality, cameras=cameras)
        if detections is not None:
            accepted, evidence = filter_detections(detections[index], differences, source_size, config)
            result.update(accepted_detections=accepted, detection_evidence=evidence)
        results.append(result)
    return results
