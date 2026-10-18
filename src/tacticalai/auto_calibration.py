"""Experimental automatic court correspondences and short-term floor tracking.

No manual coordinates enter detection or fitting. A geometric pass is not an
accuracy measurement. KaliCalib's two elevated basket channels are excluded.
"""
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from .calibration import transform


def grid_points():
    """KaliCalib getFieldPoints: first 91 floor points, centimetres -> metres."""
    points, offset, step = [], 0, 175
    for _ in range(7):
        points.extend([[i * 28 / 12, 15 - offset / 100] for i in range(13)])
        offset += step
        step += 30
    return np.asarray(points, dtype=np.float64)


class KaliDetector:
    def __init__(self, folder, threads=4):
        import torch
        self.torch = torch
        torch.set_num_threads(threads)
        spec = importlib.util.spec_from_file_location("tacticalai_kalicalib", Path(folder)/"model_resnet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.makeModel()
        self.model.load_state_dict(torch.load(Path(folder)/"model_test.pth", map_location="cpu", weights_only=True))
        self.model.eval()

    def detect(self, frame, probability=.3, min_area=3):
        rgb = cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2RGB)
        array = (rgb.astype(np.float32)/255 - [.485, .456, .406]) / [.229, .224, .225]
        tensor = self.torch.from_numpy(array.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
        with self.torch.inference_mode():
            heatmaps = self.model(tensor)[0].numpy()
        winner = heatmaps.argmax(axis=0)
        points = []
        # Channels 91/92 describe elevated baskets; 93 is background.
        for index, world in enumerate(grid_points()):
            mask = ((winner == index) & (heatmaps[index] >= probability)).astype(np.uint8)
            count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
            if count <= 1:
                continue
            component = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            if stats[component, cv2.CC_STAT_AREA] < min_area:
                continue
            x, y = centres[component]
            points.append(dict(index=index, world_xy_m=world.tolist(),
                               image_xy=[float(x*4*frame.shape[1]/960), float(y*4*frame.shape[0]/540)],
                               confidence=float(heatmaps[index][labels == component].mean()),
                               heatmap_area=int(stats[component, cv2.CC_STAT_AREA])))
        return points


def fit_candidate(points, image_size, threshold_960=8., min_inliers=8, min_ratio=.4):
    """RANSAC plus conservative degeneracy/support checks, not verified accuracy."""
    quality = dict(method="unavailable", reason="insufficient_correspondences", candidates=len(points),
                   inliers=0, accuracy_metrics=None)
    if len(points) < min_inliers:
        return None, quality
    world = np.asarray([p["world_xy_m"] for p in points], dtype=float)
    pixels = np.asarray([p["image_xy"] for p in points], dtype=float)
    if not np.isfinite([world, pixels]).all():
        return None, dict(quality, reason="nonfinite_correspondences")
    cv2.setRNGSeed(0)
    threshold = threshold_960 * image_size[0]/960
    matrix, mask = cv2.findHomography(world, pixels, cv2.RANSAC, threshold, maxIters=3000)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        return None, dict(quality, reason="degenerate_fit")
    good = mask.ravel().astype(bool)
    ratio = float(good.mean())
    hull = cv2.convexHull(pixels[good].astype(np.float32))
    support = float(cv2.contourArea(hull)/(image_size[0]*image_size[1]))
    residual = np.linalg.norm(transform(world[good], matrix)-pixels[good], axis=1)
    quality.update(inliers=int(good.sum()), inlier_ratio=ratio, support_fraction=support,
                   inlier_indices=[p["index"] for p, g in zip(points, good) if g],
                   support_polygon_image_xy=hull.reshape(-1, 2).tolist(),
                   fit_rmse_px=float(np.sqrt(np.mean(residual**2))))
    # Diagnostic only: rejected matrices must never reach player projection.
    quality["diagnostic_court_to_image_h"] = matrix.tolist()
    if good.sum() < min_inliers or ratio < min_ratio or support < .005:
        return None, dict(quality, reason="weak_inlier_support")
    # Reject projective horizons crossing the court: no matrix/default fallback.
    corners = np.array([[0, 0, 1], [28, 0, 1], [28, 15, 1], [0, 15, 1]])
    denominator = corners @ matrix[2]
    if np.min(denominator)*np.max(denominator) <= 0:
        return None, dict(quality, reason="horizon_crosses_court")
    return np.linalg.inv(matrix), dict(quality, method="automatic_candidate_unverified", reason=None)


class FloorTracker:
    """Propagate an automatic fit using masked floor features; bound the age."""
    def __init__(self, max_age=4):
        self.previous = self.features = self.matrix = None
        self.age = 0
        self.max_age = max_age

    def reset(self, gray, matrix, boxes, court):
        self.matrix = matrix
        self.previous = gray.copy()
        self.age = 0
        self.features = None
        if matrix is None:
            return
        polygon = transform([[0, 0], [28, 0], [28, 15], [0, 15]], np.linalg.inv(matrix))
        if not np.isfinite(polygon).all() or np.abs(polygon).max() > 1e6:
            self.matrix = None
            return
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(mask, (max(0, int(x1)-5), max(0, int(y1)-5)),
                          (int(x2)+5, int(y2)+5), 0, -1)
        # Broadcast score graphics lie near the bottom; exclude this region.
        mask[int(.85*gray.shape[0]):] = 0
        self.features = cv2.goodFeaturesToTrack(gray, 250, .01, 7, mask=mask)

    def advance(self, gray, boxes):
        unavailable = dict(method="unavailable", reason="floor_tracking_lost", accuracy_metrics=None)
        if self.matrix is None or self.features is None or len(self.features) < 12 or self.age >= self.max_age:
            self.matrix = None
            return None, unavailable
        # A coarse cut test supplements, but does not guarantee, cut detection.
        if np.mean(cv2.absdiff(cv2.resize(gray, (64, 36)), cv2.resize(self.previous, (64, 36)))) > 45:
            self.matrix = None
            return None, dict(unavailable, reason="possible_camera_cut")
        forward, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.features, None)
        if forward is None:
            self.matrix = None
            return None, unavailable
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, forward, None)
        if backward is None:
            self.matrix = None
            return None, unavailable
        good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
        good &= np.linalg.norm(backward-self.features, axis=2).ravel() < 1.5
        xy = forward.reshape(-1, 2)
        for x1, y1, x2, y2 in boxes:
            good &= ~((xy[:, 0] >= x1-5) & (xy[:, 0] <= x2+5) & (xy[:, 1] >= y1-5) & (xy[:, 1] <= y2+5))
        a, b = self.features.reshape(-1, 2)[good], xy[good]
        if len(a) < 12:
            self.matrix = None
            return None, unavailable
        motion, mask = cv2.findHomography(a, b, cv2.RANSAC, 2*gray.shape[1]/640)
        if motion is None or np.linalg.matrix_rank(motion) < 3 or mask.sum() < 12 or mask.mean() < .6:
            self.matrix = None
            return None, unavailable
        self.matrix = self.matrix @ np.linalg.inv(motion)
        self.previous = gray.copy()
        self.features = b[mask.ravel().astype(bool)].reshape(-1, 1, 2)
        self.age += 1
        return self.matrix.copy(), dict(method="automatic_floor_flow_unverified", reason=None,
                                        flow_inliers=int(mask.sum()), frames_since_detection=self.age,
                                        accuracy_metrics=None)
