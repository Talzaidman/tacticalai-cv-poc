"""Adapt KaliCalib's published heatmap extraction and camera refinement.

Reference: CEA-LIST/KaliCalib @ 3b3f52b59e8e38e155ba7e9c681281d44748c173
kalicalib/estimateHomography.py and modeling/example_camera_model.py.
Original algorithm: winner-channel centroids, 35px RANSAC, constrained pinhole
camera refinement and 100..1800cm floor-span check. Unlike upstream, failures
return null rather than a precomputed mean camera. All fitting uses 960x540
coordinates consistently; final matrices are explicitly scaled to the source.
"""
import cv2
import numpy as np

from .auto_calibration import KaliDetector, grid_points
from .calibration import transform


class ReferenceDetector(KaliDetector):
    def __init__(self, folder, checkpoint):
        # Reuse architecture loading; load the requested weights explicitly.
        super().__init__(folder)
        self.model.load_state_dict(self.torch.load(folder/checkpoint, map_location="cpu", weights_only=True))

    def heatmaps(self, frame):
        rgb = cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2RGB)
        array = rgb.astype(np.float32)/255
        array = (array-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
        tensor = self.torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
        with self.torch.inference_mode():
            return self.model(tensor)[0].numpy()


def extract_centroids(heatmaps):
    """Same winning-channel mask and whole-mask moments as upstream."""
    maximum = heatmaps.max(axis=0)
    points = []
    for index, world in enumerate(grid_points()):
        mask = ((heatmaps[index] == maximum) & (heatmaps[index] > 0)).astype(np.uint8)
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        points.append(dict(index=index, world_xy_cm=(world*100).tolist(),
                           image_xy_960=[4*moments["m10"]/moments["m00"], 4*moments["m01"]/moments["m00"]],
                           winning_pixels=int(mask.sum())))
    return points


def refine_camera(world_cm, pixels):
    """Equivalent OpenCV portion of upstream compute_camera_model, no fallback."""
    if len(world_cm) <= 5:
        return None, dict(reason="fewer_than_six_camera_points")
    objects = np.column_stack((world_cm, np.zeros(len(world_cm)))).astype(np.float32)
    flags = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
             cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST)
    try:
        rmse, intrinsic, distortion, rotations, translations = cv2.calibrateCamera(
            [objects], [pixels.astype(np.float32)], (960, 540), None, None, flags=flags)
        rotation = cv2.Rodrigues(rotations[0])[0]
        matrix = intrinsic @ np.column_stack((rotation[:, :2], translations[0]))
        if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
            return None, dict(reason="degenerate_camera")
        probe = transform([[240, 270], [720, 270]], np.linalg.inv(matrix))
        distance = float(np.linalg.norm(probe[1]-probe[0]))
        diagnostic = dict(camera_rmse_px=float(rmse), probe_floor_span_cm=distance,
                          intrinsic=intrinsic.tolist(), distortion=distortion.ravel().tolist(),
                          rotation=rotation.tolist(), translation_cm=translations[0].ravel().tolist(),
                          reason=None if 100 <= distance <= 1800 else "upstream_floor_span_rejected")
        if not np.isfinite(distance) or not 100 <= distance <= 1800:
            return None, diagnostic
        return matrix, diagnostic
    except (cv2.error, np.linalg.LinAlgError) as error:
        return None, dict(reason="camera_refinement_failed", error=str(error))


def estimate(points, image_size):
    result = dict(raw_image_to_court_h=None, refined_image_to_court_h=None,
                  correspondences=len(points), inliers=0, accuracy_metrics=None)
    if len(points) < 4:
        return dict(result, reason="fewer_than_four_points")
    world = np.array([p["world_xy_cm"] for p in points], np.float64)
    pixels = np.array([p["image_xy_960"] for p in points], np.float64)
    cv2.setRNGSeed(0)
    raw, mask = cv2.findHomography(world, pixels, cv2.RANSAC, 35, maxIters=2000)
    if raw is None or not np.isfinite(raw).all() or np.linalg.matrix_rank(raw) < 3:
        return dict(result, reason="ransac_failed")
    good = mask.ravel().astype(bool)
    scale = np.diag([image_size[0]/960, image_size[1]/540, 1.])
    centimetres = np.diag([100., 100., 1.])
    result.update(raw_image_to_court_h=np.linalg.inv(scale @ raw @ centimetres).tolist(),
                  inliers=int(good.sum()), inlier_indices=[p["index"] for p, keep in zip(points, good) if keep],
                  raw_fit_rmse_960_px=float(np.sqrt(np.mean(np.sum((transform(world[good], raw)-pixels[good])**2, axis=1)))))
    refined, diagnostic = refine_camera(world[good], pixels[good])
    result["camera"] = diagnostic
    if refined is not None:
        result["refined_image_to_court_h"] = np.linalg.inv(scale @ refined @ centimetres).tolist()
    return result
