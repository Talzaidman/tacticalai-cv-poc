"""CPU-only homography fitting and conservative, bounded keyframe interpolation."""
import cv2
import numpy as np

from .court import landmarks


def transform(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    result = np.full((len(points), 2), np.nan)
    good = np.abs(homogeneous[:, 2]) > 1e-9
    result[good] = homogeneous[good, :2] / homogeneous[good, 2, None]
    return result


def fit(anchors, court, threshold_px=4.0):
    """Fit court -> image with a pixel RANSAC threshold; invert for projection.

    Validation points never participate in fitting. Their residuals are checks
    against manual clicks, not independently measured player-position accuracy.
    """
    known = landmarks(court)
    names = [name for name, a in anchors.items() if a.get("use", "fit") == "fit"]
    if len(names) < 4:
        raise ValueError("At least four fit anchors are required")
    if any(name not in known for name in anchors):
        raise ValueError("Unknown court landmark")
    world = np.asarray([known[n] for n in names], dtype=np.float64)
    pixels = np.asarray([anchors[n]["image_xy"] for n in names], dtype=np.float64)
    if not np.isfinite(pixels).all():
        raise ValueError("Non-finite anchor")
    if any(a.get("use", "fit") not in ("fit", "check") for a in anchors.values()):
        raise ValueError("Anchor use must be fit or check")
    # Detect any collinear triple when fitting exactly four: rank alone misses it.
    from itertools import combinations
    for pts in (world, pixels):
        if np.linalg.matrix_rank(np.column_stack((pts, np.ones(len(pts))))) < 3:
            raise ValueError("Collinear anchors cannot define a homography")
        if len(pts) == 4 and any(abs(np.linalg.det(np.column_stack((pts[list(t)], np.ones(3))))) < 1e-6
                                 for t in combinations(range(4), 3)):
            raise ValueError("Four-anchor fit contains a collinear triple")
    cv2.setRNGSeed(0)
    matrix, mask = cv2.findHomography(world, pixels, cv2.RANSAC, threshold_px)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Degenerate homography")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise ValueError("Fewer than four inlier anchors")
    inverse = np.linalg.inv(matrix)
    errors = np.linalg.norm(transform(world, matrix) - pixels, axis=1)
    checks = []
    for name, a in anchors.items():
        if a.get("use") == "check":
            px = np.asarray(a["image_xy"], dtype=float)
            pred_px = transform([known[name]], matrix)[0]
            pred_m = transform([px], inverse)[0]
            if not np.isfinite([*px, *pred_px, *pred_m]).all():
                raise ValueError("Non-finite validation projection")
            checks.append(dict(name=name, error_px=float(np.linalg.norm(pred_px-px)),
                               error_m=float(np.linalg.norm(pred_m-known[name]))))
    diagnostics = dict(fit_anchors=len(names), inliers=int(inliers.sum()),
                       inlier_names=[n for n, good in zip(names, inliers) if good],
                       support_polygon_image_xy=cv2.convexHull(pixels[inliers].astype(np.float32)).reshape(-1, 2).tolist(),
                       fit_rmse_px=float(np.sqrt(np.mean(errors[inliers]**2))),
                       checks=checks, accuracy_metrics=None)
    return inverse, diagnostics


class Calibration:
    def __init__(self, document):
        self.document = document
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported calibration schema")
        self.court = document["court"]
        for key in ("length_m", "width_m", "key_depth_m", "key_width_m"):
            if not np.isfinite(self.court[key]) or self.court[key] <= 0:
                raise ValueError("Court dimensions must be finite and positive")
        if self.court["key_depth_m"] >= self.court["length_m"]/2 or self.court["key_width_m"] >= self.court["width_m"]:
            raise ValueError("Key dimensions do not fit the court")
        self.keys = sorted(document["keyframes"], key=lambda k: k["frame_idx"])
        ids = [k["frame_idx"] for k in self.keys]
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Keyframe indices must be unique nonnegative integers")
        self.threshold = document.get("ransac_threshold_px", 4.0)
        self.max_gap = document.get("max_interpolation_gap_frames", 0)
        if type(self.max_gap) is not int or self.max_gap < 0 or not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("Invalid fitting/interpolation settings")
        video = document["video"]
        for keyframe in self.keys:
            if keyframe["frame_idx"] >= video["frames"]:
                raise ValueError("Keyframe outside video")
            for anchor in keyframe["anchors"].values():
                x, y = anchor["image_xy"]
                if not (0 <= x < video["width"] and 0 <= y < video["height"]):
                    raise ValueError("Anchor outside source image")
        self.fitted = {k["frame_idx"]: fit(k["anchors"], self.court, self.threshold) for k in self.keys}

    def at(self, frame_idx):
        if frame_idx in self.fitted:
            matrix, diagnostic = self.fitted[frame_idx]
            return matrix, dict(diagnostic, method="manual_keyframe", keyframes=[frame_idx])
        for left, right in zip(self.keys, self.keys[1:]):
            a, b = left["frame_idx"], right["frame_idx"]
            if a < frame_idx < b and b-a <= self.max_gap:
                # An interval requires explicit review; no silent interpolation
                # through cuts, uncertain annotations or long unsupported gaps.
                if not left.get("interpolate_to_next", False):
                    break
                names = set(left["anchors"]) & set(right["anchors"])
                names = sorted(n for n in names if all(k["anchors"][n].get("use", "fit") == "fit" for k in (left, right)))
                if len(names) < 4:
                    break
                t = (frame_idx-a)/(b-a)
                anchors = {n: {"image_xy": ((1-t)*np.asarray(left["anchors"][n]["image_xy"])
                                            + t*np.asarray(right["anchors"][n]["image_xy"])).tolist()}
                           for n in names}
                try:
                    matrix, diagnostic = fit(anchors, self.court, self.threshold)
                except ValueError:
                    break
                # Synthetic fit residuals aren't temporal validation measurements.
                diagnostic["fit_rmse_px"] = None
                return matrix, dict(diagnostic, method="interpolated_anchors_unvalidated",
                                    keyframes=[a, b])
        return None, dict(method="unavailable", keyframes=[], accuracy_metrics=None)


def project_track(row, matrix, quality, court, image_size):
    result = dict(row)
    x1, y1, x2, y2 = row["bbox_xyxy"]
    foot = [(x1+x2)/2, y2]
    clipped = x1 <= 1 or y1 <= 1 or x2 >= image_size[0]-1 or y2 >= image_size[1]-1
    xy = transform([foot], matrix)[0] if matrix is not None and not clipped else None
    valid = xy is not None and np.isfinite(xy).all()
    support = quality.get("support_polygon_image_xy")
    inside_support = cv2.pointPolygonTest(np.asarray(support, dtype=np.float32), tuple(foot), False) >= 0 if support else None
    result.update(footpoint_image_xy=foot, footpoint_method="bbox_bottom_center",
                  foot_contact_visibility="unknown", court_xy_m=xy.tolist() if valid else None,
                  position_source="floor_projection_approximation" if valid else "unavailable",
                  court_projection_quality=dict(calibration_method=quality["method"],
                                                anchor_support="inside" if inside_support else ("extrapolated" if inside_support is not None else "unknown"),
                                                footpoint_quality="clipped" if clipped else "unverified",
                                                reason=None if valid else ("clipped_box" if clipped else "no_valid_calibration")),
                  court_region=("inside" if 0 <= xy[0] <= court["length_m"] and 0 <= xy[1] <= court["width_m"] else "outside") if valid else "unknown")
    return result
