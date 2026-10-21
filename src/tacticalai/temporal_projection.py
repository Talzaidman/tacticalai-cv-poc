"""Offline robust Kalman/RTS smoothing in court coordinates, never H entries.

Camera measurements are the court positions of nine fixed image-plane probes.
They have bounded, useful scale even when most of the court is off-screen.
The smoothed correspondences are refitted to a homography. No player is created
when absent; camera interpolation is bounded, explicitly labelled and cut-aware.
"""
import cv2
import numpy as np

from .calibration import transform


def kalman_rts(values, fps, measurement_sigma=.25, acceleration_sigma=6., variance_scale=None):
    """Independent constant-velocity states with shared time model; NaN=missing.

    RTS uses future observations: appropriate for offline presentation, not a
    claim of causal/live filtering. Call separately for independent shots/tracks.
    """
    values = np.asarray(values, float)
    count, dimensions = values.shape
    if count == 0:
        return values.copy()
    observed = np.isfinite(values).all(axis=1)
    indices = np.flatnonzero(observed)
    if not len(indices):
        return values.copy()
    dt = 1/fps
    transition = np.array([[1., dt], [0., 1.]])
    process = acceleration_sigma**2*np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
    state = np.column_stack((values[indices[0]], np.zeros(dimensions)))
    if len(indices) >= 2:
        state[:, 1] = (values[indices[1]]-values[indices[0]])/((indices[1]-indices[0])*dt)
    covariance = np.broadcast_to(np.diag([measurement_sigma**2*10, 100.]), (dimensions, 2, 2)).copy()
    filtered, covariances, predicted, predicted_covariances = [], [], [], []
    for index in range(count):
        if index:
            state = state @ transition.T
            covariance = transition @ covariance @ transition.T + process
        predicted.append(state.copy())
        predicted_covariances.append(covariance.copy())
        if observed[index]:
            noise = measurement_sigma**2*(variance_scale[index] if variance_scale is not None else 1.)
            innovation_variance = covariance[:, 0, 0]+noise
            gain = covariance[:, :, 0]/innovation_variance[:, None]
            state += gain*(values[index]-state[:, 0])[:, None]
            # Joseph form maintains positive covariance under roundoff.
            operator = np.eye(2)[None]-gain[:, :, None]*np.array([1., 0.])[None, None, :]
            covariance = operator @ covariance @ operator.transpose(0, 2, 1)+noise*gain[:, :, None]*gain[:, None, :]
        filtered.append(state.copy())
        covariances.append(covariance.copy())
    smoothed = np.array(filtered)
    covariance_next = covariances[-1]
    for index in range(count-2, -1, -1):
        gain = covariances[index] @ transition.T @ np.linalg.inv(predicted_covariances[index+1])
        smoothed[index] += np.einsum('dij,dj->di', gain, smoothed[index+1]-predicted[index+1])
        covariance_next = covariances[index]+gain @ (covariance_next-predicted_covariances[index+1]) @ gain.transpose(0, 2, 1)
    return smoothed[:, :, 0]


def image_probes(image_size):
    width, height = image_size
    return np.array([[x*width, y*height] for y in (.62, .75, .88) for x in (.2, .5, .8)])


def robust_camera_outliers(values, cuts, radius=4, threshold_m=.8):
    """Leave-one-out robust local linear prediction, preserving smooth pans."""
    rejected = np.zeros(len(values), bool)
    errors = np.full(len(values), np.nan)
    shot = np.cumsum(cuts)
    for index, current in enumerate(values):
        if not np.isfinite(current).all():
            continue
        neighbors = [j for j in range(max(0, index-radius), min(len(values), index+radius+1))
                     if j != index and shot[j] == shot[index] and np.isfinite(values[j]).all()]
        if len(neighbors) < 4:
            continue
        offsets = np.array(neighbors)-index
        nearby = values[neighbors]
        slopes = [(nearby[b]-nearby[a])/(offsets[b]-offsets[a])
                  for a in range(len(neighbors)) for b in range(a+1, len(neighbors))]
        slope = np.median(slopes, axis=0)
        prediction = np.median(nearby-offsets[:, None]*slope, axis=0)
        error = float(np.sqrt(np.mean((current-prediction)**2)))
        scatter = np.median(np.sqrt(np.mean((nearby-prediction-offsets[:, None]*slope)**2, axis=1)))
        errors[index] = error
        rejected[index] = error > max(threshold_m, 4*scatter)
    return rejected, errors


def smooth_cameras(matrices, image_size, fps, cuts, max_gap=5):
    probes = image_probes(image_size)
    measured = np.full((len(matrices), 18), np.nan)
    for index, matrix in enumerate(matrices):
        if matrix is not None:
            points = transform(probes, matrix)
            if np.isfinite(points).all() and np.abs(points).max() < 100:
                measured[index] = points.ravel()
    rejected, errors = robust_camera_outliers(measured, cuts)
    usable = measured.copy()
    usable[rejected] = np.nan
    result = [None]*len(matrices)
    diagnostics = [dict(method="unavailable", reason="unsupported_camera_interval", accuracy_metrics=None) for _ in matrices]
    # Split on scene cuts and gaps longer than the explicit interpolation bound.
    shot = np.cumsum(cuts)
    for label in np.unique(shot):
        indices = np.flatnonzero((shot == label) & np.isfinite(usable).all(axis=1))
        if not len(indices):
            continue
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > max_gap+1)+1)
        for group in groups:
            first, last = int(group[0]), int(group[-1])
            filtered = kalman_rts(usable[first:last+1], fps)
            for offset, points in enumerate(filtered):
                index = first+offset
                # Refit correspondences, never average matrix coefficients.
                matrix, _ = cv2.findHomography(probes, points.reshape(-1, 2), 0)
                if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
                    continue
                residual = float(np.sqrt(np.mean((transform(probes, matrix)-points.reshape(-1, 2))**2)))
                if residual > .25:
                    continue
                result[index] = matrix
                estimated = not np.isfinite(usable[index]).all()
                diagnostics[index] = dict(method="kalman_rts_camera_gap_estimate" if estimated else "kalman_rts_camera_smoothed",
                    reason=None, measurement_rejected=bool(rejected[index]),
                    local_measurement_error_m=float(errors[index]) if np.isfinite(errors[index]) else None,
                    correspondence_refit_rmse_m=residual, shot=int(shot[index]), offline_future_frames_used=True,
                    accuracy_metrics=None)
    return result, diagnostics, measured, rejected


def smooth_players(frames, fps, cuts, identity_field=None, role_filter='player'):
    """Filter each contiguous observed ID/role segment; never fill absent rows."""
    segments = []
    active = {}
    for index, frame in enumerate(frames):
        next_active = {}
        for row in frame['tracks']:
            if row['court_xy_m'] is None or (role_filter is not None and row.get('role') != role_filter):
                continue
            if identity_field is not None and row.get(identity_field) is None:
                continue
            key = (row['track_id'], row['role'])
            if identity_field is not None:
                key += (row[identity_field],)
            segment = active.get(key)
            if segment and not cuts[index]:
                previous = segment[-1][1]['court_xy_m']
                # Do not blend a likely ID reassociation or large projection jump.
                if np.linalg.norm(np.array(row['court_xy_m'])-previous) > .6+12/fps:
                    segment = None
            if segment is None:
                segment = []
                segments.append(segment)
            segment.append((index, row))
            next_active[key] = segment
        active = next_active
    for segment in segments:
        points = np.array([row['court_xy_m'] for _, row in segment])
        filtered = kalman_rts(points, fps, measurement_sigma=.14, acceleration_sigma=12.)
        for (_, row), point in zip(segment, filtered):
            row['court_xy_camera_smoothed_m'] = row['court_xy_m']
            row['court_xy_m'] = point.tolist()
            row['court_region'] = 'inside' if 0 <= point[0] <= 28 and 0 <= point[1] <= 15 else 'outside'
            row['position_temporal_filter'] = dict(method='kalman_rts_observed_track_segment',
                                                  segment_observations=len(segment), offline_future_frames_used=True)


def roughness(points):
    """Second-difference displacement proxy, not acceleration/accuracy ground truth."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return None
    difference = np.diff(points, n=2, axis=0)
    magnitudes = np.linalg.norm(difference.reshape(len(difference), -1, 2), axis=2)
    finite = magnitudes[np.isfinite(magnitudes)]
    return dict(median=float(np.median(finite)), p90=float(np.percentile(finite, 90)), samples=int(len(finite))) if len(finite) else None
