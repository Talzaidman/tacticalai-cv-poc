"""Experimental display-only floor support, fitted automatically per frame.

An image-floor model is not a role classifier or ground-truth court mask.
Negative local floor evidence can hide unlinked/sideline people, while measured
tracks and all suppression decisions remain in exports.
"""
import cv2
import numpy as np

from .calibration import transform


DEFAULTS = dict(training_inset_m=2., minimum_training_pixels=300, clusters=3,
                minimum_cluster_fraction=.08, minimum_lightness=35.,
                color_distance_lab=18., minimum_context_pixels=35,
                minimum_floor_fraction=.12, context_width=.45,
                context_above_bottom=.3, context_below_bottom=.25)


def floor_model(image, matrix, masks, config):
    if matrix is None:
        return None
    h = np.asarray(matrix, dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        return None
    height, width = image.shape[:2]
    sx, sy = 640/width, 360/height
    inset = config['training_inset_m']
    polygon = transform([[inset, inset], [28-inset, inset], [28-inset, 15-inset], [inset, 15-inset]], np.linalg.inv(h))
    if not np.isfinite(polygon).all() or np.max(np.abs(polygon)) > 1e7:
        return None
    reduced = cv2.resize(image, (640, 360))
    training = np.zeros((360, 640), np.uint8)
    cv2.fillPoly(training, [np.rint(polygon*[sx, sy]).astype(np.int32)], 1)
    occupied = np.zeros_like(training)
    for row in masks:
        if row.get('silhouette_xy'):
            points = np.rint(np.asarray(row['silhouette_xy'])*[sx, sy]).astype(np.int32)
            cv2.fillPoly(occupied, [points], 1)
        else:
            x1, y1, x2, y2 = row['bbox_xyxy']
            cv2.rectangle(occupied, (round(x1*sx), round(y1*sy)), (round(x2*sx), round(y2*sy)), 1, -1)
    occupied = cv2.dilate(occupied, np.ones((3, 3), np.uint8))
    lab = cv2.cvtColor(reduced.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    pixels = lab[(training > 0) & (occupied == 0) & (lab[:, :, 0] >= config['minimum_lightness'])]
    if len(pixels) < config['minimum_training_pixels']:
        return None
    # Deterministic spatially ordered subsampling keeps CPU fitting inexpensive.
    pixels = np.ascontiguousarray(pixels[::max(1, len(pixels)//4096)], np.float32)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(pixels, config['clusters'], None,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 25, .2), 1, cv2.KMEANS_PP_CENTERS)
    weights = np.bincount(labels.ravel(), minlength=config['clusters'])/len(pixels)
    centers = centers[weights >= config['minimum_cluster_fraction']]
    if not len(centers):
        return None
    distance = np.linalg.norm(lab[:, :, None]-centers[None, None], axis=3).min(axis=2)
    return dict(floor_pixels=distance <= config['color_distance_lab'], occupied=occupied > 0,
                centers_lab=centers.tolist(), training_pixels=len(pixels), scale=(sx, sy))


def context_evidence(box, model, config):
    result = dict(floor_fraction=None, context_pixels=0, reason='no_floor_model')
    if model is None:
        return result
    sx, sy = model['scale']
    x1, y1, x2, y2 = np.asarray(box)*[sx, sy, sx, sy]
    width, height = x2-x1, y2-y1
    left, right = max(0, int(x1-config['context_width']*width)), min(640, int(x2+config['context_width']*width))
    top, bottom = max(0, int(y2-config['context_above_bottom']*height)), min(360, int(y2+config['context_below_bottom']*height))
    if right <= left or bottom <= top:
        return dict(result, reason='no_visible_context')
    available = ~model['occupied'][top:bottom, left:right]
    n = int(available.sum())
    if n < config['minimum_context_pixels']:
        return dict(result, context_pixels=n, reason='occluded_or_insufficient_context')
    fraction = float(model['floor_pixels'][top:bottom, left:right][available].mean())
    return dict(floor_fraction=fraction, context_pixels=n,
                reason='floor_context_supported' if fraction >= config['minimum_floor_fraction'] else 'little_floor_context')


def apply_display_filter(image, frame, config):
    model = floor_model(image, frame.get('image_to_court_h'), frame['tracks'], config)
    stats = dict(shown=0, hidden=0, unknown_context=0)
    for row in frame['tracks']:
        evidence = context_evidence(row['bbox_xyxy'], model, config)
        # Protect existing automatically classified players from this appearance
        # heuristic: lack of visible floor is also possible in crowded play.
        eligible = row.get('role') != 'player'
        hidden = eligible and evidence['reason'] == 'little_floor_context'
        row.update(display_on_video=not hidden, floor_context_evidence=evidence,
            display_filter_reason='low_floor_context_unlinked_or_nonplayer' if hidden else
            'preserved_classified_player' if row.get('role') == 'player' else evidence['reason'])
        stats['hidden' if hidden else 'shown'] += 1
        stats['unknown_context'] += evidence['floor_fraction'] is None
    frame['display_floor_filter'] = dict(stats, model_centers_lab=model['centers_lab'] if model else None)
    return stats
