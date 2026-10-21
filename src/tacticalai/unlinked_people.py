"""Display actual person detections that have no confirmed identity.

No tracker thresholds, person identities or role/team decisions are changed.
Box/mask associations are geometric evidence, never identity assignments.
"""
import copy

import numpy as np
from scipy.optimize import linear_sum_assignment

from .court_gate import court_gate


def box_matches(a, b, minimum_iou):
    """Maximum-cardinality one-to-one IoU matching; never consume two people."""
    if not len(a) or not len(b):
        return []
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    areas_a = np.maximum(0, a[:, 2:]-a[:, :2]).prod(axis=1)
    areas_b = np.maximum(0, b[:, 2:]-b[:, :2]).prod(axis=1)
    iou = intersection / np.maximum(areas_a[:, None]+areas_b-intersection, 1e-9)
    valid = iou >= minimum_iou
    rr, cc = linear_sum_assignment(-(valid*(min(iou.shape)+1)+iou))
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rr, cc) if valid[r, c]]


def unlinked_row(detection, frame, observation_id, source, index):
    row = copy.deepcopy(detection)
    row.update(track_id=None, player_id=None, identity_segment_id=None,
        identity_status='unlinked_detection', role=None, team_id=None, jersey_number=None,
        classification_evidence_id=None, role_scores=None, role_margin=None,
        role_agreement=None, role_samples=0, role_reason='unlinked_detection',
        team_reason='unlinked_detection', team_distances_lab=None,
        team_margin_lab=None, team_agreement=None, team_samples=0,
        classification_source=None, classification_temporal_scope=None,
        court_xy_m=None, court_xy_camera_smoothed_m=None, court_region='unknown',
        shown_on_minimap=False, position_source='image_detection_unlinked',
        court_projection_quality=None, foot_contact_visibility='unknown',
        silhouette_xy=None, silhouette_confidence=None, silhouette_box_iou=None,
        display_label='?', display_caption='Unlinked', display_color_hex='#aaaaaa',
        observation_kind='unlinked_person_detection', observation_id=observation_id,
        detector_source=source, detection_index=index, segmentation_source=None,
        silhouette_alpha=.22, display_label_scale=.34,
        frame_idx=frame['frame_idx'], timestamp_s=frame['timestamp_s'])
    return row


def same_head_duplicate(candidate, reference, minimum_iou=.5, head_tolerance=.1):
    """Only suppress maskless near-identical-head boxes, not generic overlaps."""
    if candidate.get('silhouette_xy') is not None:
        return False
    a, b = np.asarray(candidate['bbox_xyxy']), np.asarray(reference['bbox_xyxy'])
    if not box_matches([a], [b], minimum_iou):
        return False
    size = np.maximum(np.minimum(a[2:]-a[:2], b[2:]-b[:2]), 1)
    head_a, head_b = np.array([(a[0]+a[2])/2, a[1]]), np.array([(b[0]+b[2])/2, b[1]])
    return bool(np.max(np.abs(head_a-head_b)/size) <= head_tolerance)


def add_unlinked(frame, masks, image_size, config):
    if masks['frame_idx'] != frame['frame_idx'] or abs(masks['timestamp_s']-frame['timestamp_s']) > 1e-6:
        raise ValueError('Mask/frame alignment mismatch')
    if not len(masks['boxes']) == len(masks['polygons']) == len(masks['confidences']):
        raise ValueError('Mask arrays differ in length')
    result = copy.deepcopy(frame)
    tracks = result['tracks']
    for index, row in enumerate(tracks):
        row.update(observation_kind='original_track',
                   observation_id=f"{frame['frame_idx']}:track:{index}",
                   detector_source='baseline_yolo26s_track', detection_index=None,
                   segmentation_source='baseline_mask' if row.get('silhouette_xy') else None)
        if row.get('player_id') is None:
            row['display_caption'] = f"Unlinked / T{row['track_id']}"
    track_boxes = [r['bbox_xyxy'] for r in tracks]
    detections = frame.get('detections', [])
    represented = {j for _, j, _ in box_matches(track_boxes,
        [r['bbox_xyxy'] for r in detections], config['detection_track_iou'])}
    rejected = []
    def accept(row, source, index):
        gate = court_gate(row['bbox_xyxy'], frame.get('image_to_court_h'),
                          frame['calibration'], image_size,
                          margin_m=config['court_margin_m'], minimum_overlap=config['minimum_body_court_overlap'])
        if config['court_filter'] and not gate['keep']:
            rejected.append(dict(source=source, index=index, bbox_xyxy=row['bbox_xyxy'], **gate))
            return None
        return dict(row, **gate)
    pending = []
    for index, detection in enumerate(detections):
        if index in represented:
            continue
        accepted = accept(detection, 'yolo26s_detection', index)
        if accepted is not None:
            pending.append(unlinked_row(accepted, frame, f"{frame['frame_idx']}:det:{index}",
                                        'yolo26s_detection', index))
    # Original tracks take precedence when reserving segmentation observations.
    assigned = box_matches(track_boxes, masks['boxes'], config['mask_iou'])
    used_masks = {j for _, j, _ in assigned}
    def assign_mask(row, index, iou):
        polygon = masks['polygons'][index]
        if len(polygon) >= 3:
            row.update(silhouette_xy=copy.deepcopy(polygon),
                silhouette_confidence=masks['confidences'][index], silhouette_box_iou=iou,
                segmentation_source='cached_yolo26s_seg_1280', segmentation_detection_index=index)
    for track_idx, mask_idx, iou in assigned:
        if tracks[track_idx].get('silhouette_xy') is None:
            assign_mask(tracks[track_idx], mask_idx, iou)
    available = [j for j in range(len(masks['boxes'])) if j not in used_masks]
    for pending_idx, available_idx, iou in box_matches([r['bbox_xyxy'] for r in pending],
            [masks['boxes'][j] for j in available], config['mask_iou']):
        index = available[available_idx]
        used_masks.add(index)
        assign_mask(pending[pending_idx], index, iou)
    # The segmenter is also a person detector. Keep its otherwise unmatched
    # observations, provided no already displayed box adequately represents them.
    represented_boxes = track_boxes + [r['bbox_xyxy'] for r in pending]
    for index, box in enumerate(masks['boxes']):
        if index in used_masks or not config['include_segmentation_detections']:
            continue
        if box_matches([box], represented_boxes, config['duplicate_iou']):
            continue
        detection = dict(bbox_xyxy=box, det_confidence=masks['confidences'][index],
                         class_name='person', class_id=0)
        accepted = accept(detection, 'yolo26s_seg_detection', index)
        if accepted is not None:
            row = unlinked_row(accepted, frame, f"{frame['frame_idx']}:seg:{index}",
                               'yolo26s_seg_detection', index)
            assign_mask(row, index, 1.)
            pending.append(row); represented_boxes.append(box)
    duplicate_rows = []
    if config.get('suppress_maskless_same_head_duplicates', False):
        # Prefer measured masks and stronger detections; existing tracks always win.
        pending.sort(key=lambda r: (r['silhouette_xy'] is None, -r['det_confidence']))
        retained = []
        for row in pending:
            if any(same_head_duplicate(row, r, config['duplicate_iou'],
                    config['duplicate_head_tolerance']) for r in tracks+retained):
                duplicate_rows.append(dict(observation_id=row['observation_id'],
                    reason='maskless_overlapping_box_with_near_identical_head', bbox_xyxy=row['bbox_xyxy']))
            else:
                retained.append(row)
        pending = retained
    for row in tracks+pending:
        row['draw_detection_box_fallback'] = row.get('silhouette_xy') is None
    result.update(tracks=tracks+pending, unlinked_display=dict(
        original_tracks=len(tracks), raw_detections=len(detections),
        represented_detections=len(represented), added_unlinked=len(pending),
        added_detector_unlinked=sum(r['detector_source'] == 'yolo26s_detection' for r in pending),
        added_segmentation_unlinked=sum(r['detector_source'] == 'yolo26s_seg_detection' for r in pending),
        added_unlinked_with_mask=sum(r['silhouette_xy'] is not None for r in pending),
        masks_recovered_on_original=sum(r.get('segmentation_source') == 'cached_yolo26s_seg_1280' for r in tracks),
        fallback_boxes=sum(r['draw_detection_box_fallback'] for r in tracks+pending),
        suppressed_duplicates=duplicate_rows,
        rejected_outside=rejected))
    return result
