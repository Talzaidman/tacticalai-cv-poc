"""Experimental automatic roles and anonymous uniform groups; no label inputs.

CLIP cosine similarities and jersey-color distances are evidence scores, not
calibrated probabilities. Team A/B are local to a clip, with two teams assumed.
"""
from collections import defaultdict

import cv2
import numpy as np


def identity_key(row):
    # Never propagate a role across a raw track whose physical identity is unknown.
    return str(row['player_id']) if row.get('player_id') is not None else None


def select_samples(frames, config, width, height):
    """Time-spaced, approximately uniform samples of untruncated observations."""
    candidates = defaultdict(list)
    for frame in frames:
        for index, row in enumerate(frame['tracks']):
            key = identity_key(row)
            if key is None:
                continue
            x1, y1, x2, y2 = row['bbox_xyxy']
            if (x2-x1 < config['minimum_crop_width'] or y2-y1 < config['minimum_crop_height']
                    or x1 < 1 or y1 < 1 or x2 > width-1 or y2 > height-1):
                continue
            candidates[key].append((frame['frame_idx'], index))
    selected = {}
    for key, values in candidates.items():
        spaced = []
        for item in values:
            if not spaced or item[0]-spaced[-1][0] >= config['sample_spacing_frames']:
                spaced.append(item)
        indices = np.unique(np.rint(np.linspace(0, len(spaced)-1,
                                 min(len(spaced), config['samples_per_identity']))).astype(int))
        selected[key] = [spaced[i] for i in indices]
    return selected


def torso_color(image, row, config):
    """Masked central torso median in true CIELAB units; missing masks abstain."""
    if row.get('silhouette_xy') is None:
        return None
    x1, y1, x2, y2 = row['bbox_xyxy']
    a, b, c, d = config['torso_fraction_xyxy']
    left, top = max(0, int(x1+a*(x2-x1))), max(0, int(y1+b*(y2-y1)))
    right, bottom = min(image.shape[1], int(x1+c*(x2-x1))), min(image.shape[0], int(y1+d*(y2-y1)))
    if right <= left or bottom <= top:
        return None
    mask = np.zeros((bottom-top, right-left), np.uint8)
    polygon = np.rint(np.asarray(row['silhouette_xy'])-[left, top]).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    pixels = image[top:bottom, left:right][mask.astype(bool)]
    if len(pixels) < config['minimum_torso_pixels']:
        return None
    lab = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0).tolist()


def decide_role(samples, config):
    labels = list(config['role_prompts'])
    result = dict(role=None, role_reason='insufficient_samples', role_samples=len(samples),
                  role_scores=None, role_margin=None, role_agreement=None)
    if not samples:
        return result
    scores = np.asarray([[s['role_scores'][k] for k in labels] for s in samples])
    median = np.median(scores, axis=0)
    order = np.argsort(median)[::-1]
    winner = int(order[0])
    margin = float(median[winner]-median[order[1]])
    agreement = float(np.mean(np.argmax(scores, axis=1) == winner))
    result.update(role_scores=dict(zip(labels, median.tolist())), role_margin=margin,
                  role_agreement=agreement)
    if len(samples) < config['minimum_samples']:
        return result
    if median[winner] < config['minimum_role_similarity']:
        result['role_reason'] = 'low_similarity'
    elif margin < config['minimum_role_margin']:
        result['role_reason'] = 'ambiguous_role'
    elif agreement < config['minimum_role_agreement']:
        result['role_reason'] = 'inconsistent_samples'
    else:
        result.update(role=labels[winner], role_reason='automatic_clip_temporal_vote')
    return result


def fit_teams(vectors, config):
    """Deterministic two-means; reject unsupported or indistinguishable groups."""
    x = np.asarray(vectors, dtype=float)
    minimum = config['minimum_team_members']
    if len(x) < 2*minimum:
        return None
    distances = ((x[:, None]-x[None, :])**2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    centers = x[[i, j]].copy()
    for _ in range(50):
        labels = ((x[:, None]-centers)**2).sum(axis=2).argmin(axis=1)
        if any(np.sum(labels == k) < minimum for k in (0, 1)):
            return None
        updated = np.stack([np.mean(x[labels == k], axis=0) for k in (0, 1)])
        if np.max(np.abs(updated-centers)) < 1e-6:
            centers = updated
            break
        centers = updated
    if np.linalg.norm(centers[0]-centers[1]) < config['minimum_team_separation_lab']:
        return None
    # Labels depend on lightness, not hard-coded jersey colors or person IDs.
    return centers[np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))].tolist()


def decide_team(samples, role, centers, config):
    colors = [s['torso_lab'] for s in samples if s.get('torso_lab') is not None]
    result = dict(team_id=None, team_reason='role_not_player', team_samples=len(colors),
                  team_distances_lab=None, team_margin_lab=None, team_agreement=None,
                  torso_lab=np.median(colors, axis=0).tolist() if colors else None)
    if role != 'player':
        return result
    if len(colors) < config['minimum_samples']:
        result['team_reason'] = 'insufficient_color_samples'
        return result
    if centers is None:
        result['team_reason'] = 'unsupported_two_team_model'
        return result
    distances = np.linalg.norm(np.asarray(colors)[:, None]-np.asarray(centers), axis=2)
    medians = np.median(distances, axis=0)
    winner = int(np.argmin(medians))
    margin = float(medians[1-winner]-medians[winner])
    agreement = float(np.mean(np.argmin(distances, axis=1) == winner))
    result.update(team_distances_lab=medians.tolist(), team_margin_lab=margin,
                  team_agreement=agreement)
    if medians[winner] > config['maximum_team_distance_lab']:
        result['team_reason'] = 'far_from_uniform_groups'
    elif margin < config['minimum_team_margin_lab']:
        result['team_reason'] = 'ambiguous_uniform'
    elif agreement < config['minimum_team_agreement']:
        result['team_reason'] = 'inconsistent_uniform_samples'
    else:
        result.update(team_id=['A', 'B'][winner], team_reason='automatic_torso_color_cluster')
    return result


def classify_identities(samples, config):
    decisions = {key: decide_role(values, config) for key, values in samples.items()}
    eligible = []
    for key, decision in decisions.items():
        colors = [s['torso_lab'] for s in samples[key] if s.get('torso_lab') is not None]
        if decision['role'] == 'player' and len(colors) >= config['minimum_samples']:
            eligible.append(np.median(colors, axis=0))
    centers = fit_teams(eligible, config)
    for key, decision in decisions.items():
        decision.update(decide_team(samples[key], decision['role'], centers, config))
    return decisions, centers


class ClipRoles:
    """Generic pretrained CLIP, separate from the fine-tuned identity encoder."""
    def __init__(self, config, cache, device='cpu', threads=4):
        import open_clip
        import torch
        from .export import sha256

        torch.set_num_threads(threads)
        self.device = device
        self.crop_view = config.get('crop_view', 'center_crop')
        checkpoint = open_clip.download_pretrained(
            open_clip.get_pretrained_cfg(config['model'], config['pretrained']), cache_dir=str(cache))
        self.checkpoint_sha256 = sha256(checkpoint)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config['model'], pretrained=config['pretrained'], device=device, cache_dir=str(cache))
        self.model.eval()
        self.labels = list(config['role_prompts'])
        with torch.inference_mode():
            prototypes = []
            for label in self.labels:
                encoded = self.model.encode_text(open_clip.tokenize(config['role_prompts'][label]).to(device)).float()
                encoded = torch.nn.functional.normalize(encoded, dim=1).mean(dim=0)
                prototypes.append(torch.nn.functional.normalize(encoded, dim=0))
            self.prototypes = torch.stack(prototypes)

    def score(self, crops):
        import torch
        from PIL import Image
        if self.crop_view == 'aspect_pad_gray':
            crops = [square_pad(c) for c in crops]
        batch = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops])
        with torch.inference_mode():
            features = torch.nn.functional.normalize(self.model.encode_image(batch.to(self.device)).float(), dim=1)
            scores = (features @ self.prototypes.T).cpu().numpy()
        return [dict(zip(self.labels, row.tolist())) for row in scores]


def square_pad(crop):
    """Preserve head, jersey and legs instead of center-cropping a tall person."""
    h, w = crop.shape[:2]
    side = max(h, w)
    image = np.full((side, side, 3), 127, np.uint8)
    top, left = (side-h)//2, (side-w)//2
    image[top:top+h, left:left+w] = crop
    return image
