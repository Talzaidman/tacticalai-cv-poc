"""CLIP-ReIdent inference adapted from the MIT-licensed 2022 winning code.

Reference: DeepSportradar/2022-winners-player-reidentification-challenge,
commit e93af99ac3f7924a4b06bae4753b2f69408feea1, model.py / transforms.py.
See docs/CLIP_REIDENT.md and docs/licenses/CLIP_REIDENT_MIT.txt.
"""
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
UPSTREAM_COMMIT = "e93af99ac3f7924a4b06bae4753b2f69408feea1"


def preprocess(crop_bgr, size=224):
    """Match upstream RectResize + Normalize + ToTensorV2, without Albumentations."""
    if crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 1:
        raise ValueError("Empty ReID crop")
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR_EXACT)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    padded[y:y + nh, x:x + nw] = resized
    # Albumentations Normalize multiplies by the reciprocal denominator.
    image = (padded.astype(np.float32) - MEAN * 255) * (1 / (STD * 255))
    return np.ascontiguousarray(image.transpose(2, 0, 1))


class ClipReIdent:
    """Require the fine-tuned checkpoint; never fall back to generic CLIP weights."""

    def __init__(self, checkpoint, device="cpu", batch_size=4):
        import torch
        import open_clip
        from .export import sha256

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device, self.batch_size = torch.device(device), batch_size
        checkpoint = Path(checkpoint).resolve()
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = sha256(checkpoint)
        # OpenAI initialization in the original factory selects QuickGELU.
        # All retained parameters are loaded strictly, so no base-weight download
        # or pretrained fallback is needed.
        self.model = open_clip.create_model("ViT-L-14", pretrained="", force_quick_gelu=True)
        for name in ("transformer", "token_embedding", "ln_final", "positional_embedding", "text_projection"):
            delattr(self.model, name)
        self.model.visual.proj = None
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(k.startswith("model.") for k in state):
            raise ValueError("Expected the original CLIP-ReIdent model.* state dictionary")
        self.model.load_state_dict({k[6:]: v for k, v in state.items()}, strict=True)
        self.model.eval().to(self.device)

    def encode(self, crops):
        import torch
        if not crops:
            return np.empty((0, 1024), dtype=np.float32)
        output = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = torch.from_numpy(np.stack([preprocess(c) for c in crops[start:start + self.batch_size]]))
                features = self.model.encode_image(batch.to(self.device))
                features = torch.nn.functional.normalize(features.float(), dim=1)
                output.append(features.cpu().numpy())
        features = np.concatenate(output)
        if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) < 0.99):
            raise RuntimeError("Invalid CLIP-ReIdent embeddings")
        return features
