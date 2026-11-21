"""Conservative geometry-normalized transfer of retrieved lesion masks.

This module avoids direct full-image resize of a retrieved binary mask. A prior is first
cropped to its own lesion box and then affinely mapped to the query box. Outside the
query box, the logit prior is neutral (zero), so it cannot force global background.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_MIN_IOU = 0.15
DEFAULT_MIN_AREA_RATIO = 0.20
DEFAULT_MAX_AREA_RATIO = 5.00
DEFAULT_MAX_CENTER_DISTANCE = 0.45
DEFAULT_LOGIT_SCALE = 2.0


def clamp_box(box, width: int, height: int) -> np.ndarray:
    """Clamp an xyxy box to image bounds and return ordered float coordinates."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    x1, x2 = sorted((float(np.clip(x1, 0, width - 1)), float(np.clip(x2, 0, width - 1))))
    y1, y2 = sorted((float(np.clip(y1, 0, height - 1)), float(np.clip(y2, 0, height - 1))))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the tight xyxy box of a boolean mask, or None for an empty mask."""
    mask = np.asarray(mask, dtype=bool)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_iou(box1, box2) -> float:
    """Compute IoU for xyxy boxes with inclusive pixel coordinates."""
    if box1 is None or box2 is None:
        return 0.0
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)
    area1 = max(0.0, float(box1[2]) - float(box1[0]) + 1.0) * max(0.0, float(box1[3]) - float(box1[1]) + 1.0)
    area2 = max(0.0, float(box2[2]) - float(box2[0]) + 1.0) * max(0.0, float(box2[3]) - float(box2[1]) + 1.0)
    return inter / max(area1 + area2 - inter, 1e-8)


def _box_area(box) -> float:
    return max(0.0, float(box[2]) - float(box[0]) + 1.0) * max(0.0, float(box[3]) - float(box[1]) + 1.0)


def _center_distance(box1, box2, width: int, height: int) -> float:
    c1 = np.array([(box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0], dtype=np.float32)
    c2 = np.array([(box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0], dtype=np.float32)
    return float(np.linalg.norm((c1 - c2) / np.array([max(width, 1), max(height, 1)], dtype=np.float32)))


def load_resized_prior(mask_path: str, image_size: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a prior, resize it only for geometry checks, and return mask plus its box."""
    if not mask_path or mask_path == "none" or not os.path.exists(mask_path):
        return None, None
    width, height = image_size
    try:
        prior = Image.open(mask_path).convert("L").resize((width, height), resample=Image.NEAREST)
        prior_mask = np.asarray(prior) > 127
    except (OSError, ValueError):
        return None, None
    return prior_mask, bbox_from_mask(prior_mask)


def is_prior_geometry_compatible(
    mask_path: str,
    query_box,
    image_size: Tuple[int, int],
    min_iou: float = DEFAULT_MIN_IOU,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
    max_center_distance: float = DEFAULT_MAX_CENTER_DISTANCE,
) -> bool:
    """Gate transfer using coarse normalized position, scale, and overlap checks."""
    width, height = image_size
    _, prior_box = load_resized_prior(mask_path, image_size)
    if prior_box is None:
        return False
    query_box = clamp_box(query_box, width, height)
    area_ratio = _box_area(prior_box) / max(_box_area(query_box), 1e-8)
    return (
        box_iou(query_box, prior_box) >= min_iou
        and min_area_ratio <= area_ratio <= max_area_ratio
        and _center_distance(query_box, prior_box, width, height) <= max_center_distance
    )


def prepare_aligned_prior(
    mask_path: str,
    query_box,
    image_size: Tuple[int, int],
    device: str,
    target_size: int = 256,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """Affinely map a prior lesion crop into the query box and build a soft SAM2 prior.

    The returned binary map is in query-image coordinates for candidate consistency
    scoring. The returned tensor has shape (1, 1, target_size, target_size). The
    background outside the query box remains neutral rather than being assigned a
    high-magnitude negative logit.
    """
    if not mask_path or mask_path == "none" or not os.path.exists(mask_path):
        return None, None
    width, height = image_size
    try:
        raw_prior = np.asarray(Image.open(mask_path).convert("L")) > 127
    except (OSError, ValueError):
        return None, None
    source_box = bbox_from_mask(raw_prior)
    if source_box is None:
        return None, None

    query_box = clamp_box(query_box, width, height).astype(int)
    qx1, qy1, qx2, qy2 = query_box
    if qx2 <= qx1 or qy2 <= qy1:
        return None, None
    sx1, sy1, sx2, sy2 = source_box.astype(int)
    source_crop = raw_prior[sy1:sy2 + 1, sx1:sx2 + 1].astype(np.uint8)
    target_width, target_height = qx2 - qx1 + 1, qy2 - qy1 + 1
    warped_crop = cv2.resize(source_crop, (target_width, target_height), interpolation=cv2.INTER_NEAREST) > 0

    aligned_mask = np.zeros((height, width), dtype=bool)
    aligned_mask[qy1:qy2 + 1, qx1:qx2 + 1] = warped_crop

    # Build a soft local logit field. Only the query box receives prior evidence;
    # outside it SAM2 receives zero (unknown), preventing global background forcing.
    blur_kernel = max(3, int(round(min(target_width, target_height) * 0.06)) | 1)
    softened = cv2.GaussianBlur(warped_crop.astype(np.float32), (blur_kernel, blur_kernel), 0)
    local_logits = float(logit_scale) * (2.0 * softened - 1.0)
    logit_canvas = np.zeros((height, width), dtype=np.float32)
    logit_canvas[qy1:qy2 + 1, qx1:qx2 + 1] = local_logits
    logits_256 = cv2.resize(logit_canvas, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    logits_tensor = torch.from_numpy(logits_256).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    return logits_tensor, aligned_mask
