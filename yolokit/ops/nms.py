"""NMS: torchvision when available, otherwise a vectorised NumPy fallback.

Within one parity run every backend goes through the same NMS, so the choice
does not affect parity numbers.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on environment
    import torch
    import torchvision

    HAS_TORCHVISION = True
except ImportError:  # pragma: no cover
    HAS_TORCHVISION = False


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, (N,4) x (M,4) xyxy -> (N,M)."""
    a = a[:, None, :]
    b = b[None, :, :]
    iw = (np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])).clip(0)
    ih = (np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])).clip(0)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, use_torch: bool | None = None) -> np.ndarray:
    """Indices of kept boxes, sorted by descending score."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if use_torch is None:
        use_torch = HAS_TORCHVISION
    if use_torch:
        keep = torchvision.ops.nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), iou_thr)
        return keep.numpy()
    return _nms_numpy(boxes, scores, iou_thr)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: float,
    use_torch: bool | None = None,
) -> np.ndarray:
    """Class-aware NMS via the coordinate-offset trick (one NMS call for all classes)."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    offset = classes.astype(np.float32)[:, None] * (float(boxes.max()) + 1.0)
    return nms(boxes + offset, scores, iou_thr, use_torch=use_torch)
