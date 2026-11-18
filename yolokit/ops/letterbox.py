"""Aspect-preserving resize + constant pad, with the info needed to undo it."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Meta:
    orig_hw: tuple[int, int]
    input_hw: tuple[int, int]
    ratio: float
    pad: tuple[int, int]  # (left, top) pixels actually added


def letterbox(
    img: np.ndarray,
    new_hw: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, Meta]:
    h0, w0 = img.shape[:2]
    H, W = new_hw
    r = min(H / h0, W / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (W - new_w) / 2, (H - new_h) / 2
    if (new_w, new_h) != (w0, h0):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    assert img.shape[:2] == (H, W), (img.shape, new_hw)
    return img, Meta(orig_hw=(h0, w0), input_hw=(H, W), ratio=r, pad=(left, top))


def scale_boxes(xyxy: np.ndarray, meta: Meta) -> np.ndarray:
    """Map xyxy boxes from letterboxed input coords back to the original image, clipped."""
    out = xyxy.astype(np.float32, copy=True)
    left, top = meta.pad
    out[:, [0, 2]] -= left
    out[:, [1, 3]] -= top
    out /= meta.ratio
    h0, w0 = meta.orig_hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w0)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h0)
    return out
