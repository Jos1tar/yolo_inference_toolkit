from __future__ import annotations

import cv2
import numpy as np

from yolokit.adapters.base import Detections


def _color(k: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(k)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def draw(img_bgr: np.ndarray, dets: Detections, names: dict[int, str]) -> np.ndarray:
    out = img_bgr.copy()
    for (x1, y1, x2, y2), c, k in zip(dets.xyxy.astype(int), dets.conf, dets.cls, strict=True):
        col = _color(int(k))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"{names.get(int(k), int(k))} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1, th + 4)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text), col, -1)
        cv2.putText(out, label, (x1 + 2, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
