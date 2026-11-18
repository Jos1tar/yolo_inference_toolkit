"""Layout A ``objectness``: (1, N, 5+nc) = [cx, cy, w, h, obj, cls_0 .. cls_nc-1].

Produced by the original yolov5 ``export.py``. Score is obj * cls, as in
yolov5's ``non_max_suppression``: first gate on obj, then on obj * cls.
"""

from __future__ import annotations

import numpy as np

from yolokit.adapters import register
from yolokit.adapters.base import Adapter, xywh2xyxy
from yolokit.layout import LayoutSig

# below this many rows a (1, N, 6) output is treated as an end-to-end head (max_det rows)
E2E_MAX_ROWS = 1000


@register
class ObjectnessAdapter(Adapter):
    name = "objectness"

    @classmethod
    def matches(cls, sig: LayoutSig) -> bool:
        return (
            sig.n_outputs == 1
            and sig.out0_rank == 3
            and not sig.channels_first
            and sig.n_channels >= 6
            and sig.n_boxes > E2E_MAX_ROWS
        )

    @property
    def num_classes(self) -> int:
        return self.sig.n_channels - 5

    def decode(self, outputs: list[np.ndarray], conf_thr: float) -> np.ndarray:
        p = outputs[0][0]  # (N, 5+nc), batch 1
        p = p[p[:, 4] > conf_thr]
        if p.shape[0] == 0:
            return np.zeros((0, 6), np.float32)
        scores = p[:, 5:] * p[:, 4:5]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        m = conf > conf_thr
        boxes = xywh2xyxy(p[m, :4])
        return np.concatenate([boxes, conf[m, None], cls[m, None].astype(np.float32)], axis=1)
