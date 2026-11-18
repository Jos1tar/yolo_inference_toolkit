"""Layout C ``e2e``: (1, max_det, 6) = [x1, y1, x2, y2, conf, cls], NMS-free.

Produced by ultralytics exports of YOLOv10 / YOLO26 (one-to-one head).
M1 stretch goal: decode is trivial, but torch-reference parity is not wired up yet.
"""

from __future__ import annotations

import numpy as np

from yolokit.adapters import register
from yolokit.adapters.base import Adapter
from yolokit.adapters.objectness import E2E_MAX_ROWS
from yolokit.layout import LayoutSig


@register
class E2EAdapter(Adapter):
    name = "e2e"
    needs_nms = False

    @classmethod
    def matches(cls, sig: LayoutSig) -> bool:
        return (
            sig.n_outputs == 1
            and sig.out0_rank == 3
            and not sig.channels_first
            and sig.n_channels == 6
            and sig.n_boxes <= E2E_MAX_ROWS
        )

    @property
    def num_classes(self) -> int:
        return -1  # not encoded in the output shape; use model metadata

    def decode(self, outputs: list[np.ndarray], conf_thr: float) -> np.ndarray:
        p = outputs[0][0].astype(np.float32)
        return p[p[:, 4] > conf_thr]
