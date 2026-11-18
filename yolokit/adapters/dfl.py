"""Layout B ``dfl``: (1, 4+nc, N) = [cx, cy, w, h, cls_0 .. cls_nc-1], channels first.

Produced by ultralytics exports of v8 / v11 / v12 / v5u. The DFL box
distribution is already integrated inside the graph, so the output is plain
xywh; there is no objectness term.
"""

from __future__ import annotations

import numpy as np

from yolokit.adapters import register
from yolokit.adapters.base import Adapter, xywh2xyxy
from yolokit.layout import LayoutSig


@register
class DFLAdapter(Adapter):
    name = "dfl"

    @classmethod
    def matches(cls, sig: LayoutSig) -> bool:
        return sig.n_outputs == 1 and sig.out0_rank == 3 and sig.channels_first and sig.n_channels >= 5

    @property
    def num_classes(self) -> int:
        return self.sig.n_channels - 4

    def decode(self, outputs: list[np.ndarray], conf_thr: float) -> np.ndarray:
        p = outputs[0][0].T  # (N, 4+nc), batch 1
        scores = p[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        m = conf > conf_thr
        boxes = xywh2xyxy(p[m, :4])
        return np.concatenate([boxes, conf[m, None], cls[m, None].astype(np.float32)], axis=1)
