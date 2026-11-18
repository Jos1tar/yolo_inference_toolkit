"""Adapter contract: preprocess -> (backend) -> decode -> postprocess.

All three M1 layouts share the same preprocessing (letterbox, BGR->RGB, /255,
HWC->NCHW). They differ only in how the raw output tensor is decoded into
candidate boxes, and whether NMS is needed afterwards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from yolokit.layout import LayoutSig
from yolokit.ops.letterbox import Meta, letterbox, scale_boxes
from yolokit.ops.nms import batched_nms


@dataclass
class Detections:
    xyxy: np.ndarray  # (M, 4) float32, original-image pixels
    conf: np.ndarray  # (M,) float32
    cls: np.ndarray  # (M,) int64

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))

    def to_dicts(self, names: dict[int, str] | None = None) -> list[dict]:
        out = []
        for box, c, k in zip(self.xyxy, self.conf, self.cls, strict=True):
            d = {"xyxy": [round(float(v), 2) for v in box], "conf": round(float(c), 4), "cls": int(k)}
            if names:
                d["name"] = names.get(int(k), str(int(k)))
            out.append(d)
        return out


def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    half_w, half_h = b[:, 2] / 2, b[:, 3] / 2
    out[:, 0] = b[:, 0] - half_w
    out[:, 1] = b[:, 1] - half_h
    out[:, 2] = b[:, 0] + half_w
    out[:, 3] = b[:, 1] + half_h
    return out


class Adapter(ABC):
    name: ClassVar[str]
    needs_nms: ClassVar[bool] = True

    def __init__(self, sig: LayoutSig, max_det: int = 300) -> None:
        self.sig = sig
        self.max_det = max_det

    # -- layout matching -------------------------------------------------
    @classmethod
    @abstractmethod
    def matches(cls, sig: LayoutSig) -> bool: ...

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    # -- pipeline stages ---------------------------------------------------
    def preprocess(self, img_bgr: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, Meta]:
        padded, meta = letterbox(img_bgr, size)
        x = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0
        return x, meta

    @abstractmethod
    def decode(self, outputs: list[np.ndarray], conf_thr: float) -> np.ndarray:
        """Raw outputs -> (K, 6) [x1, y1, x2, y2, conf, cls] in letterbox coords, conf-filtered."""

    def postprocess(self, cand: np.ndarray, meta: Meta, iou_thr: float) -> Detections:
        if cand.shape[0] == 0:
            return Detections.empty()
        boxes, conf, cls = cand[:, :4], cand[:, 4], cand[:, 5].astype(np.int64)
        if self.needs_nms:
            keep = batched_nms(boxes, conf, cls, iou_thr)
        else:
            keep = np.argsort(-conf, kind="stable")
        keep = keep[: self.max_det]
        return Detections(
            xyxy=scale_boxes(boxes[keep], meta),
            conf=conf[keep].astype(np.float32),
            cls=cls[keep],
        )
