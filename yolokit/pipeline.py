"""Glue: ONNX file -> layout signature -> adapter, plus a backend to run it."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from yolokit.adapters import get_adapter
from yolokit.adapters.base import Adapter, Detections
from yolokit.backends import Backend, get_backend
from yolokit.layout import ModelIO, detect_layout, read_onnx_io, sig_from_shapes
from yolokit.names import COCO80

DEFAULT_IMGSZ = 640


@dataclass
class StageTimes:
    pre_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.pre_ms + self.infer_ms + self.post_ms


@dataclass
class Pipeline:
    """``Pipeline("models/yolov8n.onnx", backend="openvino")(img_bgr) -> Detections``.

    The ONNX file is always the source of truth for the layout, even when the
    backend actually runs something else (torch: sibling ``.pt``; openvino: sibling ``.xml``).
    """

    model: str | Path
    backend: str = "onnxrt"
    device: str = "cpu"
    layout: str | None = None
    conf: float = 0.25
    iou: float = 0.45
    backend_opts: dict = field(default_factory=dict)

    io: ModelIO = field(init=False)
    adapter: Adapter = field(init=False)
    runner: Backend = field(init=False)
    imgsz: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.model = Path(self.model)
        self.io = read_onnx_io(self.model)
        sig = sig_from_shapes(self.io.output_shapes)
        self.layout = self.layout or detect_layout(sig)
        self.adapter = get_adapter(self.layout)(sig)

        h, w = self.io.input_shape[2:4]
        self.imgsz = (h, w) if h > 0 and w > 0 else (DEFAULT_IMGSZ, DEFAULT_IMGSZ)

        self.runner = get_backend(self.backend)
        self.runner.load(self.model, self.device, layout=self.layout, imgsz=self.imgsz[0], **self.backend_opts)

    @property
    def names(self) -> dict[int, str]:
        return self.io.names or COCO80

    def __call__(self, img_bgr: np.ndarray) -> Detections:
        return self.run(img_bgr)[0]

    def run(self, img_bgr: np.ndarray) -> tuple[Detections, StageTimes]:
        t = StageTimes()
        t0 = time.perf_counter()
        x, meta = self.adapter.preprocess(img_bgr, self.imgsz)
        t1 = time.perf_counter()
        outputs = self.runner.infer(x)
        t2 = time.perf_counter()
        cand = self.adapter.decode(outputs, self.conf)
        dets = self.adapter.postprocess(cand, meta, self.iou)
        t3 = time.perf_counter()
        t.pre_ms, t.infer_ms, t.post_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
        return dets, t
