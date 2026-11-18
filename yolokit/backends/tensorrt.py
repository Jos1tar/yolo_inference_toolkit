"""TensorRT backend — M2.

Plan: build an FP16 engine from the ONNX file via ``trtexec`` (wrapped in tools/),
deserialize with the TensorRT Python API, run on CUDA. Blackwell GPUs (RTX 50xx)
need TensorRT >= 10.8 and CUDA >= 12.8.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TensorRTBackend:
    name = "tensorrt"

    def load(self, model_path: Path, device: str = "cuda", **_) -> None:
        raise NotImplementedError("TensorRT backend is planned for M2")

    def input_shape(self) -> tuple[int, int, int, int]:
        raise NotImplementedError

    def infer(self, x: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError
