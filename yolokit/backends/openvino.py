"""OpenVINO backend.

Given ``m.onnx`` it uses a sibling ``m.xml`` IR if present (see ``yk export --to openvino``),
otherwise it reads the ONNX file directly.

Precision is pinned to f32 by default: on CPUs with AMX/AVX512-BF16 OpenVINO
otherwise silently runs in bf16, which shows up as a parity failure rather
than as a backend difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov


class OpenVINOBackend:
    name = "openvino"

    def load(self, model_path: Path, device: str = "cpu", precision: str = "f32", **_) -> None:
        model_path = Path(model_path)
        xml = model_path.with_suffix(".xml")
        src = xml if model_path.suffix == ".onnx" and xml.exists() else model_path
        core = ov.Core()
        model = core.read_model(str(src))
        config = {"INFERENCE_PRECISION_HINT": precision} if precision else {}
        self.compiled = core.compile_model(model, device.upper(), config)
        self._outputs = list(self.compiled.outputs)
        ps = self.compiled.input(0).get_partial_shape()
        self._input_shape = tuple(d.get_length() if d.is_static else -1 for d in ps)
        self.source = src
        self.device = device

    def input_shape(self) -> tuple[int, int, int, int]:
        return self._input_shape  # type: ignore[return-value]

    def infer(self, x: np.ndarray) -> list[np.ndarray]:
        res = self.compiled(x)
        return [np.asarray(res[o]) for o in self._outputs]
