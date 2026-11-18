from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


class OnnxRuntimeBackend:
    name = "onnxrt"

    def load(self, model_path: Path, device: str = "cpu", threads: int | None = None, **_) -> None:
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_path), sess_options=so, providers=_PROVIDERS[device])
        inp = self.sess.get_inputs()[0]
        self._input_name = inp.name
        self._input_shape = tuple(d if isinstance(d, int) else -1 for d in inp.shape)
        self.device = device

    def input_shape(self) -> tuple[int, int, int, int]:
        return self._input_shape  # type: ignore[return-value]

    def infer(self, x: np.ndarray) -> list[np.ndarray]:
        return self.sess.run(None, {self._input_name: x})
