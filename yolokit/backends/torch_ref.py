"""PyTorch reference backend (parity baseline, dev/test only).

Runs the original ``.pt`` forward pass and returns the raw head output in the
same layout as the ONNX export, so it goes through exactly the same adapter.
Parity therefore measures backend + runtime differences, not decode differences.

Loading a ``.pt`` checkpoint means unpickling the training repo's classes, so
this backend needs the model's source code at load time:

* layout ``objectness`` (yolov5): a local clone of ultralytics/yolov5
  (``tools/fetch_models.py`` puts one at ``models/_yolov5``; override with ``YOLOKIT_YOLOV5_REPO``)
* layouts ``dfl`` / ``e2e``: the ``ultralytics`` package

That is why it is an optional extra and is never imported by the other backends.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

DEFAULT_V5_REPO = Path("models/_yolov5")


def _first_tensor(y):
    """Heads return (pred, feature_maps) tuples / lists; the decoded pred is always first."""
    while isinstance(y, (list, tuple)):
        y = y[0]
    if isinstance(y, dict):  # some e2e heads return {"one2one": ..., ...}
        y = y.get("one2one", next(iter(y.values())))
        return _first_tensor(y)
    return y


class TorchRefBackend:
    name = "torch"

    def load(self, model_path: Path, device: str = "cpu", layout: str | None = None, imgsz: int = 640, **_) -> None:
        model_path = Path(model_path)
        pt = model_path.with_suffix(".pt")
        if not pt.exists():
            raise FileNotFoundError(f"torch backend needs the original checkpoint next to the ONNX file: {pt}")
        self.device = torch.device(device)
        if layout == "objectness":
            self.model = self._load_yolov5(pt)
        else:
            self.model = self._load_ultralytics(pt)
        self.model.to(self.device).float().eval()
        self._input_shape = (1, 3, imgsz, imgsz)

    @staticmethod
    def _load_yolov5(pt: Path) -> torch.nn.Module:
        # default: the clone tools/fetch_models.py puts next to the checkpoints (models/_yolov5)
        repo = Path(os.environ.get("YOLOKIT_YOLOV5_REPO", pt.resolve().parent / DEFAULT_V5_REPO.name)).resolve()
        if not (repo / "hubconf.py").exists():
            raise FileNotFoundError(
                f"yolov5 repo not found at {repo}; run tools/fetch_models.py or set YOLOKIT_YOLOV5_REPO"
            )
        # yolov5 checkpoints are full pickles; torch>=2.6 defaults to weights_only=True
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        m = torch.hub.load(str(repo), "custom", path=str(pt.resolve()), source="local", autoshape=False, _verbose=False)
        # DetectMultiBackend -> underlying DetectionModel, so forward returns (pred, feats)
        return getattr(m, "model", m)

    @staticmethod
    def _load_ultralytics(pt: Path) -> torch.nn.Module:
        from ultralytics import YOLO  # AGPL-3.0: only ever imported by this dev-only backend

        return YOLO(str(pt)).model.fuse(verbose=False)

    def input_shape(self) -> tuple[int, int, int, int]:
        return self._input_shape

    @torch.inference_mode()
    def infer(self, x: np.ndarray) -> list[np.ndarray]:
        y = self.model(torch.from_numpy(x).to(self.device))
        return [_first_tensor(y).float().cpu().numpy()]
