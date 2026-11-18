"""Multi-backend inference toolkit for YOLO-family detectors.

Runtime depends only on numpy / opencv / onnx plus whichever backend you use
(onnxruntime, openvino, torch). Model download/export lives in ``tools/``.
"""

from yolokit.adapters.base import Detections
from yolokit.layout import LayoutError, LayoutSig, detect_layout, read_onnx_io, sig_from_shapes
from yolokit.pipeline import Pipeline

__all__ = [
    "Detections",
    "LayoutError",
    "LayoutSig",
    "Pipeline",
    "detect_layout",
    "read_onnx_io",
    "sig_from_shapes",
]
__version__ = "0.1.0.dev0"
