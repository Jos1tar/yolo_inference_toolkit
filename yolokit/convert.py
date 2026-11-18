"""ONNX -> derived formats. No ultralytics import here (that lives in tools/)."""

from __future__ import annotations

from pathlib import Path


def onnx_to_openvino(onnx_path: str | Path, fp16: bool = False) -> Path:
    """Write ``<stem>.xml/.bin`` next to the ONNX file.

    Default is FP32 so the IR is directly comparable in parity; FP16 weight
    compression is opt-in. No mean/scale or channel reversal is baked in —
    the adapter already does BGR->RGB and /255.
    """
    import openvino as ov

    onnx_path = Path(onnx_path)
    xml = onnx_path.with_suffix(".xml")
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(xml), compress_to_fp16=fp16)
    return xml
