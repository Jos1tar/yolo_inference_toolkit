"""Backend registry. Imports are lazy so that a missing runtime only fails when used."""

from __future__ import annotations

import importlib

from yolokit.backends.base import Backend, BackendUnavailable

_BACKENDS: dict[str, str] = {
    "torch": "yolokit.backends.torch_ref:TorchRefBackend",
    "onnxrt": "yolokit.backends.onnxrt:OnnxRuntimeBackend",
    "openvino": "yolokit.backends.openvino:OpenVINOBackend",
    "tensorrt": "yolokit.backends.tensorrt:TensorRTBackend",
}

BACKEND_NAMES = tuple(_BACKENDS)


def get_backend(name: str) -> Backend:
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {list(_BACKENDS)}") from None
    mod_name, cls_name = target.split(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise BackendUnavailable(f"backend {name!r} needs an extra dependency: {e}") from e
    return getattr(mod, cls_name)()


__all__ = ["BACKEND_NAMES", "Backend", "BackendUnavailable", "get_backend"]
