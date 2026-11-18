"""Backend contract. Backends run the graph and return raw outputs; they never decode."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Backend(Protocol):
    name: str

    def load(self, model_path: Path, device: str = "cpu", **opts) -> None: ...

    def input_shape(self) -> tuple[int, int, int, int]: ...  # NCHW

    def infer(self, x: np.ndarray) -> list[np.ndarray]: ...  # every output, unmodified


class BackendUnavailable(RuntimeError):
    """The backend's runtime package is not installed."""
