"""Output-layout signatures.

Adapters are selected by the *shape* of the model's output tensors, not by a
version string: YOLOv8 / v11 / v12 / v5u exports all share one layout, so a
version number is the wrong key. If no adapter (or more than one) claims a
signature we raise instead of guessing; pass ``layout=`` explicitly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

Shape = tuple[int, ...]  # dynamic dims are -1


class LayoutError(ValueError):
    """Output layout could not be identified unambiguously."""


@dataclass(frozen=True)
class LayoutSig:
    n_outputs: int
    out0_rank: int
    channels_first: bool  # (1, C, N) vs (1, N, C)
    last_dim: int | None  # size of the innermost dim when static
    out0_shape: Shape = field(default=(), compare=False)

    @property
    def n_boxes(self) -> int:
        """Number of candidate boxes (N) in output 0."""
        return self.out0_shape[-1] if self.channels_first else self.out0_shape[-2]

    @property
    def n_channels(self) -> int:
        """Per-box vector length (C) in output 0."""
        return self.out0_shape[-2] if self.channels_first else self.out0_shape[-1]


@dataclass(frozen=True)
class ModelIO:
    input_name: str
    input_shape: Shape
    output_names: tuple[str, ...]
    output_shapes: tuple[Shape, ...]
    metadata: dict[str, str]

    @property
    def names(self) -> dict[int, str] | None:
        """Class names if the exporter embedded them (ultralytics and yolov5 both do)."""
        raw = self.metadata.get("names")
        if not raw:
            return None
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
        if isinstance(val, list):
            val = dict(enumerate(val))
        return {int(k): str(v) for k, v in val.items()}


def _dims(type_proto) -> Shape:
    dims = type_proto.tensor_type.shape.dim
    return tuple(d.dim_value if d.HasField("dim_value") else -1 for d in dims)


def read_onnx_io(path: str | Path) -> ModelIO:
    import onnx  # lazy: only needed when reading .onnx files

    model = onnx.load(str(path), load_external_data=False)
    g = model.graph
    initializers = {i.name for i in g.initializer}
    inputs = [i for i in g.input if i.name not in initializers]
    if len(inputs) != 1:
        raise LayoutError(f"expected 1 image input, found {len(inputs)}: {[i.name for i in inputs]}")
    return ModelIO(
        input_name=inputs[0].name,
        input_shape=_dims(inputs[0].type),
        output_names=tuple(o.name for o in g.output),
        output_shapes=tuple(_dims(o.type) for o in g.output),
        metadata={p.key: p.value for p in model.metadata_props},
    )


def sig_from_shapes(output_shapes: list[Shape] | tuple[Shape, ...]) -> LayoutSig:
    if not output_shapes:
        raise LayoutError("model has no outputs")
    s0 = tuple(output_shapes[0])
    if len(s0) != 3:
        raise LayoutError(f"output 0 has rank {len(s0)}, expected (batch, a, b); shape={s0}")
    a, b = s0[1], s0[2]
    if a < 0 or b < 0:
        raise LayoutError(f"output 0 has dynamic box/channel dims {s0}; export with a fixed imgsz")
    # The per-box channel dim (4+nc, 5+nc, 6) is always far smaller than the box count.
    channels_first = a < b
    return LayoutSig(
        n_outputs=len(output_shapes),
        out0_rank=len(s0),
        channels_first=channels_first,
        last_dim=b,
        out0_shape=s0,
    )


def detect_layout(sig: LayoutSig) -> str:
    """Return the name of the single registered adapter that claims ``sig``."""
    from yolokit.adapters import REGISTRY

    hits = [name for name, cls in REGISTRY.items() if cls.matches(sig)]
    if len(hits) == 1:
        return hits[0]
    choices = ",".join(REGISTRY)
    if not hits:
        raise LayoutError(f"no adapter matches {sig}; pass --layout one of [{choices}]")
    raise LayoutError(f"ambiguous layout {sig}: {hits}; pass --layout explicitly")
