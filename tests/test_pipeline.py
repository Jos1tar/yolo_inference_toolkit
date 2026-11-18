"""End-to-end through real ONNX files, without downloading any YOLO weights.

A tiny graph emits a fixed raw tensor in a given layout (plus 0 * sum(input),
so the image input is actually consumed). This exercises read_onnx_io ->
layout detection -> adapter -> backend -> CLI on every CI run.
"""

import importlib.util

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from tests.test_adapters import _dfl_output, _e2e_output, _objectness_output  # noqa: E402
from yolokit.layout import read_onnx_io  # noqa: E402

BACKENDS = [b for b, mod in (("onnxrt", "onnxruntime"), ("openvino", "openvino")) if importlib.util.find_spec(mod)]


def make_onnx(path, raw: np.ndarray, names: dict | None = None):
    const = numpy_helper.from_array(raw, "raw")
    zero = numpy_helper.from_array(np.zeros((), np.float32), "zero")
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["raw", "z"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(raw.shape))],
        initializer=[const, zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    if names:
        helper.set_model_props(model, {"names": str(names)})
    onnx.save(model, str(path))
    return path


@pytest.fixture(params=["dfl", "objectness", "e2e"])
def synthetic_model(request, tmp_path):
    raw = {"dfl": _dfl_output, "objectness": _objectness_output, "e2e": _e2e_output}[request.param]()
    return request.param, make_onnx(tmp_path / f"{request.param}.onnx", raw, {0: "person", 2: "car"})


def test_read_onnx_io(synthetic_model):
    _, path = synthetic_model
    io = read_onnx_io(path)
    assert io.input_name == "images" and io.input_shape == (1, 3, 640, 640)
    assert io.names == {0: "person", 2: "car"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_pipeline_end_to_end(synthetic_model, backend):
    from yolokit.pipeline import Pipeline

    layout, path = synthetic_model
    pipe = Pipeline(path, backend=backend)
    assert pipe.layout == layout
    dets = pipe(np.zeros((640, 640, 3), np.uint8))
    assert dets.cls.tolist() == [0, 2]
    np.testing.assert_allclose(dets.xyxy[0], [80, 160, 120, 240], atol=1e-3)


def test_cli_inspect(synthetic_model):
    from typer.testing import CliRunner

    from yolokit.cli import app

    layout, path = synthetic_model
    res = CliRunner().invoke(app, ["inspect", str(path)])
    assert res.exit_code == 0, res.output
    assert f"layout   : {layout}" in res.output
