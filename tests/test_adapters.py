"""Adapters on synthetic raw outputs: two objects, plus a duplicate that NMS must remove."""

import numpy as np
import pytest

from yolokit.adapters import get_adapter
from yolokit.layout import sig_from_shapes
from yolokit.ops.letterbox import letterbox

NC = 80
N = 8400
# (cx, cy, w, h, cls, score) in letterbox coords
OBJECTS = [(100, 200, 40, 80, 0, 0.9), (400, 300, 100, 50, 2, 0.8), (102, 201, 40, 80, 0, 0.6)]


def _dfl_output():
    out = np.zeros((1, 4 + NC, N), np.float32)
    for i, (cx, cy, w, h, k, s) in enumerate(OBJECTS):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + k, i] = s
    return out


def _objectness_output(n=25200):
    out = np.zeros((1, n, 5 + NC), np.float32)
    for i, (cx, cy, w, h, k, s) in enumerate(OBJECTS):
        out[0, i, :4] = (cx, cy, w, h)
        out[0, i, 4] = 1.0  # obj
        out[0, i, 5 + k] = s
    out[0, 10, :5] = (300, 300, 10, 10, 0.2)  # obj below threshold, high cls: must be dropped
    out[0, 10, 5] = 1.0
    return out


def _e2e_output():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = (80, 160, 120, 240, 0.9, 0)
    out[0, 1] = (350, 275, 450, 325, 0.8, 2)
    return out


@pytest.mark.parametrize(
    "layout, raw, n_expected",
    [("dfl", _dfl_output(), 2), ("objectness", _objectness_output(), 2), ("e2e", _e2e_output(), 2)],
)
def test_decode_and_postprocess(layout, raw, n_expected):
    adapter = get_adapter(layout)(sig_from_shapes([raw.shape]))
    cand = adapter.decode([raw], conf_thr=0.25)
    assert cand.shape[1] == 6
    _, meta = letterbox(np.zeros((640, 640, 3), np.uint8))  # identity letterbox
    dets = adapter.postprocess(cand, meta, iou_thr=0.45)
    assert len(dets) == n_expected
    assert dets.cls.tolist() == [0, 2]
    np.testing.assert_allclose(dets.xyxy[0], [80, 160, 120, 240], atol=1e-4)
    np.testing.assert_allclose(dets.conf, [0.9, 0.8], atol=1e-6)


def test_objectness_score_is_obj_times_cls():
    raw = np.zeros((1, 25200, 85), np.float32)
    raw[0, 0, :5] = (100, 100, 20, 20, 0.5)
    raw[0, 0, 5 + 7] = 0.8
    adapter = get_adapter("objectness")(sig_from_shapes([raw.shape]))
    cand = adapter.decode([raw], conf_thr=0.25)
    assert cand.shape == (1, 6)
    assert cand[0, 4] == pytest.approx(0.4) and cand[0, 5] == 7


def test_num_classes():
    assert get_adapter("dfl")(sig_from_shapes([(1, 84, 8400)])).num_classes == 80
    assert get_adapter("objectness")(sig_from_shapes([(1, 25200, 85)])).num_classes == 80


def test_preprocess_layout():
    adapter = get_adapter("dfl")(sig_from_shapes([(1, 84, 8400)]))
    img = np.zeros((480, 640, 3), np.uint8)
    img[..., 0] = 255  # pure blue in BGR
    x, meta = adapter.preprocess(img, (640, 640))
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and x.flags.c_contiguous
    # after BGR->RGB the blue channel is last
    assert x[0, 2, 320, 320] == pytest.approx(1.0) and x[0, 0, 320, 320] == 0.0
    assert x[0, 0, 0, 0] == pytest.approx(114 / 255)  # pad row
