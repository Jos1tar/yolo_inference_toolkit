import numpy as np
import pytest

from yolokit.ops import batched_nms, box_iou, letterbox, nms, scale_boxes
from yolokit.ops.nms import HAS_TORCHVISION

NMS_IMPLS = [False, pytest.param(True, marks=pytest.mark.skipif(not HAS_TORCHVISION, reason="no torchvision"))]


@pytest.mark.parametrize("hw", [(480, 640), (640, 480), (1080, 1920), (333, 777), (640, 640), (100, 50)])
def test_letterbox_shape_and_roundtrip(hw):
    img = np.zeros((*hw, 3), np.uint8)
    out, meta = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3)
    assert meta.orig_hw == hw
    # a box covering the whole original image maps to the unpadded region and back
    h0, w0 = hw
    full = np.array([[0, 0, w0, h0]], np.float32)
    lb = full * meta.ratio + np.array([meta.pad[0], meta.pad[1]] * 2, np.float32)
    back = scale_boxes(lb, meta)
    np.testing.assert_allclose(back, full, atol=1e-3)


def test_letterbox_pad_value():
    img = np.full((320, 640, 3), 255, np.uint8)
    out, meta = letterbox(img, (640, 640))
    assert meta.pad == (0, 160)
    assert (out[0, 0] == 114).all() and (out[320, 320] == 255).all()


def test_scale_boxes_clips():
    _, meta = letterbox(np.zeros((100, 200, 3), np.uint8), (640, 640))
    out = scale_boxes(np.array([[-50, -50, 700, 700]], np.float32), meta)
    assert out.min() >= 0 and out[0, 2] <= 200 and out[0, 3] <= 100


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


BOXES = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]], np.float32)
SCORES = np.array([0.9, 0.8, 0.7, 0.95], np.float32)


@pytest.mark.parametrize("use_torch", NMS_IMPLS)
def test_nms(use_torch):
    keep = nms(BOXES, SCORES, 0.5, use_torch=use_torch)
    assert keep.tolist() == [3, 2]


@pytest.mark.parametrize("use_torch", NMS_IMPLS)
def test_batched_nms_separates_classes(use_torch):
    cls = np.array([0, 1, 0, 0])
    keep = batched_nms(BOXES, SCORES, cls, 0.5, use_torch=use_torch)
    assert sorted(keep.tolist()) == [1, 2, 3]  # box 1 survives: different class from box 3


def test_nms_empty():
    assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.5).size == 0
