import pytest

from yolokit.layout import LayoutError, detect_layout, sig_from_shapes


@pytest.mark.parametrize(
    "shapes, expected",
    [
        ([(1, 25200, 85)], "objectness"),  # yolov5n export.py @640
        ([(1, 6300, 85)], "objectness"),  # yolov5n @320
        ([(1, 84, 8400)], "dfl"),  # yolov8n / yolo11n @640
        ([(1, 5, 8400)], "dfl"),  # single-class dfl
        ([(1, 300, 6)], "e2e"),  # yolov10 / yolo26
    ],
)
def test_detect_layout(shapes, expected):
    assert detect_layout(sig_from_shapes(shapes)) == expected


def test_sig_fields():
    sig = sig_from_shapes([(1, 84, 8400)])
    assert sig.channels_first and sig.n_boxes == 8400 and sig.n_channels == 84
    sig = sig_from_shapes([(1, 25200, 85)])
    assert not sig.channels_first and sig.n_boxes == 25200 and sig.n_channels == 85


def test_sig_key_ignores_shape_details():
    # LayoutSig equality is the registry key: v8n and v11n (same layout, same nc) collide on purpose
    assert sig_from_shapes([(1, 84, 8400)]) == sig_from_shapes([(1, 84, 8400)])


@pytest.mark.parametrize(
    "shapes",
    [
        [(1, -1, 85)],  # dynamic box dim
        [(1, 84)],  # wrong rank
        [],
    ],
)
def test_bad_shapes_raise(shapes):
    with pytest.raises(LayoutError):
        sig_from_shapes(shapes)


def test_unknown_layout_raises():
    # segmentation exports have two outputs; M1 refuses instead of guessing
    with pytest.raises(LayoutError, match="no adapter"):
        detect_layout(sig_from_shapes([(1, 116, 8400), (1, 32, 160, 160)]))
