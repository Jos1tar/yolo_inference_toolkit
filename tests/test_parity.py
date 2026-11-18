"""Parity matcher unit tests + the real cross-backend parity run (needs `make models`)."""

import importlib.util

import numpy as np
import pytest

from yolokit.adapters.base import Detections
from yolokit.parity import Tolerance, compare, list_images, run_parity, to_markdown


def _dets(boxes, conf, cls):
    return Detections(np.asarray(boxes, np.float32), np.asarray(conf, np.float32), np.asarray(cls, np.int64))


REF = _dets([[0, 0, 100, 100], [200, 200, 300, 260]], [0.9, 0.5], [0, 2])


def test_identical_passes():
    r = compare(REF, REF)
    assert r.passed and r.min_iou == pytest.approx(1.0) and r.max_dconf == 0.0


def test_small_jitter_passes():
    cand = _dets([[0.5, 0, 100, 100.5], [200, 200, 300, 260]], [0.905, 0.497], [0, 2])
    assert compare(REF, cand).passed


def test_order_independent():
    cand = _dets(REF.xyxy[::-1], REF.conf[::-1], REF.cls[::-1])
    assert compare(REF, cand).passed


def test_conf_drift_fails():
    cand = _dets(REF.xyxy, [0.9, 0.53], REF.cls)
    r = compare(REF, cand)
    assert not r.passed and r.max_dconf == pytest.approx(0.03, abs=1e-6)


def test_class_mismatch_fails():
    cand = _dets(REF.xyxy, REF.conf, [0, 3])
    r = compare(REF, cand)
    assert not r.passed and r.n_matched == 1


def test_box_count_mismatch_fails():
    cand = _dets(REF.xyxy[:1], REF.conf[:1], REF.cls[:1])
    assert not compare(REF, cand).passed


def test_low_iou_fails():
    cand = _dets([[10, 10, 110, 110], [200, 200, 300, 260]], REF.conf, REF.cls)
    r = compare(REF, cand, Tolerance(min_iou=0.95))
    assert not r.passed and r.min_iou < 0.95


def test_both_empty_passes():
    assert compare(Detections.empty(), Detections.empty()).passed


# ---------------------------------------------------------------- integration
CANDIDATES = [b for b, mod in (("onnxrt", "onnxruntime"), ("openvino", "openvino")) if importlib.util.find_spec(mod)]


@pytest.mark.models
def test_backend_parity(models_dir, fixtures_dir):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch reference backend not installed (pip install -e .[ref])")
    models = sorted(p for p in models_dir.glob("*.onnx") if p.with_suffix(".pt").exists())
    if not models:
        pytest.skip("no models; run `make models`")
    if not list_images(fixtures_dir):
        pytest.skip("no fixture images in tests/fixtures")
    rows = run_parity(models, CANDIDATES, fixtures_dir)
    print("\n" + to_markdown(rows))
    failed = [f"{r.model}/{r.backend}: {r.failed_images}" for r in rows if not r.passed]
    assert not failed, failed
