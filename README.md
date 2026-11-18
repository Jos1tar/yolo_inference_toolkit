# yolokit

Run YOLO-family detectors on several CPU inference backends from one ONNX file, and check in CI that every
backend produces the same detections as the original PyTorch model.

- The decoder is chosen from the shape of the ONNX output tensor, not from a version string. YOLOv8, v11, v12
  and v5u share one output layout, the original YOLOv5 export has another, and YOLOv10 / YOLO26 (NMS-free) a
  third. A layout that matches no adapter, or more than one, is an error rather than a guess.
- Backends: PyTorch (reference), ONNX Runtime, OpenVINO. TensorRT is planned next.
- Parity tests compare every (model, backend) pair against the PyTorch forward pass on fixed images.
- The runtime needs only numpy, OpenCV, onnx and a backend. Ultralytics and yolov5 code is used in `tools/` to
  download and export checkpoints, and by the optional PyTorch reference backend.

## Output layouts

| layout | produced by | ONNX output | decode | NMS |
|---|---|---|---|---|
| `objectness` | yolov5 `export.py` | `(1, N, 5+nc)` | xywh, score = obj * cls | yes |
| `dfl` | ultralytics export (v8, v11, v12, v5u) | `(1, 4+nc, N)` | xywh, class scores (DFL already applied in the graph) | yes |
| `e2e` | ultralytics export (v10, YOLO26) | `(1, 300, 6)` | xyxy, conf, cls | no |

The `e2e` adapter is implemented and unit-tested on synthetic outputs but not yet checked against a real model.

## Install

```
pip install -e ".[cpu]"            # runtime: onnxruntime + openvino
pip install -e ".[cpu,ref,dev]"    # + PyTorch reference backend, export tools, tests
```

## Usage

```
python tools/fetch_models.py                     # yolov5n, yolov8n, yolo11n -> models/*.pt, *.onnx
python tools/fetch_fixtures.py                   # parity images -> tests/fixtures/
yk export models/yolov8n.onnx --to openvino      # optional: OpenVINO IR next to the ONNX file
yk inspect models/yolov8n.onnx                   # I/O shapes and detected layout
yk predict -m models/yolov8n.onnx -b openvino -s image.jpg --save
yk parity models/yolov5n.onnx models/yolov8n.onnx models/yolo11n.onnx --markdown
yk bench -m models/yolo11n.onnx -b torch,onnxrt,openvino --runs 100 --markdown
```

From Python:

```python
import cv2
from yolokit import Pipeline

pipe = Pipeline("models/yolo11n.onnx", backend="onnxrt", conf=0.25, iou=0.45)
dets = pipe(cv2.imread("image.jpg"))  # Detections(xyxy, conf, cls) in original image pixels
```

## Parity

Reference is the PyTorch model; input 640x640 letterbox, conf 0.25, NMS IoU 0.45, FP32 on CPU. A row passes when,
on every fixture image, the box counts are equal, every reference box is matched to a candidate of the same class,
every matched IoU is at least 0.95 and every confidence differs by at most 0.01. Totals are over the four images.

| model | backend | boxes ref/cand | min IoU | max Δconf | result |
|---|---|---|---|---|---|
| yolov5n | onnxrt | 53/53 | 0.999997 | 1.8e-06 | pass |
| yolov5n | openvino | 53/53 | 0.999997 | 1.8e-06 | pass |
| yolov8n | onnxrt | 49/49 | 0.999998 | 1.2e-06 | pass |
| yolov8n | openvino | 49/49 | 0.999997 | 1.1e-06 | pass |
| yolo11n | onnxrt | 62/62 | 0.999997 | 1.7e-06 | pass |
| yolo11n | openvino | 62/62 | 0.999996 | 2.2e-06 | pass |

Parity alone cannot catch a decoding bug shared by all backends, so the decoders were also checked against the
upstream pipelines on the same images: yolov5's own letterbox and `non_max_suppression`, and ultralytics
`predict` for v8n and 11n. Box counts are identical, IoU is 1.0 and confidences differ by about 1e-6.
Note that upstream yolov5 `AutoShape` and ultralytics `.pt` prediction use rectangular (minimum-padding) input
by default, which gives slightly different results on non-square images; yolokit always uses a fixed square input.

Fixtures are four COCO val2017 images chosen by a fixed rule in `tools/fetch_fixtures.py` (redistributable
license, at least 3 categories, one image per aspect ratio bucket). Sources and licenses are listed in
`tests/fixtures/ATTRIBUTION.md`.

## Latency

AMD Ryzen 9 9955HX (16 cores), Windows 11, onnxruntime 1.32, OpenVINO 2026.4, PyTorch 2.15 (CPU), default
thread settings. 100 runs after 10 warm-up runs on one fixture image (612x612). `infer` is the backend call alone;
`e2e` adds letterbox, decoding and NMS. Laptop numbers; expect a few ms of run-to-run variation.

| model | backend | infer p50 (ms) | infer p95 (ms) | e2e p50 (ms) | e2e p95 (ms) | FPS |
|---|---|---|---|---|---|---|
| yolov5n | torch | 20.34 | 24.49 | 24.16 | 28.11 | 41.4 |
| yolov5n | onnxrt | 10.07 | 11.43 | 15.06 | 16.19 | 66.4 |
| yolov5n | openvino | 7.32 | 8.69 | 10.78 | 12.69 | 92.8 |
| yolov8n | torch | 20.06 | 23.87 | 25.47 | 29.78 | 39.3 |
| yolov8n | onnxrt | 13.85 | 15.05 | 19.58 | 21.15 | 51.1 |
| yolov8n | openvino | 7.38 | 8.66 | 13.27 | 15.56 | 75.4 |
| yolo11n | torch | 23.35 | 28.40 | 29.51 | 33.08 | 33.9 |
| yolo11n | onnxrt | 14.53 | 15.72 | 20.48 | 21.81 | 48.8 |
| yolo11n | openvino | 7.59 | 8.88 | 13.48 | 14.90 | 74.2 |

## Not supported yet

Segmentation, pose, OBB and classification heads; dynamic input shapes; batch size above 1; Darknet `.weights`;
TensorRT; Nsight Systems profiling.

## Repository layout

```
yolokit/
  layout.py         ONNX I/O -> layout signature -> registered adapter
  adapters/         objectness.py  dfl.py  e2e.py   (preprocess / decode / postprocess)
  backends/         torch_ref.py  onnxrt.py  openvino.py  tensorrt.py (planned)
  ops/              letterbox.py  nms.py
  parity.py  bench.py  pipeline.py  cli.py
tools/
  fetch_models.py   download and export checkpoints (the only place that imports ultralytics)
  fetch_fixtures.py select the COCO val2017 parity images
tests/              unit tests, synthetic-ONNX end-to-end tests, parity test```

## License

MIT. Model weights are not distributed; `tools/fetch_models.py` downloads them under their own licenses.
