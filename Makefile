PY      ?= python
MODELS  ?= yolov5n yolov8n yolo11n
FIX     ?= tests/fixtures

.PHONY: install fixtures models test test-unit parity bench lint clean

install:
	$(PY) -m pip install -e ".[cpu,ref,dev]"

fixtures:
	$(PY) tools/fetch_fixtures.py --out $(FIX)

models:
	$(PY) tools/fetch_models.py $(MODELS) --out models
	for m in $(MODELS); do yk export models/$$m.onnx --to openvino; done

test-unit:
	$(PY) -m pytest -m "not models"

test:
	$(PY) -m pytest

parity:
	yk parity models/*.onnx --backends onnxrt,openvino --fixtures $(FIX) --markdown

bench:
	for m in $(MODELS); do yk bench -m models/$$m.onnx -b torch,onnxrt,openvino --runs 100 --warmup 10 --markdown; done

lint:
	ruff check . && ruff format --check .

clean:
	rm -rf runs .pytest_cache build *.egg-info
