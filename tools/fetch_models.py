"""Download reference checkpoints and export them to ONNX.

This is the ONLY place in the repo that imports ``ultralytics`` (AGPL-3.0) or
runs yolov5 code; the ``yolokit`` runtime never does. Usage:

    python tools/fetch_models.py                    # M1 set: yolov5n yolov8n yolo11n
    python tools/fetch_models.py yolo26n --out models

Output: models/<name>.pt and models/<name>.onnx (fixed 1x3x640x640, opset 12).
yolov5 models are exported with the original yolov5 export.py (layout
``objectness``), not ultralytics' v5u re-implementation (layout ``dfl``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

M1_MODELS = ["yolov5n", "yolov8n", "yolo11n"]
YOLOV5_REPO_URL = "https://github.com/ultralytics/yolov5"
# yolov5 checkpoints are full pickles; torch>=2.6 defaults torch.load(weights_only=True)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def ensure_yolov5_repo(dest: Path) -> Path:
    if not (dest / "export.py").exists():
        if shutil.which("git") is None:
            sys.exit("git is required to clone the yolov5 repo")
        subprocess.run(["git", "clone", "--depth", "1", YOLOV5_REPO_URL, str(dest)], check=True)
    return dest


def export_yolov5(name: str, out: Path, imgsz: int, opset: int) -> Path:
    # absolute: the subprocess below runs with cwd=repo, so a relative path would be joined twice
    repo = ensure_yolov5_repo(out / "_yolov5").resolve()
    pt = (out / f"{name}.pt").resolve()
    cmd = [
        sys.executable, str(repo / "export.py"),
        "--weights", str(pt),  # export.py downloads the release asset if missing
        "--include", "onnx",
        "--imgsz", str(imgsz),
        "--opset", str(opset),
        # no --simplify: onnx-simplifier has no wheels for recent Pythons and is not needed for correctness
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=repo)
    return pt.with_suffix(".onnx")


def export_ultralytics(name: str, out: Path, imgsz: int, opset: int) -> Path:
    from ultralytics import YOLO

    pt = out / f"{name}.pt"
    model = YOLO(str(pt))  # downloads into `out` if missing
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False)
    return Path(onnx_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=M1_MODELS)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--force", action="store_true", help="re-export even if the .onnx exists")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        name = Path(name).stem
        target = args.out / f"{name}.onnx"
        if target.exists() and not args.force:
            print(f"[skip] {target} exists")
            continue
        fn = export_yolov5 if name.startswith("yolov5") and not name.endswith("u") else export_ultralytics
        path = fn(name, args.out, args.imgsz, args.opset)
        print(f"[ok]   {name} -> {path}")


if __name__ == "__main__":
    main()
