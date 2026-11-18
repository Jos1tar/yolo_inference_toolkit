"""``yk`` command line: inspect / export / predict / parity / bench."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Multi-backend YOLO inference toolkit.")

REPO_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = REPO_ROOT / "tools" / "fetch_models.py"

ModelOpt = Annotated[Path, typer.Option("--model", "-m", exists=True, dir_okay=False, help="ONNX model file.")]
BackendOpt = Annotated[str, typer.Option("--backend", "-b", help="torch | onnxrt | openvino")]
LayoutOpt = Annotated[str | None, typer.Option("--layout", help="Override layout detection.")]


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@app.command()
def inspect(model: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Print I/O shapes and the detected output layout."""
    from yolokit.layout import LayoutError, detect_layout, read_onnx_io, sig_from_shapes

    io = read_onnx_io(model)
    typer.echo(f"model    : {model}")
    typer.echo(f"input    : {io.input_name} {list(io.input_shape)}")
    for name, shape in zip(io.output_names, io.output_shapes, strict=True):
        typer.echo(f"output   : {name} {list(shape)}")
    try:
        sig = sig_from_shapes(io.output_shapes)
        layout = detect_layout(sig)
    except LayoutError as e:
        typer.secho(f"layout   : ? ({e})", fg="red")
        raise typer.Exit(1) from e
    from yolokit.adapters import get_adapter

    nc = get_adapter(layout)(sig).num_classes
    names = io.names
    n_cls = len(names) if names else (nc if nc > 0 else "?")
    typer.echo(f"layout   : {layout}  (boxes={sig.n_boxes}, channels={sig.n_channels})")
    typer.echo(f"classes  : {n_cls}" + ("" if names else " (no names metadata)"))


@app.command()
def export(
    source: Annotated[str, typer.Argument(help="yolov8n.pt / model name (needs ultralytics), or an .onnx file")],
    to: Annotated[str, typer.Option(help="Comma-separated targets: onnx, openvino")] = "onnx",
    out_dir: Annotated[Path, typer.Option(help="Where .pt/.onnx go")] = Path("models"),
    imgsz: int = 640,
    fp16: Annotated[bool, typer.Option(help="FP16-compress OpenVINO IR weights")] = False,
) -> None:
    """Export .pt -> .onnx (delegates to tools/, needs ultralytics) and/or .onnx -> OpenVINO IR."""
    targets = set(_csv(to))
    unknown = targets - {"onnx", "openvino"}
    if unknown:
        raise typer.BadParameter(f"unknown targets {sorted(unknown)}")

    src = Path(source)
    if src.suffix == ".onnx":
        onnx_path = src
    else:
        if not FETCH_SCRIPT.exists():
            typer.secho("tools/fetch_models.py not found; .pt export only works from a source checkout", fg="red")
            raise typer.Exit(1)
        name = src.stem if src.suffix == ".pt" else source
        cmd = [sys.executable, str(FETCH_SCRIPT), name, "--out", str(out_dir), "--imgsz", str(imgsz)]
        subprocess.run(cmd, check=True)
        onnx_path = out_dir / f"{name}.onnx"

    if "openvino" in targets:
        from yolokit.convert import onnx_to_openvino

        xml = onnx_to_openvino(onnx_path, fp16=fp16)
        typer.echo(f"openvino : {xml}")
    typer.echo(f"onnx     : {onnx_path}")


@app.command()
def predict(
    model: ModelOpt,
    source: Annotated[Path, typer.Option("--source", "-s", exists=True, help="Image file or folder.")],
    backend: BackendOpt = "onnxrt",
    device: str = "cpu",
    layout: LayoutOpt = None,
    conf: float = 0.25,
    iou: float = 0.45,
    save: Annotated[bool, typer.Option(help="Write annotated images to --out.")] = False,
    out: Path = Path("runs/predict"),
    as_json: Annotated[bool, typer.Option("--json", help="Print detections as JSON.")] = False,
) -> None:
    """Run detection on an image or a folder of images."""
    import cv2

    from yolokit.parity import list_images
    from yolokit.pipeline import Pipeline

    pipe = Pipeline(model, backend=backend, device=device, layout=layout, conf=conf, iou=iou)
    images = list_images(source) if source.is_dir() else [source]
    names = pipe.names
    report = {}
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            typer.secho(f"skip unreadable {p}", fg="yellow")
            continue
        dets, t = pipe.run(img)
        report[p.name] = dets.to_dicts(names)
        if not as_json:
            summary = ", ".join(f"{names.get(int(k), k)} {c:.2f}" for k, c in zip(dets.cls, dets.conf, strict=True))
            typer.echo(f"{p.name}: {len(dets)} boxes in {t.total_ms:.1f} ms  [{summary}]")
        if save:
            from yolokit.draw import draw

            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / p.name), draw(img, dets, names))
    if as_json:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    elif save:
        typer.echo(f"saved to {out}")


@app.command()
def parity(
    models: Annotated[list[Path], typer.Argument(exists=True, help="ONNX files, e.g. models/*.onnx")],
    backends: Annotated[str, typer.Option(help="Comma-separated candidate backends.")] = "onnxrt,openvino",
    fixtures: Path = Path("tests/fixtures"),
    reference: str = "torch",
    conf: float = 0.25,
    iou: float = 0.45,
    min_iou: float = 0.95,
    max_dconf: float = 0.01,
    markdown: Annotated[bool, typer.Option("--markdown", help="Print a README-ready table.")] = False,
) -> None:
    """Compare backends against the reference backend on fixture images. Exit 1 on any failure."""
    from yolokit.parity import HEADER, Tolerance, _cells, run_parity, to_markdown

    rows = run_parity(
        models, _csv(backends), fixtures, reference=reference, conf=conf, iou=iou, tol=Tolerance(min_iou, max_dconf)
    )
    if markdown:
        typer.echo(to_markdown(rows))
    else:
        typer.echo("  ".join(f"{h:<14}" for h in HEADER))
        for r in rows:
            typer.echo("  ".join(f"{c:<14}" for c in _cells(r)))
            if r.failed_images:
                typer.echo(f"    failed: {', '.join(r.failed_images)}")
    if not all(r.passed for r in rows):
        raise typer.Exit(1)


@app.command()
def bench(
    model: ModelOpt,
    backend: Annotated[str, typer.Option("--backend", "-b", help="One or more, comma-separated.")] = "onnxrt",
    source: Annotated[Path | None, typer.Option(help="Image to benchmark on (default: first fixture).")] = None,
    runs: int = 100,
    warmup: int = 10,
    device: str = "cpu",
    markdown: Annotated[bool, typer.Option("--markdown")] = False,
) -> None:
    """Latency p50/p95 per backend."""
    import cv2

    from yolokit.bench import bench as run_bench
    from yolokit.bench import host_info, to_markdown
    from yolokit.parity import list_images
    from yolokit.pipeline import Pipeline

    if source is None:
        found = list_images(Path("tests/fixtures"))
        if not found:
            raise typer.BadParameter("no --source given and tests/fixtures is empty")
        source = found[0]
    img = cv2.imread(str(source))
    results = [run_bench(Pipeline(model, backend=b, device=device), img, runs, warmup) for b in _csv(backend)]
    typer.echo(to_markdown(results) if markdown else "\n".join(map(str, results)))
    typer.echo(f"\nhost: {host_info()} | runs={runs} warmup={warmup} | image={source.name}")


if __name__ == "__main__":
    app()
