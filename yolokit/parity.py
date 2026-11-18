"""Cross-backend parity: does backend X produce the same detections as the torch reference?

For each fixture image and (model, backend):
  1. reference = torch backend Detections; candidate = other backend, same conf/iou/imgsz
  2. greedy match: reference boxes by descending conf, each takes the unused
     same-class candidate with the highest IoU
  3. pass iff box counts equal, every reference box matched, every matched IoU >= min_iou,
     every |Δconf| <= max_dconf (class agreement is implied by same-class matching)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from yolokit.adapters.base import Detections
from yolokit.ops.nms import box_iou

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Tolerance:
    min_iou: float = 0.95
    max_dconf: float = 0.01


@dataclass
class ImageParity:
    n_ref: int
    n_cand: int
    n_matched: int
    min_iou: float
    max_dconf: float
    passed: bool


def compare(ref: Detections, cand: Detections, tol: Tolerance = Tolerance()) -> ImageParity:
    n_ref, n_cand = len(ref), len(cand)
    ious: list[float] = []
    dconfs: list[float] = []
    if n_ref and n_cand:
        iou = box_iou(ref.xyxy, cand.xyxy)
        used = np.zeros(n_cand, dtype=bool)
        for i in np.argsort(-ref.conf, kind="stable"):
            ok = (cand.cls == ref.cls[i]) & ~used
            if not ok.any():
                continue
            row = np.where(ok, iou[i], -1.0)
            j = int(row.argmax())
            if row[j] <= 0:
                continue
            used[j] = True
            ious.append(float(row[j]))
            dconfs.append(abs(float(ref.conf[i]) - float(cand.conf[j])))
    n_matched = len(ious)
    min_iou = min(ious) if ious else (1.0 if n_ref == n_cand == 0 else 0.0)
    max_dconf = max(dconfs) if dconfs else 0.0
    passed = n_ref == n_cand and n_matched == n_ref and min_iou >= tol.min_iou and max_dconf <= tol.max_dconf
    return ImageParity(n_ref, n_cand, n_matched, min_iou, max_dconf, passed)


@dataclass
class ParityRow:
    model: str
    backend: str
    n_ref: int
    n_cand: int
    min_iou: float
    max_dconf: float
    passed: bool
    failed_images: list[str] = field(default_factory=list)


def aggregate(model: str, backend: str, per_image: dict[str, ImageParity]) -> ParityRow:
    vals = list(per_image.values())
    return ParityRow(
        model=model,
        backend=backend,
        n_ref=sum(v.n_ref for v in vals),
        n_cand=sum(v.n_cand for v in vals),
        min_iou=min((v.min_iou for v in vals), default=1.0),
        max_dconf=max((v.max_dconf for v in vals), default=0.0),
        passed=all(v.passed for v in vals),
        failed_images=[k for k, v in per_image.items() if not v.passed],
    )


def list_images(fixtures: Path) -> list[Path]:
    return sorted(p for p in Path(fixtures).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run_parity(
    models: Sequence[Path],
    backends: Sequence[str],
    fixtures: Path,
    reference: str = "torch",
    conf: float = 0.25,
    iou: float = 0.45,
    tol: Tolerance = Tolerance(),
) -> list[ParityRow]:
    import cv2

    from yolokit.pipeline import Pipeline

    images = {p.name: cv2.imread(str(p)) for p in list_images(fixtures)}
    if not images:
        raise FileNotFoundError(f"no fixture images in {fixtures}")

    rows: list[ParityRow] = []
    for model in models:
        ref_pipe = Pipeline(model, backend=reference, conf=conf, iou=iou)
        ref_dets = {k: ref_pipe(img) for k, img in images.items()}
        for b in backends:
            pipe = Pipeline(model, backend=b, conf=conf, iou=iou)
            per_image = {k: compare(ref_dets[k], pipe(img), tol) for k, img in images.items()}
            rows.append(aggregate(Path(model).stem, b, per_image))
    return rows


HEADER = ("model", "backend", "boxes ref/cand", "min IoU", "max Δconf", "pass")


def _cells(r: ParityRow) -> tuple[str, ...]:
    return (
        r.model,
        r.backend,
        f"{r.n_ref}/{r.n_cand}",
        f"{r.min_iou:.6f}",
        f"{r.max_dconf:.1e}",
        "pass" if r.passed else "FAIL",
    )


def to_markdown(rows: Iterable[ParityRow]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(_cells(r)) + " |" for r in rows]
    return "\n".join(lines)
