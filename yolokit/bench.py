"""Latency benchmark: warmup, then N timed runs; report p50 / p95.

``infer`` times only ``backend.infer`` on a fixed preprocessed tensor (what
backend comparisons should use); ``e2e`` includes preprocess + decode + NMS.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from yolokit.pipeline import Pipeline


@dataclass
class BenchResult:
    model: str
    backend: str
    runs: int
    infer_p50: float
    infer_p95: float
    e2e_p50: float
    e2e_p95: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.e2e_p50 if self.e2e_p50 else 0.0


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q))


def bench(pipe: Pipeline, img_bgr: np.ndarray, runs: int = 100, warmup: int = 10) -> BenchResult:
    x, _ = pipe.adapter.preprocess(img_bgr, pipe.imgsz)
    for _ in range(warmup):
        pipe.runner.infer(x)

    infer_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pipe.runner.infer(x)
        infer_ms.append((time.perf_counter() - t0) * 1e3)

    e2e_ms = [pipe.run(img_bgr)[1].total_ms for _ in range(runs)]

    return BenchResult(
        model=pipe.model.stem,
        backend=pipe.backend,
        runs=runs,
        infer_p50=_pct(infer_ms, 50),
        infer_p95=_pct(infer_ms, 95),
        e2e_p50=_pct(e2e_ms, 50),
        e2e_p95=_pct(e2e_ms, 95),
    )


HEADER = ("model", "backend", "infer p50 (ms)", "infer p95 (ms)", "e2e p50 (ms)", "e2e p95 (ms)", "FPS")


def host_info() -> str:
    return f"{platform.processor() or platform.machine()} | {platform.system()} {platform.release()}"


def to_markdown(results: Iterable[BenchResult]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    for r in results:
        cells = (
            r.model,
            r.backend,
            f"{r.infer_p50:.2f}",
            f"{r.infer_p95:.2f}",
            f"{r.e2e_p50:.2f}",
            f"{r.e2e_p95:.2f}",
            f"{r.fps:.1f}",
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
