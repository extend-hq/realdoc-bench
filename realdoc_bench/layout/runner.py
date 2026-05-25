"""Layout benchmark runner.

For each (sample, processor) the runner emits one ``CellResult`` capturing the
predicted document, ground truth, evaluation metrics, and cost/latency. Results
serialize as JSON under ``runs/<run_id>/``.
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from realdoc_bench.layout.data.loader import DEFAULT_HF_DATASET, LayoutSample, load
from realdoc_bench.layout.metrics import structural_errors as struct
from realdoc_bench.layout.metrics.f1 import PRF, aggregate, aggregate_per_class, evaluate, per_class
from realdoc_bench.layout.metrics.adjacency import score_pairs
from realdoc_bench.layout.metrics.adjacency.aggregate import MetricBreakdown
from realdoc_bench.layout.normalizers.base import LayoutBlock, LayoutDocument
from realdoc_bench.layout.processors.base import build as build_processor
from realdoc_bench.shared.pricing.meter import catalog_sha

# Selectable layout scorer:
#   "f1"       — realdoc-bench's native scorer (greedy-Hungarian + strict type
#                gate). Backwards-compatible default.
#   "adjacency" — Hungarian 1:1 with adjacency split/merge recovery; adds
#                  an `adjusted` metric per cell on top of the strict one.
LayoutScorer = Literal["f1", "adjacency"]


@dataclass
class CellResult:
    page_id: str
    domain: str
    processor: str
    version: str
    status: str
    error: str | None = None
    latency_sec: float | None = None
    cost_estimate_usd: float | None = None
    confidence_present: bool = False
    strict: PRF | None = None
    # Populated only by the "adjacency" scorer — F1 after adjacency split/merge
    # recovery. ``None`` under the default "f1" scorer.
    adjusted: PRF | None = None
    per_class: dict[str, PRF] = field(default_factory=dict)
    structural: struct.StructuralErrors | None = None
    pred_count: int = 0
    gt_count: int = 0


@dataclass
class RunSummary:
    run_id: str
    started_at: float
    finished_at: float
    catalog_sha: str
    processors: list[str]
    n_pages: int
    cells: list[CellResult]
    scorer: LayoutScorer = "f1"


def _page_dims(*docs: LayoutDocument) -> tuple[int, int]:
    """Largest (width, height) across every page of the given documents."""
    w = max((p.width or 0 for d in docs for p in d.pages), default=0)
    h = max((p.height or 0 for d in docs for p in d.pages), default=0)
    return w, h


def _breakdown_to_prf(bd: MetricBreakdown) -> PRF:
    """Adapt an adjacency-scorer ``MetricBreakdown`` (micro totals) to a ``PRF``."""
    return PRF(
        tp=bd.tp,
        fp=bd.fp,
        fn=bd.fn,
        precision=bd.micro_precision,
        recall=bd.micro_recall,
        f1=bd.micro_f1,
    )


def _score_adjacency(
    preds: list[LayoutBlock],
    gts: list[LayoutBlock],
    page_dims: tuple[int, int],
    iou_threshold: float,
) -> tuple[PRF, dict[str, PRF], PRF]:
    """Score one cell with the adjacency scorer — returns (strict, per_class, adjusted)."""
    page_w, page_h = page_dims
    result = score_pairs([(preds, gts, page_w, page_h)], iou_threshold=iou_threshold)
    strict = _breakdown_to_prf(result.strict)
    adjusted = _breakdown_to_prf(result.adjusted)
    per_cls = {
        cls: PRF(
            tp=cs.tp,
            fp=cs.fp,
            fn=cs.fn,
            precision=cs.precision,
            recall=cs.recall,
            f1=cs.f1,
        )
        for cls, cs in result.strict.per_class.items()
    }
    return strict, per_cls, adjusted


def _evaluate_cell(
    sample: LayoutSample,
    pred_doc: LayoutDocument,
    *,
    iou_threshold: float = 0.5,
    scorer: LayoutScorer = "f1",
) -> tuple[PRF, dict[str, PRF], struct.StructuralErrors, PRF | None]:
    """Score a (prediction, ground-truth) cell with the selected scorer.

    Returns ``(strict, per_class, structural, adjusted)``. ``adjusted`` is
    ``None`` for the "f1" scorer and an adjacency-recovered PRF for "adjacency".
    Structural-error analysis is scorer-independent and always computed.
    """
    preds = pred_doc.blocks
    gts = sample.ground_truth.blocks
    errors = struct.analyze(preds, gts, iou_threshold=iou_threshold)
    if scorer == "adjacency":
        strict, per_cls, adjusted = _score_adjacency(
            preds, gts, _page_dims(sample.ground_truth, pred_doc), iou_threshold
        )
        return strict, per_cls, errors, adjusted
    strict = evaluate(preds, gts, iou_threshold=iou_threshold, type_sensitive=True)
    per_cls = per_class(preds, gts, iou_threshold=iou_threshold)
    return strict, per_cls, errors, None


def run(
    *,
    processors: list[str],
    samples: Iterable[LayoutSample] | None = None,
    iou_threshold: float = 0.5,
    scorer: LayoutScorer = "f1",
    run_id: str | None = None,
    run_dir: Path = Path("runs"),
    limit: int | None = None,
    domains: Iterable[str] | None = None,
    dataset: str = DEFAULT_HF_DATASET,
) -> RunSummary:
    if samples is None:
        samples = list(load(hf_dataset=dataset, domains=domains, limit=limit))
    else:
        samples = list(samples)

    run_id = run_id or time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:6]
    run_root = run_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    procs = [build_processor(name) for name in processors]
    cells: list[CellResult] = []
    started_at = time.time()
    for sample in samples:
        for proc in procs:
            cell_dir = run_root / sample.page_id / proc.name
            cell_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = proc.predict(sample.image_path, gt=sample.ground_truth)
                strict, per_cls, errors, adjusted = _evaluate_cell(
                    sample, result.document, iou_threshold=iou_threshold, scorer=scorer
                )
                pred_confidences = {b.confidence for b in result.document.blocks}
                cell = CellResult(
                    page_id=sample.page_id,
                    domain=sample.domain,
                    processor=proc.name,
                    version=proc.version,
                    status="ok",
                    latency_sec=result.latency_sec,
                    cost_estimate_usd=result.cost_estimate_usd,
                    confidence_present=len(pred_confidences) > 1 or pred_confidences != {1.0},
                    strict=strict,
                    adjusted=adjusted,
                    per_class=per_cls,
                    structural=errors,
                    pred_count=len(result.document.blocks),
                    gt_count=len(sample.ground_truth.blocks),
                )
                (cell_dir / "prediction.json").write_text(
                    result.document.model_dump_json(indent=2)
                )
            except Exception as e:
                cell = CellResult(
                    page_id=sample.page_id,
                    domain=sample.domain,
                    processor=proc.name,
                    version=proc.version,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                )
                (cell_dir / "error.txt").write_text(traceback.format_exc())
            cells.append(cell)

    finished_at = time.time()
    summary = RunSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        catalog_sha=catalog_sha(),
        processors=[p.name for p in procs],
        n_pages=len(samples),
        cells=cells,
        scorer=scorer,
    )
    (run_root / "summary.json").write_text(
        json.dumps(_summary_to_jsonable(summary), indent=2, default=str)
    )
    return summary


def _summary_to_jsonable(s: RunSummary) -> dict:
    return {
        "run_id": s.run_id,
        "started_at": s.started_at,
        "finished_at": s.finished_at,
        "catalog_sha": s.catalog_sha,
        "scorer": s.scorer,
        "processors": s.processors,
        "n_pages": s.n_pages,
        "cells": [_cell_to_jsonable(c) for c in s.cells],
    }


def _cell_to_jsonable(c: CellResult) -> dict:
    return {
        "page_id": c.page_id,
        "domain": c.domain,
        "processor": c.processor,
        "version": c.version,
        "status": c.status,
        "error": c.error,
        "latency_sec": c.latency_sec,
        "cost_estimate_usd": c.cost_estimate_usd,
        "confidence_present": c.confidence_present,
        "strict": asdict(c.strict) if c.strict else None,
        "adjusted": asdict(c.adjusted) if c.adjusted else None,
        "per_class": {k: asdict(v) for k, v in c.per_class.items()},
        "structural": (
            {
                "misses": c.structural.misses,
                "hallucinations": c.structural.hallucinations,
                "misclassifications": c.structural.misclassifications,
                "confusion": {f"{pt}|{gt}": n for (pt, gt), n in c.structural.confusion.items()},
            }
            if c.structural
            else None
        ),
        "pred_count": c.pred_count,
        "gt_count": c.gt_count,
    }


def aggregate_summary(summary: RunSummary) -> dict[str, dict]:
    """Roll cell results up to per-processor aggregates for the leaderboard."""
    by_proc: dict[str, list[CellResult]] = {}
    for cell in summary.cells:
        by_proc.setdefault(cell.processor, []).append(cell)

    out: dict[str, dict] = {}
    for proc, cells in by_proc.items():
        ok = [c for c in cells if c.status == "ok" and c.strict is not None]
        if not ok:
            continue
        strict_total = aggregate([c.strict for c in ok if c.strict is not None])
        adjusted_cells = [c.adjusted for c in ok if c.adjusted is not None]
        adjusted_total = aggregate(adjusted_cells) if adjusted_cells else None
        per_class_total = aggregate_per_class([c.per_class for c in ok])
        struct_total = struct.aggregate([c.structural for c in ok if c.structural is not None])
        latencies = [c.latency_sec for c in ok if c.latency_sec is not None]
        costs = [c.cost_estimate_usd for c in ok if c.cost_estimate_usd is not None]
        confidence_present = any(c.confidence_present for c in ok)
        out[proc] = {
            "version": ok[0].version,
            "n_pages": len(ok),
            "n_errors": len(cells) - len(ok),
            "strict": asdict(strict_total),
            "adjusted": asdict(adjusted_total) if adjusted_total is not None else None,
            "per_class": {k: asdict(v) for k, v in per_class_total.items()},
            "structural": {
                "misses": struct_total.misses,
                "hallucinations": struct_total.hallucinations,
                "misclassifications": struct_total.misclassifications,
                "top_confusions": [
                    {"pred": pt, "gt": gt, "count": n}
                    for pt, gt, n in struct.top_confusions(struct_total, k=5)
                ],
            },
            "mean_latency_sec": sum(latencies) / len(latencies) if latencies else None,
            "total_cost_usd": sum(costs) if costs else None,
            "mean_cost_usd_per_page": (sum(costs) / len(costs)) if costs else None,
            "confidence_present": confidence_present,
        }
    return out
