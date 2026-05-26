"""Re-score an existing layout run with the adjacency metric.

Reads the ``prediction.json`` files a layout run already wrote under
``runs/<run_id>/<page_id>/<processor>/`` and scores them with the Hungarian
1:1 matcher + adjacency split/merge recovery in this package — instead of
realdoc-bench's own ``coco_eval`` / ``f1`` scorer.

Exposed as the ``realdoc-bench layout rescore`` CLI command::

    realdoc-bench layout rescore --run-id v500 --processor extend_v2_0_0 \\
        --ignore-mode exclude_matching

If ``--processor`` is omitted every processor subdirectory under the run is
scored. Ground truth is loaded through the standard dataset loader, so the
same dataset-env requirements as ``realdoc-bench layout eval`` apply (the CLI
auto-loads ``.env`` / ``.env.local``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from realdoc_bench.layout.data.loader import load
from realdoc_bench.layout.metrics.adjacency.aggregate import AggregateResult, MetricBreakdown, score_pairs
from realdoc_bench.layout.metrics.adjacency.ignore import IGNORABLE_BLOCK_TYPES, apply_ignore_filter
from realdoc_bench.layout.normalizers.base import LayoutBlock, LayoutDocument


def _discover_processors(run_root: Path) -> list[str]:
    """Every processor subdirectory present somewhere under ``run_root``."""
    procs: set[str] = set()
    for page_dir in run_root.iterdir():
        if not page_dir.is_dir():
            continue
        for proc_dir in page_dir.iterdir():
            if proc_dir.is_dir() and (proc_dir / "prediction.json").exists():
                procs.add(proc_dir.name)
    return sorted(procs)


def _page_dims(doc: LayoutDocument) -> tuple[int, int]:
    """Largest (width, height) across a document's pages (0 if unknown)."""
    w = max((p.width or 0 for p in doc.pages), default=0)
    h = max((p.height or 0 for p in doc.pages), default=0)
    return w, h


def _prepare(blocks: list[LayoutBlock], ignore_mode: str) -> list[LayoutBlock]:
    """Apply the optional ignore filter (no class-merge step; the public
    9-class taxonomy is already uniform)."""
    if ignore_mode != "disabled":
        blocks, _ = apply_ignore_filter(blocks, ignore_mode, set(IGNORABLE_BLOCK_TYPES))  # type: ignore[arg-type]
    return blocks


def _print_breakdown(title: str, bd: MetricBreakdown) -> None:
    print(f"=== {title} ===")
    print(
        f"{'class':22s}  {'F1':>7s}  {'P':>7s}  {'R':>7s}  "
        f"{'TP':>7s}  {'FP':>7s}  {'FN':>7s}  {'N_GT':>7s}"
    )
    print("-" * 78)
    for cls in sorted(bd.per_class.values(), key=lambda c: -c.f1):
        print(
            f"{cls.block_type:22s}  {cls.f1:>7.3f}  {cls.precision:>7.3f}  {cls.recall:>7.3f}  "
            f"{cls.tp:>7d}  {cls.fp:>7d}  {cls.fn:>7d}  {cls.n_gt:>7d}"
        )
    print("-" * 78)
    print(
        f"{'MICRO (overall)':22s}  {bd.micro_f1:>7.3f}  {bd.micro_precision:>7.3f}  "
        f"{bd.micro_recall:>7.3f}  {bd.tp:>7d}  {bd.fp:>7d}  {bd.fn:>7d}"
    )
    print(
        f"{'MACRO (>=1 GT)':22s}  {bd.macro_f1:>7.3f}  {bd.macro_precision:>7.3f}  "
        f"{bd.macro_recall:>7.3f}"
    )
    print()


def rescore_processor(
    run_root: Path,
    processor: str,
    samples: list,
    *,
    ignore_mode: str,
    iou_threshold: float,
    gap_threshold: float,
) -> AggregateResult | None:
    """Score one processor's predictions for every sample that has one."""
    pairs = []
    for sample in samples:
        pred_path = run_root / sample.page_id / processor / "prediction.json"
        if not pred_path.exists():
            continue
        pred_doc = LayoutDocument.model_validate_json(pred_path.read_text())

        pred_blocks = _prepare(list(pred_doc.blocks), ignore_mode)
        gt_blocks = _prepare(list(sample.ground_truth.blocks), ignore_mode)

        pred_w, pred_h = _page_dims(pred_doc)
        gt_w, gt_h = _page_dims(sample.ground_truth)
        page_w = max(pred_w, gt_w, 1)
        page_h = max(pred_h, gt_h, 1)

        pairs.append((pred_blocks, gt_blocks, page_w, page_h))

    if not pairs:
        return None
    return score_pairs(pairs, iou_threshold=iou_threshold, gap_threshold=gap_threshold)


IGNORE_MODES = ("disabled", "exclude_matching", "include_exclude_metrics")


def run_rescore(
    *,
    run_dir: Path,
    run_id: str,
    processors: list[str],
    limit: int | None = None,
    ignore_mode: str = "disabled",
    iou_threshold: float = 0.5,
    gap_threshold: float = 0.02,
) -> int:
    """Re-score a layout run with the Hungarian + adjacency metric and print the
    per-processor breakdown.

    ``processors`` empty → every processor with a ``prediction.json`` under the
    run. Returns the number of processors that had no predictions (0 = all
    scored). Raises ``FileNotFoundError`` (missing run dir) or ``ValueError``
    (bad choice / no processors) on invalid input.
    """
    if ignore_mode not in IGNORE_MODES:
        raise ValueError(f"ignore_mode must be one of {IGNORE_MODES}, got {ignore_mode!r}")

    run_root = run_dir / run_id
    if not run_root.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_root}")

    procs = processors or _discover_processors(run_root)
    if not procs:
        raise ValueError(f"no processors with prediction.json under {run_root}")

    samples = list(load(limit=limit))
    print(
        f"Run {run_id}: {len(samples)} GT samples, processors {procs}\n"
        f"Metric: match_blocks_with_adjacency "
        f"(IoU={iou_threshold:.2f}, gap={gap_threshold}, "
        f"ignore_mode={ignore_mode})\n",
        file=sys.stderr,
    )

    missing = 0
    for proc in procs:
        result = rescore_processor(
            run_root,
            proc,
            samples,
            ignore_mode=ignore_mode,
            iou_threshold=iou_threshold,
            gap_threshold=gap_threshold,
        )
        if result is None:
            print(f"### {proc}: no predictions found — skipped\n", file=sys.stderr)
            missing += 1
            continue
        print(f"\n##### {proc} — {result.n_pages} pages "
              f"(splits resolved {result.splits_resolved}, merges resolved {result.merges_resolved})")
        _print_breakdown("Strict F1 (Hungarian 1:1, no confidence sweep)", result.strict)
        _print_breakdown("Adjusted F1 (with adjacency split/merge recovery)", result.adjusted)

    return missing
