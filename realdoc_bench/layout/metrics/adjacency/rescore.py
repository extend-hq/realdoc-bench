"""Re-score an existing layout run with the adjacency metric.

Reads the ``prediction.json`` files a layout run already wrote under
``runs/<run_id>/<page_id>/<processor>/`` and scores them with the Hungarian
1:1 matcher + adjacency split/merge recovery in this package — instead of
realdoc-bench's own ``coco_eval`` / ``f1`` scorer.

Usage::

    python -m realdoc_bench.layout.metrics.adjacency.rescore \\
        --run-id v500 --processor extend_v2_0_0_beta \\
        --merge-profile extended --ignore-mode exclude_matching

If ``--processor`` is omitted every processor subdirectory under the run is
scored. Ground truth is loaded through the standard dataset loader, so the
same ``HF_TOKEN`` / dataset-env requirements as ``realdoc-bench layout eval``
apply.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from realdoc_bench.layout.data.loader import load
from realdoc_bench.layout.metrics.adjacency.aggregate import AggregateResult, MetricBreakdown, score_pairs
from realdoc_bench.layout.metrics.adjacency.ignore import IGNORABLE_BLOCK_TYPES, apply_ignore_filter
from realdoc_bench.layout.metrics.adjacency.label_merge import apply_merge_profile
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


def _prepare(
    blocks: list[LayoutBlock],
    merge_profile: str,
    ignore_mode: str,
) -> list[LayoutBlock]:
    """Apply the class-merge profile then the ignore filter (in that order)."""
    if merge_profile != "none":
        blocks = apply_merge_profile(blocks, merge_profile)  # type: ignore[arg-type]
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
    merge_profile: str,
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

        pred_blocks = _prepare(list(pred_doc.blocks), merge_profile, ignore_mode)
        gt_blocks = _prepare(list(sample.ground_truth.blocks), merge_profile, ignore_mode)

        pred_w, pred_h = _page_dims(pred_doc)
        gt_w, gt_h = _page_dims(sample.ground_truth)
        page_w = max(pred_w, gt_w, 1)
        page_h = max(pred_h, gt_h, 1)

        pairs.append((pred_blocks, gt_blocks, page_w, page_h))

    if not pairs:
        return None
    return score_pairs(pairs, iou_threshold=iou_threshold, gap_threshold=gap_threshold)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-score a layout run with the Hungarian + adjacency metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", required=True, help="Run directory name under --run-dir.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--processor",
        action="append",
        default=[],
        help="Processor to score (repeatable). Default: every processor in the run.",
    )
    parser.add_argument(
        "--merge-profile",
        choices=["none", "core5", "extended", "full"],
        default="none",
        help="Class-merge profile applied to both predictions and GT.",
    )
    parser.add_argument(
        "--ignore-mode",
        choices=["disabled", "exclude_matching", "include_exclude_metrics"],
        default="disabled",
        help="Ignore-filter mode for header/footer/caption-style block types.",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold.")
    parser.add_argument("--gap-threshold", type=float, default=0.02, help="Adjacency gap fraction.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N samples.")
    parser.add_argument("--split", action="append", default=[], help="Restrict to dataset split(s).")
    args = parser.parse_args(argv)

    run_root = args.run_dir / args.run_id
    if not run_root.is_dir():
        print(f"error: run directory not found: {run_root}", file=sys.stderr)
        return 2

    processors = args.processor or _discover_processors(run_root)
    if not processors:
        print(f"error: no processors with prediction.json under {run_root}", file=sys.stderr)
        return 2

    samples = list(load(splits=args.split or None, limit=args.limit))
    print(
        f"Run {args.run_id}: {len(samples)} GT samples, processors {processors}\n"
        f"Metric: match_blocks_with_adjacency "
        f"(IoU={args.iou:.2f}, gap={args.gap_threshold}, "
        f"merge_profile={args.merge_profile}, ignore_mode={args.ignore_mode})\n",
        file=sys.stderr,
    )

    exit_code = 0
    for proc in processors:
        result = rescore_processor(
            run_root,
            proc,
            samples,
            merge_profile=args.merge_profile,
            ignore_mode=args.ignore_mode,
            iou_threshold=args.iou,
            gap_threshold=args.gap_threshold,
        )
        if result is None:
            print(f"### {proc}: no predictions found — skipped\n", file=sys.stderr)
            exit_code = 1
            continue
        print(f"\n##### {proc} — {result.n_pages} pages "
              f"(splits resolved {result.splits_resolved}, merges resolved {result.merges_resolved})")
        _print_breakdown("Strict F1 (Hungarian 1:1, no confidence sweep)", result.strict)
        _print_breakdown("Adjusted F1 (with adjacency split/merge recovery)", result.adjusted)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
