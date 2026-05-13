"""Threshold sweep for confidence-emitting layout processors.

Given a run dir produced by ``realdoc-bench layout eval`` with a processor
that returns raw per-block confidences (e.g. ``elrnd_l_api`` at ``conf=0.0``),
load the predictions + GT and sweep per-class confidence thresholds to
maximize F1.

This mirrors the methodology of the original ``Extend v2 Benchmark Snapshot``
on Notion: raw model output with the F1-optimal post-hoc threshold per class.

Usage:
    python scripts/sweep_threshold.py \\
        --run-id sanity_elrnd \\
        --processor elrnd_l_api \\
        [--global] [--iou 0.5]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from realdoc_bench.layout.metrics.matching import match_blocks
from realdoc_bench.layout.normalizers.base import (
    BBox,
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from realdoc_bench.layout.normalizers.coco import normalize_coco
from realdoc_bench.shared.io.hf import fetch_dataset

DEFAULT_HF_DATASET = "extend-ai/realdoc-layout"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "datasets" / "layout" / "manifest.csv"


def _block_from_dict(d: dict) -> LayoutBlock:
    bb = d["bbox"]
    return LayoutBlock(
        id=d.get("id") or "",
        block_type=d["block_type"],
        bbox=BBox(x=bb["x"], y=bb["y"], w=bb["w"], h=bb["h"]),
        confidence=float(d.get("confidence", 1.0)),
        text=d.get("text"),
        reading_order=d.get("reading_order"),
    )


def _doc_from_pred_json(path: Path) -> LayoutDocument:
    raw = json.loads(path.read_text())
    pages: list[LayoutPage] = []
    for p in raw.get("pages", []):
        blocks = [_block_from_dict(b) for b in p.get("blocks", [])]
        pages.append(
            LayoutPage(
                page_number=p.get("page_number", 1),
                width=p.get("width", 0),
                height=p.get("height", 0),
                blocks=blocks,
            )
        )
    return LayoutDocument(
        document_id=raw.get("document_id") or path.parent.parent.name,
        source=raw.get("source"),
        pages=pages,
    )


def _load_gt(snapshot: Path, page_id: str) -> LayoutDocument | None:
    ann_path = snapshot / "annotations" / f"{page_id}.json"
    if not ann_path.exists():
        return None
    return normalize_coco(json.loads(ann_path.read_text()), source=page_id)


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _aggregate_at_threshold(
    docs: list[tuple[LayoutDocument, LayoutDocument]],
    *,
    per_class_thresh: dict[str, float] | None = None,
    global_thresh: float | None = None,
    iou_threshold: float = 0.5,
) -> tuple[float, float, float, dict[str, tuple[int, int, int]]]:
    """Filter preds by thresholds, match against GT, return aggregate P/R/F1 and per-class counts."""
    total_tp = total_fp = total_fn = 0
    by_class: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [tp, fp, fn]

    for pred_doc, gt_doc in docs:
        # Flatten all pages → one list each
        pred_blocks: list[LayoutBlock] = []
        for pg in pred_doc.pages:
            for b in pg.blocks:
                threshold = (
                    per_class_thresh.get(b.block_type, 1.01)
                    if per_class_thresh is not None
                    else (global_thresh if global_thresh is not None else 0.0)
                )
                if b.confidence >= threshold:
                    pred_blocks.append(b)
        gt_blocks: list[LayoutBlock] = [b for pg in gt_doc.pages for b in pg.blocks]

        matched, up, ug = match_blocks(
            pred_blocks, gt_blocks, iou_threshold=iou_threshold, type_sensitive=True
        )
        # Only count matches where types agree as TP per class; cross-type matches → FP+FN
        for pi, gi in matched:
            pb = pred_blocks[pi]
            gb = gt_blocks[gi]
            if pb.block_type == gb.block_type:
                by_class[pb.block_type][0] += 1
                total_tp += 1
            else:
                by_class[pb.block_type][1] += 1
                by_class[gb.block_type][2] += 1
                total_fp += 1
                total_fn += 1
        for pi in up:
            by_class[pred_blocks[pi].block_type][1] += 1
            total_fp += 1
        for gi in ug:
            by_class[gt_blocks[gi].block_type][2] += 1
            total_fn += 1

    p, r, f1 = _f1(total_tp, total_fp, total_fn)
    return p, r, f1, {k: tuple(v) for k, v in by_class.items()}


def _per_class_best(
    docs: list[tuple[LayoutDocument, LayoutDocument]],
    *,
    classes: list[str],
    grid: list[float],
    iou_threshold: float,
    baseline: float,
) -> dict[str, float]:
    """For each class, sweep its threshold while holding others at ``baseline``.

    Sweep each class independently around a non-noisy operating point so
    cross-class FPs from one floating class don't pollute the matcher.
    """
    best: dict[str, float] = {c: baseline for c in classes}
    for c in classes:
        scores: list[tuple[float, float]] = []
        for t in grid:
            thresh = {**best, c: t}
            _, _, _, by_cls = _aggregate_at_threshold(
                docs, per_class_thresh=thresh, iou_threshold=iou_threshold
            )
            tp, fp, fn = by_cls.get(c, (0, 0, 0))
            _, _, f1 = _f1(tp, fp, fn)
            scores.append((t, f1))
        scores.sort(key=lambda x: (-x[1], x[0]))
        best[c] = scores[0][0]
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--global-threshold", action="store_true", help="Sweep a single threshold across all classes")
    parser.add_argument(
        "--grid",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80",
        help="Comma-separated confidence thresholds to try",
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    grid = sorted(set([float(x) for x in args.grid.split(",")]))

    snapshot = fetch_dataset(args.hf_dataset)
    manifest_rows = list(csv.DictReader(args.manifest.open()))
    manifest_by_id = {r["pageId"]: r for r in manifest_rows}

    run_root = args.run_dir / args.run_id
    pred_paths = sorted(run_root.glob(f"*/{args.processor}/prediction.json"))
    print(f"Loaded {len(pred_paths)} predictions from {run_root}", file=sys.stderr)

    docs: list[tuple[LayoutDocument, LayoutDocument]] = []
    for path in pred_paths:
        page_id = path.parent.parent.name
        gt = _load_gt(snapshot, page_id)
        if gt is None:
            continue
        pred = _doc_from_pred_json(path)
        docs.append((pred, gt))
    print(f"Pairs with GT: {len(docs)}", file=sys.stderr)
    if not docs:
        print("No valid (pred, gt) pairs to score.", file=sys.stderr)
        return 2

    # Baseline at conf=0.0 (everything kept)
    _, _, baseline_f1, _ = _aggregate_at_threshold(docs, global_thresh=0.0, iou_threshold=args.iou)
    print(f"\nBaseline F1 (conf >= 0.0): {baseline_f1:.4f}")

    # Global sweep
    print("\nGlobal threshold sweep:")
    global_scores: list[tuple[float, float, float, float]] = []
    for t in grid:
        p, r, f1, _ = _aggregate_at_threshold(docs, global_thresh=t, iou_threshold=args.iou)
        global_scores.append((t, p, r, f1))
        print(f"  conf >= {t:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    best_global = max(global_scores, key=lambda x: x[3])
    print(f"\nBest global: conf >= {best_global[0]:.2f}  →  F1={best_global[3]:.4f}")

    if args.global_threshold:
        return 0

    # Per-class sweep
    classes_seen: set[str] = set()
    for pred, _gt in docs:
        for pg in pred.pages:
            for b in pg.blocks:
                classes_seen.add(b.block_type)
    classes_ordered = sorted(classes_seen)
    print(f"\nPer-class sweep over {len(classes_ordered)} classes (others held at global best {best_global[0]:.2f})...")
    best_thresh = _per_class_best(
        docs, classes=classes_ordered, grid=grid, iou_threshold=args.iou, baseline=best_global[0]
    )
    print("\nBest per-class thresholds:")
    for c in classes_ordered:
        print(f"  {c:18s}  {best_thresh[c]:.2f}")

    p, r, f1, by_cls = _aggregate_at_threshold(
        docs, per_class_thresh=best_thresh, iou_threshold=args.iou
    )
    print(f"\nFinal (per-class F1-optimal): P={p:.3f}  R={r:.3f}  F1={f1:.4f}")
    print("\nPer-class final F1:")
    for c in classes_ordered:
        tp, fp, fn = by_cls.get(c, (0, 0, 0))
        _, _, cf = _f1(tp, fp, fn)
        print(f"  {c:18s}  tp={tp:>4} fp={fp:>4} fn={fn:>4}  F1={cf:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
