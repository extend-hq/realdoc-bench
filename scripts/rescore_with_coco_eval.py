"""Re-score a realdoc-bench layout run using the elrnd-style COCO matcher.

Takes the predictions stored under ``runs/<run-id>/<page_id>/<processor>/prediction.json``
and re-runs scoring through :mod:`realdoc_bench.layout.metrics.coco_eval` — the
same greedy by-class IoU matching + per-class F1-optimal threshold sweep that
elrnd's ``scripts/evaluate.py`` uses. The result is directly comparable to
the official ELRND numbers in the Notion benchmark page.

Optional class_remap mirrors the elrnd ``eval.class_remap`` knob (e.g. fold
Caption into Text, Form into Key Value). Watermark is excluded by default.

Usage:
    .venv/bin/python scripts/rescore_with_coco_eval.py \\
        --run-id bench_paddle_full \\
        --processor paddle_structure_v3 \\
        --remap-default 9class
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from realdoc_bench.layout.metrics.coco_eval import aggregate
from realdoc_bench.layout.normalizers.base import (
    BBox,
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from realdoc_bench.layout.normalizers.coco import normalize_coco
from realdoc_bench.shared.io.hf import fetch_dataset

DEFAULT_HF_DATASET = "extend-ai/realdoc-layout"

# 9-class remap matching the Notion doc (Form → Key Value, not Text).
# Applied AFTER realdoc-bench's coco normalizer collapses chunkr's 23 categories
# down to its LayoutBlockType vocab (most of the collapses already happened
# there — the remaining ones below align pred and GT on the final 9 buckets).
REMAP_9CLASS: dict[str, str] = {
    "figure_caption": "text",
    "footnote": "text",
    "formula": "text",
    "list_item": "text",
    "barcode": "figure",
}

# Permissive scheme for models with limited vocab (dots.mocr, Paddle V3 etc):
# everything they can't structurally predict gets folded into Text on BOTH
# pred and GT side. Mirrors the original Notion benchmark's
# "Key-values treated as text here to ensure fair comparison" note,
# generalized for additional classes.
REMAP_PERMISSIVE: dict[str, str] = {
    **REMAP_9CLASS,
    "title": "text",
    "key_value": "text",
    "page_number": "text",
}

SKIP_9CLASS = frozenset({"signature"})  # not part of the 9-class scheme


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--remap-default",
        choices=("none", "9class", "permissive"),
        default="9class",
        help=(
            "Apply built-in class_remap. 9class: the standard benchmark scheme. "
            "permissive: also folds Title/Key Value/Page Number into Text (for "
            "models that structurally can't predict those, like dots.mocr)."
        ),
    )
    parser.add_argument(
        "--remap",
        action="append",
        default=[],
        help="Additional class remaps as 'src=dst'. Stack multiple --remap flags.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Block types to skip entirely. Stack multiple --skip flags.",
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--adjusted", action="store_true", help="Also report adjusted F1 with adjacency merging.")
    parser.add_argument("--gap-threshold", type=float, default=0.02, help="Adjacency gap threshold (fraction of max page dim).")
    args = parser.parse_args()

    remap: dict[str, str] = {}
    if args.remap_default == "9class":
        remap.update(REMAP_9CLASS)
    elif args.remap_default == "permissive":
        remap.update(REMAP_PERMISSIVE)
    for r in args.remap:
        if "=" not in r:
            raise SystemExit(f"--remap must be src=dst, got {r!r}")
        src, dst = r.split("=", 1)
        remap[src.strip()] = dst.strip()

    skip = set(SKIP_9CLASS if args.remap_default == "9class" else ())
    skip.update(args.skip)

    snapshot = fetch_dataset(args.hf_dataset)

    run_root = args.run_dir / args.run_id
    pred_paths = sorted(run_root.glob(f"*/{args.processor}/prediction.json"))
    print(f"Loaded {len(pred_paths)} predictions from {run_root}", file=sys.stderr)

    pairs: list[tuple[LayoutDocument, LayoutDocument]] = []
    for path in pred_paths:
        page_id = path.parent.parent.name
        gt = _load_gt(snapshot, page_id)
        if gt is None:
            continue
        pred = _doc_from_pred_json(path)
        pairs.append((pred, gt))
    print(f"Pairs with GT: {len(pairs)}", file=sys.stderr)
    if not pairs:
        print("No valid (pred, gt) pairs to score.", file=sys.stderr)
        return 2

    print(f"\nClass remap: {remap}", file=sys.stderr)
    print(f"Skip: {sorted(skip)}", file=sys.stderr)
    print(f"IoU threshold: {args.iou}\n", file=sys.stderr)

    result = aggregate(
        pairs,
        iou_threshold=args.iou,
        class_remap=remap,
        skip_classes=frozenset(skip),
        with_adjusted=args.adjusted,
        gap_threshold=args.gap_threshold,
    )

    # Strict per-class table sorted by F1 descending
    rows = sorted(result.per_class.values(), key=lambda r: -r.f1)
    print(f"=== Strict F1 (IoU {args.iou:.2f}) ===")
    print(f"{'class':24s}  {'F1':>6s}  {'P':>6s}  {'R':>6s}  {'conf':>5s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  {'N_GT':>5s}")
    print("-" * 80)
    for r in rows:
        if r.num_gt == 0:
            continue
        print(
            f"{r.cls:24s}  {r.f1:>6.3f}  {r.precision:>6.3f}  {r.recall:>6.3f}  "
            f"{r.best_conf:>5.2f}  {r.tp:>5d}  {r.fp:>5d}  {r.fn:>5d}  {r.num_gt:>5d}"
        )
    print("-" * 80)
    print(f"{'MACRO (≥1 GT)':24s}  {result.macro_f1:>6.3f}  {result.macro_precision:>6.3f}  {result.macro_recall:>6.3f}")

    if args.adjusted and result.adjusted_per_class is not None:
        print(f"\n=== Adjusted F1 (with adjacency merging, gap ≤ {args.gap_threshold:.2%} of max page dim) ===")
        print(f"  Splits resolved: {result.splits_resolved}, Merges resolved: {result.merges_resolved}")
        adj_rows = sorted(result.adjusted_per_class.values(), key=lambda r: -r.f1)
        print(f"{'class':24s}  {'F1':>6s}  {'P':>6s}  {'R':>6s}  {'conf':>5s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  {'N_GT':>5s}")
        print("-" * 80)
        for r in adj_rows:
            if r.num_gt == 0:
                continue
            print(
                f"{r.cls:24s}  {r.f1:>6.3f}  {r.precision:>6.3f}  {r.recall:>6.3f}  "
                f"{r.best_conf:>5.2f}  {r.tp:>5d}  {r.fp:>5d}  {r.fn:>5d}  {r.num_gt:>5d}"
            )
        print("-" * 80)
        print(
            f"{'MACRO (≥1 GT)':24s}  {result.adjusted_macro_f1:>6.3f}  "
            f"{result.adjusted_macro_precision:>6.3f}  {result.adjusted_macro_recall:>6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
