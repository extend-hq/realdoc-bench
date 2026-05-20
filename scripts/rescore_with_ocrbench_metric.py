"""Re-score a realdoc-bench layout run using **ocr-bench's** metric.

Goal: produce numbers directly comparable to the original Extend v2 Benchmark
Snapshot (which used ocr-bench's ``metrics/layout.py``) without re-running any
evals. We just re-read the existing ``prediction.json`` files and pass them
through ocr-bench's matcher.

Key differences vs ``rescore_with_coco_eval.py`` (realdoc-bench's own scorer):
  - **Matching**: Hungarian 1:1 (ocr-bench) vs greedy by-class IoU (realdoc-bench).
  - **Confidence**: ocr-bench treats every prediction as positive; there is no
    per-class confidence-threshold sweep. For models that emit conf=1.0
    everywhere (paddle_structure, dots.mocr) the two scorers agree; for ELRND
    (rich conf distribution) ocr-bench will count low-conf garbage as FPs.
  - **Adjusted F1**: ocr-bench's adjacency merger handles both splits and
    merges, returning a single (strict, adjusted) pair per doc.

Usage:
    .venv/bin/python scripts/rescore_with_ocrbench_metric.py \\
        --ocr-bench /Users/ameyajoshi/Projects/ocr-bench \\
        --run-id   dots_full_1500_baseten \\
        --processor dots_ocr \\
        --remap-default notion9
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# realdoc-bench imports (for loading prediction.json + dataset)
from realdoc_bench.shared.io.hf import fetch_dataset

DEFAULT_HF_DATASET = "extend-ai/realdoc-layout"

# Notion 9-class scheme: text / title / section_header / table / figure /
# page_header / page_footer / page_number / key_value. Caption/footnote/
# list_item/formula → text; bar/QR → figure; form → key_value; watermark/
# signature excluded by being dropped before scoring.
REMAP_NOTION9: dict[str, str] = {
    "figure_caption": "text",
    "footnote": "text",
    "list_item": "text",
    "formula": "text",
    "table_caption": "text",
    "table_footer": "text",
    "table_cell": "table",
    "figure_footnote": "text",
    "barcode": "figure",
}
SKIP_NOTION9 = {"watermark", "signature"}

# "Permissive" — for models whose taxonomy structurally lacks Title / KeyValue /
# PageNumber (e.g. dots.mocr); folds those into Text so the model isn't
# penalized for classes it can't predict.
REMAP_PERMISSIVE: dict[str, str] = {
    **REMAP_NOTION9,
    "title": "text",
    "key_value": "text",
    "page_number": "text",
}

# The "extend-v2" 8-class vocab — matches ocr-bench's chunkr_human GT
# normalizer at ``ocr-bench/models/chunkr_human/model.py``. Applied to BOTH
# pred and GT so taxonomy disagreements (e.g. ELRND emits ``key_value`` while
# GT calls it ``text``) don't false-penalize. The surviving classes are:
# text, title, section_header, page_header, page_footer, page_number, figure, table.
REMAP_EXTEND_V2: dict[str, str] = {
    "key_value": "text",
    "list_item": "text",
    "formula": "text",
    "footnote": "text",
    "figure_caption": "text",
    "table_caption": "text",
    "table_footer": "text",
    "table_cell": "table",
    "figure_footnote": "text",
    "barcode": "figure",
    "signature": "text",
}
SKIP_EXTEND_V2 = {"watermark"}


def _doc_blocks_to_ocrbench(pred_doc: dict, ocrbench_layout_mod, ocrbench_bbox_mod) -> tuple[list, int, int]:
    """Convert a realdoc-bench prediction.json (or coco-derived GT) into a flat
    list of ocr-bench ``LayoutBlock``s plus (page_width, page_height)."""
    LayoutBlock = ocrbench_layout_mod.LayoutBlock
    BBox = ocrbench_bbox_mod.BBox

    page_w = page_h = 0
    out_blocks = []
    for pg in pred_doc.get("pages", []):
        page_w = max(page_w, int(pg.get("width") or 0))
        page_h = max(page_h, int(pg.get("height") or 0))
        for b in pg.get("blocks", []):
            bb = b.get("bbox")
            if not bb:
                continue
            block = LayoutBlock(
                id=b.get("id"),
                block_type=b["block_type"],
                bbox=BBox(x=int(bb["x"]), y=int(bb["y"]), w=int(bb["w"]), h=int(bb["h"])),
                confidence=float(b.get("confidence", 1.0)),
                text=b.get("text"),
            )
            out_blocks.append(block)
    return out_blocks, page_w, page_h


def _load_gt(snapshot_dir: Path, page_id: str, ocrbench_layout_mod, ocrbench_bbox_mod) -> tuple[list, int, int] | None:
    """Load the COCO GT for a page_id from the staged dataset and convert to
    ocr-bench LayoutBlocks. Returns None if blobs are missing."""
    ann_path = snapshot_dir / "annotations" / f"{page_id}.json"
    if not ann_path.exists():
        return None
    with ann_path.open() as f:
        coco = json.load(f)

    # Realdoc-bench's normalizers/coco.py converts the COCO categories to the
    # LayoutBlockType vocab — reuse it so this script stays in sync.
    from realdoc_bench.layout.normalizers.coco import normalize_coco

    gt_doc = normalize_coco(coco, source=f"{page_id}.png")
    # gt_doc is a realdoc-bench LayoutDocument — schema-identical to ocr-bench's
    # but a different class. Serialize → dict → convert.
    gt_dict = gt_doc.model_dump()
    return _doc_blocks_to_ocrbench(gt_dict, ocrbench_layout_mod, ocrbench_bbox_mod)


def _remap_blocks(blocks: list, remap: dict[str, str], skip: set[str]) -> list:
    """Apply class remap (in-place safe — uses model_copy) and drop skipped types."""
    out = []
    for b in blocks:
        if b.block_type in skip:
            continue
        new_type = remap.get(b.block_type, b.block_type)
        if new_type != b.block_type:
            b = b.model_copy(update={"block_type": new_type})
        out.append(b)
    return out


def _filter_conf(blocks: list, threshold: float) -> list:
    """Optional pre-filter — only keeps preds with confidence ≥ threshold. Used
    to make confidence-rich models (ELRND) comparable; default off (=0)."""
    if threshold <= 0:
        return blocks
    return [b for b in blocks if (b.confidence or 0.0) >= threshold]


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rescore a realdoc-bench layout run with ocr-bench's metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ocr-bench", required=True, type=Path,
                        help="Path to the ocr-bench repo (so we can import metrics.layout).")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--gap-threshold", type=float, default=0.02)
    parser.add_argument("--remap-default", choices=["none", "notion9", "permissive", "ocrbench", "extend_v2"], default="notion9",
                        help="Built-in class remap. notion9 = 9-class Notion scheme. "
                        "ocrbench = use ocr-bench's full eval pipeline (EXTENDED_MERGE + apply_ignore_filter)."
                        " Lands at ~10-11 classes after dropping footnote/figure_caption/table_caption/figure_footnote/table_footer. "
                        "extend_v2 = ocr-bench's chunkr_human GT 8-class collapse "
                        "(key_value/list_item/formula/footnote/figure_caption/signature/etc → text). "
                        "This is what the original ocr-bench ELRND eval used.")
    parser.add_argument("--remap", action="append", default=[],
                        help="Extra remap pairs as 'src=dst'. Stack multiple --remap.")
    parser.add_argument("--skip", action="append", default=[],
                        help="Extra block_types to skip entirely. Stack multiple --skip.")
    parser.add_argument("--conf-threshold", type=float, default=0.0,
                        help="Pre-filter preds by confidence (≥ threshold). Default 0 = ocr-bench default.")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    args = parser.parse_args()

    # Wire ocr-bench into sys.path so we can import its metric module.
    sys.path.insert(0, str(args.ocr_bench.resolve()))
    from metrics.layout import match_blocks_with_adjacency  # type: ignore
    from models.normalized import layout as ocrbench_layout_mod  # type: ignore
    from helpers import boxes as ocrbench_bbox_mod  # type: ignore
    # ocrbench-native pipeline helpers (only needed for --remap-default ocrbench)
    from metrics.label_merge import apply_merge_profile  # type: ignore
    from metrics.ignore import apply_ignore_filter, IGNORABLE_BLOCK_TYPES  # type: ignore

    use_ocrbench_pipeline = (args.remap_default == "ocrbench")

    remap: dict[str, str] = {}
    if args.remap_default == "notion9":
        remap.update(REMAP_NOTION9)
    elif args.remap_default == "permissive":
        remap.update(REMAP_PERMISSIVE)
    elif args.remap_default == "extend_v2":
        remap.update(REMAP_EXTEND_V2)
    # ocrbench: skip the dict-based remap — we'll run apply_merge_profile("extended") + apply_ignore_filter
    # inside the per-doc loop so the implementation matches ocr-bench's own benchmark.py path verbatim.
    for r in args.remap:
        if "=" not in r:
            raise SystemExit(f"--remap must be src=dst, got {r!r}")
        s, d = r.split("=", 1)
        remap[s.strip()] = d.strip()

    skip: set[str] = set()
    if args.remap_default == "notion9":
        skip.update(SKIP_NOTION9)
    elif args.remap_default == "permissive":
        skip.update(SKIP_NOTION9 - {"signature"})  # permissive keeps signature out too
    elif args.remap_default == "extend_v2":
        skip.update(SKIP_EXTEND_V2)
    skip.update(args.skip)

    snapshot = fetch_dataset(args.hf_dataset)
    run_root = args.run_dir / args.run_id
    pred_paths = sorted(run_root.glob(f"*/{args.processor}/prediction.json"))
    print(f"Loaded {len(pred_paths)} predictions from {run_root}", file=sys.stderr)
    print(f"\nMetric: ocr-bench match_blocks_with_adjacency (IoU={args.iou:.2f}, gap={args.gap_threshold})",
          file=sys.stderr)
    print(f"Remap: {remap}", file=sys.stderr)
    print(f"Skip: {sorted(skip)}", file=sys.stderr)
    if args.conf_threshold > 0:
        print(f"Conf pre-filter: keep preds with confidence ≥ {args.conf_threshold}", file=sys.stderr)
    print()

    # Aggregate strict + adjusted TP/FP/FN, both micro (overall) and by class.
    strict_tot = [0, 0, 0]   # tp, fp, fn
    adj_tot = [0, 0, 0]
    strict_by_cls: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    adj_by_cls: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    gt_cls_counts: dict[str, int] = defaultdict(int)
    splits_total = merges_total = 0
    n_scored = 0

    for path in pred_paths:
        page_id = path.parent.parent.name
        with path.open() as f:
            pred_doc = json.load(f)
        pred_blocks, pred_pw, pred_ph = _doc_blocks_to_ocrbench(pred_doc, ocrbench_layout_mod, ocrbench_bbox_mod)
        gt = _load_gt(snapshot, page_id, ocrbench_layout_mod, ocrbench_bbox_mod)
        if gt is None:
            continue
        gt_blocks, gt_pw, gt_ph = gt

        pred_blocks = _filter_conf(pred_blocks, args.conf_threshold)
        if use_ocrbench_pipeline:
            # ocr-bench's own benchmark.py flow: EXTENDED_MERGE class collapse, then
            # apply_ignore_filter in "exclude_matching" mode (drops the IGNORABLE
            # types entirely so they don't enter Hungarian matching).
            pred_blocks = apply_merge_profile(pred_blocks, "extended")
            gt_blocks = apply_merge_profile(gt_blocks, "extended")
            pred_blocks, _ = apply_ignore_filter(pred_blocks, "exclude_matching", set(IGNORABLE_BLOCK_TYPES))
            gt_blocks, _ = apply_ignore_filter(gt_blocks, "exclude_matching", set(IGNORABLE_BLOCK_TYPES))
        else:
            pred_blocks = _remap_blocks(pred_blocks, remap, skip)
            gt_blocks = _remap_blocks(gt_blocks, remap, skip)

        page_w = max(pred_pw, gt_pw, 1000)
        page_h = max(pred_ph, gt_ph, 1000)

        # Overall (type-sensitive) ─ produces both strict + adjusted in one pass
        result = match_blocks_with_adjacency(
            pred_blocks, gt_blocks, page_w, page_h,
            iou_threshold=args.iou, gap_threshold=args.gap_threshold, type_sensitive=True,
        )
        strict_tot[0] += result.strict_tp; strict_tot[1] += result.strict_fp; strict_tot[2] += result.strict_fn
        adj_tot[0] += result.adjusted_tp; adj_tot[1] += result.adjusted_fp; adj_tot[2] += result.adjusted_fn
        splits_total += result.splits_resolved
        merges_total += result.merges_resolved

        # Per-class: count GTs (for sorting / context) and re-run match_blocks_with_adjacency
        # on the per-class subsets (type_sensitive irrelevant when subsets are single-class).
        cls_present = {b.block_type for b in pred_blocks} | {b.block_type for b in gt_blocks}
        for cls in cls_present:
            p_sub = [b for b in pred_blocks if b.block_type == cls]
            g_sub = [b for b in gt_blocks if b.block_type == cls]
            gt_cls_counts[cls] += len(g_sub)
            if not p_sub and not g_sub:
                continue
            r_sub = match_blocks_with_adjacency(
                p_sub, g_sub, page_w, page_h,
                iou_threshold=args.iou, gap_threshold=args.gap_threshold, type_sensitive=False,
            )
            strict_by_cls[cls][0] += r_sub.strict_tp; strict_by_cls[cls][1] += r_sub.strict_fp; strict_by_cls[cls][2] += r_sub.strict_fn
            adj_by_cls[cls][0] += r_sub.adjusted_tp; adj_by_cls[cls][1] += r_sub.adjusted_fp; adj_by_cls[cls][2] += r_sub.adjusted_fn
        n_scored += 1

    if n_scored == 0:
        print("No (pred, gt) pairs scored.", file=sys.stderr)
        return 2

    def _print_table(title: str, totals: list[int], by_cls: dict[str, list[int]]) -> None:
        p, r, f1 = _f1(*totals)
        rows = []
        for cls, (tp, fp, fn) in by_cls.items():
            if (tp + fn) == 0:
                continue
            cp, cr, cf1 = _f1(tp, fp, fn)
            rows.append((cls, cf1, cp, cr, tp, fp, fn, gt_cls_counts[cls]))
        rows.sort(key=lambda r: -r[1])

        print(f"=== {title} ===")
        print(f"{'class':24s}  {'F1':>6s}  {'P':>6s}  {'R':>6s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}  {'N_GT':>6s}")
        print("-" * 80)
        for row in rows:
            print(f"{row[0]:24s}  {row[1]:>6.3f}  {row[2]:>6.3f}  {row[3]:>6.3f}  "
                  f"{row[4]:>6d}  {row[5]:>6d}  {row[6]:>6d}  {row[7]:>6d}")
        # Macro = unweighted mean of per-class F1 over classes with ≥1 GT
        if rows:
            macro_f1 = sum(r[1] for r in rows) / len(rows)
            macro_p = sum(r[2] for r in rows) / len(rows)
            macro_r = sum(r[3] for r in rows) / len(rows)
        else:
            macro_f1 = macro_p = macro_r = 0.0
        print("-" * 80)
        print(f"{'MICRO (overall)':24s}  {f1:>6.3f}  {p:>6.3f}  {r:>6.3f}  "
              f"{totals[0]:>6d}  {totals[1]:>6d}  {totals[2]:>6d}")
        print(f"{'MACRO (≥1 GT)':24s}  {macro_f1:>6.3f}  {macro_p:>6.3f}  {macro_r:>6.3f}")
        print()

    _print_table("Strict F1 (ocr-bench Hungarian, no conf sweep)", strict_tot, strict_by_cls)
    _print_table(
        f"Adjusted F1 (with adjacency merging) — splits resolved {splits_total}, merges resolved {merges_total}",
        adj_tot, adj_by_cls,
    )
    print(f"Scored {n_scored} pages.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
