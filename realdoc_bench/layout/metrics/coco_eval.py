"""COCO-style layout evaluation with per-class F1-optimal confidence sweep.

Ported from ``elrnd/src/evaluation/coco_eval.py``. Two pieces:

- :func:`compute_tp_flags` — greedy by-class IoU matching at threshold 0.5.
  Sort predictions by confidence descending; each is a TP iff there is an
  unmatched GT with the same class and IoU ≥ threshold.
- :func:`compute_best_f1_per_class` — sweep cumulatively over each class's
  (score, is_tp) tuples (sorted by descending score) to find the threshold
  that maximizes per-class F1. Mirrors ``elrnd/src/evaluation/coco_eval.py``
  ``_compute_best_f1_per_class`` exactly.

This is the same methodology used to score the production ELRND
checkpoints. Re-using it across all realdoc-bench processors keeps
cross-provider comparisons apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from realdoc_bench.layout.normalizers.base import BBox, LayoutBlock, LayoutDocument


@dataclass(frozen=True)
class PerClassResult:
    cls: str
    precision: float
    recall: float
    f1: float
    best_conf: float
    tp: int
    fp: int
    fn: int
    num_gt: int


@dataclass(frozen=True)
class AggregateResult:
    per_class: dict[str, PerClassResult]
    macro_precision: float
    macro_recall: float
    macro_f1: float


def _iou(a: BBox, b: BBox) -> float:
    """Standard pixel-IoU. Matches realdoc-bench's BBox.iou semantics."""
    return a.iou(b)


def compute_tp_flags(
    preds: list[LayoutBlock],
    gts: list[LayoutBlock],
    *,
    iou_threshold: float = 0.5,
) -> list[bool]:
    """Greedy by-class IoU matcher. Returns one bool per pred.

    Algorithm — identical to ``elrnd._compute_tp_flags``:

    1. Sort predictions by confidence descending.
    2. For each pred (in confidence order):
       a. Find the unmatched GT of the same class with the highest IoU
          (and IoU ≥ ``iou_threshold``).
       b. If found, mark that GT used and the pred as TP.
    3. Unmatched preds are FP; unmatched GTs are FN.
    """
    n = len(preds)
    m = len(gts)
    tp_flags = [False] * n
    if n == 0 or m == 0:
        return tp_flags

    order = sorted(range(n), key=lambda i: -float(preds[i].confidence))
    gt_used = [False] * m

    for pi in order:
        pb = preds[pi]
        if pb.bbox is None:
            continue
        best_iou = iou_threshold
        best_gi = -1
        for gi in range(m):
            if gt_used[gi]:
                continue
            gb = gts[gi]
            if gb.block_type != pb.block_type or gb.bbox is None:
                continue
            i = _iou(pb.bbox, gb.bbox)
            if i >= best_iou:
                best_iou = i
                best_gi = gi
        if best_gi >= 0:
            gt_used[best_gi] = True
            tp_flags[pi] = True
    return tp_flags


def compute_best_f1_per_class(
    pred_records: list[tuple[float, bool, str]],
    gt_counts: dict[str, int],
) -> dict[str, PerClassResult]:
    """Per-class confidence sweep for F1-optimal operating point.

    Args:
        pred_records: list of ``(score, is_tp, class_name)`` for every pred
            across the eval set, after :func:`compute_tp_flags` has run.
        gt_counts: number of GT instances per class across the eval set.

    Algorithm — identical to ``elrnd._compute_best_f1_per_class``:

    For each class, sort that class's predictions by confidence descending
    and cumsum TP/FP. At each step, compute P=TP/(TP+FP), R=TP/N_gt,
    F1=2PR/(P+R). Track the maximum F1 and the score at which it was hit.

    Returns a per-class :class:`PerClassResult`.
    """
    by_cls: dict[str, list[tuple[float, bool]]] = {}
    for score, is_tp, cls in pred_records:
        by_cls.setdefault(cls, []).append((score, is_tp))
    classes = sorted(set(by_cls.keys()) | set(gt_counts.keys()))

    out: dict[str, PerClassResult] = {}
    for cls in classes:
        rows = by_cls.get(cls, [])
        num_gt = int(gt_counts.get(cls, 0))
        if not rows or num_gt == 0:
            out[cls] = PerClassResult(
                cls=cls, precision=0.0, recall=0.0, f1=0.0, best_conf=0.5,
                tp=0, fp=0, fn=num_gt, num_gt=num_gt,
            )
            continue

        rows.sort(key=lambda r: -r[0])
        tp_cum = fp_cum = 0
        best_f1 = best_p = best_r = 0.0
        best_conf = 0.5
        best_tp = best_fp = 0
        for score, is_tp in rows:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            denom = tp_cum + fp_cum
            p = tp_cum / denom if denom > 0 else 0.0
            r = tp_cum / num_gt if num_gt > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_p, best_r = p, r
                best_conf = float(score)
                best_tp, best_fp = tp_cum, fp_cum
        fn = num_gt - best_tp
        out[cls] = PerClassResult(
            cls=cls, precision=best_p, recall=best_r, f1=best_f1,
            best_conf=best_conf, tp=best_tp, fp=best_fp, fn=max(fn, 0), num_gt=num_gt,
        )
    return out


def aggregate(
    pred_docs: Iterable[tuple[LayoutDocument, LayoutDocument]],
    *,
    iou_threshold: float = 0.5,
    class_remap: dict[str, str] | None = None,
    skip_classes: frozenset[str] = frozenset(),
) -> AggregateResult:
    """Run the full evaluate.py-style pipeline against an iterable of
    ``(pred, gt)`` document pairs.

    Args:
        pred_docs: stream of paired :class:`LayoutDocument`s. Both must be
            normalized to the same vocabulary (e.g. via the layout
            normalizers).
        iou_threshold: IoU cutoff for greedy TP matching. Default 0.5.
        class_remap: optional name→name remap applied to both pred and GT
            block types *before* matching (mirrors evaluate.py's
            ``eval.class_remap``).
        skip_classes: classes to drop from both pred and GT before matching
            (e.g. ``{"watermark"}``).
    """
    remap = class_remap or {}
    skip = set(skip_classes)
    pred_records: list[tuple[float, bool, str]] = []
    gt_counts: dict[str, int] = {}

    def _remap(block: LayoutBlock) -> LayoutBlock:
        new_type = remap.get(block.block_type, block.block_type)
        if new_type == block.block_type:
            return block
        return block.model_copy(update={"block_type": new_type})

    for pred_doc, gt_doc in pred_docs:
        # Flatten across pages
        pred_blocks: list[LayoutBlock] = []
        for pg in pred_doc.pages:
            for b in pg.blocks:
                rb = _remap(b)
                if rb.block_type in skip:
                    continue
                pred_blocks.append(rb)
        gt_blocks: list[LayoutBlock] = []
        for pg in gt_doc.pages:
            for b in pg.blocks:
                rb = _remap(b)
                if rb.block_type in skip:
                    continue
                gt_blocks.append(rb)

        tp_flags = compute_tp_flags(pred_blocks, gt_blocks, iou_threshold=iou_threshold)
        for b, is_tp in zip(pred_blocks, tp_flags):
            pred_records.append((float(b.confidence), is_tp, b.block_type))
        for b in gt_blocks:
            gt_counts[b.block_type] = gt_counts.get(b.block_type, 0) + 1

    per_class = compute_best_f1_per_class(pred_records, gt_counts)

    # Macro means across classes with ≥1 GT
    keep = [r for r in per_class.values() if r.num_gt > 0]
    if keep:
        macro_p = sum(r.precision for r in keep) / len(keep)
        macro_r = sum(r.recall for r in keep) / len(keep)
        macro_f1 = sum(r.f1 for r in keep) / len(keep)
    else:
        macro_p = macro_r = macro_f1 = 0.0
    return AggregateResult(
        per_class=per_class,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
    )
