"""Layout F1 / Precision / Recall metrics.

Two flavors:

- **Strict**: count matches where both bbox IoU >= threshold AND the block type
  matches (matched-but-wrong-type pairs are treated as FP + FN).
- **Type-tolerant**: count matches at the IoU threshold regardless of type. Used
  by structural-error analysis to surface misclassifications.

Per-class F1 is computed by partitioning preds + GT to a single class before
running the strict pipeline (so cross-type matches between e.g. heading→title
remain hidden, which is the desired behavior for per-class breakdowns).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from realdoc_bench.layout.metrics.matching import match_blocks
from realdoc_bench.layout.normalizers.base import LayoutBlock


@dataclass(frozen=True)
class PRF:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    *,
    iou_threshold: float = 0.5,
    type_sensitive: bool = True,
) -> PRF:
    matched, unmatched_pred, unmatched_gt = match_blocks(
        preds, gts, iou_threshold=iou_threshold, type_sensitive=type_sensitive
    )
    if type_sensitive:
        # Reject matched-but-wrong-type pairs from the TP count.
        good_matches = [(p, g) for p, g in matched if preds[p].block_type == gts[g].block_type]
        bad_matches = [(p, g) for p, g in matched if preds[p].block_type != gts[g].block_type]
        tp = len(good_matches)
        fp = len(unmatched_pred) + len(bad_matches)
        fn = len(unmatched_gt) + len(bad_matches)
    else:
        tp = len(matched)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    return PRF(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def per_class(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, PRF]:
    """Return strict per-class PRF. Partitions inputs to one class before matching."""
    classes: set[str] = {b.block_type for b in preds} | {b.block_type for b in gts}
    out: dict[str, PRF] = {}
    for cls in sorted(classes):
        cls_preds = [b for b in preds if b.block_type == cls]
        cls_gts = [b for b in gts if b.block_type == cls]
        out[cls] = evaluate(cls_preds, cls_gts, iou_threshold=iou_threshold, type_sensitive=True)
    return out


def aggregate(prfs: list[PRF]) -> PRF:
    """Micro-aggregate a list of per-document PRFs into one."""
    tp = sum(x.tp for x in prfs)
    fp = sum(x.fp for x in prfs)
    fn = sum(x.fn for x in prfs)
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    return PRF(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def aggregate_per_class(per_doc: list[dict[str, PRF]]) -> dict[str, PRF]:
    bucket: dict[str, list[PRF]] = defaultdict(list)
    for doc in per_doc:
        for cls, prf in doc.items():
            bucket[cls].append(prf)
    return {cls: aggregate(prfs) for cls, prfs in bucket.items()}
