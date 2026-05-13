"""Per-class confusion + structural-error breakdown.

The matcher in :mod:`realdoc_bench.layout.metrics.matching` allows cross-type
pairs to win when IoU is high; this module surfaces those as
``misclassification`` events (rather than burying them as one FP + one FN).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from realdoc_bench.layout.metrics.matching import match_blocks
from realdoc_bench.layout.normalizers.base import LayoutBlock


@dataclass
class StructuralErrors:
    """Per-document structural error counts."""

    misses: int = 0              # GT blocks with no matched pred
    hallucinations: int = 0      # Pred blocks with no matched GT
    misclassifications: int = 0  # Matched pairs with different block types
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)  # (pred_type, gt_type) → count


def analyze(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    *,
    iou_threshold: float = 0.5,
) -> StructuralErrors:
    matched, unmatched_pred, unmatched_gt = match_blocks(
        preds, gts, iou_threshold=iou_threshold, type_sensitive=False
    )
    result = StructuralErrors(
        misses=len(unmatched_gt),
        hallucinations=len(unmatched_pred),
    )
    for pi, gi in matched:
        pt, gt = preds[pi].block_type, gts[gi].block_type
        if pt != gt:
            result.misclassifications += 1
            key = (pt, gt)
            result.confusion[key] = result.confusion.get(key, 0) + 1
    return result


def aggregate(per_doc: list[StructuralErrors]) -> StructuralErrors:
    out = StructuralErrors()
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for d in per_doc:
        out.misses += d.misses
        out.hallucinations += d.hallucinations
        out.misclassifications += d.misclassifications
        for k, v in d.confusion.items():
            confusion[k] += v
    out.confusion = dict(confusion)
    return out


def top_confusions(errors: StructuralErrors, k: int = 5) -> list[tuple[str, str, int]]:
    """Top-k ``(pred_type, gt_type, count)`` confusion entries."""
    items = [(pt, gt, n) for (pt, gt), n in errors.confusion.items()]
    items.sort(key=lambda x: x[2], reverse=True)
    return items[:k]
