"""Hungarian-IoU block matcher.

Type-sensitive matching adds a
penalty (0.25) to cross-type pairings; we still allow them at high IoU so
misclassifications surface as matched-but-wrong-type rather than disappearing
as unmatched FP/FN pairs (see :mod:`realdoc_bench.layout.metrics.structural_errors`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from realdoc_bench.layout.normalizers.base import BBox, LayoutBlock


def iou_matrix(preds: Sequence[BBox | None], gts: Sequence[BBox | None]) -> np.ndarray:
    n, m = len(preds), len(gts)
    out = np.zeros((n, m), dtype=np.float32)
    for i, p in enumerate(preds):
        if p is None:
            continue
        for j, g in enumerate(gts):
            if g is None:
                continue
            out[i, j] = p.iou(g)
    return out


def match_blocks(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    *,
    iou_threshold: float = 0.5,
    type_sensitive: bool = True,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return ``(matched_pairs, unmatched_pred_idx, unmatched_gt_idx)``.

    Matched pairs satisfy ``IoU >= iou_threshold``. With ``type_sensitive=True``,
    a per-pair penalty of 0.25 is added to the cost matrix when block types
    disagree — pairs can still survive if the IoU is high enough.
    """
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    pboxes = [b.bbox for b in preds]
    gboxes = [b.bbox for b in gts]
    ious = iou_matrix(pboxes, gboxes)
    cost = 1.0 - ious
    if type_sensitive:
        for i, pb in enumerate(preds):
            for j, gb in enumerate(gts):
                if pb.block_type != gb.block_type:
                    cost[i, j] += 0.25

    row_ind, col_ind = linear_sum_assignment(cost)
    matched: list[tuple[int, int]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for pi, gi in zip(row_ind, col_ind):
        pb, gb = preds[pi].bbox, gts[gi].bbox
        if pb is None or gb is None:
            continue
        if pb.iou(gb) >= iou_threshold:
            matched.append((int(pi), int(gi)))
            used_pred.add(int(pi))
            used_gt.add(int(gi))

    unmatched_pred = [i for i in range(len(preds)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(gts)) if i not in used_gt]
    return matched, unmatched_pred, unmatched_gt
