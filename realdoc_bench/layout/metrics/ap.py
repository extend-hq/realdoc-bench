"""Per-class average precision (AP) and mean AP (mAP).

Implements COCO-style AP@IoU using a confidence-sorted greedy match (no
``ultralytics`` dependency — the algorithm is short and clear). Reports
AP@0.50, AP@0.75, and AP@[.50:.95:.05].

If every prediction for a class has identical confidence (e.g. all 1.0
because the provider doesn't expose scores), AP is still well-defined but
degenerates to a single-threshold P/R point — the snapshot report flags this
processor as "mAP-N/A" rather than printing a misleading number.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from realdoc_bench.layout.normalizers.base import LayoutBlock


@dataclass(frozen=True)
class APResult:
    per_iou: dict[float, float]   # iou_threshold → AP
    average: float                # mean over per_iou values

    @property
    def ap50(self) -> float:
        return self.per_iou.get(0.5, 0.0)

    @property
    def ap75(self) -> float:
        return self.per_iou.get(0.75, 0.0)


def _per_class_buckets(
    preds_by_doc: dict[str, Sequence[LayoutBlock]],
    gts_by_doc: dict[str, Sequence[LayoutBlock]],
) -> dict[str, tuple[list[tuple[str, int, float]], dict[str, int]]]:
    """Group predictions and GT counts by class.

    Returns ``{class: (preds, gt_counts)}`` where
    - ``preds`` is a list of ``(doc_id, pred_index, confidence)``
    - ``gt_counts`` is ``{doc_id: gt_count_for_class}``.
    """
    buckets: dict[str, tuple[list[tuple[str, int, float]], dict[str, int]]] = {}

    classes: set[str] = set()
    for blocks in preds_by_doc.values():
        classes.update(b.block_type for b in blocks)
    for blocks in gts_by_doc.values():
        classes.update(b.block_type for b in blocks)

    for cls in classes:
        preds: list[tuple[str, int, float]] = []
        gt_counts: dict[str, int] = {}
        for doc_id, blocks in preds_by_doc.items():
            for i, b in enumerate(blocks):
                if b.block_type == cls:
                    preds.append((doc_id, i, b.confidence))
        for doc_id, blocks in gts_by_doc.items():
            gt_counts[doc_id] = sum(1 for b in blocks if b.block_type == cls)
        buckets[cls] = (preds, gt_counts)
    return buckets


def _ap_at_iou(
    preds: list[tuple[str, int, float]],
    preds_by_doc: dict[str, Sequence[LayoutBlock]],
    gts_by_doc: dict[str, Sequence[LayoutBlock]],
    gt_counts: dict[str, int],
    iou_threshold: float,
) -> float:
    """COCO 11-point interpolated AP for a single class at a single IoU threshold."""
    total_gt = sum(gt_counts.values())
    if total_gt == 0 or not preds:
        return 0.0

    preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
    matched_gt_by_doc: dict[str, set[int]] = defaultdict(set)
    tps: list[int] = []
    fps: list[int] = []

    for doc_id, pred_idx, _conf in preds_sorted:
        pred_blocks = preds_by_doc.get(doc_id, [])
        gt_blocks = gts_by_doc.get(doc_id, [])
        pred = pred_blocks[pred_idx]
        if pred.bbox is None:
            tps.append(0); fps.append(1); continue

        best_iou, best_gt = 0.0, -1
        for gi, gt in enumerate(gt_blocks):
            if gt.block_type != pred.block_type or gi in matched_gt_by_doc[doc_id]:
                continue
            if gt.bbox is None:
                continue
            iou = pred.bbox.iou(gt.bbox)
            if iou > best_iou:
                best_iou, best_gt = iou, gi
        if best_iou >= iou_threshold and best_gt >= 0:
            matched_gt_by_doc[doc_id].add(best_gt)
            tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)

    tp_cum = np.cumsum(tps).astype(np.float64)
    fp_cum = np.cumsum(fps).astype(np.float64)
    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # 11-point interpolated AP (Pascal VOC). COCO uses 101 points; 11 is plenty
    # for our class counts and avoids tiny-N jitter.
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        if not np.any(recalls >= t):
            continue
        ap += float(precisions[recalls >= t].max())
    return ap / 11.0


def compute_map(
    preds_by_doc: dict[str, Sequence[LayoutBlock]],
    gts_by_doc: dict[str, Sequence[LayoutBlock]],
    *,
    iou_thresholds: Sequence[float] | None = None,
) -> tuple[dict[str, APResult], APResult]:
    """Return ``(per_class_AP, mean_AP)``.

    ``mean_AP`` is the macro mean over classes that have at least one GT instance.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * k, 2) for k in range(10)]  # 0.50…0.95

    buckets = _per_class_buckets(preds_by_doc, gts_by_doc)
    per_class: dict[str, APResult] = {}
    for cls, (cls_preds, gt_counts) in buckets.items():
        per_iou: dict[float, float] = {}
        for t in iou_thresholds:
            per_iou[t] = _ap_at_iou(cls_preds, preds_by_doc, gts_by_doc, gt_counts, t)
        per_class[cls] = APResult(per_iou=per_iou, average=float(np.mean(list(per_iou.values()))))

    eligible = [r for cls, r in per_class.items() if sum(buckets[cls][1].values()) > 0]
    if not eligible:
        mean = APResult(per_iou={t: 0.0 for t in iou_thresholds}, average=0.0)
    else:
        mean_per_iou = {
            t: float(np.mean([r.per_iou[t] for r in eligible])) for t in iou_thresholds
        }
        mean = APResult(per_iou=mean_per_iou, average=float(np.mean(list(mean_per_iou.values()))))
    return per_class, mean
