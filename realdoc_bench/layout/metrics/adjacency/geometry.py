"""Vectorized axis-aligned IoU helpers used by the adjacency matcher.

A Shapely-based ``polygon_iou`` is intentionally not provided — this metric
operates on axis-aligned boxes only, so the Shapely dependency would be
deadweight.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from realdoc_bench.layout.normalizers.base import BBox


def iou(a: BBox, b: BBox) -> float:
    return a.iou(b)


def iou_matrix(boxes_a: Sequence[BBox], boxes_b: Sequence[BBox]) -> np.ndarray:
    """Vectorized IoU for two sets of axis-aligned boxes.

    Returns an array of shape (len(boxes_a), len(boxes_b)). Empty inputs return empty arrays.
    """
    na, nb = len(boxes_a), len(boxes_b)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float32)

    ax1 = np.array([b.x for b in boxes_a], dtype=np.float32)[:, None]
    ay1 = np.array([b.y for b in boxes_a], dtype=np.float32)[:, None]
    ax2 = ax1 + np.array([b.w for b in boxes_a], dtype=np.float32)[:, None]
    ay2 = ay1 + np.array([b.h for b in boxes_a], dtype=np.float32)[:, None]

    bx1 = np.array([b.x for b in boxes_b], dtype=np.float32)[None, :]
    by1 = np.array([b.y for b in boxes_b], dtype=np.float32)[None, :]
    bx2 = bx1 + np.array([b.w for b in boxes_b], dtype=np.float32)[None, :]
    by2 = by1 + np.array([b.h for b in boxes_b], dtype=np.float32)[None, :]

    inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))
    union = area_a + area_b - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        iou_mat = np.where(union > 0.0, inter / union, 0.0)
    return iou_mat.astype(np.float32)
