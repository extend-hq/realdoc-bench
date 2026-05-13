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
    # Adjusted (with adjacency-merging tolerance for split / merge errors).
    # Populated by :func:`aggregate_with_adjacency`.
    adjusted_per_class: dict[str, PerClassResult] | None = None
    adjusted_macro_precision: float | None = None
    adjusted_macro_recall: float | None = None
    adjusted_macro_f1: float | None = None
    splits_resolved: int = 0
    merges_resolved: int = 0


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


def _are_adjacent(
    a: LayoutBlock,
    b: LayoutBlock,
    page_width: int,
    page_height: int,
    gap_threshold: float = 0.02,
) -> bool:
    """True when two same-type blocks are spatially adjacent.

    Ported from ``ocr-bench/metrics/layout.py:are_adjacent``: same block type
    and small gap (under ``gap_threshold * max(page_w, page_h)``) in the
    separating direction with overlap in the perpendicular direction.
    """
    if a.bbox is None or b.bbox is None or a.block_type != b.block_type:
        return False
    ab, bb = a.bbox, b.bbox
    px = gap_threshold * max(page_width, page_height)
    x_overlap = not (ab.x + ab.w < bb.x or bb.x + bb.w < ab.x)
    y_overlap = not (ab.y + ab.h < bb.y or bb.y + bb.h < ab.y)
    v_gap = max(0, max(ab.y, bb.y) - min(ab.y + ab.h, bb.y + bb.h))
    h_gap = max(0, max(ab.x, bb.x) - min(ab.x + ab.w, bb.x + bb.w))
    return (x_overlap and v_gap < px) or (y_overlap and h_gap < px)


def _merge_bboxes(blocks: list[LayoutBlock]) -> BBox | None:
    valid = [b.bbox for b in blocks if b.bbox is not None]
    if not valid:
        return None
    x0 = min(b.x for b in valid)
    y0 = min(b.y for b in valid)
    x1 = max(b.x + b.w for b in valid)
    y1 = max(b.y + b.h for b in valid)
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def _adjacent_group(
    start: int,
    candidates: list[int],
    blocks: list[LayoutBlock],
    page_width: int,
    page_height: int,
    gap_threshold: float,
) -> list[int]:
    """Flood-fill all blocks adjacent (transitively) to ``start`` within ``candidates``."""
    if start not in candidates:
        return [start]
    group = [start]
    remaining = [c for c in candidates if c != start]
    changed = True
    while changed:
        changed = False
        for c in remaining[:]:
            if any(
                _are_adjacent(blocks[c], blocks[g], page_width, page_height, gap_threshold)
                for g in group
            ):
                group.append(c)
                remaining.remove(c)
                changed = True
    return sorted(group)


def _try_merge_to_match(
    target: LayoutBlock,
    candidates: list[int],
    source_blocks: list[LayoutBlock],
    page_width: int,
    page_height: int,
    iou_threshold: float,
    gap_threshold: float,
) -> list[int] | None:
    """Find an adjacent group of ``source_blocks[candidates]`` that IoU-matches ``target``.

    Returns the candidate indices used, or ``None`` if no group reaches the IoU bar.
    """
    if target.bbox is None or len(candidates) < 2:
        return None
    best_single = 0.0
    for ci in candidates:
        b = source_blocks[ci].bbox
        if b is not None:
            best_single = max(best_single, b.iou(target.bbox))

    best: tuple[list[int], float] | None = None
    for start in candidates:
        group = _adjacent_group(start, candidates, source_blocks, page_width, page_height, gap_threshold)
        if len(group) < 2:
            continue
        merged = _merge_bboxes([source_blocks[i] for i in group])
        if merged is None:
            continue
        iou = merged.iou(target.bbox)
        if iou >= iou_threshold and iou > best_single and (best is None or iou > best[1]):
            best = (group, iou)
    return best[0] if best else None


def _match_with_adjacency(
    preds: list[LayoutBlock],
    gts: list[LayoutBlock],
    page_width: int,
    page_height: int,
    iou_threshold: float,
    gap_threshold: float = 0.02,
) -> tuple[int, int, int, int, int, dict[str, tuple[int, int, int]]]:
    """Run strict + adjacency-tolerant matching on one image.

    Returns (strict_tp, splits_resolved, merges_resolved, adjusted_fp, adjusted_fn,
    per_class_counts).  ``per_class_counts`` is {cls: (tp, fp, fn)} on the
    *adjusted* side.
    """
    # Strict greedy by-class IoU pass — record matches and unmatched
    n, m = len(preds), len(gts)
    gt_used = [False] * m
    pred_used = [False] * n
    matched: list[tuple[int, int]] = []
    order = sorted(range(n), key=lambda i: -float(preds[i].confidence))
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
            iou = pb.bbox.iou(gb.bbox)
            if iou >= best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi >= 0:
            gt_used[best_gi] = True
            pred_used[pi] = True
            matched.append((pi, best_gi))

    strict_tp = len(matched)
    splits = merges = 0

    # Track which class each resolution credits a TP to.
    # Same-class adjacency requirement guarantees all members share block_type.
    resolution_tps: list[str] = []  # one entry per split or merge event

    # Step 2: merge unmatched preds to match unmatched GTs (split recovery)
    for gi in [i for i in range(m) if not gt_used[i]]:
        gb = gts[gi]
        if gb.bbox is None:
            continue
        cands = [
            pi
            for pi in range(n)
            if not pred_used[pi]
            and preds[pi].bbox is not None
            and preds[pi].block_type == gb.block_type
            and preds[pi].bbox.iou(gb.bbox) > 0
        ]
        group = _try_merge_to_match(
            gb, cands, preds, page_width, page_height, iou_threshold, gap_threshold
        )
        if group:
            for pi in group:
                pred_used[pi] = True
            gt_used[gi] = True
            splits += 1
            resolution_tps.append(gb.block_type)

    # Step 3: merge unmatched GTs to match unmatched preds (merge recovery)
    for pi in [i for i in range(n) if not pred_used[i]]:
        pb = preds[pi]
        if pb.bbox is None:
            continue
        cands = [
            gi
            for gi in range(m)
            if not gt_used[gi]
            and gts[gi].bbox is not None
            and gts[gi].block_type == pb.block_type
            and gts[gi].bbox.iou(pb.bbox) > 0
        ]
        group = _try_merge_to_match(
            pb, cands, gts, page_width, page_height, iou_threshold, gap_threshold
        )
        if group:
            for gi in group:
                gt_used[gi] = True
            pred_used[pi] = True
            merges += 1
            resolution_tps.append(pb.block_type)

    adjusted_fp = sum(1 for u in pred_used if not u)
    adjusted_fn = sum(1 for u in gt_used if not u)

    # Per-class adjusted counts. Adjusted TP = strict matches + resolution events
    # (each split or merge counts as one TP). FP / FN are the residual unmatched
    # preds / GTs after both recovery passes.
    by_cls: dict[str, list[int]] = {}
    for pi, _gi in matched:
        cls = preds[pi].block_type
        by_cls.setdefault(cls, [0, 0, 0])
        by_cls[cls][0] += 1
    for cls in resolution_tps:
        by_cls.setdefault(cls, [0, 0, 0])
        by_cls[cls][0] += 1
    for pi in range(n):
        if not pred_used[pi]:
            cls = preds[pi].block_type
            by_cls.setdefault(cls, [0, 0, 0])
            by_cls[cls][1] += 1
    for gi in range(m):
        if not gt_used[gi]:
            cls = gts[gi].block_type
            by_cls.setdefault(cls, [0, 0, 0])
            by_cls[cls][2] += 1

    per_class_counts = {cls: (v[0], v[1], v[2]) for cls, v in by_cls.items()}
    return strict_tp, splits, merges, adjusted_fp, adjusted_fn, per_class_counts


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def aggregate(
    pred_docs: Iterable[tuple[LayoutDocument, LayoutDocument]],
    *,
    iou_threshold: float = 0.5,
    class_remap: dict[str, str] | None = None,
    skip_classes: frozenset[str] = frozenset(),
    with_adjusted: bool = False,
    gap_threshold: float = 0.02,
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
        with_adjusted: also compute adjacency-merged adjusted F1 (mirrors
            the original Notion benchmark's "Adjusted F1" column).
        gap_threshold: adjacency-gap as a fraction of max page dim.
            Default 0.02 (= ~20px on a 1000px page) matches ocr-bench.
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

    # Materialize so we can stream pairs twice (strict pass + optional adjusted pass)
    pairs: list[tuple[list[LayoutBlock], list[LayoutBlock], int, int]] = []
    for pred_doc, gt_doc in pred_docs:
        pred_blocks: list[LayoutBlock] = []
        for pg in pred_doc.pages:
            for b in pg.blocks:
                rb = _remap(b)
                if rb.block_type in skip:
                    continue
                pred_blocks.append(rb)
        gt_blocks: list[LayoutBlock] = []
        page_w = page_h = 0
        for pg in gt_doc.pages:
            page_w = max(page_w, pg.width)
            page_h = max(page_h, pg.height)
            for b in pg.blocks:
                rb = _remap(b)
                if rb.block_type in skip:
                    continue
                gt_blocks.append(rb)
        pairs.append((pred_blocks, gt_blocks, page_w, page_h))

        tp_flags = compute_tp_flags(pred_blocks, gt_blocks, iou_threshold=iou_threshold)
        for b, is_tp in zip(pred_blocks, tp_flags):
            pred_records.append((float(b.confidence), is_tp, b.block_type))
        for b in gt_blocks:
            gt_counts[b.block_type] = gt_counts.get(b.block_type, 0) + 1

    per_class = compute_best_f1_per_class(pred_records, gt_counts)

    keep = [r for r in per_class.values() if r.num_gt > 0]
    if keep:
        macro_p = sum(r.precision for r in keep) / len(keep)
        macro_r = sum(r.recall for r in keep) / len(keep)
        macro_f1 = sum(r.f1 for r in keep) / len(keep)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    adjusted_per_class: dict[str, PerClassResult] | None = None
    adjusted_macro_p = adjusted_macro_r = adjusted_macro_f1 = None
    splits_total = merges_total = 0

    if with_adjusted:
        # Adjusted F1 uses ALL predictions (no threshold sweep) — matches what
        # the original Notion benchmark reports as "Adjusted F1": the model's
        # raw output evaluated with adjacency-merge tolerance for split / merge
        # errors. Models without per-block confidence (e.g. dots.mocr) need this
        # path because their strict sweep is meaningless on tied scores.
        adj_by_cls: dict[str, list[int]] = {}  # class -> [tp, fp, fn]
        for pred_blocks, gt_blocks, page_w, page_h in pairs:
            (_, splits, merges, _, _, per_class_counts) = _match_with_adjacency(
                pred_blocks,
                gt_blocks,
                page_w or 1000,
                page_h or 1000,
                iou_threshold=iou_threshold,
                gap_threshold=gap_threshold,
            )
            splits_total += splits
            merges_total += merges
            for cls, (tp, fp, fn) in per_class_counts.items():
                row = adj_by_cls.setdefault(cls, [0, 0, 0])
                row[0] += tp
                row[1] += fp
                row[2] += fn

        adjusted_per_class = {}
        for cls in sorted(adj_by_cls.keys()):
            tp, fp, fn = adj_by_cls[cls]
            p, r, f1 = _f1(tp, fp, fn)
            adjusted_per_class[cls] = PerClassResult(
                cls=cls,
                precision=p,
                recall=r,
                f1=f1,
                best_conf=0.0,  # adjusted F1 uses all preds; no threshold sweep
                tp=tp,
                fp=fp,
                fn=fn,
                num_gt=gt_counts.get(cls, 0),
            )
        keep_adj = [r for r in adjusted_per_class.values() if r.num_gt > 0]
        if keep_adj:
            adjusted_macro_p = sum(r.precision for r in keep_adj) / len(keep_adj)
            adjusted_macro_r = sum(r.recall for r in keep_adj) / len(keep_adj)
            adjusted_macro_f1 = sum(r.f1 for r in keep_adj) / len(keep_adj)
        else:
            adjusted_macro_p = adjusted_macro_r = adjusted_macro_f1 = 0.0

    return AggregateResult(
        per_class=per_class,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        adjusted_per_class=adjusted_per_class,
        adjusted_macro_precision=adjusted_macro_p,
        adjusted_macro_recall=adjusted_macro_r,
        adjusted_macro_f1=adjusted_macro_f1,
        splits_resolved=splits_total,
        merges_resolved=merges_total,
    )
