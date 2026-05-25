"""Per-class / micro / macro roll-up over the adjacency layout matcher.

The aggregation layer that turns per-page
:func:`~realdoc_bench.layout.metrics.adjacency.layout.match_blocks_with_adjacency`
results into a dataset-level score table — strict + adjusted, micro + macro,
and a per-class breakdown.

This module only *accumulates* matcher outputs; it applies no ignore
filtering of its own — callers prepare the blocks first (e.g. via
:func:`apply_ignore_filter`).

Roll-up definition:

- **Overall (micro)** — every page is run through ``match_blocks_with_adjacency``
  with ``type_sensitive=True``; the strict and adjusted TP/FP/FN are pooled and
  ``precision_recall_f1`` is applied to the totals.
- **Per-class** — for each block type present on a page, the single-class
  subsets of preds and GTs are run through ``match_blocks_with_adjacency`` with
  ``type_sensitive=False`` (the subset is already one class, so the type
  penalty is moot) and pooled per class.
- **Macro** — the unweighted mean of the per-class precision / recall / F1 over
  classes with at least one GT block.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from realdoc_bench.layout.metrics.adjacency.layout import (
    match_blocks_with_adjacency,
    precision_recall_f1,
)
from realdoc_bench.layout.normalizers.base import LayoutBlock

# One page's input to the scorer: (predicted blocks, GT blocks, page_w, page_h).
PagePair = tuple[Sequence[LayoutBlock], Sequence[LayoutBlock], int, int]


@dataclass
class ClassScore:
    """Precision / recall / F1 for a single block type."""

    block_type: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    n_gt: int  # total GT blocks of this type (for sorting / context)


@dataclass
class MetricBreakdown:
    """One metric variant — either ``strict`` or ``adjusted``."""

    tp: int
    fp: int
    fn: int
    # micro = pooled over every block (precision_recall_f1 of the totals)
    micro_precision: float
    micro_recall: float
    micro_f1: float
    # macro = unweighted mean over classes with >= 1 GT block
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, ClassScore] = field(default_factory=dict)


@dataclass
class AggregateResult:
    """Dataset-level score: strict + adjusted, each micro / macro / per-class."""

    strict: MetricBreakdown
    adjusted: MetricBreakdown
    splits_resolved: int
    merges_resolved: int
    n_pages: int


def _macro(per_class: dict[str, ClassScore]) -> tuple[float, float, float]:
    """Unweighted mean of P / R / F1 over classes with >= 1 GT block."""
    rows = [c for c in per_class.values() if (c.tp + c.fn) > 0]
    if not rows:
        return 0.0, 0.0, 0.0
    n = len(rows)
    return (
        sum(c.precision for c in rows) / n,
        sum(c.recall for c in rows) / n,
        sum(c.f1 for c in rows) / n,
    )


def _breakdown(
    totals: list[int],
    by_cls: dict[str, list[int]],
    gt_counts: dict[str, int],
) -> MetricBreakdown:
    """Turn pooled (tp, fp, fn) totals + per-class counts into a breakdown."""
    tp, fp, fn = totals
    mp, mr, mf1 = precision_recall_f1(tp, fp, fn)

    per_class: dict[str, ClassScore] = {}
    for cls, (c_tp, c_fp, c_fn) in by_cls.items():
        if (c_tp + c_fn) == 0:
            # No GT of this class — skip it from the table (matches the bridge
            # roll-up, which sorts/reports only classes with >= 1 GT block).
            continue
        cp, cr, cf1 = precision_recall_f1(c_tp, c_fp, c_fn)
        per_class[cls] = ClassScore(
            block_type=cls,
            tp=c_tp,
            fp=c_fp,
            fn=c_fn,
            precision=cp,
            recall=cr,
            f1=cf1,
            n_gt=gt_counts.get(cls, 0),
        )

    macro_p, macro_r, macro_f1 = _macro(per_class)
    return MetricBreakdown(
        tp=tp,
        fp=fp,
        fn=fn,
        micro_precision=mp,
        micro_recall=mr,
        micro_f1=mf1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        per_class=per_class,
    )


def score_pairs(
    pairs: Iterable[PagePair],
    *,
    iou_threshold: float = 0.5,
    gap_threshold: float = 0.02,
) -> AggregateResult:
    """Score an iterable of ``(preds, gts, page_width, page_height)`` page pairs.

    Each page is run through ``match_blocks_with_adjacency`` twice:

    - ``type_sensitive=True`` for the overall (micro) strict + adjusted totals.
    - ``type_sensitive=False`` on the single-class subsets for the per-class
      table.

    Args:
        pairs: Iterable of per-page ``(preds, gts, page_width, page_height)``
            tuples. Blocks should already be remapped / ignore-filtered to the
            desired class scheme — this function applies none of its own.
        iou_threshold: Minimum IoU for a 1:1 match (default 0.5).
        gap_threshold: Adjacency gap as a fraction of ``max(page_w, page_h)``
            (default 0.02).

    Returns:
        An :class:`AggregateResult` with strict and adjusted breakdowns, each
        carrying pooled micro totals, the per-class table, and the unweighted
        macro average over classes with >= 1 GT block.
    """
    strict_tot = [0, 0, 0]  # tp, fp, fn
    adj_tot = [0, 0, 0]
    strict_by_cls: dict[str, list[int]] = {}
    adj_by_cls: dict[str, list[int]] = {}
    gt_counts: dict[str, int] = {}
    splits_total = 0
    merges_total = 0
    n_pages = 0

    for preds, gts, page_w, page_h in pairs:
        pw = int(page_w)
        ph = int(page_h)

        # Overall (type-sensitive) — strict + adjusted in one pass.
        result = match_blocks_with_adjacency(
            preds,
            gts,
            pw,
            ph,
            iou_threshold=iou_threshold,
            gap_threshold=gap_threshold,
            type_sensitive=True,
        )
        strict_tot[0] += result.strict_tp
        strict_tot[1] += result.strict_fp
        strict_tot[2] += result.strict_fn
        adj_tot[0] += result.adjusted_tp
        adj_tot[1] += result.adjusted_fp
        adj_tot[2] += result.adjusted_fn
        splits_total += result.splits_resolved
        merges_total += result.merges_resolved

        # Per-class — single-class subsets (type penalty is moot within a class).
        classes = {b.block_type for b in preds} | {b.block_type for b in gts}
        for cls in classes:
            p_sub = [b for b in preds if b.block_type == cls]
            g_sub = [b for b in gts if b.block_type == cls]
            gt_counts[cls] = gt_counts.get(cls, 0) + len(g_sub)
            if not p_sub and not g_sub:
                continue
            r_sub = match_blocks_with_adjacency(
                p_sub,
                g_sub,
                pw,
                ph,
                iou_threshold=iou_threshold,
                gap_threshold=gap_threshold,
                type_sensitive=False,
            )
            s = strict_by_cls.setdefault(cls, [0, 0, 0])
            s[0] += r_sub.strict_tp
            s[1] += r_sub.strict_fp
            s[2] += r_sub.strict_fn
            a = adj_by_cls.setdefault(cls, [0, 0, 0])
            a[0] += r_sub.adjusted_tp
            a[1] += r_sub.adjusted_fp
            a[2] += r_sub.adjusted_fn

        n_pages += 1

    return AggregateResult(
        strict=_breakdown(strict_tot, strict_by_cls, gt_counts),
        adjusted=_breakdown(adj_tot, adj_by_cls, gt_counts),
        splits_resolved=splits_total,
        merges_resolved=merges_total,
        n_pages=n_pages,
    )
