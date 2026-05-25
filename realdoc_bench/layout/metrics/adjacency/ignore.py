"""Ignore handling for document layout evaluation.

Configurable filtering of certain block types (headers, footers, captions,
etc.) from metric calculations, per OmniDocBench's Ignore-Handling strategy.
Reference: https://github.com/opendatalab/OmniDocBench
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from realdoc_bench.layout.normalizers.base import LayoutBlock, LayoutBlockType

# Default block types to ignore in evaluation. Under the public 9-class
# vocabulary every surviving class is meaningful, so the default set is
# empty; callers can opt in to a per-study ignore list by passing
# ``ignored_types`` explicitly to :func:`apply_ignore_filter`.
IGNORABLE_BLOCK_TYPES: set[LayoutBlockType] = set()

IgnoreMode = Literal["exclude_matching", "include_exclude_metrics", "disabled"]


def apply_ignore_filter(
    blocks: Sequence[LayoutBlock],
    mode: IgnoreMode,
    ignored_types: set[str] | None = None,
) -> tuple[list[LayoutBlock], list[LayoutBlock]]:
    """Filter blocks based on ignore mode.

    Args:
        blocks: List of layout blocks to filter
        mode: Ignore handling mode:
            - "exclude_matching": Remove ignored blocks entirely (not used in matching)
            - "include_exclude_metrics": Keep for matching, but flag for metric exclusion
            - "disabled": No filtering, include all blocks
        ignored_types: Set of block types to ignore (defaults to IGNORABLE_BLOCK_TYPES)

    Returns:
        Tuple of (blocks_for_matching, ignored_blocks)
        - In "exclude_matching" mode, ignored_blocks are removed from matching
        - In "include_exclude_metrics" mode, ignored_blocks are still matched but
          have their `ignore` flag set to True for later metric exclusion
    """
    if ignored_types is None:
        ignored_types = IGNORABLE_BLOCK_TYPES

    if mode == "disabled":
        return list(blocks), []

    ignored: list[LayoutBlock] = []
    kept: list[LayoutBlock] = []

    for block in blocks:
        is_ignored = block.block_type in ignored_types

        if mode == "exclude_matching" and is_ignored:
            # Completely exclude from matching
            ignored.append(block)
        else:
            # Keep block, but mark as ignored if applicable
            if is_ignored and not block.ignore:
                # Create copy with ignore flag set
                block = block.model_copy(update={"ignore": True})
            kept.append(block)

    return kept, ignored


def count_ignored_in_matches(
    matched_pairs: list[tuple[int, int]],
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
) -> int:
    """Count how many matched pairs involve ignored blocks.

    This is useful for adjusting TP counts when using "include_exclude_metrics" mode.
    """
    count = 0
    for pred_idx, gt_idx in matched_pairs:
        if preds[pred_idx].ignore or gts[gt_idx].ignore:
            count += 1
    return count


def adjust_metrics_for_ignored(
    tp: int,
    fp: int,
    fn: int,
    matched_pairs: list[tuple[int, int]],
    unmatched_preds: list[int],
    unmatched_gts: list[int],
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
) -> tuple[int, int, int]:
    """Adjust TP/FP/FN counts by excluding ignored blocks.

    In "include_exclude_metrics" mode, ignored blocks participate in matching
    but are excluded from metric calculations.

    Returns:
        Tuple of (adjusted_tp, adjusted_fp, adjusted_fn)
    """
    # Count ignored matches
    ignored_matches = count_ignored_in_matches(matched_pairs, preds, gts)

    # Count ignored unmatched preds (should not count as FP)
    ignored_fp = sum(1 for pi in unmatched_preds if preds[pi].ignore)

    # Count ignored unmatched GTs (should not count as FN)
    ignored_fn = sum(1 for gi in unmatched_gts if gts[gi].ignore)

    adjusted_tp = tp - ignored_matches
    adjusted_fp = fp - ignored_fp
    adjusted_fn = fn - ignored_fn

    return max(0, adjusted_tp), max(0, adjusted_fp), max(0, adjusted_fn)
