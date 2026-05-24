"""Layout-detection metric: Hungarian matcher + adjacency split/merge recovery.

The matcher is Hungarian 1:1 on ``1 - IoU`` with a +0.25 type-disagreement
cost penalty; a pair survives only if its IoU clears the threshold. After
the 1:1 pass, an adjacency-merge step unions same-type unmatched preds
(symmetrically: same-type unmatched GTs) when the merged box would clear
the IoU threshold, recovering matches lost to over- and under-segmentation.

Both strict (1:1-only) and adjusted (after adjacency recovery) F1 are
computed in a single pass and returned together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from realdoc_bench.layout.metrics.adjacency.alignment import hungarian
from realdoc_bench.layout.metrics.adjacency.geometry import iou_matrix
from realdoc_bench.layout.metrics.adjacency.ignore import adjust_metrics_for_ignored
from realdoc_bench.layout.normalizers.base import BBox, LayoutBlock


@dataclass
class MergeRecord:
    """Records a merge operation for visualization/debugging."""

    source_indices: list[int]  # Original block indices that were merged
    merged_bbox: BBox  # Resulting merged bounding box
    matched_target_idx: int | None  # Which GT/Pred it matched (None if still unmatched)
    iou_before: float  # Best IoU of any single source block
    iou_after: float  # IoU after merging


@dataclass
class LayoutEvaluationResult:
    """Complete evaluation result with both strict and adjusted metrics."""

    # Strict metrics (1:1 matching only, no merging)
    strict_tp: int
    strict_fp: int
    strict_fn: int
    strict_precision: float
    strict_recall: float
    strict_f1: float

    # Adjusted metrics (after adjacency merging)
    adjusted_tp: int
    adjusted_fp: int
    adjusted_fn: int
    adjusted_precision: float
    adjusted_recall: float
    adjusted_f1: float

    # Error breakdown (per LED Benchmark categories)
    splits_resolved: int  # Pred merges that successfully matched GT
    merges_resolved: int  # GT merges that successfully matched Pred
    true_misses: int  # GT blocks unmatched even after merging
    true_false_positives: int  # Pred blocks unmatched even after merging
    misclassifications: int  # Matched but wrong block type

    # Merge tracking for visualization
    pred_merge_records: list[MergeRecord] = field(default_factory=list)
    gt_merge_records: list[MergeRecord] = field(default_factory=list)

    # Original blocks preserved
    pred_blocks: list[LayoutBlock] = field(default_factory=list)
    gt_blocks: list[LayoutBlock] = field(default_factory=list)


def _match_blocks_vanilla(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    iou_threshold: float = 0.5,
    type_sensitive: bool = True,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predicted and GT blocks via Hungarian on 1-IoU (and type penalty).

    Returns (matched_pairs, unmatched_pred_indices, unmatched_gt_indices)
    """
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    # Vectorized bbox IoU for speed, then add type penalties
    pboxes = [b.bbox for b in preds]
    gboxes = [b.bbox for b in gts]
    ious = iou_matrix(pboxes, gboxes)
    cm = ious.copy()
    cm = 1.0 - cm
    if type_sensitive:
        for i, pb in enumerate(preds):
            for j, gb in enumerate(gts):
                if pb.block_type != gb.block_type:
                    cm[i, j] += 0.25
    cm = cm.tolist()
    pairs, _ = hungarian(cm)

    matched: list[tuple[int, int]] = []
    used_pred = set()
    used_gt = set()
    for pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        pb = preds[pi].bbox
        gb = gts[gi].bbox
        i = pb.iou(gb)
        if i >= iou_threshold:
            matched.append((pi, gi))
            used_pred.add(pi)
            used_gt.add(gi)

    unmatched_pred = [i for i in range(len(preds)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(gts)) if i not in used_gt]
    return matched, unmatched_pred, unmatched_gt


def match_blocks(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    iou_threshold: float = 0.5,
    type_sensitive: bool = True,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    matched, up, ug = _match_blocks_vanilla(preds, gts, iou_threshold=iou_threshold, type_sensitive=type_sensitive)
    return matched, up, ug


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate_layout_detection(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    iou_threshold: float = 0.5,
    type_sensitive: bool = True,
) -> dict[str, float]:
    matched, unmatched_pred, unmatched_gt = match_blocks(preds, gts, iou_threshold=iou_threshold, type_sensitive=type_sensitive)
    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    # Exclude ignored blocks from metric counts (but keep them in matching for visualization)
    tp, fp, fn = adjust_metrics_for_ignored(tp, fp, fn, matched, unmatched_pred, unmatched_gt, preds, gts)

    p, r, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


# =============================================================================
# Layout Adjacency Matching (Spatial Proximity Merging)
# =============================================================================


def are_adjacent(
    block_a: LayoutBlock,
    block_b: LayoutBlock,
    page_width: int,
    page_height: int,
    gap_threshold: float = 0.02,
) -> bool:
    """Check if two blocks are spatially adjacent.

    Two blocks are adjacent if:
    1. Same or compatible block types
    2. They are close together (gap < threshold in the separating direction)
    3. Aligned in the perpendicular direction (overlap)

    For example:
    - Vertically stacked blocks: small vertical gap AND horizontal overlap
    - Horizontally adjacent blocks: small horizontal gap AND vertical overlap
    """
    if block_a.bbox is None or block_b.bbox is None:
        return False

    # Check block type compatibility
    if block_a.block_type != block_b.block_type:
        return False

    a, b = block_a.bbox, block_b.bbox
    threshold_px = gap_threshold * max(page_width, page_height)

    # Check horizontal overlap (aligned vertically - stacked)
    x_overlap = not (a.x + a.w < b.x or b.x + b.w < a.x)
    # Check vertical overlap (aligned horizontally - side by side)
    y_overlap = not (a.y + a.h < b.y or b.y + b.h < a.y)

    # Calculate gaps
    # Vertical gap between stacked blocks
    if a.y + a.h <= b.y:
        v_gap = b.y - (a.y + a.h)
    elif b.y + b.h <= a.y:
        v_gap = a.y - (b.y + b.h)
    else:
        v_gap = 0  # Overlapping vertically

    # Horizontal gap between side-by-side blocks
    if a.x + a.w <= b.x:
        h_gap = b.x - (a.x + a.w)
    elif b.x + b.w <= a.x:
        h_gap = a.x - (b.x + b.w)
    else:
        h_gap = 0  # Overlapping horizontally

    # Adjacent if:
    # - Vertically stacked (x overlap) with small vertical gap, OR
    # - Horizontally adjacent (y overlap) with small horizontal gap, OR
    # - Overlapping (both gaps = 0)
    vertically_adjacent = x_overlap and v_gap < threshold_px
    horizontally_adjacent = y_overlap and h_gap < threshold_px

    return vertically_adjacent or horizontally_adjacent


def merge_blocks(blocks: list[LayoutBlock]) -> LayoutBlock:
    """Merge multiple blocks into one by union of bboxes and concat text."""
    if not blocks:
        raise ValueError("Cannot merge empty list")
    if len(blocks) == 1:
        return blocks[0]

    # Filter blocks with valid bboxes
    blocks_with_bbox = [b for b in blocks if b.bbox is not None]
    if not blocks_with_bbox:
        return blocks[0]

    # Union of bounding boxes
    min_x = min(b.bbox.x for b in blocks_with_bbox)
    min_y = min(b.bbox.y for b in blocks_with_bbox)
    max_x = max(b.bbox.x + b.bbox.w for b in blocks_with_bbox)
    max_y = max(b.bbox.y + b.bbox.h for b in blocks_with_bbox)

    merged_bbox = BBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y)

    # Concatenate text (if available)
    texts = [b.text for b in blocks if b.text]
    merged_text = "\n".join(texts) if texts else None

    return LayoutBlock(
        id=f"merged_{blocks[0].id}",
        block_type=blocks[0].block_type,
        bbox=merged_bbox,
        text=merged_text,
        ignore=any(b.ignore for b in blocks),
    )


def _find_adjacent_group(
    start_idx: int,
    candidates: list[int],
    blocks: Sequence[LayoutBlock],
    page_width: int,
    page_height: int,
    gap_threshold: float,
) -> list[int]:
    """Find all blocks adjacent to start_idx within candidates."""
    if start_idx not in candidates:
        return [start_idx]

    group = [start_idx]
    remaining = [i for i in candidates if i != start_idx]

    changed = True
    while changed:
        changed = False
        for idx in remaining[:]:
            # Check if idx is adjacent to any block in group
            for grp_idx in group:
                if are_adjacent(blocks[idx], blocks[grp_idx], page_width, page_height, gap_threshold):
                    group.append(idx)
                    remaining.remove(idx)
                    changed = True
                    break

    return sorted(group)


def _try_merge_for_gt(
    gt_block: LayoutBlock,
    candidate_pred_indices: list[int],
    preds: Sequence[LayoutBlock],
    page_width: int,
    page_height: int,
    gap_threshold: float,
    iou_threshold: float,
) -> tuple[list[int], float, float] | None:
    """Try to merge candidate predictions to match a GT block.

    Returns (merged_indices, iou_before, iou_after) if successful, None otherwise.
    """
    if len(candidate_pred_indices) < 2:
        return None

    if gt_block.bbox is None:
        return None

    # Find best single block IoU
    best_single_iou = 0.0
    for pi in candidate_pred_indices:
        if preds[pi].bbox:
            iou = preds[pi].bbox.iou(gt_block.bbox)
            best_single_iou = max(best_single_iou, iou)

    # Try merging adjacent groups
    best_merge: tuple[list[int], float] | None = None

    for start_idx in candidate_pred_indices:
        # Find adjacent blocks starting from this one
        adj_group = _find_adjacent_group(start_idx, candidate_pred_indices, preds, page_width, page_height, gap_threshold)

        if len(adj_group) < 2:
            continue

        # Merge and check IoU
        merged = merge_blocks([preds[i] for i in adj_group])
        if merged.bbox is None:
            continue

        merged_iou = merged.bbox.iou(gt_block.bbox)

        if merged_iou >= iou_threshold and merged_iou > best_single_iou:
            if best_merge is None or merged_iou > best_merge[1]:
                best_merge = (adj_group, merged_iou)

    if best_merge:
        return best_merge[0], best_single_iou, best_merge[1]
    return None


def match_blocks_with_adjacency(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    page_width: int,
    page_height: int,
    iou_threshold: float = 0.5,
    gap_threshold: float = 0.02,
    type_sensitive: bool = True,
) -> LayoutEvaluationResult:
    """Match blocks with adjacency merging, returning both strict and adjusted metrics.

    Algorithm:
    1. Compute strict 1:1 matching (existing Hungarian approach)
    2. For unmatched GTs: try merging adjacent preds to match (handles over-segmentation)
    3. For unmatched preds: try merging adjacent GTs to match (handles under-segmentation)
    4. Report both metric sets + error breakdown
    """
    preds_list = list(preds)
    gts_list = list(gts)

    # Step 1: Strict 1:1 matching
    strict_matched, strict_unmatched_pred, strict_unmatched_gt = match_blocks(preds_list, gts_list, iou_threshold, type_sensitive)

    strict_tp = len(strict_matched)
    strict_fp = len(strict_unmatched_pred)
    strict_fn = len(strict_unmatched_gt)
    strict_p, strict_r, strict_f1 = precision_recall_f1(strict_tp, strict_fp, strict_fn)

    # Track used blocks and merge records
    used_preds: set[int] = {pi for pi, _ in strict_matched}
    used_gts: set[int] = {gi for _, gi in strict_matched}
    pred_merge_records: list[MergeRecord] = []
    gt_merge_records: list[MergeRecord] = []
    splits_resolved = 0
    merges_resolved = 0

    # Count misclassifications from strict matching
    misclassifications = 0
    for pi, gi in strict_matched:
        if preds_list[pi].block_type != gts_list[gi].block_type:
            misclassifications += 1

    # Step 2: Try merging preds to match unmatched GTs (handles over-segmentation)
    remaining_unmatched_gt = [i for i in strict_unmatched_gt]

    for gt_idx in remaining_unmatched_gt[:]:
        gt_block = gts_list[gt_idx]
        if gt_block.bbox is None:
            continue

        # Find unmatched pred blocks that overlap with this GT
        candidate_pred_indices = []
        for pi in strict_unmatched_pred:
            if pi in used_preds:
                continue
            pred_block = preds_list[pi]
            if pred_block.bbox is None:
                continue
            # Check for any overlap
            if pred_block.bbox.iou(gt_block.bbox) > 0:
                candidate_pred_indices.append(pi)

        if len(candidate_pred_indices) >= 2:
            result = _try_merge_for_gt(
                gt_block,
                candidate_pred_indices,
                preds_list,
                page_width,
                page_height,
                gap_threshold,
                iou_threshold,
            )

            if result:
                merged_indices, iou_before, iou_after = result
                merged_block = merge_blocks([preds_list[i] for i in merged_indices])

                pred_merge_records.append(
                    MergeRecord(
                        source_indices=merged_indices,
                        merged_bbox=merged_block.bbox,
                        matched_target_idx=gt_idx,
                        iou_before=iou_before,
                        iou_after=iou_after,
                    )
                )

                # Mark as used
                for pi in merged_indices:
                    used_preds.add(pi)
                used_gts.add(gt_idx)
                splits_resolved += 1
                remaining_unmatched_gt.remove(gt_idx)

    # Step 3: Try merging GTs to match unmatched preds (handles under-segmentation)
    remaining_unmatched_pred = [i for i in strict_unmatched_pred if i not in used_preds]

    for pred_idx in remaining_unmatched_pred[:]:
        if pred_idx in used_preds:
            continue
        pred_block = preds_list[pred_idx]
        if pred_block.bbox is None:
            continue

        # Find unmatched GT blocks that overlap with this pred
        candidate_gt_indices = []
        for gi in strict_unmatched_gt:
            if gi in used_gts:
                continue
            gt_block = gts_list[gi]
            if gt_block.bbox is None:
                continue
            if gt_block.bbox.iou(pred_block.bbox) > 0:
                candidate_gt_indices.append(gi)

        if len(candidate_gt_indices) >= 2:
            # Try merging adjacent GTs
            best_merge: tuple[list[int], float, float] | None = None
            best_single_iou = 0.0

            for gi in candidate_gt_indices:
                if gts_list[gi].bbox:
                    iou = gts_list[gi].bbox.iou(pred_block.bbox)
                    best_single_iou = max(best_single_iou, iou)

            for start_idx in candidate_gt_indices:
                adj_group = _find_adjacent_group(start_idx, candidate_gt_indices, gts_list, page_width, page_height, gap_threshold)

                if len(adj_group) < 2:
                    continue

                merged = merge_blocks([gts_list[i] for i in adj_group])
                if merged.bbox is None:
                    continue

                merged_iou = merged.bbox.iou(pred_block.bbox)

                if merged_iou >= iou_threshold and merged_iou > best_single_iou:
                    if best_merge is None or merged_iou > best_merge[2]:
                        best_merge = (adj_group, best_single_iou, merged_iou)

            if best_merge:
                merged_indices, iou_before, iou_after = best_merge
                merged_block = merge_blocks([gts_list[i] for i in merged_indices])

                gt_merge_records.append(
                    MergeRecord(
                        source_indices=merged_indices,
                        merged_bbox=merged_block.bbox,
                        matched_target_idx=pred_idx,
                        iou_before=iou_before,
                        iou_after=iou_after,
                    )
                )

                for gi in merged_indices:
                    used_gts.add(gi)
                used_preds.add(pred_idx)
                merges_resolved += 1
                remaining_unmatched_pred.remove(pred_idx)

    # Step 4: Calculate adjusted metrics
    # TP = strict matches + splits resolved + merges resolved
    # FP = unmatched preds that couldn't be merged
    # FN = unmatched GTs that couldn't be matched even with merging
    adjusted_tp = strict_tp + splits_resolved + merges_resolved
    true_false_positives = len([i for i in range(len(preds_list)) if i not in used_preds])
    true_misses = len([i for i in range(len(gts_list)) if i not in used_gts])
    adjusted_fp = true_false_positives
    adjusted_fn = true_misses

    adjusted_p, adjusted_r, adjusted_f1 = precision_recall_f1(adjusted_tp, adjusted_fp, adjusted_fn)

    return LayoutEvaluationResult(
        # Strict metrics
        strict_tp=strict_tp,
        strict_fp=strict_fp,
        strict_fn=strict_fn,
        strict_precision=strict_p,
        strict_recall=strict_r,
        strict_f1=strict_f1,
        # Adjusted metrics
        adjusted_tp=adjusted_tp,
        adjusted_fp=adjusted_fp,
        adjusted_fn=adjusted_fn,
        adjusted_precision=adjusted_p,
        adjusted_recall=adjusted_r,
        adjusted_f1=adjusted_f1,
        # Error breakdown
        splits_resolved=splits_resolved,
        merges_resolved=merges_resolved,
        true_misses=true_misses,
        true_false_positives=true_false_positives,
        misclassifications=misclassifications,
        # Merge records for visualization
        pred_merge_records=pred_merge_records,
        gt_merge_records=gt_merge_records,
        # Original blocks
        pred_blocks=preds_list,
        gt_blocks=gts_list,
    )


def get_adjusted_blocks(
    preds: Sequence[LayoutBlock],
    gts: Sequence[LayoutBlock],
    page_width: int,
    page_height: int,
    iou_threshold: float = 0.5,
    gap_threshold: float = 0.02,
    type_sensitive: bool = True,
) -> tuple[list[LayoutBlock], list[LayoutBlock]]:
    """Apply adjacency merging and return adjusted prediction and GT blocks.

    This creates new block lists where:
    - Merged predictions are replaced by a single merged block
    - Merged GTs are replaced by a single merged block

    Useful for computing mAP on "adjusted" predictions after handling splits/merges.

    Returns:
        adjusted_preds: Predictions with merges applied
        adjusted_gts: Ground truths with merges applied
    """
    result = match_blocks_with_adjacency(
        preds=preds,
        gts=gts,
        page_width=page_width,
        page_height=page_height,
        iou_threshold=iou_threshold,
        gap_threshold=gap_threshold,
        type_sensitive=type_sensitive,
    )

    preds_list = list(preds)
    gts_list = list(gts)

    # Track which indices have been merged
    merged_pred_indices: set[int] = set()
    merged_gt_indices: set[int] = set()

    adjusted_preds: list[LayoutBlock] = []
    adjusted_gts: list[LayoutBlock] = []

    # Add merged predictions (from pred_merge_records - handling splits)
    for record in result.pred_merge_records:
        merged_block = merge_blocks([preds_list[i] for i in record.source_indices])
        adjusted_preds.append(merged_block)
        merged_pred_indices.update(record.source_indices)

    # Add merged GTs (from gt_merge_records - handling merges by model)
    for record in result.gt_merge_records:
        merged_block = merge_blocks([gts_list[i] for i in record.source_indices])
        adjusted_gts.append(merged_block)
        merged_gt_indices.update(record.source_indices)

    # Add non-merged predictions
    for i, pred in enumerate(preds_list):
        if i not in merged_pred_indices:
            adjusted_preds.append(pred)

    # Add non-merged GTs
    for i, gt in enumerate(gts_list):
        if i not in merged_gt_indices:
            adjusted_gts.append(gt)

    return adjusted_preds, adjusted_gts
