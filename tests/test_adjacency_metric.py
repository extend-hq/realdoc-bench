"""Tests for the adjacency layout-metric package.

Exercises the Hungarian matcher, adjacency split/merge recovery, optional
ignore filter, and the ``aggregate.score_pairs`` roll-up that builds
dataset-level strict + adjusted F1 from per-page results.
"""

from __future__ import annotations

import pytest

from realdoc_bench.layout.metrics.adjacency import (
    IGNORABLE_BLOCK_TYPES,
    AggregateResult,
    ClassScore,
    LayoutEvaluationResult,
    apply_ignore_filter,
    evaluate_layout_detection,
    get_adjusted_blocks,
    match_blocks,
    match_blocks_with_adjacency,
    precision_recall_f1,
    score_pairs,
)
from realdoc_bench.layout.metrics.adjacency.geometry import iou, iou_matrix
from realdoc_bench.layout.metrics.adjacency.layout import are_adjacent, merge_blocks
from realdoc_bench.layout.normalizers.base import BBox, LayoutBlock


def nb(
    block_type: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    text: str | None = None,
    ignore: bool = False,
    conf: float = 1.0,
) -> LayoutBlock:
    """Build a layout block — test helper."""
    return LayoutBlock(
        block_type=block_type,
        bbox=BBox(x=x, y=y, w=w, h=h),
        text=text,
        ignore=ignore,
        confidence=conf,
    )


# =============================================================================
# Geometry — IoU fidelity
# =============================================================================


class TestGeometry:
    """Pin down realdoc-bench's BBox.iou on the canonical cases."""

    def test_iou_identical_boxes(self):
        a = BBox(x=0, y=0, w=100, h=100)
        assert iou(a, a) == 1.0

    def test_iou_half_overlap(self):
        # inter = 50*100 = 5000; union = 10000 + 10000 - 5000 = 15000
        a = BBox(x=0, y=0, w=100, h=100)
        b = BBox(x=50, y=0, w=100, h=100)
        assert iou(a, b) == pytest.approx(5000 / 15000)

    def test_iou_disjoint(self):
        a = BBox(x=0, y=0, w=100, h=100)
        b = BBox(x=500, y=500, w=100, h=100)
        assert iou(a, b) == 0.0

    def test_iou_matrix_shape_and_values(self):
        a = [BBox(x=0, y=0, w=100, h=100), BBox(x=0, y=0, w=10, h=10)]
        b = [BBox(x=0, y=0, w=100, h=100)]
        m = iou_matrix(a, b)
        assert m.shape == (2, 1)
        assert m[0, 0] == pytest.approx(1.0)
        assert m[1, 0] == pytest.approx(100 / 10000)

    def test_iou_matrix_empty(self):
        assert iou_matrix([], []).shape == (0, 0)
        assert iou_matrix([BBox(x=0, y=0, w=1, h=1)], []).shape == (1, 0)


# =============================================================================
# match_blocks — Hungarian 1:1 matcher
# =============================================================================


class TestMatchBlocks:
    def test_match_blocks_type_sensitive(self):
        preds = [nb("title", 0, 0, 10, 10), nb("text", 50, 50, 10, 10)]
        gts = [nb("title", 0, 0, 10, 10), nb("text", 50, 50, 10, 10)]
        pairs, up, ug = match_blocks(preds, gts, iou_threshold=0.5, type_sensitive=True)
        assert len(pairs) == 2 and not up and not ug

    def test_evaluate_layout_detection_precision_recall(self):
        preds = [nb("title", 0, 0, 10, 10)]
        gts = [nb("title", 0, 0, 10, 10), nb("text", 100, 100, 10, 10)]
        stats = evaluate_layout_detection(preds, gts, iou_threshold=0.5, type_sensitive=True)
        assert stats["tp"] == 1.0 and stats["fn"] == 1.0 and stats["fp"] == 0.0

    def test_misclassification_rejected_by_iou_pair(self):
        # type penalty (+0.25) does not stop the pair forming, but the strict
        # IoU>=0.5 gate keeps a wrong-type overlap from counting as a match.
        preds = [nb("table", 0, 0, 100, 100)]
        gts = [nb("text", 0, 0, 100, 100)]
        pairs, up, ug = match_blocks(preds, gts, type_sensitive=True)
        # Identical boxes → IoU 1.0 ≥ 0.5 → still matched (type is not the gate).
        assert len(pairs) == 1

    def test_empty_inputs(self):
        pairs, up, ug = match_blocks([], [])
        assert pairs == [] and up == [] and ug == []
        pairs, up, ug = match_blocks([nb("text", 0, 0, 10, 10)], [])
        assert pairs == [] and up == [0] and ug == []


# =============================================================================
# are_adjacent / merge_blocks
# =============================================================================


class TestAreAdjacent:
    def test_overlapping_same_type(self):
        a = nb("text", 0, 0, 100, 50)
        b = nb("text", 50, 0, 100, 50)
        assert are_adjacent(a, b, 1000, 1000, gap_threshold=0.02)

    def test_close_vertical_blocks(self):
        a = nb("text", 0, 0, 100, 50)
        b = nb("text", 0, 55, 100, 50)  # 5px gap, threshold 2% = 20px
        assert are_adjacent(a, b, 1000, 1000, gap_threshold=0.02)

    def test_far_apart_blocks(self):
        a = nb("text", 0, 0, 100, 50)
        b = nb("text", 0, 500, 100, 50)
        assert not are_adjacent(a, b, 1000, 1000, gap_threshold=0.02)

    def test_different_types_not_adjacent(self):
        a = nb("text", 0, 0, 100, 50)
        b = nb("figure", 0, 55, 100, 50)
        assert not are_adjacent(a, b, 1000, 1000, gap_threshold=0.02)

    def test_missing_bbox(self):
        a = LayoutBlock(block_type="text", bbox=None)
        b = nb("text", 0, 0, 100, 50)
        assert not are_adjacent(a, b, 1000, 1000, gap_threshold=0.02)


class TestMergeBlocks:
    def test_merge_two_blocks(self):
        merged = merge_blocks(
            [
                nb("text", 0, 0, 100, 50, text="Hello"),
                nb("text", 0, 60, 100, 50, text="World"),
            ]
        )
        assert merged.bbox == BBox(x=0, y=0, w=100, h=110)
        assert merged.text == "Hello\nWorld"
        assert merged.block_type == "text"

    def test_merge_single_block(self):
        block = nb("text", 10, 20, 100, 50)
        assert merge_blocks([block]) == block

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError):
            merge_blocks([])

    def test_merge_preserves_ignore(self):
        merged = merge_blocks(
            [
                nb("text", 0, 0, 100, 50, ignore=False),
                nb("text", 0, 60, 100, 50, ignore=True),
            ]
        )
        assert merged.ignore is True


# =============================================================================
# match_blocks_with_adjacency — strict + adjusted recovery
# =============================================================================


class TestMatchBlocksWithAdjacency:
    def test_perfect_match(self):
        preds = [nb("text", 0, 0, 100, 50)]
        gts = [nb("text", 0, 0, 100, 50)]
        r = match_blocks_with_adjacency(preds, gts, page_width=1000, page_height=1000)
        assert (r.strict_tp, r.strict_fp, r.strict_fn) == (1, 0, 0)
        assert r.strict_f1 == 1.0

    def test_over_segmentation_resolved(self):
        preds = [nb("text", 0, 0, 100, 40), nb("text", 0, 50, 100, 40)]
        gts = [nb("text", 0, 0, 100, 100)]
        r = match_blocks_with_adjacency(
            preds, gts, page_width=1000, page_height=1000, gap_threshold=0.05
        )
        assert r.adjusted_tp >= r.strict_tp
        assert r.adjusted_f1 >= r.strict_f1

    def test_no_match(self):
        preds = [nb("text", 0, 0, 100, 50)]
        gts = [nb("text", 500, 500, 100, 50)]
        r = match_blocks_with_adjacency(preds, gts, page_width=1000, page_height=1000)
        assert (r.strict_tp, r.strict_fp, r.strict_fn) == (0, 1, 1)

    def test_empty_inputs(self):
        r = match_blocks_with_adjacency([], [], page_width=1000, page_height=1000)
        assert (r.strict_tp, r.strict_fp, r.strict_fn) == (0, 0, 0)
        assert r.adjusted_f1 == 0.0

    def test_strict_equals_adjusted_on_perfect_match(self):
        preds = [nb("text", 0, 0, 100, 50), nb("text", 0, 100, 100, 50)]
        gts = [nb("text", 0, 0, 100, 50), nb("text", 0, 100, 100, 50)]
        r = match_blocks_with_adjacency(preds, gts, page_width=1000, page_height=1000)
        assert r.strict_tp == r.adjusted_tp
        assert r.strict_fp == r.adjusted_fp
        assert r.strict_fn == r.adjusted_fn
        assert r.strict_f1 == r.adjusted_f1

    def test_error_breakdown_categories(self):
        preds = [
            nb("text", 0, 0, 100, 40),
            nb("text", 0, 45, 100, 45),
            nb("text", 500, 500, 100, 50),  # true FP
        ]
        gts = [
            nb("text", 0, 0, 100, 100),  # split by preds
            nb("text", 800, 800, 100, 50),  # true miss
        ]
        r = match_blocks_with_adjacency(
            preds, gts, page_width=1000, page_height=1000, gap_threshold=0.05
        )
        assert isinstance(r.true_misses, int)
        assert isinstance(r.true_false_positives, int)
        assert isinstance(r.splits_resolved, int)
        assert r.true_misses >= 1
        assert r.true_false_positives >= 1


# =============================================================================
# precision_recall_f1
# =============================================================================


class TestPrecisionRecallF1:
    def test_all_correct(self):
        assert precision_recall_f1(10, 0, 0) == (1.0, 1.0, 1.0)

    def test_all_false_positives(self):
        assert precision_recall_f1(0, 10, 0) == (0.0, 0.0, 0.0)

    def test_all_false_negatives(self):
        assert precision_recall_f1(0, 0, 10) == (0.0, 0.0, 0.0)

    def test_mixed_results(self):
        p, r, f1 = precision_recall_f1(5, 2, 3)
        assert p == pytest.approx(5 / 7)
        assert r == pytest.approx(5 / 8)
        assert f1 == pytest.approx(2 * (5 / 7) * (5 / 8) / ((5 / 7) + (5 / 8)))

    def test_zeros_everywhere(self):
        assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


class TestLayoutEvaluationResult:
    def test_dataclass_defaults(self):
        r = LayoutEvaluationResult(
            strict_tp=0, strict_fp=0, strict_fn=0,
            strict_precision=0.0, strict_recall=0.0, strict_f1=0.0,
            adjusted_tp=0, adjusted_fp=0, adjusted_fn=0,
            adjusted_precision=0.0, adjusted_recall=0.0, adjusted_f1=0.0,
            splits_resolved=0, merges_resolved=0,
            true_misses=0, true_false_positives=0, misclassifications=0,
        )
        assert r.pred_merge_records == []
        assert r.gt_merge_records == []
        assert r.pred_blocks == []
        assert r.gt_blocks == []


# =============================================================================
# Ignore filter (default set is empty under the public 9-class vocab; the
# filter is now a per-study knob, exercised here with an explicit set).
# =============================================================================


class TestIgnoreFilter:
    def test_default_ignorable_set_empty(self):
        # Under the public 9-class taxonomy every surviving class is meaningful,
        # so the default ignore set is empty; callers opt in explicitly.
        assert IGNORABLE_BLOCK_TYPES == set()

    def test_disabled_keeps_everything(self):
        blocks = [nb("text", 0, 0, 10, 10), nb("header", 0, 20, 10, 10)]
        kept, ignored = apply_ignore_filter(blocks, "disabled")
        assert len(kept) == 2 and ignored == []

    def test_exclude_matching_with_explicit_set(self):
        blocks = [nb("text", 0, 0, 10, 10), nb("header", 0, 20, 10, 10)]
        kept, ignored = apply_ignore_filter(blocks, "exclude_matching", {"header"})
        assert [b.block_type for b in kept] == ["text"]
        assert [b.block_type for b in ignored] == ["header"]

    def test_include_exclude_metrics_flags_ignorable(self):
        blocks = [nb("text", 0, 0, 10, 10), nb("header", 0, 20, 10, 10)]
        kept, ignored = apply_ignore_filter(blocks, "include_exclude_metrics", {"header"})
        assert len(kept) == 2
        flagged = {b.block_type: b.ignore for b in kept}
        assert flagged["text"] is False and flagged["header"] is True


# =============================================================================
# Legacy 18-class label coercion (so old prediction.json files still load).
# =============================================================================


class TestLegacyLabelCoercion:
    def test_legacy_18class_labels_fold_into_9class(self):
        cases = [
            ("title", "heading"),
            ("section_header", "section_heading"),
            ("page_header", "header"),
            ("page_footer", "footer"),
            ("list_item", "text"),
            ("formula", "text"),
            ("footnote", "text"),
            ("signature", "text"),
            ("figure_caption", "text"),
            ("table_cell", "table"),
            ("barcode", "figure"),
        ]
        for legacy, public in cases:
            b = nb(legacy, 0, 0, 10, 10)
            assert b.block_type == public, f"{legacy!r} should coerce to {public!r}, got {b.block_type!r}"


# =============================================================================
# get_adjusted_blocks
# =============================================================================


class TestGetAdjustedBlocks:
    def test_over_segmentation_collapses_preds(self):
        preds = [nb("text", 0, 0, 100, 40), nb("text", 0, 45, 100, 45)]
        gts = [nb("text", 0, 0, 100, 100)]
        adj_preds, adj_gts = get_adjusted_blocks(
            preds, gts, page_width=1000, page_height=1000, gap_threshold=0.05
        )
        # The two split preds merge into a single adjusted prediction.
        assert len(adj_preds) == 1
        assert len(adj_gts) == 1


# =============================================================================
# aggregate.score_pairs — this package's roll-up layer
# =============================================================================


class TestScorePairs:
    def test_perfect_page(self):
        preds = [nb("text", 0, 0, 100, 100), nb("table", 0, 200, 100, 100)]
        gts = [nb("text", 0, 0, 100, 100), nb("table", 0, 200, 100, 100)]
        result = score_pairs([(preds, gts, 1000, 1000)])
        assert isinstance(result, AggregateResult)
        assert result.n_pages == 1
        assert result.strict.micro_f1 == 1.0
        assert result.strict.macro_f1 == 1.0
        assert result.adjusted.micro_f1 == 1.0

    def test_per_class_breakdown(self):
        preds = [nb("text", 0, 0, 100, 100)]
        gts = [nb("text", 0, 0, 100, 100), nb("table", 0, 200, 100, 100)]
        result = score_pairs([(preds, gts, 1000, 1000)])
        per_class = result.strict.per_class
        assert isinstance(per_class["text"], ClassScore)
        assert (per_class["text"].tp, per_class["text"].fp, per_class["text"].fn) == (1, 0, 0)
        # table has 1 GT and 0 preds → pure miss.
        assert (per_class["table"].tp, per_class["table"].fp, per_class["table"].fn) == (0, 0, 1)
        assert per_class["table"].n_gt == 1

    def test_micro_pools_across_pages(self):
        # Page 1: 1 hit. Page 2: 1 miss. micro = pooled = 1 TP, 1 FN.
        page1 = ([nb("text", 0, 0, 100, 100)], [nb("text", 0, 0, 100, 100)], 1000, 1000)
        page2 = ([], [nb("text", 0, 0, 100, 100)], 1000, 1000)
        result = score_pairs([page1, page2])
        assert result.n_pages == 2
        assert (result.strict.tp, result.strict.fp, result.strict.fn) == (1, 0, 1)
        assert result.strict.micro_f1 == pytest.approx(2 / 3)

    def test_macro_is_unweighted_mean(self):
        # text: perfect (F1 1.0). table: pure miss (F1 0.0). macro = 0.5.
        preds = [nb("text", 0, 0, 100, 100)]
        gts = [nb("text", 0, 0, 100, 100), nb("table", 0, 200, 100, 100)]
        result = score_pairs([(preds, gts, 1000, 1000)])
        assert result.strict.macro_f1 == pytest.approx(0.5)

    def test_empty_iterable(self):
        result = score_pairs([])
        assert result.n_pages == 0
        assert result.strict.micro_f1 == 0.0
        assert result.strict.per_class == {}

    def test_class_with_no_gt_excluded_from_table(self):
        # A pure false-positive class (preds only, no GT) is not a table row.
        preds = [nb("text", 0, 0, 100, 100), nb("figure", 0, 200, 100, 100)]
        gts = [nb("text", 0, 0, 100, 100)]
        result = score_pairs([(preds, gts, 1000, 1000)])
        assert "figure" not in result.strict.per_class
        # ...but it still pollutes the micro totals as a false positive.
        assert result.strict.fp == 1
