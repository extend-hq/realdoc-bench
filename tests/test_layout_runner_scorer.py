"""Tests for the ``--scorer {f1,adjacency}`` wiring in the layout runner.

Covers the parts that don't need the dataset loader or live processors:
``_evaluate_cell``'s scorer branch, ``CellResult``'s new ``adjusted`` field,
and ``aggregate_summary``'s roll-up of adjusted PRFs across cells.
"""

from __future__ import annotations

from pathlib import Path

from realdoc_bench.layout.data.loader import LayoutSample
from realdoc_bench.layout.metrics.f1 import PRF
from realdoc_bench.layout.metrics.structural_errors import StructuralErrors
from realdoc_bench.layout.normalizers.base import BBox, LayoutBlock, LayoutDocument, LayoutPage
from realdoc_bench.layout.runner import (
    CellResult,
    RunSummary,
    _evaluate_cell,
    aggregate_summary,
)


def _block(x: int, y: int, w: int, h: int, t: str = "text") -> LayoutBlock:
    return LayoutBlock(block_type=t, bbox=BBox(x=x, y=y, w=w, h=h))


def _doc(blocks: list[LayoutBlock], *, w: int = 1000, h: int = 1000) -> LayoutDocument:
    return LayoutDocument(pages=[LayoutPage(page_number=1, width=w, height=h, blocks=blocks)])


def _sample(gt_blocks: list[LayoutBlock]) -> LayoutSample:
    return LayoutSample(
        page_id="p1",
        domain="x",
        image_path=Path("/dev/null"),
        ground_truth=_doc(gt_blocks),
        source_url=None,
        original_image_url=None,
    )


# =============================================================================
# _evaluate_cell — scorer branch
# =============================================================================


def test_evaluate_cell_default_scorer_returns_no_adjusted():
    """f1 scorer (the default) leaves the adjusted slot None."""
    preds = [_block(0, 0, 100, 100)]
    gts = [_block(0, 0, 100, 100)]
    sample = _sample(gts)
    strict, per_cls, errors, adjusted = _evaluate_cell(sample, _doc(preds))
    assert isinstance(strict, PRF)
    assert strict.tp == 1 and strict.f1 == 1.0
    assert "text" in per_cls
    assert isinstance(errors, StructuralErrors)
    assert adjusted is None


def test_evaluate_cell_adjacency_returns_adjusted():
    """adjacency scorer produces both strict + adjusted PRFs from one pass."""
    preds = [_block(0, 0, 100, 100), _block(0, 200, 100, 100, t="table")]
    gts = [_block(0, 0, 100, 100), _block(0, 200, 100, 100, t="table")]
    sample = _sample(gts)
    strict, per_cls, errors, adjusted = _evaluate_cell(
        sample, _doc(preds), scorer="adjacency"
    )
    assert isinstance(adjusted, PRF)
    assert strict.f1 == 1.0
    assert adjusted.f1 == 1.0
    assert {"text", "table"} <= set(per_cls)
    assert isinstance(errors, StructuralErrors)


def test_evaluate_cell_adjacency_adjusted_recovers_split():
    """An over-segmented pred that strict matching rejects should be recovered
    by adjacency merging in adjusted F1."""
    # GT: one 100x100 block. Preds: two adjacent vertical halves — neither
    # individually clears IoU 0.5 against the GT.
    preds = [_block(0, 0, 100, 40), _block(0, 45, 100, 45)]  # 5px gap, threshold 50px
    gts = [_block(0, 0, 100, 100)]
    sample = _sample(gts)
    _, _, _, adjusted = _evaluate_cell(sample, _doc(preds), scorer="adjacency")
    assert adjusted is not None
    # The merged pred should match the GT after adjacency recovery, so adjusted
    # F1 strictly improves over the strict (which would score 0/1/1).
    assert adjusted.tp >= 1


# =============================================================================
# aggregate_summary — rolls adjusted PRFs across cells
# =============================================================================


def _ok_cell(processor: str, *, strict_tp: int, adjusted_tp: int | None) -> CellResult:
    strict = PRF(tp=strict_tp, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
    adj = (
        PRF(tp=adjusted_tp, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
        if adjusted_tp is not None
        else None
    )
    return CellResult(
        page_id=f"p{strict_tp}",
        domain="x",
        processor=processor,
        version="v1",
        status="ok",
        strict=strict,
        adjusted=adj,
        per_class={},
        structural=StructuralErrors(misses=0, hallucinations=0, misclassifications=0, confusion={}),
    )


def _summary(cells: list[CellResult], scorer: str = "f1") -> RunSummary:
    return RunSummary(
        run_id="r",
        started_at=0.0,
        finished_at=1.0,
        catalog_sha="sha",
        processors=sorted({c.processor for c in cells}),
        n_pages=len(cells),
        cells=cells,
        scorer=scorer,  # type: ignore[arg-type]
    )


def test_aggregate_summary_includes_adjusted_when_present():
    cells = [_ok_cell("p", strict_tp=3, adjusted_tp=4), _ok_cell("p", strict_tp=2, adjusted_tp=3)]
    out = aggregate_summary(_summary(cells, scorer="adjacency"))
    agg = out["p"]
    assert agg["adjusted"] is not None
    assert agg["adjusted"]["tp"] == 7  # 4 + 3 pooled


def test_aggregate_summary_adjusted_none_under_f1_scorer():
    """When every cell's adjusted is None (the f1 scorer), the rolled-up
    'adjusted' is also None — the report should hide its column."""
    cells = [_ok_cell("p", strict_tp=3, adjusted_tp=None), _ok_cell("p", strict_tp=2, adjusted_tp=None)]
    out = aggregate_summary(_summary(cells))
    assert out["p"]["adjusted"] is None
    # strict still rolls up normally.
    assert out["p"]["strict"]["tp"] == 5
