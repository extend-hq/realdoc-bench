"""Tests for the AWS Textract + Azure DI layout normalizers.

These exercise the raw-response → ``LayoutDocument`` mapping with small
hand-built fixtures (no SDK / network), covering the verbatim ocr-bench mapping
(folded to the public 9 classes), geometry projection, reading order, the Azure
roleless-paragraph dedup, and the dropped categories that preserve leaderboard
parity (AWS key-value / list; Azure table cells).
"""

from __future__ import annotations

from realdoc_bench.layout.normalizers.aws_textract import normalize_aws_textract
from realdoc_bench.layout.normalizers.azure_di import normalize_azure_di
from realdoc_bench.layout.normalizers.base import BBox

# =============================================================================
# BBox helpers added for the Azure containment dedup
# =============================================================================


def test_bbox_contains_and_overlap_ratio():
    outer = BBox(x=0, y=0, w=100, h=100)
    inner = BBox(x=10, y=10, w=20, h=20)
    assert outer.contains(inner)
    assert not inner.contains(outer)
    # whole of `inner` is inside `outer` → overlap vs. "other" (inner) is 1.0
    assert outer.overlap_ratio(inner, "other") == 1.0

    half = BBox(x=50, y=0, w=100, h=100)  # 50x100 intersection with outer
    assert abs(outer.overlap_ratio(half) - (5000 / 15000)) < 1e-9


# =============================================================================
# AWS Textract
# =============================================================================


def _aws_block(block_id: str, btype: str, left: float, top: float, w: float, h: float) -> dict:
    return {
        "Id": block_id,
        "BlockType": btype,
        "Geometry": {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}},
    }


def test_aws_maps_layout_regions_and_scales_to_pixels():
    resp = {
        "Blocks": [
            _aws_block("1", "LAYOUT_TITLE", 0.1, 0.0, 0.5, 0.1),  # → heading
            _aws_block("2", "LAYOUT_TEXT", 0.0, 0.2, 1.0, 0.1),  # → text
            _aws_block("3", "LAYOUT_TABLE", 0.0, 0.4, 0.5, 0.2),  # → table
            _aws_block("4", "TABLE_TITLE", 0.0, 0.65, 0.4, 0.05),  # table_caption → text
            _aws_block("5", "SIGNATURE", 0.6, 0.7, 0.2, 0.05),  # signature → text
            # WORD is finer-grained content, not a layout region → dropped.
            _aws_block("6", "WORD", 0.0, 0.0, 0.01, 0.01),
        ]
    }
    doc = normalize_aws_textract(resp, page_width=1000, page_height=2000, source="p.png")
    blocks = doc.blocks

    assert [b.block_type for b in blocks] == ["heading", "text", "table", "text", "text"]
    assert [b.reading_order for b in blocks] == [0, 1, 2, 3, 4]

    title = blocks[0].bbox
    assert (title.x, title.y, title.w, title.h) == (100, 0, 500, 200)
    assert doc.pages[0].width == 1000
    assert doc.pages[0].height == 2000


def test_aws_maps_key_value_but_drops_list():
    """LAYOUT_KEY_VALUE → key_value (realdoc has the class); LAYOUT_LIST dropped."""
    resp = {
        "Blocks": [
            _aws_block("1", "LAYOUT_KEY_VALUE", 0.0, 0.0, 0.3, 0.1),
            _aws_block("2", "LAYOUT_LIST", 0.0, 0.2, 0.5, 0.1),
            _aws_block("3", "LAYOUT_TEXT", 0.0, 0.4, 0.5, 0.1),
        ]
    }
    doc = normalize_aws_textract(resp, page_width=1000, page_height=1000)
    assert [b.block_type for b in doc.blocks] == ["key_value", "text"]


def test_aws_skips_blocks_without_geometry():
    resp = {"Blocks": [{"Id": "1", "BlockType": "LAYOUT_TEXT"}]}
    doc = normalize_aws_textract(resp, page_width=100, page_height=100)
    assert doc.blocks == []


# =============================================================================
# Azure Document Intelligence
# =============================================================================


def _rect_poly(x0: int, y0: int, x1: int, y1: int) -> list[int]:
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def test_azure_emits_regions_orders_by_span_dedups_and_drops_cells():
    data = {
        "pages": [{"pageNumber": 1, "width": 1000, "height": 1000}],
        "tables": [
            {
                "spans": [{"offset": 50, "length": 10}],
                "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(100, 100, 300, 300)}],
                # cell is used for the dedup pass but dropped from the output.
                "cells": [
                    {
                        "rowIndex": 0,
                        "columnIndex": 0,
                        "content": "x",
                        "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(100, 100, 200, 160)}],
                    }
                ],
            }
        ],
        "figures": [
            {
                "spans": [{"offset": 200, "length": 5}],
                "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(400, 400, 500, 500)}],
            }
        ],
        "paragraphs": [
            {
                "role": "title",
                "content": "T",
                "spans": [{"offset": 0, "length": 1}],
                "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(0, 0, 200, 40)}],
            },
            {
                # roleless paragraph wholly inside the table region → dropped.
                "role": None,
                "content": "inside",
                "spans": [{"offset": 52, "length": 3}],
                "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(120, 120, 200, 160)}],
            },
            {
                "role": None,
                "content": "outside",
                "spans": [{"offset": 300, "length": 3}],
                "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(0, 600, 200, 640)}],
            },
        ],
    }
    doc = normalize_azure_di(data, source="p.png")
    blocks = doc.pages[0].blocks

    # table region + figure + title + the one un-contained roleless paragraph.
    # The table cell is dedup-only and excluded from output (ocr-bench parity).
    assert len(blocks) == 4
    assert sorted(b.block_type for b in blocks) == ["figure", "heading", "table", "text"]

    # Reading order tracks span offsets: title(0) < table(50) < figure(200) < outside(300)
    assert [b.block_type for b in blocks] == ["heading", "table", "figure", "text"]
    assert [b.reading_order for b in blocks] == [0, 1, 2, 3]

    assert doc.pages[0].width == 1000
    assert doc.pages[0].height == 1000


def test_azure_table_caption_and_footnote_fold_to_text():
    data = {
        "pages": [{"pageNumber": 1, "width": 1000, "height": 1000}],
        "tables": [
            {
                "spans": [{"offset": 50}],
                "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(100, 500, 400, 700)}],
                "caption": {
                    "spans": [{"offset": 40}],
                    "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(100, 470, 400, 495)}],
                },
                "footnotes": [
                    {
                        "spans": [{"offset": 90}],
                        "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(100, 705, 400, 725)}],
                    }
                ],
            }
        ],
    }
    doc = normalize_azure_di(data)
    # caption (offset 40) → text, table region (50) → table, footnote (90) → text
    assert [b.block_type for b in doc.pages[0].blocks] == ["text", "table", "text"]


def test_azure_maps_page_adornment_roles():
    data = {
        "pages": [{"pageNumber": 1, "width": 800, "height": 1000}],
        "paragraphs": [
            {"role": "pageHeader", "content": "h", "spans": [{"offset": 0, "length": 1}],
             "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(0, 0, 800, 30)}]},
            {"role": "pageFooter", "content": "f", "spans": [{"offset": 10, "length": 1}],
             "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(0, 970, 800, 1000)}]},
            {"role": "pageNumber", "content": "1", "spans": [{"offset": 20, "length": 1}],
             "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(380, 980, 420, 1000)}]},
            {"role": "sectionHeading", "content": "s", "spans": [{"offset": 30, "length": 1}],
             "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(0, 40, 400, 70)}]},
        ],
    }
    doc = normalize_azure_di(data)
    assert [b.block_type for b in doc.pages[0].blocks] == [
        "header",
        "footer",
        "page_number",
        "section_heading",
    ]


def test_azure_keeps_span_less_region_but_still_drops_cells():
    """A table region missing `spans` is surfaced (appended), not dropped; its
    cells (emit=False) stay out of the output regardless."""
    data = {
        "pages": [{"pageNumber": 1, "width": 1000, "height": 1000}],
        "tables": [
            {
                # no "spans" → span offset None, but it's a real region.
                "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(100, 100, 300, 300)}],
                "cells": [
                    {"boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(100, 100, 200, 160)}]}
                ],
            }
        ],
    }
    doc = normalize_azure_di(data)
    # table region kept (1 block); cell excluded.
    assert [b.block_type for b in doc.pages[0].blocks] == ["table"]


def test_azure_malformed_polygon_skipped_not_crash():
    """An odd-length polygon is skipped, not fed to BBox.from_poly (which would
    raise on reshape and abort the whole page)."""
    data = {
        "pages": [{"pageNumber": 1, "width": 1000, "height": 1000}],
        "paragraphs": [
            {"role": None, "content": "bad", "spans": [{"offset": 0}],
             "boundingRegions": [{"pageNumber": 1, "polygon": [10, 20, 30]}]},  # odd length
            {"role": None, "content": "ok", "spans": [{"offset": 5}],
             "boundingRegions": [{"pageNumber": 1, "polygon": _rect_poly(0, 0, 100, 40)}]},
        ],
    }
    doc = normalize_azure_di(data)  # must not raise
    # malformed region dropped; valid one kept.
    assert [b.block_type for b in doc.pages[0].blocks] == ["text"]


def test_azure_empty_result_yields_empty_page():
    doc = normalize_azure_di({"pages": [{"pageNumber": 1, "width": 100, "height": 200}]})
    assert len(doc.pages) == 1
    assert doc.pages[0].blocks == []
    assert doc.pages[0].width == 100
