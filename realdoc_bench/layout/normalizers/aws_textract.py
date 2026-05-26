"""AWS Textract ``analyze_document`` response → LayoutDocument.

Textract returns a flat ``Blocks[]`` graph; the page-layout regions are the
``LAYOUT_*`` blocks (plus a top-level ``SIGNATURE`` block when the SIGNATURES
feature is on, and ``TABLE_TITLE`` / ``TABLE_FOOTER`` sub-blocks). Block
geometry is normalized to 0..1 against the page, so the caller passes the
source image's pixel dimensions to project boxes back into pixel space — the
space the COCO ground truth lives in.

The block-type map below mirrors ocr-bench's ``_AWS_TO_LAYOUT`` (extend-hq/
ocr-bench ``models/aws_textract/analyze_document/model.py``), with one
deliberate divergence: ``LAYOUT_KEY_VALUE`` → ``key_value`` (see the map
comment). ``LAYOUT_LIST`` stays dropped, as in ocr-bench. The raw vendor strings
emitted here are folded into the public 9 classes by ``LayoutBlock``'s
``block_type`` validator — the same collapse that ingested ocr-bench's
predictions when the leaderboard rows were first scored.
"""

from __future__ import annotations

from typing import Any, cast

from realdoc_bench.layout.normalizers.base import (
    BBox,
    LayoutBlock,
    LayoutBlockType,
    LayoutDocument,
    LayoutPage,
)

# Mostly verbatim from ocr-bench. Values are ocr-bench's 15-class names;
# LayoutBlock's validator folds them (title→heading, section_header→
# section_heading, page_header→header, page_footer→footer, signature/
# table_caption/table_footer→text).
#
# One deliberate divergence from ocr-bench: ``LAYOUT_KEY_VALUE`` → ``key_value``.
# ocr-bench's 15-class vocab had no key-value class so it dropped these regions;
# realdoc's taxonomy has ``key_value`` (and the GT carries 4774 of them), so
# dropping Textract's key-value regions would discard genuine, reasonably
# precise detections (P≈0.46). ``LAYOUT_LIST`` stays dropped, matching ocr-bench
# (mapping it to text adds net false positives here).
_AWS_TO_LAYOUT: dict[str, str] = {
    "LAYOUT_TABLE": "table",
    "LAYOUT_FIGURE": "figure",
    "LAYOUT_TITLE": "title",
    "LAYOUT_SECTION_HEADER": "section_header",
    "LAYOUT_HEADER": "page_header",
    "LAYOUT_FOOTER": "page_footer",
    "LAYOUT_PAGE_NUMBER": "page_number",
    "LAYOUT_TEXT": "text",
    "LAYOUT_KEY_VALUE": "key_value",
    "SIGNATURE": "signature",
    "TABLE_TITLE": "table_caption",
    "TABLE_FOOTER": "table_footer",
}


def normalize_aws_textract(
    data: dict[str, Any],
    *,
    page_width: int,
    page_height: int,
    source: str | None = None,
) -> LayoutDocument:
    """Project Textract layout regions onto a single-page ``LayoutDocument``.

    ``page_width`` / ``page_height`` are the source image's pixel dimensions,
    used to scale Textract's normalized 0..1 geometry into pixels. Blocks are
    emitted in Textract's order with sequential reading-order indices, matching
    ocr-bench.
    """
    blocks: list[LayoutBlock] = []
    reading_order = 0
    for b in data.get("Blocks") or []:
        bt = _AWS_TO_LAYOUT.get(b.get("BlockType", ""))
        if bt is None:
            continue
        bb = (b.get("Geometry") or {}).get("BoundingBox")
        if not bb:
            continue
        left = float(bb.get("Left", 0.0))
        top = float(bb.get("Top", 0.0))
        width = float(bb.get("Width", 0.0))
        height = float(bb.get("Height", 0.0))
        blocks.append(
            LayoutBlock(
                id=b.get("Id"),
                block_type=cast(LayoutBlockType, bt),
                bbox=BBox(
                    x=round(left * page_width),
                    y=round(top * page_height),
                    w=round(width * page_width),
                    h=round(height * page_height),
                ),
                reading_order=reading_order,
            )
        )
        reading_order += 1

    return LayoutDocument(
        document_id=source or "",
        source=source,
        pages=[LayoutPage(page_number=1, width=page_width, height=page_height, blocks=blocks)],
    )
