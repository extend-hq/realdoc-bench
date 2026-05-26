"""Paddle PP-Structure V3 response → LayoutDocument.


PP-Structure V3 emits a single page with a flat ``parsing_res_list`` of blocks
labelled by ``block_label`` and bounded by ``block_bbox`` (LTRB).
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

_PADDLE_TO_LAYOUT: dict[str, LayoutBlockType] = {
    "text": "text",
    "doc_title": "heading",
    "paragraph_title": "section_heading",
    "header": "header",
    "footer": "footer",
    "footnote": "text",
    "caption": "text",
    "page_number": "page_number",
    "table": "table",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "formula": "text",
    "seal": "figure",
    "Text": "text",
    "Heading": "heading",
    "Subheading": "section_heading",
    "Header": "header",
    "Footer": "footer",
    "Caption": "text",
    "Page Number": "page_number",
    "Table": "table",
    "Picture": "figure",
    "Figure": "figure",
    "Image": "figure",
    "Chart": "figure",
    "Formula": "text",
    "List Item": "text",
    "Form Region": "key_value",
    "Key-Value Pair": "key_value",
    "Bar/QR Codes": "figure",
}


def _bbox_from_ltrb(ltrb: list[float]) -> BBox:
    left, top, right, bottom = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
    return BBox(
        x=int(round(left)),
        y=int(round(top)),
        w=int(round(right - left)),
        h=int(round(bottom - top)),
    )


def normalize_paddle(data: dict[str, Any], *, source: str | None = None) -> LayoutDocument:
    pages = data.get("pages") or []
    res = pages[0].get("res", {}) if pages else {}
    blocks_raw = res.get("parsing_res_list", [])
    width = int(res.get("width", 0) or 0)
    height = int(res.get("height", 0) or 0)

    blocks: list[LayoutBlock] = []
    for block in blocks_raw:
        bt = _PADDLE_TO_LAYOUT.get(block.get("block_label", ""))
        if bt is None:
            continue
        bbox_raw = block.get("block_bbox")
        if not bbox_raw or len(bbox_raw) < 4:
            continue
        block_id = str(block.get("block_id", len(blocks)))
        blocks.append(
            LayoutBlock(
                id=f"block_{block_id}",
                block_type=cast(LayoutBlockType, bt),
                bbox=_bbox_from_ltrb(bbox_raw),
                text=block.get("block_content") or block.get("block_text") or "",
                reading_order=block.get("block_order"),
            )
        )

    return LayoutDocument(
        document_id=source or "",
        source=source,
        pages=[LayoutPage(page_number=1, width=width, height=height, blocks=blocks)],
    )
