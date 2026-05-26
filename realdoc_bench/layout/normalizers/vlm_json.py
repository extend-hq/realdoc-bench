"""VLM JSON response → LayoutDocument.

Used by VLM layout processors that emit JSON (e.g. ``dots_ocr``). They prompt the
model to emit a JSON object whose shape mirrors Extend's ``ParserRun``
(chunks → blocks with ``type``, ``content``, ``bounding_box``, and
``metadata.page``). This normalizer walks that dict and produces a
LayoutDocument directly — no SDK objects involved.

Block-type mapping mirrors the Extend normalizer; the prompts use the same
vocabulary so cross-comparisons are honest.
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

_VLM_TO_LAYOUT: dict[str, LayoutBlockType] = {
    "text": "text",
    "heading": "heading",
    "title": "heading",
    "section_heading": "section_heading",
    "section_header": "section_heading",
    "header": "header",
    "page_header": "header",
    "footer": "footer",
    "page_footer": "footer",
    "page_number": "page_number",
    "figure": "figure",
    "barcode": "figure",
    "table": "table",
    "table_head": "table",
    "key_value": "key_value",
    # The public 9 fold these into text; we accept the legacy aliases for
    # backward-compat with prompts/responses that still emit them.
    "caption": "text",
    "figure_caption": "text",
    "formula": "text",
    "list_item": "text",
    "footnote": "text",
    "signature": "text",
}


def _bbox_from(node: dict[str, Any] | None) -> BBox | None:
    if not node:
        return None
    left = node.get("left")
    top = node.get("top")
    right = node.get("right")
    bottom = node.get("bottom")
    if left is None or top is None or right is None or bottom is None:
        # Fallback: some prompts return [x, y, w, h] or [l, t, r, b] lists
        return None
    return BBox(
        x=int(round(left)),
        y=int(round(top)),
        w=int(round(right - left)),
        h=int(round(bottom - top)),
    )


def normalize_vlm_json(data: dict[str, Any], *, source: str | None = None) -> LayoutDocument:
    pages_map: dict[int, list[LayoutBlock]] = {}
    page_dims: dict[int, tuple[int, int]] = {}

    for chunk in data.get("chunks", []) or []:
        for block in chunk.get("blocks", []) or []:
            block_type_raw = (block.get("type") or "").lower()
            bt = _VLM_TO_LAYOUT.get(block_type_raw)
            if bt is None:
                continue

            meta = block.get("metadata") or {}
            page_meta = meta.get("page") or {}
            page_num = page_meta.get("number", 1)
            if page_meta.get("width") and page_meta.get("height"):
                page_dims[page_num] = (int(page_meta["width"]), int(page_meta["height"]))

            bbox = _bbox_from(block.get("bounding_box") or block.get("boundingBox"))
            if bbox is None:
                continue

            page_blocks = pages_map.setdefault(page_num, [])
            page_blocks.append(
                LayoutBlock(
                    id=block.get("id") or f"block_{len(page_blocks)}",
                    block_type=cast(LayoutBlockType, bt),
                    bbox=bbox,
                    text=block.get("content") or "",
                    reading_order=meta.get("reading_order") if isinstance(meta, dict) else None,
                )
            )

    pages: list[LayoutPage] = []
    for page_num in sorted(pages_map):
        w, h = page_dims.get(page_num, (0, 0))
        pages.append(LayoutPage(page_number=page_num, width=w, height=h, blocks=pages_map[page_num]))

    return LayoutDocument(
        document_id=data.get("id") or "",
        source=source,
        pages=pages,
    )
