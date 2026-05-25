"""Extend SDK response → LayoutDocument.

Propagates
per-block confidence when the production response carries it (falls back to 1.0
when absent — needed for mAP). No post-hoc filtering by confidence; Extend's
pipeline applies whatever cutoffs it considers production-correct.
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

# Extend SDK ``BlockType`` enum → 9-class public ``LayoutBlockType``.
# Mirrors ``extend_ai.types.block_type.BlockType``: text / heading /
# section_heading / table / figure / table_head / table_cell / key_value /
# page_number / barcode / formula / header / footer. Unknown types drop.
_EXTEND_TO_LAYOUT: dict[str, LayoutBlockType] = {
    "text": "text",
    "heading": "heading",
    "section_heading": "section_heading",
    "header": "header",
    "footer": "footer",
    "page_number": "page_number",
    "figure": "figure",
    "barcode": "figure",
    "table": "table",
    "table_head": "table",  # fold; the GT taxonomy has no separate table-head class
    "key_value": "key_value",
    "formula": "text",       # extend emits formula; the public 9 fold it into text
    # ``table_cell`` is intentionally absent — it is not part of this eval.
}


def _classify(block: Any) -> LayoutBlockType | None:
    raw_type = getattr(block.type, "value", block.type)
    bt = _EXTEND_TO_LAYOUT.get(raw_type)
    if bt is not None:
        return bt
    details = block.details
    if isinstance(details, dict):
        if details.get("type") == "figure_details" and details.get("image_url"):
            return "figure"
        if details.get("type") == "table_details":
            return "table"
    elif details is not None:
        detail_type = getattr(details, "type", None)
        if detail_type == "figure_details" and getattr(details, "image_url", None):
            return "figure"
        if detail_type == "table_details":
            return "table"
    return None


def _confidence(block: Any) -> float:
    """Extract per-block confidence from a few known locations on the Extend response.

    Returns 1.0 when no score is present so mAP remains computable; the report
    annotates the processor as "mAP-N/A" when confidence is uniformly 1.0.
    """
    for attr in ("confidence", "score"):
        value = getattr(block, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    meta = getattr(block, "metadata", None)
    if meta is not None:
        for attr in ("confidence", "score"):
            value = getattr(meta, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
    details = getattr(block, "details", None)
    if isinstance(details, dict):
        value = details.get("confidence") or details.get("score")
        if isinstance(value, (int, float)):
            return float(value)
    return 1.0


def to_layout_document(run: Any, *, source: str | None = None) -> LayoutDocument:
    pages_map: dict[int, list[LayoutBlock]] = {}
    page_dimensions: dict[int, tuple[int, int]] = {}

    output = getattr(run, "output", None)
    chunks = getattr(output, "chunks", None) if output is not None else None
    if chunks is None:
        chunks = getattr(run, "chunks", []) or []
    for chunk in chunks:
        for block in chunk.blocks:
            bt = _classify(block)
            if bt is None:
                continue

            page_num: int | None = None
            meta = block.metadata
            page_meta = getattr(meta, "page", None)
            if page_meta is not None:
                page_num = page_meta.number
                if page_meta.width is not None and page_meta.height is not None:
                    page_dimensions[page_num] = (int(round(page_meta.width)), int(round(page_meta.height)))
            elif getattr(meta, "page_number", None) is not None:
                page_num = meta.page_number
            if page_num is None:
                continue

            bb = block.bounding_box
            if bb is None or bb.left is None or bb.top is None or bb.right is None or bb.bottom is None:
                continue
            bbox = BBox(
                x=int(round(bb.left)),
                y=int(round(bb.top)),
                w=int(round(bb.right - bb.left)),
                h=int(round(bb.bottom - bb.top)),
            )

            page_blocks = pages_map.setdefault(page_num, [])
            ro = getattr(meta, "reading_order", None)
            if ro is None:
                ro = len(page_blocks)
            page_blocks.append(
                LayoutBlock(
                    id=block.id,
                    block_type=cast(LayoutBlockType, bt),
                    bbox=bbox,
                    confidence=_confidence(block),
                    text=block.content,
                    reading_order=ro,
                )
            )

    pages: list[LayoutPage] = []
    for page_num in sorted(pages_map):
        w, h = page_dimensions.get(page_num, (0, 0))
        pages.append(LayoutPage(page_number=page_num, width=w, height=h, blocks=pages_map[page_num]))

    return LayoutDocument(
        document_id=getattr(run, "id", None),
        source=source,
        pages=pages,
        latency_sec=getattr(run, "latency_sec", None),
    )
