"""ELRND raw-model response → LayoutDocument.

The ``/elrnd_l/infer`` endpoint returns a flat list of detections, each:

    {
      "name": "Text",                        # class name (16-class schema)
      "widget_type": "Text",
      "confidence": 0.95,
      "class": <int>,
      "page_width": <int>,
      "page_height": <int>,
      "box": {"x0": <0..1>, "y0": <0..1>, "x1": <0..1>, "y1": <0..1>},
      "rotation": 0,
      "page": <request-index, 0-based>,
    }

Class names follow the chunkr 16-class schema (qvmdexq lineage). Collapse to
realdoc-bench's :class:`LayoutBlockType` using the same rules the COCO GT
normalizer applies — otherwise pred-vs-GT class labels can't agree on the
ambiguous classes (Caption, Line Number, Watermark).
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

# Mirrors realdoc_bench.layout.normalizers.coco._CATEGORY_ID_TO_BLOCK_TYPE so
# preds and GT agree on the collapsed taxonomy. None = skipped (Watermark).
_ELRND_TO_LAYOUT: dict[str, LayoutBlockType | None] = {
    "Bar/QR Codes": "figure",
    "Caption": "text",
    "Code Block": "text",
    "Footer": "page_footer",
    "Footnote/Endnote": "footnote",
    "Form": "key_value",
    "Form Region / Key-Value Pair": "key_value",
    "Formula": "formula",
    "Graphical Item": "figure",
    "Header": "page_header",
    "Heading": "title",
    "Key Value": "key_value",
    "Legend": "text",
    "Line Number": "text",
    "List Item": "list_item",
    "Page Number": "page_number",
    "Picture/Figure/Image/Chart": "figure",
    "Signature / Signature Block": "signature",
    "Subheading": "section_header",
    "Table": "table",
    "Table_of_contents": "table",
    "Text": "text",
    "Unknown/Other": None,
    "Watermark": None,
}


def _bbox_from_norm(box: dict[str, float], width: int, height: int) -> BBox | None:
    x0 = box.get("x0")
    y0 = box.get("y0")
    x1 = box.get("x1")
    y1 = box.get("y1")
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None
    left = max(0, int(round(x0 * width)))
    top = max(0, int(round(y0 * height)))
    right = max(left, int(round(x1 * width)))
    bottom = max(top, int(round(y1 * height)))
    return BBox(x=left, y=top, w=right - left, h=bottom - top)


def normalize_elrnd(
    detections: list[dict[str, Any]],
    *,
    source: str | None = None,
    page_width: int | None = None,
    page_height: int | None = None,
) -> LayoutDocument:
    """Convert ELRND ``/elrnd_l/infer`` response into a LayoutDocument.

    A single image → single page. Confidence is carried through so callers can
    do downstream threshold sweeping.
    """
    pages_map: dict[int, list[LayoutBlock]] = {}
    page_dims: dict[int, tuple[int, int]] = {}

    for det in detections:
        name = det.get("name") or det.get("widget_type") or ""
        bt = _ELRND_TO_LAYOUT.get(name)
        if bt is None:
            continue

        page_idx = int(det.get("page", 0))
        w = int(det.get("page_width") or page_width or 0)
        h = int(det.get("page_height") or page_height or 0)
        if w and h:
            page_dims[page_idx] = (w, h)

        bbox = _bbox_from_norm(det.get("box") or {}, w, h)
        if bbox is None:
            continue

        confidence = det.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = 1.0

        page_blocks = pages_map.setdefault(page_idx, [])
        page_blocks.append(
            LayoutBlock(
                id=f"elrnd_{page_idx}_{len(page_blocks)}",
                block_type=cast(LayoutBlockType, bt),
                bbox=bbox,
                confidence=float(confidence),
            )
        )

    pages: list[LayoutPage] = []
    for page_idx in sorted(pages_map):
        w, h = page_dims.get(page_idx, (page_width or 0, page_height or 0))
        pages.append(
            LayoutPage(
                page_number=page_idx + 1,
                width=w,
                height=h,
                blocks=pages_map[page_idx],
            )
        )

    return LayoutDocument(document_id=source or "", source=source, pages=pages)
