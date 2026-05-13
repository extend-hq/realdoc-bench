"""COCO ground-truth normalizer for the realdoc-layout dataset.

Adapted from ``ocr-bench/models/chunkr_human/model.py``. Two-step category mapping
folded into one dict: raw 23-class annotations → 18-class realdoc-layout vocabulary.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel

from realdoc_bench.layout.normalizers.base import (
    BBox,
    LayoutBlock,
    LayoutBlockType,
    LayoutDocument,
    LayoutPage,
)


class CocoImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int


class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]  # [x, y, width, height]
    area: float | None = None
    iscrowd: int = 0


# Raw COCO category id → block_type. Skipped ids (Watermark, Unknown/Other) are absent.
_CATEGORY_ID_TO_BLOCK_TYPE: dict[int, str] = {
    0: "figure",          # Bar/QR codes
    1: "text",            # Caption
    2: "text",            # Code block
    3: "page_footer",
    4: "footnote",        # Footnote / endnote
    5: "key_value",       # Form
    6: "formula",
    7: "figure",          # Graphical item
    8: "page_header",
    9: "title",           # Heading
    10: "key_value",
    11: "text",           # Legend
    12: "text",           # Line number
    13: "list_item",
    14: "page_number",
    15: "figure",
    16: "signature",
    17: "section_header", # Subheading
    18: "table",
    19: "table",          # Table of contents
    20: "text",
}
_SKIP_CATEGORY_IDS: frozenset[int] = frozenset({21, 22})


def coco_to_layout_document(
    image: CocoImage,
    annotations: list[CocoAnnotation],
    *,
    source: str | None = None,
) -> LayoutDocument:
    blocks: list[LayoutBlock] = []
    for ann in annotations:
        if ann.category_id in _SKIP_CATEGORY_IDS:
            continue
        block_type = _CATEGORY_ID_TO_BLOCK_TYPE.get(ann.category_id)
        if block_type is None:
            continue
        x, y, w, h = ann.bbox
        blocks.append(
            LayoutBlock(
                id=str(ann.id),
                block_type=cast(LayoutBlockType, block_type),
                bbox=BBox(x=int(x), y=int(y), w=int(w), h=int(h)),
            )
        )
    return LayoutDocument(
        document_id=str(image.id),
        source=source,
        pages=[LayoutPage(page_number=1, width=image.width, height=image.height, blocks=blocks)],
    )


def normalize_coco(data: dict, *, source: str | None = None) -> LayoutDocument:
    """Normalize a COCO-style annotation dict into a LayoutDocument.

    Accepts both single-image envelopes (``{"image": {...}, "annotations": [...]}``)
    and the legacy ``page_info``/``layout_dets`` shape used by some training pipelines.
    """
    image_data = data.get("image") or data.get("page_info") or {}
    if "file_name" not in image_data and "image_path" in image_data:
        image_data = {
            "id": image_data.get("page_no", 1),
            "file_name": image_data.get("image_path", ""),
            "width": image_data.get("width", 0),
            "height": image_data.get("height", 0),
        }
    image = CocoImage.model_validate(image_data)
    raw_annotations = data.get("annotations") or data.get("layout_dets") or []
    annotations = [CocoAnnotation.model_validate(ann) for ann in raw_annotations]
    return coco_to_layout_document(image, annotations, source=source)
