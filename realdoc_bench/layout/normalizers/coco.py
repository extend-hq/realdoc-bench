"""COCO ground-truth normalizer for the realdoc-layout dataset.

The dataset's per-file ``categories`` block names the 9 public block
classes directly; this normalizer looks each ``category_id`` up in that
block, takes the ``name``, and uses it as the ``block_type``. A
:class:`LayoutBlock` ``field_validator`` collapses any legacy 18-class
string into the 9-class equivalent, so older annotation files that still
carry the raw COCO names also load correctly.
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


class CocoCategory(BaseModel):
    id: int
    name: str
    supercategory: str | None = None


# Fold legacy raw COCO names into the 9 public class names so older
# annotation files (pre-collapse) also load. Names that already match a
# public class pass through unchanged via the ``.get(name, name)`` default.
_LEGACY_NAME_TO_BLOCK_TYPE: dict[str, str] = {
    "Bar/QR Codes": "figure",
    "Caption": "text",
    "Code Block": "text",
    "Footer": "footer",
    "Footnote/Endnote": "text",
    "Form": "key_value",
    "Formula": "text",
    "Graphical Item": "figure",
    "Header": "header",
    "Heading": "heading",
    "Key Value": "key_value",
    "Legend": "text",
    "Line Number": "text",
    "List Item": "text",
    "Page Number": "page_number",
    "Picture/Figure/Image/Chart": "figure",
    "Signature / Signature Block": "text",
    "Subheading": "section_heading",
    "Table": "table",
    "Table_of_contents": "table",
    "Text": "text",
    "Unknown/Other": "",  # drop
    "Watermark": "",      # drop
}


def _resolve_block_type(name: str) -> str:
    """Return the public 9-class ``block_type`` for a category name, or ``""``
    if the category should be dropped."""
    return _LEGACY_NAME_TO_BLOCK_TYPE.get(name, name)


def coco_to_layout_document(
    image: CocoImage,
    annotations: list[CocoAnnotation],
    categories: list[CocoCategory],
    *,
    source: str | None = None,
) -> LayoutDocument:
    id_to_block_type: dict[int, str] = {}
    for cat in categories:
        bt = _resolve_block_type(cat.name)
        if bt:
            id_to_block_type[cat.id] = bt

    blocks: list[LayoutBlock] = []
    for ann in annotations:
        block_type = id_to_block_type.get(ann.category_id)
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

    Accepts both single-image envelopes (``{"image": {...}, "annotations": [...],
    "categories": [...]}``) and the legacy ``page_info``/``layout_dets`` shape
    used by some training pipelines.
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
    categories = [CocoCategory.model_validate(c) for c in (data.get("categories") or [])]
    return coco_to_layout_document(image, annotations, categories, source=source)
