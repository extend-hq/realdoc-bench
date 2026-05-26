"""Layout block / document data model.

The block-type vocabulary aligns with extend v2's native block-type
enum, surfacing the nine classes the benchmark actually evaluates.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
from pydantic import BaseModel, field_validator

LayoutBlockType = Literal[
    "text",
    "heading",
    "section_heading",
    "header",
    "footer",
    "page_number",
    "figure",
    "table",
    "key_value",
]

# Backwards-compat coercion for older 18-class block-type strings that
# may still appear in cached ``prediction.json`` files on disk. Anything
# extend v2 cannot natively emit (or that this benchmark folds away) maps
# into the nearest surviving class.
_LEGACY_BLOCK_TYPE_REMAP: dict[str, str] = {
    "title": "heading",
    "section_header": "section_heading",
    "page_header": "header",
    "page_footer": "footer",
    "formula": "text",
    "list_item": "text",
    "footnote": "text",
    "signature": "text",
    "figure_caption": "text",
    "figure_footnote": "text",
    "table_caption": "text",
    "table_footer": "text",
    "table_cell": "table",
    "barcode": "figure",
    "watermark": "text",
}


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

    def right(self) -> int:
        return self.x + self.w

    def bottom(self) -> int:
        return self.y + self.h

    def area(self) -> int:
        return int(self.w * self.h) if self.w > 0 and self.h > 0 else 0

    def intersection(self, other: BBox) -> BBox | None:
        ix0 = max(self.x, other.x)
        iy0 = max(self.y, other.y)
        ix1 = min(self.right(), other.right())
        iy1 = min(self.bottom(), other.bottom())
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw <= 0 or ih <= 0:
            return None
        return BBox(x=ix0, y=iy0, w=iw, h=ih)

    def inter_area(self, other: BBox) -> int:
        inter = self.intersection(other)
        return inter.area() if inter is not None else 0

    def iou(self, other: BBox) -> float:
        inter = self.inter_area(other)
        if inter == 0:
            return 0.0
        union_area = self.area() + other.area() - inter
        return float(inter / union_area) if union_area > 0 else 0.0

    def contains(self, other: BBox, *, inclusive: bool = True) -> bool:
        """True if ``other`` lies wholly within this box."""
        if inclusive:
            return (
                self.x <= other.x
                and self.y <= other.y
                and self.right() >= other.right()
                and self.bottom() >= other.bottom()
            )
        return (
            self.x < other.x
            and self.y < other.y
            and self.right() > other.right()
            and self.bottom() > other.bottom()
        )

    def overlap_ratio(self, other: BBox, denom: str = "union") -> float:
        """Intersection area as a fraction of the chosen denominator.

        ``denom`` selects the divisor: ``union`` (IoU), ``self``/``other`` (the
        respective box area), or ``min``/``max`` of the two areas.
        """
        inter = self.inter_area(other)
        if inter == 0:
            return 0.0
        if denom == "union":
            return inter / (self.area() + other.area() - inter)
        if denom == "self":
            return inter / self.area()
        if denom == "other":
            return inter / other.area()
        if denom == "min":
            return inter / min(self.area(), other.area())
        if denom == "max":
            return inter / max(self.area(), other.area())
        raise ValueError("denom must be one of: union, self, other, min, max")

    @staticmethod
    def from_poly(poly: list[int | float]) -> BBox:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        min_x = int(np.floor(pts[:, 0].min()))
        min_y = int(np.floor(pts[:, 1].min()))
        max_x = int(np.ceil(pts[:, 0].max()))
        max_y = int(np.ceil(pts[:, 1].max()))
        return BBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y)

    @staticmethod
    def union(boxes: Iterable[BBox]) -> BBox:
        boxes = list(boxes)
        if not boxes:
            raise ValueError("Cannot union empty boxes")
        min_x = min(b.x for b in boxes)
        min_y = min(b.y for b in boxes)
        max_x = max(b.x + b.w for b in boxes)
        max_y = max(b.y + b.h for b in boxes)
        return BBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y)


class LayoutBlock(BaseModel):
    id: str | None = None
    block_type: LayoutBlockType
    bbox: BBox | None = None
    confidence: float = 1.0
    text: str | None = None
    ignore: bool = False
    reading_order: int | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: float | None) -> float:
        return 1.0 if v is None else float(v)

    @field_validator("block_type", mode="before")
    @classmethod
    def _coerce_block_type(cls, v: str) -> str:
        """Collapse legacy/fine-grained block-type strings into the public 9.

        Lets cached ``prediction.json`` files written under the old 18-class
        vocabulary keep loading after the taxonomy collapse.
        """
        if not isinstance(v, str):
            return v
        return _LEGACY_BLOCK_TYPE_REMAP.get(v, v)


class LayoutPage(BaseModel):
    page_number: int
    width: int | None = None
    height: int | None = None
    blocks: list[LayoutBlock]


class LayoutDocument(BaseModel):
    document_id: str | None = None
    source: str | None = None
    latency_sec: float | None = None
    pages: list[LayoutPage]

    @property
    def blocks(self) -> list[LayoutBlock]:
        out: list[LayoutBlock] = []
        for p in self.pages:
            out.extend(p.blocks)
        return out
