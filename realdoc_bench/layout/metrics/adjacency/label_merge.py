"""Class-merge profiles applied to predictions + GT before scoring."""

from __future__ import annotations

from typing import Literal

from realdoc_bench.layout.normalizers.base import LayoutBlock

MergeProfile = Literal["core5", "extended", "full"]

CORE5_MERGE: dict[str, str] = {
    "text": "text",
    "list_item": "text",
    "formula": "text",
    "footnote": "text",
    "page_header": "text",
    "page_footer": "text",
    "page_number": "text",
    "signature": "text",
    "key_value": "text",
    "title": "title",
    "section_header": "title",
    "table": "table",
    "table_cell": "table",
    "table_caption": "caption",
    "table_footer": "text",
    "figure": "figure",
    "figure_caption": "caption",
    "figure_footnote": "text",
    "watermark": "text",
    "barcode": "text",
}

EXTENDED_MERGE: dict[str, str] = {
    "text": "text",
    "title": "title",
    "section_header": "section_header",
    "list_item": "list_item",
    "formula": "formula",
    "key_value": "key_value",
    "table": "table",
    "figure": "figure",
    "figure_caption": "figure_caption",
    "table_caption": "figure_caption",
    "footnote": "footnote",
    "page_header": "page_header",
    "page_footer": "page_footer",
    "page_number": "text",
    "signature": "text",
    "table_cell": "table",
    "table_footer": "text",
    "figure_footnote": "text",
    "watermark": "text",
    "barcode": "text",
}


def get_merge_mapping(profile: MergeProfile) -> dict[str, str] | None:
    if profile == "core5":
        return CORE5_MERGE
    if profile == "extended":
        return EXTENDED_MERGE
    return None


def apply_merge_profile(blocks: list[LayoutBlock], profile: MergeProfile) -> list[LayoutBlock]:
    mapping = get_merge_mapping(profile)
    if mapping is None:
        return blocks
    return [
        b.model_copy(update={"block_type": mapping.get(b.block_type, b.block_type)})
        for b in blocks
    ]
