"""Azure Document Intelligence ``prebuilt-layout`` response → LayoutDocument.

Walks the ``as_dict()`` form of the analyze result (camelCase keys, matching
the REST wire shape). This is a **verbatim port of ocr-bench's** Azure
``to_layout_document`` (extend-hq/ocr-bench
``models/azure_document_intelligence/prebuilt_layout/model.py``) so the
processor reproduces the published layout-leaderboard rows:

- ``tables[]`` → one ``table`` block per bounding region; captions →
  ``table_caption``, footnotes → ``table_footer``; cells → ``table_cell``.
- ``figures[]`` → ``figure``; captions → ``figure_caption``.
- ``paragraphs[]`` → role-mapped; a roleless paragraph becomes ``text`` unless
  it sits inside an already-emitted region (containment dedup against tables /
  cells / figures so cell text isn't double-counted).

Two ocr-bench behaviors are reproduced deliberately: (1) reading order follows
each item's first span offset; (2) table **cells** join the containment-dedup
pass but are excluded from the final blocks (carried as ``emit=False`` rather
than keyed on a missing span, so a region that happens to lack spans is still
surfaced instead of silently dropped). The raw vendor strings emitted
here are folded into the public 9 classes by ``LayoutBlock``'s validator (the
same collapse that ingested ocr-bench's predictions when those rows were
scored): ``table_cell``→table, ``table_caption`` / ``table_footer`` /
``figure_caption`` / ``footnote``→text, ``title``→heading, etc.

Geometry is pixel-space polygons: the input is a page image, so ``unit`` is
``pixel`` and the coordinates align with the COCO ground truth.
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

# Verbatim from ocr-bench (_AZURE_PARA_ROLE_TO_LAYOUT). Values are ocr-bench's
# 15-class names; LayoutBlock's validator folds them to the public 9.
_AZURE_ROLE_TO_LAYOUT: dict[str, str] = {
    "title": "title",
    "sectionHeading": "section_header",
    "pageHeader": "page_header",
    "pageFooter": "page_footer",
    "pageNumber": "page_number",
    "footnote": "footnote",
}

# A roleless paragraph this far covered by an existing region is treated as part
# of it (e.g. text inside a table) and dropped.
_CONTAINMENT_OVERLAP = 0.90


def _span_offset(spans: list[dict] | None) -> int | None:
    """First span offset — the reading-order key Azure assigns."""
    if not spans:
        return None
    return spans[0].get("offset")


def _region_bbox(region: dict) -> BBox | None:
    poly = region.get("polygon")
    # Need an even count of >= 4 ints (>= 2 points) to form a box; anything
    # malformed (e.g. a truncated odd-length payload) is skipped rather than
    # letting BBox.from_poly's reshape(-1, 2) raise and abort the whole page.
    if not poly or len(poly) < 4 or len(poly) % 2:
        return None
    return BBox.from_poly(poly)


def _block(block_type: str, bbox: BBox, *, text: str | None = None) -> LayoutBlock:
    return LayoutBlock(block_type=cast(LayoutBlockType, block_type), bbox=bbox, text=text)


def _contained(bb: BBox, existing: list[tuple[int | None, LayoutBlock, bool]]) -> bool:
    for _, blk, _emit in existing:
        if blk.bbox is None:
            continue
        if blk.bbox.contains(bb, inclusive=True):
            return True
        if blk.bbox.overlap_ratio(bb, "other") >= _CONTAINMENT_OVERLAP:
            return True
    return False


def normalize_azure_di(data: dict[str, Any], *, source: str | None = None) -> LayoutDocument:
    # page_number -> list[(span_offset | None, block, emit)]. ``emit=False``
    # blocks (table cells) join the containment-dedup pass but are excluded
    # from the final output, reproducing ocr-bench.
    page_blocks: dict[int, list[tuple[int | None, LayoutBlock, bool]]] = {}

    def add(page_num: int, span_offset: int | None, block: LayoutBlock, *, emit: bool = True) -> None:
        page_blocks.setdefault(page_num, []).append((span_offset, block, emit))

    for tbl in data.get("tables") or []:
        so = _span_offset(tbl.get("spans"))
        for br in tbl.get("boundingRegions") or []:
            bb = _region_bbox(br)
            if bb is not None:
                add(br.get("pageNumber", 1), so, _block("table", bb))
        cap = tbl.get("caption")
        if cap:
            cso = _span_offset(cap.get("spans"))
            for br in cap.get("boundingRegions") or []:
                bb = _region_bbox(br)
                if bb is not None:
                    add(br.get("pageNumber", 1), cso, _block("table_caption", bb))
        for fn in tbl.get("footnotes") or []:
            fso = _span_offset(fn.get("spans"))
            for br in fn.get("boundingRegions") or []:
                bb = _region_bbox(br)
                if bb is not None:
                    add(br.get("pageNumber", 1), fso, _block("table_footer", bb))
        # Cells join the dedup pass but are excluded from output (emit=False).
        for cell in tbl.get("cells") or []:
            for br in cell.get("boundingRegions") or []:
                bb = _region_bbox(br)
                if bb is not None:
                    add(br.get("pageNumber", 1), None, _block("table_cell", bb), emit=False)

    for fig in data.get("figures") or []:
        so = _span_offset(fig.get("spans"))
        for br in fig.get("boundingRegions") or []:
            bb = _region_bbox(br)
            if bb is not None:
                add(br.get("pageNumber", 1), so, _block("figure", bb))
        cap = fig.get("caption")
        if cap:
            cso = _span_offset(cap.get("spans"))
            for br in cap.get("boundingRegions") or []:
                bb = _region_bbox(br)
                if bb is not None:
                    add(br.get("pageNumber", 1), cso, _block("figure_caption", bb))

    for para in data.get("paragraphs") or []:
        role = para.get("role")
        layout_type = _AZURE_ROLE_TO_LAYOUT.get(role) if role else None
        so = _span_offset(para.get("spans"))
        content = para.get("content")
        for br in para.get("boundingRegions") or []:
            bb = _region_bbox(br)
            if bb is None:
                continue
            page_num = br.get("pageNumber", 1)
            if layout_type is not None:
                add(page_num, so, _block(layout_type, bb, text=content))
            elif not _contained(bb, page_blocks.get(page_num, [])):
                add(page_num, so, _block("text", bb, text=content))

    dims: dict[int, tuple[int | None, int | None]] = {
        p.get("pageNumber", 1): (p.get("width"), p.get("height")) for p in data.get("pages") or []
    }

    pages: list[LayoutPage] = []
    for page_num, items in sorted(page_blocks.items()):
        # Cells (emit=False) were dedup-only; everything else is output. Blocks
        # with a span offset are ordered by it; a span-less region (rare — Azure
        # normally spans every table/figure/paragraph) is appended afterward so
        # it is surfaced rather than silently dropped.
        emitted = [(so, blk) for so, blk, emit in items if emit]
        with_order = sorted(
            ((so, blk) for so, blk in emitted if so is not None), key=lambda x: x[0]
        )
        for reading_order, (_, blk) in enumerate(with_order):
            blk.reading_order = reading_order
        ordered = [blk for _, blk in with_order] + [blk for so, blk in emitted if so is None]
        width, height = dims.get(page_num, (None, None))
        pages.append(LayoutPage(page_number=page_num, width=width, height=height, blocks=ordered))

    if not pages:
        width, height = next(iter(dims.values()), (None, None))
        pages = [LayoutPage(page_number=1, width=width, height=height, blocks=[])]

    return LayoutDocument(document_id=source or "", source=source, pages=pages)
