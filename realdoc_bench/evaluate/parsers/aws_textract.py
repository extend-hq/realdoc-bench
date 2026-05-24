"""AWS Textract parse adapter — PDF in, markdown out.

Textract's synchronous ``analyze_document`` only accepts single-page docs
(image or 1-page PDF). For multi-page PDFs we render each page to a PNG
via pymupdf and call ``analyze_document`` per page in a small thread pool,
then stitch per-page markdown — same shape as the VLM parser base.

The Textract response is a flat ``Blocks[]`` graph (LINE / WORD / TABLE /
CELL / LAYOUT_*) connected by ``Relationships``. We do a minimal honest
serialization:

- LAYOUT_TITLE → ``#``, LAYOUT_SECTION_HEADER → ``##``.
- LAYOUT_TABLE → emit the table as a pipe-table (walk CELL rows/columns).
- LAYOUT_TEXT / LAYOUT_LIST → emit child LINE text in order.
- LAYOUT_PAGE_NUMBER / LAYOUT_HEADER / LAYOUT_FOOTER → skipped.

Per the master plan: the JSON-to-markdown serializer **is part of the
system under test**. This is the minimal honest version — don't optimize.

Auth: default boto3 credential chain (``AWS_PROFILE`` env var honored).
Pricing: ``aws_textract`` in catalog.yaml (per-page rate).
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.evaluate.parsers.base import (
    ParseProvider,
    ParseResult,
    register_parser,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost


_HEADING_TYPES = {"LAYOUT_TITLE": "# ", "LAYOUT_SECTION_HEADER": "## "}
_SKIP_LAYOUT_TYPES = {"LAYOUT_PAGE_NUMBER", "LAYOUT_HEADER", "LAYOUT_FOOTER"}


def _bbox_of(block: dict) -> tuple[float, float, float, float] | None:
    """Return (left, top, right, bottom) in normalized 0..1 page coords."""
    geom = block.get("Geometry") or {}
    bb = geom.get("BoundingBox")
    if not bb:
        return None
    L = bb.get("Left", 0.0); T = bb.get("Top", 0.0)
    W = bb.get("Width", 0.0); H = bb.get("Height", 0.0)
    return (L, T, L + W, T + H)


def _bbox_inside(inner: tuple, outer: tuple, tol: float = 0.005) -> bool:
    """True if inner bbox is mostly inside outer bbox (with small tolerance)."""
    iL, iT, iR, iB = inner
    oL, oT, oR, oB = outer
    return (iL >= oL - tol and iT >= oT - tol
            and iR <= oR + tol and iB <= oB + tol)


def _line_ids_inside_tables(blocks: list[dict]) -> set[str]:
    """Return LINE block IDs whose bbox is inside any TABLE block.

    Textract emits LINE blocks for every visible text line on the page,
    INCLUDING text inside table cells. Those LINEs and the TABLE block
    describe the same content via different lenses. Without dedup, we'd
    emit both — a vertical cell-by-cell text dump above the markdown table.
    """
    table_bboxes: list[tuple[float, float, float, float]] = []
    for b in blocks:
        if b.get("BlockType") == "TABLE":
            bb = _bbox_of(b)
            if bb is not None:
                table_bboxes.append(bb)
    if not table_bboxes:
        return set()
    inside: set[str] = set()
    for b in blocks:
        if b.get("BlockType") != "LINE":
            continue
        bb = _bbox_of(b)
        if bb is None:
            continue
        if any(_bbox_inside(bb, tb) for tb in table_bboxes):
            inside.add(b["Id"])
    return inside


def _render_pdf_pages(pdf_path: Path, dpi: int = 150) -> list[bytes]:
    import pymupdf

    pngs: list[bytes] = []
    doc = pymupdf.open(str(pdf_path))
    try:
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            pngs.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pngs


def _index_blocks(blocks: list[dict]) -> dict[str, dict]:
    return {b["Id"]: b for b in blocks}


def _child_ids(block: dict) -> list[str]:
    out: list[str] = []
    for rel in block.get("Relationships", []) or []:
        if rel.get("Type") == "CHILD":
            out.extend(rel.get("Ids", []))
    return out


def _line_text(block: dict) -> str:
    return (block.get("Text") or "").strip()


def _cell_text(cell: dict, by_id: dict[str, dict]) -> str:
    parts: list[str] = []
    for cid in _child_ids(cell):
        c = by_id.get(cid)
        if c is None:
            continue
        if c.get("BlockType") == "WORD":
            t = c.get("Text") or ""
            if t:
                parts.append(t)
    return " ".join(parts).replace("|", "\\|")


def _table_to_md(table: dict, by_id: dict[str, dict]) -> str:
    cells = [by_id[i] for i in _child_ids(table) if by_id.get(i, {}).get("BlockType") == "CELL"]
    if not cells:
        return ""
    rows = max(c.get("RowIndex", 1) for c in cells)
    cols = max(c.get("ColumnIndex", 1) for c in cells)
    grid: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    for c in cells:
        r = c.get("RowIndex", 1) - 1
        col = c.get("ColumnIndex", 1) - 1
        if 0 <= r < rows and 0 <= col < cols:
            grid[r][col] = _cell_text(c, by_id)
    lines = ["| " + " | ".join(grid[0]) + " |", "|" + "|".join(["---"] * cols) + "|"]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _layout_block_to_md(layout: dict, by_id: dict[str, dict],
                        lines_inside_tables: set[str] | None = None) -> str:
    btype = layout.get("BlockType", "")
    if btype in _SKIP_LAYOUT_TYPES:
        return ""
    lines_inside_tables = lines_inside_tables or set()
    # Walk children. Skip LINE blocks whose geometry is inside any TABLE block
    # (their content is duplicated in the markdown table emitted below).
    line_texts: list[str] = []
    table_md: list[str] = []
    for cid in _child_ids(layout):
        c = by_id.get(cid)
        if c is None:
            continue
        bt = c.get("BlockType")
        if bt == "LINE":
            if cid in lines_inside_tables:
                continue  # dedup: line is inside a table block
            t = _line_text(c)
            if t:
                line_texts.append(t)
        elif bt == "TABLE":
            md = _table_to_md(c, by_id)
            if md:
                table_md.append(md)
    # Defensive dedup: if this layout owns a TABLE child, the remaining
    # un-flagged LINEs are still likely duplicates → drop them.
    if table_md:
        line_texts = []
    body = "\n".join(line_texts)
    if btype in _HEADING_TYPES and body:
        body = _HEADING_TYPES[btype] + body
    parts = [body] if body else []
    parts.extend(table_md)
    return "\n\n".join(p for p in parts if p)


def _response_to_md(response: dict) -> str:
    blocks = response.get("Blocks", []) or []
    by_id = _index_blocks(blocks)
    layout_blocks = [b for b in blocks if b.get("BlockType", "").startswith("LAYOUT_")]
    # Precompute LINE IDs that are inside any TABLE block by geometry — used
    # to dedupe paragraph-style emission of text that's already in a table.
    lines_inside_tables = _line_ids_inside_tables(blocks)
    parts: list[str] = []
    seen_table_ids: set[str] = set()
    for lb in layout_blocks:
        md = _layout_block_to_md(lb, by_id, lines_inside_tables)
        if md:
            parts.append(md)
        # Track tables surfaced via LAYOUT_TABLE so we don't dup them below.
        for cid in _child_ids(lb):
            if by_id.get(cid, {}).get("BlockType") == "TABLE":
                seen_table_ids.add(cid)
    # If layout never anchored a table, walk top-level LINE blocks (skipping
    # those inside table bboxes) so we still get the non-table prose.
    if not layout_blocks:
        for b in blocks:
            if b.get("BlockType") == "LINE" and b["Id"] not in lines_inside_tables:
                t = _line_text(b)
                if t:
                    parts.append(t)
    for b in blocks:
        if b.get("BlockType") == "TABLE" and b["Id"] not in seen_table_ids:
            md = _table_to_md(b, by_id)
            if md:
                parts.append(md)
    return "\n\n".join(parts)


@register_parser("aws_textract", version="1.0.0")
class AWSTextractParser(ParseProvider):
    """AWS Textract via per-page rendering + sync analyze_document."""

    features: list[str] = ["TABLES", "FORMS", "LAYOUT", "SIGNATURES"]
    dpi: int = 150
    page_concurrency: int = 4

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import boto3

            profile = os.environ.get("AWS_PROFILE")
            region = os.environ.get("AWS_REGION", "us-east-1")
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self._client = session.client("textract", region_name=region)
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            f"{self.name}|{self.version}|features={','.join(sorted(self.features))}|dpi={self.dpi}"
        )[:7]

    def _analyze_page(self, png_bytes: bytes) -> dict:
        client = self._ensure_client()
        return client.analyze_document(
            Document={"Bytes": png_bytes}, FeatureTypes=self.features
        )

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        t0 = time.perf_counter()
        pages = _render_pdf_pages(pdf_path, self.dpi)
        concurrency = max(1, int(os.environ.get("RDB_PAGE_CONCURRENCY", self.page_concurrency)))

        page_responses: list[dict] = [{}] * len(pages)
        if concurrency == 1 or len(pages) <= 1:
            for i, png in enumerate(pages):
                page_responses[i] = self._analyze_page(png)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = {pool.submit(self._analyze_page, png): i for i, png in enumerate(pages)}
                for fut in concurrent.futures.as_completed(futs):
                    page_responses[futs[fut]] = fut.result()
        latency = time.perf_counter() - t0

        page_md = [
            f"## Page {i + 1}\n\n{_response_to_md(r)}".rstrip()
            for i, r in enumerate(page_responses)
        ]
        markdown = "\n\n".join(page_md)

        return ParseResult(
            markdown=markdown,
            page_count=len(pages),
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=len(pages)),
            pages_processed=len(pages),
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
