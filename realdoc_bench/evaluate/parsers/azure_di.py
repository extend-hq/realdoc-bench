"""Azure Document Intelligence parse adapter — PDF in, markdown out.

Uses ``prebuilt-layout``, which handles multi-page PDFs natively (poller).
The structured response (``paragraphs[]``, ``tables[]``, ``pages[]``) is
serialized into a minimal-but-honest markdown:

- ``paragraphs`` are emitted in their natural order. ``role`` is mapped to
  heading levels (``title`` → ``#``, ``sectionHeading`` → ``##``,
  ``pageHeader`` / ``pageFooter`` / ``pageNumber`` skipped — they tend to
  inflate the agent's context without helping extraction).
- ``tables`` are emitted as pipe-tables right after the page they belong
  to (preserves reading order and lets the agent see structure).

Per the master plan: the JSON-to-markdown serializer **is part of the
system under test**. This is the minimal honest version — don't optimize.

Auth: ``AZURE_DI_KEY`` + ``AZURE_DI_ENDPOINT`` env vars. Pricing:
``azure_di`` in catalog.yaml (per-page rate).
"""

from __future__ import annotations

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


# Roles whose content we drop from the markdown (noise for downstream agents).
_SKIP_ROLES = {"pageHeader", "pageFooter", "pageNumber"}


def _paragraph_to_md(p: Any) -> str:
    content = (getattr(p, "content", "") or "").strip()
    if not content:
        return ""
    role = getattr(p, "role", None)
    if role in _SKIP_ROLES:
        return ""
    if role == "title":
        return f"# {content}"
    if role == "sectionHeading":
        return f"## {content}"
    return content


def _table_to_md(table: Any) -> str:
    """Serialize an Azure DI table to a pipe-table.

    DI tables don't reliably distinguish header rows from body rows; we
    treat row 0 as the header. Single-cell tables degenerate to plain text.
    """
    rows = getattr(table, "row_count", 0) or 0
    cols = getattr(table, "column_count", 0) or 0
    if rows == 0 or cols == 0:
        return ""

    grid: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in getattr(table, "cells", []) or []:
        r = getattr(cell, "row_index", 0)
        c = getattr(cell, "column_index", 0)
        content = (getattr(cell, "content", "") or "").replace("\n", " ").replace("|", "\\|")
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = content

    lines = ["| " + " | ".join(grid[0]) + " |", "|" + "|".join(["---"] * cols) + "|"]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _page_number_of(item: Any) -> int | None:
    """DI items reference pages via ``bounding_regions[0].page_number``."""
    regions = getattr(item, "bounding_regions", None) or []
    if regions:
        pn = getattr(regions[0], "page_number", None)
        if isinstance(pn, int):
            return pn
    return None


def _spans_of(item: Any) -> list[tuple[int, int]]:
    """Extract (offset, end) tuples from an Azure DI item's spans[]."""
    out: list[tuple[int, int]] = []
    for s in (getattr(item, "spans", None) or []):
        offset = getattr(s, "offset", None)
        length = getattr(s, "length", None)
        if isinstance(offset, int) and isinstance(length, int):
            out.append((offset, offset + length))
    return out


def _paragraph_inside_table(p: Any, table_cell_spans: list[tuple[int, int]]) -> bool:
    """True iff every span of paragraph p falls inside some table cell span.

    Azure DI duplicates a paragraph's content across both ``result.paragraphs[]``
    (sequential page reading order) AND inside ``result.tables[].cells[]``.
    Without dedup, our markdown emits the paragraph form (1 cell per line) AND
    the structured table — wrecking downstream Q&A. We filter out any paragraph
    whose entire text comes from inside a table cell.
    """
    p_spans = _spans_of(p)
    if not p_spans:
        return False
    for (po, pe) in p_spans:
        if not any(co <= po and pe <= ce for (co, ce) in table_cell_spans):
            return False
    return True


def _serialize(result: Any) -> tuple[str, int]:
    """Walk the DI result, emit page-by-page markdown.

    For each page: paragraphs (excluding skipped roles AND those whose text
    falls entirely inside a table cell) in order, then any tables anchored
    to that page after a blank line. Pages are separated by ``\\n\\n---\\n\\n``.
    """
    pages = getattr(result, "pages", None) or []
    paragraphs = getattr(result, "paragraphs", None) or []
    tables = getattr(result, "tables", None) or []

    # Collect every span owned by a table cell — used to dedupe paragraphs
    table_cell_spans: list[tuple[int, int]] = []
    for t in tables:
        for cell in (getattr(t, "cells", None) or []):
            table_cell_spans.extend(_spans_of(cell))

    para_by_page: dict[int, list[Any]] = {}
    for p in paragraphs:
        if _paragraph_inside_table(p, table_cell_spans):
            continue  # dedup — content already emitted as table cell
        pn = _page_number_of(p)
        if pn is not None:
            para_by_page.setdefault(pn, []).append(p)
    table_by_page: dict[int, list[Any]] = {}
    for t in tables:
        pn = _page_number_of(t)
        if pn is not None:
            table_by_page.setdefault(pn, []).append(t)

    chunks: list[str] = []
    n_pages = len(pages) or (max(para_by_page.keys(), default=0) or 1)
    for pn in range(1, n_pages + 1):
        page_parts: list[str] = []
        for p in para_by_page.get(pn, []):
            md = _paragraph_to_md(p)
            if md:
                page_parts.append(md)
        for t in table_by_page.get(pn, []):
            md = _table_to_md(t)
            if md:
                page_parts.append(md)
        if page_parts:
            chunks.append("\n\n".join(page_parts))
    return ("\n\n---\n\n".join(chunks), n_pages)


@register_parser("azure_di", version="1.0.0")
class AzureDIParser(ParseProvider):
    """Azure Document Intelligence — ``prebuilt-layout`` over a multi-page PDF."""

    model_id: str = "prebuilt-layout"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            endpoint = os.environ.get("AZURE_DI_ENDPOINT")
            key = os.environ.get("AZURE_DI_KEY")
            if not endpoint or not key:
                raise RuntimeError("AZURE_DI_ENDPOINT / AZURE_DI_KEY not set")
            self._client = DocumentIntelligenceClient(
                endpoint=endpoint, credential=AzureKeyCredential(key)
            )
        return self._client

    def config_hash(self) -> str:
        return sha256_text(f"{self.name}|{self.version}|model={self.model_id}")[:7]

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        client = self._ensure_client()
        t0 = time.perf_counter()
        with pdf_path.open("rb") as f:
            poller = client.begin_analyze_document(self.model_id, body=f)
        result = poller.result()
        latency = time.perf_counter() - t0
        markdown, pages = _serialize(result)
        return ParseResult(
            markdown=markdown,
            page_count=pages,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=pages) if pages else None,
            pages_processed=pages,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
