"""Reducto parse adapter — uploads a PDF, gets markdown back.

Reducto's parse API returns a ``result.chunks[].blocks[]`` graph (each block
has a ``content`` field that's already markdown-ish). We concatenate block
contents in reading order. Page boundaries are inferred from chunk-level
page-range metadata when available.

Auth: ``REDUCTO_API_KEY`` env var. Pricing: ``reducto`` in catalog.yaml
(per-page rate). Page count comes from the response's chunk page-ranges.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.extraction.parsers.base import (
    ParseProvider,
    ParseResult,
    register_parser,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost


# Block type → markdown prefix. The block's ``content`` field already carries
# the text; we just decorate headings so the agent sees structure. (Reducto
# tables come back as HTML or markdown depending on Reducto config; we leave
# them as-is — the agent can read either.)
_HEADING_TYPES = {"Title", "Header", "Section Header"}


def _block_to_md(b: Any) -> str:
    btype = getattr(b, "type", None)
    btype_str = btype.value if hasattr(btype, "value") else str(btype or "")
    content = (getattr(b, "content", "") or "").rstrip()
    if not content:
        return ""
    if btype_str == "Title":
        return f"# {content}"
    if btype_str == "Section Header" or btype_str == "Header":
        return f"## {content}"
    return content


def _chunks_to_markdown(chunks: list[Any]) -> tuple[str, int]:
    """Concatenate block contents in order; track the max page number seen."""
    parts: list[str] = []
    max_page = 0
    for chunk in chunks or []:
        # Page range may live on the chunk or on individual block bboxes.
        pr = getattr(chunk, "page_range", None) or getattr(chunk, "pageRange", None)
        if pr is not None:
            end = getattr(pr, "end", None) or getattr(pr, "stop", None)
            if isinstance(end, int):
                max_page = max(max_page, end)
        for b in getattr(chunk, "blocks", None) or []:
            md = _block_to_md(b)
            if md:
                parts.append(md)
            bbox = getattr(b, "bbox", None)
            if bbox is not None and isinstance(getattr(bbox, "page", None), int):
                max_page = max(max_page, bbox.page)
    return "\n\n".join(parts), max_page


@register_parser("reducto", version="1.0.0")
class ReductoParser(ParseProvider):
    """Reducto parse API — PDF in, markdown out via the chunk/block graph."""

    chunk_mode: str = "page"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from reducto import Reducto

            api_key = os.environ.get("REDUCTO_API_KEY")
            if not api_key:
                raise RuntimeError("REDUCTO_API_KEY not set")
            self._client = Reducto(api_key=api_key)
        return self._client

    def config_hash(self) -> str:
        return sha256_text(f"{self.name}|{self.version}|chunk_mode={self.chunk_mode}")[:7]

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        client = self._ensure_client()
        t0 = time.perf_counter()

        upload = client.upload(file=pdf_path)
        resp = client.parse.run(
            input=upload.file_id,
            retrieval={"chunking": {"chunk_mode": self.chunk_mode}},
            settings={"extraction_mode": "ocr"},
        )
        latency = time.perf_counter() - t0

        chunks = []
        result = getattr(resp, "result", None)
        if result is not None:
            chunks = getattr(result, "chunks", None) or []

        markdown, pages = _chunks_to_markdown(chunks)
        pages = pages or None

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
