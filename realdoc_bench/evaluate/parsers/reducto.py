"""Reducto parse adapter — uploads a PDF, gets markdown back.

Reducto's parse API returns either:

  - **inline result** for small docs: ``result.chunks[].blocks[]``, where
    each block has a ``content`` field that's already markdown-ish, OR
  - **URL result** for large docs (typically > a few pages): the
    ``ParseResponse`` carries ``result={type: "url", url: "..."}`` and
    the actual ``{chunks: [...]}`` JSON is stored at the (pre-signed S3)
    URL. The adapter fetches the URL and parses the same shape.

Without the URL branch, large-doc parses silently return 0 chunks. We
walk both shapes uniformly via attribute/dict accessors.

Auth: ``REDUCTO_API_KEY`` env var. Pricing: ``reducto`` in catalog.yaml
(per-page rate). Page count comes from the response's chunk page-ranges.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from realdoc_bench.evaluate.parsers.base import (
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


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Accessor that works on both Pydantic models and plain dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _block_to_md(b: Any) -> str:
    btype = _get(b, "type")
    btype_str = btype.value if hasattr(btype, "value") else str(btype or "")
    content = (_get(b, "content", "") or "").rstrip()
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
        pr = _get(chunk, "page_range") or _get(chunk, "pageRange")
        if pr is not None:
            end = _get(pr, "end") or _get(pr, "stop")
            if isinstance(end, int):
                max_page = max(max_page, end)
        for b in _get(chunk, "blocks", None) or []:
            md = _block_to_md(b)
            if md:
                parts.append(md)
            bbox = _get(b, "bbox")
            if bbox is not None:
                pn = _get(bbox, "page")
                if isinstance(pn, int):
                    max_page = max(max_page, pn)
    return "\n\n".join(parts), max_page


def _fetch_url_result(url: str, timeout: float = 60.0) -> dict[str, Any]:
    """Fetch the Reducto URL-result and parse it as JSON.

    Reducto stores the full result on S3 for large docs (> ~tens of pages)
    and returns a pre-signed URL; the body is the same ``{chunks: [...]}``
    JSON we'd otherwise get inline.
    """
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read()
    return json.loads(body)


@register_parser("reducto", version="1.0.0")
class ReductoParser(ParseProvider):
    """Reducto parse API — PDF in, markdown out via the chunk/block graph."""

    chunk_mode: str = "page"
    extraction_mode: str = "ocr"          # SDK allows only "ocr" | "hybrid"
    # enhance.agentic — VLM-assisted extraction, one entry per scope. The base
    # parser runs the figure agent only; the agentic variant adds table + text.
    agentic: list = [{"scope": "figure", "advanced_chart_agent": True}]
    intelligent_ordering: bool = False

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
        scopes = ",".join(sorted(a["scope"] for a in self.agentic))
        return sha256_text(
            f"{self.name}|{self.version}|chunk_mode={self.chunk_mode}"
            f"|mode={self.extraction_mode}|agentic={scopes}"
            f"|order={self.intelligent_ordering}|sig=1"
        )[:7]

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        client = self._ensure_client()
        t0 = time.perf_counter()

        upload = client.upload(file=pdf_path)
        resp = client.parse.run(
            input=upload.file_id,
            retrieval={"chunking": {"chunk_mode": self.chunk_mode}},
            settings={"extraction_mode": self.extraction_mode},
            formatting={"include": ["signatures"]},
            enhance={"agentic": self.agentic,
                     "intelligent_ordering": self.intelligent_ordering},
        )
        latency = time.perf_counter() - t0

        # Two response shapes:
        #   small docs → resp.result.chunks inline
        #   large docs → resp.result = {type: "url", url: "..."} pointing at
        #                an S3 JSON blob with the actual chunks. Fetch it.
        result = _get(resp, "result")
        chunks: list[Any] = []
        if result is not None:
            rtype = _get(result, "type")
            if rtype == "url":
                url = _get(result, "url")
                if url:
                    fetched = _fetch_url_result(str(url))
                    chunks = fetched.get("chunks") or []
                    latency = time.perf_counter() - t0  # include URL fetch
            else:
                chunks = _get(result, "chunks") or []

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


@register_parser("reducto_agentic", version="1.0.0")
class ReductoAgenticParser(ReductoParser):
    """Reducto with agentic VLM enhancement on every scope — text (form
    key-value regions), table and figure. Config matches the reference clip
    exactly; the base parser runs the figure agent only."""

    agentic = [
        {"scope": "text"},
        {"scope": "table"},
        {"scope": "figure", "advanced_chart_agent": True},
    ]
