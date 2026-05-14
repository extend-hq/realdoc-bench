"""Shared base for vision-model parse providers (dots.mocr, GLM-OCR, …).

These models are per-image, OpenAI-compatible vLLM endpoints. The base:

1. Renders each PDF page to a PNG via PyMuPDF (configurable DPI).
2. Sends each page image + a markdown-extraction prompt to the endpoint.
3. Concatenates per-page markdown into one document.

Subclasses provide the prompt and a ``_page_to_markdown`` that turns one raw
model response into markdown for that page. Connection errors are retried
with backoff (the endpoints are typically reached over a flappy SSH tunnel).
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.extraction.parsers.base import ParseProvider, ParseResult
from realdoc_bench.shared.pricing.meter import parse_cost

DEFAULT_DPI = 150
DEFAULT_TIMEOUT_SEC = 300.0
MAX_RETRIES = 6
RETRY_BACKOFF_SEC = 5.0


def _render_pdf_pages(pdf_path: Path, dpi: int) -> list[bytes]:
    """Render each PDF page to PNG bytes."""
    import pymupdf

    out: list[bytes] = []
    doc = pymupdf.open(str(pdf_path))
    try:
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            out.append(pix.tobytes("png"))
    finally:
        doc.close()
    return out


class VisionParserBase(ParseProvider):
    """Base class for OpenAI-compatible vision parse providers.

    Subclass contract:
    - ``base_url_env`` / ``model_env`` — env vars for endpoint + model id
    - ``default_base_url`` / ``default_model``
    - ``prompt`` — the per-page extraction prompt
    - ``_page_to_markdown(raw_text, page_no)`` — raw response → markdown
    """

    base_url_env: str = ""
    model_env: str = ""
    default_base_url: str = "http://localhost:8000/v1"
    default_model: str = ""
    prompt: str = "Convert this document page to clean Markdown."
    max_tokens: int = 12000
    dpi: int = DEFAULT_DPI

    def __init__(self) -> None:
        self._client: Any | None = None
        self._model_id: str = ""

    def _new_client(self) -> Any:
        from openai import OpenAI

        base_url = os.environ.get(self.base_url_env, self.default_base_url)
        api_key = os.environ.get(f"{self.base_url_env}_KEY") or "EMPTY"
        self._model_id = os.environ.get(self.model_env, "") or self.default_model
        return OpenAI(api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT_SEC, max_retries=0)

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._new_client()
        return self._client

    def _page_to_markdown(self, raw_text: str, page_no: int) -> str:
        """Subclasses override to post-process the raw model response."""
        return raw_text.strip()

    def _call_page(self, png_bytes: bytes) -> str:
        encoded = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{encoded}"
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                client = self._ensure_client()
                resp = client.chat.completions.create(
                    model=self._model_id,
                    max_completion_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                )
                return resp.choices[0].message.content or "" if resp.choices else ""
            except Exception as e:  # noqa: BLE001 — connection/timeout/5xx all retryable
                last_err = e
                self._client = None
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
        raise RuntimeError(f"{self.name} page request failed after {MAX_RETRIES} attempts: {last_err}")

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        t0 = time.perf_counter()
        pages = _render_pdf_pages(pdf_path, self.dpi)
        page_md: list[str] = []
        for i, png in enumerate(pages, start=1):
            raw = self._call_page(png)
            md = self._page_to_markdown(raw, i)
            page_md.append(f"## Page {i}\n\n{md}".rstrip())
        latency = time.perf_counter() - t0
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
