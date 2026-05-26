"""Shared base for vision-model parse providers.

These models are per-image: a PDF is rendered page-by-page, each page image is
sent to the model with a markdown-extraction prompt, and the per-page markdown
is concatenated.

- :class:`GeminiVisionParserBase` — the Google Gemini cloud API.

Cost is page-based when the catalog rate is a ``PageRate`` and token-based when
it's a ``TokenRate`` (cloud VLMs). Connection errors are retried with backoff.
"""

from __future__ import annotations

import os
import re
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from realdoc_bench.evaluate.parsers.base import ParseProvider, ParseResult
from realdoc_bench.shared.pricing.meter import agent_cost, parse_cost

DEFAULT_DPI = 150
MAX_RETRIES = 6
RETRY_BACKOFF_SEC = 5.0

# Gemini sometimes wraps single-page markdown output in
# ```markdown ... ``` code fences despite the prompt saying "Output only the
# Markdown for this page". The fences themselves leak into the agent's input,
# confusing table/structure parsing. Strip them defensively.
_FENCE_OPEN_RE = re.compile(r"^```(?:markdown|md)?\s*\n", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\n```\s*$")


def _strip_outer_code_fence(text: str) -> str:
    """Strip a single outer ``` fence pair if the entire blob is wrapped in one.

    Idempotent: text without an outer fence is returned unchanged. Inner fenced
    blocks (e.g. real code samples inside the markdown) are preserved.
    """
    if not text:
        return text
    m_open = _FENCE_OPEN_RE.match(text)
    if not m_open:
        return text
    m_close = _FENCE_CLOSE_RE.search(text)
    if not m_close:
        return text
    inner = text[m_open.end() : m_close.start()].strip()
    return inner if inner else text


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


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _load_pages(input_path: Path, dpi: int) -> list[bytes]:
    """Load an input file as a list of per-page PNG-encoded bytes.

    PDFs are rasterized at ``dpi``. Single-image inputs (PNG/JPEG/WEBP) are
    read as one "page" — JPEG/WEBP are re-encoded to PNG so downstream calls
    that hardcode ``image/png`` mime type stay correct.
    """
    suffix = input_path.suffix.lower()
    if suffix not in _IMAGE_EXTS:
        return _render_pdf_pages(input_path, dpi)
    data = input_path.read_bytes()
    if suffix == ".png":
        return [data]
    # Re-encode JPEG/WEBP to PNG so the existing image/png mime contract holds.
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return [buf.getvalue()]


class VisionParserBase(ParseProvider):
    """Renders a PDF to page images and concatenates per-page markdown.

    Subclass contract:
    - ``prompt`` — the per-page extraction prompt.
    - ``_call_page(png_bytes)`` → ``(raw_text, input_tokens, output_tokens)``.
    - ``_page_to_markdown(raw_text, page_no)`` — raw response → markdown.
    - ``pricing_key`` — catalog key for cost (page- or token-rate).
    """

    prompt: str = "Convert this document page to clean, faithful Markdown."
    max_tokens: int = 12000
    dpi: int = DEFAULT_DPI
    pricing_key: str = ""
    # Pages are independent — fan them out concurrently. The Gemini API handles
    # this fine; it turns an N-page sequential parse into ~N/page_concurrency
    # wall-clock. Override per subclass or via the ``RDB_PAGE_CONCURRENCY`` env var.
    page_concurrency: int = 8

    @abstractmethod
    def _call_page(self, png_bytes: bytes) -> tuple[str, int, int]: ...

    def _page_to_markdown(self, raw_text: str, page_no: int) -> str:
        return _strip_outer_code_fence(raw_text.strip())

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        import concurrent.futures
        import os as _os

        t0 = time.perf_counter()
        pages = _load_pages(pdf_path, self.dpi)
        concurrency = max(1, int(_os.environ.get("RDB_PAGE_CONCURRENCY", self.page_concurrency)))

        results: list[tuple[str, int, int]] = [("", 0, 0)] * len(pages)
        if concurrency == 1 or len(pages) <= 1:
            for i, png in enumerate(pages):
                results[i] = self._call_page(png)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = {pool.submit(self._call_page, png): i for i, png in enumerate(pages)}
                for fut in concurrent.futures.as_completed(futs):
                    results[futs[fut]] = fut.result()

        page_md: list[str] = []
        in_tok = out_tok = 0
        for i, (raw, it, ot) in enumerate(results, start=1):
            in_tok += it
            out_tok += ot
            page_md.append(f"## Page {i}\n\n{self._page_to_markdown(raw, i)}".rstrip())
        latency = time.perf_counter() - t0
        markdown = "\n\n".join(page_md)

        key = self.pricing_key or self.name
        cost = parse_cost(key, pages=len(pages))
        if cost is None:  # token-rated provider
            cost = agent_cost(key, input_tokens=in_tok, output_tokens=out_tok)

        return ParseResult(
            markdown=markdown,
            page_count=len(pages),
            latency_sec=latency,
            cost_estimate_usd=cost,
            pages_processed=len(pages),
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
            raw={"input_tokens": in_tok, "output_tokens": out_tok} if (in_tok or out_tok) else None,
        )


# ── Google Gemini ───────────────────────────────────────────────────────────


class GeminiVisionParserBase(VisionParserBase):
    """For the Google Gemini API (Gemini 3 as a parse pipeline)."""

    model: str = ""

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _call_page(self, png_bytes: bytes) -> tuple[str, int, int]:
        from google.genai import types

        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                client = self._ensure_client()
                resp = client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                        self.prompt,
                    ],
                )
                text = resp.text or ""
                usage = getattr(resp, "usage_metadata", None)
                it = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
                ot = getattr(usage, "candidates_token_count", 0) or 0 if usage else 0
                return text, int(it), int(ot)
            except Exception as e:  # noqa: BLE001
                last_err = e
                self._client = None
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
        raise RuntimeError(f"{self.name} page request failed after {MAX_RETRIES} attempts: {last_err}")
