"""Mistral OCR 4 parse adapter -- PDF in, markdown out.

Uses Mistral's OCR endpoint directly instead of the Python SDK so the adapter
stays consistent with the repo's existing lightweight HTTP clients. Local PDFs
are sent as base64 data URLs, which the API documents as the supported path for
files that are not publicly hosted.

Auth: ``MISTRAL_API_KEY`` env var. Pricing: ``mistral_ocr_4`` in catalog.yaml
(per-page rate).
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import httpx

from realdoc_bench.evaluate.parsers.base import (
    ParseProvider,
    ParseResult,
    register_parser,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost

DEFAULT_BASE_URL = "https://api.mistral.ai"


def _pdf_data_url(pdf_path: Path) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pages_to_markdown(response: dict[str, Any]) -> tuple[str, int]:
    pages = response.get("pages") or []
    chunks: list[str] = []
    for page in pages:
        markdown = (_get(page, "markdown", "") or "").strip()
        if markdown:
            chunks.append(markdown)
    return "\n\n---\n\n".join(chunks), len(pages)


@register_parser("mistral_ocr_4", version="mistral-ocr-latest")
class MistralOCR4Parser(ParseProvider):
    """Mistral OCR 4 via ``POST /v1/ocr``."""

    model: str = "mistral-ocr-latest"
    table_format: str | None = None
    confidence_scores_granularity: str | None = None

    def config_hash(self) -> str:
        return sha256_text(
            f"{self.name}|{self.version}|model={self.model}"
            f"|table={self.table_format}|conf={self.confidence_scores_granularity}"
        )[:7]

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")

        base_url = os.environ.get("MISTRAL_OCR_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        payload: dict[str, Any] = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": _pdf_data_url(pdf_path),
            },
        }
        if self.table_format is not None:
            payload["table_format"] = self.table_format
        if self.confidence_scores_granularity is not None:
            payload["confidence_scores_granularity"] = self.confidence_scores_granularity

        t0 = time.perf_counter()
        with httpx.Client(timeout=None) as client:
            resp = client.post(
                f"{base_url}/v1/ocr",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()
        latency = time.perf_counter() - t0

        markdown, pages = _pages_to_markdown(raw)
        return ParseResult(
            markdown=markdown,
            page_count=pages,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=pages) if pages else None,
            pages_processed=pages,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
            raw={"model": raw.get("model"), "usage_info": raw.get("usage_info")},
        )
