"""Extend parse adapters for the extraction-quality benchmark.

Each adapter calls the Extend parse API with a specific engine + version and
returns the resulting markdown. The Extend SDK is imported lazily so the
package stays usable without the dependency installed.
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

_PROD_URL = "https://api.extend.ai"


def _base_url() -> str:
    """Always hit prod unless EXTEND_BASE_URL is set explicitly."""
    return os.environ.get("EXTEND_BASE_URL") or _PROD_URL


class _ExtendParserBase(ParseProvider):
    engine: str = "parse_performance"
    engine_version: str = ""
    target: str = "markdown"
    config_extra: dict[str, Any] = {}

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from extend_ai import Extend

            token = os.environ.get("EXTEND_API_KEY")
            if not token:
                raise RuntimeError("EXTEND_API_KEY not set")
            self._client = Extend(token=token, base_url=_base_url())
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            "|".join(
                [
                    self.name,
                    self.version,
                    self.engine,
                    self.engine_version,
                    self.target,
                    repr(sorted(self.config_extra.items())),
                ]
            )
        )[:7]

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        client = self._ensure_client()
        t0 = time.perf_counter()
        config = {
            "engine": self.engine,
            "engineVersion": self.engine_version,
            "target": self.target,
            **self.config_extra,
        }
        with pdf_path.open("rb") as f:
            uploaded = client.files.upload(file=f)
        run = client.parse_runs.create_and_poll(
            file={"id": uploaded.id},
            config=config,
        )
        latency = time.perf_counter() - t0
        markdown = _markdown_from_run(run)
        pages = _page_count(run)
        return ParseResult(
            markdown=markdown,
            page_count=pages,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=pages),
            pages_processed=pages,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )


def _markdown_from_run(run: Any) -> str:
    output = getattr(run, "output", None)
    chunks = getattr(output, "chunks", None) if output is not None else None
    if chunks is None:
        chunks = getattr(run, "chunks", None) or []
    pieces: list[str] = []
    for c in chunks:
        content = getattr(c, "content", "") or ""
        if content:
            pieces.append(content)
    if pieces:
        return "\n\n".join(pieces)
    if output is not None:
        md = getattr(output, "markdown", None)
        if isinstance(md, str):
            return md
        if isinstance(output, dict) and isinstance(output.get("markdown"), str):
            return output["markdown"]
    return ""


def _page_count(run: Any) -> int | None:
    metrics = getattr(run, "metrics", None)
    if metrics is not None:
        pages = getattr(metrics, "page_count", None)
        if isinstance(pages, (int, float)):
            return int(pages)
        if isinstance(metrics, dict):
            pages = metrics.get("page_count") or metrics.get("pageCount")
            if isinstance(pages, (int, float)):
                return int(pages)
    output = getattr(run, "output", None)
    if isinstance(output, dict):
        pages = output.get("pageCount") or output.get("page_count")
        if isinstance(pages, (int, float)):
            return int(pages)
    return None


@register_parser("extend_performance_v1_0_0", version="1.0.0")
class ExtendPerformanceV1(_ExtendParserBase):
    engine = "parse_performance"
    engine_version = "1.0.0"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {"enabled": True, "figureImageClippingEnabled": True},
            "tables": {"enabled": True, "targetFormat": "html"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }


# Full v2.0.0 config: every v2.0.0+ block option ON except agentic (deferred).
# See: https://docs.extend.ai/product/parsing/configuration-options
_PERFORMANCE_V2_CONFIG_EXTRA = {
    "chunkingStrategy": {"type": "page"},
    "blockOptions": {
        "figures": {
            "enabled": True,
            "figureImageClippingEnabled": True,
            "advancedChartExtractionEnabled": True,   # charts → structured tables (v2.0.0+)
        },
        "tables": {
            "enabled": True,
            "targetFormat": "markdown",
            "tableHeaderContinuationEnabled": True,   # propagate headers across page breaks
        },
        "text": {
            "enabled": True,
            "styleFormattingEnabled": False,
            "signatureDetectionEnabled": True,        # handwritten signature detection
            # "agentic": {"enabled": True},           # DEFERRED — adds VLM correction passes
        },
        "formulas": {"enabled": True},
        "keyValue": {"blankFieldFormattingEnabled": True},  # emits <blank /> for empty (v2.0.0+)
        "barcodes": {"readingEnabled": True, "imageClippingEnabled": False},
    },
    "advancedOptions": {"pageRotationEnabled": True},
}


@register_parser("extend_light_v1_0_0", version="1.0.0")
class ExtendLightV1(_ExtendParserBase):
    """Parse Light v1.0.0 with advanced chart extraction enabled."""

    engine = "parse_light"
    engine_version = "1.0.0"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {
                "enabled": True,
                "figureImageClippingEnabled": True,
                "advancedChartExtractionEnabled": True,
            },
            "tables": {"enabled": True, "targetFormat": "html"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
            "formulas": {"enabled": True},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }


@register_parser("extend_performance_v2_0_0", version="2.0.0")
class ExtendPerformanceV2(_ExtendParserBase):
    """Performance v2.0.0 GA with EVERY v2.0.0+ block option ON (no agentic).

    Capabilities (vs v1):
      • advancedChartExtractionEnabled — bar/line/pie → structured table
      • tableHeaderContinuationEnabled — multi-page table headers
      • signatureDetectionEnabled — handwritten signatures
      • blankFieldFormattingEnabled — explicit <blank /> on empty form fields
      • formulas — LaTeX representation of math
    plus barcode reading and (existing) figure image clipping.
    """

    engine = "parse_performance"
    engine_version = "2.0.0"
    config_extra = _PERFORMANCE_V2_CONFIG_EXTRA
