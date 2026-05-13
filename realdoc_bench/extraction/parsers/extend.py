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

from realdoc_bench.extraction.parsers.base import (
    ParseProvider,
    ParseResult,
    register_parser,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost


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

            self._client = Extend(api_key=os.environ.get("EXTEND_API_KEY"))
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
        run = client.parse.run(file_path=str(pdf_path), config=config, wait=True)
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
    chunks = getattr(run, "chunks", None) or []
    pieces: list[str] = []
    for c in chunks:
        content = getattr(c, "content", "") or ""
        if content:
            pieces.append(content)
    if not pieces:
        output = getattr(run, "output", None) or {}
        if isinstance(output, dict) and isinstance(output.get("markdown"), str):
            return output["markdown"]
    return "\n\n".join(pieces)


def _page_count(run: Any) -> int | None:
    output = getattr(run, "output", None) or {}
    if isinstance(output, dict):
        pages = output.get("pageCount") or output.get("page_count")
        if isinstance(pages, int):
            return pages
    metrics = getattr(run, "metrics", None) or {}
    pages = metrics.get("page_count") if isinstance(metrics, dict) else None
    return pages if isinstance(pages, int) else None


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


@register_parser("extend_performance_v2_0_0_beta", version="2.0.0-beta")
class ExtendPerformanceV2(_ExtendParserBase):
    engine = "parse_performance"
    engine_version = "2.0.0-beta"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {"enabled": True, "figureImageClippingEnabled": True},
            "tables": {"enabled": True, "targetFormat": "markdown"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
            "formulas": {"enabled": True},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }


@register_parser("extend_light_v1_0_0", version="1.0.0")
class ExtendLightV1(_ExtendParserBase):
    engine = "parse_light"
    engine_version = "1.0.0"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {"enabled": True, "figureImageClippingEnabled": True},
            "tables": {"enabled": True, "targetFormat": "html"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
            "formulas": {"enabled": True},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }


@register_parser("extend_light_v0_0_2_beta", version="0.0.2-beta")
class ExtendLightV002(_ExtendParserBase):
    engine = "parse_light"
    engine_version = "0.0.2-beta"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {"enabled": True, "figureImageClippingEnabled": True},
            "tables": {"enabled": True, "targetFormat": "html"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }
