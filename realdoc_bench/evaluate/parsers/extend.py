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


def _block_type(block: Any) -> str:
    """Normalize ``block.type`` (enum) → lowercase string."""
    t = getattr(block, "type", None)
    return (t.value if hasattr(t, "value") else str(t or "")).lower()


# Block-aware renderer: walks chunk.blocks[] and emits markdown that surfaces
# Extend's typed-block taxonomy more aggressively than the server-side
# `chunk.content` renderer does. Differences:
#   - section_heading → ``## ``  (server uses ``### ``)
#   - key_value → ``**key:** value``  (server emits plain text)
#   - page_number / footer / header → SKIPPED (server emits as plain text → noise)
#   - everything else: use block.content verbatim (already markdown / HTML / table)
#
# Goal: parity with AWS Textract / Azure DI adapters, which both do explicit
# block-type → markdown mapping. Empirical question: does this make Extend's
# downstream extraction score noticeably better?
_SKIP_BLOCK_TYPES = {"page_number", "footer", "header"}


def _kv_block_to_md(block: Any) -> str:
    """Render a ``key_value`` block as ``**key:** value``.

    Extend's key_value blocks carry metadata like ``{key: ..., value: ...}``;
    fall back to ``block.content`` if metadata isn't structured.
    """
    md = getattr(block, "metadata", None)
    key = value = None
    if md is not None:
        key = getattr(md, "key", None) or getattr(md, "field_name", None)
        value = getattr(md, "value", None) or getattr(md, "field_value", None)
        if isinstance(md, dict):
            key = key or md.get("key") or md.get("fieldName")
            value = value or md.get("value") or md.get("fieldValue")
    if isinstance(key, str) and key.strip():
        v = (value if value is not None else getattr(block, "content", "")) or ""
        return f"**{key.strip()}:** {str(v).strip()}"
    return (getattr(block, "content", "") or "").strip()


def _markdown_from_blocks(run: Any) -> str:
    """Walk ``run.output.chunks[].blocks[]`` and emit our own block-aware md."""
    output = getattr(run, "output", None)
    chunks = getattr(output, "chunks", None) if output is not None else None
    if chunks is None:
        chunks = getattr(run, "chunks", None) or []
    parts: list[str] = []
    for chunk in chunks:
        blocks = getattr(chunk, "blocks", None) or []
        for b in blocks:
            t = _block_type(b)
            if t in _SKIP_BLOCK_TYPES:
                continue
            content = (getattr(b, "content", "") or "").strip()
            if not content:
                continue
            if t == "heading":
                # Keep '# ' as the server emits it; some headings already have it
                parts.append(content if content.startswith("#") else f"# {content}")
            elif t == "section_heading":
                # Upgrade from server's '### ' to '## '
                stripped = content.lstrip("# ").strip() if content.startswith("#") else content
                parts.append(f"## {stripped}")
            elif t == "key_value":
                parts.append(_kv_block_to_md(b))
            else:
                # text / table / figure / formula / barcode / table_head / table_cell
                parts.append(content)
    if not parts:
        # Fallback if blocks aren't populated for some reason
        return _markdown_from_run(run)
    return "\n\n".join(parts)


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


_PERFORMANCE_V2_CONFIG_EXTRA = {
    "chunkingStrategy": {"type": "page"},
    "blockOptions": {
        "figures": {"enabled": True, "figureImageClippingEnabled": True},
        "tables": {"enabled": True, "targetFormat": "markdown"},
        "text": {"enabled": True, "styleFormattingEnabled": False},
        "formulas": {"enabled": True},
    },
    "advancedOptions": {"pageRotationEnabled": True},
}


@register_parser("extend_performance_v2_0_0", version="2.0.0")
class ExtendPerformanceV2(_ExtendParserBase):
    """Performance v2.0.0 — release of the new pipeline.
    """

    engine = "parse_performance"
    engine_version = "2.0.0"
    config_extra = _PERFORMANCE_V2_CONFIG_EXTRA




# Full advanced config: every v2.0.0+ block option ON except agentic (deferred).
# See: https://docs.extend.ai/product/parsing/configuration-options
_PERFORMANCE_V2_ADVANCED_CONFIG_EXTRA = {
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


@register_parser("extend_performance_v2_0_0_advanced", version="2.0.0+advanced")
class ExtendPerformanceV2Advanced(_ExtendParserBase):
    """Performance v2.0.0 GA with EVERY v2.0.0+ block option ON (no agentic).

    Used by the realdoc-bench v1.2_final pilot to showcase the within-vendor
    delta from v1.0.0 → v2.0.0+advanced. Six new capabilities relative to v1:
      • advancedChartExtractionEnabled — bar/line/pie → structured table
      • cellBlocksEnabled — per-cell row/col indices for table extraction
      • tableHeaderContinuationEnabled — multi-page table headers
      • signatureDetectionEnabled — handwritten signatures
      • blankFieldFormattingEnabled — explicit <blank /> on empty form fields
      • formulas — LaTeX representation of math
    plus barcode reading and (existing) figure image clipping.
    """

    engine = "parse_performance"
    engine_version = "2.0.0"
    config_extra = _PERFORMANCE_V2_ADVANCED_CONFIG_EXTRA


# Agentic config: the full +advanced config PLUS the three agentic / VLM-review
# passes the +advanced run deliberately left off.
#   • text.agentic        — VLM review/correction pass over OCR text
#   • tables.agentic      — VLM review/correction pass over extracted tables
#   • formattingDetection — change_tracking: detect tracked changes (insertions,
#     deletions, substitutions) shown via strikethrough / colored text / underline
# Heavier + slower + pricier than +advanced; recorded as its own perf run.
_PERFORMANCE_V2_AGENTIC_CONFIG_EXTRA = {
    "chunkingStrategy": {"type": "page"},
    "blockOptions": {
        "figures": {
            "enabled": True,
            "figureImageClippingEnabled": True,
            "advancedChartExtractionEnabled": True,
        },
        "tables": {
            "enabled": True,
            "targetFormat": "markdown",
            "tableHeaderContinuationEnabled": True,
            "agentic": {"enabled": True},          # agentic tables — VLM review/correction
        },
        "text": {
            "enabled": True,
            "styleFormattingEnabled": False,
            "signatureDetectionEnabled": True,
            "agentic": {"enabled": True},          # agentic OCR — VLM review/correction
        },
        "formulas": {"enabled": True},
        "keyValue": {"blankFieldFormattingEnabled": True},
        "barcodes": {"readingEnabled": True, "imageClippingEnabled": False},
    },
    "advancedOptions": {
        "pageRotationEnabled": True,
        "formattingDetection": [{"type": "change_tracking"}],   # change detection
    },
}


@register_parser("extend_performance_v2_0_0_agentic", version="2.0.0+agentic")
class ExtendPerformanceV2Agentic(_ExtendParserBase):
    """Performance v2.0.0 + advanced + the three agentic passes.

    Everything ``extend_performance_v2_0_0_advanced`` enables, plus:
      • agentic OCR      — VLM review/correction pass over text (text.agentic)
      • agentic tables   — VLM review/correction pass over tables (tables.agentic)
      • change detection — formattingDetection ``change_tracking`` (strikethrough /
        colored / underlined insertions, deletions, substitutions)
    Separate performance run; heavier + slower than +advanced.
    """

    engine = "parse_performance"
    engine_version = "2.0.0"
    config_extra = _PERFORMANCE_V2_AGENTIC_CONFIG_EXTRA


@register_parser("extend_performance_v2_0_0_blocks", version="2.0.0+blocks")
class ExtendPerformanceV2Blocks(_ExtendParserBase):
    """Performance v2.0.0 + **block-aware** markdown renderer.

    Same API call as ``extend_performance_v2_0_0`` (engine_version="2.0.0"),
    but we walk ``chunk.blocks[]`` ourselves to emit markdown that:
      - upgrades ``section_heading`` to ``## ``
      - skips ``page_number`` / ``footer`` / ``header`` blocks (server emits as plain text → noise)
      - renders ``key_value`` blocks as ``**key:** value``
    """

    engine = "parse_performance"
    engine_version = "2.0.0"
    config_extra = _PERFORMANCE_V2_CONFIG_EXTRA

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
        markdown = _markdown_from_blocks(run)
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

