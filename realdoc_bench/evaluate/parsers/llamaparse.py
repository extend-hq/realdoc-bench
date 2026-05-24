"""LlamaParse (LlamaCloud) parse adapter — uploads a PDF, gets markdown back.

LlamaCloud's parse API returns a single ``markdown_full`` string for the
whole document when ``expand=["markdown_full"]`` is passed. We use that
directly. Tier defaults to ``agentic`` (their best); version defaults to
``latest``. Page count is taken from the response's job metadata.

Auth: ``LLAMA_CLOUD_API_KEY`` env var. Pricing: ``llamaparse`` in
catalog.yaml (per-page rate).
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


def _page_count(parse_result: Any) -> int | None:
    # LlamaCloud surfaces page_count in several shapes across response
    # versions. Walk the obvious candidates.
    for attr in ("page_count", "pages", "total_pages", "num_pages"):
        v = getattr(parse_result, attr, None)
        if isinstance(v, int):
            return v
    job = getattr(parse_result, "job", None)
    if job is not None:
        for attr in ("page_count", "pages", "total_pages", "num_pages"):
            v = getattr(job, attr, None)
            if isinstance(v, int):
                return v
    return None


@register_parser("llamaparse", version="1.0.0")
class LlamaParseParser(ParseProvider):
    """LlamaParse parse API — PDF in, ``markdown_full`` out."""

    tier: str = "agentic"
    version_str: str = "latest"
    # Match Extend v2 advanced + Reducto signature flag — turn on the
    # provider's strongest chart-parsing pipeline so each parser is configured
    # for its best-effort chart extraction. ``agentic`` here pairs with the
    # agentic tier; ``agentic_plus`` is heavier and more expensive.
    # specialized_chart_parsing only applies to the agentic tiers — the
    # cost_effective / fast tiers leave it None (see LlamaParseCostEffective).
    chart_parsing: str | None = "agentic"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from llama_cloud import LlamaCloud

            api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
            if not api_key:
                raise RuntimeError("LLAMA_CLOUD_API_KEY not set")
            self._client = LlamaCloud(api_key=api_key)
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            f"{self.name}|{self.version}|tier={self.tier}"
            f"|version_str={self.version_str}|chart={self.chart_parsing}"
        )[:7]

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        client = self._ensure_client()
        t0 = time.perf_counter()

        file_obj = client.files.create(file=pdf_path, purpose="parse")
        parse_kwargs: dict[str, Any] = dict(
            file_id=file_obj.id,
            tier=self.tier,
            version=self.version_str,
            output_options={
                "markdown": {"tables": {"output_tables_as_markdown": True}},
            },
            expand=["markdown_full"],
        )
        # specialized_chart_parsing only applies to the agentic tiers; the
        # cost_effective / fast tiers leave it unset.
        if self.chart_parsing:
            parse_kwargs["processing_options"] = {
                "specialized_chart_parsing": self.chart_parsing,
            }
        result = client.parsing.parse(**parse_kwargs)
        latency = time.perf_counter() - t0

        markdown = getattr(result, "markdown_full", "") or ""
        pages = _page_count(result)

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


@register_parser("llamaparse_cost_effective", version="1.0.0")
class LlamaParseCostEffective(LlamaParseParser):
    """LlamaParse ``cost_effective`` tier — the balanced low-cost parsing
    option, vs the default ``llamaparse`` which runs the ``agentic`` tier.

    Recorded as a separate performance run. No specialized chart parsing —
    that pipeline only runs on the agentic tiers.
    """

    tier = "cost_effective"
    chart_parsing = None
