"""Local fallback parser: extract plain text from each PDF page via PyMuPDF.

Intended as a no-API baseline so the harness end-to-end works offline. Output
is "page-flat markdown": each page becomes a `## Page N` heading followed by
its text. Not competitive, just useful for smoke tests and CI.
"""

from __future__ import annotations

import time
from pathlib import Path

from realdoc_bench.evaluate.parsers.base import (
    ParseProvider,
    ParseResult,
    register_parser,
)
from realdoc_bench.shared.pricing.meter import parse_cost


@register_parser("pymupdf", version="1.24")
class PymupdfParser(ParseProvider):
    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        import pymupdf

        t0 = time.perf_counter()
        doc = pymupdf.open(str(pdf_path))
        try:
            chunks: list[str] = []
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                chunks.append(f"## Page {i}\n\n{text.strip()}")
            n_pages = len(doc)
        finally:
            doc.close()
        latency = time.perf_counter() - t0
        markdown = "\n\n".join(chunks)
        return ParseResult(
            markdown=markdown,
            page_count=n_pages,
            latency_sec=latency,
            cost_estimate_usd=parse_cost("pymupdf", pages=n_pages),
            pages_processed=n_pages,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
