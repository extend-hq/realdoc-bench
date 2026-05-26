"""Cloud VLMs as extraction parse providers (PDF → markdown).

Gemini 3.5 Flash renders a PDF page-by-page and converts each page to Markdown,
concatenating the result for the downstream extraction agent — the "frontier VLM
as a parse pipeline" entry.

Costs are token-billed via the catalog (see ``shared/pricing/catalog.yaml``).

Env:
- ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` — Gemini.
"""

from __future__ import annotations

from realdoc_bench.evaluate.parsers._vision_base import GeminiVisionParserBase
from realdoc_bench.evaluate.parsers.base import register_parser

_MARKDOWN_PROMPT = (
    "Convert this document page to clean, faithful Markdown.\n"
    "- Preserve reading order and heading hierarchy (use `#`, `##`, `###`).\n"
    "- Render tables as GitHub-flavored markdown tables. If a cell spans "
    "multiple rows or columns, repeat its value in each cell so the table "
    "stays rectangular.\n"
    "- For figures and charts: if the figure shows numeric data (bar / line / "
    "pie / scatter / etc.), emit the underlying data as a markdown table. "
    "Otherwise write a one-line italic caption: `*Figure: <brief description>*`.\n"
    "- Render form fields as `**<label>:** <value>` on their own line. If the "
    "value is empty, emit `**<label>:** <blank>`.\n"
    "- Render checkboxes inline as ☒ (checked) or ☐ (unchecked), paired with "
    "their adjacent label on the same line.\n"
    "- Skip running page headers, page footers, and page numbers — they are "
    "page furniture, not document content.\n"
    "- Transcribe text exactly as shown — do not translate, summarize, "
    "paraphrase, or invent content.\n"
    "Output only the Markdown for this page, no commentary, no surrounding "
    "code fences."
)


@register_parser("gemini_3_5_flash", version="gemini-3.5-flash")
class Gemini35FlashParser(GeminiVisionParserBase):
    model = "gemini-3.5-flash"
    pricing_key = "gemini_3_5_flash"
    prompt = _MARKDOWN_PROMPT
