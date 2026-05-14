"""Cloud VLMs as extraction parse providers (PDF → markdown).

Gemini 3, Claude Sonnet 4.7, and Claude Opus 4.6 each render a PDF page-by-page
and convert each page to Markdown, concatenating the result for the downstream
extraction agent. These are the "frontier VLM as a parse pipeline" entries.

Costs are token-billed via the catalog (see ``shared/pricing/catalog.yaml``).

Env:
- ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` — Gemini.
- ``ANTHROPIC_API_KEY`` — Claude (Sonnet + Opus).
"""

from __future__ import annotations

from realdoc_bench.extraction.parsers._vision_base import (
    AnthropicVisionParserBase,
    GeminiVisionParserBase,
)
from realdoc_bench.extraction.parsers.base import register_parser

_MARKDOWN_PROMPT = (
    "Convert this document page to clean, faithful Markdown.\n"
    "- Preserve reading order and heading hierarchy.\n"
    "- Render tables as HTML <table> markup; render formulas as LaTeX.\n"
    "- Transcribe text exactly as shown — do not translate, summarize, or invent.\n"
    "- Render checkboxes as ☒ (checked) or ☐ (unchecked).\n"
    "Output only the Markdown for this page, no commentary."
)


@register_parser("gemini_3_flash", version="gemini-3-flash-preview")
class Gemini3FlashParser(GeminiVisionParserBase):
    model = "gemini-3-flash-preview"
    pricing_key = "gemini_3_flash"
    prompt = _MARKDOWN_PROMPT


@register_parser("claude_sonnet_4_7", version="claude-sonnet-4-7")
class SonnetParser(AnthropicVisionParserBase):
    model = "claude-sonnet-4-7"
    pricing_key = "claude_sonnet_4_7"
    prompt = _MARKDOWN_PROMPT
    max_tokens = 16384


@register_parser("claude_opus_4_6", version="claude-opus-4-6")
class OpusParser(AnthropicVisionParserBase):
    model = "claude-opus-4-6"
    pricing_key = "claude_opus_4_6"
    prompt = _MARKDOWN_PROMPT
    max_tokens = 16384
