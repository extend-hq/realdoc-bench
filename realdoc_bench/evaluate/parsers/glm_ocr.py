"""GLM-OCR as an extraction parse provider (PDF → markdown).

GLM-OCR is a vision OCR model served over an OpenAI-compatible vLLM endpoint.
Each PDF page is rendered to an image and converted to Markdown (tables as
HTML). The per-page outputs are concatenated for the downstream extraction
agent.

Configure via env:
- ``GLM_OCR_BASE_URL`` — vLLM OpenAI base, e.g. ``http://localhost:8002/v1``
- ``GLM_OCR_MODEL``    — model id, default ``zai-org/GLM-OCR``

Note: there are two GLM-OCRs in play —
  * ``zai-org/GLM-OCR`` — the original open-source weights (host yourself).
  * the Extend-finetuned ``glm_ocr`` app on ``models.api.*.extend.ai`` — a
    different model; point ``GLM_OCR_BASE_URL`` at it only if you specifically
    want the finetuned variant in the comparison.
"""

from __future__ import annotations

from realdoc_bench.evaluate.parsers._vision_base import OpenAIVisionParserBase
from realdoc_bench.evaluate.parsers.base import register_parser

_MARKDOWN_PROMPT = (
    "Convert this document page to clean, faithful Markdown.\n"
    "- Preserve the reading order and heading hierarchy.\n"
    "- Render tables as HTML <table> markup.\n"
    "- Transcribe text exactly as shown; do not translate, summarize, or invent content.\n"
    "- Omit running headers/footers only if they are pure page furniture.\n"
    "Output only the Markdown, no commentary."
)


@register_parser("glm_ocr", version="zai-org/GLM-OCR")
class GlmOcrParser(OpenAIVisionParserBase):
    base_url_env = "GLM_OCR_BASE_URL"
    model_env = "GLM_OCR_MODEL"
    default_base_url = "http://localhost:8002/v1"
    default_model = "zai-org/GLM-OCR"
    prompt = _MARKDOWN_PROMPT
    pricing_key = "glm_ocr"
    max_tokens = 12000

    def _page_to_markdown(self, raw_text: str, page_no: int) -> str:
        s = raw_text.strip()
        # Strip a leading ```markdown / ``` fence if the model wrapped its output.
        if s.startswith("```"):
            nl = s.find("\n")
            if nl > 0:
                s = s[nl + 1 :]
            if s.endswith("```"):
                s = s[:-3]
        return s.strip()
