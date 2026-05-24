"""dots.mocr as an extraction parse provider (PDF → markdown).

Uses dots.mocr's official ``prompt_layout_all_en`` so each page comes back as
a JSON array of layout elements (bbox + category + text). We then emit the
``text`` fields in reading order, with tables kept as HTML and formulas as
LaTeX (the model already formats them that way per the prompt). Pictures are
dropped (no text). The result is structure-preserving markdown for the
downstream extraction agent.

Configure via env:
- ``DOTS_OCR_BASE_URL`` — vLLM OpenAI base, default ``http://localhost:8001/v1``
- ``DOTS_OCR_MODEL`` — HF id, default ``rednote-hilab/dots.mocr``
"""

from __future__ import annotations

import json
from typing import Any

from realdoc_bench.evaluate.parsers._vision_base import OpenAIVisionParserBase
from realdoc_bench.evaluate.parsers.base import register_parser

_LAYOUT_ALL_PROMPT = (
    "Please output the layout information from the PDF image, including each "
    "layout element's bbox, its category, and the corresponding text content "
    "within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', "
    "'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', "
    "'Section-header', 'Table', 'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object.\n"
)

# How to render each element's text into the page markdown.
_HEADING_CATS = {"title": "# ", "section-header": "## "}


def _extract_elements(raw: str) -> list[dict[str, Any]]:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    def _as_list(obj: Any) -> list[dict[str, Any]]:
        if isinstance(obj, list):
            return [e for e in obj if isinstance(e, dict)]
        if isinstance(obj, dict):
            for key in ("layout", "blocks", "elements", "items"):
                if isinstance(obj.get(key), list):
                    return [e for e in obj[key] if isinstance(e, dict)]
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return [e for e in v if isinstance(e, dict)]
        return []

    try:
        return _as_list(json.loads(s))
    except json.JSONDecodeError:
        for lo, hi in ((s.find("["), s.rfind("]")), (s.find("{"), s.rfind("}"))):
            if lo >= 0 and hi > lo:
                try:
                    return _as_list(json.loads(s[lo : hi + 1]))
                except json.JSONDecodeError:
                    continue
    return []


@register_parser("dots_mocr", version="rednote-hilab/dots.mocr")
class DotsMocrParser(OpenAIVisionParserBase):
    base_url_env = "DOTS_OCR_BASE_URL"
    model_env = "DOTS_OCR_MODEL"
    default_base_url = "http://localhost:8001/v1"
    default_model = "rednote-hilab/dots.mocr"
    prompt = _LAYOUT_ALL_PROMPT
    pricing_key = "dots_mocr"
    max_tokens = 12000

    def _page_to_markdown(self, raw_text: str, page_no: int) -> str:
        elements = _extract_elements(raw_text)
        if not elements:
            # Fall back to whatever the model returned verbatim.
            return raw_text.strip()
        lines: list[str] = []
        for e in elements:
            cat = str(e.get("category", "Text")).strip().lower()
            text = (e.get("text") or "").strip()
            if cat == "picture" or not text:
                continue
            prefix = _HEADING_CATS.get(cat, "")
            lines.append(f"{prefix}{text}")
        return "\n\n".join(lines)
