"""Self-hosted OSS layout VLM: rednote-hilab/dots.mocr 1.5.

Hits a vLLM OpenAI-compatible endpoint
(e.g. ``vllm serve rednote-hilab/dots.mocr``, default port 8000).

The model is pinned to ``rednote-hilab/dots.mocr``. Configure via env:
- ``DOTS_OCR_BASE_URL`` — vLLM base URL, default ``http://localhost:8000/v1``
- ``DOTS_OCR_API_KEY`` — optional bearer (default ``EMPTY``)

The model emits a JSON array of ``{category, bbox=[l,t,r,b], text}``;
we re-shape into the ``normalize_vlm_json`` expected dict.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.layout.normalizers.vlm_json import normalize_vlm_json
from realdoc_bench.layout.processors.base import (
    LayoutProcessor,
    ProcessorResult,
    register_layout_processor,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost

DOTS_PROMPT = (
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

# Map dots.mocr categories to realdoc-bench's vlm_json intermediate vocabulary.
# Keys are normalized (lowercase, hyphen/space → underscore) so case + separator
# variants from the model all hit the same bucket.
_DOTS_RAW_TO_VLM: dict[str, str] = {
    "text": "text",
    "title": "heading",
    "section_header": "section_heading",
    "section_heading": "section_heading",
    "subheading": "section_heading",
    "heading": "heading",
    "table": "table",
    "picture": "figure",
    "figure": "figure",
    "image": "figure",
    "chart": "figure",
    "caption": "figure_caption",
    "figure_caption": "figure_caption",
    "table_caption": "figure_caption",
    "formula": "formula",
    "equation": "formula",
    "list_item": "list_item",
    "list": "list_item",
    "page_footer": "page_footer",
    "footer": "page_footer",
    "page_header": "page_header",
    "header": "page_header",
    "footnote": "footnote",
    "endnote": "footnote",
    "page_number": "page_number",
    "key_value": "key_value",
    "form": "key_value",
}


def _normalize_category(name: str) -> str:
    """Lowercase + collapse hyphens / spaces to underscores so 'Page-header',
    'page header', 'Page_Header' all hit the same key."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _category_to_vlm_type(name: str) -> str | None:
    return _DOTS_RAW_TO_VLM.get(_normalize_category(name))


# Backwards-compatible export — used by tests / external code.
DOTS_CATEGORY_TO_VLM_TYPE: dict[str, str] = {
    "Text": "text",
    "Title": "heading",
    "Section-header": "section_heading",
    "Table": "table",
    "Picture": "figure",
    "Caption": "figure_caption",
    "Formula": "formula",
    "List-item": "list_item",
    "Page-footer": "page_footer",
    "Page-header": "page_header",
    "Footnote": "footnote",
}


def _image_dims(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        return int(im.width), int(im.height)


def _detect_mime(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/jpeg"


def _parse_dots_array(text: str) -> list[dict[str, Any]]:
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    def _extract_array(obj: Any) -> list[dict[str, Any]]:
        if isinstance(obj, list):
            return [e for e in obj if isinstance(e, dict)]
        if isinstance(obj, dict):
            # Newer dots prompt returns a single JSON object — look for the array under
            # known keys, otherwise pick the first list-of-dicts value.
            for key in ("layout", "blocks", "elements", "items"):
                if isinstance(obj.get(key), list):
                    return [e for e in obj[key] if isinstance(e, dict)]
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return [e for e in v if isinstance(e, dict)]
        return []

    try:
        return _extract_array(json.loads(s))
    except json.JSONDecodeError:
        lo, hi = s.find("["), s.rfind("]")
        if lo >= 0 and hi > lo:
            try:
                return _extract_array(json.loads(s[lo : hi + 1]))
            except json.JSONDecodeError:
                pass
        lo2, hi2 = s.find("{"), s.rfind("}")
        if lo2 >= 0 and hi2 > lo2:
            try:
                return _extract_array(json.loads(s[lo2 : hi2 + 1]))
            except json.JSONDecodeError:
                pass
    return []


def _dots_to_vlm_json(elements: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for idx, e in enumerate(elements, start=1):
        cat = e.get("category") or e.get("type") or e.get("label") or "Text"
        bt = _category_to_vlm_type(cat) or "text"
        bb = e.get("bbox") or e.get("box") or []
        if len(bb) < 4:
            continue
        blocks.append(
            {
                "id": f"block_{idx:03d}",
                "type": bt,
                "content": e.get("text") or "",
                "metadata": {"page": {"number": 1, "width": width, "height": height}},
                "bounding_box": {
                    "left": bb[0],
                    "top": bb[1],
                    "right": bb[2],
                    "bottom": bb[3],
                },
            }
        )
    return {"chunks": [{"blocks": blocks}]}


@register_layout_processor("dots_ocr", version="rednote-hilab/dots.mocr")
class DotsOcrLayout(LayoutProcessor):
    model: str = "rednote-hilab/dots.mocr"
    max_tokens: int = 12000

    def __init__(self) -> None:
        self._client: Any | None = None

    # Connection errors are common against a self-hosted vLLM endpoint whose
    # link can flap. Retry the whole request a few times with backoff; rebuild
    # the client each attempt so a dropped connection pool doesn't poison
    # subsequent calls.
    max_retries: int = 6
    retry_backoff_sec: float = 5.0

    def _new_client(self) -> Any:
        from openai import OpenAI

        base_url = os.environ.get("DOTS_OCR_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("DOTS_OCR_API_KEY") or "EMPTY"
        # Per-request timeout generous enough for a 12k-token layout dump.
        return OpenAI(api_key=api_key, base_url=base_url, timeout=300.0, max_retries=0)

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._new_client()
        return self._client

    def config_hash(self) -> str:
        return sha256_text("|".join([self.name, self.version, self.model, str(self.max_tokens)]))[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt
        width, height = _image_dims(image_path)
        raw = image_path.read_bytes()
        mime = _detect_mime(raw)
        encoded = base64.b64encode(raw).decode("utf-8")
        data_url = f"data:{mime};base64,{encoded}"

        t0 = time.perf_counter()
        response = None
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                client = self._ensure_client()
                response = client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": DOTS_PROMPT},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                )
                break
            except Exception as e:  # noqa: BLE001 — connection / timeout / 5xx all retryable
                last_err = e
                self._client = None  # force a fresh client (+ fresh conn pool) next try
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
        if response is None:
            raise RuntimeError(
                f"dots_ocr request failed after {self.max_retries} attempts: {last_err}"
            )
        latency = time.perf_counter() - t0

        content = response.choices[0].message.content if response.choices else ""
        elements = _parse_dots_array(content or "")
        vlm_data = _dots_to_vlm_json(elements, width, height)
        document = normalize_vlm_json(vlm_data, source=image_path.name)

        return ProcessorResult(
            document=document,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=1),
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
