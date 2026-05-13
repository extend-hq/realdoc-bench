"""Self-hosted OSS layout VLM: rednote-hilab/dots.ocr 1.5.

Adapted from ``ocr-bench/processors/dots_ocr.py``. Hits a vLLM
OpenAI-compatible endpoint (typically `vllm serve rednote-hilab/dots.ocr`
on awsl4 port 8000, exposed via SSH tunnel).

Configure via env:
- ``DOTS_OCR_BASE_URL`` — vLLM base URL, default ``http://localhost:8000/v1``
- ``DOTS_OCR_MODEL`` — HF repo id, default ``rednote-hilab/dots.ocr``
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
    "within the bbox.\n"
    "Bbox format: [x1, y1, x2, y2]\n"
    "Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', "
    "'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', "
    "'Text', 'Title']\n"
    "Format Rules:\n"
    "- Picture: text should be omitted\n"
    "- Formula: formatted as LaTeX\n"
    "- Table: formatted as HTML\n"
    "- All others: formatted as Markdown\n"
    "Output as JSON array sorted by reading order."
)

# Map dots-ocr categories to the realdoc-bench LayoutBlockType vocabulary,
# already pre-collapsed where appropriate so downstream 9-class remap works.
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
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else []
    except json.JSONDecodeError:
        lo, hi = s.find("["), s.rfind("]")
        if lo >= 0 and hi > lo:
            try:
                out = json.loads(s[lo : hi + 1])
                return out if isinstance(out, list) else []
            except json.JSONDecodeError:
                pass
    return []


def _dots_to_vlm_json(elements: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for idx, e in enumerate(elements, start=1):
        cat = e.get("category", "Text")
        bt = DOTS_CATEGORY_TO_VLM_TYPE.get(cat, "text")
        bb = e.get("bbox") or []
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


@register_layout_processor("dots_ocr", version="rednote-hilab/dots.ocr-1.5")
class DotsOcrLayout(LayoutProcessor):
    model: str = "rednote-hilab/dots.ocr"
    max_tokens: int = 12000

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            base_url = os.environ.get("DOTS_OCR_BASE_URL", "http://localhost:8000/v1")
            api_key = os.environ.get("DOTS_OCR_API_KEY") or "EMPTY"
            self._model_id = os.environ.get("DOTS_OCR_MODEL") or self.model
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def config_hash(self) -> str:
        return sha256_text("|".join([self.name, self.version, self.model, str(self.max_tokens)]))[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt
        client = self._ensure_client()
        width, height = _image_dims(image_path)
        raw = image_path.read_bytes()
        mime = _detect_mime(raw)
        encoded = base64.b64encode(raw).decode("utf-8")
        data_url = f"data:{mime};base64,{encoded}"

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self._model_id,
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
