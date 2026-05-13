"""Cloud VLM layout processor: OpenAI GPT.

Adapted from ``ocr-bench/processors/openai_gpt.py``. Calls the OpenAI chat
completions API with an image and a layout-detection prompt; the model is
expected to emit JSON whose shape matches Extend's ParserRun (chunks → blocks
with ``type``, ``content``, ``bounding_box``, ``metadata.page``), parsed via
:mod:`realdoc_bench.layout.normalizers.vlm_json`.

Configure via env:
- ``OPENAI_API_KEY`` — required.
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
from realdoc_bench.shared.pricing.meter import agent_cost

_SYSTEM_PROMPT = (
    "You are an advanced document layout and OCR engine. "
    "Extract all document elements accurately with precise bounding boxes in PIXEL units. "
    "For tables: reproduce values faithfully; preserve structure with <table>/<thead>/<tbody>. "
    "Use <th> for headers, <td> for data. Combine continuous columns into ONE table. "
    "Render checkboxes as ☒/☐. Always respond with valid JSON only."
)


def _build_user_prompt(width: int, height: int) -> str:
    return f"""Analyze this document image ({width}x{height} pixels) and extract its layout and text content.

Identify all blocks including:
- text: paragraphs and body text
- heading: document titles
- section_heading: section headers
- table: data tables (extract as a SINGLE table block per logical table)
- figure: images and diagrams
- figure_caption: captions for figures

For each block provide:
1. Bounding box (left, top, right, bottom in PIXELS, between 0 and image dimensions)
2. Block type
3. Content

Return results as JSON with this exact structure:
{{
  "chunks": [{{
    "blocks": [{{
      "id": "block_001",
      "type": "text",
      "content": "...",
      "metadata": {{"page": {{"number": 1, "width": {width}, "height": {height}}}}},
      "bounding_box": {{"left": 0, "top": 0, "right": {width}, "bottom": {height}}}
    }}]
  }}]
}}
"""


def _image_dims(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        return int(im.width), int(im.height)


def _detect_mime(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


class _GPTLayoutBase(LayoutProcessor):
    model: str = "gpt-5.4"
    pricing_key: str = "gpt_5_4"
    max_tokens: int = 16384

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
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
            model=self.model,
            max_completion_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_user_prompt(width, height)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        latency = time.perf_counter() - t0

        content = response.choices[0].message.content if response.choices else ""
        content = _strip_code_fence(content or "")
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"chunks": []}

        for chunk in data.get("chunks", []) or []:
            for block in chunk.get("blocks", []) or []:
                meta = block.setdefault("metadata", {})
                meta.setdefault("page", {"number": 1, "width": width, "height": height})

        document = normalize_vlm_json(data, source=image_path.name)
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = agent_cost(self.pricing_key, input_tokens=int(input_tokens), output_tokens=int(output_tokens))

        return ProcessorResult(
            document=document,
            latency_sec=latency,
            cost_estimate_usd=cost,
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )


@register_layout_processor("gpt_5_4", version="gpt-5.4")
class GPT54Layout(_GPTLayoutBase):
    model = "gpt-5.4"
    pricing_key = "gpt_5_4"
