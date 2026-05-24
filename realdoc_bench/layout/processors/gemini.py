"""Cloud VLM layout processor: Google Gemini.

Uses ``google.genai`` to
prompt a Gemini model for a JSON layout decomposition, then normalizes via
:mod:`realdoc_bench.layout.normalizers.vlm_json`.

Gemini is prompted to return NORMALIZED 0–1 bbox coordinates (more reliable
than asking it for pixel values). We scale them up to pixels before handing
to the shared normalizer, which expects pixel coords.

Configure via env:
- ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` — required (either is read).
"""

from __future__ import annotations

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


def _build_user_prompt(width: int, height: int) -> str:
    return f"""Analyze this document image ({width}x{height} pixels) and extract its layout and text content.

Identify all blocks including:
- text: paragraphs, body text, and key-value/form regions
- heading: document titles
- section_heading: section headers
- table: data tables (extract as a SINGLE table block per logical table)
- figure: images, diagrams, barcodes, and QR codes
- header: running page headers
- footer: running page footers
- page_number: standalone printed page numbers
- figure_caption: captions for figures

For each block provide:
1. Bounding box as NORMALIZED coordinates (values between 0.0 and 1.0, where (0,0) is top-left and (1,1) is bottom-right)
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
      "bounding_box": {{"left": 0.05, "top": 0.02, "right": 0.95, "bottom": 0.15}}
    }}]
  }}]
}}
"""


_SYSTEM_PROMPT = (
    "You are an advanced document layout and OCR engine. "
    "Detect all document elements with precise bounding boxes returned as "
    "NORMALIZED coordinates (0.0 to 1.0). "
    "For tables: reproduce values faithfully; preserve structure; merge "
    "continuous columns into ONE logical table. "
    "Use <table>/<thead>/<tbody>, render checkboxes as ☒/☐, no styling. "
    "Always respond with valid JSON only."
)


def _image_dims(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        return int(im.width), int(im.height)


def _scale_bboxes(data: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    for chunk in data.get("chunks", []) or []:
        for block in chunk.get("blocks", []) or []:
            bb = block.get("bounding_box") or {}
            left = bb.get("left")
            top = bb.get("top")
            right = bb.get("right")
            bottom = bb.get("bottom")
            if any(v is None for v in (left, top, right, bottom)):
                continue
            if all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in (left, top, right, bottom)):
                block["bounding_box"] = {
                    "left": left * width,
                    "top": top * height,
                    "right": right * width,
                    "bottom": bottom * height,
                }
            meta = block.setdefault("metadata", {})
            meta.setdefault("page", {"number": 1, "width": width, "height": height})
    return data


class _GeminiLayoutBase(LayoutProcessor):
    model: str = "gemini-3-flash-preview"
    pricing_key: str = "gemini_3_flash"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def config_hash(self) -> str:
        return sha256_text("|".join([self.name, self.version, self.model]))[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt
        from google.genai import types

        client = self._ensure_client()
        width, height = _image_dims(image_path)
        raw = image_path.read_bytes()
        ext = image_path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        image_part = types.Part.from_bytes(data=raw, mime_type=mime)

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=self.model,
            contents=[image_part, _build_user_prompt(width, height)],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )
        latency = time.perf_counter() - t0

        try:
            data = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            data = {"chunks": []}
        data = _scale_bboxes(data, width, height)
        document = normalize_vlm_json(data, source=image_path.name)

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        cost = agent_cost(self.pricing_key, input_tokens=int(input_tokens or 0), output_tokens=int(output_tokens or 0))

        return ProcessorResult(
            document=document,
            latency_sec=latency,
            cost_estimate_usd=cost,
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )


@register_layout_processor("gemini_3_flash", version="gemini-3-flash-preview")
class Gemini3FlashLayout(_GeminiLayoutBase):
    model = "gemini-3-flash-preview"
    pricing_key = "gemini_3_flash"


@register_layout_processor("gemini_flash_lite", version="gemini-3.1-flash-lite-preview")
class GeminiFlashLiteLayout(_GeminiLayoutBase):
    model = "gemini-3.1-flash-lite-preview"
    pricing_key = "gemini_flash_lite"


@register_layout_processor("gemini_pro", version="gemini-3-pro-preview")
class GeminiProLayout(_GeminiLayoutBase):
    model = "gemini-3-pro-preview"
    pricing_key = "gemini_pro"
