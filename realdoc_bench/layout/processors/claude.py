"""Cloud VLM layout processor: Anthropic Claude.

The model is
prompted to emit a JSON structure mirroring Extend's ``ParserRun`` shape
(chunks → blocks with ``type``, ``content``, ``bounding_box``,
``metadata.page``) which the :mod:`realdoc_bench.layout.normalizers.vlm_json`
module then turns into a ``LayoutDocument``.

Pricing is token-based — ``cost_estimate_usd`` is computed from the API's
usage counts at the per-model rate in ``shared/pricing/catalog.yaml``.

Configure via env:
- ``ANTHROPIC_API_KEY`` — required.
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
    "Extract all document elements accurately with precise bounding boxes. "
    "For tables: reproduce values faithfully, fix OCR artifacts, preserve structure and semantics. "
    "Use <table> with <thead> and <tbody>. Use <th> for headers, <td> for data. "
    "Combine related sections with same columns into ONE logical table. "
    "Render checkboxes as ☒ (checked) or ☐ (unchecked). "
    "Always respond with valid JSON only."
)


def _build_user_prompt(width: int, height: int) -> str:
    return f"""Analyze this document image ({width}x{height} pixels) and extract its layout and text content.

Identify all blocks including:
- text: paragraphs and body text (including captions, footnotes, formulas, list items)
- heading: document titles
- section_heading: section headers
- header: running page headers
- footer: running page footers
- page_number: page-number indicators
- table: data tables (extract as a SINGLE table block per logical table)
- figure: images, diagrams, charts, barcodes
- key_value: form fields / key-value pairs

For each block provide:
1. Bounding box (left, top, right, bottom in pixels)
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


def _detect_mime(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def _image_dims(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        return int(im.width), int(im.height)


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


class _ClaudeLayoutBase(LayoutProcessor):
    model: str = "claude-sonnet-4-6"
    pricing_key: str = "claude_sonnet_4_6"
    max_tokens: int = 16384
    max_retries: int = 3

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = Anthropic(api_key=api_key)
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            "|".join([self.name, self.version, self.model, str(self.max_tokens)])
        )[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt
        client = self._ensure_client()
        width, height = _image_dims(image_path)
        raw = image_path.read_bytes()
        mime = _detect_mime(raw)
        encoded = base64.b64encode(raw).decode("utf-8")

        t0 = time.perf_counter()
        response = None
        for attempt in range(self.max_retries):
            try:
                response = client.messages.create(
                    model=self.model,
                    system=_SYSTEM_PROMPT,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": mime, "data": encoded},
                                },
                                {"type": "text", "text": _build_user_prompt(width, height)},
                            ],
                        }
                    ],
                )
                break
            except Exception as e:
                msg = str(e).lower()
                if "rate" in msg or "overload" in msg or "429" in msg:
                    if attempt + 1 == self.max_retries:
                        raise
                    time.sleep(min(10 * (2**attempt), 30))
                else:
                    raise
        assert response is not None
        latency = time.perf_counter() - t0

        text = response.content[0].text if response.content else ""
        text = _strip_code_fence(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Treat as empty result so the runner records a zero-block prediction
            data = {"chunks": []}

        # Stamp page metadata if the model omitted it
        for chunk in data.get("chunks", []) or []:
            for block in chunk.get("blocks", []) or []:
                meta = block.setdefault("metadata", {})
                meta.setdefault("page", {"number": 1, "width": width, "height": height})

        document = normalize_vlm_json(data, source=image_path.name)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
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


@register_layout_processor("sonnet_4_6", version="claude-sonnet-4-6")
class SonnetLayout(_ClaudeLayoutBase):
    model = "claude-sonnet-4-6"
    pricing_key = "claude_sonnet_4_6"


@register_layout_processor("opus_4_6", version="claude-opus-4-6")
class OpusLayout(_ClaudeLayoutBase):
    model = "claude-opus-4-6"
    pricing_key = "claude_opus_4_6"
