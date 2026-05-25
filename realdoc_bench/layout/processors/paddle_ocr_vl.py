"""Self-hosted OSS layout VLM: PaddleOCR-VL 1.5.

PaddleOCR-VL doesn't
have an OpenAI-compatible vLLM serving path, so realdoc-bench talks to a thin
FastAPI wrapper around ``paddleocr.PaddleOCRVL`` (default port 8001).

The server returns ``{"blocks": [{"label", "content", "bbox":[l,t,r,b]},
"width", "height"]}``; this adapter maps PaddleOCR-VL labels into
realdoc-bench's LayoutBlockType vocabulary.

Configure via env:
- ``PADDLE_OCR_VL_BASE_URL`` — server base, default ``http://localhost:8001``
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, cast

from realdoc_bench.layout.normalizers.base import (
    BBox,
    LayoutBlock,
    LayoutBlockType,
    LayoutDocument,
    LayoutPage,
)
from realdoc_bench.layout.processors.base import (
    LayoutProcessor,
    ProcessorResult,
    register_layout_processor,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost

# PP-DocLayoutV3 label → LayoutBlockType, expressed in realdoc-bench's
# LayoutBlockType vocab. Unknown labels fall back to "text".
_PADDLE_VL_TO_LAYOUT: dict[str, LayoutBlockType] = {
    "text": "text",
    "title": "heading",
    "doc_title": "heading",
    "paragraph_title": "section_heading",
    "section_header": "section_heading",
    "subheading": "section_heading",
    "header": "header",
    "footer": "footer",
    "page_number": "page_number",
    "figure": "figure",
    "seal": "figure",
    "table": "table",
    "key_value": "key_value",
    "form": "key_value",
    # Folded into text in the 9-class public vocab:
    "formula": "text",
    "equation": "text",
    "reference": "text",
    "abstract": "text",
    "figure_caption": "text",
    "table_caption": "text",
    "list_item": "text",
}

DEFAULT_BASE = "http://localhost:8001"
DEFAULT_TIMEOUT_SEC = 300.0


@register_layout_processor("paddle_ocr_vl", version="paddleocr-vl-1.5")
class PaddleOcrVlLayout(LayoutProcessor):
    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            base = os.environ.get("PADDLE_OCR_VL_BASE_URL", DEFAULT_BASE).rstrip("/")
            self._client = httpx.Client(base_url=base, timeout=DEFAULT_TIMEOUT_SEC)
        return self._client

    def config_hash(self) -> str:
        return sha256_text("|".join([self.name, self.version]))[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt
        client = self._ensure_client()
        t0 = time.perf_counter()
        with image_path.open("rb") as f:
            resp = client.post(
                "/predict",
                files={"file": (image_path.name, f, "image/png")},
            )
        resp.raise_for_status()
        body = resp.json()
        latency = time.perf_counter() - t0

        width = int(body.get("width") or 0)
        height = int(body.get("height") or 0)
        blocks: list[LayoutBlock] = []
        for idx, entry in enumerate(body.get("blocks") or [], start=1):
            label_raw = (entry.get("label") or "text").lower()
            bt = _PADDLE_VL_TO_LAYOUT.get(label_raw)
            if bt is None:
                continue
            bb = entry.get("bbox") or []
            if len(bb) < 4:
                continue
            left, top, right, bottom = bb[:4]
            blocks.append(
                LayoutBlock(
                    id=f"block_{idx:03d}",
                    block_type=cast(LayoutBlockType, bt),
                    bbox=BBox(
                        x=int(round(left)),
                        y=int(round(top)),
                        w=int(round(right - left)),
                        h=int(round(bottom - top)),
                    ),
                    text=entry.get("content") or "",
                )
            )

        document = LayoutDocument(
            document_id=image_path.stem,
            source=image_path.name,
            pages=[
                LayoutPage(
                    page_number=1,
                    width=width,
                    height=height,
                    blocks=blocks,
                )
            ],
        )

        return ProcessorResult(
            document=document,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=1),
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
