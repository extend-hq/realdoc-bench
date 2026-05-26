"""Cloud layout processor: Reducto.

Wraps Reducto's parse API as a ``LayoutProcessor``: upload a page image, run
the parse pipeline with ``chunk_mode=page`` + ``extraction_mode=ocr``, then
map the returned ``chunks[].blocks[]`` (with normalized 0-1 bboxes) into the
9-class public ``LayoutBlockType`` vocabulary.

Configure via env:
- ``REDUCTO_API_KEY`` — required.
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

# Reducto's parse API emits a small fixed vocab. Map to the 9 public classes.
# Anything not listed drops to "text" via the ``.get(..., "text")`` fallback.
_REDUCTO_TO_LAYOUT: dict[str, LayoutBlockType] = {
    "Title":          "heading",
    "Header":         "header",
    "Footer":         "footer",
    "Section Header": "section_heading",
    "Text":           "text",
    "Table":          "table",
    "Figure":         "figure",
    "Key Value":      "key_value",
    "List Item":      "text",       # folded
    "Page Number":    "page_number",
    "Comment":        "text",       # folded
    "Signature":      "text",       # folded
}


def _image_dims(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        return int(im.width), int(im.height)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Accessor that works on both Pydantic models and plain dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@register_layout_processor("reducto", version="parse-v1")
class ReductoLayout(LayoutProcessor):
    chunk_mode: str = "page"
    extraction_mode: str = "ocr"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from reducto import Reducto  # lazy import

            api_key = os.environ.get("REDUCTO_API_KEY")
            if not api_key:
                raise RuntimeError("REDUCTO_API_KEY not set")
            self._client = Reducto(api_key=api_key)
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            "|".join([self.name, self.version, self.chunk_mode, self.extraction_mode])
        )[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt
        client = self._ensure_client()
        width, height = _image_dims(image_path)

        t0 = time.perf_counter()
        upload = client.upload(file=image_path)
        parse_response = client.parse.run(
            input=upload.file_id,
            retrieval={"chunking": {"chunk_mode": self.chunk_mode}},
            settings={"extraction_mode": self.extraction_mode},
        )
        latency = time.perf_counter() - t0

        chunks = []
        result = _get(parse_response, "result")
        if result is not None:
            chunks = _get(result, "chunks") or []

        blocks: list[LayoutBlock] = []
        for chunk in chunks:
            for idx, b in enumerate(_get(chunk, "blocks") or [], start=1):
                btype_raw = _get(b, "type") or "Text"
                btype_str = btype_raw.value if hasattr(btype_raw, "value") else str(btype_raw)
                public = _REDUCTO_TO_LAYOUT.get(btype_str, "text")

                bbox = _get(b, "bbox")
                if bbox is None:
                    continue
                left   = _get(bbox, "left",   0.0) or 0.0
                top    = _get(bbox, "top",    0.0) or 0.0
                bwidth = _get(bbox, "width",  0.0) or 0.0
                bheight= _get(bbox, "height", 0.0) or 0.0

                blocks.append(
                    LayoutBlock(
                        id=_get(b, "id") or f"block_{idx:04d}",
                        block_type=cast(LayoutBlockType, public),
                        bbox=BBox(
                            x=int(round(left   * width)),
                            y=int(round(top    * height)),
                            w=int(round(bwidth * width)),
                            h=int(round(bheight* height)),
                        ),
                        text=_get(b, "content") or "",
                    )
                )

        document = LayoutDocument(
            document_id=image_path.stem,
            source=image_path.name,
            pages=[LayoutPage(page_number=1, width=width, height=height, blocks=blocks)],
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
