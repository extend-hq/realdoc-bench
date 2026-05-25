"""Paddle PP-Structure V3 layout processor.

PP-Structure V3 is
hosted as an HTTP endpoint (Ray Model Server or self-hosted equivalent).
Configure via env:

- ``PADDLE_STRUCTURE_BASE_URL`` — endpoint root, e.g. ``http://localhost:8080``
- ``PADDLE_STRUCTURE_AUTH_TOKEN`` — optional bearer token

The processor uploads the image as multipart/form-data and parses the JSON
response shape ``{pages: [{res: {parsing_res_list, width, height}}]}``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.layout.normalizers.paddle import normalize_paddle
from realdoc_bench.layout.processors.base import (
    LayoutProcessor,
    ProcessorResult,
    register_layout_processor,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost

DEFAULT_ENDPOINT = "/paddle/pp_structure_v3/blocks"
DEFAULT_TIMEOUT_SEC = 300.0


@register_layout_processor("paddle_structure_v3", version="3.0.0")
class PaddleStructureV3(LayoutProcessor):
    endpoint: str = DEFAULT_ENDPOINT
    page_rotation_enabled: bool = True
    predict_config: dict[str, Any] = {}

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            base = os.environ.get("PADDLE_STRUCTURE_BASE_URL", "").rstrip("/")
            if not base:
                raise RuntimeError("PADDLE_STRUCTURE_BASE_URL not set")
            token = os.environ.get("PADDLE_STRUCTURE_AUTH_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            self._client = httpx.Client(base_url=base, headers=headers, timeout=DEFAULT_TIMEOUT_SEC)
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            "|".join(
                [
                    self.name,
                    self.version,
                    self.endpoint,
                    str(self.page_rotation_enabled),
                    json.dumps(self.predict_config, sort_keys=True),
                ]
            )
        )[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt
        client = self._ensure_client()
        t0 = time.perf_counter()
        with image_path.open("rb") as f:
            resp = client.post(
                self.endpoint,
                files={"files": (image_path.name, f)},
                data={
                    "page_rotation_enabled": "true" if self.page_rotation_enabled else "false",
                    "predict_config": json.dumps(self.predict_config),
                },
            )
        resp.raise_for_status()
        body = resp.json()
        latency = time.perf_counter() - t0
        document = normalize_paddle(body, source=image_path.name)
        return ProcessorResult(
            document=document,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=1),
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
