"""ELRND raw-model layout processor — hits ``models.api.extend.{ai,app}/elrnd_l/infer``.

This is the "raw model" path: uploads an image, gets back per-detection
confidences with no post-processing or block merging applied. Use this when
you want to sweep confidence thresholds yourself for F1-optimal numbers,
matching the methodology in the original Extend v2 Benchmark Snapshot.

Configure via env:

- ``EXTEND_MODELS_API_BASE`` — default ``https://models.api.staging.extend.ai``;
  set to ``https://models.api.extend.ai`` for production.
- ``EXTEND_MODELS_API_TOKEN`` — bearer token. Required for production (and
  ``us2-stg`` / ``eu1`` / ``us2``); not required for ``dev1`` or ``staging``.
- ``ELRND_L_APP`` — model app name; default ``elrnd_l``. Set to
  ``elrnd_l_dev`` to hit the dev slot (next-candidate checkpoint).

Query params:

- ``conf`` — confidence cutoff. We default to ``0.0`` so the response carries
  every detection and downstream code can sweep thresholds.
- ``page_rotation_enabled`` — default ``True``; mirrors the production path.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.layout.normalizers.elrnd import normalize_elrnd
from realdoc_bench.layout.processors.base import (
    LayoutProcessor,
    ProcessorResult,
    register_layout_processor,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost

DEFAULT_BASE = "https://models.api.staging.extend.ai"
DEFAULT_TIMEOUT_SEC = 60.0


class _ElrndLApiBase(LayoutProcessor):
    app: str = "elrnd_l"
    conf: float = 0.0  # request every detection so callers can sweep
    page_rotation_enabled: bool = True

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            base = os.environ.get("EXTEND_MODELS_API_BASE", DEFAULT_BASE).rstrip("/")
            token = os.environ.get("EXTEND_MODELS_API_TOKEN")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            self._client = httpx.Client(base_url=base, headers=headers, timeout=DEFAULT_TIMEOUT_SEC)
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            "|".join(
                [
                    self.name,
                    self.version,
                    self.app,
                    f"conf={self.conf}",
                    f"rot={self.page_rotation_enabled}",
                ]
            )
        )[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt
        client = self._ensure_client()
        t0 = time.perf_counter()
        with image_path.open("rb") as f:
            resp = client.post(
                f"/{os.environ.get('ELRND_L_APP', self.app)}/infer",
                files={"file": (image_path.name, f, "image/png")},
                params={
                    "conf": str(self.conf),
                    "page_rotation_enabled": "true" if self.page_rotation_enabled else "false",
                },
            )
        resp.raise_for_status()
        detections = resp.json()
        latency = time.perf_counter() - t0
        document = normalize_elrnd(detections, source=image_path.name)
        return ProcessorResult(
            document=document,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=1),
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )


@register_layout_processor("elrnd_l_api", version="qvmdexq-2100")
class ElrndLApi(_ElrndLApiBase):
    app = "elrnd_l"


@register_layout_processor("elrnd_l_dev_api", version="light-pond-274-2400")
class ElrndLDevApi(_ElrndLApiBase):
    app = "elrnd_l_dev"
