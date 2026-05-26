"""Azure Document Intelligence layout processor.

Calls ``prebuilt-layout`` on a single page image (poller) and projects the
result's regions into a LayoutDocument. Auth: ``AZURE_DI_ENDPOINT`` +
``AZURE_DI_KEY``. Pricing: the ``azure_di`` per-page rate in catalog.yaml.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.layout.normalizers.azure_di import normalize_azure_di
from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.layout.processors.base import (
    LayoutProcessor,
    ProcessorResult,
    register_layout_processor,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost


@register_layout_processor("azure_di", version="1.0.0")
class AzureDI(LayoutProcessor):
    """Azure Document Intelligence ``prebuilt-layout`` over a single page image."""

    model_id: str = "prebuilt-layout"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            endpoint = os.environ.get("AZURE_DI_ENDPOINT")
            key = os.environ.get("AZURE_DI_KEY")
            if not endpoint or not key:
                raise RuntimeError("AZURE_DI_ENDPOINT / AZURE_DI_KEY not set")
            self._client = DocumentIntelligenceClient(
                endpoint=endpoint, credential=AzureKeyCredential(key)
            )
        return self._client

    def config_hash(self) -> str:
        return sha256_text(f"{self.name}|{self.version}|model={self.model_id}")[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt  # not used
        client = self._ensure_client()
        t0 = time.perf_counter()
        with image_path.open("rb") as f:
            poller = client.begin_analyze_document(self.model_id, body=f)
        result = poller.result()
        latency = time.perf_counter() - t0

        document = normalize_azure_di(result.as_dict(), source=image_path.name)
        return ProcessorResult(
            document=document,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=1),
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
