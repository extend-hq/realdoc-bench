"""AWS Textract layout processor.

Calls ``analyze_document`` on a single page image with the LAYOUT feature set
and projects the ``LAYOUT_*`` regions into a LayoutDocument. Auth uses the
default boto3 credential chain (``AWS_PROFILE`` / ``AWS_REGION`` honored).
Pricing: the ``aws_textract`` per-page rate in catalog.yaml.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.layout.normalizers.aws_textract import normalize_aws_textract
from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.layout.processors.base import (
    LayoutProcessor,
    ProcessorResult,
    register_layout_processor,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost


@register_layout_processor("aws_textract", version="1.0.0")
class AWSTextract(LayoutProcessor):
    """AWS Textract ``analyze_document`` over a single page image."""

    features: list[str] = ["FORMS", "TABLES", "LAYOUT", "SIGNATURES"]

    def __init__(self) -> None:
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import boto3

            profile = os.environ.get("AWS_PROFILE")
            region = os.environ.get("AWS_REGION", "us-east-1")
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self._client = session.client("textract", region_name=region)
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            f"{self.name}|{self.version}|features={','.join(sorted(self.features))}"
        )[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt  # not used
        client = self._ensure_client()
        with image_path.open("rb") as f:
            image_bytes = f.read()
        t0 = time.perf_counter()
        response = client.analyze_document(
            Document={"Bytes": image_bytes}, FeatureTypes=self.features
        )
        latency = time.perf_counter() - t0

        from PIL import Image

        with Image.open(image_path) as im:
            page_width, page_height = im.size

        document = normalize_aws_textract(
            response, page_width=page_width, page_height=page_height, source=image_path.name
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
