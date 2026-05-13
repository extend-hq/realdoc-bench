"""Baseline processor: returns the ground truth as the prediction.

Useful for smoke-testing the runner / metrics / report pipelines without
touching any external API.
"""

from __future__ import annotations

import time
from pathlib import Path

from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.layout.processors.base import (
    LayoutProcessor,
    ProcessorResult,
    register_layout_processor,
)
from realdoc_bench.shared.pricing.meter import parse_cost


@register_layout_processor("gt_self", version="1.0.0")
class GtSelfProcessor(LayoutProcessor):
    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        if gt is None:
            raise ValueError("gt_self requires a ground-truth document to return")
        t0 = time.perf_counter()
        latency = time.perf_counter() - t0
        return ProcessorResult(
            document=gt.model_copy(deep=True),
            latency_sec=latency,
            cost_estimate_usd=parse_cost("gt_self", pages=1),
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
