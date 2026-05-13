"""Extend layout processor adapters.

Wraps the Extend parse API in the ``LayoutProcessor`` interface. Multiple engine
versions register as distinct processor names so the leaderboard can compare them.

The Extend SDK is imported lazily so the package remains importable without the
dependency installed (useful for offline tests). Set ``EXTEND_API_KEY``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.layout.normalizers.extend import to_layout_document
from realdoc_bench.layout.processors.base import (
    LayoutProcessor,
    ProcessorResult,
    register_layout_processor,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost


class _ExtendBase(LayoutProcessor):
    engine: str = "parse_performance"
    engine_version: str = ""
    target: str = "markdown"
    config_extra: dict[str, Any] = {}

    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from extend_ai import Extend  # imported lazily

            self._client = Extend(api_key=os.environ.get("EXTEND_API_KEY"))
        return self._client

    def config_hash(self) -> str:
        return sha256_text(
            "|".join(
                [
                    self.name,
                    self.version,
                    self.engine,
                    self.engine_version,
                    self.target,
                    repr(sorted(self.config_extra.items())),
                ]
            )
        )[:7]

    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        del gt  # not used
        client = self._ensure_client()
        t0 = time.perf_counter()
        config = {
            "engine": self.engine,
            "engineVersion": self.engine_version,
            "target": self.target,
            **self.config_extra,
        }
        run = client.parse.run(file_path=str(image_path), config=config, wait=True)
        latency = time.perf_counter() - t0
        document = to_layout_document(run, source=image_path.name)
        return ProcessorResult(
            document=document,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=1),
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )


@register_layout_processor("extend_v2_0_0_beta", version="2.0.0-beta")
class ExtendV2(_ExtendBase):
    engine = "parse_performance"
    engine_version = "2.0.0-beta"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {"enabled": True, "figureImageClippingEnabled": True},
            "tables": {"enabled": True, "targetFormat": "markdown"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
            "formulas": {"enabled": True},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }


@register_layout_processor("extend_v1_0_1", version="1.0.1")
class ExtendV1(_ExtendBase):
    engine = "parse_performance"
    engine_version = "1.0.1"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {"enabled": True, "figureImageClippingEnabled": True},
            "tables": {"enabled": True, "targetFormat": "html"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }


@register_layout_processor("extend_light_v1_0_0", version="1.0.0")
class ExtendLightV1(_ExtendBase):
    engine = "parse_light"
    engine_version = "1.0.0"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {"enabled": True, "figureImageClippingEnabled": True},
            "tables": {"enabled": True, "targetFormat": "html"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
            "formulas": {"enabled": True},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }


@register_layout_processor("extend_light_v0_0_2_beta", version="0.0.2-beta")
class ExtendLightV002(_ExtendBase):
    engine = "parse_light"
    engine_version = "0.0.2-beta"
    config_extra = {
        "chunkingStrategy": {"type": "page"},
        "blockOptions": {
            "figures": {"enabled": True, "figureImageClippingEnabled": True},
            "tables": {"enabled": True, "targetFormat": "html"},
            "text": {"enabled": True, "styleFormattingEnabled": False},
        },
        "advancedOptions": {"pageRotationEnabled": True},
    }
