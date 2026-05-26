"""Layout processor base + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.registry import Registry


class ProcessorResult(BaseModel):
    document: LayoutDocument
    latency_sec: float
    cost_estimate_usd: float | None = None
    pages_processed: int = 1
    provider: str
    version: str
    config_hash: str
    raw: dict[str, Any] | None = None


class LayoutProcessor(ABC):
    """Predict a LayoutDocument for a single page image.

    Subclasses register via the module-level :data:`registry` decorator.
    """

    name: str = ""
    version: str = ""

    def config_hash(self) -> str:
        """Hash that varies when meaningful config changes; cache buster."""
        return sha256_text(f"{self.name}|{self.version}")[:7]

    @abstractmethod
    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        """Return a ``ProcessorResult`` for ``image_path``.

        ``gt`` is optionally provided for trivial baselines (e.g. ``gt_self``).
        """


registry: Registry[type[LayoutProcessor]] = Registry("layout processor")


def register_layout_processor(name: str, *, version: str) -> Any:
    """Class decorator: register a layout processor under ``name`` + ``version``."""

    def _wrap(cls: type[LayoutProcessor]) -> type[LayoutProcessor]:
        cls.name = name
        cls.version = version
        registry.register(name, cls)
        return cls

    return _wrap


def build(name: str, **kwargs: Any) -> LayoutProcessor:
    return registry.get(name)(**kwargs)
