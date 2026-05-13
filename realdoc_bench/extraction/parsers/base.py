"""Parse-provider base + registry.

A ``ParseProvider`` takes a PDF path and returns markdown plus metadata.
Crucially, **markdown is the only thing the agent sees** — JSON-emitting
providers serialize internally inside the adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.registry import Registry


class ParseResult(BaseModel):
    markdown: str
    page_count: int | None = None
    latency_sec: float
    cost_estimate_usd: float | None = None
    pages_processed: int | None = None
    provider: str
    version: str
    config_hash: str
    raw: dict[str, Any] | None = None


class ParseProvider(ABC):
    name: str = ""
    version: str = ""

    def config_hash(self) -> str:
        return sha256_text(f"{self.name}|{self.version}")[:7]

    @abstractmethod
    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult: ...


registry: Registry[type[ParseProvider]] = Registry("parse provider")


def register_parser(name: str, *, version: str) -> Any:
    def _wrap(cls: type[ParseProvider]) -> type[ParseProvider]:
        cls.name = name
        cls.version = version
        registry.register(name, cls)
        return cls

    return _wrap


def build(name: str, **kwargs: Any) -> ParseProvider:
    return registry.get(name)(**kwargs)
