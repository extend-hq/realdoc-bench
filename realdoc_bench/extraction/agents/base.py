"""Extraction agent base + registry.

Critical contract: ``extract(markdown, schema, doc_id)`` is the only entry
point. Agents must never read the original PDF — they only see the parsed
markdown. The runner validates this by passing markdown text into ``extract``
directly; agents that try to open files inside that call will produce a
``SCHEMA_VIOLATION`` or ``FAILED`` status because the path is not available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel

from realdoc_bench.shared.registry import Registry

AgentStatus = Literal["PROCESSED", "FAILED", "INVALID_JSON", "SCHEMA_VIOLATION"]


class AgentResult(BaseModel):
    value: dict[str, Any] | None
    status: AgentStatus
    latency_sec: float
    usage: dict[str, Any] = {}
    cost_estimate_usd: float | None = None
    retries: int = 0
    failure_reason: str | None = None


class ExtractionAgent(ABC):
    name: str = ""
    version: str = ""

    @abstractmethod
    def extract(
        self,
        markdown: str,
        schema: dict[str, Any],
        doc_id: str,
    ) -> AgentResult: ...


registry: Registry[type[ExtractionAgent]] = Registry("extraction agent")


def register_agent(name: str, *, version: str = "1.0.0") -> Any:
    def _wrap(cls: type[ExtractionAgent]) -> type[ExtractionAgent]:
        cls.name = name
        cls.version = version
        registry.register(name, cls)
        return cls

    return _wrap


def build(name: str, **kwargs: Any) -> ExtractionAgent:
    return registry.get(name)(**kwargs)
