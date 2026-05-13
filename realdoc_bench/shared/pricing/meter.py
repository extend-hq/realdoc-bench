"""Cost metering for parse providers and LLM agents.

Rates live in ``catalog.yaml`` next to this file. ``catalog_sha()`` is included in
every run's metadata so old runs remain interpretable when prices drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from realdoc_bench.shared.io.cache import sha256_bytes

_CATALOG_PATH = Path(__file__).with_name("catalog.yaml")


@dataclass(frozen=True)
class PageRate:
    usd: float


@dataclass(frozen=True)
class TokenRate:
    input_usd_per_1k: float
    output_usd_per_1k: float


Rate = PageRate | TokenRate


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, Rate], str]:
    raw_bytes = _CATALOG_PATH.read_bytes()
    raw = yaml.safe_load(raw_bytes) or {}
    rates: dict[str, Rate] = {}
    for name, entry in raw.items():
        unit = entry.get("unit")
        if unit == "page":
            rates[name] = PageRate(usd=float(entry["usd"]))
        elif unit == "tokens":
            rates[name] = TokenRate(
                input_usd_per_1k=float(entry["input_usd_per_1k"]),
                output_usd_per_1k=float(entry["output_usd_per_1k"]),
            )
        else:
            raise ValueError(f"unknown pricing unit {unit!r} for {name}")
    return rates, sha256_bytes(raw_bytes)[:12]


def catalog_sha() -> str:
    return _load()[1]


def get_rate(name: str) -> Rate | None:
    rates, _ = _load()
    return rates.get(name)


def parse_cost(provider: str, *, pages: int | None) -> float | None:
    rate = get_rate(provider)
    if rate is None or pages is None:
        return None
    if isinstance(rate, PageRate):
        return rate.usd * pages
    return None


def agent_cost(agent: str, *, input_tokens: int, output_tokens: int) -> float | None:
    rate = get_rate(agent)
    if rate is None:
        return None
    if isinstance(rate, TokenRate):
        return (
            rate.input_usd_per_1k * input_tokens / 1000
            + rate.output_usd_per_1k * output_tokens / 1000
        )
    return None
