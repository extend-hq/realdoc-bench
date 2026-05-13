"""Per-dataset / per-provider canonicalization hooks.

A manifest can declare a small set of rewrites that fire on actual values
before the scoring engine sees them — useful for benign provider quirks like
list-wrapping single results. Visible in run metadata; never silent.

Inspired by ``research/benchmarks/reducto_deep_extract/scoring_rules.py`` but
reimplemented standalone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Rewriter = Callable[[Any], Any]

_REWRITES: dict[str, Rewriter] = {}


def register(name: str, fn: Rewriter) -> Rewriter:
    if name in _REWRITES:
        raise ValueError(f"rewriter {name!r} already registered")
    _REWRITES[name] = fn
    return fn


def apply_rules(actual: Any, rule_names: list[str]) -> Any:
    out = actual
    for rule in rule_names:
        fn = _REWRITES.get(rule)
        if fn is None:
            raise KeyError(f"unknown scoring rule: {rule}")
        out = fn(out)
    return out


# ── Built-in rewrites ───────────────────────────────────────────────────────


def _unwrap_item(value: Any) -> Any:
    """Some providers wrap a single object in a single-element list. Unwrap it."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return value


register("unwrap_item", _unwrap_item)


def _zero_to_none(value: Any) -> Any:
    """Treat literal 0 as absent (some pipelines emit 0 instead of null)."""

    def _walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_walk(x) for x in v]
        if v == 0:
            return None
        return v

    return _walk(value)


register("zero_to_none", _zero_to_none)
