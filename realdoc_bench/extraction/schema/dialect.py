"""Coerce Extend-flavored JSON-Schema to a plain draft-07 dialect.

Removes / rewrites things that LLM tool-use APIs reject:

- ``type: "currency"`` → ``{"type": "number"}``
- ``type: "date"``     → ``{"type": "string", "format": "date"}``
- ``type: "signature"``→ ``{"type": "string"}``
- ``x-*`` keys are stripped (vendor extensions)
- Nullable expressed as ``{"type": ["string", "null"], ...}`` is preserved
  (draft-07 understands union-typed nullables natively).

Pattern lifted from ``stela_builder/extractors/anthropic_claude.py:_coerce_json_schema``
and re-implemented to keep realdoc-bench standalone.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_EXTEND_TYPE_REWRITES: dict[str, dict[str, Any]] = {
    "currency": {"type": "number"},
    "date": {"type": "string", "format": "date"},
    "signature": {"type": "string"},
}


def to_plain_jsonschema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied, draft-07-compatible version of ``schema``."""
    return _coerce(deepcopy(schema))


def _coerce(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k.startswith("x-"):
            continue
        out[k] = v

    t = out.get("type")
    if isinstance(t, str) and t in _EXTEND_TYPE_REWRITES:
        rewrite = _EXTEND_TYPE_REWRITES[t]
        for rk, rv in rewrite.items():
            out.setdefault(rk, rv)
        out["type"] = rewrite["type"]

    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {k: _coerce(v) for k, v in out["properties"].items()}
    if "items" in out:
        items = out["items"]
        if isinstance(items, list):
            out["items"] = [_coerce(i) for i in items]
        else:
            out["items"] = _coerce(items)
    for key in ("oneOf", "anyOf", "allOf"):
        if key in out and isinstance(out[key], list):
            out[key] = [_coerce(s) for s in out[key]]
    if "additionalProperties" in out:
        ap = out["additionalProperties"]
        if isinstance(ap, dict):
            out["additionalProperties"] = _coerce(ap)
    return out
