"""Value normalizers for the field-level scoring engine.

Each normalizer takes a raw value (anything an LLM emits) and produces a
canonical form. Comparison happens after normalization.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+")


def normalize_string(value: Any, *, ignore_punctuation: bool = True) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    out = _WHITESPACE_RE.sub(" ", value).strip().casefold()
    if ignore_punctuation:
        out = _PUNCT_RE.sub("", out).strip()
        out = _WHITESPACE_RE.sub(" ", out)
    return out


def normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    try:
        from dateutil import parser as dt_parser

        parsed = dt_parser.parse(value, default=None)
        return parsed.date().isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def normalize_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def normalize_integer(value: Any) -> int | None:
    n = normalize_number(value)
    if n is None:
        return None
    if abs(n - round(n)) < 1e-9:
        return int(round(n))
    return None


def normalize_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "y", "1"}:
            return True
        if lower in {"false", "no", "n", "0"}:
            return False
    return None
