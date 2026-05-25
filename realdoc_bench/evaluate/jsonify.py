"""Bank → typed-JSON-template prep.

A raw bank item has `response_format` ("Return exactly: amount=<number>") and
`gold_answer` ("amount=12800") as free-form strings. The scorer wants two
machine-readable fields instead:

- `template`  — typed JSON template string, e.g.  '{"amount": <number>}'
- `gold_dict` — the gold value parsed into the matching JSON types,
                 e.g. {"amount": 12800}

`jsonify_bank(items) -> items` mutates each item in place. The type
vocabulary and `_to_value` rules are intentionally stable across bank
revisions so the same item always jsonifies to an identical template.

Library-only; no CLI verb. The current bank ships pre-jsonified, so this is
used when a new bank rev lands.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

# ── type vocab ─────────────────────────────────────────────────────────────
_INT_LIKE = {"digits", "digit", "year", "qty", "quantity", "count", "integer", "int"}
_NUMBER_LIKE = {"amount", "money", "cost", "price", "fee", "balance", "total",
                "subtotal", "premium", "payment", "deposit", "percent",
                "percentage", "rate", "decimal", "float", "number"}
_DATE_LIKE = {"date", "mm/dd/yyyy", "mm-dd-yyyy", "yyyy-mm-dd", "yyyy/mm/dd"}
_STRING_LIKE = {"string", "str", "text", "value", "val", "name"}
_BOOL_LIKE = {"boolean", "bool"}
_BOOL_PAIRS = {("true", "false")}


def _placeholder_to_type_hint(inner: str) -> str:
    raw = inner.strip()
    options = [o.strip() for o in raw.split("|") if o.strip()]
    lower = [o.lower() for o in options]
    nullable = "null" in lower
    non_null = [o for o, lo in zip(options, lower) if lo != "null"]
    non_null_lo = [o.lower() for o in non_null]

    def with_null(t: str) -> str:
        return f"{t} | null" if nullable else t

    if len(non_null) == 1:
        tok = non_null_lo[0]
        if tok in _BOOL_LIKE:
            return with_null("boolean")
        if tok in _INT_LIKE:
            return with_null("integer")
        if tok in _NUMBER_LIKE:
            return with_null("number")
        if tok in _DATE_LIKE:
            return with_null("date string YYYY-MM-DD")
        if tok in _STRING_LIKE:
            return with_null("string")
        return with_null("string")
    if non_null:
        tup = tuple(non_null_lo)
        if tup in _BOOL_PAIRS or tuple(reversed(tup)) in _BOOL_PAIRS:
            return with_null("boolean")
        seen, uniq = set(), []
        for o in non_null:
            if o.lower() not in seen:
                seen.add(o.lower()); uniq.append(o)
        return with_null("one of: " + " | ".join(uniq))
    return with_null("string")


def render_template(key_types: dict[str, str]) -> str:
    lines = ["{"]
    visible = [(k, t) for k, t in key_types.items() if not k.startswith("__")]
    for i, (k, t) in enumerate(visible):
        comma = "," if i < len(visible) - 1 else ""
        lines.append(f'  {json.dumps(k)}: <{t}>{comma}')
    lines.append("}")
    return "\n".join(lines)


_DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y", "%m-%d-%Y",
                 "%m.%d.%Y", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%y", "%m-%d-%y",
                 "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
                 "%B %d %Y", "%b %d %Y"]


def _to_iso_date(text: str) -> str | None:
    t = re.sub(r"\bSept\b", "Sep", text.strip().rstrip(".,;:"))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _to_value(raw: Any, type_hint: str) -> Any:
    if raw is None:
        return None
    text = str(raw).strip()
    low = text.lower()
    base = type_hint.replace(" | null", "").strip()
    is_enum = base.startswith("one of:")
    enum_tokens: set[str] = set()
    if is_enum:
        enum_tokens = {t.strip().lower()
                       for t in base[len("one of:"):].split("|") if t.strip()}
    null_literals = {"null", "none", "", "—", "–"}
    if base in {"integer", "number", "boolean"} or base.startswith("date"):
        if not (is_enum and "-" in enum_tokens):
            null_literals.add("-")
    if low in null_literals:
        return None
    if low in {"n/a", "na"} and not is_enum and "string" not in type_hint:
        return None
    if base == "integer":
        cleaned = re.sub(r"[,$\s]", "", text).lstrip("+").rstrip("%")
        try:
            return int(cleaned)
        except ValueError:
            try:
                return float(cleaned)
            except ValueError:
                return text
    if base == "number":
        cleaned = re.sub(r"[,$\s]", "", text).lstrip("+").rstrip("%")
        try:
            return float(cleaned) if "." in cleaned else int(cleaned)
        except ValueError:
            return text
    if base.startswith("date"):
        return _to_iso_date(text) or text
    if base == "boolean":
        if low in {"y", "yes", "true", "checked", "1"}:
            return True
        if low in {"n", "no", "false", "unchecked", "0"}:
            return False
        return text
    if base.startswith("one of:"):
        for tok in [t.strip() for t in base[len("one of:"):].split("|") if t.strip()]:
            if tok.lower() == low:
                return tok
        return text
    return re.sub(r"\s+", " ", text).strip()


def widen_types_from_gold(key_types: dict[str, str], gold: dict[str, Any]) -> None:
    """Widen a key's type when the gold value doesn't fit the declared type so
    the template never forces the model into a format the gold can't match."""
    for k, v in list(gold.items()):
        ht = key_types.get(k, "")
        base = ht.replace(" | null", "").strip()
        nullable = "null" in ht
        if base in {"integer", "number"} and isinstance(v, str) and v.strip():
            iso = _to_iso_date(v)
            if iso:
                key_types[k] = "date string YYYY-MM-DD" + (" | null" if nullable else "")
                gold[k] = iso
            else:
                key_types[k] = "string | null" if nullable else "string"
        elif base.startswith("date") and isinstance(v, str) and \
                not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            key_types[k] = "string | null" if nullable else "string"
        elif base.startswith("one of:") and isinstance(v, str):
            toks = {t.strip().lower()
                    for t in base[len("one of:"):].split("|") if t.strip()}
            if v.lower() not in toks:
                key_types[k] = "string | null" if nullable else "string"


_DATEISH = re.compile(r"m+[/\-. ]d+[/\-. ]y+|y{4}|month\s+y{2,4}|mm/yy", re.I)
_NUMWORD = re.compile(
    r"\b(number|amount|dollar|currency|percentage|percent|numeric|decimal)\b", re.I)


def our_rf_to_placeholder(rf: str) -> str:
    """Map a freeform response_format string to a <placeholder> body."""
    s = (rf or "").strip()
    s = re.sub(r"^\((.*)\)$", r"\1", s).strip()
    inner = re.findall(r"<([^<>]+)>", s)
    if inner:
        s = inner[0] if len(inner) == 1 else " | ".join(inner)
    low = s.lower()
    if "|" in s:
        return s
    if "/" in s and len(s) <= 60 and not _DATEISH.search(s):
        opts = [o.strip() for o in s.split("/") if o.strip()]
        if 2 <= len(opts) <= 8:
            return " | ".join(opts)
    if low in _DATE_LIKE or _DATEISH.search(low) or low in ("date", "currency date"):
        return "date"
    if re.search(r"x{2,}|#{2,}|sequence of digits|\(xxx\)", low):
        return "text"
    if _NUMWORD.search(low) or low in ("integer", "a number", "a two-digit number"):
        return "integer" if low in ("integer", "a two-digit number") else "number"
    return "text"


def jsonify_bank(items: list[dict]) -> list[dict]:
    """Mutates each item to add `template` + `gold_dict`, in place. Items that
    already have both are left alone (idempotent)."""
    for it in items:
        if "template" in it and "gold_dict" in it:
            continue
        body = our_rf_to_placeholder(it.get("response_format", ""))
        hint = _placeholder_to_type_hint(body)
        key_types = {"answer": hint}
        gold_dict = {"answer": _to_value(it.get("gold_answer", ""), hint)}
        widen_types_from_gold(key_types, gold_dict)
        it["template"] = render_template(key_types)
        it["gold_dict"] = gold_dict
    return items
