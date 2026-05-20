#!/usr/bin/env python3
"""Migrate a question bank to canonically FLAT gold + response_format.

Any question whose gold_answer is 2-level nested ('OUTER: k=v, k=v; ...' or
brace style) is rewritten to flat compound keys ('OUTER.k=v; OUTER.k=v; ...').
response_format is flattened in parallel (per-key type hints preserved when
they line up, else regenerated as <value>). Clean flat / scalar questions are
left untouched.

After this, gold_to_struct decomposes every question into single-value items,
so the answer agent always produces a flat {item_id: value} map — no more
whole-string fallback.

Usage: flatten_bank.py <bank.json>   (writes in place, backs up to .pre_flatten.bak)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_eval import flatten_nested   # noqa: E402


def is_clean_flat(g: str) -> bool:
    """True if g is already a clean 'k=v; k=v' string (no nesting)."""
    if "=" not in g:
        return True                     # scalar — leave alone
    out: set[str] = set()
    for seg in (s.strip() for s in g.split(";") if s.strip()):
        if seg.count("=") != 1:
            return False
        k = seg.split("=", 1)[0].strip()
        if not k or ":" in k or k in out:
            return False
        out.add(k)
    return bool(out)


def serialize(d: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in d.items())


def main() -> int:
    bank_path = Path(sys.argv[1])
    bank = json.loads(bank_path.read_text())
    items = bank.get("items", [])

    n_migrated = 0
    n_fmt_regen = 0
    changed_sources: set[str] = set()
    for it in items:
        gold = str(it.get("gold_answer", "")).strip()
        flat_gold = flatten_nested(gold)
        if len(flat_gold) < 2:
            continue                    # genuine scalar / single — leave as-is
        if serialize(flat_gold) == gold:
            continue                    # already canonically flat — no-op
        # parallel-flatten the response_format; keep hints if keys line up
        flat_fmt = flatten_nested(it.get("response_format", ""))
        if set(flat_fmt) == set(flat_gold):
            new_fmt = "Return exactly: " + serialize(flat_fmt)
        else:
            new_fmt = ("Return exactly: "
                       + "; ".join(f"{k}=<value>" for k in flat_gold))
            n_fmt_regen += 1
        it["_gold_answer_nested_orig"] = gold          # keep the original
        it["gold_answer"] = serialize(flat_gold)
        it["response_format"] = new_fmt
        n_migrated += 1
        if it.get("source_file"):
            changed_sources.add(it["source_file"])

    backup = bank_path.with_suffix(".json.pre_flatten.bak")
    if not backup.exists():
        backup.write_text(json.dumps(bank, indent=2, ensure_ascii=False))
    bank_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")

    # list of changed docs — consumed by the re-eval step
    (bank_path.parent / "flatten_changed_docs.json").write_text(
        json.dumps(sorted(changed_sources), indent=2))

    print(f"migrated {n_migrated} nested questions -> flat compound keys "
          f"({n_fmt_regen} response_formats regenerated generically)")
    print(f"{len(changed_sources)} docs changed -> "
          f"{bank_path.parent / 'flatten_changed_docs.json'}")
    print(f"backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
