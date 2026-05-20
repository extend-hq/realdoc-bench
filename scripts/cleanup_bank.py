#!/usr/bin/env python3
"""Drop degenerate questions from the bank:
  - questions on blank-template pages (blank_pages.json) — nothing to extract
  - questions with genuinely corrupt gold ('=' artifacts, comma-split numbers)
Salvages flat gold wrapped in a stray ["..."] / ['...'] bracket.
Everything removed is archived to qa/dropped_blank_corrupt.json (recoverable).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def is_corrupt(g: str) -> bool:
    return bool("'='" in g or '"="' in g
                or re.search(r"\b\d{3}/[A-Z]", g)        # comma-split number key
                or re.search(r"PER \$\d;", g)            # truncated number
                or re.search(r"=\s*'\(", g))             # stray-quote value


def unwrap(g: str) -> str:
    """Strip a stray ["...flat k=v..."] / ['...'] wrapper."""
    s = g.strip()
    m = re.match(r"""^\[\s*["'](.+)["']\s*\]$""", s, re.DOTALL)
    if m and "=" in m.group(1):
        return m.group(1).strip()
    return g


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    qa = run / "qa"
    bank = json.loads((qa / "bank_r1.json").read_text())
    blank = json.loads((qa / "blank_pages.json").read_text()) \
        if (qa / "blank_pages.json").exists() else []
    blank_stems = {Path(b["dest_path"]).stem for b in blank}

    kept, dropped = [], []
    n_salvaged = 0
    for it in bank["items"]:
        if it.get("source_file") in blank_stems:
            dropped.append({**it, "_drop_reason": "blank_page"})
            continue
        g = str(it.get("gold_answer", ""))
        if is_corrupt(g):
            dropped.append({**it, "_drop_reason": "corrupt_gold"})
            continue
        u = unwrap(g)
        if u != g:
            it["gold_answer"] = u
            n_salvaged += 1
        kept.append(it)

    backup = qa / "bank_r1.json.pre_cleanup.bak"
    if not backup.exists():
        backup.write_text(json.dumps(bank, indent=2, ensure_ascii=False))
    bank["items"] = kept
    (qa / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    (qa / "dropped_blank_corrupt.json").write_text(
        json.dumps(dropped, indent=2, ensure_ascii=False))

    rc = Counter(d["_drop_reason"] for d in dropped)
    print(f"dropped {len(dropped)} questions  ({dict(rc)})")
    print(f"salvaged {n_salvaged} bracket-wrapped gold strings")
    print(f"bank: kept {len(kept)} questions  (backup: {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
