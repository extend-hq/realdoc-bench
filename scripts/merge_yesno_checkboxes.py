#!/usr/bin/env python3
"""Merge per-checkbox YES/NO checklist pairs into single Yes/No/null questions.

A checklist row gets atomized into TWO questions — one about the YES box,
one about the NO box — in any of several phrasings:
  'is the YES checkbox checked (checked or unchecked)?'
  'is the NO checkbox selected (selected or unselected)?'
  'what is the selection state for the NO checkbox ... (selected/unselected)?'

This pairs the two halves PHRASING-AGNOSTICALLY (by the row's normalized
text with the YES/NO token stripped) and emits ONE question:
  '<row> — what is the marked answer (Yes, No, or null)?'
answered Yes / No / both / null  ('both' = data-quality flag).

Writes the merged bank in place (backup .pre_yesno.bak) + the affected docs
to qa/yesno_changed_docs.json for a targeted re-eval.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

POS = {"checked", "selected", "x", "✓", "marked", "yes", "true", "on"}
SIDE_RE = re.compile(r"\bthe ['\"]?(YES|NO)['\"]? checkbox", re.I)
FMT_RE = re.compile(r"\s*\([^)]*\)\s*\??\s*$")
TAIL_RE = re.compile(r"\s*(?:,?\s*or )?(?:un)?(?:selected|checked|marked)\s*\??\s*$", re.I)

# the clause that names the specific YES/NO box — stripped from the merged Q
CLAUSE_RES = [
    re.compile(r",?\s*what is the (?:selection |checkbox )?state (?:of|for) "
               r"(?:the )?['\"]?(?:YES|NO)['\"]? checkbox", re.I),
    re.compile(r",?\s*what is the status of (?:the )?['\"]?(?:YES|NO)['\"]? checkbox", re.I),
    re.compile(r",?\s*is (?:the )?['\"]?(?:YES|NO)['\"]? checkbox(?: (?:selected|checked))?", re.I),
    re.compile(r",?\s*(?:the )?['\"]?(?:YES|NO)['\"]? checkbox", re.I),
]


def side(it: dict) -> str | None:
    """'yes' / 'no' if this atomic question is about a YES/NO checkbox."""
    if it.get("answer_type") == "checklist_yesno":
        return None
    m = SIDE_RE.search(it.get("question", ""))
    return m.group(1).lower() if m else None


def row_norm(it: dict) -> str:
    """Section + row identity with the YES/NO token removed — so the two
    halves of one row collapse to the same key regardless of phrasing."""
    q = re.sub(r"['\"]?\b(?:YES|NO)\b['\"]?", "", it.get("question", ""), flags=re.I)
    q = FMT_RE.sub("", q)
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def merged_gold(yes_g, no_g) -> str:
    yc = str(yes_g).strip().lower() in POS
    nc = str(no_g).strip().lower() in POS
    if yc and nc:
        return "both"
    if yc:
        return "Yes"
    if nc:
        return "No"
    return "null"


def merged_question(text: str) -> str:
    t = text
    for rx in CLAUSE_RES:
        new = rx.sub(" ", t, count=1)
        if new != t:
            t = new
            break
    t = FMT_RE.sub("", t)
    t = TAIL_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,;:])", r"\1", t).strip(" ,;:.")
    if not t:
        t = "For this checklist row"
    return t + " — what is the marked answer (Yes, No, or null)?"


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    qa = run / "qa"
    bank = json.loads((qa / "bank_r1.json").read_text())
    items = bank["items"]

    # pair the two halves by (doc, normalized row text)
    rows: dict[tuple, dict] = defaultdict(dict)
    other = []
    for it in items:
        s = side(it)
        if s in ("yes", "no"):
            rows[(it.get("source_file"), row_norm(it))][s] = it
        else:
            other.append(it)

    merged, unpaired, changed = [], [], set()
    n_both = 0
    for (_, _), pair in rows.items():
        if "yes" in pair and "no" in pair:
            y, n = pair["yes"], pair["no"]
            g = merged_gold(y.get("gold_answer"), n.get("gold_answer"))
            if g == "both":
                n_both += 1
            mq = {**y}
            mq["question"] = merged_question(y.get("question", ""))
            mq["response_format"] = "<Yes|No|both|null>"
            mq["gold_answer"] = g
            mq["answer_type"] = "checklist_yesno"
            mq["_merged_from"] = [y["question_id"], n["question_id"]]
            merged.append(mq)
            changed.add(y.get("source_file"))
        else:
            unpaired.extend(pair.values())

    bank["items"] = other + merged + unpaired
    backup = qa / "bank_r1.json.pre_yesno.bak"
    if not backup.exists():
        backup.write_text(json.dumps(
            {"items": items, **{k: v for k, v in bank.items() if k != "items"}},
            indent=2, ensure_ascii=False))
    (qa / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    (qa / "yesno_changed_docs.json").write_text(
        json.dumps(sorted(c for c in changed if c), indent=2))

    print(f"merged {len(merged)} Yes/No checklist rows "
          f"(from {len(merged) * 2} per-checkbox questions)")
    print(f"  unpaired left as-is: {len(unpaired)}")
    print(f"  rows with BOTH boxes marked (gold flag): {n_both}")
    print(f"  affected docs: {len(changed)}")
    print(f"  bank: {len(items)} -> {len(bank['items'])} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
