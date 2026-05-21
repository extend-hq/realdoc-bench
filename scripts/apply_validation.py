#!/usr/bin/env python3
"""Apply human validation decisions to the v500 bank.

Reads qa/validation_decisions.json (the validation.html localStorage export):
  - question reject   -> drop the question
  - question keep     -> keep
  - question gold     -> overwrite gold_answer with the human value
  - doc reject        -> drop every question of that doc

Decisions on questions rewritten AFTER the user validated them
(``_pre_nullfix_question`` present) are STALE and skipped; decisions whose
qid is no longer in the bank are ORPHANED and skipped. Both are reported.

Writes the bank in place (backup .pre_validation.bak) + the affected docs to
qa/validation_changed_docs.json for a targeted re-eval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    qa = run / "qa"
    bank = json.loads((qa / "bank_r1.json").read_text())
    items = bank["items"]
    dec = json.loads((qa / "validation_decisions.json").read_text())
    qdec = dec.get("q", {})
    ddec = dec.get("doc", {})

    by_id = {it["question_id"]: it for it in items}
    reject_docs = {d for d, v in ddec.items() if (v or {}).get("v") == "reject"}

    new_items: list[dict] = []
    changed: set[str] = set()
    n_doc_drop = n_reject = n_gold = n_stale = 0

    for it in items:
        qid = it["question_id"]
        sf = it.get("source_file")
        if sf in reject_docs:                       # whole doc rejected
            n_doc_drop += 1
            changed.add(sf)
            continue
        d = qdec.get(qid)
        if not d:                                   # no decision -> keep
            new_items.append(it)
            continue
        if "_pre_nullfix_question" in it:           # rewritten after validation -> stale
            n_stale += 1
            new_items.append(it)
            continue
        if d.get("v") == "reject":
            n_reject += 1
            changed.add(sf)
            continue
        # keep — apply a human gold correction if present
        if "gold" in d and str(d["gold"]) != str(it.get("gold_answer", "")):
            it = dict(it)
            it["_pre_validation_gold"] = it.get("gold_answer", "")
            it["gold_answer"] = d["gold"]
            n_gold += 1
            changed.add(sf)
        new_items.append(it)

    n_orphan = sum(1 for k in qdec if k not in by_id)

    backup = qa / "bank_r1.json.pre_validation.bak"
    if not backup.exists():
        backup.write_text((qa / "bank_r1.json").read_text())
    bank["items"] = new_items
    (qa / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    (qa / "validation_changed_docs.json").write_text(
        json.dumps(sorted(c for c in changed if c), indent=2))

    print(f"applied human validation decisions:")
    print(f"  questions dropped (reject):      {n_reject}")
    print(f"  questions dropped (doc reject):  {n_doc_drop}  "
          f"(docs: {', '.join(sorted(reject_docs))})")
    print(f"  gold corrections applied:        {n_gold}")
    print(f"  skipped — stale (rewritten):     {n_stale}")
    print(f"  skipped — orphaned (qid gone):   {n_orphan}")
    print(f"  bank: {len(items)} -> {len(new_items)} questions")
    print(f"  affected docs: {len(changed)}  -> validation_changed_docs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
