#!/usr/bin/env python3
"""Atomize a question bank: every multi-part question becomes N self-contained
single-value questions — one per sub-field — each rewritten to name its
section anchor, ask for exactly one value, and state the answer format inline.

After this the bank has only atomic questions; the eval asks each one directly
(no flatten, no 'extract only sub-field X' bolt-on).

  --sample N   print N atomized examples and exit (no write)
  (default)    atomize the whole bank in place (backup .pre_atomize.bak)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_eval import gold_to_struct, format_to_struct          # noqa: E402
from pilot_50_gen_qa import gemini_generate, load_env_file          # noqa: E402

ATOMIZE_SYSTEM = (
    "You split ONE multi-part document-extraction question into SEPARATE "
    "self-contained single-value questions — exactly one per listed sub-field.\n\n"
    "Each rewritten question MUST:\n"
    "  - Name the SAME section / table / block anchor the parent uses (a "
    "printed label on the page), so it is answerable on its own.\n"
    "  - Ask for EXACTLY ONE value — the named sub-field — and nothing else.\n"
    "  - State the answer format in parentheses at the end "
    "(e.g. '(checked or unchecked)', '(a dollar amount)', '(mm/dd/yyyy)').\n"
    "  - Read naturally — not 'extract sub-field X', but a real question.\n\n"
    "Example — parent: 'In the INDICATE SECTIONS ATTACHED grid (three checkbox "
    "columns), provide the checkbox states for EQUIPMENT FLOATER, GLASS AND "
    "SIGN, COMMERCIAL LIABILITY' → for sub-field 'COMMERCIAL LIABILITY': "
    "'In the INDICATE SECTIONS ATTACHED grid (three checkbox columns), what is "
    "the status of the COMMERCIAL LIABILITY checkbox (checked or unchecked)?'\n\n"
    "Return exactly: {\"atomic\": {\"<key>\": {\"question\": \"...\", "
    "\"response_format\": \"...\"}, ...}} covering EXACTLY the given keys, same "
    "spelling. response_format is the single value form, e.g. "
    "'<checked|unchecked>' or '<dollar amount>'."
)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:40] or "k"


def atomize_question(it: dict) -> list[dict]:
    """Return the list of atomic question items for one bank question."""
    gs = gold_to_struct(it.get("gold_answer"))
    keys = list(gs.keys())
    base_caps = it.get("capabilities") or []
    src = it.get("source_file")
    qid = it["question_id"]

    if keys == ["answer"]:                     # already a single-value question
        a = dict(it)
        a["parent_id"] = qid
        a["answer_key"] = "answer"
        return [a]

    fs = format_to_struct(it.get("response_format", ""))
    prompt = (
        f"PARENT QUESTION:\n{it.get('question','')}\n\n"
        f"SUB-FIELDS (one atomic question each): {json.dumps(keys, ensure_ascii=False)}\n\n"
        f"per-field answer formats: {json.dumps(fs, ensure_ascii=False)}"
    )
    try:
        payload = gemini_generate([{"text": prompt}], system=ATOMIZE_SYSTEM,
                                  temperature=0.0)
        atomic = payload.get("atomic", {}) if isinstance(payload, dict) else {}
    except Exception:
        atomic = {}

    out = []
    for k in keys:
        spec = atomic.get(k) if isinstance(atomic.get(k), dict) else {}
        q_text = spec.get("question") or (
            f"{it.get('question','')}  (Answer only for: {k})")
        rf = spec.get("response_format") or fs.get(k) or "<value>"
        out.append({
            "question_id": f"{qid}__{slug(k)}",
            "parent_id": qid,
            "answer_key": k,
            "source_file": src,
            "question": q_text,
            "response_format": rf,
            "gold_answer": gs[k],
            "capabilities": base_caps,
            "answer_type": it.get("answer_type", "atomic_exact"),
            "difficulty": it.get("difficulty", ""),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="v500")
    ap.add_argument("--sample", type=int, default=0,
                    help="print N atomized examples and exit (no write)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run = REPO / "runs" / args.run
    load_env_file(REPO / ".env")

    bank = json.loads((run / "qa" / "bank_r1.json").read_text())
    items = bank["items"]
    multi = [it for it in items if len(gold_to_struct(it.get("gold_answer"))) >= 2]

    if args.sample:
        for it in multi[: args.sample]:
            print("=" * 70)
            print(f"PARENT  {it['question_id']}")
            print(f"  question: {it['question']}")
            print(f"  gold    : {it['gold_answer']}")
            print("  -> ATOMIC:")
            for a in atomize_question(it):
                print(f"    [{a['question_id']}]")
                print(f"      Q: {a['question']}")
                print(f"      format: {a['response_format']}   gold: {a['gold_answer']}")
        return 0

    print(f"atomizing {len(items)} questions ({len(multi)} multi-part)...", flush=True)
    t0 = time.time()
    atomic_items: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(atomize_question, it): it for it in items}
        for fut in as_completed(futs):
            atomic_items.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(items):
                print(f"  [{done}/{len(items)}] -> {len(atomic_items)} atomic "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)

    backup = run / "qa" / "bank_r1.json.pre_atomize.bak"
    if not backup.exists():
        backup.write_text(json.dumps(bank, indent=2, ensure_ascii=False))
    bank["items"] = atomic_items
    bank["atomized"] = True
    (run / "qa" / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")

    # atomize rewrites every question_id — the per-(doc,parser) eval cache is
    # keyed only by doc+parser, so it would NOT auto-invalidate. Wipe it so the
    # next eval genuinely re-runs against the atomic questions.
    cache = run / "eval" / "answers_flat"
    if cache.exists():
        n_cache = len(list(cache.glob("*.json")))
        shutil.rmtree(cache)
        print(f"wiped {n_cache} stale eval-cache files (atomize invalidated all)")

    print(f"\n{len(items)} -> {len(atomic_items)} atomic questions")
    print(f"backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
