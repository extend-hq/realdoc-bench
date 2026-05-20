#!/usr/bin/env python3
"""Rewrite every curated question into a clean, self-contained question and
re-verify its gold answer against the page image (vision LLM).

The curated bank (bank_curated.json) is the discriminative 4-5/doc subset,
but the questions are atomization leftovers — often clumsy/vague. This pass
gives each one a natural, anchored, single-value phrasing and re-reads the
gold straight off the page so the benchmark answer is image-verified.

  --sample N   rewrite N examples, print them, no write
  (default)    rewrite the whole curated bank -> qa/bank_r1.json
               (backup qa/bank_r1.json.pre_curate.bak)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_gen_qa import gemini_generate, encoded_image, load_env_file  # noqa: E402

REWRITE_SYSTEM = (
    "You are shown a DOCUMENT PAGE IMAGE and ONE extraction question with its "
    "current machine-generated answer. The question came from an automatic "
    "'atomization' step and is often clumsy, vague, or awkwardly phrased.\n\n"
    "Do TWO things:\n\n"
    "1. REWRITE the question into ONE clean, natural, self-contained question:\n"
    "   - Name the printed section / table / row / label anchor visible on the "
    "page, so the question is answerable on its own.\n"
    "   - Ask for EXACTLY ONE value.\n"
    "   - Only ask for values PRINTED on the page — never anything computed, "
    "summed, or derived.\n"
    "   - End with the answer format in parentheses.\n"
    "   - Real, natural phrasing — not 'extract sub-field X'.\n\n"
    "2. RE-VERIFY the answer against the page image. Read the actual value / "
    "checkbox state / mark off the image. If the current answer is correct, "
    "echo it. If it is wrong, return the correct value. Use the value VERBATIM "
    "as printed — do not normalize, round, or reformat numbers or dates.\n\n"
    "Keep the answer TYPE consistent:\n"
    "  - checklist_yesno  -> answer exactly one of: Yes, No, both, null\n"
    "  - multiple_choice  -> answer exactly one of the listed options, or null\n"
    "  - otherwise        -> the single printed value, or null if blank/absent\n\n"
    "The 'response_format' field is a SHORT TEMPLATE for the SHAPE of the "
    "answer — for example '<Yes|No|both|null>', '<X|null>', 'text', "
    "'<dollar amount>', 'mm/dd/yyyy', '<number>', '<percentage>', or a "
    "pipe-list of the literal options for a multiple-choice question. It is "
    "NEVER the words 'checklist_yesno', 'multiple_choice', "
    "'hard_structural_exact' — those are answer_type labels, not formats.\n\n"
    "Return exactly: {\"question\": \"...\", \"response_format\": \"...\", "
    "\"answer\": \"...\", \"gold_status\": \"confirmed\" | \"corrected\"}"
)

_ATYPE_NAMES = {
    "hard_structural_exact", "chart_figure_exact", "complex_checkmark_exact",
    "checklist_yesno", "multiple_choice", "atomic_exact", "atomic",
}


def rewrite_one(it: dict, run: Path) -> dict:
    """Return the rewritten item (falls back to the original on any failure)."""
    out = dict(it)
    img = run / "images" / f"{it.get('source_file','')}.png"
    if not img.exists():
        out["_rewrite"] = "no_image"
        return out
    atype = it.get("answer_type", "atomic_exact")
    prompt = (
        f"answer_type: {atype}\n"
        f"CURRENT QUESTION: {it.get('question','')}\n"
        f"CURRENT ANSWER: {it.get('gold_answer','')}\n"
        f"CURRENT FORMAT: {it.get('response_format','')}"
    )
    try:
        r = gemini_generate(
            [{"text": prompt},
             {"inline_data": {"mime_type": "image/png", "data": encoded_image(img)}}],
            system=REWRITE_SYSTEM, temperature=0.0)
    except Exception as e:
        out["_rewrite"] = f"error:{type(e).__name__}"
        return out
    if not isinstance(r, dict) or not r.get("question"):
        out["_rewrite"] = "bad_payload"
        return out

    out["_pre_rewrite_question"] = it.get("question", "")
    out["_pre_rewrite_gold"] = it.get("gold_answer", "")
    out["question"] = str(r["question"]).strip()

    # response_format: MC / yes-no keep their fixed templates; otherwise take
    # the LLM's if it is a sane shape-template, else keep the original.
    if atype == "multiple_choice":
        pass                                    # keep original option list
    elif atype == "checklist_yesno":
        out["response_format"] = "<Yes|No|both|null>"
    else:
        rf = str(r.get("response_format", "")).strip()
        if rf and rf.lower() not in _ATYPE_NAMES:
            out["response_format"] = rf

    new_gold = r.get("answer")
    status = r.get("gold_status", "confirmed")
    changed = new_gold is not None and str(new_gold) != str(it.get("gold_answer", ""))
    if changed:
        out["gold_answer"] = new_gold
    out["_gold_status"] = status
    out["_rewrite"] = "gold_corrected" if (changed and status == "corrected") else "ok"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="v500")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run = REPO / "runs" / args.run
    load_env_file(REPO / ".env")

    qa = run / "qa"
    bank = json.loads((qa / "bank_curated.json").read_text())
    items = bank["items"]

    if args.sample:
        for it in items[: args.sample]:
            res = rewrite_one(it, run)
            print("=" * 72)
            print(f"[{res.get('_rewrite')}]  {it['question_id']}")
            print(f"  OLD Q : {it.get('question','')[:200]}")
            print(f"  NEW Q : {res.get('question','')[:200]}")
            print(f"  format: {res.get('response_format')}")
            print(f"  gold  : {it.get('gold_answer')!r}  ->  {res.get('gold_answer')!r}")
        return 0

    print(f"rewriting {len(items)} curated questions...", flush=True)
    t0 = time.time()
    done: list[dict] = []
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(rewrite_one, it, run) for it in items]
        for fut in as_completed(futs):
            done.append(fut.result())
            n += 1
            if n % 200 == 0 or n == len(items):
                print(f"  [{n}/{len(items)}] ({(time.time()-t0)/60:.1f}min)", flush=True)

    from collections import Counter
    stats = Counter(d.get("_rewrite", "ok") for d in done)
    backup = qa / "bank_r1.json.pre_curate.bak"
    if not backup.exists():
        backup.write_text((qa / "bank_r1.json").read_text())
    bank["items"] = done
    bank["curated"] = True
    bank["rewritten"] = True
    (qa / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")

    print(f"\nrewrote {len(done)} questions ({(time.time()-t0)/60:.1f}min)")
    for k, v in stats.most_common():
        print(f"  {k:>16}: {v}")
    print(f"backup: {backup}")
    print(f"wrote:  {qa/'bank_r1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
