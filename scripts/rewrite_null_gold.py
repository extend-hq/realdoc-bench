#!/usr/bin/env python3
"""Re-point null-gold questions at an actual FILLED value.

~180 curated questions have gold == null/blank — atomization aimed them at
empty fields / rows / unchecked boxes, which makes them weak benchmark
questions. Each is rewritten (vision LLM) into a question about a real,
printed, filled value on the SAME page, with the gold read off the image.

A question that genuinely can't be salvaged (truly blank page) is dropped.

  --sample N   rewrite N examples, print them, no write
  (default)    rewrite all null-gold questions in place -> qa/bank_r1.json
               (backup .pre_nullfix.bak) + qa/nullfix_changed_docs.json
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

NULL_TOKENS = {"null", "none", "", "blank", "n/a", "na", "<blank>", "-"}


def is_null(g) -> bool:
    return str(g).strip().lower() in NULL_TOKENS


# answer_type / category labels the model sometimes wrongly emits as a "format"
_ATYPE_NAMES = {
    "hard_structural_exact", "chart_figure_exact", "complex_checkmark_exact",
    "checklist_yesno", "multiple_choice", "atomic_exact", "atomic",
    "short_text", "long_text", "text_exact", "value",
}


REWRITE_NULL_SYSTEM = (
    "You are shown a DOCUMENT PAGE IMAGE and ONE extraction question whose "
    "answer is null — it targets an EMPTY field, blank row, or unchecked "
    "checkbox, which makes it a weak benchmark question.\n\n"
    "Rewrite it into a question that targets an ACTUAL FILLED value on the "
    "page — a printed value, a checked box, a table cell that has content.\n\n"
    "The new question MUST:\n"
    "  - Prefer the SAME section / table / region as the original; only if "
    "that whole region is empty, pick another clearly-filled area of the page.\n"
    "  - Name the printed section / row / label anchor so it is self-contained.\n"
    "  - Ask for EXACTLY ONE value that is genuinely PRESENT — never blank.\n"
    "  - Only ask for values printed on the page — nothing computed or derived.\n"
    "  - End with the answer format in parentheses.\n\n"
    "Read the answer straight off the image, VERBATIM — keep commas, decimal "
    "points, trailing zeros, currency symbols and dates exactly as printed.\n\n"
    "Keep the answer consistent with answer_type:\n"
    "  checklist_yesno -> pick a checklist row that IS marked; answer Yes / "
    "No / both\n"
    "  multiple_choice -> pick an option group where one option IS selected; "
    "answer that option's label\n"
    "  otherwise       -> the single printed value\n\n"
    "If the ENTIRE page is a blank template with nothing filled in, return "
    '{"question": null}.\n\n'
    'Return exactly: {"question": "...", "response_format": "...", '
    '"answer": "<the real filled value>"}'
)


def rewrite_one(it: dict, run: Path) -> dict:
    out = dict(it)
    img = run / "images" / f"{it.get('source_file','')}.png"
    if not img.exists():
        out["_rewrite"] = "no_image"
        return out
    prompt = (f"answer_type: {it.get('answer_type','atomic_exact')}\n"
              f"CURRENT (null-gold) QUESTION: {it.get('question','')}")
    r = None
    for attempt in range(3):
        try:
            r = gemini_generate(
                [{"text": prompt},
                 {"inline_data": {"mime_type": "image/png",
                                  "data": encoded_image(img)}}],
                system=REWRITE_NULL_SYSTEM, temperature=0.0)
            break
        except Exception as e:
            if attempt == 2:
                out["_rewrite"] = f"error:{type(e).__name__}"
                return out
            time.sleep(2)
    if not isinstance(r, dict) or not r.get("question"):
        out["_rewrite"] = "page_blank"          # whole page is an empty template
        return out
    ans = r.get("answer")
    if ans is None or is_null(ans):
        out["_rewrite"] = "still_null"
        return out
    out["_pre_nullfix_question"] = it.get("question", "")
    out["_pre_nullfix_gold"] = it.get("gold_answer", "")
    out["question"] = str(r["question"]).strip()
    rf = str(r.get("response_format", "")).strip()
    if not rf or rf.lower() in _ATYPE_NAMES:        # model emitted a category, not a format
        orig = str(it.get("response_format", "")).strip()
        rf = orig if (orig and orig.lower() not in _ATYPE_NAMES) else "<value>"
    out["response_format"] = rf
    out["gold_answer"] = ans
    out["_rewrite"] = "ok"
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
    bank = json.loads((qa / "bank_r1.json").read_text())
    items = bank["items"]
    null_items = [it for it in items if is_null(it.get("gold_answer"))]
    print(f"{len(null_items)} null-gold questions of {len(items)}", flush=True)

    if args.sample:
        for it in null_items[: args.sample]:
            res = rewrite_one(it, run)
            print("=" * 72)
            print(f"[{res.get('_rewrite')}]  {it['question_id']}")
            print(f"  OLD: {it.get('question','')[:200]}")
            print(f"       gold: {it.get('gold_answer')!r}")
            print(f"  NEW: {res.get('question','')[:200]}")
            print(f"       gold: {res.get('gold_answer')!r}   fmt: {res.get('response_format')}")
        return 0

    t0 = time.time()
    rewritten: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(rewrite_one, it, run) for it in null_items]
        for fut in as_completed(futs):
            res = fut.result()
            rewritten[res["question_id"]] = res
            done += 1
            if done % 40 == 0 or done == len(null_items):
                print(f"  [{done}/{len(null_items)}] ({(time.time()-t0)/60:.1f}min)",
                      flush=True)

    from collections import Counter
    stats = Counter(r.get("_rewrite") for r in rewritten.values())

    # rebuild bank: ok -> rewritten; unsalvageable null-gold -> dropped
    new_items, dropped = [], []
    changed = set()
    for it in items:
        qid = it["question_id"]
        if qid in rewritten:
            res = rewritten[qid]
            changed.add(it.get("source_file"))
            if res.get("_rewrite") == "ok":
                new_items.append(res)
            else:
                dropped.append(qid)
        else:
            new_items.append(it)

    backup = qa / "bank_r1.json.pre_nullfix.bak"
    if not backup.exists():
        backup.write_text((qa / "bank_r1.json").read_text())
    bank["items"] = new_items
    (qa / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    (qa / "nullfix_changed_docs.json").write_text(
        json.dumps(sorted(c for c in changed if c), indent=2))

    print(f"\nnull-gold rewritten ({(time.time()-t0)/60:.1f}min):")
    for k, v in stats.most_common():
        print(f"  {k:>14}: {v}")
    print(f"dropped (unsalvageable): {len(dropped)}")
    print(f"bank: {len(items)} -> {len(new_items)} questions")
    print(f"affected docs: {len(changed)}  (-> nullfix_changed_docs.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
