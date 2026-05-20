#!/usr/bin/env python3
"""Merge pick-one (radio / mutually-exclusive) checkbox groups into a single
'which option is selected?' question.

Per-checkbox atomic questions are grouped by parent. A vision LLM classifies
each group as pick_one (exactly one option meant to be selected — claim
status, entity type, filing status, billing plan) vs select_many (independent
checklist — attachments, coverages, symptoms). Only pick_one groups are
merged; select_many / independent groups are left untouched.

Writes the merged bank in place (.pre_mc.bak) + qa/mc_changed_docs.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from pilot_50_gen_qa import gemini_generate, encoded_image, load_env_file  # noqa: E402

CLASSIFY_SYSTEM = (
    "You are shown a document page and a group of checkbox options that each "
    "currently have a separate question. Decide whether this group is:\n"
    "  pick_one   — a mutually-exclusive / radio-style choice where exactly "
    "ONE option is meant to be selected (e.g. claim status Open/Closed, "
    "entity type, filing status, billing plan Direct/Agency, gender).\n"
    "  select_many — independent checkboxes where any number may be checked "
    "(e.g. attachments, coverages, lines of business, a symptom checklist).\n\n"
    "Return exactly: {\"type\": \"pick_one\" | \"select_many\", "
    "\"field\": \"<short name of the field this group belongs to>\"}"
)


def opt_label(it: dict) -> str | None:
    """Extract just the option name that sits immediately before 'checkbox'.

    Atomic questions read 'In the X section, what is the status of the
    <LABEL> checkbox (...)?'. A non-greedy 'the (.+?) checkbox' anchors on
    the FIRST 'the' and swallows the whole question prefix — so we anchor
    on the LAST 'the ... checkbox' via a greedy '.*' lead-in instead.
    """
    q = it.get("question", "")
    m = re.search(r".*\bthe ['\"]?(.+?)['\"]? checkbox", q, re.I)
    if not m:
        return None
    lbl = m.group(1).strip().strip("'\"").strip()
    # guard against the old greedy capture leaking back in
    if len(lbl) > 60 or "what is the status" in lbl.lower():
        m2 = re.search(r"status of the ['\"]?(.+?)['\"]? checkbox", q, re.I)
        lbl = m2.group(1).strip().strip("'\"").strip() if m2 else lbl
    return lbl or None


def classify(opts: list[str], img: Path) -> dict:
    try:
        r = gemini_generate(
            [{"text": f"Checkbox option group: {opts}\n"
                      "Is this pick_one (mutually exclusive) or select_many?"},
             {"inline_data": {"mime_type": "image/png", "data": encoded_image(img)}}],
            system=CLASSIFY_SYSTEM, temperature=0.0)
        return r if isinstance(r, dict) else {"type": "select_many"}
    except Exception:
        return {"type": "select_many"}


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    load_env_file(REPO / ".env")
    qa = run / "qa"
    bank = json.loads((qa / "bank_r1.json").read_text())
    items = bank["items"]

    # group per-checkbox atomic questions by parent
    groups: dict[str, list] = defaultdict(list)
    other = []
    for it in items:
        if ("checkbox_state" in it.get("capabilities", [])
                and it.get("answer_type") != "checklist_yesno"
                and opt_label(it)):
            groups[it.get("parent_id", it["question_id"].rsplit("__", 1)[0])].append(it)
        else:
            other.append(it)
    # only multi-option groups are candidates; pure YES/NO groups belong to
    # the yes/no merge (single Yes/No/null question), never to MC.
    def _all_yesno(its: list) -> bool:
        labs = {(opt_label(it) or "").strip().upper() for it in its}
        return bool(labs) and labs <= {"YES", "NO", "Y", "N"}

    cand = {pid: its for pid, its in groups.items()
            if len(its) >= 2 and not _all_yesno(its)}
    other += [it for pid, its in groups.items()
              if len(its) < 2 or _all_yesno(its) for it in its]
    print(f"{len(cand)} multi-option checkbox groups to classify", flush=True)

    def do(pid_its):
        pid, its = pid_its
        img = run / "images" / f"{its[0].get('source_file','')}.png"
        opts = [opt_label(it) for it in its]
        res = classify(opts, img) if img.exists() else {"type": "select_many"}
        return pid, its, res

    classified = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(do, kv) for kv in cand.items()]
        for i, fut in enumerate(as_completed(futs), 1):
            pid, its, res = fut.result()
            classified[pid] = (its, res)
            if i % 40 == 0 or i == len(cand):
                print(f"  [{i}/{len(cand)}] classified ({(time.time()-t0)/60:.1f}min)",
                      flush=True)

    merged, changed, n_pick = [], set(), 0
    for pid, (its, res) in classified.items():
        if res.get("type") != "pick_one":
            other.extend(its)
            continue
        n_pick += 1
        field = res.get("field") or "this field"
        opts = [opt_label(it) for it in its]
        sel = [opt_label(it) for it in its
               if str(it.get("gold_answer", "")).strip().lower() in ("selected", "checked")]
        gold = sel[0] if len(sel) == 1 else ("null" if not sel else " | ".join(sel))
        src = its[0].get("source_file")
        # anchor on the first option question's section context if present
        ctx = ""
        m = re.match(r"(In .+?,)", its[0].get("question", ""))
        if m:
            ctx = m.group(1) + " "
        merged.append({
            "question_id": pid + "__mc",
            "parent_id": pid,
            "source_file": src,
            "question": f"{ctx}which option is selected for {field}? "
                        f"(one of: {', '.join(o for o in opts if o)}, or null)",
            "response_format": "<" + "|".join(o for o in opts if o) + "|null>",
            "gold_answer": gold,
            "capabilities": its[0].get("capabilities", []),
            "answer_type": "multiple_choice",
            "_merged_from": [it["question_id"] for it in its],
        })
        changed.add(src)

    bank["items"] = other + merged
    backup = qa / "bank_r1.json.pre_mc.bak"
    if not backup.exists():
        backup.write_text(json.dumps(
            {"items": items, **{k: v for k, v in bank.items() if k != "items"}},
            indent=2, ensure_ascii=False))
    (qa / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    (qa / "mc_changed_docs.json").write_text(json.dumps(sorted(changed), indent=2))

    print(f"\npick_one groups merged: {n_pick}  "
          f"(select_many left as-is: {len(cand)-n_pick})")
    print(f"affected docs: {len(changed)}")
    print(f"bank: {len(items)} -> {len(bank['items'])} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
