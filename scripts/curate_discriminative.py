#!/usr/bin/env python3
"""Curate the v500 bank down to ~4-5 *discriminative* questions per doc.

Atomization exploded every multi-part question into many repetitive
single-value fragments ("status of the first / second / third checkbox").
This script reverses the bloat: per doc it keeps only the questions that
best SEPARATE the 6 parsers, deduped so no one section/capability dominates.

  discriminative score   disc(n) = n * (6 - n)     n = #parsers correct
  3/3 split -> 9 (max) ........ 6/6 or 0/6 -> 0 (no signal at all)

MC questions were just re-merged (their text changed) so their current
eval is stale; their disc is estimated from the pre-merge per-checkbox
questions (eval_iter4_rescored, via the _merged_from list).

Writes qa/bank_curated.json + qa/curate_report.json. Does NOT touch
bank_r1.json — the rewrite step consumes bank_curated.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent

PER_DOC = 5          # target questions kept per doc
MIN_DOC = 4          # never drop below this if the doc has the questions
BUCKET_CAP = 2       # max picks sharing one (section, capability) bucket
N_PARSERS = 6

_ORDINALS = (r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
             r"ninth|tenth|eleventh|twelfth|1st|2nd|3rd|\d+th)\b")
_TYPE_PRIORITY = {              # diversity tie-break only
    "chart_figure_exact": 3,
    "complex_checkmark_exact": 2,
    "multiple_choice": 1,
    "checklist_yesno": 1,
}


def stem_to_key(eval_obj: dict) -> dict[str, str]:
    """Map a bare doc stem -> the full eval doc-path key."""
    out = {}
    for k in eval_obj:
        out[Path(k).stem] = k
    return out


def disc_of(entry: dict | None) -> tuple[float, int]:
    if not entry:
        return 0.0, -1
    n = len(entry.get("parsers_correct") or [])
    return float(n * (N_PARSERS - n)), n


def section_anchor(q: str) -> str:
    m = re.match(r"\s*In (?:the )?(.+?)(?: section| grid| table| block| area)\b",
                 q, re.I)
    if m:
        return m.group(1).strip().lower()[:48]
    m = re.match(r"\s*(In .+?,)", q)
    return m.group(1).strip().lower()[:48] if m else "_"


def primary_cap(it: dict) -> str:
    caps = it.get("capabilities") or []
    return (caps[0] if caps else it.get("answer_type", "?"))


def norm_q(q: str) -> str:
    """Collapse ordinals / digits / quoted labels so checkbox-grid siblings
    map to the same string (used as a hard dedup key)."""
    t = q.lower()
    t = re.sub(_ORDINALS, "#", t)
    t = re.sub(r"['\"][^'\"]{1,40}['\"]", "#", t)
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"[^a-z#]+", " ", t).strip()
    return t


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    qa = run / "qa"
    bank = json.loads((qa / "bank_r1.json").read_text())
    items = bank["items"]

    cur = json.loads((run / "eval" / "eval_r1_rescored.json").read_text())
    iter4 = json.loads((run / "eval" / "eval_iter4_rescored.json").read_text())
    cur_key = stem_to_key(cur)
    it4_key = stem_to_key(iter4)

    def q_entry(obj, keymap, stem, qid):
        dk = keymap.get(stem)
        if not dk:
            return None
        return (obj[dk].get("questions") or {}).get(qid)

    # --- discriminative score for every question --------------------------
    score: dict[str, float] = {}
    nbin: dict[str, int] = {}
    miss = 0
    for it in items:
        qid = it["question_id"]
        stem = it.get("source_file", "")
        if it.get("answer_type") == "multiple_choice":
            # stale text -> estimate from pre-merge per-checkbox questions
            ds = []
            for cb in it.get("_merged_from") or []:
                d, _ = disc_of(q_entry(iter4, it4_key, stem, cb))
                if _ >= 0:
                    ds.append(d)
            score[qid] = mean(ds) if ds else 4.0
            nbin[qid] = -1
        else:
            d, n = disc_of(q_entry(cur, cur_key, stem, qid))
            if n < 0:
                # fall back to pre-merge children, else neutral
                ds = [disc_of(q_entry(iter4, it4_key, stem, cb))[0]
                      for cb in it.get("_merged_from") or []]
                ds = [x for x in ds if x is not None]
                score[qid] = mean(ds) if ds else 0.0
                miss += 1
            else:
                score[qid] = d
            nbin[qid] = n
    print(f"scored {len(items)} questions  ({miss} had no eval entry)", flush=True)

    # --- per-doc selection -------------------------------------------------
    by_doc: dict[str, list] = defaultdict(list)
    for it in items:
        by_doc[it.get("source_file", "")].append(it)

    kept: list[dict] = []
    report = {"per_doc": {}, "totals": {}}
    disc_hist = Counter()
    for stem, doc_items in by_doc.items():
        # hard dedup: collapse exact checkbox-grid siblings, keep best disc
        seen: dict[str, dict] = {}
        for it in doc_items:
            k = (norm_q(it["question"]), primary_cap(it))
            if k not in seen or score[it["question_id"]] > score[seen[k]["question_id"]]:
                seen[k] = it
        cand = list(seen.values())

        def sort_key(it):
            return (-score[it["question_id"]],
                    -_TYPE_PRIORITY.get(it.get("answer_type", ""), 0),
                    len(it.get("question", "")))

        cand.sort(key=sort_key)

        picks, bcount, pid_set = [], Counter(), set()
        for it in cand:
            if len(picks) >= PER_DOC:
                break
            b = (section_anchor(it["question"]), primary_cap(it))
            if bcount[b] >= BUCKET_CAP:
                continue
            picks.append(it)
            bcount[b] += 1
            pid_set.add(it["question_id"])
        # relax bucket cap to reach the floor
        if len(picks) < MIN_DOC:
            for it in cand:
                if len(picks) >= MIN_DOC:
                    break
                if it["question_id"] not in pid_set:
                    picks.append(it)
                    pid_set.add(it["question_id"])

        kept.extend(picks)
        ds = [score[it["question_id"]] for it in picks]
        for it in picks:
            disc_hist[nbin[it["question_id"]]] += 1
        report["per_doc"][stem] = {
            "in": len(doc_items), "deduped": len(cand), "out": len(picks),
            "disc": [round(x, 1) for x in ds],
        }

    bank_out = {k: v for k, v in bank.items() if k != "items"}
    bank_out["items"] = kept
    bank_out["curated"] = True
    (qa / "bank_curated.json").write_text(
        json.dumps(bank_out, indent=2, ensure_ascii=False) + "\n")

    report["totals"] = {
        "docs": len(by_doc),
        "questions_in": len(items),
        "questions_out": len(kept),
        "mean_per_doc": round(len(kept) / max(len(by_doc), 1), 2),
        "n_correct_hist": {str(k): v for k, v in sorted(disc_hist.items())},
    }
    (qa / "curate_report.json").write_text(json.dumps(report, indent=2))

    print(f"\ncurated {len(items)} -> {len(kept)} questions "
          f"across {len(by_doc)} docs ({report['totals']['mean_per_doc']}/doc)")
    print("kept-question parser-correct histogram (n correct -> count):")
    for k in sorted(disc_hist):
        lab = "MC (est.)" if k < 0 else f"{k}/6"
        print(f"  {lab:>10}: {disc_hist[k]}")
    print(f"\nwrote {qa/'bank_curated.json'}")
    print(f"wrote {qa/'curate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
