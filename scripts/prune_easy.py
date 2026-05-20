#!/usr/bin/env python3
"""Prune non-discriminating questions from a run's bank.

A question every parser answers correctly contributes no signal to a
parser-comparison benchmark. This drops questions whose parsers-correct count
(LLM-judge, question-level) is >= --threshold, keeping at least --floor
questions per document so every doc stays represented.

Dropped questions are archived to qa/dropped_questions.json (recoverable),
eval_r1_rescored.json is filtered to the kept set, the iteration-history tail
is recomputed, and the dashboard is rebuilt. No re-eval needed — dropping
questions only removes rows from aggregation.

Usage: prune_easy.py [run] [--threshold 6] [--floor 1]
  --threshold 6  drop only 6/6 (all parsers correct) — default
  --threshold 5  drop >=5/6 (more aggressive)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from v500_harden_chain import per_parser_scores   # noqa: E402

PY = str(REPO / ".venv" / "bin" / "python")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="v500")
    ap.add_argument("--threshold", type=int, default=6,
                    help="drop questions with >= this many parsers correct (LLM)")
    ap.add_argument("--floor", type=int, default=1,
                    help="keep at least this many questions per document")
    ap.add_argument("--keep-frac", type=float, default=0.30,
                    help="keep this fraction of the would-be-dropped easy "
                         "questions (random) as an easy tier; rest archived")
    args = ap.parse_args()
    run = REPO / "runs" / args.run
    qa, ev = run / "qa", run / "eval"

    bank = json.loads((qa / "bank_r1.json").read_text())
    items = bank["items"]
    rescored = json.loads((ev / "eval_r1_rescored.json").read_text())

    # question_id -> parsers-correct count (LLM-judge, question level)
    n_correct = {}
    for page in rescored.values():
        for qid, q in page["questions"].items():
            n_correct[qid] = len(q.get("parsers_correct_llm", []))

    # group bank items by doc, decide drops with a per-doc floor
    by_doc: dict[str, list] = {}
    for it in items:
        by_doc.setdefault(it.get("source_file"), []).append(it)

    keep_ids, drop_records, candidates = set(), [], []
    for src, doc_items in by_doc.items():
        # hardest first (fewest parsers correct) — floor keeps the hardest
        ranked = sorted(doc_items, key=lambda it: n_correct.get(it["question_id"], 0))
        for rank, it in enumerate(ranked):
            qid = it["question_id"]
            nc = n_correct.get(qid, 0)
            if nc >= args.threshold and rank >= args.floor:
                candidates.append((it, src, nc))      # eligible to drop
            else:
                keep_ids.add(qid)

    # keep a random sample of the easy candidates as an easy tier; the dropped
    # ones are archived (recoverable — can be added back later)
    random.seed(0)
    random.shuffle(candidates)
    n_keep = int(round(len(candidates) * args.keep_frac))
    for it, src, nc in candidates[:n_keep]:
        keep_ids.add(it["question_id"])
    for it, src, nc in candidates[n_keep:]:
        drop_records.append({
            "question_id": it["question_id"], "source_file": src,
            "parsers_correct": nc, "question": it.get("question"),
            "gold_answer": it.get("gold_answer"),
        })

    kept_items = [it for it in items if it["question_id"] in keep_ids]
    bank["items"] = kept_items
    (qa / "bank_r1.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    (qa / "dropped_questions.json").write_text(
        json.dumps(drop_records, indent=2, ensure_ascii=False))

    # filter eval data to the kept set so the scoreboard stays consistent
    for page in rescored.values():
        page["questions"] = {q: v for q, v in page["questions"].items()
                             if q in keep_ids}
    rescored = {dp: pg for dp, pg in rescored.items() if pg["questions"]}
    (ev / "eval_r1_rescored.json").write_text(
        json.dumps(rescored, indent=2, ensure_ascii=False))

    # recompute the iteration-history tail against the pruned set
    hist_path = ev / "iteration_history.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())
        if hist:
            hist[-1]["n_questions"] = len(kept_items)
            hist[-1]["scores"] = per_parser_scores(ev / "eval_r1_rescored.json")
            hist[-1]["label"] = hist[-1].get("label", "") + " · pruned"
            hist_path.write_text(json.dumps(hist, indent=2))

    print(f"easy candidates (>= {args.threshold}/6 correct, floor {args.floor}): "
          f"{len(candidates)}  ->  kept {n_keep} as easy tier, "
          f"dropped {len(drop_records)} (archived, recoverable)")
    print(f"bank: {len(items)} -> {len(kept_items)} questions")
    print(f"archived: {qa / 'dropped_questions.json'}")

    rc = subprocess.run([PY, "scripts/pilot_50_build_html.py",
                         "--run-dir", str(run), "--mode", "post"],
                        cwd=REPO).returncode
    if rc == 0 and (run / "review_post.html").exists():
        shutil.copy(run / "review_post.html", run / "dashboard.html")
        print("dashboard rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
