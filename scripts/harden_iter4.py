#!/usr/bin/env python3
"""Harden iteration 4 for v500.

Replaces both EASY questions (>=5/6 parsers correct, LLM-judge) and COMPUTED
questions (gold requires arithmetic) with harder, verbatim-only replacements.
Old questions are versioned in place (question_id stable, prior content kept
in the item's `versions[]` list — handled by pilot_50_gen_qa replace mode).

Steps: detect computed -> build results (force computed = replace-worthy) ->
replace-easy -> normalize -> swap bank -> re-eval changed docs -> rescore ->
append 'harden 4' to iteration_history -> rebuild dashboard.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from v500_harden_chain import per_parser_scores, n_questions, normalize_bank_formats  # noqa: E402

RUN = REPO / "runs" / "v500"
QA, EV = RUN / "qa", RUN / "eval"
PY = str(REPO / ".venv" / "bin" / "python")


def sh(cmd: list, log: Path) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    with log.open("w") as f:
        return subprocess.run([str(c) for c in cmd], stdout=f,
                              stderr=subprocess.STDOUT, cwd=REPO).returncode


def main() -> int:
    t0 = time.time()

    # 1. detect computed questions
    if sh([PY, "scripts/detect_computed.py", "v500"],
          RUN / "harden_iter4_computed.log") != 0:
        print("detect_computed failed"); return 1
    computed = set(json.loads((QA / "computed_qids.json").read_text()))
    print(f"computed questions flagged: {len(computed)}", flush=True)

    # 2. eval -> results; force computed questions to count as replace-worthy
    results = EV / "results_for_iter4.json"
    sh([PY, "scripts/eval_to_results.py", EV / "eval_r1_rescored.json", results, "llm"],
       RUN / "harden_iter4_adapter.log")
    rj = json.loads(results.read_text())
    for row in rj["results"]:
        if row["question_id"] in computed:
            row["exact_match"] = True            # force into the "easy" set
    results.write_text(json.dumps(rj))

    # 3. replace-easy (versioned + verbatim-only prompt)
    new_bank = QA / "bank_iter4.json"
    if new_bank.exists():
        new_bank.unlink()
    art = QA / "artifacts" / "harden_iter4"
    rc = sh([PY, "scripts/pilot_50_gen_qa.py", "--replace-easy",
             "--bank", QA / "bank_r1.json", "--results", results,
             "--out", new_bank, "--images", RUN / "images",
             "--artifact-dir", art, "--env-file", REPO / ".env",
             "--easy-threshold", "5", "--auto-fix-gold"],
            RUN / "harden_iter4_replace.log")
    if rc != 0 or not new_bank.exists():
        print(f"replace-easy failed rc={rc}"); return 1

    # 4. normalize placeholders, swap bank in, wipe changed docs' cache
    n_norm = normalize_bank_formats(new_bank)
    print(f"normalized {n_norm} response_format placeholders", flush=True)
    shutil.copy(QA / "bank_r1.json", QA / "bank_before_iter4.json")
    shutil.copy(new_bank, QA / "bank_r1.json")
    changed = set()
    log = art / "replace_easy_log.json"
    if log.exists():
        for e in json.loads(log.read_text()):
            if str(e.get("outcome", "")).startswith("replaced"):
                changed.add(e["source_file"])
    cache = EV / "answers_flat"
    n_wiped = 0
    for sf in changed:
        for f in cache.glob(f"{sf}__*.json"):
            f.unlink()
            n_wiped += 1
    print(f"{len(changed)} docs changed — wiped {n_wiped} cache files", flush=True)

    # 5. re-eval + rescore
    if sh([PY, "scripts/pilot_50_eval.py", "--run-dir", RUN, "--max-workers", "12"],
          RUN / "harden_iter4_eval.log") != 0:
        print("eval failed"); return 1
    if sh([PY, "scripts/pilot_50_rescore.py", "--run-dir", RUN, "--max-workers", "12"],
          RUN / "harden_iter4_rescore.log") != 0:
        print("rescore failed"); return 1

    # 6. history + dashboard
    hist_path = EV / "iteration_history.json"
    hist = json.loads(hist_path.read_text())
    rescored = EV / "eval_r1_rescored.json"
    hist.append({"iter": len(hist), "label": "harden 4",
                 "n_questions": n_questions(rescored),
                 "scores": per_parser_scores(rescored)})
    hist_path.write_text(json.dumps(hist, indent=2))
    shutil.copy(rescored, EV / "eval_iter4_rescored.json")
    sh([PY, "scripts/pilot_50_build_html.py", "--run-dir", RUN, "--mode", "post"],
       RUN / "harden_iter4_html.log")
    if (RUN / "review_post.html").exists():
        shutil.copy(RUN / "review_post.html", RUN / "dashboard.html")
    print(f"DONE in {(time.time()-t0)/60:.1f}min — dashboard rebuilt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
