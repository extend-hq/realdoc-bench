#!/usr/bin/env python3
"""Overnight question-hardening chain for a run dir (default: runs/v500).

For each iteration:
  1. Convert the current eval into a {results:[...]} file (eval_to_results.py).
  2. pilot_50_gen_qa.py --replace-easy: every question >= --easy-threshold
     parsers nailed (LLM-judge) is replaced with a HARDER one generated from
     the page image, gold validated by an image-baseline call.
  3. Normalize the new bank's response_format placeholders (<digits> -> <value>).
  4. Swap the new bank in; wipe the eval cache for docs whose questions changed.
  5. Re-run eval (batched, 1 call/doc/parser) + 3-layer rescore.
  6. Snapshot eval_iter<N>_rescored.json + append to iteration_history.json.

Finally rebuilds the HTML dashboard (review_post.html / dashboard.html), which
renders the iteration-history panel from iteration_history.json.

Idempotent-ish: each iteration snapshots its outputs, so a mid-chain failure
still leaves earlier iterations intact and the dashboard buildable.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / "bin" / "python")
PARSERS = ["extend_performance_v1_0_0", "extend_performance_v2_0_0_advanced",
           "aws_textract", "azure_di", "reducto", "llamaparse"]
N_ITERS = 3
EASY_THRESHOLD = 5          # >= this many of 6 parsers correct (LLM) = "easy"


def sh(cmd: list[str], log: Path) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        return subprocess.run([str(c) for c in cmd], stdout=f,
                              stderr=subprocess.STDOUT, cwd=REPO).returncode


def per_parser_scores(rescored: Path) -> dict:
    """Per-parser sub-answer + question tallies across all 3 scoring layers."""
    data = json.loads(rescored.read_text())
    out = {}
    for p in PARSERS:
        sc = {k: 0 for k in ("sub_strict", "sub_fuzzy", "sub_llm", "sub_t",
                              "q_strict", "q_fuzzy", "q_llm", "q_t")}
        for page in data.values():
            for q in page.get("questions", {}).values():
                if p not in q.get("parser_answers", {}):
                    continue
                sc["q_t"] += 1
                for layer in ("strict", "fuzzy", "llm"):
                    if p in q.get(f"parsers_correct_{layer}", []):
                        sc[f"q_{layer}"] += 1
                for k in q.get("keys", []):
                    sc["sub_t"] += 1
                    for layer in ("strict", "fuzzy", "llm"):
                        if p in q.get(f"sub_correct_{layer}", {}).get(k, []):
                            sc[f"sub_{layer}"] += 1
        out[p] = sc
    return out


_DIGITS = re.compile(r"<[^<>]*digits[^<>]*>")


def normalize_bank_formats(bank_path: Path) -> int:
    """Drop any <digits>-style placeholder from a freshly generated bank — they
    make the answer LLM strip commas/decimals (5,280.00 -> 528000)."""
    bank = json.loads(bank_path.read_text())
    n = 0
    for it in bank.get("items", []):
        fmt = it.get("response_format", "")
        new = _DIGITS.sub(
            lambda m: "<value|null>" if ("null" in m.group(0).lower()
                                         or re.search(r"[-—–]", m.group(0)))
            else "<value>", fmt)
        new = re.sub(r"\bdigits\b", "value", new)
        if new != fmt:
            it["response_format"] = new
            n += 1
    if n:
        bank_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    return n


def n_questions(rescored: Path) -> int:
    data = json.loads(rescored.read_text())
    return sum(len(p.get("questions", {})) for p in data.values())


def main() -> int:
    run = REPO / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "v500")
    eval_dir = run / "eval"
    qa_dir = run / "qa"
    rescored = eval_dir / "eval_r1_rescored.json"
    if not rescored.exists():
        print(f"FATAL: {rescored} missing — run the baseline eval+rescore first")
        return 1

    t0 = time.time()
    hist_path = eval_dir / "iteration_history.json"

    # iter 0 = current baseline
    shutil.copy(rescored, eval_dir / "eval_iter0_rescored.json")
    history = [{
        "iter": 0, "label": "baseline",
        "n_questions": n_questions(rescored),
        "scores": per_parser_scores(rescored),
    }]
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"iter 0 (baseline): {history[0]['n_questions']} questions", flush=True)

    for it in range(1, N_ITERS + 1):
        print(f"\n=== HARDEN ITERATION {it}/{N_ITERS} ({(time.time()-t0)/60:.1f}min) ===",
              flush=True)
        art = qa_dir / "artifacts" / f"harden_iter{it}"
        results = eval_dir / f"results_for_iter{it}.json"

        # 1. adapter
        if sh([PY, "scripts/eval_to_results.py", rescored, results, "llm"],
              run / f"harden_iter{it}_adapter.log") != 0:
            print("  adapter failed — stopping"); break

        # 2. replace-easy (vision-LLM harder questions + image-baseline gold check)
        new_bank = qa_dir / f"bank_iter{it}.json"
        if new_bank.exists():
            new_bank.unlink()
        rc = sh([PY, "scripts/pilot_50_gen_qa.py", "--replace-easy",
                 "--bank", qa_dir / "bank_r1.json",
                 "--results", results,
                 "--out", new_bank,
                 "--images", run / "images",
                 "--artifact-dir", art,
                 "--env-file", REPO / ".env",
                 "--easy-threshold", EASY_THRESHOLD, "--auto-fix-gold"],
                run / f"harden_iter{it}_replace.log")
        if rc != 0 or not new_bank.exists():
            print(f"  replace-easy failed (rc={rc}) — stopping at iter {it-1}")
            break

        # 3. normalize placeholders on the new bank
        n_norm = normalize_bank_formats(new_bank)
        print(f"  normalized {n_norm} response_format placeholders", flush=True)

        # 4. swap bank in, wipe changed docs' eval cache
        shutil.copy(qa_dir / "bank_r1.json", qa_dir / f"bank_before_iter{it}.json")
        shutil.copy(new_bank, qa_dir / "bank_r1.json")
        changed = set()
        log = art / "replace_easy_log.json"
        if log.exists():
            for e in json.loads(log.read_text()):
                if str(e.get("outcome", "")).startswith("replaced"):
                    changed.add(e["source_file"])
        cache = eval_dir / "answers_flat"
        n_wiped = 0
        for sf in changed:
            for f in cache.glob(f"{sf}__*.json"):
                f.unlink()
                n_wiped += 1
        print(f"  {len(changed)} docs changed — wiped {n_wiped} cache files", flush=True)

        # 5. re-eval + rescore
        if sh([PY, "scripts/pilot_50_eval.py", "--run-dir", run, "--max-workers", "12"],
              run / f"harden_iter{it}_eval.log") != 0:
            print("  eval failed — stopping"); break
        if sh([PY, "scripts/pilot_50_rescore.py", "--run-dir", run, "--max-workers", "12"],
              run / f"harden_iter{it}_rescore.log") != 0:
            print("  rescore failed — stopping"); break

        # 6. snapshot + history
        shutil.copy(rescored, eval_dir / f"eval_iter{it}_rescored.json")
        history.append({
            "iter": it, "label": f"harden {it}",
            "n_questions": n_questions(rescored),
            "scores": per_parser_scores(rescored),
        })
        hist_path.write_text(json.dumps(history, indent=2))
        sc = history[-1]["scores"]
        line = "  ".join(
            f"{p.split('_')[0][:6]}:{sc[p]['sub_llm']/max(1,sc[p]['sub_t'])*100:.0f}%"
            for p in PARSERS)
        print(f"  iter {it} done — sub-answer LLM acc: {line}", flush=True)

    # dashboard
    print(f"\n=== building dashboard ({(time.time()-t0)/60:.1f}min) ===", flush=True)
    sh([PY, "scripts/pilot_50_build_html.py", "--run-dir", run, "--mode", "post"],
       run / "harden_dashboard.log")
    if (run / "review_post.html").exists():
        shutil.copy(run / "review_post.html", run / "dashboard.html")
    print(f"\nDONE in {(time.time()-t0)/60:.1f}min — dashboard: "
          f"{run / 'dashboard.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
