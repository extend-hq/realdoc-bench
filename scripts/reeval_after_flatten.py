#!/usr/bin/env python3
"""Re-eval v500 after the flatten migration.

Wipes the answers_flat cache for every doc whose questions were flattened,
re-runs eval + rescore (so those questions are now scored per-compound-key),
appends a 'flatten fix' entry to iteration_history.json, and rebuilds the
dashboard.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from v500_harden_chain import per_parser_scores, n_questions   # noqa: E402

RUN = REPO / "runs" / "v500"
PY = str(REPO / ".venv" / "bin" / "python")


def run(cmd: list[str], log: Path) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    with log.open("w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode


def main() -> int:
    changed = json.loads((RUN / "qa" / "flatten_changed_docs.json").read_text())
    cache = RUN / "eval" / "answers_flat"
    n_wiped = 0
    for sf in changed:
        for f in cache.glob(f"{sf}__*.json"):
            f.unlink()
            n_wiped += 1
    print(f"wiped {n_wiped} cache files for {len(changed)} flattened docs", flush=True)

    if run([PY, "scripts/pilot_50_eval.py", "--run-dir", str(RUN), "--max-workers", "12"],
           RUN / "reeval_flatten_eval.log") != 0:
        print("eval failed"); return 1
    if run([PY, "scripts/pilot_50_rescore.py", "--run-dir", str(RUN), "--max-workers", "12"],
           RUN / "reeval_flatten_rescore.log") != 0:
        print("rescore failed"); return 1

    # append a 'flatten fix' entry to the iteration history
    hist_path = RUN / "eval" / "iteration_history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
    if hist and hist[-1].get("label") == "flatten fix":
        hist.pop()                      # replace prior flatten entry, don't stack
    rescored = RUN / "eval" / "eval_r1_rescored.json"
    hist.append({
        "iter": len(hist), "label": "flatten fix",
        "n_questions": n_questions(rescored),
        "scores": per_parser_scores(rescored),
    })
    hist_path.write_text(json.dumps(hist, indent=2))

    if run([PY, "scripts/pilot_50_build_html.py", "--run-dir", str(RUN), "--mode", "post"],
           RUN / "reeval_flatten_html.log") != 0:
        print("build_html failed"); return 1
    shutil.copy(RUN / "review_post.html", RUN / "dashboard.html")
    print("DONE — dashboard rebuilt at runs/v500/dashboard.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
