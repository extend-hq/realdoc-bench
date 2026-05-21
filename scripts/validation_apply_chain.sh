#!/usr/bin/env bash
# After apply_validation.py: re-eval only the docs whose question set / golds
# changed (incremental — others stay cached), rescore, rebuild HTMLs.
set -e
cd "$(dirname "$0")/.."

LOG=runs/v500/validation_apply_chain.log
echo "[$(date +%T)] validation apply: wiping caches for changed docs" | tee "$LOG"
.venv/bin/python - <<'PY' | tee -a "$LOG"
import json, glob, os
docs = json.load(open('runs/v500/qa/validation_changed_docs.json'))
n = 0
for stem in docs:
    for f in glob.glob(f'runs/v500/eval/answers_flat/{stem}__*.json'):
        os.remove(f); n += 1
print(f'wiped {n} answers_flat files for {len(docs)} changed docs')
PY
rm -rf runs/v500/eval/llm_judge_struct

echo "[$(date +%T)] eval — consensus, changed docs only..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_eval.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/validation_apply_eval.log 2>&1
echo "[$(date +%T)] eval done (exit $?)" | tee -a "$LOG"
tail -14 runs/v500/validation_apply_eval.log | tee -a "$LOG"

echo "[$(date +%T)] rescore..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_rescore.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/validation_apply_rescore.log 2>&1
echo "[$(date +%T)] rescore done (exit $?)" | tee -a "$LOG"
tail -22 runs/v500/validation_apply_rescore.log | tee -a "$LOG"

echo "[$(date +%T)] rebuild validation.html + dashboard..." | tee -a "$LOG"
.venv/bin/python -u scripts/build_validation_html.py v500 >> "$LOG" 2>&1
.venv/bin/python -u scripts/pilot_50_build_html.py \
  --run-dir runs/v500 --mode post --out runs/v500/review_post.html >> "$LOG" 2>&1
cp runs/v500/review_post.html runs/v500/dashboard.html

echo "[$(date +%T)] validation apply chain COMPLETE" | tee -a "$LOG"
