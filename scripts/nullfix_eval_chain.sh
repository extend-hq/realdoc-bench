#!/usr/bin/env bash
# After rewrite_null_gold.py: re-eval only the docs whose questions changed
# (incremental — the other ~320 docs stay cached), rescore, rebuild HTMLs.
set -e
cd "$(dirname "$0")/.."

LOG=runs/v500/nullfix_chain.log
echo "[$(date +%T)] nullfix eval: wiping caches for changed docs" | tee "$LOG"
.venv/bin/python - <<'PY' | tee -a "$LOG"
import json, glob, os
docs = json.load(open('runs/v500/qa/nullfix_changed_docs.json'))
n = 0
for stem in docs:
    for f in glob.glob(f'runs/v500/eval/answers_flat/{stem}__*.json'):
        os.remove(f); n += 1
print(f'wiped {n} answers_flat files for {len(docs)} changed docs')
PY
rm -rf runs/v500/eval/llm_judge_struct

echo "[$(date +%T)] eval — consensus, changed docs only..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_eval.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/nullfix_eval.log 2>&1
echo "[$(date +%T)] eval done (exit $?)" | tee -a "$LOG"
tail -12 runs/v500/nullfix_eval.log | tee -a "$LOG"

echo "[$(date +%T)] rescore..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_rescore.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/nullfix_rescore.log 2>&1
echo "[$(date +%T)] rescore done (exit $?)" | tee -a "$LOG"
tail -20 runs/v500/nullfix_rescore.log | tee -a "$LOG"

echo "[$(date +%T)] rebuild validation.html + dashboard..." | tee -a "$LOG"
.venv/bin/python -u scripts/build_validation_html.py v500 >> "$LOG" 2>&1
.venv/bin/python -u scripts/pilot_50_build_html.py \
  --run-dir runs/v500 --mode post --out runs/v500/review_post.html >> "$LOG" 2>&1
cp runs/v500/review_post.html runs/v500/dashboard.html

echo "[$(date +%T)] nullfix eval chain COMPLETE" | tee -a "$LOG"
