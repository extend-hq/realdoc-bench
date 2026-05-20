#!/usr/bin/env bash
# After rewrite_curated.py finishes: wipe the stale eval caches, re-eval the
# curated+rewritten bank, rescore (3-layer), rebuild validation.html + the
# dashboard. Idempotent enough to re-run.
set -e
cd "$(dirname "$0")/.."

LOG=runs/v500/curate_chain.log
echo "[$(date +%T)] curate-eval chain: waiting for rewrite to finish" | tee "$LOG"
until grep -q "^wrote:" runs/v500/rewrite.log 2>/dev/null; do
  sleep 20
done
echo "[$(date +%T)] rewrite done" | tee -a "$LOG"
tail -10 runs/v500/rewrite.log | tee -a "$LOG"

# every question's text + gold changed -> both caches are stale
rm -rf runs/v500/eval/answers_flat runs/v500/eval/llm_judge_struct
echo "[$(date +%T)] wiped answers_flat + llm_judge_struct caches" | tee -a "$LOG"

echo "[$(date +%T)] eval (curated bank)..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_eval.py \
  --run-dir runs/v500 --max-workers 12 >> runs/v500/curate_eval.log 2>&1
echo "[$(date +%T)] eval done (exit $?)" | tee -a "$LOG"
tail -6 runs/v500/curate_eval.log | tee -a "$LOG"

echo "[$(date +%T)] rescore..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_rescore.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/curate_rescore.log 2>&1
echo "[$(date +%T)] rescore done (exit $?)" | tee -a "$LOG"
tail -16 runs/v500/curate_rescore.log | tee -a "$LOG"

echo "[$(date +%T)] building validation.html..." | tee -a "$LOG"
.venv/bin/python -u scripts/build_validation_html.py v500 >> "$LOG" 2>&1

echo "[$(date +%T)] building dashboard..." | tee -a "$LOG"
.venv/bin/python -u scripts/pilot_50_build_html.py \
  --run-dir runs/v500 --mode post --out runs/v500/review_post.html >> "$LOG" 2>&1
cp runs/v500/review_post.html runs/v500/dashboard.html

echo "[$(date +%T)] curate-eval chain COMPLETE" | tee -a "$LOG"
echo "  validation: runs/v500/qa/validation.html" | tee -a "$LOG"
echo "  dashboard : runs/v500/dashboard.html" | tee -a "$LOG"
