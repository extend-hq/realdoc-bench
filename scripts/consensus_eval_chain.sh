#!/usr/bin/env bash
# Re-eval v500 with the cross-model consensus grader: every (doc,parser) batch
# answered by Sonnet ×3 + Gemini ×3, per-question majority vote. Kills the
# single-model temp-1.0 noise (the random all-null batches). Then rescore +
# rebuild validation.html / dashboard. All 7 parsers.
set -e
cd "$(dirname "$0")/.."

LOG=runs/v500/consensus_chain.log
echo "[$(date +%T)] consensus eval: wiping stale single-pass caches" | tee "$LOG"
rm -rf runs/v500/eval/answers_flat runs/v500/eval/llm_judge_struct

echo "[$(date +%T)] eval — Sonnet×3 + Gemini×3 majority vote, 7 parsers..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_eval.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/consensus_eval.log 2>&1
echo "[$(date +%T)] eval done (exit $?)" | tee -a "$LOG"
tail -10 runs/v500/consensus_eval.log | tee -a "$LOG"

echo "[$(date +%T)] rescore..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_rescore.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/consensus_rescore.log 2>&1
echo "[$(date +%T)] rescore done (exit $?)" | tee -a "$LOG"
tail -20 runs/v500/consensus_rescore.log | tee -a "$LOG"

echo "[$(date +%T)] rebuild validation.html + dashboard..." | tee -a "$LOG"
.venv/bin/python -u scripts/build_validation_html.py v500 >> "$LOG" 2>&1
.venv/bin/python -u scripts/pilot_50_build_html.py \
  --run-dir runs/v500 --mode post --out runs/v500/review_post.html >> "$LOG" 2>&1
cp runs/v500/review_post.html runs/v500/dashboard.html

echo "[$(date +%T)] consensus eval chain COMPLETE" | tee -a "$LOG"
echo "  validation: runs/v500/qa/validation.html" | tee -a "$LOG"
echo "  dashboard : runs/v500/dashboard.html" | tee -a "$LOG"
