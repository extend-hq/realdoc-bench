#!/usr/bin/env bash
# Parse v500 with the new extend_performance_v2_0_0_agentic parser (agentic OCR
# + agentic tables + change detection), then re-eval (incremental — only the new
# parser, the other 6 stay cached), rescore, rebuild validation.html + dashboard.
set -e
cd "$(dirname "$0")/.."

LOG=runs/v500/agentic_chain.log
echo "[$(date +%T)] agentic chain: parse 529 docs with extend_performance_v2_0_0_agentic" | tee "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_parse.py \
  --run-dir runs/v500 --parsers extend_performance_v2_0_0_agentic --workers 8 \
  >> runs/v500/agentic_parse.log 2>&1
echo "[$(date +%T)] parse done (exit $?)" | tee -a "$LOG"
tail -6 runs/v500/agentic_parse.log | tee -a "$LOG"

echo "[$(date +%T)] eval (incremental — agentic only, other 6 cached)..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_eval.py \
  --run-dir runs/v500 --max-workers 12 >> runs/v500/agentic_eval.log 2>&1
echo "[$(date +%T)] eval done (exit $?)" | tee -a "$LOG"
tail -8 runs/v500/agentic_eval.log | tee -a "$LOG"

echo "[$(date +%T)] rescore..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_rescore.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/agentic_rescore.log 2>&1
echo "[$(date +%T)] rescore done (exit $?)" | tee -a "$LOG"
tail -18 runs/v500/agentic_rescore.log | tee -a "$LOG"

echo "[$(date +%T)] rebuild validation.html + dashboard..." | tee -a "$LOG"
.venv/bin/python -u scripts/build_validation_html.py v500 >> "$LOG" 2>&1
.venv/bin/python -u scripts/pilot_50_build_html.py \
  --run-dir runs/v500 --mode post --out runs/v500/review_post.html >> "$LOG" 2>&1
cp runs/v500/review_post.html runs/v500/dashboard.html

echo "[$(date +%T)] agentic chain COMPLETE" | tee -a "$LOG"
echo "  validation: runs/v500/qa/validation.html" | tee -a "$LOG"
echo "  dashboard : runs/v500/dashboard.html" | tee -a "$LOG"
