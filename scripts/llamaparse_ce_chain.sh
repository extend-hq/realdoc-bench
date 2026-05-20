#!/usr/bin/env bash
# Parse v500 with llamaparse_cost_effective (LlamaParse cost_effective tier),
# then incrementally eval just that parser (consensus), rescore, rebuild HTMLs.
set -e
cd "$(dirname "$0")/.."

LOG=runs/v500/llamaparse_ce_chain.log
echo "[$(date +%T)] llamaparse cost-effective: parse 529 docs" | tee "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_parse.py \
  --run-dir runs/v500 --parsers llamaparse_cost_effective --workers 8 \
  >> runs/v500/llamaparse_ce_parse.log 2>&1
echo "[$(date +%T)] parse done (exit $?)" | tee -a "$LOG"
tail -5 runs/v500/llamaparse_ce_parse.log | tee -a "$LOG"

# don't start eval while the nullfix eval chain is still touching the caches
echo "[$(date +%T)] waiting for nullfix chain to finish..." | tee -a "$LOG"
until grep -q "nullfix eval chain COMPLETE" runs/v500/nullfix_chain.log 2>/dev/null; do
  sleep 15
done

echo "[$(date +%T)] eval (incremental — cost-effective only, consensus)..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_eval.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/llamaparse_ce_eval.log 2>&1
echo "[$(date +%T)] eval done (exit $?)" | tee -a "$LOG"
tail -12 runs/v500/llamaparse_ce_eval.log | tee -a "$LOG"

echo "[$(date +%T)] rescore..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_rescore.py \
  --run-dir runs/v500 --max-workers 8 >> runs/v500/llamaparse_ce_rescore.log 2>&1
echo "[$(date +%T)] rescore done (exit $?)" | tee -a "$LOG"
tail -22 runs/v500/llamaparse_ce_rescore.log | tee -a "$LOG"

echo "[$(date +%T)] rebuild validation.html + dashboard..." | tee -a "$LOG"
.venv/bin/python -u scripts/build_validation_html.py v500 >> "$LOG" 2>&1
.venv/bin/python -u scripts/pilot_50_build_html.py \
  --run-dir runs/v500 --mode post --out runs/v500/review_post.html >> "$LOG" 2>&1
cp runs/v500/review_post.html runs/v500/dashboard.html

echo "[$(date +%T)] llamaparse cost-effective chain COMPLETE" | tee -a "$LOG"
