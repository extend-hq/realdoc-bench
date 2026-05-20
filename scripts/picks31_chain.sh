#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
LOG=runs/picks31/chain.log
echo "[$(date +%T)] picks31 chain runner started" | tee "$LOG"

echo "[$(date +%T)] waiting for parse.log → 'Done in'" | tee -a "$LOG"
until grep -q "^Done in " runs/picks31/parse.log 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] parse done" | tee -a "$LOG"

echo "[$(date +%T)] waiting for qa.log → 'Wrote '" | tee -a "$LOG"
until grep -q "^Wrote " runs/picks31/qa.log 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] qa-gen done" | tee -a "$LOG"

echo "[$(date +%T)] running eval (6 parsers)..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_eval.py \
  --run-dir runs/picks31 --max-workers 8 \
  >> runs/picks31/eval.log 2>&1
echo "[$(date +%T)] eval done" | tee -a "$LOG"

echo "[$(date +%T)] running rescore..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_rescore.py \
  --run-dir runs/picks31 --max-workers 8 \
  >> runs/picks31/rescore.log 2>&1
echo "[$(date +%T)] rescore done" | tee -a "$LOG"

echo "[$(date +%T)] building HTML..." | tee -a "$LOG"
.venv/bin/python -u scripts/pilot_50_build_html.py --run-dir runs/picks31 --mode post \
  --out runs/picks31/review_post.html 2>&1 | tee -a "$LOG"
echo "[$(date +%T)] picks31 chain COMPLETE" | tee -a "$LOG"
