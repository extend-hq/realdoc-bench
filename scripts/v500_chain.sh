#!/usr/bin/env bash
# Chain runner: wait for parse + qa-gen to finish, then run eval → rescore → HTML.
# Idempotent — picks up wherever the prior stages ended.

set -e
cd "$(dirname "$0")/.."

LOG=runs/v500/chain.log
echo "[$(date +%T)] chain runner started" | tee "$LOG"

# 1. Wait for parse to finish
echo "[$(date +%T)] waiting for parse.log → 'Done in'" | tee -a "$LOG"
until grep -q "^Done in " runs/v500/parse.log 2>/dev/null; do
  sleep 30
done
echo "[$(date +%T)] parse done" | tee -a "$LOG"
tail -3 runs/v500/parse.log | tee -a "$LOG"

# 2. Wait for qa-gen to finish
echo "[$(date +%T)] waiting for qa.log → 'Wrote '" | tee -a "$LOG"
until grep -q "^Wrote " runs/v500/qa.log 2>/dev/null; do
  sleep 30
done
echo "[$(date +%T)] qa-gen done" | tee -a "$LOG"
tail -3 runs/v500/qa.log | tee -a "$LOG"

# 3. Run eval
echo "[$(date +%T)] running eval..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_eval.py \
  --run-dir runs/v500 --max-workers 10 \
  >> runs/v500/eval.log 2>&1
echo "[$(date +%T)] eval done (exit $?)" | tee -a "$LOG"
tail -8 runs/v500/eval.log | tee -a "$LOG"

# 4. Run rescore (strict → fuzzy → llm)
echo "[$(date +%T)] running rescore..." | tee -a "$LOG"
caffeinate -i .venv/bin/python -u scripts/pilot_50_rescore.py \
  --run-dir runs/v500 --max-workers 8 \
  >> runs/v500/rescore.log 2>&1
echo "[$(date +%T)] rescore done (exit $?)" | tee -a "$LOG"
tail -8 runs/v500/rescore.log | tee -a "$LOG"

# 5. Build HTMLs (pre + post)
echo "[$(date +%T)] building HTMLs..." | tee -a "$LOG"
.venv/bin/python -u scripts/pilot_50_build_html.py \
  --run-dir runs/v500 --mode pre \
  --out runs/v500/review_pre.html 2>&1 | tee -a "$LOG"
.venv/bin/python -u scripts/pilot_50_build_html.py \
  --run-dir runs/v500 --mode post \
  --out runs/v500/review_post.html 2>&1 | tee -a "$LOG"

echo "[$(date +%T)] chain COMPLETE. Open: runs/v500/review_post.html" | tee -a "$LOG"
