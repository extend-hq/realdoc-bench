#!/bin/bash
# Overnight: wait for the v500 batched baseline chain to finish, then run the
# 3-iteration question-hardening chain and build the dashboard. Fully unattended.
cd /Users/ameyajoshi/Projects/realdoc-bench || exit 1

echo "=== v500 overnight wrapper started $(date) ==="

# 1. Wait for the baseline chain (eval -> rescore -> build_html) to complete.
#    Its build_html step prints 'Wrote ...' only after eval+rescore succeeded.
echo "waiting for v500 baseline chain..."
for _ in $(seq 1 240); do          # up to 2h
  if [ -f runs/v500/build_html_flat.log ] && grep -q "Wrote runs/v500" runs/v500/build_html_flat.log; then
    echo "baseline chain done at $(date)"
    break
  fi
  sleep 30
done

if ! grep -q "Wrote runs/v500" runs/v500/build_html_flat.log 2>/dev/null; then
  echo "FATAL: baseline chain never completed — aborting overnight run"
  exit 1
fi

if [ ! -f runs/v500/eval/eval_r1_rescored.json ]; then
  echo "FATAL: eval_r1_rescored.json missing — aborting"
  exit 1
fi

# 2. Run the 3-iteration hardening chain + dashboard build.
echo "=== starting harden chain $(date) ==="
caffeinate -i .venv/bin/python -u scripts/v500_harden_chain.py v500
echo "=== v500 overnight wrapper finished $(date) exit=$? ==="
