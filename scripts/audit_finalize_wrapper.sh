#!/bin/bash
# Wait for both the flatten re-eval and the vision audit, then run the audit
# finalize phase (hallucination check + needs_review.html).
cd /Users/ameyajoshi/Projects/realdoc-bench || exit 1
echo "audit-finalize wrapper: waiting for flatten re-eval + vision audit $(date)"
for _ in $(seq 1 240); do
  grep -q "DONE" runs/v500/reeval_flatten.log 2>/dev/null && \
  grep -q "vision phase done" runs/v500/audit_vision.log 2>/dev/null && break
  sleep 30
done
if ! grep -q "DONE" runs/v500/reeval_flatten.log 2>/dev/null; then
  echo "FATAL: flatten re-eval never finished"; exit 1
fi
if ! grep -q "vision phase done" runs/v500/audit_vision.log 2>/dev/null; then
  echo "FATAL: vision audit never finished"; exit 1
fi
echo "both done — running audit finalize $(date)"
caffeinate -i .venv/bin/python -u scripts/audit_qa.py --run-dir runs/v500 --phase finalize
echo "audit-finalize wrapper done $(date) exit=$?"
