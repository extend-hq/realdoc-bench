#!/usr/bin/env bash
# Entrypoint for the truss-train OCR-host job. Runs inside a vLLM-OpenAI
# container on a Baseten H200 (or H100) GPU.
#
# Boot sequence:
#   1. `vllm serve <HF_MODEL>` on :8000 in the background, with the same flags
#      the docker_server config used to pass.
#   2. Wait until `/health` returns 200 — that's our own readiness gate
#      (no Baseten gateway in the path).
#   3. Hold open with `sleep` so the interactive session stays alive long
#      enough for realdoc-bench to drive the model from the laptop via the
#      SSH tunnel that `truss train isession` sets up.
#
# How realdoc-bench talks to it:
#   The training-job HAS no public HTTP endpoint by itself. After this script
#   reports "Ready for SSH", on the laptop:
#
#     truss train isession --job-id <ID>      # prints ssh-creds
#     ssh -L 8000:localhost:8000 <baseten-host>
#     export DOTS_OCR_BASE_URL=http://localhost:8000/v1
#     export DOTS_OCR_BASE_URL_KEY=EMPTY     # vLLM ignores the key
#     realdoc-bench extract run --dataset v0 -p dots_mocr -a claude_sonnet_4_6
#
# Stop with: `truss train stop --job-id <ID>`  (or wait $SLEEP_HOURS).

set -euo pipefail

HF_MODEL="${HF_MODEL:?HF_MODEL env var is required}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
SLEEP_HOURS="${SLEEP_HOURS:-4}"
PORT="${PORT:-8000}"

echo "=========================================="
echo "realdoc-bench OCR host  —  $HF_MODEL"
echo "  max_model_len=$MAX_MODEL_LEN  gpu_mem_util=$GPU_MEM_UTIL"
echo "  extra: $EXTRA_VLLM_ARGS"
echo "  hold-open: ${SLEEP_HOURS}h on :$PORT"
echo "=========================================="

# vllm/vllm-openai already has `vllm` on PATH; install only if missing
# (e.g. when the user overrode the base image with a plain PyTorch one).
if ! command -v vllm >/dev/null 2>&1; then
  echo "[boot] vllm CLI not present — pip install vllm"
  pip install --no-cache-dir vllm
fi

mkdir -p /tmp/vllm
LOG=/tmp/vllm/serve.log

# shellcheck disable=SC2086  # EXTRA_VLLM_ARGS is intentionally word-split
vllm serve "$HF_MODEL" \
  --port "$PORT" \
  --served-model-name "$HF_MODEL" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  $EXTRA_VLLM_ARGS \
  >"$LOG" 2>&1 &

VLLM_PID=$!
echo "[boot] vllm pid=$VLLM_PID  logging to $LOG"

# Wait up to 15 min for /health to come up
for i in $(seq 1 90); do
  if curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[boot] vLLM ready after ${i}0s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[boot] vllm died during startup — last 50 log lines:"
    tail -n 50 "$LOG" || true
    exit 1
  fi
  sleep 10
done

if ! curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "[boot] vLLM did not become healthy within 15min — log tail:"
  tail -n 50 "$LOG" || true
  exit 1
fi

echo "=========================================="
echo "Ready for SSH."
echo "  On laptop:  truss train isession --job-id \$JOB_ID"
echo "              ssh -L $PORT:localhost:$PORT <baseten-host-from-isession>"
echo "  Then point realdoc-bench at http://localhost:$PORT/v1"
echo "=========================================="

# Keep the container alive for the requested window. The interactive session
# can extend its own timeout via `truss train isession --update-timeout`.
SECS=$(( SLEEP_HOURS * 3600 ))
echo "[hold] sleeping ${SECS}s (vllm pid $VLLM_PID still serving)"
sleep "$SECS"

# Clean shutdown if we ever reach the end
echo "[shutdown] stopping vllm..."
kill -TERM "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
