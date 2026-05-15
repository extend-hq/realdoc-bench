#!/usr/bin/env python
"""Spin up a realdoc-bench OCR/layout model on a Baseten H200 via ``truss train``.

This replaces the previous ``truss push`` (model-deployment) path. Rationale:
``truss push`` routes through Baseten's gateway, whose readiness handshake
disagrees with the stock ``vllm/vllm-openai`` image for these OCR models
(predict requests never reach the container). ``truss train`` skips the
gateway — the job is an SSH-accessible H200 box running vLLM directly on
:8000, and realdoc-bench's adapters reach it via a local SSH tunnel.

Flow:
  1. ``baseten/deploy.py --preset dots_mocr`` sets the BASETEN_* env vars,
     invokes ``truss train push baseten/train_config.py --entrypoint
     "bash baseten/serve_vllm.sh"``.
  2. Baseten allocates an H200, runs ``serve_vllm.sh`` → boots vLLM on :8000,
     prints "Ready for SSH", sleeps.
  3. On the laptop:
        truss train isession --job-id <ID>
        ssh -L 8000:localhost:8000 <baseten-host>
        export DOTS_OCR_BASE_URL=http://localhost:8000/v1
        export DOTS_OCR_BASE_URL_KEY=EMPTY
        realdoc-bench extract run --dataset v0 -p dots_mocr -a claude_sonnet_4_6
  4. ``truss train stop --job-id <ID>`` when done (or wait for sleep to expire).

Prereqs:
  - ``truss`` CLI on PATH (``pipx install truss``; this repo's box has 0.14.1).
  - ``BASETEN_API_KEY`` exported, or ``truss login`` already done.
  - For gated HF repos: an ``hf_access_token`` secret in Baseten (Workspace →
    Secrets); the train config maps it to ``HF_TOKEN``. The three shipped
    presets (dots_mocr / glm_ocr / paddle_ocr_vl) are public — no token needed.

Examples:
  python baseten/deploy.py --preset dots_mocr                  # H200, 4h
  python baseten/deploy.py --preset glm_ocr --hours 2
  python baseten/deploy.py --hf-model rednote-hilab/dots.mocr \\
      --accelerator H100 --max-model-len 16384
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from baseten.presets import PRESETS

_TRAIN_CONFIG = Path(__file__).parent / "train_config.py"
_ENTRYPOINT = "bash baseten/serve_vllm.sh"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Curated model preset.")
    parser.add_argument("--hf-model", help="HF repo id (overrides --preset's hf_model).")
    parser.add_argument(
        "--accelerator",
        default="H200",
        help="Baseten GPU type (A10G, A100, H100, H200). Default H200.",
    )
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument(
        "--base-image",
        default="vllm/vllm-openai:latest",
        help="Docker base image — must ship the `vllm` CLI. Default vllm/vllm-openai:latest.",
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument(
        "--extra-vllm-args",
        default=None,
        help="Extra args appended to `vllm serve` (overrides preset).",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=4,
        help="Hours to hold the container open before auto-shutdown.",
    )
    parser.add_argument(
        "--project-name",
        default="realdoc-bench-ocr",
        help="Truss-train project name (groups jobs in the Baseten UI).",
    )
    parser.add_argument(
        "--tail", action="store_true", help="Pass --tail to `truss train push` (stream logs)."
    )
    args = parser.parse_args()

    if not args.preset and not args.hf_model:
        parser.error("provide --preset or --hf-model")

    preset = dict(PRESETS.get(args.preset, {})) if args.preset else {}
    hf_model = args.hf_model or preset.get("hf_model")
    if not hf_model:
        parser.error("could not resolve an HF model id")

    extra_args = (
        args.extra_vllm_args
        if args.extra_vllm_args is not None
        else str(preset.get("extra_vllm_args", ""))
    )

    truss = shutil.which("truss")
    if not truss:
        print("ERROR: `truss` not on PATH. `pipx install truss` (or use mise).", file=sys.stderr)
        return 2

    env = {
        **os.environ,
        "BASETEN_HF_MODEL": str(hf_model),
        "BASETEN_MAX_MODEL_LEN": str(args.max_model_len),
        "BASETEN_GPU_MEM_UTIL": str(args.gpu_mem_util),
        "BASETEN_EXTRA_VLLM_ARGS": extra_args,
        "BASETEN_SLEEP_HOURS": str(args.hours),
        "BASETEN_GPU_TYPE": args.accelerator,
        "BASETEN_GPU_COUNT": str(args.gpu_count),
        "BASETEN_BASE_IMAGE": args.base_image,
        "BASETEN_PROJECT_NAME": args.project_name,
    }

    print(
        f"\nLaunching truss-train job for {hf_model} on {args.accelerator}×{args.gpu_count} "
        f"(hold-open {args.hours}h, project '{args.project_name}'):\n",
        file=sys.stderr,
    )

    cmd = [
        truss, "train", "push", str(_TRAIN_CONFIG),
        "--entrypoint", _ENTRYPOINT,
        "--accelerator", f"{args.accelerator}:{args.gpu_count}",
        "--interactive", "on_startup",
        "--interactive-timeout-minutes", str(args.hours * 60),
    ]
    if args.tail:
        cmd.append("--tail")

    print(f"$ {' '.join(cmd)}\n", file=sys.stderr)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        return result.returncode

    env_var = preset.get("env_var", f"OCR_BASE_URL")
    print(
        "\n────────────────────────────────────────────────────────────────\n"
        "Once the job logs say 'Ready for SSH', on this laptop:\n\n"
        "  truss train isession --job-id <JOB_ID>           # prints SSH creds\n"
        "  ssh -L 8000:localhost:8000 <baseten-ssh-host>     # tunnel vLLM\n\n"
        f"Then point realdoc-bench at the tunnel:\n\n"
        f'  export {env_var}=http://localhost:8000/v1\n'
        f'  export {env_var}_KEY=EMPTY                       # vLLM ignores the key\n'
        "\nStop with:  truss train stop --job-id <JOB_ID>\n"
        "────────────────────────────────────────────────────────────────\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
