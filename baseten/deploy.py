#!/usr/bin/env python
"""Deploy a realdoc-bench OCR / layout model to Baseten as an OpenAI-compatible
vLLM endpoint.

Renders ``baseten/vllm_truss/config.yaml`` for the requested HF model + GPU,
drops it into a temp Truss directory, and runs ``truss push``. Baseten then
serves the model at an OpenAI-compatible URL — point realdoc-bench's existing
adapters at it via the ``*_BASE_URL`` / ``*_BASE_URL_KEY`` env vars (the
adapters already build an ``openai.OpenAI(base_url=..., api_key=...)`` client).

Prereqs:
  - ``truss`` CLI on PATH (``pipx install truss`` or mise).
  - ``BASETEN_API_KEY`` exported, or ``truss login`` already done.
  - For gated HF repos, a ``hf_access_token`` secret configured in Baseten
    (Workspace → Secrets) — the rendered config references it.

Examples:
  # dots.mocr on an H100
  python baseten/deploy.py --preset dots_mocr

  # GLM-OCR on H200, custom name
  python baseten/deploy.py --preset glm_ocr --accelerator H200 --name glm-ocr-rdb

  # arbitrary HF model
  python baseten/deploy.py --hf-model rednote-hilab/dots.mocr --name dots-mocr \\
      --accelerator H100 --max-model-len 16384

After ``truss push`` finishes, Baseten prints the model URL. Use it like:
  export DOTS_OCR_BASE_URL="https://model-XXXX.api.baseten.co/environments/production/sync/v1"
  export DOTS_OCR_BASE_URL_KEY="$BASETEN_API_KEY"
  realdoc-bench layout eval -p dots_ocr ...
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_TEMPLATE = Path(__file__).parent / "vllm_truss" / "config.yaml"

# Curated presets for the OCR / layout models realdoc-bench benchmarks. Each
# entry maps a short name → (HF repo, vLLM args). GPU defaults to H100; override
# with --accelerator. ``env_var`` is the realdoc-bench env var to point at the
# resulting endpoint.
PRESETS: dict[str, dict[str, object]] = {
    "dots_mocr": {
        "hf_model": "rednote-hilab/dots.mocr",
        "max_model_len": 16384,
        "gpu_mem_util": 0.90,
        "extra_vllm_args": "",
        "env_var": "DOTS_OCR_BASE_URL",
    },
    "glm_ocr": {
        "hf_model": "zai-org/GLM-OCR",
        "max_model_len": 16384,
        "gpu_mem_util": 0.90,
        "extra_vllm_args": "--quantization fp8 --kv-cache-dtype fp8_e4m3",
        "env_var": "GLM_OCR_BASE_URL",
    },
    "paddle_ocr_vl": {
        # PaddleOCR-VL is served through vLLM in recent PaddleOCR releases.
        "hf_model": "PaddlePaddle/PaddleOCR-VL",
        "max_model_len": 16384,
        "gpu_mem_util": 0.90,
        "extra_vllm_args": "",
        "env_var": "PADDLE_OCR_VL_BASE_URL",
    },
}


def _render(template: str, values: dict[str, object]) -> str:
    out = template
    for key, val in values.items():
        out = out.replace("{{ " + key + " }}", str(val))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Curated model preset.")
    parser.add_argument("--hf-model", help="HF repo id (overrides / replaces --preset).")
    parser.add_argument("--name", help="Truss/Baseten model name. Defaults to a slug of the HF id.")
    parser.add_argument("--accelerator", default="H100", help="Baseten GPU (A10G, A100, H100, H200). Default H100.")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--extra-vllm-args", default="", help="Extra args appended to `vllm serve`.")
    parser.add_argument("--predict-concurrency", type=int, default=32)
    parser.add_argument("--hf-token-secret", default="hf_access_token",
                        help="Name of the Baseten secret holding the HF token (referenced in config).")
    parser.add_argument("--dry-run", action="store_true", help="Render the truss but don't `truss push`.")
    parser.add_argument("--publish", action="store_true", help="Pass --publish to `truss push` (production deploy).")
    args = parser.parse_args()

    if not args.preset and not args.hf_model:
        parser.error("provide --preset or --hf-model")

    preset = dict(PRESETS.get(args.preset, {})) if args.preset else {}
    hf_model = args.hf_model or preset.get("hf_model")
    if not hf_model:
        parser.error("could not resolve an HF model id")

    name = args.name or str(hf_model).split("/")[-1].replace(".", "-").replace("_", "-").lower()
    volume_folder = name.replace("-", "_")

    values = {
        "model_name": name,
        "hf_model": hf_model,
        "accelerator": args.accelerator,
        "max_model_len": args.max_model_len,
        "gpu_mem_util": args.gpu_mem_util,
        "extra_vllm_args": args.extra_vllm_args or preset.get("extra_vllm_args", ""),
        "predict_concurrency": args.predict_concurrency,
        "hf_token_secret": args.hf_token_secret,
        "volume_folder": volume_folder,
    }
    # Preset can override numeric defaults if the user didn't pass them explicitly.
    for k in ("max_model_len", "gpu_mem_util"):
        if args.preset and f"--{k.replace('_', '-')}" not in sys.argv and k in preset:
            values[k] = preset[k]

    rendered = _render(_TEMPLATE.read_text(), values)

    work = Path(tempfile.mkdtemp(prefix=f"truss-{name}-"))
    (work / "config.yaml").write_text(rendered)
    print(f"Rendered truss → {work}/config.yaml\n", file=sys.stderr)
    print(rendered, file=sys.stderr)

    if args.dry_run:
        print(f"\n[dry-run] truss dir left at {work}", file=sys.stderr)
        return 0

    truss = shutil.which("truss")
    if not truss:
        print("ERROR: `truss` not on PATH. `pipx install truss` (or use mise).", file=sys.stderr)
        return 2

    cmd = [truss, "push", str(work)]
    if args.publish:
        cmd.append("--publish")
    print(f"\n$ {' '.join(cmd)}\n", file=sys.stderr)
    result = subprocess.run(cmd)
    if result.returncode == 0:
        env_var = preset.get("env_var", f"{name.upper().replace('-', '_')}_BASE_URL")
        print(
            f"\nDeployed. Once Baseten reports the model URL, wire it up:\n"
            f"  export {env_var}=\"https://model-XXXX.api.baseten.co/environments/production/sync/v1\"\n"
            f"  export {env_var}_KEY=\"$BASETEN_API_KEY\"\n",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
