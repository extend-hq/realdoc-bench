"""Baseten ``truss train`` config — read by ``truss train push train_config.py``.

This is *not* a model deployment (``truss push``) — it's a **training job**
that Baseten allocates an H200 (or H100) GPU for. The job's entrypoint is
``baseten/serve_vllm.sh``, which boots vLLM in OpenAI mode on :8000 and then
holds the container open for ``$SLEEP_HOURS``. realdoc-bench then drives the
endpoint from a laptop via the SSH tunnel that ``truss train isession`` opens.

Why this shape rather than ``truss push``:
  - ``truss push`` (model deployment) routes traffic through Baseten's gateway,
    which has a readiness handshake that disagrees with stock vLLM-OpenAI
    images for OCR models (see ``baseten/README.md``'s prior known-issue note).
  - ``truss train push`` skips the gateway. We hit vLLM directly through the
    SSH tunnel — same UX as ``ssh awsl4``, just on a real H200.
  - Scale-to-zero by ending the job (``truss train stop``) — no idle billing.

Parameters arrive as env vars set by ``baseten/deploy.py``:

  BASETEN_HF_MODEL          HF repo id, e.g. rednote-hilab/dots.mocr (required)
  BASETEN_MAX_MODEL_LEN     vLLM --max-model-len (default 16384)
  BASETEN_GPU_MEM_UTIL      vLLM --gpu-memory-utilization (default 0.90)
  BASETEN_EXTRA_VLLM_ARGS   extra args appended to `vllm serve` (default "")
  BASETEN_SLEEP_HOURS       how long to hold the container open (default 4)
  BASETEN_GPU_TYPE          H100 | H200 | A100 | A10G (default H200)
  BASETEN_GPU_COUNT         GPU count (default 1)
  BASETEN_BASE_IMAGE        Docker base image (default vllm/vllm-openai:latest)
  BASETEN_PROJECT_NAME      truss-train project name (default realdoc-bench-ocr)
"""

from __future__ import annotations

import os

from truss.base import truss_config
from truss_train import definitions

# ─ Parameters (set by baseten/deploy.py via env vars) ──────────────────────
hf_model = os.environ.get("BASETEN_HF_MODEL", "")
if not hf_model:
    raise RuntimeError("BASETEN_HF_MODEL is required")

max_model_len = int(os.environ.get("BASETEN_MAX_MODEL_LEN", "16384"))
gpu_mem_util = float(os.environ.get("BASETEN_GPU_MEM_UTIL", "0.90"))
extra_vllm_args = os.environ.get("BASETEN_EXTRA_VLLM_ARGS", "")
sleep_hours = int(os.environ.get("BASETEN_SLEEP_HOURS", "4"))
gpu_type_name = os.environ.get("BASETEN_GPU_TYPE", "H200")
gpu_count = int(os.environ.get("BASETEN_GPU_COUNT", "1"))
project_name = os.environ.get("BASETEN_PROJECT_NAME", "realdoc-bench-ocr")
base_image_name = os.environ.get("BASETEN_BASE_IMAGE", "vllm/vllm-openai:latest")

# ─ Image ───────────────────────────────────────────────────────────────────
# vllm-openai ships the `vllm` CLI + CUDA libs; serve_vllm.sh just runs it.
image = definitions.Image(base_image=base_image_name)

# ─ Compute ─────────────────────────────────────────────────────────────────
gpu_type = getattr(truss_config.Accelerator, gpu_type_name)
compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(accelerator=gpu_type, count=gpu_count),
    cpu_count=gpu_count * 8,
    memory="64Gi",
    node_count=1,
)

# ─ Runtime ─────────────────────────────────────────────────────────────────
# `serve_vllm.sh` reads HF_MODEL / MAX_MODEL_LEN / GPU_MEM_UTIL / EXTRA_VLLM_ARGS
# / SLEEP_HOURS / PORT from the env. We also mount HF_HOME into the cache
# volume so a re-run of the same model is a cache hit.
runtime = definitions.Runtime(
    environment_variables={
        "HF_MODEL": hf_model,
        "MAX_MODEL_LEN": str(max_model_len),
        "GPU_MEM_UTIL": str(gpu_mem_util),
        "EXTRA_VLLM_ARGS": extra_vllm_args,
        "SLEEP_HOURS": str(sleep_hours),
        "PORT": "8000",
        # HF_TOKEN is only needed for gated repos. The presets we ship (dots,
        # GLM-OCR, PaddleOCR-VL) are public so this is harmless when missing.
        "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
        mount_base_path="/root/.cache",
        require_cache_affinity=False,
    ),
)

# ─ Interactive session ─────────────────────────────────────────────────────
# ON_STARTUP makes Baseten provision SSH access as soon as the job is up,
# so `truss train isession --job-id <ID>` returns creds immediately. The
# laptop forwards :8000 through that SSH tunnel to reach vLLM.
interactive_session = definitions.InteractiveSession(
    trigger=definitions.InteractiveSessionTrigger.ON_STARTUP,
    timeout_minutes=sleep_hours * 60,
)

# ─ Job ─────────────────────────────────────────────────────────────────────
training_job = definitions.TrainingJob(
    image=image,
    compute=compute,
    runtime=runtime,
    interactive_session=interactive_session,
    name=f"ocr-{hf_model.split('/')[-1].replace('.', '-').replace('_', '-').lower()}",
)

# ─ Project ─────────────────────────────────────────────────────────────────
# `truss train push` reads the module-level `project` symbol.
project = definitions.TrainingProject(name=project_name, job=training_job)
