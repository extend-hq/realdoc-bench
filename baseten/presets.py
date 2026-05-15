"""HF-model presets for the realdoc-bench OCR / layout deployments.

Used by both the (deprecated) ``deploy_push.py`` (truss-push deployment) path
and the current ``deploy.py`` (truss-train interactive-job) path.

Each preset maps a short name to an HF repo + vLLM args + the realdoc-bench
env var that should point at the resulting endpoint.
"""

from __future__ import annotations

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
        "hf_model": "PaddlePaddle/PaddleOCR-VL",
        "max_model_len": 16384,
        "gpu_mem_util": 0.90,
        "extra_vllm_args": "",
        "env_var": "PADDLE_OCR_VL_BASE_URL",
    },
}
