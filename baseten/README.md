# Baseten deployer

Host realdoc-bench's self-hosted OCR / layout models (dots.mocr, GLM-OCR,
PaddleOCR-VL) on a Baseten **H200** via ``truss train``, and drive them from
the laptop over an SSH tunnel. realdoc-bench's existing OpenAI-SDK adapters
hit ``http://localhost:8000/v1`` unchanged — just `export <MODEL>_BASE_URL`.

## Why ``truss train`` and not ``truss push``

This repo tried ``truss push`` (model deployments) first. Build + boot worked,
vLLM came up healthy in the container, but **Baseten's gateway** kept
returning `Model is unhealthy, it is not ready to make predictions` — the
gateway↔container readiness handshake disagrees with the stock
``vllm/vllm-openai`` image for OCR models, and predict requests never reach
the container.

``truss train`` is the right primitive here:

| Concern | ``truss push`` | ``truss train`` (this) |
|---|---|---|
| GPU | H100/H200 deployment replica | H100/H200 **job** |
| Path to vLLM | Baseten gateway → predict_endpoint | SSH tunnel → vLLM directly |
| Readiness gate | Baseten's handshake (the problem) | Our own `serve_vllm.sh` health-loop |
| Idle cost | Replica scale-to-zero | End the job (``truss train stop``) |
| Public URL | Yes (when healthy) | No — laptop-only via SSH-forward |

A training job is essentially a managed remote GPU box. We use it the same
way we use ``awsl4`` — vLLM listens on :8000, the laptop forwards :8000
through SSH, realdoc-bench points at ``http://localhost:8000/v1``. Just on
real H200 instead of an L4 (23 GiB → 141 GiB).

## What it does

`deploy.py` renders nothing — it just shells out to
``truss train push baseten/train_config.py`` with the right `BASETEN_*` env
vars + ``--entrypoint "bash baseten/serve_vllm.sh"`` and
``--interactive on_startup``. Baseten allocates the GPU, runs
``serve_vllm.sh`` (which `vllm serve`s your model on :8000, waits for
`/health`, and `sleep`s for the requested window). ``truss train isession``
then returns SSH creds; you forward the port and you're done.

## Prereqs

- `truss` CLI on PATH — `pipx install truss` (or mise; this repo's box has 0.14.1).
- `BASETEN_API_KEY` exported, or `truss login` already done.
- Gated HF repos: an `hf_access_token` secret in the Baseten workspace —
  `train_config.py` wires it into the container's `HF_TOKEN`. dots.mocr /
  GLM-OCR / PaddleOCR-VL are public — no token needed.

## Usage

```bash
# Curated presets (dots_mocr, glm_ocr, paddle_ocr_vl) on an H200 for 4 hours
python baseten/deploy.py --preset dots_mocr
python baseten/deploy.py --preset glm_ocr --hours 2
python baseten/deploy.py --preset paddle_ocr_vl --accelerator H100

# Arbitrary HF model
python baseten/deploy.py --hf-model rednote-hilab/dots.mocr \
    --accelerator H200 --max-model-len 16384 --hours 6

# Stream logs while pushing (omit and use `truss train logs --job-id ...` later)
python baseten/deploy.py --preset dots_mocr --tail
```

Once `deploy.py` returns and the job logs say `Ready for SSH`, follow the
printed instructions:

```bash
truss train isession --job-id <JOB_ID>          # prints SSH creds + host
ssh -L 8000:localhost:8000 <baseten-ssh-host>   # tunnel vLLM to laptop

# Then in another shell:
export DOTS_OCR_BASE_URL=http://localhost:8000/v1
export DOTS_OCR_BASE_URL_KEY=EMPTY              # vLLM ignores the key

# Layout benchmark
realdoc-bench layout eval -p dots_ocr --run-id dots_baseten

# Extraction benchmark (dots.mocr as a parse provider)
realdoc-bench extract run --dataset v0 -p dots_mocr -a claude_sonnet_4_6
```

When the run is done, stop the job:

```bash
truss train stop --job-id <JOB_ID>
```

## Adapter ↔ env-var map

| Model | Layout processor | Extraction parser | Base-URL env var |
| --- | --- | --- | --- |
| dots.mocr | `dots_ocr` | `dots_mocr` | `DOTS_OCR_BASE_URL` (+ `_KEY`) |
| GLM-OCR | — | `glm_ocr` | `GLM_OCR_BASE_URL` (+ `_KEY`) |
| PaddleOCR-VL | `paddle_ocr_vl` | — | `PADDLE_OCR_VL_BASE_URL` (+ `_KEY`) |

The `_KEY` suffix env var is read by realdoc-bench's `OpenAIVisionParserBase` /
processor clients and passed as the OpenAI `api_key`. For vLLM through the
tunnel, set it to anything non-empty (vLLM doesn't auth) — `EMPTY` is the
convention.

## Notes

- **`predict_concurrency`** is irrelevant on this path — there's no gateway.
  vLLM does continuous batching internally, so a single replica handles
  many in-flight requests directly.
- **Cost**: H200 jobs are ~$0.10–0.15/min on Baseten. The 4-hour default
  means worst-case ~$36/job; for a benchmark that's usually 30–90 min you'll
  spend $5–$20 and tear down with `truss train stop`.
- **HF cache** is mounted at `/root/.cache` via `CacheConfig` so re-runs of
  the same model on the same project skip the HF download.
- **The L4 fallback** (`ssh awsl4`) still works — same SSH-tunnel pattern,
  just one model at a time and limited to ~23 GiB.

## File layout

```
baseten/
├── deploy.py          # CLI: shells out to `truss train push`
├── train_config.py    # truss_train.TrainingProject + TrainingJob
├── serve_vllm.sh      # container entrypoint — vllm serve + health-wait + sleep
├── presets.py         # HF-model presets (dots_mocr, glm_ocr, paddle_ocr_vl)
└── vllm_truss/        # (deprecated) truss-push template — kept for reference
```
