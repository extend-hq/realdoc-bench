# Baseten deployer

Host realdoc-bench's OCR / layout models on Baseten GPUs as **OpenAI-compatible
vLLM endpoints**, so the existing realdoc-bench adapters can hit them with no
code changes — just point the `*_BASE_URL` env vars at the Baseten URL.

## Why

The self-hosted models (dots.mocr, GLM-OCR, PaddleOCR-VL) need a GPU. `awsl4`
(one L4, 23 GiB) can hold one at a time and the SSH tunnel flaps. Baseten gives
a stable H100/H200 endpoint per model so the layout + extraction benchmarks can
run several in parallel without babysitting a tunnel.

## How it works

`deploy.py` renders [`vllm_truss/config.yaml`](vllm_truss/config.yaml) — a
Truss `docker_server` config that just runs `vllm serve <model>` — for the
requested HF model + GPU, then `truss push`es it. Baseten serves the result at
an OpenAI-compatible URL (`…/sync/v1/chat/completions`). realdoc-bench's
adapters already build an `openai.OpenAI(base_url=…, api_key=…)` client, so they
work against Baseten unchanged.

## Prereqs

- `truss` CLI on PATH — `pipx install truss` (or mise; this repo's box has 0.14.1).
- `BASETEN_API_KEY` exported, or `truss login` already done.
- Gated HF repos: add an `hf_access_token` secret in the Baseten workspace and
  wire it per the comment in `vllm_truss/config.yaml`. dots.mocr / GLM-OCR /
  PaddleOCR-VL are public — no token needed.

## Usage

```bash
# Curated presets (dots_mocr, glm_ocr, paddle_ocr_vl)
python baseten/deploy.py --preset dots_mocr --accelerator H100
python baseten/deploy.py --preset glm_ocr  --accelerator H200   # fp8 weights+KV

# Arbitrary HF model
python baseten/deploy.py --hf-model rednote-hilab/dots.mocr --name dots-mocr \
    --accelerator H100 --max-model-len 16384

# Inspect the rendered Truss without pushing
python baseten/deploy.py --preset dots_mocr --dry-run

# Production deploy (truss push --publish)
python baseten/deploy.py --preset dots_mocr --publish
```

After `truss push` finishes, Baseten prints the model URL. Wire it up:

```bash
export DOTS_OCR_BASE_URL="https://model-XXXX.api.baseten.co/environments/production/sync/v1"
export DOTS_OCR_BASE_URL_KEY="$BASETEN_API_KEY"

# Layout benchmark
realdoc-bench layout eval -p dots_ocr --run-id dots_baseten

# Extraction benchmark (dots.mocr as a parse provider)
realdoc-bench extract run --dataset v0 -p dots_mocr -a claude_sonnet_4_6
```

## Adapter ↔ env-var map

| Model | Layout processor | Extraction parser | Base-URL env var |
| --- | --- | --- | --- |
| dots.mocr | `dots_ocr` | `dots_mocr` | `DOTS_OCR_BASE_URL` (+ `_KEY`) |
| GLM-OCR | — | `glm_ocr` | `GLM_OCR_BASE_URL` (+ `_KEY`) |
| PaddleOCR-VL | `paddle_ocr_vl` | — | `PADDLE_OCR_VL_BASE_URL` (+ `_KEY`) |

The `_KEY` suffix env var is read by realdoc-bench's `OpenAIVisionParserBase` /
processor clients and passed as the OpenAI `api_key` — set it to your
`BASETEN_API_KEY`.

## Notes

- `predict_concurrency` (default 32) lets one replica batch many in-flight
  requests — vLLM handles the batching. Bump it for throughput, or add replicas
  in the Baseten UI / config.
- PaddleOCR-VL: the preset assumes the model is vLLM-servable (true for recent
  PaddleOCR-VL releases). If a given release isn't, fall back to the FastAPI
  wrapper in `~/paddle_ocr_vl_server.py` on a raw GPU box.
- Cost: an H100 replica is ~$0.06–0.10/min on Baseten; scale-to-zero keeps idle
  cost ~0. Tear down with `truss` or the Baseten UI when a benchmark finishes.
