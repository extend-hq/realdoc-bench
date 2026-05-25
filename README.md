<p align="center">
  <img src="docs/assets/realdocbench_logo.png" alt="RealDocBench — A Real-World Benchmark for Document Agents" width="100%">
</p>

# RealDocBench

A document parsing benchmark that measures two things on real documents:

1. **Layout quality** — bounding-box and block-type predictions versus human annotation, with F1, adjusted F1, mAP, and per-block-type breakdowns.
2. **QA extraction quality** — for each (question, parser) pair, an LLM extractor is asked to answer the question from the parser's markdown alone. Per-field scoring with `deep_equal` against a typed-JSON gold (with a conservative fuzzy fall-back for plain-string fields).

Each mode reports cost and latency alongside quality.

## Full Leaderboard (1,359 questions)

| Parser | Per-field Accuracy | Per-question Accuracy |
| --- | ---: | ---: |
| **Extend v2 - Performance** | **95.7%** | **90.3%** |
| LlamaParse (Agentic) | 92.1% | 84.3% |
| Reducto (Agentic) | 91.1% | 83.1% |
| Extend v1 | 90.4% | 81.8% |
| Gemini 3.5 Flash | 89.0% | 81.6% |
| LlamaParse | 89.0% | 80.4% |
| Azure DI | 88.8% | 78.9% |
| Reducto | 88.5% | 80.2% |
| AWS Textract | 70.5% | 53.6% |

## Installation
```bash
uv sync
```

## API Keys

All secrets load from `.env` at the repo root (auto-loaded by the CLI). Keys needed per command:

- `score` — `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- `parse` — whatever the parsers require: `ANTHROPIC_API_KEY`, `EXTEND_API_KEY`, `LLAMA_CLOUD_API_KEY`, AWS / Azure creds, …
- `download` and the layout dataset fetch — `HF_TOKEN` (for the private datasets)

**Dataset overrides** — point `REALDOC_BENCH_DATASET_EXTEND_AI_REALDOCBENCH_LAYOUT` at a local snapshot to skip the HF download.

## QA-extraction benchmark — `evaluate`

The full pipeline is download → parse → score → report. Every stage is per-parser scoped, cached, and reusable.

```bash
# 1. Pull the bench dataset (qa_bank.json + docs/) from the HF Hub
realdoc-bench evaluate download --run-dir runs/qa --dataset Extend-AI/Realdoc-Bench

# 2. Parse every PDF with one or more parsers (idempotent; --force to re-parse)
realdoc-bench evaluate parse --run-dir runs/qa -p reducto -p aws_textract -p azure_di

# 3. Score every (question × parser) pair with Gemini deep_equal
#    No -p means: score every parser found under <run-dir>/parses/
realdoc-bench evaluate score --run-dir runs/qa

# 4. Build the dashboard
realdoc-bench evaluate report --run-dir runs/qa
```

Or all at once for a known parser set:

```bash
realdoc-bench evaluate run --run-dir runs/qa -p reducto -p aws_textract
```

**Run-dir layout**:

```
runs/qa/
  docs/                          # input — documents (PDFs)
  qa_bank.json                   # input — typed-template QA bank
  parses/<parser>/<stem>.md      # parser output (+ <stem>.json meta)
  eval/cache/<qid>__<parser>.json   # per-(question, parser) scoring cache
  eval/results.json              # flat snapshot rebuilt by `report`
  dashboard.html                 # the dashboard
```

**Cache semantics**:

- `parse` skips `(parser, doc)` when `.md` + `.json` exist and the json's `ok=true`. `--force` busts the cache for the listed parsers only.
- `score` skips `(qid, parser)` when the cache file exists. **On hit, the cached answer is re-scored with the current scoring code**, so editing scoring thresholds + re-running `score` updates verdicts without spending a single API call. `--force` re-calls the model.
- Per-parser scope is strict: scoring `-p X` never touches any other parser's cache files.

**API keys** — see [API Keys](#api-keys). `score` needs `GEMINI_API_KEY`, `parse` needs whatever the parsers require, `download` needs `HF_TOKEN`.

## Layout benchmark — `layout`

```bash
# 1. Run one or more processors over the layout dataset and score every page
realdoc-bench layout eval --dataset Extend-AI/RealDocBench-Layout --processor gt_self --limit 5

# 2. Build the markdown + Pareto-plot report from a previous run
realdoc-bench layout report --run-id <id>
```

`--dataset` defaults to `Extend-AI/RealDocBench-Layout` and can be omitted; pass a different HF repo id to point at another snapshot. Set `HF_TOKEN` for private repos.

### Scorer — `--scorer {f1,adjacency}`

Two scorers are wired into `layout eval`; both build on a Hungarian matcher.

- **`f1`** (default) — Hungarian on `1 − IoU`, IoU ≥ 0.5, wrong-type pair → FP + FN. Backwards-compatible with the legacy leaderboard.
- **`adjacency`** — Hungarian on `1 − IoU + 0.25·type` (wrong-type pair stays TP, surfaced as `misclassifications`) with adjacency split/merge recovery; reports an additional **adjusted F1** that lets the matcher merge adjacent same-type fragments to recover from over/under-segmentation. See [`realdoc_bench/layout/metrics/adjacency/`](realdoc_bench/layout/metrics/adjacency/).

```bash
realdoc-bench layout eval -p extend_v2_0_0_beta --limit 50 --scorer adjacency
```

The adjacency scorer also has a Python API and a standalone re-score CLI:

```python
from realdoc_bench.layout.metrics.adjacency import score_pairs
result = score_pairs(pairs, iou_threshold=0.5, gap_threshold=0.02)   # pairs := [(pred_blocks, gt_blocks, page_w, page_h), ...]
print(result.strict.micro_f1, result.adjusted.micro_f1)
```

```bash
python -m realdoc_bench.layout.metrics.adjacency.rescore --run-id <id> --merge-profile extended --ignore-mode exclude_matching
```

### Assumptions

- **Dataset:** [`Extend-AI/RealDocBench-Layout`](https://huggingface.co/datasets/Extend-AI/RealDocBench-Layout) (1,500 pages, private) by default; override with `--dataset <repo-id>`. Manifest + image + COCO blobs all come from HF on first use (set `HF_TOKEN`, or `REALDOC_BENCH_DATASET_EXTEND_AI_REALDOCBENCH_LAYOUT` to a local snapshot).
- **Matching:** axis-aligned IoU, threshold 0.5 (`--iou-threshold` to change). No confidence sweep — every prediction counts positive at IoU ≥ 0.5; mAP is computed separately when `confidence_present=True`.
- **Adjusted F1 (adjacency only):** after the 1:1 match, unmatched same-type preds with gap < `0.02 × max(page_w, page_h)` are merged and re-scored; symmetric for unmatched GTs.
- **Cost & latency:** each processor reports `cost_estimate_usd` / `latency_sec` per page when available; aggregates carry `$/page` and `latency_sec/page`, and the snapshot Pareto plots use them.

## Datasets

Distributed via the [Hugging Face Hub](https://huggingface.co/Extend-AI) under the `Extend-AI` org. Private during development; gated by a one-time legal review for the public sets.

- [`Extend-AI/RealDocBench-Layout`](https://huggingface.co/datasets/Extend-AI/RealDocBench-Layout) — 1,500-page layout benchmark with COCO annotations.
- [`Extend-AI/Realdoc-Bench`](https://huggingface.co/datasets/Extend-AI/Realdoc-Bench) — QA-extraction bench (1,359 questions × 581 documents, private). What `evaluate download` pulls.

The layout set ships its `manifest.csv` (page list + per-page metadata) alongside `images/` and `annotations/` on HF.

## License

Code: Apache-2.0. Schemas and ground-truth answers: CC-BY-4.0. Source documents in `v1-public/sources/` carry per-doc licenses tracked in `SOURCES.md`.

## Adding a provider

See [`docs/adding_a_parser.md`](docs/adding_a_parser.md) (extraction) and [`docs/adding_a_layout_processor.md`](docs/adding_a_layout_processor.md). Each is roughly 150 LOC + a smoke test.
