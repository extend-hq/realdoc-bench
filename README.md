# realdoc-bench

A document parsing benchmark that measures two things on real documents:

1. **Layout quality** — bounding-box and block-type predictions versus human annotation, with F1, adjusted F1, mAP, and per-block-type breakdowns.
2. **QA extraction quality** — for each (question, parser) pair, an LLM extractor is asked to answer the question from the parser's markdown alone. Per-field scoring with `deep_equal` against a typed-JSON gold (with a conservative fuzzy fall-back for plain-string fields).

Each mode reports cost and latency alongside quality.

## Install

```bash
uv sync
```

## QA-extraction benchmark — `evaluate`

The full pipeline is download → parse → score → report. Every stage is per-parser scoped, cached, and reusable.

```bash
# 1. Pull the bench dataset (bank.json + pdfs/) from the HF Hub
realdoc-bench evaluate download --run-dir runs/v6 --dataset extend-ai/realdoc-bench-qa

# 2. Parse every PDF with one or more parsers (idempotent; --force to re-parse)
realdoc-bench evaluate parse --run-dir runs/v6 -p reducto -p aws_textract -p azure_di

# 3. Score every (question × parser) pair with Gemini deep_equal
#    No -p means: score every parser found under <run-dir>/parses/
realdoc-bench evaluate score --run-dir runs/v6

# 4. Build the dashboard
realdoc-bench evaluate report --run-dir runs/v6
```

Or all at once for a known parser set:

```bash
realdoc-bench evaluate run --run-dir runs/v6 -p reducto -p aws_textract
```

**Run-dir layout**:

```
runs/v6/
  pdfs/                          # input — PDFs
  bank.json                      # input — typed-template QA bank
  parses/<parser>/<stem>.md      # parser output (+ <stem>.json meta)
  eval/cache/<qid>__<parser>.json   # per-(question, parser) scoring cache
  eval/results.json              # flat snapshot rebuilt by `report`
  dashboard.html                 # the dashboard
```

**Cache semantics**:

- `parse` skips `(parser, doc)` when `.md` + `.json` exist and the json's `ok=true`. `--force` busts the cache for the listed parsers only.
- `score` skips `(qid, parser)` when the cache file exists. **On hit, the cached answer is re-scored with the current scoring code**, so editing scoring thresholds + re-running `score` updates verdicts without spending a single API call. `--force` re-calls the model.
- Per-parser scope is strict: scoring `-p X` never touches any other parser's cache files.

**API keys** — drop them in `.env` at the repo root; the CLI loads it automatically. `score` needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`); `parse` needs whatever the parsers require (`ANTHROPIC_API_KEY`, `EXTEND_API_KEY`, `LLAMA_CLOUD_API_KEY`, AWS / Azure creds, ...); `download` needs `HF_TOKEN` for private datasets.

## Layout benchmark — `layout`

```bash
realdoc-bench layout list
realdoc-bench layout eval --processor gt_self --limit 5
realdoc-bench layout report --run-id <id>
```

## Datasets

Distributed via the [Hugging Face Hub](https://huggingface.co/extend-ai) under the `extend-ai` org. Private during development; gated by a one-time legal review for the public sets.

- `extend-ai/realdoc-layout` — 1,500-page layout benchmark with COCO annotations.
- `extend-ai/realdoc-bench-qa` — QA-extraction bench (TBD; what `evaluate download` pulls).

The list of pages and per-page metadata for the layout set ships in `datasets/layout/manifest.csv`; image and annotation blobs live on HF.

## License

Code: Apache-2.0. Schemas and ground-truth answers: CC-BY-4.0. Source documents in `v1-public/sources/` carry per-doc licenses tracked in `SOURCES.md`.

## Adding a provider

See [`docs/adding_a_parser.md`](docs/adding_a_parser.md) (extraction) and [`docs/adding_a_layout_processor.md`](docs/adding_a_layout_processor.md). Each is roughly 150 LOC + a smoke test.
