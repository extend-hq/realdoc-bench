# realdoc-bench

A document parsing benchmark that measures two things on real documents:

1. **Layout quality** — bounding-box and block-type predictions versus human annotation, with F1, adjusted F1, mAP, and per-block-type breakdowns.
2. **Extraction quality** — downstream JSON extraction accuracy when an LLM agent (Claude, Gemini, ...) is given only the parsed markdown plus a schema. This proxies for "did the parse preserve enough information to be useful?"

Each mode reports cost and latency alongside quality, and emits Pareto-frontier plots so it's easy to see which providers actually dominate.

## Install

```bash
uv sync
```

## Quickstart

```bash
# Layout
realdoc-bench layout list
realdoc-bench layout eval --processor gt_self --limit 5
realdoc-bench layout report --run-id <id>

# Extraction
realdoc-bench extract list datasets
realdoc-bench extract run --dataset v0 --parser pymupdf --agent claude_sonnet_4_6 --doc-id fin_10k_001
realdoc-bench extract report --run-id <id>
```

## Datasets

Datasets are distributed via the [Hugging Face Hub](https://huggingface.co/extend-ai) under the `extend-ai` org. They are **private by default** during development; the `v1-public` extraction set and the public layout set are gated by a one-time legal review.

- `extend-ai/realdoc-layout` — 1,500-page layout benchmark with COCO annotations.
- `extend-ai/realdoc-extract-v0` — internal dogfood extraction set.
- `extend-ai/realdoc-extract-v1-public` — public extraction set (TBD).

The list of pages and per-page metadata for the layout set ships in `datasets/layout/manifest.csv`; image and annotation blobs live on HF.

## License

Code: Apache-2.0. Schemas and ground-truth answers: CC-BY-4.0. Source documents in `v1-public/sources/` carry per-doc licenses tracked in `SOURCES.md`.

## Adding a provider

See [`docs/adding_a_parser.md`](docs/adding_a_parser.md) (extraction) and [`docs/adding_a_layout_processor.md`](docs/adding_a_layout_processor.md). Each is roughly 150 LOC + a smoke test.
