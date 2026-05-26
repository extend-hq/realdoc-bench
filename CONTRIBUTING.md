# Contributing

Thanks for your interest in realdoc-bench. This guide covers the two most common contributions: adding a **layout processor** (for `layout eval`) and adding a **parse provider** (for `evaluate parse`).

## Dev setup

```bash
uv sync
uv run pytest
uv run ruff check
```

Secrets live in `.env` (or `.env.local`, gitignored). Only the parsers/processors you exercise need credentials.

## Adding a layout processor

A layout processor takes a single page image and returns a `LayoutDocument` of predicted blocks. The runner handles caching, scoring, and reporting — your job is just the predict step.

**1. Create a new module under `realdoc_bench/layout/processors/`.** Subclass `LayoutProcessor` and register it with `@register_layout_processor("<name>", version="<semver-or-tag>")`. Implement `predict(image_path, *, gt=None) -> ProcessorResult`.

Skeleton:

```python
# realdoc_bench/layout/processors/myproc.py
from __future__ import annotations

import time
from pathlib import Path

from realdoc_bench.layout.normalizers.base import (
    BBox, LayoutBlock, LayoutDocument, LayoutPage,
)
from realdoc_bench.layout.processors.base import (
    LayoutProcessor, ProcessorResult, register_layout_processor,
)
from realdoc_bench.shared.pricing.meter import parse_cost


@register_layout_processor("myproc", version="1.0.0")
class MyProcessor(LayoutProcessor):
    def predict(self, image_path: Path, *, gt: LayoutDocument | None = None) -> ProcessorResult:
        t0 = time.perf_counter()
        # ... call your provider, get back blocks ...
        blocks = [
            LayoutBlock(
                id="block_0001",
                block_type="text",          # see vocab below
                bbox=BBox(x=10, y=20, w=300, h=40),
                text="hello",
            ),
        ]
        document = LayoutDocument(
            document_id=image_path.stem,
            source=image_path.name,
            pages=[LayoutPage(page_number=1, width=W, height=H, blocks=blocks)],
        )
        return ProcessorResult(
            document=document,
            latency_sec=time.perf_counter() - t0,
            cost_estimate_usd=parse_cost(self.name, pages=1),
            pages_processed=1,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
```

**2. Register the module for side-effect import** in `realdoc_bench/layout/processors/__init__.py` — wrap the import in `try/except ImportError` so the package stays usable when an optional SDK isn't installed:

```python
try:
    from realdoc_bench.layout.processors import myproc  # noqa: F401
except ImportError:
    pass
```

**3. Normalize block types into the 9-class public vocab** (`text`, `heading`, `section_heading`, `header`, `footer`, `page_number`, `figure`, `table`, `key_value`). Anything outside that set will fail Pydantic validation. The typical pattern is a `dict[str, LayoutBlockType]` mapping the vendor's labels to public classes, with a fallback to `"text"`.

**4. BBox is in pixels** in the source image's coordinate space — convert from normalized 0–1 with `int(round(value * width))` before constructing `BBox`.

**5. (Optional) Pricing.** If you want `$/page` to render in the report, add an entry to the pricing catalog and read it via `parse_cost(self.name, pages=...)`. Otherwise pass `cost_estimate_usd=None`.

**6. Verify.** Smoke-test against the trivial baseline path:

```bash
uv run realdoc-bench layout list                    # confirm "myproc" shows up
uv run realdoc-bench layout eval -p myproc --limit 2
```

## Adding a parse provider

A parse provider takes a PDF and returns markdown. The scorer only sees the markdown — JSON-emitting providers must serialize internally inside the adapter.

**1. Create a new module under `realdoc_bench/evaluate/parsers/`.** Subclass `ParseProvider` and register it with `@register_parser("<name>", version="<tag>")`. Implement `parse(pdf_path, *, cache_dir=None) -> ParseResult`.

Skeleton:

```python
# realdoc_bench/evaluate/parsers/myparser.py
from __future__ import annotations

import time
from pathlib import Path

from realdoc_bench.evaluate.parsers.base import (
    ParseProvider, ParseResult, register_parser,
)
from realdoc_bench.shared.pricing.meter import parse_cost


@register_parser("myparser", version="v1")
class MyParser(ParseProvider):
    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        t0 = time.perf_counter()
        # ... call your provider, get markdown back ...
        markdown = "# Title\n\nbody…"
        pages = 1
        return ParseResult(
            markdown=markdown,
            page_count=pages,
            latency_sec=time.perf_counter() - t0,
            cost_estimate_usd=parse_cost(self.name, pages=pages),
            pages_processed=pages,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
        )
```

**2. Register for side-effect import** in `realdoc_bench/evaluate/parsers/__init__.py`, wrapped in `try/except ImportError` like the others.

**3. Markdown is the whole contract.** Anything you want the scoring model to see — headings, tables, key/values — has to be in the markdown string. Tables can be markdown or HTML; the scorer handles both.

**4. (Optional) Override `config_hash`** if your adapter has knobs (mode, model name, prompt) whose changes should bust the parse cache. Mixing a content nonce into the hash is enough — any change to it invalidates cached parses.

**5. Verify.**

```bash
uv run realdoc-bench evaluate list                  # confirm "myparser" shows up
uv run realdoc-bench evaluate parse --run-dir runs/qa -p myparser --limit 2
uv run realdoc-bench evaluate score --run-dir runs/qa -p myparser --limit 2
```

## Tests, lint, and PRs

- `uv run pytest` — the suite is small and fast; please keep it green.
- `uv run ruff check` — match the project style; the existing Typer `B008` warnings are expected.
- Open a PR against `main`. Keep commits focused; describe the why in the PR body, not the commit message.
