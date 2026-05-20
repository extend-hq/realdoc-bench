#!/usr/bin/env python3
"""Schema-grounded extraction across all parser outputs in a sweep.

For each (parser, doc, schema_type) triple:
  - load schema from datasets/extraction/v1.1/_schemas/<type>.json
  - load parsed markdown from runs/parse_sweep/<ts>/<parser>/<dom>/<type>/<f>.md
  - coerce schema to plain JSON-Schema (strip Extend x-* extensions)
  - call Claude with tool-use, where tool input_schema = the coerced schema
  - parse tool_use output → structured JSON
  - save to runs/parse_sweep/<ts>/<parser>/_extractions/<dom>/<type>/<f>.json

Parallelizes across (parser, doc) pairs. Skips done extractions (idempotent).

Usage:
  python -u scripts/extract_from_parses.py --sweep-dir runs/parse_sweep/<ts>
  python -u scripts/extract_from_parses.py --sweep-dir … --parser azure_di
  python -u scripts/extract_from_parses.py --sweep-dir … --limit 5  # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
SCHEMAS_DIR = V1_1 / "_schemas"


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k):
            os.environ[k] = v
load_dotenv()


_anthropic = None
def get_anthropic():
    global _anthropic
    if _anthropic is None:
        from anthropic import Anthropic
        _anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic


def coerce_schema(schema: dict) -> dict:
    """Use realdoc_bench's schema coercer to strip Extend x-* extensions."""
    from realdoc_bench.extraction.schema.dialect import to_plain_jsonschema
    coerced = to_plain_jsonschema(schema)
    # Tool input_schema must be a top-level object; strip $schema/title
    coerced.pop("$schema", None)
    # Anthropic tool input_schema requires `type: object` at root
    if coerced.get("type") != "object":
        coerced["type"] = "object"
    return coerced


def truncate_md(md: str, limit: int = 60000) -> str:
    """Truncate to fit in context. 60k chars ≈ 15k tokens, leaves room for schema."""
    if len(md) <= limit:
        return md
    head = limit * 2 // 3
    tail = limit - head
    return md[:head] + "\n\n…[truncated middle]…\n\n" + md[-tail:]


EXTRACTION_PROMPT = """You are a document extraction assistant. Given the parsed markdown of a document and a structured schema (provided as the tool's input schema), call the `extract_fields` tool with values for every field you can determine from the markdown.

Rules:
- For fields you cannot find: set to `null` (do not fabricate)
- For boolean checkbox fields: set true ONLY if there's clear evidence the box is checked; otherwise false (or null if unclear)
- For enum fields: pick exactly one option that matches; null if unclear
- For table fields (arrays of objects): emit one entry per visible row; empty array `[]` if no rows visible

The markdown was produced by an automatic OCR/parser, so some values may be slightly noisy. Use your judgment but err on the side of null over guessing.

=== PARSED MARKDOWN ===
{markdown}"""


def extract_one(parser_name: str, md_path: Path, schema_path: Path,
                model: str) -> dict:
    """Returns dict with keys: ok, extracted (dict|None), error (str|None),
    token_usage (dict), latency_sec (float)."""
    t0 = time.time()
    try:
        schema = json.loads(schema_path.read_text())
    except Exception as e:
        return {"ok": False, "error": f"schema load fail: {e}",
                "latency_sec": 0, "token_usage": {}, "extracted": None}
    try:
        md = md_path.read_text()
    except Exception as e:
        return {"ok": False, "error": f"md load fail: {e}",
                "latency_sec": 0, "token_usage": {}, "extracted": None}
    if not md.strip():
        return {"ok": False, "error": "empty markdown",
                "latency_sec": 0, "token_usage": {}, "extracted": None}

    coerced = coerce_schema(schema)
    tool = {
        "name": "extract_fields",
        "description": f"Extract structured data from the document. Schema title: {schema.get('title', '?')}",
        "input_schema": coerced,
    }
    prompt = EXTRACTION_PROMPT.format(markdown=truncate_md(md))

    client = get_anthropic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=8000,
            tools=[tool],
            tool_choice={"type": "tool", "name": "extract_fields"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return {"ok": False, "error": f"anthropic call fail: {type(e).__name__}: {str(e)[:300]}",
                "latency_sec": time.time() - t0, "token_usage": {}, "extracted": None}

    # Pull tool_use block
    extracted = None
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            extracted = block.input
            break
    if extracted is None:
        # Try text-fallback parsing
        text = resp.content[0].text if resp.content else ""
        return {"ok": False,
                "error": f"no tool_use in response; text: {text[:200]}",
                "latency_sec": time.time() - t0,
                "token_usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
                "extracted": None}

    return {
        "ok": True, "error": None,
        "extracted": extracted,
        "latency_sec": time.time() - t0,
        "token_usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
    }


def discover_work(sweep_dir: Path, schemas_dir: Path,
                  parser_filter: list[str] | None = None,
                  ) -> list[tuple[str, str, str, Path, Path, Path]]:
    """Yield (parser, domain, type, md_path, schema_path, out_path).

    Skips entries where out_path already exists.
    """
    out: list[tuple[str, str, str, Path, Path, Path]] = []
    for parser_dir in sorted(sweep_dir.iterdir()):
        if not parser_dir.is_dir() and not parser_dir.is_symlink():
            continue
        # Follow symlinks (we symlink Extend's existing run dir)
        try:
            real = parser_dir.resolve()
        except Exception:
            continue
        parser = parser_dir.name
        if parser_filter and parser not in parser_filter:
            continue
        extractions_root = parser_dir / "_extractions"
        for md_path in sorted(real.rglob("*.md")):
            # Skip quality_grades.md / summary mds in run root
            rel = md_path.relative_to(real)
            if len(rel.parts) < 3:
                continue
            domain, typ, fname = rel.parts[0], rel.parts[1], rel.parts[2]
            schema_path = schemas_dir / f"{typ}.json"
            if not schema_path.exists():
                continue
            out_path = extractions_root / domain / typ / fname.replace(".md", ".json")
            if out_path.exists():
                continue
            out.append((parser, domain, typ, md_path, schema_path, out_path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True,
                    help="runs/parse_sweep/<ts>/ root containing per-parser subdirs")
    ap.add_argument("--parser", action="append", default=None,
                    help="Restrict to specific parser(s) (repeatable)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="claude-sonnet-4-5")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        print(f"!! sweep dir not found: {sweep_dir}")
        return 1

    work = discover_work(sweep_dir, SCHEMAS_DIR, parser_filter=args.parser)
    if args.limit:
        work = work[:args.limit]
    print(f"Found {len(work)} (parser, doc) extraction tasks")
    if not work:
        print("  nothing to do (all extractions exist, or no parses yet)")
        return 0

    # Count per parser
    from collections import Counter
    by_parser = Counter(t[0] for t in work)
    for p, n in sorted(by_parser.items()):
        print(f"  {p:35} {n} tasks")
    print(f"  workers: {args.workers}, model: {args.model}")

    t0 = time.time()
    n_ok = n_fail = 0
    total_in = total_out = 0

    def _job(parser, domain, typ, md_path, schema_path, out_path):
        r = extract_one(parser, md_path, schema_path, args.model)
        # Always write a metadata sidecar; only write the extraction on success
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if r["ok"]:
            out_path.write_text(json.dumps(r["extracted"], indent=2, default=str))
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({
            "parser": parser, "domain": domain, "type": typ,
            "md_path": str(md_path.relative_to(REPO)),
            "schema_path": str(schema_path.relative_to(REPO)),
            "ok": r["ok"], "error": r.get("error"),
            "latency_sec": r["latency_sec"],
            "token_usage": r.get("token_usage", {}),
        }, indent=2))
        return (parser, typ, out_path.name, r)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, *w) for w in work]
        for i, fut in enumerate(as_completed(futures), 1):
            parser, typ, fname, r = fut.result()
            if r["ok"]:
                n_ok += 1
                total_in += r["token_usage"].get("input", 0)
                total_out += r["token_usage"].get("output", 0)
                nfields = len(r["extracted"]) if isinstance(r["extracted"], dict) else 0
                nonnull = sum(1 for v in (r["extracted"] or {}).values() if v not in (None, "", [], {}))
                if i % 20 == 0 or i == len(work):
                    print(f"  [{i}/{len(work)}] ✓ {parser}/{typ}/{fname}  "
                          f"{nonnull}/{nfields} non-null, "
                          f"in={r['token_usage'].get('input',0)} out={r['token_usage'].get('output',0)} "
                          f"{r['latency_sec']:.1f}s", flush=True)
            else:
                n_fail += 1
                print(f"  [{i}/{len(work)}] ✗ {parser}/{typ}/{fname}  {r['error'][:120]}", flush=True)

    wall = time.time() - t0
    # Claude Sonnet 4.5 pricing: $3/M input, $15/M output (approx)
    cost_est = (total_in / 1e6) * 3.0 + (total_out / 1e6) * 15.0
    print(f"\nDone — {n_ok} ok / {n_fail} fail / {len(work)} total in {wall:.0f}s")
    print(f"  tokens: {total_in:,} in / {total_out:,} out, est. ${cost_est:.2f}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
