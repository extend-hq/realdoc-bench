#!/usr/bin/env python3
"""QA-style extraction across all parser outputs.

For each (parser, doc) pair:
  - load the QA schema (datasets/extraction/v1.1/_schemas_qa/<type>.json)
  - load the parser's markdown
  - call Claude Sonnet with the questions list + markdown
  - Claude answers all questions in one structured response
  - save:  runs/parse_sweep/<ts>/<parser>/_qa_extractions/<dom>/<type>/<doc>.json

Output JSON shape: {field_id: {answer: <value>, evidence: "<snippet>" | null,
                                confidence: 0-1}}

Idempotent. Parallel.

Usage:
  python -u scripts/extract_qa_from_parses.py --sweep-dir runs/parse_sweep/<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
SCHEMAS_QA_DIR = V1_1 / "_schemas_qa"


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


def truncate_md(md: str, limit: int = 60000) -> str:
    if len(md) <= limit:
        return md
    head = limit * 2 // 3
    tail = limit - head
    return md[:head] + "\n\n…[truncated middle]…\n\n" + md[-tail:]


def build_answer_tool(qa_schema: dict) -> dict:
    """Build a tool input_schema where each question's answer is a property.

    Each answer is wrapped: {answer: <typed>, evidence: <string|null>, confidence: <num>}
    Type per question is loose (string|number|boolean|null) since markdown is noisy.
    """
    properties: dict = {}
    for q in qa_schema.get("questions", []):
        qid = q["id"]
        atype = q.get("answer_type", "string")
        nullable = q.get("nullable", True)
        enum_opts = q.get("enum_options") or []

        # Define the answer's allowed type
        if atype == "boolean":
            ans_def = {"type": ["boolean", "null"] if nullable else "boolean"}
        elif atype == "number":
            ans_def = {"type": ["number", "null"] if nullable else "number"}
        elif atype == "enum":
            ans_def = {
                "type": ["string", "null"] if nullable else "string",
                "enum": (enum_opts + [None]) if nullable else enum_opts,
            }
        elif atype == "date":
            ans_def = {"type": ["string", "null"] if nullable else "string"}
        elif atype == "array_string":
            ans_def = {"type": ["array", "null"] if nullable else "array",
                       "items": {"type": "string"}}
        elif atype == "array_object":
            ans_def = {"type": ["array", "null"] if nullable else "array",
                       "items": {"type": "object"}}
        else:  # string default
            ans_def = {"type": ["string", "null"] if nullable else "string"}

        properties[qid] = {
            "type": ["object", "null"],
            "properties": {
                "answer": ans_def,
                "evidence": {"type": ["string", "null"],
                              "description": "Short verbatim quote from the markdown that supports the answer; null if no evidence found"},
                "confidence": {"type": ["number", "null"],
                                "description": "0.0 to 1.0 — how sure are you this is right based on what's in the markdown"},
            },
            "required": ["answer"],
        }

    return {
        "name": "answer_questions",
        "description": f"Answer every question in the QA schema using the parsed document. "
                       f"For each question, provide an answer (or null if not in doc), "
                       f"a short evidence snippet from the markdown, and a confidence.",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys())[:0],  # don't require any specific id
        },
    }


QA_PROMPT = """You are answering a series of extraction questions about a document. The document is provided below as parsed markdown (output of an OCR/parser; some structure may be noisy).

For EACH question in the QA schema, call the `answer_questions` tool with:
- `answer`: the value (use the answer_type — null if not found in the document)
- `evidence`: a short verbatim snippet (≤ 40 words) from the markdown that supports your answer; null if no evidence
- `confidence`: 0.0 to 1.0 — how confident you are. Use ≤0.5 if the evidence is weak/inferred, ≥0.85 if exact match in the markdown.

Be conservative — return null + low confidence rather than guess. For boolean checkbox questions: return true ONLY when there's clear "checked" evidence in the markdown.

=== QUESTIONS ({n_questions} total) ===
{questions_block}

=== PARSED MARKDOWN (may be truncated) ===
{markdown}"""


def render_questions_block(qa_schema: dict, limit_chars: int = 8000) -> str:
    parts = []
    total = 0
    for q in qa_schema.get("questions", []):
        line = f"- [{q['id']}] {q['question']}"
        if q.get("answer_format"):
            line += f" (format: {q['answer_format']})"
        if q.get("extraction_hint"):
            line += f"  — hint: {q['extraction_hint']}"
        parts.append(line)
        total += len(line)
        if total > limit_chars:
            parts.append(f"…[+{len(qa_schema['questions']) - len(parts)} more questions, see schema]…")
            break
    return "\n".join(parts)


def extract_one(parser_name: str, md_path: Path, qa_schema_path: Path,
                model: str) -> dict:
    t0 = time.time()
    try:
        qa_schema = json.loads(qa_schema_path.read_text())
    except Exception as e:
        return {"ok": False, "error": f"qa schema load fail: {e}",
                "latency_sec": 0, "token_usage": {}, "extracted": None}
    try:
        md = md_path.read_text()
    except Exception as e:
        return {"ok": False, "error": f"md load fail: {e}",
                "latency_sec": 0, "token_usage": {}, "extracted": None}
    if not md.strip():
        return {"ok": False, "error": "empty markdown",
                "latency_sec": 0, "token_usage": {}, "extracted": None}

    n_q = len(qa_schema.get("questions", []))
    tool = build_answer_tool(qa_schema)
    prompt = QA_PROMPT.format(
        n_questions=n_q,
        questions_block=render_questions_block(qa_schema),
        markdown=truncate_md(md),
    )

    client = get_anthropic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=12000,
            tools=[tool],
            tool_choice={"type": "tool", "name": "answer_questions"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return {"ok": False,
                "error": f"anthropic fail: {type(e).__name__}: {str(e)[:300]}",
                "latency_sec": time.time() - t0, "token_usage": {}, "extracted": None}

    answered = None
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            answered = block.input
            break
    if answered is None:
        return {"ok": False, "error": "no tool_use returned",
                "latency_sec": time.time() - t0,
                "token_usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
                "extracted": None}

    return {
        "ok": True, "error": None,
        "extracted": answered,
        "latency_sec": time.time() - t0,
        "token_usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
    }


def discover_work(sweep_dir: Path,
                  parser_filter: list[str] | None = None,
                  ) -> list[tuple[str, str, str, Path, Path, Path]]:
    """Yield (parser, domain, type, md_path, qa_schema_path, out_path). Skips done."""
    out: list = []
    for parser_dir in sorted(sweep_dir.iterdir()):
        if not (parser_dir.is_dir() or parser_dir.is_symlink()):
            continue
        parser = parser_dir.name
        if parser == "comparison":
            continue
        if parser_filter and parser not in parser_filter:
            continue
        try:
            real = parser_dir.resolve()
        except Exception:
            continue
        qa_root = parser_dir / "_qa_extractions"
        for md_path in sorted(real.rglob("*.md")):
            rel = md_path.relative_to(real)
            if len(rel.parts) < 3:
                continue
            domain, typ, fname = rel.parts[0], rel.parts[1], rel.parts[2]
            qa_path = SCHEMAS_QA_DIR / f"{typ}.json"
            if not qa_path.exists():
                continue
            out_path = qa_root / domain / typ / fname.replace(".md", ".json")
            if out_path.exists():
                continue
            out.append((parser, domain, typ, md_path, qa_path, out_path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--parser", action="append", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="claude-sonnet-4-5")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    work = discover_work(sweep_dir, parser_filter=args.parser)
    if args.limit:
        work = work[:args.limit]
    print(f"Found {len(work)} (parser, doc) QA-extraction tasks")
    if not work:
        return 0

    from collections import Counter
    by_parser = Counter(t[0] for t in work)
    for p, n in sorted(by_parser.items()):
        print(f"  {p:35} {n} tasks")
    print(f"  workers: {args.workers}, model: {args.model}")

    t0 = time.time()
    n_ok = n_fail = 0
    total_in = total_out = 0

    def _job(parser, domain, typ, md_path, qa_path, out_path):
        r = extract_one(parser, md_path, qa_path, args.model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if r["ok"]:
            out_path.write_text(json.dumps(r["extracted"], indent=2, default=str))
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({
            "parser": parser, "domain": domain, "type": typ,
            "md_path": str(md_path.relative_to(REPO)),
            "qa_schema_path": str(qa_path.relative_to(REPO)),
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
                n_answered = sum(1 for v in (r["extracted"] or {}).values()
                                 if isinstance(v, dict) and v.get("answer") not in (None, "", []))
                n_total = len(r["extracted"] or {})
                if i % 20 == 0 or i == len(work):
                    print(f"  [{i}/{len(work)}] ✓ {parser}/{typ}/{fname}  "
                          f"{n_answered}/{n_total} answered, "
                          f"in={r['token_usage'].get('input',0)} out={r['token_usage'].get('output',0)} "
                          f"{r['latency_sec']:.1f}s", flush=True)
            else:
                n_fail += 1
                print(f"  [{i}/{len(work)}] ✗ {parser}/{typ}/{fname}  {r['error'][:140]}", flush=True)

    wall = time.time() - t0
    cost_est = (total_in / 1e6) * 3.0 + (total_out / 1e6) * 15.0
    print(f"\nDone — {n_ok} ok / {n_fail} fail in {wall:.0f}s")
    print(f"  tokens: {total_in:,} in / {total_out:,} out, est. ${cost_est:.2f}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
