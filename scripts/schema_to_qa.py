#!/usr/bin/env python3
"""Convert each v1.1 JSON Schema into a Q&A-format schema.

For each schema field, ask Claude Sonnet to produce a natural-language
question that, when answered against the document, gives the same value
the schema field would capture.

Why: QA-format extraction is often more accurate than tool-use schema
filling — the LLM reasons about each question on its own terms, can cite
evidence, and isn't forced into a JSON-shaped output.

Input:  datasets/extraction/v1.1/_schemas/<type>.json
Output: datasets/extraction/v1.1/_schemas_qa/<type>.json

Output format:
{
  "$qa_schema_version": "1.0",
  "source_schema": "<type>.json",
  "title": "...",
  "x-doc-type": "...",
  "x-domain": "...",
  "questions": [
    {
      "id": "<field_name>",
      "question": "What is X?",
      "answer_type": "string|number|boolean|date|enum|array_string|array_object",
      "answer_format": "MM/DD/YYYY" | "USD 2 decimals" | ...,
      "enum_options": [...],   // if applicable
      "nullable": true,
      "extraction_hint": "where to look",
      "field_kind": "checkbox|table|key_value|other"
    },
    ...
  ]
}
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
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
SCHEMAS_IN = V1_1 / "_schemas"
SCHEMAS_QA_OUT = V1_1 / "_schemas_qa"


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


CONVERT_PROMPT = """You are converting a JSON Schema (Extend dialect) into a Q&A extraction format.

For each field in the schema, write a clear natural-language QUESTION that an extractor agent should ask when reading the document. The answer to the question is exactly the value the schema field should hold.

Guidelines:
- Questions should be SPECIFIC and reference the doc location when known. Good: "What value is filled in Box 1a (insured's ID number)?" — Bad: "What is the ID number?"
- For booleans / checkboxes: phrase as a yes/no question. Good: "Is the box for 'Employment-related condition' checked in Box 10a?" — Bad: "is_employment_related?"
- For enums with mutually-exclusive options: list the options in the question. Good: "Which insurance type is checked in Box 1: Medicare, Medicaid, TRICARE, CHAMPVA, Group Health Plan, FECA Black Lung, or Other?"
- For tables: ask for the LIST of rows with their fields. Good: "List all service-line rows from Box 24, each with date, place of service, CPT code, diagnosis pointer, charges, and units."
- For dates: specify the expected format in `answer_format`.
- Include an `extraction_hint` that tells the extractor where to look (which box, section, page area).

Each question gets these keys:
  - `id`: the original schema field name (preserve EXACTLY)
  - `question`: the NL question (1 sentence)
  - `answer_type`: "string" | "number" | "boolean" | "date" | "enum" | "array_string" | "array_object"
  - `answer_format`: format hint (e.g. "MM/DD/YYYY", "USD with 2 decimals", "ICD-10 code") — empty string if N/A
  - `enum_options`: list of valid options (only for enum types) or empty list
  - `nullable`: true if the schema allows null (most should be true)
  - `extraction_hint`: short string ("Box 1, top of page 1", "Service line table FL42-49")
  - `field_kind`: "checkbox" | "table" | "key_value" | "other" (carry over from x-capabilities)

Return ONLY a JSON object with this shape:
{{
  "$qa_schema_version": "1.0",
  "source_schema": "{source_schema_name}",
  "title": "<copy from source>",
  "x-doc-type": "<copy from source>",
  "x-domain": "<copy from source>",
  "questions": [ ... one entry per schema property ... ]
}}

NO preamble, NO fences. Pure JSON.

=== SOURCE SCHEMA ===
{schema_json}"""


def field_kind(prop: dict) -> str:
    caps = prop.get("x-capabilities") or []
    if "checkbox" in caps:
        return "checkbox"
    if "table_structured" in caps or prop.get("type") == "array":
        return "table"
    if "form_region" in caps or "key_value" in caps:
        return "key_value"
    return "other"


def convert_one(schema_path: Path, model: str) -> dict:
    schema = json.loads(schema_path.read_text())
    # Truncate schema if too large — but keep all properties
    schema_text = json.dumps(schema, indent=2)
    if len(schema_text) > 30000:
        # Strip examples / very long descriptions if any
        compact = dict(schema)
        compact["properties"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "examples"} if isinstance(v, dict) else v
            for k, v in (schema.get("properties") or {}).items()
        }
        schema_text = json.dumps(compact, indent=2)

    prompt = CONVERT_PROMPT.format(
        source_schema_name=schema_path.name,
        schema_json=schema_text,
    )
    client = get_anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    qa = json.loads(text)

    # Sanity: every schema property should appear once in questions[]
    source_props = set((schema.get("properties") or {}).keys())
    qa_ids = {q["id"] for q in qa.get("questions", []) if "id" in q}
    missing = source_props - qa_ids
    extra = qa_ids - source_props
    qa["_validation"] = {
        "source_props": len(source_props),
        "qa_questions": len(qa.get("questions", [])),
        "missing_field_ids": sorted(missing),
        "extra_field_ids": sorted(extra),
    }
    return qa


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--schema", action="append", help="Restrict to one or more schema files (basename, repeatable)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-4-5")
    args = ap.parse_args()

    SCHEMAS_QA_OUT.mkdir(parents=True, exist_ok=True)

    schemas = sorted(SCHEMAS_IN.glob("*.json"))
    if args.schema:
        allowed = set(args.schema)
        schemas = [s for s in schemas if s.name in allowed or s.stem in allowed]

    # Skip done
    todo = []
    skipped = 0
    for s in schemas:
        out = SCHEMAS_QA_OUT / s.name
        if out.exists() and not args.force:
            skipped += 1
            continue
        todo.append(s)
    print(f"Schemas: {len(schemas)} total, {skipped} already converted, {len(todo)} to do")
    print(f"  workers: {args.workers}, model: {args.model}")

    t0 = time.time()
    n_ok = n_fail = 0

    def _job(schema_path: Path):
        try:
            qa = convert_one(schema_path, args.model)
            out_path = SCHEMAS_QA_OUT / schema_path.name
            out_path.write_text(json.dumps(qa, indent=2))
            v = qa.get("_validation", {})
            kind_counts = {}
            for q in qa.get("questions", []):
                kind_counts[q.get("field_kind", "other")] = kind_counts.get(q.get("field_kind", "other"), 0) + 1
            msg = (f"ok ({v.get('qa_questions', 0)}q, kinds: {kind_counts}, "
                   f"missing={len(v.get('missing_field_ids', []))})")
            return (schema_path.name, True, msg)
        except Exception as e:
            return (schema_path.name, False, f"ERR {type(e).__name__}: {str(e)[:200]}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, s): s for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            name, ok, msg = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            print(f"  [{i}/{len(todo)}] {'✓' if ok else '✗'} {name}: {msg}", flush=True)

    wall = time.time() - t0
    print(f"\nDone — {n_ok} ok / {n_fail} fail in {wall:.0f}s")
    print(f"  output: {SCHEMAS_QA_OUT.relative_to(REPO)}/")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
