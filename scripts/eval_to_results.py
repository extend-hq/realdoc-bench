#!/usr/bin/env python3
"""Adapter: eval_r1_rescored.json -> the {results:[...]} shape that
pilot_50_gen_qa.py --replace-easy expects.

A question is "correct" for a parser iff that parser is in the question-level
parsers_correct_<layer> set (default layer: llm — most permissive, so only
questions every parser TRULY nails get flagged easy).

Usage: eval_to_results.py <eval_r1_rescored.json> <out.json> [layer]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    layer = sys.argv[3] if len(sys.argv) > 3 else "llm"

    data = json.loads(src.read_text())
    results = []
    for dp, page in data.items():
        stem = Path(dp).stem
        for qid, q in page.get("questions", {}).items():
            correct = set(q.get(f"parsers_correct_{layer}")
                          or q.get("parsers_correct") or [])
            for parser in q.get("parser_answers", {}):
                results.append({
                    "question_id": qid,
                    "source_file": stem,
                    "parser": parser,
                    "exact_match": parser in correct,
                })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"wrote {len(results)} result rows ({layer}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
