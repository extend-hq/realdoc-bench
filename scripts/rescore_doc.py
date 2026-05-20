"""Re-score every (parser × agent) cell for a single doc against the *current*
ground_truth/<doc>.json, using existing agent_output.json on disk.

Use this when a GT field is corrected — no need to re-run parses or agents.
Updates every existing score.json in place and prints the diff for every
field whose score changed.

Usage:
    python scripts/rescore_doc.py tech_datasheet_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from realdoc_bench.extraction.scoring.engine import score as score_fn, macro_f1
from realdoc_bench.extraction.schema.dialect import to_plain_jsonschema


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc_id")
    ap.add_argument("--field", help="If set, only print diffs for this field path.")
    args = ap.parse_args()

    gt_path = REPO / "datasets" / "extraction" / "v0" / "ground_truth" / f"{args.doc_id}.json"
    schema_path = REPO / "datasets" / "extraction" / "v0" / "schemas" / f"{args.doc_id}.json"
    gt = json.loads(gt_path.read_text())
    schema = to_plain_jsonschema(json.loads(schema_path.read_text()))

    updated = 0
    diffs = []

    for ao in REPO.rglob(f"runs/*/{args.doc_id}/*/*/agent_output.json"):
        agent = ao.parent.name
        parser = ao.parent.parent.name
        run = ao.parent.parent.parent.parent.name
        actual = json.loads(ao.read_text())

        old_score = None
        score_json = ao.parent / "score.json"
        if score_json.exists():
            try:
                old_data = json.loads(score_json.read_text())
                # Map field path -> old score
                old_score = {s["path"]: s["score"] for s in old_data.get("scores", [])}
            except Exception:
                old_score = {}

        rows = score_fn(gt, actual, schema)
        new_data = {
            "macro_f1": macro_f1(rows),
            "scores": [
                {
                    "path": r.path, "leaf_type": r.leaf_type,
                    "expected": r.expected, "actual": r.actual,
                    "score": r.score, "match": r.match,
                    "capabilities": r.capabilities, "difficulty": r.difficulty,
                    "required": r.required,
                }
                for r in rows
            ],
        }
        score_json.write_text(json.dumps(new_data, indent=2, default=str))
        updated += 1

        if old_score is not None:
            for r in rows:
                if args.field and r.path != args.field:
                    continue
                prev = old_score.get(r.path)
                if prev is not None and abs(prev - r.score) > 1e-9:
                    diffs.append((run, parser, agent, r.path, prev, r.score, r.actual))

    print(f"rescored {updated} score.json files for {args.doc_id}")
    if diffs:
        print(f"\n{len(diffs)} field-level score changes:")
        for run, parser, agent, path, old, new, actual in diffs:
            print(f"  [{run[:24]}] {parser:<30s} {agent:<18s} {path:<45s}  {old:.2f} -> {new:.2f}   actual={actual!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
