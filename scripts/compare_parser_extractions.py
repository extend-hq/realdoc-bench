#!/usr/bin/env python3
"""Cross-parser comparison report.

Inputs:
  runs/parse_sweep/<ts>/<parser>/_extractions/<dom>/<type>/<doc>.json   per-parser extractions

For each (doc, schema field):
  - gather all parsers' values
  - pick a CONSENSUS via majority vote (or null if no majority)
  - score each parser's value against consensus → agreement rate
  - flag fields where parsers disagree heavily (interesting / hard fields)

Outputs:
  runs/parse_sweep/<ts>/comparison/
    per_parser_summary.csv     parser, n_docs, n_fields, n_non_null, agreement_rate, est_cost
    per_field_agreement.csv    type, field, n_parsers, consensus_value, agreements
    disagreement_hotspots.md   top-N fields where parsers disagree most
    INDEX.md                   headline report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
SCHEMAS_DIR = V1_1 / "_schemas"


def normalize_value(v):
    """Hashable, fuzz-tolerant value for agreement comparison."""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, str):
        s = v.strip().lower()
        s = " ".join(s.split())  # collapse whitespace
        return s if s else None
    if isinstance(v, list):
        return ("list", len(v))  # for arrays, agree on count (loose)
    if isinstance(v, dict):
        return ("dict", len(v))
    return str(v)[:200]


def load_extractions(sweep_dir: Path) -> dict:
    """Returns nested dict:
       data[doc_key][parser][field] = raw value
       doc_key = (domain, type, filename)"""
    data: dict = defaultdict(lambda: defaultdict(dict))
    parsers = set()
    for parser_dir in sorted(sweep_dir.iterdir()):
        if not (parser_dir.is_dir() or parser_dir.is_symlink()):
            continue
        parser = parser_dir.name
        if parser in {"comparison"}:
            continue
        ext_root = parser_dir / "_extractions"
        if not ext_root.exists():
            continue
        for jpath in ext_root.rglob("*.json"):
            if jpath.name.endswith(".meta.json"):
                continue
            rel = jpath.relative_to(ext_root)
            if len(rel.parts) < 3:
                continue
            domain, typ, fname = rel.parts[0], rel.parts[1], rel.parts[2]
            doc_key = (domain, typ, fname.replace(".json", ""))
            try:
                blob = json.loads(jpath.read_text())
            except Exception:
                continue
            if not isinstance(blob, dict):
                continue
            data[doc_key][parser] = blob
            parsers.add(parser)
    return dict(data), sorted(parsers)


def load_meta_costs(sweep_dir: Path) -> dict:
    """Returns {parser → {'total_in': X, 'total_out': Y, 'n_ok': N, 'n_fail': F}}"""
    out: dict = defaultdict(lambda: {"total_in": 0, "total_out": 0, "n_ok": 0, "n_fail": 0, "total_lat": 0.0})
    for parser_dir in sorted(sweep_dir.iterdir()):
        if not (parser_dir.is_dir() or parser_dir.is_symlink()):
            continue
        parser = parser_dir.name
        if parser == "comparison":
            continue
        ext_root = parser_dir / "_extractions"
        if not ext_root.exists():
            continue
        for mpath in ext_root.rglob("*.meta.json"):
            try:
                m = json.loads(mpath.read_text())
            except Exception:
                continue
            if m.get("ok"):
                out[parser]["n_ok"] += 1
            else:
                out[parser]["n_fail"] += 1
            tu = m.get("token_usage") or {}
            out[parser]["total_in"] += tu.get("input", 0) or 0
            out[parser]["total_out"] += tu.get("output", 0) or 0
            out[parser]["total_lat"] += m.get("latency_sec", 0) or 0
    return out


def load_parse_manifests(sweep_dir: Path) -> dict:
    """Returns {parser → {'pages': total_pp, 'docs_ok': n, 'docs_fail': n, 'parse_cost_usd': X}}"""
    out: dict = defaultdict(lambda: {"pages": 0, "docs_ok": 0, "docs_fail": 0,
                                     "parse_cost_usd": 0.0, "parse_latency_sec": 0.0})
    for parser_dir in sorted(sweep_dir.iterdir()):
        if not (parser_dir.is_dir() or parser_dir.is_symlink()):
            continue
        parser = parser_dir.name
        if parser == "comparison":
            continue
        real = parser_dir.resolve()
        m = real / "manifest.csv"
        if not m.exists():
            continue
        with m.open() as f:
            for row in csv.DictReader(f):
                if row.get("ok") == "1":
                    out[parser]["docs_ok"] += 1
                    out[parser]["pages"] += int(row.get("page_count") or 0)
                    out[parser]["parse_cost_usd"] += float(row.get("cost_usd") or 0)
                    out[parser]["parse_latency_sec"] += float(row.get("latency_sec") or 0)
                else:
                    out[parser]["docs_fail"] += 1
    return out


def compute_consensus_and_agreement(data: dict, parsers: list[str]) -> tuple[dict, dict]:
    """For each (doc, field):
         - tally normalized values across parsers
         - consensus = majority (≥50%) of non-null values
         - per-parser agreement = 1 if matches consensus, 0 otherwise, null if parser had no value
    Returns:
      per_field: list of dicts (type, field, n_parsers_with_value, consensus, votes, n_unique_values)
      per_parser_per_field: {parser → list of (type, field, agreed: 1|0|None, value)}
    """
    field_rows: list[dict] = []
    parser_agreement: dict = defaultdict(list)  # parser → list of (1|0|None)
    parser_nonnull: dict = defaultdict(int)
    parser_total_fields_seen: dict = defaultdict(int)

    for (domain, typ, fname), parser_outputs in data.items():
        # union of fields across parsers' extractions
        all_fields: set = set()
        for p in parsers:
            if p in parser_outputs and isinstance(parser_outputs[p], dict):
                all_fields.update(parser_outputs[p].keys())
        for fld in all_fields:
            values: dict = {}  # parser → normalized value
            raw_values: dict = {}
            for p in parsers:
                if p not in parser_outputs:
                    continue
                v = parser_outputs[p].get(fld)
                nv = normalize_value(v)
                if nv is None:
                    values[p] = None
                else:
                    values[p] = nv
                    raw_values[p] = v
            # tally non-null values
            non_null_vals = [v for v in values.values() if v is not None]
            if not non_null_vals:
                consensus = None
            else:
                tally = Counter(non_null_vals)
                top, top_count = tally.most_common(1)[0]
                # majority of non-null parsers
                consensus = top if top_count * 2 > len(non_null_vals) else None
            field_rows.append({
                "domain": domain, "type": typ, "filename": fname, "field": fld,
                "n_parsers": sum(1 for v in values.values() if v is not None),
                "n_unique_values": len(set(non_null_vals)) if non_null_vals else 0,
                "consensus": str(consensus)[:120] if consensus is not None else "",
                "has_consensus": consensus is not None,
            })
            for p, v in values.items():
                parser_total_fields_seen[p] += 1
                if v is not None:
                    parser_nonnull[p] += 1
                if consensus is None:
                    parser_agreement[p].append(None)
                else:
                    parser_agreement[p].append(1 if v == consensus else 0)

    parser_stats: dict = {}
    for p in parsers:
        agr = parser_agreement[p]
        graded = [a for a in agr if a is not None]
        parser_stats[p] = {
            "fields_seen": parser_total_fields_seen[p],
            "fields_non_null": parser_nonnull[p],
            "non_null_rate": (parser_nonnull[p] / parser_total_fields_seen[p]) if parser_total_fields_seen[p] else 0,
            "agreement_rate_vs_consensus": (sum(graded) / len(graded)) if graded else 0,
            "n_graded": len(graded),
        }
    return field_rows, parser_stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--top-disagreement", type=int, default=30,
                    help="Show top-N most-disagreed fields in hotspots report")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        print(f"!! {sweep_dir} not found")
        return 1

    out_dir = sweep_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading extractions from {sweep_dir.relative_to(REPO)}/...")
    data, parsers = load_extractions(sweep_dir)
    print(f"  {len(data)} unique docs, {len(parsers)} parsers: {parsers}")

    print("Loading parse manifests for cost/latency...")
    parse_stats = load_parse_manifests(sweep_dir)
    print("Loading extraction meta for token cost...")
    extract_meta = load_meta_costs(sweep_dir)

    print("Computing consensus + per-parser agreement...")
    field_rows, parser_stats = compute_consensus_and_agreement(data, parsers)

    # 1. Per-parser summary CSV + markdown
    summary_csv = out_dir / "per_parser_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fields = ["parser", "docs_parsed_ok", "docs_parsed_fail", "pages_parsed",
                  "parse_cost_usd", "parse_latency_sec",
                  "extractions_ok", "extractions_fail",
                  "extract_token_in", "extract_token_out", "extract_cost_usd",
                  "fields_seen", "fields_non_null", "non_null_rate",
                  "agreement_rate_vs_consensus", "n_graded",
                  "total_cost_usd", "cost_per_doc_usd"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in parsers:
            ps = parse_stats[p]
            em = extract_meta[p]
            # claude-sonnet-4-5 pricing for extraction phase
            ext_cost = (em["total_in"] / 1e6) * 3.0 + (em["total_out"] / 1e6) * 15.0
            total_cost = ps["parse_cost_usd"] + ext_cost
            stats = parser_stats.get(p, {})
            row = {
                "parser": p,
                "docs_parsed_ok": ps["docs_ok"],
                "docs_parsed_fail": ps["docs_fail"],
                "pages_parsed": ps["pages"],
                "parse_cost_usd": round(ps["parse_cost_usd"], 4),
                "parse_latency_sec": round(ps["parse_latency_sec"], 1),
                "extractions_ok": em["n_ok"],
                "extractions_fail": em["n_fail"],
                "extract_token_in": em["total_in"],
                "extract_token_out": em["total_out"],
                "extract_cost_usd": round(ext_cost, 4),
                "fields_seen": stats.get("fields_seen", 0),
                "fields_non_null": stats.get("fields_non_null", 0),
                "non_null_rate": round(stats.get("non_null_rate", 0), 4),
                "agreement_rate_vs_consensus": round(stats.get("agreement_rate_vs_consensus", 0), 4),
                "n_graded": stats.get("n_graded", 0),
                "total_cost_usd": round(total_cost, 4),
                "cost_per_doc_usd": round(total_cost / max(em["n_ok"], 1), 4),
            }
            w.writerow(row)

    # 2. Per-field agreement CSV
    field_csv = out_dir / "per_field_agreement.csv"
    with field_csv.open("w", newline="") as f:
        fnames = ["domain", "type", "filename", "field", "n_parsers", "n_unique_values",
                  "has_consensus", "consensus"]
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        for row in field_rows:
            w.writerow(row)

    # 3. Disagreement hotspots: fields with >= 4 parsers AND >= 3 unique values
    hotspots = [r for r in field_rows
                if r["n_parsers"] >= 4 and r["n_unique_values"] >= 3]
    hotspots.sort(key=lambda r: (-r["n_unique_values"], -r["n_parsers"]))
    hs_md = out_dir / "disagreement_hotspots.md"
    with hs_md.open("w") as f:
        f.write(f"# Disagreement hotspots — {sweep_dir.name}\n\n")
        f.write(f"Fields where ≥4 parsers extracted a value but ≥3 disagreed.\n")
        f.write(f"Top {args.top_disagreement} by uniqueness.\n\n")
        f.write("| Type | Field | n_parsers | unique vals | consensus? |\n|---|---|---:|---:|:---:|\n")
        for r in hotspots[:args.top_disagreement]:
            cs = "✓" if r["has_consensus"] else "—"
            f.write(f"| `{r['type']}` | `{r['field']}` | {r['n_parsers']} | "
                    f"{r['n_unique_values']} | {cs} |\n")
        f.write(f"\nTotal hotspot fields: {len(hotspots)} / {len(field_rows)} all field-rows.\n")

    # 4. Headline INDEX.md
    idx = out_dir / "INDEX.md"
    with idx.open("w") as f:
        f.write(f"# Cross-parser comparison — {sweep_dir.name}\n\n")
        f.write(f"**{len(parsers)} parsers · {len(data)} unique docs · "
                f"{len(field_rows):,} (doc × field) cells**\n\n")
        f.write("## Per-parser scoreboard\n\n")
        f.write("Sorted by **agreement rate vs cross-parser consensus**.\n\n")
        f.write("| Parser | Docs ok | Pages | Parse $ | Extract $ | Total $ | $/doc | "
                "Non-null rate | Agree rate vs consensus |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        rows = []
        for p in parsers:
            ps = parse_stats[p]
            em = extract_meta[p]
            ext_cost = (em["total_in"] / 1e6) * 3.0 + (em["total_out"] / 1e6) * 15.0
            total_cost = ps["parse_cost_usd"] + ext_cost
            stats = parser_stats.get(p, {})
            rows.append((p, ps, em, ext_cost, total_cost, stats))
        rows.sort(key=lambda r: -r[5].get("agreement_rate_vs_consensus", 0))
        for p, ps, em, ext_cost, total_cost, stats in rows:
            f.write(f"| `{p}` | {em['n_ok']} | {ps['pages']:,} | "
                    f"${ps['parse_cost_usd']:.2f} | ${ext_cost:.2f} | "
                    f"${total_cost:.2f} | "
                    f"${total_cost / max(em['n_ok'], 1):.3f} | "
                    f"{stats.get('non_null_rate', 0):.1%} | "
                    f"{stats.get('agreement_rate_vs_consensus', 0):.1%} "
                    f"({stats.get('n_graded', 0):,}) |\n")

        f.write(f"\n## Disagreement\n\nSee `disagreement_hotspots.md` for fields where parsers "
                f"diverge most.\n\n")
        f.write("## Files\n\n")
        f.write("- `per_parser_summary.csv` — full numbers per parser\n")
        f.write("- `per_field_agreement.csv` — (doc, field) → consensus + parser count\n")
        f.write("- `disagreement_hotspots.md` — sorted disagreements\n")

    # Console summary
    print(f"\nWrote to {out_dir.relative_to(REPO)}/:")
    for p in [summary_csv, field_csv, hs_md, idx]:
        print(f"  {p.relative_to(REPO)}")
    print("\n=== HEADLINE ===")
    rows.sort(key=lambda r: -r[5].get("agreement_rate_vs_consensus", 0))
    print(f"  {'parser':32} {'docs':>5} {'pages':>6} {'tot$':>7} {'$/doc':>6} {'nonnull':>7} {'agree':>6}")
    for p, ps, em, ext_cost, total_cost, stats in rows:
        print(f"  {p:32} {em['n_ok']:>5} {ps['pages']:>6} ${total_cost:>6.2f} "
              f"${total_cost/max(em['n_ok'],1):>5.3f} "
              f"{stats.get('non_null_rate',0):>6.1%} "
              f"{stats.get('agreement_rate_vs_consensus',0):>5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
