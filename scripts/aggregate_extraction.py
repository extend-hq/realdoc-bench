"""Aggregate realdoc-bench extraction runs into one per-(parser, agent) table.

Reads ``runs/<run-id>/summary.json`` for each given run-id and rolls the
per-cell ``macro_f1`` / cost / latency up by (parser, agent). Part B is split
across several run-ids (one per parser group) so summaries don't clobber —
this stitches them back together.

Usage:
    python scripts/aggregate_extraction.py \\
        --run-id bench_extract_v0 \\
        --run-id bench_extract_dots_mocr \\
        --run-id bench_extract_claude_sonnet_4_6 \\
        --run-id bench_extract_gemini_3_flash
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", action="append", required=True, dest="run_ids")
    ap.add_argument("--run-dir", type=Path, default=Path("runs"))
    ap.add_argument("--json-out", type=Path, help="Optional path to dump the aggregate as JSON.")
    args = ap.parse_args()

    cells: list[dict] = []
    for rid in args.run_ids:
        summary_path = args.run_dir / rid / "summary.json"
        if not summary_path.exists():
            print(f"  (skip {rid}: no summary.json)")
            continue
        cells.extend(json.loads(summary_path.read_text()).get("cells", []))

    # Roll up by (parser, agent)
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cells:
        groups.setdefault((c["parser"], c["agent"]), []).append(c)

    rows = []
    for (parser, agent), gcells in sorted(groups.items()):
        ok = [c for c in gcells if c.get("status") == "ok"]
        f1s = [c["macro_f1"] for c in ok if c.get("macro_f1") is not None]
        rows.append(
            {
                "parser": parser,
                "agent": agent,
                "n_docs": len(gcells),
                "n_ok": len(ok),
                "macro_f1": _mean(f1s),
                "parse_latency_s": _mean([c.get("parse_latency_sec") for c in gcells]),
                "agent_latency_s": _mean([c.get("agent_latency_sec") for c in gcells]),
                "parse_cost_usd": _mean([c.get("parse_cost_usd") for c in gcells]),
                "agent_cost_usd": _mean([c.get("agent_cost_usd") for c in gcells]),
            }
        )

    # Sort by macro_f1 desc (None last)
    rows.sort(key=lambda r: (r["macro_f1"] is None, -(r["macro_f1"] or 0)))

    w = max(len(r["parser"]) for r in rows) if rows else 10
    print(f"\n{'parser':<{w}}  {'agent':<18}  {'F1':>6}  {'ok/N':>7}  {'parse_s':>8}  {'agent_s':>8}  {'$/doc':>8}")
    print("-" * (w + 64))
    for r in rows:
        f1 = f"{r['macro_f1']:.3f}" if r["macro_f1"] is not None else "  —  "
        pl = f"{r['parse_latency_s']:.1f}" if r["parse_latency_s"] is not None else "—"
        al = f"{r['agent_latency_s']:.1f}" if r["agent_latency_s"] is not None else "—"
        pc = (r["parse_cost_usd"] or 0) + (r["agent_cost_usd"] or 0)
        print(
            f"{r['parser']:<{w}}  {r['agent']:<18}  {f1:>6}  "
            f"{r['n_ok']:>3}/{r['n_docs']:<3}  {pl:>8}  {al:>8}  {pc:>8.4f}"
        )

    # Parser-only rollup (mean over agents)
    by_parser: dict[str, list[float]] = {}
    for r in rows:
        if r["macro_f1"] is not None:
            by_parser.setdefault(r["parser"], []).append(r["macro_f1"])
    print(f"\n{'parser (mean over agents)':<{w+20}}  {'F1':>6}")
    print("-" * (w + 28))
    for parser, f1s in sorted(by_parser.items(), key=lambda kv: -statistics.fmean(kv[1])):
        print(f"{parser:<{w+20}}  {statistics.fmean(f1s):>6.3f}")

    if args.json_out:
        args.json_out.write_text(json.dumps({"rows": rows}, indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
