"""Rebuild a run's ``summary.json`` from per-cell artifacts on disk.

Useful when an under-the-same-run-id re-invocation of ``realdoc-bench extract
run`` clobbers the run's ``summary.json`` to include only the freshly-run
cells. The per-doc ``parse_result.json`` / ``score.json`` / ``agent_meta.json``
files are still on disk under ``runs/<run-id>/<doc>/<parser>/<agent>/`` — this
script walks them and rewrites the index.

Optional: merge cells from one or more *other* run-ids (typically a *_fix
or _rerun bucket) into the target run's summary, latest occurrence wins.

Usage:
    python scripts/rebuild_summary.py --run-id bench_extract_v0
    python scripts/rebuild_summary.py --run-id bench_extract_v0 \\
        --merge bench_extract_v0_v2fix
    python scripts/rebuild_summary.py --run-id bench_extract_v0 \\
        --merge bench_extract_v0_v2fix --in-place=false  # dry-run + dump to stdout
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cell_from_disk(run_dir: Path, doc_id: str, parser: str, agent: str) -> dict | None:
    """Reconstruct one cell from parse_result.json + score.json + agent_meta.json."""
    parser_dir = run_dir / doc_id / parser
    agent_dir = parser_dir / agent
    pr = _read_json(parser_dir / "parse_result.json")
    sc = _read_json(agent_dir / "score.json")
    am = _read_json(agent_dir / "agent_meta.json")
    if pr is None and sc is None and am is None:
        return None

    pr = pr or {}
    sc = sc or {}
    am = am or {}

    # Status inference: "ok" iff there is a macro_f1 *and* the agent ran;
    # other paths land at "agent_error" / "failed" depending on what's
    # present. Without a stored status field we approximate.
    macro_f1 = sc.get("macro_f1")
    has_agent = bool(am)
    has_parse = bool(pr) and (pr.get("markdown") or pr.get("page_count"))
    if macro_f1 is not None:
        status = "ok"
        error = None
    elif has_agent and macro_f1 is None:
        status = "agent_error"
        error = am.get("error")
    elif not has_parse:
        status = "parse_error"
        error = pr.get("error") if pr else None
    else:
        status = "failed"
        error = None

    return {
        "doc_id": doc_id,
        "parser": parser,
        "parser_version": pr.get("version") or "",
        "agent": agent,
        "agent_version": am.get("version") or "",
        "status": status,
        "error": error,
        "parse_latency_sec": pr.get("latency_sec"),
        "parse_cost_usd": pr.get("cost_estimate_usd"),
        "parse_pages": pr.get("page_count"),
        "agent_latency_sec": am.get("latency_sec"),
        "agent_cost_usd": am.get("cost_usd"),
        "agent_retries": am.get("retries", 0),
        "macro_f1": macro_f1,
    }


def _walk_cells(run_dir: Path) -> list[dict]:
    """Walk runs/<rid>/<doc>/<parser>/<agent>/ to enumerate present cells."""
    out: list[dict] = []
    if not run_dir.exists():
        return out
    for doc_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for parser_dir in sorted(p for p in doc_dir.iterdir() if p.is_dir()):
            # Each parser dir holds parse_result.json + one subdir per agent.
            for agent_dir in sorted(p for p in parser_dir.iterdir() if p.is_dir()):
                cell = _cell_from_disk(run_dir, doc_dir.name, parser_dir.name, agent_dir.name)
                if cell is not None:
                    out.append(cell)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="Target run-id whose summary.json to rewrite.")
    ap.add_argument("--run-dir", type=Path, default=Path("runs"))
    ap.add_argument(
        "--merge",
        action="append",
        default=[],
        help="Additional run-id(s) to fold in (later wins on (doc, parser, agent)).",
    )
    ap.add_argument(
        "--in-place",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Write the result back to runs/<run-id>/summary.json (default true). "
        "Pass --in-place=false to dump to stdout instead.",
    )
    args = ap.parse_args()

    target_dir = args.run_dir / args.run_id

    # Walk target + each --merge run-id, in order. Later wins.
    cells_by_key: dict[tuple[str, str, str], dict] = {}
    sources = [args.run_id] + list(args.merge)
    counts = {}
    for rid in sources:
        rd = args.run_dir / rid
        for c in _walk_cells(rd):
            cells_by_key[(c["doc_id"], c["parser"], c["agent"])] = c
        counts[rid] = sum(
            1 for c in cells_by_key.values()
            # crude: count cells whose source dir contains the rid
            if (rd / c["doc_id"] / c["parser"] / c["agent"]).exists()
        )

    cells = sorted(cells_by_key.values(), key=lambda c: (c["doc_id"], c["parser"], c["agent"]))

    # Pull what we can of the headline fields from any existing summary file
    # in the target dir (preserve original timing / catalog SHA).
    existing = _read_json(target_dir / "summary.json") or {}

    summary = {
        "run_id": args.run_id,
        "manifest_set_id": existing.get("manifest_set_id", "v0"),
        "started_at": existing.get("started_at"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "catalog_sha": existing.get("catalog_sha"),
        "parsers": sorted({c["parser"] for c in cells}),
        "agents": sorted({c["agent"] for c in cells}),
        "n_docs": len({c["doc_id"] for c in cells}),
        "cells": cells,
        "_rebuilt_from_disk": True,
        "_merged_from": sources,
    }

    out_text = json.dumps(summary, indent=2, default=str)
    if args.in_place:
        out_path = target_dir / "summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Back up the old one if present
        if out_path.exists():
            shutil.copyfile(out_path, out_path.with_suffix(".json.bak"))
        out_path.write_text(out_text)
        n_ok = sum(1 for c in cells if c["status"] == "ok")
        print(
            f"rebuilt {out_path}: {len(cells)} cells "
            f"({n_ok} ok), {summary['n_docs']} docs × "
            f"{len(summary['parsers'])} parsers × {len(summary['agents'])} agents; "
            f"merged from {sources}"
        )
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
