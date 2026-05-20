#!/usr/bin/env python3
"""Pilot 50: parse each of the 50 sampled pages with 4 parsers.

Output:
    runs/pilot_50/parses/<parser>/<dest_path>.md      ← markdown
    runs/pilot_50/parses/<parser>/<dest_path>.json    ← {markdown, page_count, latency, cost, error}
    runs/pilot_50/parses/_manifest.csv                 ← per (parser, page) row

Idempotent: skips parser+page combos whose .md already exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

V12F = REPO / "datasets" / "extraction" / "v1.2_final"
# Module-level globals — assigned from --run-dir CLI arg in main()
PILOT = REPO / "runs" / "pilot_50"
SAMPLE_CSV = PILOT / "sample.csv"
PARSES_ROOT = PILOT / "parses"


def _set_run_dir(run_dir: Path) -> None:
    global PILOT, SAMPLE_CSV, PARSES_ROOT
    PILOT = run_dir
    SAMPLE_CSV = PILOT / "sample.csv"
    PARSES_ROOT = PILOT / "parses"


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


def out_paths(parser_name: str, dest_path: str) -> tuple[Path, Path]:
    """parses/<parser>/<dest_path-with-.pdf-stripped>.{md,json}"""
    stem = Path(dest_path).with_suffix("").as_posix()  # strip .pdf
    base = PARSES_ROOT / parser_name / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_suffix(".md"), base.with_suffix(".json")


def parse_one(parser, parser_name: str, pdf: Path, dest_path: str) -> dict:
    md_path, json_path = out_paths(parser_name, dest_path)
    if md_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        return {**meta, "parser": parser_name, "dest_path": dest_path,
                "cached": True}
    t0 = time.time()
    try:
        result = parser.parse(pdf)
        md = result.markdown or ""
        meta = {
            "parser": parser_name,
            "dest_path": dest_path,
            "ok": True,
            "page_count": result.page_count or 0,
            "latency_sec": result.latency_sec or 0.0,
            "cost_usd": result.cost_estimate_usd or 0.0,
            "md_length": len(md),
            "elapsed_total": time.time() - t0,
            "error": "",
        }
        md_path.write_text(md)
        json_path.write_text(json.dumps(meta, indent=2))
        meta["cached"] = False
        return meta
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
        meta = {
            "parser": parser_name,
            "dest_path": dest_path,
            "ok": False,
            "page_count": 0, "latency_sec": 0.0, "cost_usd": 0.0,
            "md_length": 0, "elapsed_total": time.time() - t0,
            "error": err,
        }
        # Don't write the .md on error; do write the meta so we don't retry
        # forever — re-run with --retry-errors to retry.
        json_path.write_text(json.dumps(meta, indent=2))
        meta["cached"] = False
        return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=REPO / "runs" / "pilot_50",
                    help="run output directory (sample.csv + parses/ under here)")
    ap.add_argument("--parsers", nargs="+",
                    default=["extend_performance_v1_0_0",
                             "extend_performance_v2_0_0_advanced",
                             "aws_textract",
                             "azure_di"])
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent parses PER PARSER (each parser also runs in its own slot).")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-attempt page+parser combos that previously errored.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    _set_run_dir(args.run_dir.resolve())

    # Register parsers
    import realdoc_bench.extraction.parsers  # noqa
    from realdoc_bench.extraction.parsers import extend, aws_textract, azure_di  # noqa
    from realdoc_bench.extraction.parsers.base import registry as parser_registry

    rows = list(csv.DictReader(SAMPLE_CSV.open()))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Sample: {len(rows)} pages × {len(args.parsers)} parsers = {len(rows) * len(args.parsers)} jobs")
    print(f"Parsers: {args.parsers}\n", flush=True)

    PARSES_ROOT.mkdir(parents=True, exist_ok=True)

    # Build skip set
    if not args.retry_errors:
        # Skip if .json exists AND it succeeded — leave errored entries to retry on
        # explicit --retry-errors
        def is_done(parser_name, dest_path):
            _, j = out_paths(parser_name, dest_path)
            if not j.exists(): return False
            try: return json.loads(j.read_text()).get("ok", False)
            except: return False
    else:
        def is_done(parser_name, dest_path):
            md, j = out_paths(parser_name, dest_path)
            return md.exists() and j.exists()

    # Instantiate parsers (one each, threadsafe-ish)
    parsers = {}
    for name in args.parsers:
        cls = parser_registry.get(name)
        if cls is None:
            print(f"  ! parser '{name}' not registered, skipping", file=sys.stderr)
            continue
        parsers[name] = cls()

    # Build job list
    jobs = []  # (parser_name, parser_instance, pdf, dest_path)
    for r in rows:
        dest_path = r["dest_path"]
        pdf = V12F / dest_path
        if not pdf.exists():
            print(f"  ! missing pdf {pdf}", file=sys.stderr)
            continue
        for pname, pinst in parsers.items():
            if is_done(pname, dest_path):
                continue
            jobs.append((pname, pinst, pdf, dest_path))
    print(f"To run: {len(jobs)} (cached: {len(rows) * len(parsers) - len(jobs)})\n", flush=True)

    all_results = []
    t0 = time.time()

    # Run jobs grouped by parser (parallel within parser, sequential across)
    # so we don't blow up any single API.
    by_parser = {pname: [] for pname in parsers}
    for j in jobs: by_parser[j[0]].append(j)

    for pname, plist in by_parser.items():
        if not plist: continue
        print(f"=== {pname}: {len(plist)} jobs ===", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(parse_one, j[1], j[0], j[2], j[3]): j for j in plist}
            ok = fail = 0
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                if r["ok"]: ok += 1
                else: fail += 1
                if i % 5 == 0 or i == len(plist):
                    print(f"  {pname}: {i}/{len(plist)}  ok={ok} fail={fail}  "
                          f"({(time.time()-t0)/60:.1f}min)", flush=True)
                all_results.append(r)
        print()

    # Manifest
    if all_results:
        m_path = PARSES_ROOT / "_manifest.csv"
        fields = ["parser", "dest_path", "ok", "page_count", "latency_sec",
                  "cost_usd", "md_length", "elapsed_total", "error", "cached"]
        with m_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL,
                              extrasaction="ignore")
            w.writeheader()
            for r in sorted(all_results, key=lambda r: (r["parser"], r["dest_path"])):
                w.writerow(r)
        print(f"Manifest: {m_path.relative_to(REPO)}")

    wall = time.time() - t0
    print(f"\nDone in {wall/60:.1f}min — {sum(1 for r in all_results if r['ok'])} ok, "
          f"{sum(1 for r in all_results if not r['ok'])} fail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
