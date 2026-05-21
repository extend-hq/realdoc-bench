#!/usr/bin/env python3
"""realdoc-bench extraction pipeline — unified runner.

Usage:
  python -m realdoc_bench.pipeline <command> [options]

Commands:
  parse      run the parsers over the sampled documents    (cached; --force redoes)
  eval       run the answer-extraction agent on the parses (cached; --force redoes)
  rescore    3-layer scoring (strict / fuzzy / LLM-judge) of the eval answers
  report     build the HTML dashboard
  run        parse -> eval -> rescore -> report, end to end

Every command takes --run-dir <dir>. parse / eval / run also accept
--domain and --type to run a subset (e.g. --domain finance medical).
Caching skips work already done — pass --force to recompute.

Run `python -m realdoc_bench.pipeline <command> --help` for per-command options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from realdoc_bench.pipeline import evaluate as eval_stage
from realdoc_bench.pipeline import parse as parse_stage
from realdoc_bench.pipeline import report as report_stage
from realdoc_bench.pipeline import rescore as rescore_stage

_STAGES = {
    "parse": parse_stage.main,
    "eval": eval_stage.main,
    "rescore": rescore_stage.main,
    "report": report_stage.main,
}


def _run_all(rest: list[str]) -> int:
    """``run`` — the full pipeline; threads shared flags into each stage."""
    ap = argparse.ArgumentParser(prog="realdoc_bench.pipeline run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--parsers", nargs="+", default=None)
    ap.add_argument("--domain", nargs="*", default=None,
                    help="run only these document domains (e.g. finance)")
    ap.add_argument("--type", nargs="*", default=None,
                    help="run only these document types")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--profile", default="default",
                    help="fuzzy-scoring profile: default / strict / lenient")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when cached output exists")
    ap.add_argument("--skip-parse", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args(rest)

    subset: list[str] = []
    if a.domain:
        subset += ["--domain", *a.domain]
    if a.type:
        subset += ["--type", *a.type]
    force = ["--force"] if a.force else []

    if not a.skip_parse:
        print("\n=== [1/4] parse ===", flush=True)
        pargs = ["--run-dir", a.run_dir, "--workers", str(a.workers), *subset, *force]
        if a.parsers:
            pargs += ["--parsers", *a.parsers]
        if parse_stage.main(pargs):
            return 1
    if not a.skip_eval:
        print("\n=== [2/4] eval ===", flush=True)
        if eval_stage.main(["--run-dir", a.run_dir, "--max-workers", str(a.workers),
                            *subset, *force]):
            return 1
    print("\n=== [3/4] rescore ===", flush=True)
    if rescore_stage.main(["--run-dir", a.run_dir, "--max-workers", str(a.workers),
                           "--profile", a.profile]):
        return 1
    print("\n=== [4/4] report ===", flush=True)
    out = str(Path(a.run_dir) / "dashboard.html")
    if report_stage.main(["--run-dir", a.run_dir, "--mode", "post", "--out", out]):
        return 1
    print(f"\npipeline complete -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in _STAGES:
        return _STAGES[cmd](rest) or 0
    if cmd == "run":
        return _run_all(rest)
    print(f"unknown command: {cmd!r}\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
