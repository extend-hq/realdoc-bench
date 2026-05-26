"""realdoc-bench CLI entrypoint.

Two top-level command groups:

- ``layout``    — list processors, run a layout benchmark, render its report.
- ``evaluate``  — download a dataset, parse documents, score against the QA
                   bank, and build the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

# Side-effect imports register all decorated parsers/processors.
from realdoc_bench.evaluate import parsers as _parsers  # noqa: F401
from realdoc_bench.evaluate.parsers.base import registry as parser_registry
from realdoc_bench.evaluate import parse as ev_parse
from realdoc_bench.evaluate import report as ev_report
from realdoc_bench.evaluate import score as ev_score
from realdoc_bench.evaluate.download import download_dataset
from realdoc_bench.evaluate.dotenv import load_dotenv
from realdoc_bench.evaluate.runs import RunLayout
from realdoc_bench.layout import processors as _processors  # noqa: F401
from realdoc_bench.layout.data.loader import DEFAULT_HF_DATASET as DEFAULT_LAYOUT_DATASET
from realdoc_bench.layout.metrics.adjacency import rescore as adjacency_rescore
from realdoc_bench.layout.processors.base import registry as proc_registry
from realdoc_bench.layout.reports.snapshot import write_full as write_layout_report
from realdoc_bench.layout.runner import aggregate_summary as layout_aggregate
from realdoc_bench.layout.runner import run as layout_run
from realdoc_bench.shared.registry import Registry

console = Console()

app = typer.Typer(no_args_is_help=True, help="realdoc-bench — layout + QA-extraction benchmarks")
layout_app = typer.Typer(no_args_is_help=True, help="Layout benchmark.")
evaluate_app = typer.Typer(
    no_args_is_help=True,
    help="QA-extraction benchmark: download → parse → score → report.",
)
app.add_typer(layout_app, name="layout")
app.add_typer(evaluate_app, name="evaluate")


# ── Layout ──────────────────────────────────────────────────────────────────


@layout_app.command("list")
def layout_list() -> None:
    table = Table(title="Layout processors")
    table.add_column("name"); table.add_column("version")
    for name in proc_registry.names():
        cls = proc_registry.get(name)
        table.add_row(name, getattr(cls, "version", ""))
    console.print(table)


@layout_app.command("eval")
def layout_eval(
    processor: list[str] = typer.Option(..., "--processor", "-p"),
    dataset: str = typer.Option(
        DEFAULT_LAYOUT_DATASET, "--dataset",
        help="HF dataset repo id (e.g. Extend-AI/RealDocBench-Layout)",
    ),
    limit: int | None = typer.Option(None, "--limit"),
    domain: list[str] | None = typer.Option(None, "--domain"),
    run_id: str | None = typer.Option(None, "--run-id"),
    run_dir: Path = typer.Option(Path("runs"), "--run-dir"),
    scorer: str = typer.Option(
        "f1",
        "--scorer",
        help=(
            "Layout F1 scorer. 'f1' (default) = realdoc-bench's native greedy-"
            "Hungarian + strict type-gate scorer. 'adjacency' = Hungarian 1:1 "
            "with adjacency split/merge recovery, which also produces an "
            "'adjusted' F1 per processor."
        ),
    ),
) -> None:
    _env()
    if scorer not in ("f1", "adjacency"):
        raise typer.BadParameter(f"--scorer must be 'f1' or 'adjacency', got {scorer!r}")
    _validate_plugin_names(processor, proc_registry, "processor")
    summary = layout_run(
        processors=processor, limit=limit, domains=domain,
        scorer=scorer,  # type: ignore[arg-type]
        run_id=run_id, run_dir=run_dir, dataset=dataset,
    )
    console.print(f"run_id = [bold]{summary.run_id}[/bold]  scorer = [bold]{scorer}[/bold]")
    aggregates = layout_aggregate(summary)
    table = Table(title=f"Layout results ({scorer})")
    table.add_column("processor"); table.add_column("F1")
    if scorer == "adjacency": table.add_column("adjusted F1")
    table.add_column("$/page"); table.add_column("latency_s/page")
    for proc, agg in sorted(aggregates.items(),
                            key=lambda kv: kv[1]["strict"]["f1"], reverse=True):
        row = [proc, f"{agg['strict']['f1']:.3f}"]
        if scorer == "adjacency":
            adj = agg.get("adjusted")
            row.append(f"{adj['f1']:.3f}" if adj else "—")
        row.append(f"{agg['mean_cost_usd_per_page']:.4f}" if agg["mean_cost_usd_per_page"] is not None else "—")
        row.append(f"{agg['mean_latency_sec']:.2f}" if agg["mean_latency_sec"] is not None else "—")
        table.add_row(*row)
    console.print(table)


@layout_app.command("report")
def layout_report(
    run_id: str = typer.Option(..., "--run-id"),
    run_dir: Path = typer.Option(Path("runs"), "--run-dir"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
) -> None:
    _env()
    summary_path = run_dir / run_id / "summary.json"
    summary = json.loads(summary_path.read_text())
    aggregates_path = run_dir / run_id / "aggregates.json"
    if aggregates_path.exists():
        aggregates = json.loads(aggregates_path.read_text())
    else:
        from realdoc_bench.layout.metrics import structural_errors as struct
        from realdoc_bench.layout.metrics.f1 import PRF
        from realdoc_bench.layout.runner import (
            CellResult,
            RunSummary,
            aggregate_summary,
        )

        cells: list[CellResult] = []
        for c in summary["cells"]:
            strict = PRF(**c["strict"]) if c.get("strict") else None
            adjusted = PRF(**c["adjusted"]) if c.get("adjusted") else None
            per_cls = {k: PRF(**v) for k, v in (c.get("per_class") or {}).items()}
            s = c.get("structural")
            if s:
                conf = {tuple(k.split("|", 1)): v for k, v in (s.get("confusion") or {}).items()}
                errors = struct.StructuralErrors(
                    misses=s["misses"], hallucinations=s["hallucinations"],
                    misclassifications=s["misclassifications"], confusion=conf,
                )
            else:
                errors = None
            cells.append(
                CellResult(
                    page_id=c["page_id"], domain=c["domain"], processor=c["processor"],
                    version=c["version"], status=c["status"], error=c.get("error"),
                    latency_sec=c.get("latency_sec"), cost_estimate_usd=c.get("cost_estimate_usd"),
                    confidence_present=c.get("confidence_present", False),
                    strict=strict, adjusted=adjusted, per_class=per_cls, structural=errors,
                    pred_count=c.get("pred_count", 0), gt_count=c.get("gt_count", 0),
                )
            )
        run_summary = RunSummary(
            run_id=summary["run_id"], started_at=summary["started_at"],
            finished_at=summary["finished_at"], catalog_sha=summary["catalog_sha"],
            processors=summary["processors"], n_pages=summary["n_pages"], cells=cells,
            scorer=summary.get("scorer", "f1"),
        )
        aggregates = aggregate_summary(run_summary)

    out_dir = out_dir or (run_dir / run_id / "reports")
    md_path = write_layout_report(summary, aggregates, out_dir)
    console.print(f"report: {md_path}")


@layout_app.command("rescore")
def layout_rescore(
    run_id: str = typer.Option(..., "--run-id", help="Run directory name under --run-dir."),
    run_dir: Path = typer.Option(Path("runs"), "--run-dir"),
    processor: list[str] | None = typer.Option(
        None, "--processor", "-p",
        help="Processor to score (repeatable). Default: every processor in the run."),
    ignore_mode: str = typer.Option(
        "disabled", "--ignore-mode",
        help=f"Optional per-class ignore filter. One of {adjacency_rescore.IGNORE_MODES}."),
    iou: float = typer.Option(0.5, "--iou", help="IoU match threshold."),
    gap_threshold: float = typer.Option(0.02, "--gap-threshold", help="Adjacency gap fraction."),
    limit: int | None = typer.Option(None, "--limit", help="Score only the first N samples."),
) -> None:
    """Re-score an existing layout run with the Hungarian + adjacency metric.

    Reads each run's cached prediction.json and re-scores against ground truth
    with adjacency split/merge recovery — no model calls, no re-running parsers.
    """
    _env()
    try:
        missing = adjacency_rescore.run_rescore(
            run_dir=run_dir, run_id=run_id, processors=processor or [],
            limit=limit,
            ignore_mode=ignore_mode, iou_threshold=iou, gap_threshold=gap_threshold,
        )
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e
    if missing:
        raise typer.Exit(1)


# ── Helpers ─────────────────────────────────────────────────────────────────
# Shared by both `evaluate` and `layout` command groups.

def _env() -> None:
    # Load .env.local first so its values win (load_dotenv skips already-set vars).
    load_dotenv(Path.cwd() / ".env.local")
    load_dotenv(Path.cwd() / ".env")


def _validate_plugin_names(names: list[str], registry: Registry[Any], kind: str) -> None:
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise typer.BadParameter(
            f"unknown {kind} name(s): {unknown}. Available: {registry.names()}"
        )


@evaluate_app.command("list")
def evaluate_list() -> None:
    """List registered parse providers."""
    table = Table(title="Parse providers")
    table.add_column("name"); table.add_column("version")
    for name in parser_registry.names():
        cls = parser_registry.get(name)
        table.add_row(name, getattr(cls, "version", ""))
    console.print(table)


@evaluate_app.command("download")
def evaluate_download(
    run_dir: Path = typer.Option(..., "--run-dir", help="destination run dir"),
    dataset: str = typer.Option(..., "--dataset",
                                help="HF dataset repo id (e.g. extend-ai/realdoc-bench-qa)"),
    revision: str | None = typer.Option(None, "--revision"),
    force: bool = typer.Option(False, "--force",
                                help="re-download even if files already present"),
    limit: int | None = typer.Option(
        None, "--limit",
        help="smoke mode: pull bank + the docs referenced by the first N "
             "bank items (matches parse/score --limit semantics)"),
) -> None:
    """Pull qa_bank.json + docs/ from a HF dataset repo into <run-dir>/."""
    _env()
    layout = RunLayout.at(run_dir)
    res = download_dataset(layout, repo_id=dataset, revision=revision,
                           force=force, limit=limit)
    console.print(
        f"downloaded → {layout.root}: {res['docs']} docs, "
        f"bank={'yes' if res['bank'] else 'NO'}"
    )


@evaluate_app.command("parse")
def evaluate_parse(
    run_dir: Path = typer.Option(..., "--run-dir"),
    parser: list[str] = typer.Option(..., "--parser", "-p",
                                      help="parser name; repeat for several"),
    bank: Path | None = typer.Option(
        None, "--bank",
        help="qa_bank.json; default: <run-dir>/qa_bank.json. Only consulted "
             "when --limit is set, to pick which docs the limit covers."),
    force: bool = typer.Option(False, "--force",
                                help="re-parse even if cached"),
    retry_errors: bool = typer.Option(False, "--retry-errors",
                                       help="retry previously-errored entries"),
    workers: int = typer.Option(8, "--workers"),
    limit: int | None = typer.Option(
        None, "--limit",
        help="smoke mode: parse only the docs referenced by the first N "
             "bank items"),
) -> None:
    """Parse every document under <run-dir>/docs/ with each --parser."""
    _env()
    _validate_plugin_names(parser, parser_registry, "parser")
    layout = RunLayout.at(run_dir)
    records = ev_parse.run_parse(
        layout, parser, bank_path=bank, force=force, retry_errors=retry_errors,
        workers=workers, limit=limit,
    )
    table = Table(title="Parse results")
    for col in ("parser", "docs", "ok", "fail", "cached", "median lat (s)",
                "$/doc"):
        table.add_column(col)
    by_parser: dict[str, list] = {}
    for r in records:
        by_parser.setdefault(r.parser, []).append(r)
    for pname, rs in sorted(by_parser.items()):
        ok = sum(1 for r in rs if r.ok)
        fail = sum(1 for r in rs if not r.ok)
        cached = sum(1 for r in rs if r.cached)
        lats = sorted(r.latency_sec for r in rs if r.ok and not r.cached)
        med = lats[len(lats) // 2] if lats else 0.0
        cost = (sum(r.cost_usd for r in rs if r.ok) / max(ok, 1))
        table.add_row(pname, str(len(rs)), str(ok), str(fail), str(cached),
                      f"{med:.2f}", f"{cost:.4f}")
    console.print(table)


def _score_and_print(layout: RunLayout, parsers: list[str], *,
                     bank_path: Path | None, force: bool, workers: int,
                     limit: int | None) -> list[ev_score.ScoreRecord]:
    records = ev_score.run_score(
        layout, parsers or None, bank_path=bank_path,
        force=force, workers=workers, limit=limit,
    )
    used_parsers = sorted({r.parser for r in records})
    summary = ev_score.summarize(records, used_parsers)
    table = Table(title="QA score")
    for col in ("parser", "per-field %", "field ok/total",
                "per-question %", "q ok/total"):
        table.add_column(col)
    for p, s in sorted(summary.items(), key=lambda kv: -kv[1]["field_pct"]):
        table.add_row(
            p, f"{s['field_pct']:.1f}",
            f"{s['field_ok']}/{s['field_tot']}",
            f"{s['q_pct']:.1f}", f"{s['q_ok']}/{s['q_tot']}",
        )
    console.print(table)
    return records


@evaluate_app.command("score")
def evaluate_score(
    run_dir: Path = typer.Option(..., "--run-dir"),
    parser: list[str] = typer.Option(
        None, "--parser", "-p",
        help="parser name; repeat for several. Default: all parsers with parses."),
    bank: Path | None = typer.Option(None, "--bank",
                                      help="qa_bank.json; default: <run-dir>/qa_bank.json"),
    force: bool = typer.Option(False, "--force",
                                help="re-call the model (rescore on cached hits is free)"),
    workers: int = typer.Option(16, "--workers"),
    limit: int | None = typer.Option(None, "--limit",
                                      help="score only the first N items (smoke)"),
) -> None:
    """Score every (item × parser) pair against the bank.

    On cache hit, the cached answer is re-scored with the current scoring
    code (no model call). To re-score after editing thresholds, just re-run
    this command — there is no separate rescore verb.
    """
    _env()
    layout = RunLayout.at(run_dir)
    _score_and_print(layout, parser or [], bank_path=bank,
                     force=force, workers=workers, limit=limit)
    ev_score.aggregate_results(layout)


@evaluate_app.command("report")
def evaluate_report(
    run_dir: Path = typer.Option(..., "--run-dir"),
    bank: Path | None = typer.Option(None, "--bank"),
    out: Path | None = typer.Option(None, "--out"),
    merge_domains: str | None = typer.Option(
        None, "--merge-domains",
        help="src=dst,src2=dst2 — fold one domain into another for display"),
    capabilities: Path | None = typer.Option(
        None, "--capabilities",
        help="YAML overriding the default capability bucket groupings "
             "(see realdoc_bench/evaluate/capability_buckets.yaml for the schema)"),
) -> None:
    """Build the dashboard from cached eval results."""
    layout = RunLayout.at(run_dir)
    merge: dict[str, str] = {}
    if merge_domains:
        for pair in merge_domains.split(","):
            src, dst = pair.split("=")
            merge[src.strip()] = dst.strip()
    out_path = ev_report.build_report(
        layout, bank_path=bank, out_path=out,
        merge_domains=merge, capabilities_path=capabilities,
    )
    ev_score.aggregate_results(layout)
    console.print(f"dashboard: {out_path}")


@evaluate_app.command("run")
def evaluate_run(
    run_dir: Path = typer.Option(..., "--run-dir"),
    parser: list[str] = typer.Option(..., "--parser", "-p"),
    bank: Path | None = typer.Option(None, "--bank"),
    force: bool = typer.Option(False, "--force"),
    workers: int = typer.Option(8, "--workers"),
    limit: int | None = typer.Option(
        None, "--limit",
        help="smoke mode: parse only the first N docs + score only those Qs"),
) -> None:
    """parse → score → report end-to-end. --force is threaded through both
    parse and score, so you really do redo everything. --limit caps the doc
    set for a smoke run."""
    _env()
    _validate_plugin_names(parser, parser_registry, "parser")
    ev_score.require_api_key()  # fail fast if GEMINI_API_KEY missing — before parsing
    layout = RunLayout.at(run_dir)

    console.rule("[1/3] parse")
    ev_parse.run_parse(layout, parser, bank_path=bank,
                       force=force, workers=workers, limit=limit)

    console.rule("[2/3] score")
    _score_and_print(layout, parser, bank_path=bank, force=force,
                     workers=max(workers * 2, 16), limit=limit)
    ev_score.aggregate_results(layout)

    console.rule("[3/3] report")
    out = ev_report.build_report(layout, bank_path=bank)
    console.print(f"dashboard: {out}")
