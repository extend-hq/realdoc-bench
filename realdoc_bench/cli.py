"""realdoc-bench CLI entrypoint.

Two top-level command groups:

- ``layout`` — list processors, run a layout benchmark, render the snapshot report.
- ``extract`` — list parsers/agents, run an extraction grid, score it, render a report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Side-effect imports register all decorated parsers/agents/processors.
from realdoc_bench.extraction import agents as _agents  # noqa: F401
from realdoc_bench.extraction import parsers as _parsers  # noqa: F401
from realdoc_bench.extraction.agents.base import registry as agent_registry
from realdoc_bench.extraction.manifest import Manifest
from realdoc_bench.extraction.parsers.base import registry as parser_registry
from realdoc_bench.extraction.runner import aggregate as ext_aggregate
from realdoc_bench.extraction.runner import run as ext_run
from realdoc_bench.layout import processors as _processors  # noqa: F401
from realdoc_bench.layout.processors.base import registry as proc_registry
from realdoc_bench.layout.runner import aggregate_summary as layout_aggregate
from realdoc_bench.layout.runner import run as layout_run
from realdoc_bench.layout.reports.snapshot import write_full as write_layout_report

console = Console()

app = typer.Typer(no_args_is_help=True, help="realdoc-bench — layout + extraction benchmarks")
layout_app = typer.Typer(no_args_is_help=True, help="Layout benchmark.")
extract_app = typer.Typer(no_args_is_help=True, help="Extraction-quality benchmark.")
app.add_typer(layout_app, name="layout")
app.add_typer(extract_app, name="extract")


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
    processor: list[str] = typer.Option(..., "--processor", "-p", help="Processor names. Repeat for several."),
    dataset: str = typer.Option("realdoc-layout", "--dataset"),
    limit: int | None = typer.Option(None, "--limit"),
    split: list[str] | None = typer.Option(None, "--split"),
    domain: list[str] | None = typer.Option(None, "--domain"),
    run_id: str | None = typer.Option(None, "--run-id"),
    run_dir: Path = typer.Option(Path("runs"), "--run-dir"),
) -> None:
    del dataset  # only realdoc-layout for now
    summary = layout_run(
        processors=processor,
        limit=limit,
        splits=split,
        domains=domain,
        run_id=run_id,
        run_dir=run_dir,
    )
    console.print(f"run_id = [bold]{summary.run_id}[/bold]")
    aggregates = layout_aggregate(summary)
    table = Table(title="Layout results")
    table.add_column("processor"); table.add_column("F1"); table.add_column("$/page"); table.add_column("latency_s/page")
    for proc, agg in sorted(aggregates.items(), key=lambda kv: kv[1]["strict"]["f1"], reverse=True):
        table.add_row(
            proc,
            f"{agg['strict']['f1']:.3f}",
            f"{agg['mean_cost_usd_per_page']:.4f}" if agg["mean_cost_usd_per_page"] is not None else "—",
            f"{agg['mean_latency_sec']:.2f}" if agg["mean_latency_sec"] is not None else "—",
        )
    console.print(table)


@layout_app.command("report")
def layout_report(
    run_id: str = typer.Option(..., "--run-id"),
    run_dir: Path = typer.Option(Path("runs"), "--run-dir"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
) -> None:
    summary_path = run_dir / run_id / "summary.json"
    summary = json.loads(summary_path.read_text())
    aggregates_path = run_dir / run_id / "aggregates.json"
    if aggregates_path.exists():
        aggregates = json.loads(aggregates_path.read_text())
    else:
        # Reconstruct from summary cells.
        from realdoc_bench.layout.runner import (
            CellResult,
            RunSummary,
            aggregate_summary,
        )
        from realdoc_bench.layout.metrics.f1 import PRF
        from realdoc_bench.layout.metrics import structural_errors as struct

        cells: list[CellResult] = []
        for c in summary["cells"]:
            strict = PRF(**c["strict"]) if c.get("strict") else None
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
                    confidence_present=c.get("confidence_present", False), strict=strict, per_class=per_cls,
                    structural=errors, pred_count=c.get("pred_count", 0), gt_count=c.get("gt_count", 0),
                )
            )
        run_summary = RunSummary(
            run_id=summary["run_id"], started_at=summary["started_at"],
            finished_at=summary["finished_at"], catalog_sha=summary["catalog_sha"],
            processors=summary["processors"], n_pages=summary["n_pages"], cells=cells,
        )
        aggregates = aggregate_summary(run_summary)

    out_dir = out_dir or (run_dir / run_id / "reports")
    md_path = write_layout_report(summary, aggregates, out_dir)
    console.print(f"report: {md_path}")


# ── Extraction ──────────────────────────────────────────────────────────────


@extract_app.command("list")
def extract_list(what: str = typer.Argument("parsers", help="parsers | agents | datasets")) -> None:
    if what == "parsers":
        names = parser_registry.names()
        title = "Parse providers"
    elif what == "agents":
        names = agent_registry.names()
        title = "Extraction agents"
    elif what == "datasets":
        datasets_root = Path(__file__).resolve().parents[1] / "datasets" / "extraction"
        names = [p.parent.name for p in datasets_root.glob("*/manifest.yaml")]
        title = "Extraction datasets"
    else:
        raise typer.BadParameter("expected one of: parsers, agents, datasets")
    table = Table(title=title)
    table.add_column("name")
    for n in names:
        table.add_row(n)
    console.print(table)


def _resolve_manifest(dataset: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "datasets" / "extraction" / dataset / "manifest.yaml"
    if not path.exists():
        raise typer.BadParameter(f"no manifest at {path}")
    return path


@extract_app.command("run")
def extract_run(
    dataset: str = typer.Option("v0", "--dataset"),
    parser: list[str] = typer.Option(..., "--parser", "-p"),
    agent: list[str] = typer.Option(..., "--agent", "-a"),
    doc_id: list[str] | None = typer.Option(None, "--doc-id"),
    split: str | None = typer.Option(None, "--split"),
    profile: str = typer.Option("default", "--profile"),
    run_id: str | None = typer.Option(None, "--run-id"),
    run_dir: Path = typer.Option(Path("runs"), "--run-dir"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    manifest_path = _resolve_manifest(dataset)
    summary = ext_run(
        manifest_path=manifest_path,
        parsers=parser,
        agents=agent,
        doc_ids=doc_id,
        split=split,
        profile_name=profile,
        run_id=run_id,
        run_dir=run_dir,
        no_cache=no_cache,
    )
    console.print(f"run_id = [bold]{summary.run_id}[/bold]")
    aggregates = ext_aggregate(summary)
    table = Table(title=f"Extraction results — {dataset}")
    table.add_column("parser"); table.add_column("agent")
    table.add_column("F1"); table.add_column("$/doc"); table.add_column("latency_s/doc")
    for (p_name, a_name), agg in sorted(
        aggregates.items(),
        key=lambda kv: (kv[1].get("mean_f1") or 0.0),
        reverse=True,
    ):
        table.add_row(
            p_name,
            a_name,
            f"{agg['mean_f1']:.3f}" if agg.get("mean_f1") is not None else "—",
            f"{agg['mean_total_cost_usd']:.4f}" if agg.get("mean_total_cost_usd") is not None else "—",
            f"{(agg.get('mean_parse_latency_sec') or 0.0) + (agg.get('mean_agent_latency_sec') or 0.0):.2f}",
        )
    console.print(table)


@extract_app.command("schema-walk")
def extract_schema_walk(
    dataset: str = typer.Option("v0", "--dataset"),
    doc_id: str = typer.Option(..., "--doc-id"),
) -> None:
    """Print every leaf path the scoring engine would walk for a given doc's schema."""
    from realdoc_bench.extraction.manifest import Manifest as M
    from realdoc_bench.extraction.schema.walk import walk_leaves

    manifest_path = _resolve_manifest(dataset)
    manifest = M.load(manifest_path)
    entry = next((d for d in manifest.docs if d.doc_id == doc_id), None)
    if entry is None:
        raise typer.BadParameter(f"no such doc_id: {doc_id}")
    schema_path = manifest_path.parent / entry.schema_path
    schema = json.loads(schema_path.read_text())
    table = Table(title=f"Schema leaves — {doc_id}")
    table.add_column("path"); table.add_column("type"); table.add_column("difficulty"); table.add_column("capabilities")
    for leaf in walk_leaves(schema):
        table.add_row(leaf.path, leaf.leaf_type, leaf.difficulty or "—", ",".join(leaf.capabilities) or "—")
    console.print(table)
