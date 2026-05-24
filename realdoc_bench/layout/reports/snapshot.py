"""Notion-style snapshot report generator for the layout benchmark.

Sections produced (in order):

1. Headline table — model | F1 | $/page | latency_sec/page, sorted by F1.
2. Per-block-type F1 matrix.
3. Confusion summary.
4. Domain slice (when domain metadata available).
5. Cost / latency Pareto plot file references.

mAP columns are populated only when the processor surfaces real per-block
confidence; otherwise they read ``mAP-N/A`` to avoid misleading numbers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from realdoc_bench.shared.reporting.leaderboard import render_section, render_table
from realdoc_bench.shared.reporting.plots import write_pareto_plot


def _headline_row(proc: str, agg: Mapping[str, Any]) -> dict[str, Any]:
    adj = agg.get("adjusted") or {}
    return {
        "model": proc,
        "version": agg["version"],
        "F1": agg["strict"]["f1"],
        # Populated only by the "adjacency" scorer (adjacency split/merge recovery).
        "adjusted F1": adj.get("f1"),
        "precision": agg["strict"]["precision"],
        "recall": agg["strict"]["recall"],
        "$/page": agg.get("mean_cost_usd_per_page"),
        "latency_sec/page": agg.get("mean_latency_sec"),
        "mAP-ready": "yes" if agg.get("confidence_present") else "no (mAP-N/A)",
    }


def _per_class_matrix(aggregates: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    classes: set[str] = set()
    for agg in aggregates.values():
        classes.update(agg["per_class"].keys())
    class_order = sorted(classes)
    columns = ["block_type", *aggregates.keys()]
    rows: list[dict[str, Any]] = []
    for cls in class_order:
        row: dict[str, Any] = {"block_type": cls}
        for proc, agg in aggregates.items():
            row[proc] = agg["per_class"].get(cls, {}).get("f1")
        rows.append(row)
    return columns, rows


def _confusion_rows(aggregates: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proc, agg in aggregates.items():
        for entry in agg["structural"]["top_confusions"]:
            rows.append(
                {
                    "model": proc,
                    "predicted_as": entry["pred"],
                    "actually_was": entry["gt"],
                    "count": entry["count"],
                }
            )
    return rows


def write_report(
    run_summary: Mapping[str, Any],
    aggregates: Mapping[str, Mapping[str, Any]],
    out_path: Path,
    *,
    cost_pareto_path: Path | None = None,
    latency_pareto_path: Path | None = None,
) -> Path:
    headline = [_headline_row(proc, agg) for proc, agg in aggregates.items()]
    # Only surface the adjusted-F1 column when at least one processor was scored
    # with the "adjacency" scorer (otherwise every cell would be "—").
    has_adjusted = any(agg.get("adjusted") for agg in aggregates.values())
    headline_cols = ["model", "version", "F1"]
    if has_adjusted:
        headline_cols.append("adjusted F1")
    headline_cols += ["precision", "recall", "$/page", "latency_sec/page", "mAP-ready"]
    headline_table = render_table(headline, headline_cols, sort_by="F1")

    class_cols, class_rows = _per_class_matrix(aggregates)
    class_table = render_table(class_rows, class_cols, sort_by="block_type", descending=False)
    confusion_table = render_table(
        _confusion_rows(aggregates),
        ["model", "predicted_as", "actually_was", "count"],
        sort_by="count",
    )

    md_parts: list[str] = [
        f"# realdoc-layout snapshot — `{run_summary['run_id']}`\n",
        f"Pages evaluated: **{run_summary['n_pages']}**. Pricing catalog: `{run_summary['catalog_sha']}`.\n",
        render_section("Headline", headline_table),
        render_section("Per-block-type F1", class_table),
        render_section("Top confusions", confusion_table),
    ]

    if cost_pareto_path is not None:
        md_parts.append(render_section("Quality / cost Pareto", f"![cost pareto]({cost_pareto_path.name})"))
    if latency_pareto_path is not None:
        md_parts.append(render_section("Quality / latency Pareto", f"![latency pareto]({latency_pareto_path.name})"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_parts))
    return out_path


def build_pareto_inputs(
    aggregates: Mapping[str, Mapping[str, Any]],
    *,
    quality_key: str = "F1",
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """Return (cost_points, latency_points) suitable for the pareto plotter.

    Points missing cost or latency are skipped.
    """
    cost_points: list[tuple[str, float, float]] = []
    latency_points: list[tuple[str, float, float]] = []
    for proc, agg in aggregates.items():
        quality = agg["strict"]["f1"] if quality_key == "F1" else agg["strict"][quality_key]
        cost = agg.get("mean_cost_usd_per_page")
        latency = agg.get("mean_latency_sec")
        if cost is not None:
            cost_points.append((proc, float(cost), float(quality)))
        if latency is not None:
            latency_points.append((proc, float(latency), float(quality)))
    return cost_points, latency_points


def write_full(
    run_summary: Mapping[str, Any],
    aggregates: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
) -> Path:
    """Write the snapshot markdown + accompanying pareto plot files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_points, latency_points = build_pareto_inputs(aggregates)

    cost_path = out_dir / "pareto_cost.png"
    latency_path = out_dir / "pareto_latency.png"
    if cost_points:
        annotated_cost = write_pareto_plot(
            cost_points,
            cost_path,
            x_label="$ / page",
            y_label="F1",
            title="realdoc-layout: quality vs. cost",
        )
        (out_dir / "pareto_cost.json").write_text(
            json.dumps([p.__dict__ for p in annotated_cost], indent=2)
        )
    if latency_points:
        annotated_latency = write_pareto_plot(
            latency_points,
            latency_path,
            x_label="sec / page",
            y_label="F1",
            title="realdoc-layout: quality vs. latency",
        )
        (out_dir / "pareto_latency.json").write_text(
            json.dumps([p.__dict__ for p in annotated_latency], indent=2)
        )

    md_path = out_dir / "snapshot.md"
    write_report(
        run_summary,
        aggregates,
        md_path,
        cost_pareto_path=cost_path if cost_points else None,
        latency_pareto_path=latency_path if latency_points else None,
    )
    return md_path
