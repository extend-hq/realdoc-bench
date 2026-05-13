"""Extraction-mode runner.

For each ``(doc, parser, agent)`` cell, the runner: parses the source PDF →
feeds the markdown plus the dialect-stripped schema to the agent → validates
the agent's output → scores it against ground truth. All artifacts are written
under ``runs/<run_id>/<doc_id>/<parser>/<agent>/``.

Parses are cached by ``(source_sha256, parser_config_hash)`` so re-running the
grid with new agents doesn't repeat parse work.
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from realdoc_bench.extraction.agents.base import AgentResult
from realdoc_bench.extraction.agents.base import build as build_agent
from realdoc_bench.extraction.manifest import (
    DocEntry,
    Manifest,
    read_ground_truth,
    read_schema,
    resolve_source,
)
from realdoc_bench.extraction.parsers.base import ParseResult
from realdoc_bench.extraction.parsers.base import build as build_parser
from realdoc_bench.extraction.scoring.engine import FieldScore, macro_f1, score
from realdoc_bench.extraction.scoring.profiles import PROFILES, ScoringProfile
from realdoc_bench.extraction.scoring.rules import apply_rules
from realdoc_bench.shared.io.cache import default_cache_root, ensure_dir, sha256_file
from realdoc_bench.shared.pricing.meter import catalog_sha


@dataclass
class CellResult:
    doc_id: str
    parser: str
    parser_version: str
    agent: str
    agent_version: str
    status: str
    error: str | None = None
    parse_latency_sec: float | None = None
    parse_cost_usd: float | None = None
    parse_pages: int | None = None
    agent_latency_sec: float | None = None
    agent_cost_usd: float | None = None
    agent_retries: int = 0
    macro_f1: float | None = None
    field_scores: list[FieldScore] = field(default_factory=list)


@dataclass
class RunSummary:
    run_id: str
    manifest_set_id: str
    started_at: float
    finished_at: float
    catalog_sha: str
    parsers: list[str]
    agents: list[str]
    n_docs: int
    cells: list[CellResult]


def _parse_cache_path(source_sha: str, parser_name: str, config_hash: str) -> Path:
    root = default_cache_root() / "parses" / f"{parser_name}-{config_hash}"
    return ensure_dir(root) / f"{source_sha}.json"


def _load_or_parse(
    parser_name: str,
    source_path: Path,
    *,
    no_cache: bool,
) -> ParseResult:
    parser = build_parser(parser_name)
    source_sha = sha256_file(source_path)
    cache_path = _parse_cache_path(source_sha, parser.name, parser.config_hash())
    if not no_cache and cache_path.exists():
        return ParseResult.model_validate_json(cache_path.read_text())
    result = parser.parse(source_path)
    cache_path.write_text(result.model_dump_json(indent=2))
    return result


def _score_one(
    *,
    expected: dict,
    actual: dict | None,
    schema: dict,
    profile: ScoringProfile,
    rule_names: list[str],
) -> tuple[float, list[FieldScore]]:
    actual_canon = apply_rules(actual, rule_names) if actual is not None and rule_names else actual
    scores = score(expected, actual_canon, schema, profile=profile)
    return macro_f1(scores), scores


def run(
    *,
    manifest_path: Path,
    parsers: list[str],
    agents: list[str],
    doc_ids: list[str] | None = None,
    split: str | None = None,
    profile_name: str = "default",
    run_id: str | None = None,
    run_dir: Path = Path("runs"),
    no_cache: bool = False,
) -> RunSummary:
    manifest = Manifest.load(manifest_path)
    entries: list[DocEntry] = manifest.docs
    if doc_ids:
        keep = set(doc_ids)
        entries = [e for e in entries if e.doc_id in keep]
    if split:
        entries = [e for e in entries if split in e.splits]

    run_id = run_id or time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:6]
    run_root = run_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[profile_name]
    cells: list[CellResult] = []
    started_at = time.time()

    for entry in entries:
        try:
            schema = read_schema(manifest_path, entry)
            expected = read_ground_truth(manifest_path, entry)
            source_path = resolve_source(manifest_path, entry)
        except FileNotFoundError as e:
            for parser_name in parsers:
                for agent_name in agents:
                    cells.append(
                        CellResult(
                            doc_id=entry.doc_id,
                            parser=parser_name,
                            parser_version="",
                            agent=agent_name,
                            agent_version="",
                            status="missing_input",
                            error=str(e),
                        )
                    )
            continue

        for parser_name in parsers:
            try:
                parse_result = _load_or_parse(parser_name, source_path, no_cache=no_cache)
            except Exception as e:
                for agent_name in agents:
                    cells.append(
                        CellResult(
                            doc_id=entry.doc_id,
                            parser=parser_name,
                            parser_version="",
                            agent=agent_name,
                            agent_version="",
                            status="parse_error",
                            error=f"{type(e).__name__}: {e}",
                        )
                    )
                continue

            parse_dir = run_root / entry.doc_id / parser_name
            parse_dir.mkdir(parents=True, exist_ok=True)
            (parse_dir / "parse_result.json").write_text(parse_result.model_dump_json(indent=2))
            (parse_dir / "agent_input.md").write_text(parse_result.markdown)

            for agent_name in agents:
                cell_dir = parse_dir / agent_name
                cell_dir.mkdir(parents=True, exist_ok=True)
                try:
                    agent = build_agent(agent_name)
                    agent_result: AgentResult = agent.extract(
                        parse_result.markdown, schema, entry.doc_id
                    )
                except Exception as e:
                    (cell_dir / "error.txt").write_text(traceback.format_exc())
                    cells.append(
                        CellResult(
                            doc_id=entry.doc_id,
                            parser=parser_name,
                            parser_version=parse_result.version,
                            agent=agent_name,
                            agent_version="",
                            status="agent_error",
                            error=f"{type(e).__name__}: {e}",
                            parse_latency_sec=parse_result.latency_sec,
                            parse_cost_usd=parse_result.cost_estimate_usd,
                            parse_pages=parse_result.pages_processed,
                        )
                    )
                    continue

                (cell_dir / "agent_meta.json").write_text(agent_result.model_dump_json(indent=2))
                if agent_result.value is not None:
                    (cell_dir / "agent_output.json").write_text(
                        json.dumps(agent_result.value, indent=2)
                    )

                if agent_result.status == "PROCESSED":
                    f1, field_scores = _score_one(
                        expected=expected,
                        actual=agent_result.value,
                        schema=schema,
                        profile=profile,
                        rule_names=entry.scoring_rules,
                    )
                    (cell_dir / "score.json").write_text(
                        json.dumps(
                            {
                                "macro_f1": f1,
                                "scores": [asdict(s) for s in field_scores],
                            },
                            indent=2,
                            default=str,
                        )
                    )
                    cells.append(
                        CellResult(
                            doc_id=entry.doc_id,
                            parser=parser_name,
                            parser_version=parse_result.version,
                            agent=agent_name,
                            agent_version=agent.version,
                            status="ok",
                            parse_latency_sec=parse_result.latency_sec,
                            parse_cost_usd=parse_result.cost_estimate_usd,
                            parse_pages=parse_result.pages_processed,
                            agent_latency_sec=agent_result.latency_sec,
                            agent_cost_usd=agent_result.cost_estimate_usd,
                            agent_retries=agent_result.retries,
                            macro_f1=f1,
                            field_scores=field_scores,
                        )
                    )
                else:
                    cells.append(
                        CellResult(
                            doc_id=entry.doc_id,
                            parser=parser_name,
                            parser_version=parse_result.version,
                            agent=agent_name,
                            agent_version=agent.version,
                            status=agent_result.status.lower(),
                            error=agent_result.failure_reason,
                            parse_latency_sec=parse_result.latency_sec,
                            parse_cost_usd=parse_result.cost_estimate_usd,
                            parse_pages=parse_result.pages_processed,
                            agent_latency_sec=agent_result.latency_sec,
                            agent_cost_usd=agent_result.cost_estimate_usd,
                            agent_retries=agent_result.retries,
                        )
                    )

    finished_at = time.time()
    summary = RunSummary(
        run_id=run_id,
        manifest_set_id=manifest.set_id,
        started_at=started_at,
        finished_at=finished_at,
        catalog_sha=catalog_sha(),
        parsers=parsers,
        agents=agents,
        n_docs=len(entries),
        cells=cells,
    )
    (run_root / "summary.json").write_text(
        json.dumps(_summary_to_jsonable(summary), indent=2, default=str)
    )
    return summary


def _summary_to_jsonable(s: RunSummary) -> dict:
    return {
        "run_id": s.run_id,
        "manifest_set_id": s.manifest_set_id,
        "started_at": s.started_at,
        "finished_at": s.finished_at,
        "catalog_sha": s.catalog_sha,
        "parsers": s.parsers,
        "agents": s.agents,
        "n_docs": s.n_docs,
        "cells": [
            {
                "doc_id": c.doc_id,
                "parser": c.parser,
                "parser_version": c.parser_version,
                "agent": c.agent,
                "agent_version": c.agent_version,
                "status": c.status,
                "error": c.error,
                "parse_latency_sec": c.parse_latency_sec,
                "parse_cost_usd": c.parse_cost_usd,
                "parse_pages": c.parse_pages,
                "agent_latency_sec": c.agent_latency_sec,
                "agent_cost_usd": c.agent_cost_usd,
                "agent_retries": c.agent_retries,
                "macro_f1": c.macro_f1,
            }
            for c in s.cells
        ],
    }


def aggregate(summary: RunSummary) -> dict[tuple[str, str], dict]:
    """Roll cells up to per-(parser, agent) means for the leaderboard."""
    bucket: dict[tuple[str, str], list[CellResult]] = {}
    for c in summary.cells:
        bucket.setdefault((c.parser, c.agent), []).append(c)

    out: dict[tuple[str, str], dict] = {}
    for (parser, agent), cells in bucket.items():
        ok = [c for c in cells if c.status == "ok" and c.macro_f1 is not None]
        if not ok:
            out[(parser, agent)] = {
                "n_ok": 0,
                "n_total": len(cells),
                "mean_f1": None,
                "mean_parse_latency_sec": None,
                "mean_agent_latency_sec": None,
                "mean_total_cost_usd": None,
                "errors": [c.status for c in cells],
            }
            continue
        out[(parser, agent)] = {
            "n_ok": len(ok),
            "n_total": len(cells),
            "mean_f1": sum(c.macro_f1 for c in ok) / len(ok),
            "mean_parse_latency_sec": _mean([c.parse_latency_sec for c in ok]),
            "mean_agent_latency_sec": _mean([c.agent_latency_sec for c in ok]),
            "mean_total_cost_usd": _mean(
                [
                    (c.parse_cost_usd or 0.0) + (c.agent_cost_usd or 0.0)
                    for c in ok
                ]
            ),
        }
    return out


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
