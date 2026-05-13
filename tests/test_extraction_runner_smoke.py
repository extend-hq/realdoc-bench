"""Smoke test: feed the runner a tiny synthetic doc + a mock agent.

This exercises manifest parsing, parser caching, agent invocation, scoring,
and JSON artifact emission without touching any external API.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from realdoc_bench.extraction.agents.base import AgentResult, ExtractionAgent, registry as agent_registry
from realdoc_bench.extraction.runner import aggregate, run


class _StubAgent(ExtractionAgent):
    name = "stub_agent"
    version = "1.0.0"

    def __init__(self) -> None:
        pass

    def extract(self, markdown: str, schema: dict, doc_id: str) -> AgentResult:
        del markdown, doc_id
        # Return the expected GT (we read it from a side channel below).
        return AgentResult(
            value=_STUB_VALUE,
            status="PROCESSED",
            latency_sec=0.01,
            usage={"input_tokens": 10, "output_tokens": 5},
            cost_estimate_usd=0.0,
            retries=0,
        )


_STUB_VALUE: dict = {}


def test_runner_end_to_end(tmp_path: Path) -> None:
    if "stub_agent" not in agent_registry:
        agent_registry.register("stub_agent", _StubAgent)

    # Build a tiny manifest: one doc, one fake PDF (just bytes), one schema, one GT.
    docs_root = tmp_path
    sources = docs_root / "sources"; sources.mkdir()
    schemas = docs_root / "schemas"; schemas.mkdir()
    gts = docs_root / "ground_truth"; gts.mkdir()
    (sources / "doc1.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["name"],
    }
    expected = {"name": "alice", "n": 42}
    (schemas / "doc1.json").write_text(json.dumps(schema))
    (gts / "doc1.json").write_text(json.dumps(expected))

    global _STUB_VALUE
    _STUB_VALUE = expected

    manifest_path = docs_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "set_id": "smoke",
                "defaults": {},
                "docs": [
                    {
                        "doc_id": "doc1",
                        "source_path": "sources/doc1.pdf",
                        "schema_path": "schemas/doc1.json",
                        "ground_truth_path": "ground_truth/doc1.json",
                        "metadata": {"domain": "test", "doc_type": "synthetic"},
                        "splits": ["test"],
                        "scoring_rules": [],
                    }
                ],
            }
        )
    )

    # We need a parser that doesn't try to open the fake PDF as a real PDF.
    # Register a stub parser inline.
    from realdoc_bench.extraction.parsers.base import (
        ParseProvider,
        ParseResult,
        registry as parser_registry,
    )

    class _StubParser(ParseProvider):
        name = "stub_parser"
        version = "1.0.0"

        def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
            return ParseResult(
                markdown="hello world",
                page_count=1,
                latency_sec=0.001,
                cost_estimate_usd=0.0,
                pages_processed=1,
                provider=self.name,
                version=self.version,
                config_hash=self.config_hash(),
            )

    if "stub_parser" not in parser_registry:
        parser_registry.register("stub_parser", _StubParser)

    summary = run(
        manifest_path=manifest_path,
        parsers=["stub_parser"],
        agents=["stub_agent"],
        run_dir=tmp_path / "runs",
        no_cache=True,
    )
    assert summary.n_docs == 1
    assert len(summary.cells) == 1
    assert summary.cells[0].status == "ok"
    assert summary.cells[0].macro_f1 == 1.0

    agg = aggregate(summary)
    assert agg[("stub_parser", "stub_agent")]["mean_f1"] == 1.0

    # Artifacts exist.
    cell_dir = tmp_path / "runs" / summary.run_id / "doc1" / "stub_parser" / "stub_agent"
    assert (cell_dir / "agent_output.json").exists()
    assert (cell_dir / "agent_meta.json").exists()
    assert (cell_dir / "score.json").exists()
