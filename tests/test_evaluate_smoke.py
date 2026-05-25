"""End-to-end smoke: parse → score → re-score (cached) → report on a synthetic
2-doc / 4-question fixture. Uses a stub parser + a stub Gemini extract fn so
nothing hits the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realdoc_bench.evaluate import parse as ev_parse
from realdoc_bench.evaluate import report as ev_report
from realdoc_bench.evaluate import score as ev_score
from realdoc_bench.evaluate.runs import RunLayout
from realdoc_bench.evaluate.parsers.base import (
    ParseProvider,
    ParseResult,
    register_parser,
    registry,
)


@register_parser("stub_parser", version="1.0.0")
class _StubParser(ParseProvider):
    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        # Markdown encodes the doc stem so the stub extractor can answer
        # questions by reading it.
        return ParseResult(
            markdown=f"# {pdf_path.stem}\namount = 100\nstatus = active\n",
            page_count=1, latency_sec=0.001, cost_estimate_usd=0.0,
            pages_processed=1, provider="stub_parser", version="1.0.0",
            config_hash="stub",
        )


def _stub_extract(question: str, template: str, markdown: str):
    # Answer every key with the value embedded in the markdown.
    if "amount" in template:
        return {"amount": 100}
    if "status" in template:
        return {"status": "active"}
    return {"answer": "stub"}


def _explode(*a, **k):
    raise AssertionError("gemini_extract called on a fully-cached run")


@pytest.fixture(autouse=True)
def _fake_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy run_score's require_api_key() preflight without a real key.

    WARNING: this fixture only sets the env var. Any new test that exercises
    a path which calls ``gemini_extract`` MUST also monkeypatch it with a
    stub (see existing tests for the pattern), otherwise the test will make
    a real network call to the Gemini API.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test")


@pytest.fixture
def smoke_run(tmp_path: Path) -> RunLayout:
    layout = RunLayout.at(tmp_path / "smoke")
    layout.ensure_dirs()
    # Need real PDFs on disk; their content is irrelevant — stub parser ignores
    # the bytes. Write minimal PDF headers so file-extension checks survive.
    for name in ("doc_a", "doc_b"):
        (layout.docs_dir / f"{name}.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

    items = [
        {
            "question_id": f"q_{i}",
            "source_file": stem,
            "domain": "test",
            "question": "what is the amount?",
            "response_format": "Return exactly: amount=<number>",
            "gold_answer": "amount=100",
            "template": '{\n  "amount": <number>\n}',
            "gold_dict": {"amount": 100},
        }
        for i, stem in enumerate(["doc_a", "doc_a", "doc_b", "doc_b"])
    ]
    layout.bank_path.write_text(json.dumps({"items": items}))
    return layout


def test_parse_then_score_then_report(smoke_run: RunLayout, monkeypatch: pytest.MonkeyPatch):
    assert "stub_parser" in registry

    parse_recs = ev_parse.run_parse(smoke_run, ["stub_parser"], workers=2,
                                     progress=False)
    assert len(parse_recs) == 2
    assert all(r.ok for r in parse_recs)
    assert (smoke_run.parser_dir("stub_parser") / "doc_a.md").exists()

    monkeypatch.setattr(ev_score, "gemini_extract", _stub_extract)
    score_recs = ev_score.run_score(
        smoke_run, ["stub_parser"], workers=2, progress=False,
    )
    assert len(score_recs) == 4
    assert all(r.ok and r.match for r in score_recs)
    cache_files = list(smoke_run.cache_dir.glob("*.json"))
    assert len(cache_files) == 4

    # Second call must be entirely cache-served, NO model calls.
    monkeypatch.setattr(ev_score, "gemini_extract", _explode)
    score_recs2 = ev_score.run_score(
        smoke_run, ["stub_parser"], workers=2, progress=False,
    )
    assert all(r.cached and r.match for r in score_recs2)

    # Per-parser scope — adding a second parser leaves stub_parser caches
    # alone (verified by reading mtimes).
    mtimes_before = {f.name: f.stat().st_mtime for f in cache_files}
    # No second parser exists; just rerun and confirm mtimes unchanged.
    ev_score.run_score(smoke_run, ["stub_parser"], workers=2, progress=False)
    for f in cache_files:
        assert f.stat().st_mtime == mtimes_before[f.name]

    out = ev_report.build_report(smoke_run)
    assert out.exists()
    html = out.read_text()
    assert "stub_parser" in html or "PyMuPDF" in html or "Stub" in html or "summary" in html


def test_score_defaults_to_parsed_parsers(smoke_run: RunLayout, monkeypatch: pytest.MonkeyPatch):
    ev_parse.run_parse(smoke_run, ["stub_parser"], workers=2, progress=False)
    monkeypatch.setattr(ev_score, "gemini_extract", _stub_extract)
    # No -p list → should auto-pick "stub_parser" from parses/
    recs = ev_score.run_score(smoke_run, None, workers=2, progress=False)
    assert {r.parser for r in recs} == {"stub_parser"}


def test_score_rescore_on_cache_hit(smoke_run: RunLayout, monkeypatch: pytest.MonkeyPatch):
    ev_parse.run_parse(smoke_run, ["stub_parser"], workers=2, progress=False)
    monkeypatch.setattr(ev_score, "gemini_extract", _stub_extract)
    ev_score.run_score(smoke_run, ["stub_parser"], workers=2, progress=False)
    # Corrupt one cache entry to look like a wrong answer; re-run with same
    # gold; the cached answer is still the wrong one and must STAY wrong
    # (re-scoring against unchanged gold).
    a_cache = next(smoke_run.cache_dir.glob("q_0*.json"))
    rec = json.loads(a_cache.read_text())
    rec["answer"] = {"amount": 999}
    rec["field_matches"] = {"amount": True}  # lie — say it matched
    rec["match"] = True
    a_cache.write_text(json.dumps(rec))

    recs = ev_score.run_score(smoke_run, ["stub_parser"], workers=2,
                               progress=False)
    # The corrupt entry should now read back as cached, but re-scored to a miss.
    corrupted = [r for r in recs if r.qid == "q_0"][0]
    assert corrupted.cached
    assert corrupted.match is False        # rescore caught the lie
    rec_after = json.loads(a_cache.read_text())
    assert rec_after["match"] is False
    assert rec_after["field_matches"] == {"amount": False}
