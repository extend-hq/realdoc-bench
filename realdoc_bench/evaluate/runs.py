"""On-disk layout for an evaluation run.

A run dir holds everything for one (dataset, parser-set) experiment:

    <run-dir>/
      docs/                            input documents (PDFs)
      qa_bank.json                     typed-template QA bank
      parses/<parser>/<stem>.md        parser output
      parses/<parser>/<stem>.json      parse metadata
      eval/cache/<qid>__<parser>.json  per-(question, parser) scoring cache
      eval/results.json                flat snapshot rebuilt by `report`
      dashboard.html                   report output

`RunLayout` centralizes those paths so the rest of the package never opens its
own `Path(...)` against the run dir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_SAFE_QID = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class RunLayout:
    """Resolved paths under one run directory."""

    root: Path

    @classmethod
    def at(cls, run_dir: str | Path) -> "RunLayout":
        return cls(Path(run_dir).resolve())

    # ── Inputs ──────────────────────────────────────────────────────────────
    @property
    def docs_dir(self) -> Path:
        """Input documents (PDFs) — `<run-dir>/docs/`."""
        return self.root / "docs"

    @property
    def bank_path(self) -> Path:
        """Default bank location — `<run-dir>/qa_bank.json`. CLI commands can
        override with `--bank PATH` so the bank can live elsewhere."""
        return self.root / "qa_bank.json"

    # ── Parse stage ─────────────────────────────────────────────────────────
    @property
    def parses_root(self) -> Path:
        return self.root / "parses"

    def parser_dir(self, parser: str) -> Path:
        return self.parses_root / parser

    def parse_md(self, parser: str, stem: str) -> Path:
        return self.parser_dir(parser) / f"{stem}.md"

    def parse_meta(self, parser: str, stem: str) -> Path:
        return self.parser_dir(parser) / f"{stem}.json"

    def parsed_parsers(self) -> list[str]:
        """Every parser that has a populated dir under `parses/`. Used by
        `score` when `-p` is omitted to fall back to 'score everything that's
        been parsed'."""
        if not self.parses_root.exists():
            return []
        return sorted(
            p.name for p in self.parses_root.iterdir()
            if p.is_dir() and any(p.glob("*.md"))
        )

    # ── Score stage ─────────────────────────────────────────────────────────
    @property
    def cache_dir(self) -> Path:
        return self.root / "eval" / "cache"

    def cache_path(self, qid: str, parser: str) -> Path:
        return self.cache_dir / f"{_SAFE_QID.sub('_', qid)}__{parser}.json"

    @property
    def results_path(self) -> Path:
        return self.root / "eval" / "results.json"

    # ── Report ──────────────────────────────────────────────────────────────
    @property
    def dashboard_path(self) -> Path:
        return self.root / "dashboard.html"

    # ── Setup ───────────────────────────────────────────────────────────────
    def ensure_dirs(self) -> None:
        for d in (self.docs_dir, self.parses_root, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
