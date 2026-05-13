"""Pydantic models for extraction dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class DocMetadata(BaseModel):
    domain: str | None = None
    doc_type: str | None = None
    difficulty: str | None = None
    page_count: int | None = None
    origin: str | None = None
    license: str | None = None


class DocEntry(BaseModel):
    doc_id: str
    source_path: str                    # relative to manifest dir
    source_sha256: str | None = None
    schema_path: str                    # relative to manifest dir
    ground_truth_path: str              # relative to manifest dir
    metadata: DocMetadata = DocMetadata()
    splits: list[str] = []
    scoring_rules: list[str] = []       # names from realdoc_bench.extraction.scoring.rules


class Manifest(BaseModel):
    version: int
    set_id: str
    defaults: dict[str, Any] = {}
    docs: list[DocEntry]

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


def resolve(manifest_path: Path, rel: str) -> Path:
    return (manifest_path.parent / rel).resolve()


def read_schema(manifest_path: Path, entry: DocEntry) -> dict[str, Any]:
    return json.loads(resolve(manifest_path, entry.schema_path).read_text())


def read_ground_truth(manifest_path: Path, entry: DocEntry) -> dict[str, Any]:
    return json.loads(resolve(manifest_path, entry.ground_truth_path).read_text())


def resolve_source(manifest_path: Path, entry: DocEntry) -> Path:
    return resolve(manifest_path, entry.source_path)
