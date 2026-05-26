"""realdoc-layout dataset loader.

Resolves the manifest CSV plus image + COCO blobs from the HF dataset
(``Extend-AI/RealDocBench-Layout``) and yields validated samples joining
the two. The dataset is public — no authentication needed. Pass ``manifest_path``
to override the snapshot-shipped manifest (useful for tests or local snapshots).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from realdoc_bench.layout.normalizers.base import LayoutDocument
from realdoc_bench.layout.normalizers.coco import normalize_coco
from realdoc_bench.shared.io.hf import fetch_dataset

DEFAULT_HF_DATASET = "Extend-AI/RealDocBench-Layout"


@dataclass(frozen=True)
class LayoutSample:
    page_id: str
    domain: str
    image_path: Path
    ground_truth: LayoutDocument
    source_url: str | None
    original_image_url: str | None


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open() as f:
        return list(csv.DictReader(f))


def _load_blobs(snapshot_dir: Path, page_id: str) -> tuple[Path, dict]:
    """Locate the image + COCO annotation for ``page_id`` inside the HF snapshot."""
    image_path = next(
        (p for p in (snapshot_dir / "images").glob(f"{page_id}.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        None,
    )
    if image_path is None:
        raise FileNotFoundError(f"no image for {page_id} under {snapshot_dir}/images")
    ann_path = snapshot_dir / "annotations" / f"{page_id}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"no annotations for {page_id} at {ann_path}")
    with ann_path.open() as f:
        coco = json.load(f)
    return image_path, coco


def load(
    *,
    manifest_path: Path | None = None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_revision: str | None = None,
    domains: Iterable[str] | None = None,
    limit: int | None = None,
) -> Iterator[LayoutSample]:
    """Yield :class:`LayoutSample` records for the (optionally filtered) manifest.

    ``manifest_path`` defaults to ``<snapshot>/manifest.csv`` from the HF dataset.
    """
    snapshot = fetch_dataset(hf_dataset, revision=hf_revision)
    rows = _read_manifest(manifest_path or snapshot / "manifest.csv")
    if domains is not None:
        keep_domains = set(domains)
        rows = [r for r in rows if r["domain"] in keep_domains]
    skip_missing = bool(os.environ.get("REALDOC_BENCH_SKIP_MISSING_BLOBS"))
    if not skip_missing and limit is not None:
        rows = rows[:limit]
    yielded = 0
    for row in rows:
        page_id = row["pageId"]
        try:
            image_path, coco = _load_blobs(snapshot, page_id)
        except FileNotFoundError:
            if skip_missing:
                print(f"warning: skipping {page_id} (blob missing under {snapshot})", file=sys.stderr)
                continue
            raise
        gt = normalize_coco(coco, source=image_path.name)
        yielded += 1
        yield LayoutSample(
            page_id=page_id,
            domain=row["domain"],
            image_path=image_path,
            ground_truth=gt,
            source_url=row.get("sourceUrl") or None,
            original_image_url=row.get("originalImageUrl") or None,
        )
        if limit is not None and yielded >= limit:
            return
