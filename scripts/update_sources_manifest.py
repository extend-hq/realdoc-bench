#!/usr/bin/env python3
"""Refresh the doc_pages column in a v1_*/_sources_manifest.csv by
opening every PDF actually present under sources/ with pymupdf.

Usage:
    python scripts/update_sources_manifest.py <v1_folder> [<v1_folder> ...]

If no folders are given, scans every datasets/extraction/v1_*/ folder.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import fitz  # pymupdf


REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"


def update_one(folder: Path) -> None:
    manifest = folder / "_sources_manifest.csv"
    sources = folder / "sources"
    if not manifest.exists():
        print(f"[skip] no manifest in {folder.name}")
        return
    if not sources.exists():
        print(f"[skip] no sources/ in {folder.name}")
        return

    with manifest.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "doc_pages" not in fieldnames:
        print(f"[skip] no doc_pages column in {manifest}")
        return

    changed = 0
    listed = {r["local_filename"] for r in rows}
    on_disk = {p.name for p in sources.glob("*.pdf")}

    for row in rows:
        pdf = sources / row["local_filename"]
        if not pdf.exists():
            row["doc_pages"] = "MISSING"
            changed += 1
            continue
        try:
            with fitz.open(pdf) as doc:
                pages = doc.page_count
            new = str(pages)
            if row["doc_pages"] != new:
                row["doc_pages"] = new
                changed += 1
        except Exception as e:
            row["doc_pages"] = f"ERR:{type(e).__name__}"
            changed += 1

    # Note PDFs present on disk but not in manifest
    untracked = sorted(on_disk - listed)
    if untracked:
        print(f"[warn] {folder.name}: PDFs on disk not in manifest: {untracked}")

    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok]   {folder.name}: {changed} rows updated, {len(rows)} total")


def main() -> int:
    if len(sys.argv) > 1:
        targets = [Path(a) if Path(a).is_absolute() else DATASETS / a for a in sys.argv[1:]]
    else:
        targets = sorted(p for p in DATASETS.glob("v1_*") if p.is_dir())
    for t in targets:
        update_one(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
