#!/usr/bin/env python3
"""Split every PDF in v1.1 into per-page PDFs to create a pagewise dataset.

Input:  datasets/extraction/v1.1/<dom>/<typ>/sources/<doc>.pdf  (multi-page)
Output: datasets/extraction/v1.1_pagewise/<dom>/<typ>/<doc_stem>__p<NN>.pdf
        + manifest.csv with one row per page

Free, fast — uses pymupdf to extract single-page PDFs.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
DEST = REPO / "datasets" / "extraction" / "v1.1_pagewise"


def discover(v1_1: Path) -> list[tuple[str, str, str, Path, int]]:
    out = []
    for domain_dir in sorted(d for d in v1_1.iterdir() if d.is_dir() and not d.name.startswith("_")):
        for type_dir in sorted(t for t in domain_dir.iterdir() if t.is_dir()):
            sources = type_dir / "sources"
            if not sources.exists():
                continue
            for pdf in sorted(sources.glob("*.pdf")):
                try:
                    with fitz.open(pdf) as d:
                        pc = d.page_count
                except Exception:
                    continue
                out.append((domain_dir.name, type_dir.name, pdf.name, pdf, pc))
    return out


def split_one(args):
    domain, type_name, fname, src_pdf, pc = args
    dest_type_dir = DEST / domain / type_name
    dest_type_dir.mkdir(parents=True, exist_ok=True)
    stem = fname.removesuffix(".pdf")
    pad = max(2, len(str(pc)))  # at least 2 digits, more if doc has >99 pages
    rows: list[dict] = []
    try:
        with fitz.open(src_pdf) as src:
            for i in range(pc):
                out_name = f"{stem}__p{i+1:0{pad}d}.pdf"
                out_path = dest_type_dir / out_name
                if out_path.exists():
                    rows.append({
                        "domain": domain, "type": type_name,
                        "source_filename": fname, "page_idx": i, "page_count_total": pc,
                        "page_pdf": str(out_path.relative_to(REPO)),
                        "size_bytes": out_path.stat().st_size, "action": "skip",
                    })
                    continue
                new_doc = fitz.open()
                new_doc.insert_pdf(src, from_page=i, to_page=i)
                new_doc.save(out_path, garbage=4, deflate=True)
                new_doc.close()
                rows.append({
                    "domain": domain, "type": type_name,
                    "source_filename": fname, "page_idx": i, "page_count_total": pc,
                    "page_pdf": str(out_path.relative_to(REPO)),
                    "size_bytes": out_path.stat().st_size, "action": "wrote",
                })
        return rows
    except Exception as e:
        return [{
            "domain": domain, "type": type_name,
            "source_filename": fname, "page_idx": -1, "page_count_total": pc,
            "page_pdf": "", "size_bytes": 0, "action": f"ERR {e}",
        }]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    docs = discover(V1_1)
    total_pages = sum(d[4] for d in docs)
    print(f"{len(docs)} docs, {total_pages} pages → splitting into {DEST.relative_to(REPO)}/")

    t0 = time.time()
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(split_one, d) for d in docs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows = fut.result()
            all_rows.extend(rows)
            if i % 20 == 0 or i == len(docs):
                print(f"  [{i}/{len(docs)}] docs split", flush=True)

    # Write manifest
    manifest = DEST / "manifest.csv"
    with manifest.open("w", newline="") as f:
        fields = ["domain", "type", "source_filename", "page_idx", "page_count_total",
                  "page_pdf", "size_bytes", "action"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(all_rows, key=lambda r: (r["domain"], r["type"], r["source_filename"], r["page_idx"])):
            w.writerow(r)

    wall = time.time() - t0
    wrote = sum(1 for r in all_rows if r["action"] == "wrote")
    skipped = sum(1 for r in all_rows if r["action"] == "skip")
    errs = sum(1 for r in all_rows if r["action"].startswith("ERR"))
    print(f"\nDone in {wall:.1f}s — {wrote} wrote, {skipped} skipped (already existed), {errs} errors")
    print(f"  manifest: {manifest.relative_to(REPO)}")
    print(f"  per-page PDFs: {DEST.relative_to(REPO)}/<domain>/<type>/")

    # Quick sanity counts
    import os as _os
    total_bytes = sum(p.stat().st_size for p in DEST.rglob("*.pdf"))
    print(f"  total bytes: {total_bytes / 1024 / 1024:.1f} MB across {wrote + skipped} page PDFs")
    return 0 if errs == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
