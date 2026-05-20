#!/usr/bin/env python3
"""Apply per-page synth decisions from the HTML reviewer.

Action semantics:
  skip       — do nothing (leave existing pagewise PDFs)
  delete     — remove ALL persona PDFs for that (template, page), update v1.2
  handwritten — refill with gpt-image-2 using handwritten convention
  printed    — refill with gpt-image-2 using printed/typed convention

This script handles only DELETE (safe + fast). REFILL is a separate script
(_refill_synth.py) since it needs API calls + persona/schema regeneration.

Usage:
  python scripts/_apply_synth_decisions.py /tmp/synth_decisions.json --action delete
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "datasets" / "extraction"


def load_inventory() -> dict[int, dict]:
    inv = json.load((DATA / "_synth_review_pagewise" / "inventory.json").open())
    return {c["idx"]: c for c in inv}


def find_v12_paths_for_cell(cell: dict) -> list[Path]:
    """For one (type, template, page), find all v1.2 pagewise PDFs that came from it.

    v1.2 manifest rows have type, doc_stem (canonical-ish), page_num, persona.
    """
    t = cell["type"]
    tpl = cell["template"]
    page = cell["page"]
    # Read v1.2 manifest
    m = DATA / "v1.2" / "_manifest.csv"
    matches = []
    if not m.exists():
        return matches
    for r in csv.DictReader(m.open()):
        if r["kind"] != "synthetic":
            continue
        if r["type"] != t:
            continue
        # doc_stem may have suffixes like __p00 (from earlier batch); canonicalize
        stem = r["doc_stem"]
        canon = re.sub(r"__p\d+(?:_v\d+fill|_hw|_cmsfill)?(?:__d\d+)?$", "", stem)
        canon = re.sub(r"__persona\d+(?:__d\d+)?$", "", canon)
        if canon != tpl:
            continue
        if int(r["page_num"]) != page:
            continue
        matches.append(DATA / "v1.2" / r["dest_path"])
    return matches


def delete_cell(cell: dict, dry_run: bool = False) -> dict:
    """Delete every persona PDF + v1.2 page for this cell.

    Returns counts: {sources_deleted, v12_deleted}.
    """
    counts = {"sources_deleted": 0, "v12_deleted": 0}
    # Source PDFs in v1.1_pagewise_filled (1-page each, safe to delete)
    for pdf_str, page_idx, persona in cell["all_sources"]:
        p = Path(pdf_str)
        if "v1.1_pagewise_filled" in str(p) and p.exists():
            if not dry_run:
                p.unlink()
            counts["sources_deleted"] += 1
        # v1.1_synthetic sources are multi-page; we leave them alone — they're
        # legacy artifacts. v1.2 deletion below is what matters.

    # v1.2 pagewise PDFs (one per persona) — delete them
    v12_paths = find_v12_paths_for_cell(cell)
    for fp in v12_paths:
        if fp.exists():
            if not dry_run:
                fp.unlink()
            counts["v12_deleted"] += 1
    return counts


def prune_v12_manifest(deleted_cells: list[dict], dry_run: bool = False) -> int:
    """Remove rows from v1.2 manifest that reference deleted (type, tpl, page) cells."""
    m = DATA / "v1.2" / "_manifest.csv"
    if not m.exists():
        return 0
    rows = list(csv.DictReader(m.open()))
    fields = list(rows[0].keys())

    # Build deletion key set: (type, canon_template, page_num)
    keys = set()
    for c in deleted_cells:
        keys.add((c["type"], c["template"], c["page"]))

    def row_in_delete_set(r):
        if r["kind"] != "synthetic":
            return False
        stem = r["doc_stem"]
        canon = re.sub(r"__p\d+(?:_v\d+fill|_hw|_cmsfill)?(?:__d\d+)?$", "", stem)
        canon = re.sub(r"__persona\d+(?:__d\d+)?$", "", canon)
        return (r["type"], canon, int(r["page_num"])) in keys

    keep = [r for r in rows if not row_in_delete_set(r)]
    n_dropped = len(rows) - len(keep)
    if not dry_run:
        with m.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            w.writeheader()
            for r in keep:
                w.writerow(r)
    return n_dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("decisions", type=Path)
    ap.add_argument("--action", default="delete",
                    choices=["delete", "all-but-refill"],
                    help="Which actions to execute. delete = only delete. "
                         "all-but-refill = delete (skip/HW/PR handled elsewhere).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    decisions = json.loads(args.decisions.read_text())
    inv = load_inventory()

    targets = {"delete": []}
    for sidx, action in decisions.items():
        if action not in targets:
            continue
        idx = int(sidx)
        cell = inv.get(idx)
        if cell:
            targets[action].append(cell)

    print(f"Loaded {len(decisions)} decisions; {len(targets['delete'])} marked DELETE")
    if args.dry_run:
        print("(DRY RUN — no files will be touched)")

    tot_src = tot_v12 = 0
    for c in targets["delete"]:
        counts = delete_cell(c, dry_run=args.dry_run)
        tot_src += counts["sources_deleted"]
        tot_v12 += counts["v12_deleted"]
    print(f"  source PDFs deleted: {tot_src}")
    print(f"  v1.2 page PDFs deleted: {tot_v12}")

    n_pruned = prune_v12_manifest(targets["delete"], dry_run=args.dry_run)
    print(f"  v1.2 manifest rows pruned: {n_pruned}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
