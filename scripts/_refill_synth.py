#!/usr/bin/env python3
"""Refill specific (template, page) cells with gpt-image-2 using a per-page
handwriting convention. Driven by the synth pagewise reviewer decisions JSON.

Behaviour:
  - For each template marked HW/PR, ensure schemas + analysis + persona exist
    (regenerate via Claude if missing).
  - For each target (page, convention), override analysis.handwriting_convention
    in memory and build the fill prompt with that convention.
  - Delete the existing persona-page PDF (and the v1.2 page PDF) before
    refilling, so output is fresh.
  - 1 persona per template (matches earlier "fill once" direction).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "datasets" / "extraction"
sys.path.insert(0, str(REPO / "scripts"))

from fill_pagewise_careful import (
    extract_page_schema, build_fill_prompt, fill_page_via_gpt_image,
    OUT_ROOT,
)
from run_template_pipeline import (
    analyze_template, gen_personas_batch, TEMPLATE_LABELS,
)


def load_inventory_by_idx():
    inv = json.load((DATA / "_synth_review_pagewise" / "inventory.json").open())
    return {c["idx"]: c for c in inv}


def find_v12_paths_for_cell(t: str, tpl: str, page: int) -> list[Path]:
    m = DATA / "v1.2" / "_manifest.csv"
    if not m.exists():
        return []
    matches = []
    for r in csv.DictReader(m.open()):
        if r["kind"] != "synthetic" or r["type"] != t:
            continue
        stem = r["doc_stem"]
        canon = re.sub(r"__p\d+(?:_v\d+fill|_hw|_cmsfill)?(?:__d\d+)?$", "", stem)
        canon = re.sub(r"__persona\d+(?:__d\d+)?$", "", canon)
        if canon != tpl or int(r["page_num"]) != page:
            continue
        matches.append(DATA / "v1.2" / r["dest_path"])
    return matches


def setup_template(t: str, tpl: str, anthropic_model: str = "claude-sonnet-4-5") -> dict | None:
    """Return {pdf, n_pages, schemas, analysis, persona, work_dir}.
    Creates assets if missing."""
    src = DATA / t / "sources" / f"{tpl}.pdf"
    if not src.exists():
        print(f"  ! source missing: {src}", file=sys.stderr)
        return None
    try:
        with fitz.open(src) as d: n_pages = d.page_count
    except Exception as e:
        print(f"  ! fitz fail {src}: {e}", file=sys.stderr)
        return None
    work_dir = OUT_ROOT / t / tpl
    work_dir.mkdir(parents=True, exist_ok=True)

    # schemas
    sch_path = work_dir / "schemas.json"
    if sch_path.exists():
        schemas = json.loads(sch_path.read_text())
    else:
        print(f"    extracting {n_pages} page schemas via Claude...", flush=True)
        schemas = [extract_page_schema(src, i, anthropic_model) for i in range(n_pages)]
        sch_path.write_text(json.dumps(schemas, indent=2))

    # analysis
    an_path = work_dir / "template_analysis.json"
    if an_path.exists():
        analysis = json.loads(an_path.read_text())
    else:
        print(f"    analyzing template via Claude...", flush=True)
        analysis = analyze_template(src, schemas, anthropic_model)
        an_path.write_text(json.dumps(analysis, indent=2))

    # persona
    pers_files = sorted(work_dir.glob("persona_*.json"))
    if pers_files:
        persona = json.loads(pers_files[0].read_text())
    else:
        print(f"    generating 1 persona via Claude...", flush=True)
        personas = gen_personas_batch(src, t, schemas, analysis, 1, anthropic_model)
        persona = personas[0]
        (work_dir / "persona_01.json").write_text(json.dumps(persona, indent=2))

    return {
        "pdf": src, "n_pages": n_pages, "schemas": schemas,
        "analysis": analysis, "persona": persona, "work_dir": work_dir,
    }


def refill_one_cell(t: str, tpl: str, page: int, convention: str,
                    setup: dict, image_model: str = "gpt-image-2",
                    force: bool = False) -> dict:
    """Refill all persona PDFs for this (template, page) cell with target convention.

    Currently uses 1 persona (persona_01).
    IDEMPOTENT: if out_pdf already exists and is non-trivial (>100KB), skip.
    Pass force=True to nuke and refill anyway.
    """
    pidx = page - 1
    if pidx < 0 or pidx >= setup["n_pages"]:
        return {"ok": False, "reason": f"page {page} out of range (1..{setup['n_pages']})"}

    # Override convention in analysis (in-memory only)
    analysis_override = dict(setup["analysis"])
    analysis_override["handwriting_convention"] = convention

    # Persona-page output path
    out_pdf = setup["work_dir"] / f"{setup['pdf'].stem}__p{page:02d}__persona01.pdf"

    # IDEMPOTENT SKIP: if a valid fill already exists, don't re-do
    if out_pdf.exists() and not force:
        sz = out_pdf.stat().st_size
        if sz > 100_000:  # gpt-image-2 outputs are usually ~1MB; <100KB = junk
            return {"ok": True, "skipped": True, "out": str(out_pdf)}
        # corrupt/tiny → re-do
        out_pdf.unlink()
    elif out_pdf.exists() and force:
        out_pdf.unlink()

    # Build fill prompt with overridden convention
    prompt, _ = build_fill_prompt(setup["schemas"][pidx], setup["persona"], analysis_override)
    # Save prompt for debug
    (setup["work_dir"] / f"prompt_p{page:02d}_{convention}.txt").write_text(prompt)
    # Fill
    ok = fill_page_via_gpt_image(setup["pdf"], pidx, prompt, out_pdf, image_model)
    if not ok:
        return {"ok": False, "reason": "gpt-image-2 failed", "out": str(out_pdf)}

    # Replace v1.2 page(s) — find old paths and delete; new ones written by next build_v1_2
    for old in find_v12_paths_for_cell(t, tpl, page):
        if old.exists():
            old.unlink()

    return {"ok": True, "out": str(out_pdf)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("decisions", type=Path)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--anthropic-model", default="claude-sonnet-4-5")
    ap.add_argument("--image-model", default="gpt-image-2")
    ap.add_argument("--force", action="store_true",
                    help="Refill even cells with valid existing output (default: skip)")
    args = ap.parse_args()

    decisions = json.loads(args.decisions.read_text())
    inv = load_inventory_by_idx()

    # Collect (type, template, page, convention) cells
    tasks = []
    for sidx, action in decisions.items():
        if action not in ("handwritten", "printed"):
            continue
        c = inv.get(int(sidx))
        if not c: continue
        conv = "handwritten" if action == "handwritten" else "typed"
        tasks.append((c["type"], c["template"], c["page"], conv))

    # Group by (type, template) so setup happens once
    from collections import defaultdict
    by_tpl = defaultdict(list)
    for t, tpl, p, conv in tasks:
        by_tpl[(t, tpl)].append((p, conv))

    print(f"==> {len(tasks)} cells to refill across {len(by_tpl)} templates")
    print()

    # Setup phase (Claude calls — sequential per template to avoid rate spikes)
    setups: dict[tuple[str, str], dict] = {}
    for i, (key, pages) in enumerate(sorted(by_tpl.items()), 1):
        t, tpl = key
        print(f"  [{i:>2}/{len(by_tpl)}] setup {t}/{tpl}  ({len(pages)} cells)", flush=True)
        setup = setup_template(t, tpl, args.anthropic_model)
        if setup:
            setups[key] = setup

    print()
    print(f"==> setup done; firing {len(tasks)} gpt-image-2 refills (workers={args.max_workers})")
    print()

    # Refill phase (parallel gpt-image-2)
    t0 = time.time()
    ok = skip = fail = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for (t, tpl), pages in by_tpl.items():
            setup = setups.get((t, tpl))
            if not setup: continue
            for page, conv in pages:
                fut = ex.submit(refill_one_cell, t, tpl, page, conv,
                                setup, args.image_model, args.force)
                futures[fut] = (t, tpl, page, conv)
        for i, fut in enumerate(as_completed(futures), 1):
            t, tpl, page, conv = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
            tag = f"{t}/{tpl}/p{page}/{conv}"
            if r.get("skipped"):
                skip += 1
                print(f"  [{i:>3}/{len(futures)}] ⏭  {tag} (cached)", flush=True)
            elif r["ok"]:
                ok += 1
                print(f"  [{i:>3}/{len(futures)}] ✓ {tag}", flush=True)
            else:
                fail += 1
                print(f"  [{i:>3}/{len(futures)}] ✗ {tag}: {r.get('reason','?')}", flush=True)

    wall = time.time() - t0
    print()
    print(f"Done in {wall/60:.1f}min — {ok} ok, {skip} skip-cached, {fail} fail")
    if fail:
        print(f"Re-run the same command to retry the {fail} failed cells (idempotent — cached ones skip)")
    else:
        print(f"Next: rebuild v1.2 to absorb the refilled pages")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
