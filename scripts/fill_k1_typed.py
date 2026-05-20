#!/usr/bin/env python3
"""One-off: fill 4 IRS templates (1065 + 3 K-1 variants) with TYPED convention.

Output: runs/k1_extras/ with parses + ready for full pipeline.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import date as date_cls
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fill_pagewise_careful import (
    extract_page_schema, build_fill_prompt, fill_page_via_gpt_image, OUT_ROOT,
)
from run_template_pipeline import analyze_template, gen_personas_batch, TEMPLATE_LABELS

TEMPLATES = [
    ("irs_form_1065_2024", "v1_irs_specialty"),
    ("irs_schedule_k1_form_1065_2024", "v1_irs_specialty"),
    ("irs_schedule_k1_form_1041_2024", "v1_irs_specialty"),
    ("irs_schedule_k1_form_1120s_2024", "v1_irs_specialty"),
]

DATA = REPO / "datasets" / "extraction"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
IMAGE_MODEL = "gpt-image-2"


def setup_or_load(stem: str, type_name: str) -> dict:
    src = DATA / type_name / "sources" / f"{stem}.pdf"
    if not src.exists():
        raise FileNotFoundError(src)
    import fitz
    with fitz.open(src) as d:
        n_pages = d.page_count
    wd = OUT_ROOT / type_name / stem
    wd.mkdir(parents=True, exist_ok=True)

    # Schemas
    sp = wd / "schemas.json"
    if sp.exists():
        schemas = json.loads(sp.read_text())
    else:
        print(f"  schemas: {n_pages} pages via Claude...", flush=True)
        schemas = [extract_page_schema(src, i, ANTHROPIC_MODEL) for i in range(n_pages)]
        sp.write_text(json.dumps(schemas, indent=2))

    # Analysis — force typed
    ap = wd / "template_analysis.json"
    if ap.exists():
        analysis = json.loads(ap.read_text())
    else:
        print(f"  analysis...", flush=True)
        analysis = analyze_template(src, schemas, ANTHROPIC_MODEL)
        ap.write_text(json.dumps(analysis, indent=2))
    # OVERRIDE convention — user requested TYPED for these
    analysis_typed = {**analysis, "handwriting_convention": "typed"}

    # Persona
    pf = sorted(wd.glob("persona_*.json"))
    if pf:
        persona = json.loads(pf[0].read_text())
    else:
        print(f"  persona via Claude...", flush=True)
        personas = gen_personas_batch(src, type_name, schemas, analysis, 1, ANTHROPIC_MODEL)
        persona = personas[0]
        (wd / "persona_01.json").write_text(json.dumps(persona, indent=2))

    return {"src": src, "n_pages": n_pages, "schemas": schemas,
            "analysis": analysis_typed, "persona": persona, "work_dir": wd}


def fill(stem: str, type_name: str, setup: dict) -> int:
    """Fill all pages with 1 persona, typed convention. Returns count filled."""
    n_ok = 0
    for pi in range(setup["n_pages"]):
        out_pdf = setup["work_dir"] / f"{setup['src'].stem}__p{pi+1:02d}__persona01.pdf"
        if out_pdf.exists() and out_pdf.stat().st_size > 100_000:
            print(f"  p{pi+1}: cached", flush=True)
            n_ok += 1
            continue
        prompt, _ = build_fill_prompt(setup["schemas"][pi], setup["persona"], setup["analysis"])
        # Save prompt for debug
        (setup["work_dir"] / f"prompt_p{pi+1:02d}_typed.txt").write_text(prompt)
        ok = fill_page_via_gpt_image(setup["src"], pi, prompt, out_pdf, IMAGE_MODEL)
        if ok:
            n_ok += 1
            print(f"  p{pi+1}: ✓ filled", flush=True)
        else:
            print(f"  p{pi+1}: ✗ failed", flush=True)
    return n_ok


def main() -> int:
    # Ensure Form 1065 is in the manifest
    m = DATA / "v1_irs_specialty" / "_sources_manifest.csv"
    rows = list(csv.DictReader(m.open())) if m.exists() else []
    if not any(r["local_filename"] == "irs_form_1065_2024.pdf" for r in rows):
        fields = list(rows[0].keys()) if rows else [
            "local_filename", "source_url", "source_site", "is_filled",
            "doc_pages", "downloaded_on", "notes"]
        new_row = {f: "" for f in fields}
        new_row.update({
            "local_filename": "irs_form_1065_2024.pdf",
            "source_url": "https://www.irs.gov/pub/irs-pdf/f1065.pdf",
            "source_site": "irs.gov",
            "is_filled": "blank",
            "doc_pages": "6",
            "downloaded_on": date_cls.today().isoformat(),
            "notes": "IRS Form 1065 — U.S. Return of Partnership Income (2024)",
        })
        rows.append(new_row)
        with m.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"added 1065 to manifest")

    t0 = time.time()
    total_filled = 0
    for stem, type_name in TEMPLATES:
        print(f"\n=== {stem} ===", flush=True)
        try:
            setup = setup_or_load(stem, type_name)
            n = fill(stem, type_name, setup)
            total_filled += n
            print(f"  → {n}/{setup['n_pages']} pages filled", flush=True)
        except Exception as e:
            print(f"  ! error: {e}", flush=True)

    print(f"\nDONE: {total_filled} pages filled in {(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
