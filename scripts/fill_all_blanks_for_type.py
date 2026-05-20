#!/usr/bin/env python3
"""Run the careful pagewise fill pipeline on EVERY blank template of one or
more types. Outer concurrency = templates in flight at once. Each template
internally fills its pages in parallel via ThreadPoolExecutor.

Usage:
  python -u scripts/fill_all_blanks_for_type.py \
    --type v1_acord_125_126 v1_acord_4 \
    --concurrent-templates 4 \
    --workers-per-template 3
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
sys.path.insert(0, str(REPO / "scripts"))
from fill_pagewise_careful import (
    extract_page_schema, build_persona, build_fill_prompt,
    fill_page_via_gpt_image, OUT_ROOT, TEMPLATE_LABELS,
)


def list_blank_templates(types: list[str]) -> list[tuple[str, Path]]:
    out = []
    for t in types:
        m = Path(f'datasets/extraction/{t}/_sources_manifest.csv')
        if not m.exists():
            continue
        for r in csv.DictReader(open(m)):
            if (r.get('is_filled') or '').strip().lower() != 'blank':
                continue
            pdf = m.parent / 'sources' / r['local_filename']
            if pdf.exists():
                out.append((t, pdf))
    return out


def run_template(t_idx: int, n_total: int, type_name: str, pdf: Path,
                 anthropic_model: str, image_model: str,
                 page_workers: int) -> dict:
    """Full pipeline for one template: schemas → persona → fills (parallel pages)."""
    tag = f"[{t_idx+1}/{n_total}] {type_name}/{pdf.name}"
    t_start = time.time()
    try:
        with fitz.open(pdf) as d:
            n_pages = d.page_count
        work_dir = OUT_ROOT / type_name / pdf.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        schemas_path = work_dir / "schemas.json"
        persona_path = work_dir / "persona.json"

        # STEP 1: schemas
        if schemas_path.exists():
            schemas = json.loads(schemas_path.read_text())
            sch_status = "cached"
        else:
            schemas = []
            for i in range(n_pages):
                schemas.append(extract_page_schema(pdf, i, anthropic_model))
            schemas_path.write_text(json.dumps(schemas, indent=2))
            sch_status = "extracted"

        # STEP 2: persona
        if persona_path.exists():
            persona = json.loads(persona_path.read_text())
            per_status = "cached"
        else:
            label = TEMPLATE_LABELS.get(type_name, type_name)
            persona = build_persona(label, schemas, anthropic_model)
            persona_path.write_text(json.dumps(persona, indent=2))
            per_status = "built"

        # STEP 3: per-page parallel fill
        def _fill_page(i):
            prompt, debug = build_fill_prompt(schemas[i], persona)
            (work_dir / f"prompt_p{i+1:02d}.txt").write_text(prompt)
            out_pdf = work_dir / f"{pdf.stem}__p{i+1:02d}_filled.pdf"
            if out_pdf.exists():
                return (i, True, "skip")
            ok = fill_page_via_gpt_image(pdf, i, prompt, out_pdf, image_model)
            return (i, ok, "filled")

        n_ok = n_fail = 0
        with ThreadPoolExecutor(max_workers=page_workers) as ex:
            futures = [ex.submit(_fill_page, i) for i in range(n_pages)]
            for fut in as_completed(futures):
                i, ok, st = fut.result()
                if ok: n_ok += 1
                else: n_fail += 1

        dt = time.time() - t_start
        print(f"  {tag}: {n_ok}/{n_pages} pages filled (schemas={sch_status}, persona={per_status}, {dt:.0f}s)",
              flush=True)
        return {"type": type_name, "pdf": pdf.name, "pages": n_pages,
                "ok": n_ok, "fail": n_fail, "elapsed": dt}
    except Exception as e:
        dt = time.time() - t_start
        print(f"  {tag}: ✗ ERROR {type(e).__name__}: {str(e)[:200]} ({dt:.0f}s)", flush=True)
        return {"type": type_name, "pdf": pdf.name, "error": str(e)[:200], "elapsed": dt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", nargs="+", required=True,
                    help="One or more types (e.g. v1_acord_125_126 v1_acord_4)")
    ap.add_argument("--concurrent-templates", type=int, default=4,
                    help="How many templates to process in parallel (each template uses its own page workers)")
    ap.add_argument("--workers-per-template", type=int, default=3,
                    help="Parallel page fills within each template")
    ap.add_argument("--anthropic-model", default="claude-sonnet-4-5")
    ap.add_argument("--image-model", default="gpt-image-2")
    args = ap.parse_args()

    templates = list_blank_templates(args.type)
    print(f"Found {len(templates)} blank templates across {len(args.type)} types:")
    n_total_pages = 0
    for t, p in templates:
        with fitz.open(p) as d: pp = d.page_count
        n_total_pages += pp
        print(f"  {t}/{p.name}  ({pp}pp)")
    print()
    print(f"Total: {n_total_pages} pages")
    print(f"Outer concurrency: {args.concurrent_templates} templates in parallel")
    print(f"Inner page workers: {args.workers_per_template} per template")
    print(f"Theoretical max concurrent gpt-image-2 calls: {args.concurrent_templates * args.workers_per_template}")
    print()

    t_overall = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrent_templates) as ex:
        futures = [ex.submit(run_template, i, len(templates), t, p,
                              args.anthropic_model, args.image_model,
                              args.workers_per_template)
                   for i, (t, p) in enumerate(templates)]
        for fut in as_completed(futures):
            results.append(fut.result())

    wall = time.time() - t_overall
    n_ok_tot = sum(r.get("ok", 0) for r in results)
    n_fail_tot = sum(r.get("fail", 0) for r in results)
    n_err = sum(1 for r in results if "error" in r)
    print(f"\nDone in {wall:.0f}s — {len(results)} templates, "
          f"{n_ok_tot}/{n_total_pages} pages filled, {n_fail_tot} failed, {n_err} errored")
    return 0 if n_fail_tot == 0 and n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
