#!/usr/bin/env python3
"""Fill a template with multiple pre-generated personas in parallel.

Each persona produces a complete set of filled per-page PDFs named
`<doc_stem>__pNN__personaXX.pdf`.

Usage:
  python -u scripts/fill_persona_batch.py \
    --type v1_irs_specialty \
    --pdf datasets/extraction/v1_irs_specialty/sources/irs_form_2848_2024.pdf \
    --workers 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import fitz

# Reuse pipeline from fill_pagewise_careful
sys.path.insert(0, str(REPO / "scripts"))
from fill_pagewise_careful import (
    build_fill_prompt, fill_page_via_gpt_image, OUT_ROOT,
    TEMPLATE_LABELS,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--workers", type=int, default=6,
                    help="Parallel gpt-image-2 calls (10 personas × N pages)")
    ap.add_argument("--persona-glob", default="persona_*.json",
                    help="Glob for persona files in the work dir")
    ap.add_argument("--image-model", default="gpt-image-2")
    args = ap.parse_args()

    src_pdf = Path(args.pdf).resolve()
    work_dir = OUT_ROOT / args.type / src_pdf.stem
    schemas = json.loads((work_dir / "schemas.json").read_text())
    persona_files = sorted(work_dir.glob(args.persona_glob))
    with fitz.open(src_pdf) as d:
        n_pages = d.page_count

    print(f"Template: {src_pdf.name}  ({n_pages}pp, type={args.type})")
    print(f"Personas: {len(persona_files)}")
    print(f"Total fills: {len(persona_files) * n_pages} ({args.workers} workers)")

    # Build all (persona_idx, page_idx, prompt, out_pdf) tasks
    tasks = []
    for pf in persona_files:
        persona = json.loads(pf.read_text())
        pid = pf.stem.replace("persona_", "")  # "01", "02", etc.
        for pg in range(n_pages):
            prompt, debug = build_fill_prompt(schemas[pg], persona)
            out = work_dir / f"{src_pdf.stem}__p{pg+1:02d}__persona{pid}.pdf"
            tasks.append((pid, pg, prompt, out, debug))

    print()
    t0 = time.time()
    n_ok = n_fail = 0

    def _job(pid, pg, prompt, out, debug):
        t = time.time()
        if out.exists():
            return (pid, pg, True, out, "skip", time.time() - t)
        ok = fill_page_via_gpt_image(src_pdf, pg, prompt, out, args.image_model)
        return (pid, pg, ok, out, "filled", time.time() - t)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            pid, pg, ok, out, status, dt = fut.result()
            if ok:
                n_ok += 1
                print(f"  [{i}/{len(tasks)}] ✓ persona{pid} p{pg+1} ({status}, {dt:.0f}s)", flush=True)
            else:
                n_fail += 1
                print(f"  [{i}/{len(tasks)}] ✗ persona{pid} p{pg+1}", flush=True)

    wall = time.time() - t0
    print(f"\nDone in {wall:.0f}s — {n_ok} ok / {n_fail} fail")
    print(f"  outputs: {work_dir.relative_to(REPO)}/{src_pdf.stem}__pNN__personaXX.pdf")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
