#!/usr/bin/env python3
"""Run the full per-template pipeline (schemas + 5 personas + fills + eval)
on ALL remaining blank templates across all types.

Outer concurrency: --concurrent-templates (templates running in parallel)
Inner per-template concurrency: --fill-workers (gpt-image-2 calls per template)

Total parallel gpt-image-2 calls = concurrent-templates × fill-workers.
Tune to stay within OpenAI's rate limit (~30 concurrent images is fine).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Reuse pipeline functions
from fill_pagewise_careful import (
    extract_page_schema, build_fill_prompt, fill_page_via_gpt_image,
    render_page_b64, OUT_ROOT, TEMPLATE_LABELS, get_anthropic,
)
from run_template_pipeline import (
    analyze_template, gen_personas_batch, eval_filled_page,
)


# All types with blank templates to fill
TYPES_TO_RUN = [
    "v1_acord_125_126",
    "v1_acord_4",
    "v1_acord_25",  # most are filled but check
    "v1_fec_form3",  # 1 blank
    "v1_form_5500",  # 1 blank
    "v1_irs_specialty",  # mostly blanks
    "v1_intl_tax_ca",  # all blanks
]


def list_blank_templates(types: list[str]) -> list[tuple[str, Path]]:
    out = []
    for t in types:
        m = Path(f"datasets/extraction/{t}/_sources_manifest.csv")
        if not m.exists():
            continue
        for r in csv.DictReader(open(m)):
            if (r.get("is_filled") or "").strip().lower() != "blank":
                continue
            pdf = m.parent / "sources" / r["local_filename"]
            if pdf.exists():
                out.append((t, pdf))
    return out


def run_one_template(type_name: str, pdf: Path, n_personas: int,
                     fill_workers: int, eval_workers: int,
                     anthropic_model: str, image_model: str, eval_model: str,
                     skip_eval: bool) -> dict:
    tag = f"{type_name}/{pdf.name}"
    t_start = time.time()
    try:
        with fitz.open(pdf) as d:
            n_pages = d.page_count
        work_dir = OUT_ROOT / type_name / pdf.stem
        work_dir.mkdir(parents=True, exist_ok=True)

        # STEP 1: schemas
        schemas_path = work_dir / "schemas.json"
        if schemas_path.exists():
            schemas = json.loads(schemas_path.read_text())
        else:
            schemas = []
            for i in range(n_pages):
                schemas.append(extract_page_schema(pdf, i, anthropic_model))
            schemas_path.write_text(json.dumps(schemas, indent=2))

        # STEP 2: analysis (also caches)
        analysis_path = work_dir / "template_analysis.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text())
        else:
            analysis = analyze_template(pdf, schemas, anthropic_model)
            analysis_path.write_text(json.dumps(analysis, indent=2))

        # STEP 3: personas (cache if right number exists)
        existing = sorted(work_dir.glob("persona_??.json"))
        if len(existing) >= n_personas:
            personas = [json.loads(p.read_text()) for p in existing[:n_personas]]
            persona_status = f"cached ({len(personas)})"
        else:
            # clear old + regen
            for p in existing:
                p.unlink()
            personas = gen_personas_batch(pdf, type_name, schemas, analysis,
                                           n_personas, anthropic_model)
            for i, p in enumerate(personas, 1):
                (work_dir / f"persona_{i:02d}.json").write_text(json.dumps(p, indent=2))
            persona_status = f"generated {len(personas)}"

        # STEP 4: fill all personas × all pages
        tasks = []
        for pi, persona in enumerate(personas, 1):
            for pg in range(n_pages):
                prompt, _ = build_fill_prompt(schemas[pg], persona, analysis)
                out_pdf = work_dir / f"{pdf.stem}__p{pg+1:02d}__persona{pi:02d}.pdf"
                tasks.append((pi, pg, prompt, out_pdf))

        n_ok = n_skip = n_fail = 0
        def _fill(pi, pg, prompt, out_pdf):
            if out_pdf.exists():
                return True, "skip"
            ok = fill_page_via_gpt_image(pdf, pg, prompt, out_pdf, image_model)
            return ok, "filled"
        with ThreadPoolExecutor(max_workers=fill_workers) as ex:
            futures = [ex.submit(_fill, *t) for t in tasks]
            for fut in as_completed(futures):
                ok, st = fut.result()
                if not ok: n_fail += 1
                elif st == "skip": n_skip += 1
                else: n_ok += 1

        # STEP 5: eval (parallel)
        n_eval = 0
        avg_overall = 0.0
        if not skip_eval:
            filled_pdfs = sorted(work_dir.glob(f"{pdf.stem}__p*__persona*.pdf"))
            scores = []
            def _eval(fp: Path):
                eval_json = work_dir / f"eval_{fp.stem.split('__',2)[-1]}.json"
                if eval_json.exists():
                    try:
                        return json.loads(eval_json.read_text())
                    except Exception:
                        pass
                r = eval_filled_page(fp, eval_model)
                eval_json.write_text(json.dumps(r, indent=2))
                return r
            with ThreadPoolExecutor(max_workers=eval_workers) as ex:
                futures = [ex.submit(_eval, fp) for fp in filled_pdfs]
                for fut in as_completed(futures):
                    r = fut.result()
                    if isinstance(r.get("overall"), int):
                        scores.append(r["overall"])
            n_eval = len(scores)
            avg_overall = sum(scores)/len(scores) if scores else 0

        dt = time.time() - t_start
        return {
            "type": type_name, "pdf": pdf.name, "pages": n_pages,
            "personas": persona_status, "fills_ok": n_ok, "fills_skip": n_skip,
            "fills_fail": n_fail, "evals": n_eval, "avg_overall": round(avg_overall, 1),
            "elapsed_s": int(dt),
        }
    except Exception as e:
        return {"type": type_name, "pdf": pdf.name, "error": str(e)[:200],
                "elapsed_s": int(time.time() - t_start)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", action="append", default=None,
                    help="Restrict to one or more types (default: all in TYPES_TO_RUN)")
    ap.add_argument("--n-personas", type=int, default=5)
    ap.add_argument("--concurrent-templates", type=int, default=3)
    ap.add_argument("--fill-workers", type=int, default=4)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--anthropic-model", default="claude-sonnet-4-5")
    ap.add_argument("--image-model", default="gpt-image-2")
    ap.add_argument("--eval-model", default="gemini-2.5-flash")
    ap.add_argument("--max-templates", type=int, default=None)
    args = ap.parse_args()

    types_in_scope = args.type if args.type else TYPES_TO_RUN
    templates = list_blank_templates(types_in_scope)
    if args.max_templates:
        templates = templates[:args.max_templates]
    n_total_pages = 0
    for t, p in templates:
        with fitz.open(p) as d:
            n_total_pages += d.page_count
    n_fills_est = n_total_pages * args.n_personas
    cost_est = n_fills_est * 0.50

    print(f"Pipeline scope:")
    print(f"  templates: {len(templates)} across {len(types_in_scope)} types")
    print(f"  pages: {n_total_pages}")
    print(f"  personas/template: {args.n_personas}")
    print(f"  total fills: {n_fills_est}")
    print(f"  est cost: ${cost_est:.0f} (fills) + ~${args.n_personas*len(templates)*0.05:.0f} (personas) + ~${n_fills_est*0.001:.1f} (eval)")
    print(f"  concurrent templates: {args.concurrent_templates}")
    print(f"  fill workers/template: {args.fill_workers}  →  max concurrent gpt-image-2: {args.concurrent_templates*args.fill_workers}")
    print()

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrent_templates) as ex:
        futures = {
            ex.submit(run_one_template, t, p, args.n_personas, args.fill_workers,
                      args.eval_workers, args.anthropic_model, args.image_model,
                      args.eval_model, args.skip_eval): (t, p)
            for t, p in templates
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            tag = f"{r['type']}/{r['pdf']}"
            if "error" in r:
                print(f"  [{i}/{len(templates)}] ✗ {tag}: {r['error']}  ({r['elapsed_s']}s)", flush=True)
            else:
                print(f"  [{i}/{len(templates)}] ✓ {tag}: "
                      f"{r['fills_ok']}+{r['fills_skip']}/{r['pages']*args.n_personas} fills, "
                      f"{r['evals']} evals, avg={r['avg_overall']}, {r['personas']}  ({r['elapsed_s']}s)",
                      flush=True)

    wall = time.time() - t0
    # Summary CSV
    summary_csv = REPO / "runs" / "_logs" / f"all_templates_summary_{int(time.time())}.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as f:
        fields = ["type", "pdf", "pages", "personas", "fills_ok", "fills_skip",
                  "fills_fail", "evals", "avg_overall", "elapsed_s", "error"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["type"], r["pdf"])):
            w.writerow(r)

    n_ok = sum(1 for r in results if "error" not in r)
    print(f"\nDone in {wall/60:.1f}min — {n_ok}/{len(results)} templates ok")
    avg_overall = [r["avg_overall"] for r in results if "avg_overall" in r and r["avg_overall"] > 0]
    if avg_overall:
        print(f"  mean overall eval score: {sum(avg_overall)/len(avg_overall):.1f}/10")
    print(f"  summary: {summary_csv.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
