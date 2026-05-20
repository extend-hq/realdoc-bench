#!/usr/bin/env python3
"""Full per-template pipeline: schemas → 10 diverse personas → fill all → evaluate.

For ONE blank template PDF, executes:

  STEP 1  schema extraction per page (Claude vision) — caches schemas.json
  STEP 2  template analysis: handwriting convention, checkbox density, signature
          locations — informs persona design
  STEP 3  generate 10 DIVERSE personas with explicit checkbox profiles +
          handwriting decisions (saves persona_01.json … persona_10.json)
  STEP 4  fill all (10 personas × N pages) via gpt-image-2 in parallel
  STEP 5  evaluate each filled page with Gemini 3 Flash (cheap rubric)
          → eval_persona_NN_pNN.json + eval_summary.csv

Output:
  datasets/extraction/v1.1_pagewise_filled/<type>/<doc_stem>/
    schemas.json
    template_analysis.json
    persona_01.json … persona_10.json
    <stem>__pNN__personaXX.pdf  (10 × N pages = ~20-60 files)
    eval_persona_NN_pNN.json
    eval_summary.csv

Usage:
  python -u scripts/run_template_pipeline.py --pdf <path> --type v1_<x>
  python -u scripts/run_template_pipeline.py --pdf … --skip-eval
  python -u scripts/run_template_pipeline.py --pdf … --n-personas 12
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
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
    extract_page_schema, build_fill_prompt, fill_page_via_gpt_image,
    render_page_b64, OUT_ROOT, TEMPLATE_LABELS, get_anthropic,
)


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists(): return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[len("export "):]
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k): os.environ[k] = v
load_dotenv()


_genai = None
def get_genai():
    """Lazy Google GenAI client."""
    global _genai
    if _genai is None:
        from google import genai
        _genai = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _genai


# ────────────────────────────────────────────────────────────────────────────
# STEP 2: template analysis — does this form expect handwriting? heavy checkboxes?
# ────────────────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are analyzing a form template to decide how it should be filled out.

Look at this page image and the schema below. Decide:

1. handwriting_convention: "typed" | "handwritten" | "mixed"
   - "typed" if the form was clearly designed for typing into PDF fillable fields or printing+typing on a typewriter (CMS-1500, IRS forms, SEC filings)
   - "handwritten" if it expects pen-and-paper completion (ACORD 4 homeowner app, property loss notice, simple consent forms)
   - "mixed" if some fields are typed and some are signed/initialed

2. checkbox_strategy: brief guidance on how many checkboxes a TYPICAL filling involves
   ("most groups left blank" vs "most groups have selections" vs etc.)

3. signature_locations: list of section_ids that contain a signature field
4. notable_quirks: any unusual elements (preprinted boxes that shouldn't be filled, optional sections, etc.)

Schema:
{schema_json}

Return ONLY a JSON object:
{{
  "handwriting_convention": "typed | handwritten | mixed",
  "checkbox_strategy": "<one sentence>",
  "signature_locations": ["<section_id>", ...],
  "notable_quirks": "<one sentence>"
}}"""


def analyze_template(pdf: Path, schemas: list[dict], model: str = "claude-sonnet-4-5") -> dict:
    """Build a single analysis from page 1 image + all schemas."""
    b64 = render_page_b64(pdf, 0, dpi=120)
    client = get_anthropic()
    prompt = ANALYSIS_PROMPT.format(schema_json=json.dumps(schemas, indent=2)[:18000])
    resp = client.messages.create(
        model=model, max_tokens=600,
        messages=[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":b64}},
            {"type":"text","text":prompt},
        ]}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m: text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        from json_repair import repair_json
        return json.loads(repair_json(text))


# ────────────────────────────────────────────────────────────────────────────
# STEP 3: 10 diverse personas with explicit checkbox profiles
# ────────────────────────────────────────────────────────────────────────────

PERSONA_BATCH_PROMPT = """Design {n} DIVERSE personas for filling out this form template, ensuring meaningful variation across them.

Form type: {type_label}
Template file: {filename}

Template analysis: {analysis}

Page schemas (all pages):
{schemas_json}

Return ONLY a JSON ARRAY of {n} persona objects:
[
  {{
    "scenario_summary": "<≤ 30 words>",
    "checkbox_profile_hint": "<short description of how this persona differs in checkbox choices>",
    "scalar_complexity": "minimal | standard | comprehensive",   // how much of the form is filled
    "field_values": {{ "<field_id>": "<value>", ... }},
    "checkbox_decisions": {{ "<group_id>": ["<option_id>", ...] }},
    "table_rows": {{ "<table_id>": [{{...}}, ...] }}
  }},
  ...
]

DIVERSITY REQUIREMENTS — across the {n} personas:
- Use AT LEAST 5 different demographic/business backgrounds (geographic, scale, industry, situation)
- Vary scalar_complexity: ~3 "minimal" personas (few fields, narrow scope), ~5 "standard", ~2 "comprehensive" (lots of fields + acts authorized)
- Cover the FULL checkbox option space: across personas, ensure EVERY checkbox option in the schema gets picked by at least 1 persona (if plausible). Distribute coverage so different option combinations get exercised.
- Vary table rows: some personas use 1 row, others 2-3, occasionally 0 (if the table is optional)

PER-PERSONA RULES:
- USE EXACT field_id / group_id / option_id / column names from the schemas.
- Realistic synthetic data: 555 phone exchange, 123-45-6789 SSN style, demo addresses. NO real people.
- Be COHERENT across all pages of the SAME persona.
- Be CONSERVATIVE per persona: only check what THIS persona's scenario requires.
- Account for the handwriting convention from the analysis (handwritten signatures must look like real names a person would sign).

NAME DIVERSITY — STRICTLY REQUIRED (this is critical):
- AVOID these overused stock names — they appear in every AI-generated dataset and ruin realism:
    Maria, Jennifer, Sarah, John, James, David, Michael, Robert, Mary, Patricia, Jessica, Linda, Lisa
    (also avoid: Smith, Johnson, Williams, Brown, Davis, Garcia, Martinez, Rodriguez as last names — overused)
- Instead, sample names from a WIDE range of ethnic/cultural/generational backgrounds. Examples of good variety (rotate across these):
    * African American: Tamika, Jamal, DeAndre, Latoya, Marcus, Keisha, Tyrone, Imani, Darnell, Aaliyah
    * Hispanic/Latino: Camila, Mateo, Esperanza, Diego, Lucia, Joaquin, Valentina, Santiago, Adriana, Octavio
    * East Asian: Jin, Wei, Yuki, Hiroshi, Mei-Lin, Tetsuya, Soo-Jin, Min-Jun, Wenjie, Akari
    * South Asian: Priya, Arjun, Anika, Ravi, Deepa, Vikram, Saanvi, Aditya, Meera, Rohan
    * Middle Eastern: Layla, Omar, Yasmin, Karim, Nour, Farouk, Reema, Tariq, Salma, Hassan
    * Eastern European: Katarzyna, Dmitri, Ana, Tomasz, Olga, Pavel, Magdalena, Bogdan, Iryna, Lukas
    * Indigenous: Nizhoni, Tahoma, Winona, Chayton, Tala, Hakan, Aiyana, Kele, Sani, Catori
    * Older generations (60+ yrs): Mildred, Harold, Beatrice, Walter, Geraldine, Eugene, Dolores, Stanley, Vera, Norman
    * Pronounceable but unusual: Ramona, Cedric, Adelina, Otis, Gwendolyn, Maxwell, Roselyn, Percy
- For multi-persona batches: ensure NO two personas share a first name and NO two share a last name. Vary age, gender, ethnicity, profession deliberately.
- Avoid alliteration unless natural (e.g., "Dr. David Davis" sounds fake).
- Use uncommon but believable name combinations.

SIGNATURE FIELD RULES — CRITICAL FOR REALISM:
- For EVERY field with "signature" in field_id or label, you MUST populate field_values with the signer's full name (first + last).
- DIFFERENT signer roles on the same form (patient vs. physician vs. witness vs. interpreter vs. responsible party vs. practitioner vs. acupuncturist) MUST be DIFFERENT people. They have different names, professions, ages, and writing styles.
  - Patient signature: the patient's name (e.g., "Jennifer Martinez")
  - Physician/Practitioner/Provider signature: a DIFFERENT person (e.g., "Dr. Sarah Chen" or "Robert Yamamoto MD") — typically MD/DO/PT/RN credentials
  - Witness signature: yet ANOTHER person (e.g., a staff member "Maria Lopez RN" or family member "Carlos Martinez")
  - Interpreter signature: yet ANOTHER person, usually a clinic staff translator
  - Responsible party signature: usually the patient themselves if adult, or a parent/guardian if minor
- The SAME signer signing in two places (e.g., patient signs once on page 2 and again on page 5) uses the SAME name in both places.
- Add a "signers" section to persona output (optional but recommended) mapping each signer role to {{name, profession_or_role}} so downstream rendering can use distinct handwriting styles. Example:
    "signers": {{
      "patient": {{"name": "Jennifer Martinez", "role": "32-year-old software engineer"}},
      "physician": {{"name": "Dr. Sarah Chen", "role": "OB-GYN attending"}},
      "witness": {{"name": "Maria Lopez", "role": "RN intake nurse"}}
    }}

Output exactly {n} personas as a JSON array.
"""


def parse_personas_array(text: str, expected_n: int) -> list[dict]:
    """Robust parser for an array of {n} persona JSON objects."""
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    # Quick path
    try:
        v = json.loads(text)
        if isinstance(v, list): return v
    except json.JSONDecodeError:
        pass
    # Find the [...] envelope
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except json.JSONDecodeError: pass
        # json_repair fallback on the envelope
        try:
            from json_repair import repair_json
            v = json.loads(repair_json(m.group(0)))
            if isinstance(v, list): return v
        except Exception: pass
    # Brace-walk salvage
    objs = []
    depth = 0; in_str = False; esc = False; start = None; started_array = False
    for i, c in enumerate(text):
        if not started_array:
            if c == "[": started_array = True
            continue
        if esc: esc = False; continue
        if c == "\\": esc = True; continue
        if c == '"': in_str = not in_str; continue
        if in_str: continue
        if c == "{":
            if depth == 0: start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try: objs.append(json.loads(text[start:i+1]))
                except json.JSONDecodeError: pass
                start = None
    return objs


def gen_personas_batch(pdf: Path, type_name: str, schemas: list[dict], analysis: dict,
                      n: int, model: str = "claude-sonnet-4-5") -> list[dict]:
    label = TEMPLATE_LABELS.get(type_name, type_name)
    prompt = PERSONA_BATCH_PROMPT.format(
        n=n, type_label=label, filename=pdf.name,
        analysis=json.dumps(analysis),
        schemas_json=json.dumps(schemas, indent=2)[:25000],
    )
    client = get_anthropic()
    # Use higher max_tokens since 10 personas × ~800 tokens each = ~8000
    resp = client.messages.create(
        model=model, max_tokens=16000,
        messages=[{"role":"user","content":prompt}],
    )
    text = resp.content[0].text.strip()
    personas = parse_personas_array(text, n)
    if len(personas) < n:
        print(f"  ! only got {len(personas)}/{n} personas — short output {len(text)} chars")
    return personas


# ────────────────────────────────────────────────────────────────────────────
# STEP 5: evaluate filled PDFs with Gemini 3 Flash
# ────────────────────────────────────────────────────────────────────────────

EVAL_PROMPT = """You are evaluating a synthetically-filled form page. Compare to the original blank.

Score each rubric 1-10 (10=perfect, 1=broken). Return ONLY JSON:
{
  "form_preservation": <int>,   // are the printed labels, borders, ink colors preserved?
  "field_values_plausible": <int>,   // do the typed values look like real form data?
  "checkbox_discipline": <int>,   // are only intended checkboxes checked? (no over-checking)
  "handwriting_quality": <int>,   // signatures + hand-marks look real? (10=natural; 1=block letters or fake)
  "overall": <int>,
  "issues": "<≤ 30 words listing top problems if any>"
}"""


def eval_filled_page(filled_pdf: Path, model: str = "gemini-2.5-flash") -> dict:
    """Render the filled page → send to Gemini → parse JSON rubric.
    NOTE: gemini-3-flash if/when GA; gemini-2.5-flash is current cheap vision."""
    from google.genai import types as gtypes
    with fitz.open(filled_pdf) as d:
        pix = d[0].get_pixmap(matrix=fitz.Matrix(110/72, 110/72), alpha=False)
        png = pix.tobytes("png")
    client = get_genai()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[
                gtypes.Part.from_bytes(data=png, mime_type="image/png"),
                EVAL_PROMPT,
            ],
        )
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m: text = m.group(0)
        return json.loads(text)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--n-personas", type=int, default=10)
    ap.add_argument("--fill-workers", type=int, default=6,
                    help="Parallel gpt-image-2 fills")
    ap.add_argument("--eval-workers", type=int, default=8,
                    help="Parallel Gemini evals")
    ap.add_argument("--skip-fill", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--regen-personas", action="store_true")
    ap.add_argument("--anthropic-model", default="claude-sonnet-4-5")
    ap.add_argument("--image-model", default="gpt-image-2")
    ap.add_argument("--eval-model", default="gemini-2.5-flash")
    args = ap.parse_args()

    src_pdf = Path(args.pdf).resolve()
    with fitz.open(src_pdf) as d: n_pages = d.page_count
    work_dir = OUT_ROOT / args.type / src_pdf.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Template: {src_pdf.name}  ({n_pages}pp, type={args.type}) ===\n")

    # ── STEP 1: schemas ──
    schemas_path = work_dir / "schemas.json"
    if schemas_path.exists():
        schemas = json.loads(schemas_path.read_text())
        print(f"STEP 1 — schemas cached ({len(schemas)} pages)")
    else:
        print(f"STEP 1 — extracting schemas for {n_pages} pages…")
        schemas = []
        for i in range(n_pages):
            t0 = time.time()
            schemas.append(extract_page_schema(src_pdf, i, args.anthropic_model))
            print(f"  page {i+1} ({time.time()-t0:.0f}s)")
        schemas_path.write_text(json.dumps(schemas, indent=2))

    # ── STEP 2: template analysis ──
    analysis_path = work_dir / "template_analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        print(f"STEP 2 — template analysis cached: handwriting={analysis.get('handwriting_convention','?')}")
    else:
        print(f"STEP 2 — analyzing template handwriting/checkbox conventions…")
        analysis = analyze_template(src_pdf, schemas, args.anthropic_model)
        analysis_path.write_text(json.dumps(analysis, indent=2))
        print(f"  handwriting: {analysis.get('handwriting_convention')}")
        print(f"  checkbox strategy: {analysis.get('checkbox_strategy')}")
        print(f"  signature locations: {analysis.get('signature_locations', [])}")

    # ── STEP 3: 10 diverse personas ──
    persona_files = sorted(work_dir.glob("persona_??.json"))
    if persona_files and not args.regen_personas:
        print(f"STEP 3 — {len(persona_files)} personas cached")
    else:
        for p in persona_files: p.unlink()  # clear old
        print(f"STEP 3 — generating {args.n_personas} diverse personas (batch)…")
        t0 = time.time()
        personas = gen_personas_batch(src_pdf, args.type, schemas, analysis,
                                       args.n_personas, args.anthropic_model)
        for i, p in enumerate(personas, 1):
            (work_dir / f"persona_{i:02d}.json").write_text(json.dumps(p, indent=2))
        print(f"  generated {len(personas)} personas in {time.time()-t0:.0f}s")
        persona_files = sorted(work_dir.glob("persona_??.json"))

    # ── STEP 4: fill all personas × all pages in parallel ──
    if not args.skip_fill:
        print(f"\nSTEP 4 — filling {len(persona_files)} personas × {n_pages} pages = "
              f"{len(persona_files)*n_pages} fills (workers={args.fill_workers})…")
        tasks = []
        for pf in persona_files:
            persona = json.loads(pf.read_text())
            pid = pf.stem.replace("persona_", "")
            for pg in range(n_pages):
                prompt, _ = build_fill_prompt(schemas[pg], persona, analysis)
                out_pdf = work_dir / f"{src_pdf.stem}__p{pg+1:02d}__persona{pid}.pdf"
                tasks.append((pid, pg, prompt, out_pdf))

        t0 = time.time()
        n_ok = n_skip = n_fail = 0
        def _fill(pid, pg, prompt, out_pdf):
            if out_pdf.exists():
                return (pid, pg, True, "skip")
            ok = fill_page_via_gpt_image(src_pdf, pg, prompt, out_pdf, args.image_model)
            return (pid, pg, ok, "filled")
        with ThreadPoolExecutor(max_workers=args.fill_workers) as ex:
            futures = [ex.submit(_fill, *t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                pid, pg, ok, st = fut.result()
                if not ok:
                    n_fail += 1
                elif st == "skip":
                    n_skip += 1
                else:
                    n_ok += 1
                if i % 5 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
        print(f"  fill done: {n_ok} new + {n_skip} cached + {n_fail} failed ({time.time()-t0:.0f}s)")

    # ── STEP 5: evaluate ──
    if not args.skip_eval:
        filled_pdfs = sorted(work_dir.glob(f"{src_pdf.stem}__p*__persona*.pdf"))
        print(f"\nSTEP 5 — evaluating {len(filled_pdfs)} filled PDFs with {args.eval_model}…")
        eval_rows = []
        def _eval(pdf: Path):
            # Skip if eval cached
            eval_json = work_dir / f"eval_{pdf.stem.split('__',2)[-1]}.json"
            if eval_json.exists():
                try:
                    return (pdf, json.loads(eval_json.read_text()))
                except Exception:
                    pass
            r = eval_filled_page(pdf, args.eval_model)
            eval_json.write_text(json.dumps(r, indent=2))
            return (pdf, r)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.eval_workers) as ex:
            futures = [ex.submit(_eval, p) for p in filled_pdfs]
            for fut in as_completed(futures):
                pdf, r = fut.result()
                eval_rows.append({"filename": pdf.name, **r})
        # CSV summary
        eval_csv = work_dir / "eval_summary.csv"
        fields = ["filename", "form_preservation", "field_values_plausible",
                  "checkbox_discipline", "handwriting_quality", "overall", "issues", "error"]
        with eval_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(eval_rows, key=lambda r: r["filename"]):
                w.writerow(r)
        # Aggregate
        graded = [r for r in eval_rows if "error" not in r]
        def _avg(k):
            vals = [int(r.get(k, 0)) for r in graded if isinstance(r.get(k, 0), int)]
            return sum(vals)/len(vals) if vals else 0
        print(f"  evals done in {time.time()-t0:.0f}s — {len(graded)}/{len(filled_pdfs)} ok")
        if graded:
            print(f"  averages — preservation={_avg('form_preservation'):.1f}  "
                  f"plausible={_avg('field_values_plausible'):.1f}  "
                  f"checkboxes={_avg('checkbox_discipline'):.1f}  "
                  f"handwriting={_avg('handwriting_quality'):.1f}  "
                  f"overall={_avg('overall'):.1f}")

    print(f"\nAll artifacts in: {work_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
