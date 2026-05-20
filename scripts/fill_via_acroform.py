#!/usr/bin/env python3
"""AcroForm-based pagewise template fill — short-circuits gpt-image-2 for
PDFs that already have fillable widgets.

For each template with AcroForm widgets:
  1. Extract widget inventory (field_name → type + position + page)
  2. Generate N personas via Claude — each persona is a flat dict of
     {field_name: value} matching the widget names
  3. For each persona: apply widget values via pymupdf, flatten, save
  4. Split each multi-page filled PDF into per-page PDFs for consistency
     with v1.1_pagewise/

Output: datasets/extraction/v1.1_pagewise_filled/<type>/<stem>/
        <stem>__pNN__acroformXX.pdf

Cost: just Claude persona-gen (~$0.05/persona). gpt-image-2 NOT used.

Usage:
  python -u scripts/fill_via_acroform.py --type v1_irs_specialty
  python -u scripts/fill_via_acroform.py --all-with-widgets --n-personas 20
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
from collections import defaultdict

import fitz

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
OUT_ROOT = DATASETS / "v1.1_pagewise_filled"

TYPES_WITH_AF = [
    "v1_acord_125_126", "v1_acord_4", "v1_acord_25",
    "v1_irs_specialty",
    "v1_intl_tax_ca", "v1_intl_tax_nz",
]


def load_dotenv(p: Path = REPO / ".env") -> None:
    if not p.exists(): return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[len("export "):]
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k): os.environ[k] = v
load_dotenv()


_anthropic = None
def get_anthropic():
    global _anthropic
    if _anthropic is None:
        from anthropic import Anthropic
        _anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic


# ─── Widget inventory ──────────────────────────────────────────────────────

FT_NAMES = {1: "text", 2: "button", 3: "choice", 4: "signature"}

def extract_widget_inventory(pdf: Path) -> dict:
    """Return: {
        'pdf_filename': str,
        'pages': N,
        'widgets': [
          {'page': 1, 'name': 'taxpayer_name', 'type': 'text',
           'subtype': 'text|checkbox|radio|pushbtn|signature',
           'tooltip': '...', 'choices': [...] (for choice), 'export_value': str (for checkbox)}, ...
        ]
    }"""
    inv = {"pdf_filename": pdf.name, "pages": 0, "widgets": []}
    with fitz.open(pdf) as doc:
        inv["pages"] = doc.page_count
        for pi, page in enumerate(doc, 1):
            for w in (page.widgets() or []):
                ft = w.field_type
                rec = {
                    "page": pi,
                    "name": w.field_name or "",
                    "type": FT_NAMES.get(ft, "other"),
                    "subtype": (w.field_type_string or "").strip(),
                    "tooltip": w.field_label or "",
                }
                if ft == 3:  # choice
                    try:
                        rec["choices"] = list(w.choice_values or [])
                    except Exception:
                        rec["choices"] = []
                if ft == 2:  # button — capture on-state options safely
                    try:
                        bs = w.button_states() or {}
                        if isinstance(bs, dict):
                            normal = bs.get("normal", [])
                            if isinstance(normal, list):
                                rec["on_states"] = [s for s in normal if s != "Off"]
                            elif isinstance(normal, dict):
                                rec["on_states"] = list(normal.keys())
                    except Exception:
                        pass
                inv["widgets"].append(rec)
    return inv


# ─── Persona generation against widget inventory ──────────────────────────

PERSONA_PROMPT = """You are designing {n} DIVERSE personas to fill out a PDF form using its AcroForm fields.

Template: {filename}  ({pages} pages, {n_widgets} widgets)

Widget inventory (field_name + type + tooltip):
{inventory_block}

Return ONLY a JSON ARRAY of {n} persona objects:
[
  {{
    "scenario_summary": "<≤ 30 words>",
    "field_values": {{
      "<widget_name>": "<value>" | true | false,
      ...
    }}
  }},
  ...
]

RULES:
- Use EXACT widget names from the inventory above.
- TEXT fields: realistic synthetic values (555 phone exchange, 123-45-6789 SSN, demo addresses, plausible amounts).
- CHECKBOX fields (subtype=CheckBox or contains 'check'): true means tick, false/omit means leave unchecked. BE CONSERVATIVE — only check what the persona's scenario justifies. NEVER check every box.
- RADIO fields: provide the chosen option value (one per group). The "name" includes the group; the value is one of the group's allowed states.
- CHOICE fields (dropdowns): pick a plausible option.
- SIGNATURE fields: provide a realistic full name (the AcroForm "signature" widget stores text, not actual ink).
- DIVERSITY across {n} personas: vary scenario, geography, scale, and checkbox patterns substantially. Different personas should activate different checkbox subsets.
- COHERENCE within each persona: if persona 1 is a TX small business, all values (state, addresses, ID numbers, etc.) align with that.

Output exactly {n} personas as a JSON array. NO preamble, NO fences.
"""


def gen_personas_for_acroform(inv: dict, n: int, model: str = "claude-sonnet-4-5") -> list[dict]:
    # Compact inventory for prompt
    lines = []
    for w in inv["widgets"]:
        line = f'  p{w["page"]:>2}  name="{w["name"]}"  type={w["type"]}/{w["subtype"]}'
        if w.get("tooltip"): line += f'  tip="{w["tooltip"]}"'
        if w.get("choices"): line += f'  choices={w["choices"]}'
        lines.append(line)
    inv_text = "\n".join(lines)
    if len(inv_text) > 28000:
        inv_text = inv_text[:28000] + "\n  ... (truncated; same naming convention)"
    prompt = PERSONA_PROMPT.format(
        n=n, filename=inv["pdf_filename"], pages=inv["pages"],
        n_widgets=len(inv["widgets"]), inventory_block=inv_text,
    )
    client = get_anthropic()
    resp = client.messages.create(
        model=model, max_tokens=20000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m: text = m.group(0)
    try:
        personas = json.loads(text)
    except json.JSONDecodeError:
        # brace-walk salvage
        objs = []
        depth = 0; in_s = False; esc = False; start = None; arr = False
        for i, c in enumerate(text):
            if not arr:
                if c == "[": arr = True
                continue
            if esc: esc = False; continue
            if c == "\\": esc = True; continue
            if c == '"': in_s = not in_s; continue
            if in_s: continue
            if c == "{":
                if depth == 0: start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try: objs.append(json.loads(text[start:i+1]))
                    except: pass
                    start = None
        personas = objs
    return personas


# ─── Apply persona to AcroForm + save ──────────────────────────────────────

def fill_acroform(src_pdf: Path, persona: dict, out_pdf: Path) -> tuple[bool, int]:
    """Apply persona's field_values to a copy of src_pdf via pymupdf widgets.
    Returns (ok, n_filled)."""
    values = persona.get("field_values", {})
    if not values:
        return False, 0
    try:
        doc = fitz.open(src_pdf)
    except Exception as e:
        return False, 0
    n_filled = 0
    for page in doc:
        for w in (page.widgets() or []):
            name = w.field_name or ""
            if name not in values:
                continue
            v = values[name]
            try:
                ft = w.field_type
                if ft == 1:  # text
                    w.field_value = "" if v is None else str(v)
                elif ft == 2:  # button — checkbox/radio
                    # For checkboxes: True → set to 'Yes' (or the export "on" value), False → 'Off'
                    if isinstance(v, bool):
                        if v:
                            # Try a couple common on-states
                            states = []
                            try:
                                states = list((w.button_states() or {}).get("normal", []))
                            except Exception:
                                pass
                            on_val = next((s for s in states if s != "Off"), "Yes")
                            w.field_value = on_val
                        else:
                            w.field_value = "Off"
                    else:
                        # radio: value is the option name
                        w.field_value = str(v)
                elif ft == 3:  # choice
                    w.field_value = str(v) if v is not None else ""
                elif ft == 4:  # signature widget (text-based for AcroForm)
                    w.field_value = str(v) if v else ""
                w.update()
                n_filled += 1
            except Exception:
                continue
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Flatten so widgets become baked-in content
    try:
        doc.bake(annots=True, widgets=True)
    except Exception:
        pass  # older pymupdf
    doc.save(out_pdf, garbage=4, deflate=True)
    doc.close()
    return True, n_filled


# ─── Split filled multi-page PDF into per-page PDFs ───────────────────────

def split_to_pages(multi_pdf: Path, dest_dir: Path, stem_prefix: str, persona_id: str) -> list[Path]:
    out = []
    with fitz.open(multi_pdf) as doc:
        pad = max(2, len(str(doc.page_count)))
        for i in range(doc.page_count):
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=i, to_page=i)
            out_path = dest_dir / f"{stem_prefix}__p{i+1:0{pad}d}__acroform{persona_id}.pdf"
            new_doc.save(out_path, garbage=4, deflate=True)
            new_doc.close()
            out.append(out_path)
    return out


# ─── Orchestrator ────────────────────────────────────────────────────────

def list_acroform_blanks(types: list[str]) -> list[tuple[str, Path]]:
    out = []
    for t in types:
        m = DATASETS / t / "_sources_manifest.csv"
        if not m.exists():
            continue
        for r in csv.DictReader(open(m)):
            if (r.get("is_filled") or "").strip().lower() != "blank":
                continue
            pdf = m.parent / "sources" / r["local_filename"]
            if not pdf.exists():
                continue
            # Quick check: has widgets?
            try:
                with fitz.open(pdf) as d:
                    has_w = any(p.widgets() for p in d)
                if has_w:
                    out.append((t, pdf))
            except Exception:
                pass
    return out


def run_one(type_name: str, pdf: Path, n_personas: int, anthropic_model: str) -> dict:
    t0 = time.time()
    try:
        work_dir = OUT_ROOT / type_name / pdf.stem
        work_dir.mkdir(parents=True, exist_ok=True)

        # Widget inventory (cache)
        inv_path = work_dir / "widget_inventory.json"
        if inv_path.exists():
            inv = json.loads(inv_path.read_text())
        else:
            inv = extract_widget_inventory(pdf)
            inv_path.write_text(json.dumps(inv, indent=2))

        # Personas (cache acroform_persona_NN.json)
        existing = sorted(work_dir.glob("acroform_persona_??.json"))
        if len(existing) >= n_personas:
            personas = [json.loads(p.read_text()) for p in existing[:n_personas]]
            pstatus = f"cached({len(personas)})"
        else:
            for p in existing: p.unlink()
            personas = gen_personas_for_acroform(inv, n_personas, anthropic_model)
            for i, p in enumerate(personas, 1):
                (work_dir / f"acroform_persona_{i:02d}.json").write_text(json.dumps(p, indent=2))
            pstatus = f"gen({len(personas)})"

        # Fill + split
        n_pages_total = 0
        for pi, persona in enumerate(personas, 1):
            persona_id = f"{pi:02d}"
            multi_out = work_dir / f"_acroform_multi_persona{persona_id}.pdf"
            ok, n_filled = fill_acroform(pdf, persona, multi_out)
            if not ok:
                continue
            # Split into per-page PDFs
            pages = split_to_pages(multi_out, work_dir, pdf.stem, persona_id)
            n_pages_total += len(pages)
            # Remove the intermediate multi-page (per-page is what we want)
            multi_out.unlink(missing_ok=True)

        dt = time.time() - t0
        return {"type": type_name, "pdf": pdf.name, "pages": inv["pages"],
                "widgets": len(inv["widgets"]), "personas": len(personas),
                "pages_filled": n_pages_total, "personas_status": pstatus,
                "elapsed_s": int(dt)}
    except Exception as e:
        return {"type": type_name, "pdf": pdf.name,
                "error": f"{type(e).__name__}: {str(e)[:200]}", "elapsed_s": int(time.time()-t0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", action="append", default=None)
    ap.add_argument("--n-personas", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel templates (each template's Claude persona-gen is the slow step)")
    ap.add_argument("--anthropic-model", default="claude-sonnet-4-5")
    args = ap.parse_args()

    types = args.type if args.type else TYPES_WITH_AF
    templates = list_acroform_blanks(types)
    print(f"AcroForm blank templates: {len(templates)} across {len(types)} types")
    for t, p in templates:
        with fitz.open(p) as d:
            n_w = sum(len(list(pg.widgets() or [])) for pg in d)
            n_pp = d.page_count
        print(f"  {t}/{p.name}  ({n_pp}pp, {n_w} widgets)")
    print(f"\nWill generate {args.n_personas} personas × {len(templates)} templates = "
          f"{args.n_personas * len(templates)} filled multi-page PDFs")
    print(f"  estimated cost: ~${len(templates) * 0.10:.1f} (persona-gen only, no gpt-image-2)")
    print()

    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one, t, p, args.n_personas, args.anthropic_model)
                   for t, p in templates]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            tag = f"{r['type']}/{r['pdf']}"
            if "error" in r:
                print(f"  [{i}/{len(templates)}] ✗ {tag}: {r['error']} ({r['elapsed_s']}s)", flush=True)
            else:
                print(f"  [{i}/{len(templates)}] ✓ {tag}: {r['personas']} personas, "
                      f"{r['pages_filled']} per-page PDFs ({r['personas_status']}, {r['elapsed_s']}s)",
                      flush=True)

    wall = time.time() - t_start
    n_ok = sum(1 for r in results if "error" not in r)
    n_total_filled = sum(r.get("pages_filled", 0) for r in results)
    print(f"\nDone in {wall:.0f}s — {n_ok}/{len(results)} templates ok, "
          f"{n_total_filled} per-page PDFs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
