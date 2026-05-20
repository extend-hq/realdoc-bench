#!/usr/bin/env python3
"""Generalized form-filler for table-heavy v1_* types.

Builds on lessons from fill_cms1500.py:
  - explicit FORM PRESERVATION ABSOLUTE RULES so the model doesn't redesign the form
  - persona delivered as labelled bullet list (Claude-generated, cached per type)
  - form-specific section guidance so dense tables actually get filled
  - force-vision raster output (image-only PDF, no text layer)
  - parallel execution

Configured types:
  - v1_vaers      VAERS-1 reporting form (CDC/FDA)
  - v1_ub04       UB-04 / CMS-1450 institutional claim
  - v1_form_5500  IRS Form 5500 (DOL EFAST2) annual return

Usage:
  python -u scripts/fill_form_v2.py --type v1_vaers --workers 6
  python -u scripts/fill_form_v2.py --type v1_ub04 --doc bcbsmt_UB04_form.pdf
  python -u scripts/fill_form_v2.py --type v1_form_5500 --persona 0
"""

from __future__ import annotations

import argparse
import base64
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
DATASETS = REPO / "datasets" / "extraction"
PERSONAS_DIR = DATASETS / "_personas"


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v
load_dotenv()


# =====================================================================
# Form configs
# =====================================================================

VAERS_PROMPT = """TASK: Fill out the VAERS-1 (Vaccine Adverse Event Reporting System) form in the input image with the values below. Type values into the blank fields; preserve every printed pixel of the form exactly.

===== FORM PRESERVATION — ABSOLUTE RULES =====
- The input is the FDA/CDC VAERS-1 form (white sheet, OMB No. 0910-0834 OR earlier OMB 0910-0285). Output MUST be the same VAERS-1 form, NOT a redesigned version.
- DO NOT regenerate, redesign, or modernize the form. DO NOT change the form layout, font weights, line widths, box positions, or column headers.
- Preserve EVERY printed label, checkbox glyph, instruction line, OMB number, "Patient information" / "Vaccine information" section headers, and the official VAERS-1 footer/header watermarks EXACTLY.
- ONLY add typed values + handwritten signature where blank fields/checkboxes exist.

Persona: {persona_label}

===== SECTION GUIDANCE =====
- Section 1 — PATIENT IDENTITY: patient name (last/first/MI), patient address, age at vaccination, sex (M/F checkbox), date of birth.
- Section 2 — REPORTER: report date, reporter name + address, reporter title (e.g., MD/RN/Pt-Family).
- Section 3 — ADVERSE EVENT(S): describe adverse event(s), date of onset (vs date of vaccination), check applicable outcomes (Died / Life-threatening / Hospitalized / Disability / Birth defect / ER visit / Office visit / None of the above). Recovery status (yes/no/unknown).
- Section 4 — VACCINE INFORMATION (table): for each vaccine row — vaccine type (e.g. INFLUENZA, COVID-19), manufacturer, lot number, route/site (IM/Right arm), # of previous doses, date given.
- Section 5 — RELEVANT MEDICAL HISTORY: prior illnesses, current medications, pre-existing physician conditions, allergies.
- Section 6 — DIAGNOSTIC LAB DATA: any blood work, imaging, swab results with values + dates.
- Section 7 — FACILITY where vaccine given (name, address, type — public clinic / pharmacy / school / hospital).

===== DATA TO FILL =====
{persona_bullets}

===== FORMATTING & STYLE =====
- Type values in a typewriter monospaced or simple sans-serif font, BLACK ink. Use small block letters in narrow boxes.
- For checkboxes: hand-drawn X or ✓ marks (small, in pen). Check ONLY what the persona explicitly indicates.
- For the SIGNATURE field: realistic blue-ink CURSIVE handwriting, looks hand-signed.
- Multi-line description fields (adverse event narrative, medical history): wrap text to fit the boxed area; do not overflow.
- For the vaccine table, align values row-by-row beneath the printed column headers.
- DO NOT add text outside the printed form boxes. DO NOT alter any printed label.
"""

UB04_PROMPT = """TASK: Fill out the UB-04 / CMS-1450 institutional claim form in the input image with the values below. Type values into the 81 form locator (FL) boxes; preserve every printed pixel of the form exactly.

===== FORM PRESERVATION — ABSOLUTE RULES =====
- The input is the CMS-1450 (UB-04) institutional claim form — RED/PINK ink printed version, with 81 numbered Form Locators (FL1–FL81), the "TYPE OF BILL" 4-digit box at FL4, the diagnosis/procedure code table mid-form (FL66–FL75), and the revenue/HCPCS service line table (FL42–FL49) with 22 service lines.
- DO NOT generate the older UB-92 or any other variant. DO NOT redesign or modernize the form.
- Preserve EVERY printed FL number, label, box border, the official red/pink ink color, all column headers in the service line grid (REV. CD. / DESCRIPTION / HCPCS/RATE/HIPPS CODE / SERV. DATE / SERV. UNITS / TOT. CHARGES / NON-COVERED CHARGES), and all printed footer/header text EXACTLY.
- ONLY add typed values where blank fields/checkboxes exist.

Persona: {persona_label}

===== FORM LOCATOR GUIDANCE =====
- FL1: Billing provider name + address + phone (4 lines)
- FL2: Pay-to name/address (if different)
- FL3a/3b: Patient control number / Medical record number
- FL4: Type of bill (4-digit, e.g. "0111" = hospital inpatient admit through discharge)
- FL5: Federal tax ID (EIN xx-xxxxxxx)
- FL6: Statement covers period FROM/THROUGH (MMDDYY MMDDYY)
- FL8: Patient name (last, first, MI)
- FL9: Patient address (street/city/state/zip/country)
- FL10: Patient birth date (MMDDYYYY)
- FL11: Patient sex (M/F)
- FL12–FL15: Admission date / hour / type / source
- FL16: Discharge hour
- FL17: Patient status code
- FL18–FL28: Condition codes (2-digit alphanumeric, up to 11)
- FL31–FL34: Occurrence codes + dates (up to 8)
- FL35–FL36: Occurrence span codes + dates
- FL38: Responsible party name/address
- FL39–FL41: Value codes + amounts (up to 12)
- FL42–FL49 (THE SERVICE LINE TABLE — fill row by row):
    42 Rev. Code | 43 Description | 44 HCPCS/Rate/HIPPS | 45 Serv. Date | 46 Units | 47 Total Charges | 48 Non-Covered Charges
- FL50: Payer name (lines A/B/C for primary/secondary/tertiary)
- FL51: Health plan ID
- FL56: Billing provider NPI
- FL57: Other provider ID
- FL58: Insured's name (A/B/C)
- FL59: Patient relationship to insured (A/B/C)
- FL60: Insured's unique ID (A/B/C)
- FL63: Treatment authorization codes
- FL66: DX version qualifier ("0" = ICD-10)
- FL67: Principal diagnosis code + POA indicator
- FL67A–Q: Other diagnosis codes (up to 17)
- FL69: Admitting diagnosis code
- FL70: Patient reason for visit (up to 3)
- FL71: PPS code
- FL72: External cause of injury (E-code) up to 3
- FL74: Principal procedure code + date
- FL74a–e: Other procedure codes + dates
- FL76: Attending provider NPI + last/first name
- FL77–80: Operating / Other / Other provider
- FL81: Code-code field

===== DATA TO FILL =====
{persona_bullets}

===== FORMATTING & STYLE =====
- Type values in a typewriter monospaced or simple sans-serif font, BLACK ink. Use small block letters in narrow boxes.
- Dates: MMDDYY without separators (the form's printed format).
- Dollar amounts: digits with implicit 2-decimal (e.g. "1500.00" — no $ sign).
- For checkboxes: hand-drawn X marks (small). Check ONLY what the persona indicates.
- Service line table: align row-by-row. If the persona has service_line_N keys, put each in row N of the FL42-FL49 grid.
- Diagnosis codes FL67/67A-Q: align in the lettered slot grid.
- DO NOT add text outside printed form boxes. DO NOT alter any printed label or the red/pink ink color.
"""

FORM5500_PROMPT = """TASK: Fill out the IRS Form 5500 (Annual Return/Report of Employee Benefit Plan) in the input image with the values below. Type values into the blank fields; preserve every printed pixel of the form exactly.

===== FORM PRESERVATION — ABSOLUTE RULES =====
- The input is the IRS Form 5500 (or Form 5500-SF / 5500-EZ depending on the source) — Department of Labor + IRS + PBGC joint form. The header reads "Annual Return/Report of Employee Benefit Plan" with an OMB number, and the form has Part I (Annual Report Identification Information), Part II (Basic Plan Information), Part III (Form M-1 Compliance Information), and references to attached Schedules (A, C, D, G, H, I, R, etc.).
- DO NOT regenerate, redesign, or modernize the form. DO NOT change the layout.
- Preserve EVERY printed label, line number (e.g. "1a", "2b(2)"), box border, the IRS/DOL/PBGC seals at top, and all instruction lines (e.g. "For calendar plan year 20___ or fiscal plan year beginning ___ and ending ___") EXACTLY.
- ONLY add typed values + handwritten signature where blank fields exist.

Persona: {persona_label}

===== SECTION GUIDANCE =====
- Top header: plan year (calendar OR fiscal beginning/ending dates)
- Part I — ANNUAL REPORT IDENTIFICATION:
    Box A: check the appropriate box (multiemployer, single-employer, multiple-employer, DFE)
    Box B: check first return / amended / final / short plan year
    Box C: check ERISA Section 412(e)(3) if applicable
    Box D: check if part of group
- Part II — BASIC PLAN INFORMATION:
    1a plan name | 1b 3-digit plan number | 1c effective date
    2a plan sponsor name | 2b mailing address | 2c phone | 2d EIN | 2e business code | 2f telephone
    3a plan administrator (if different) | 3b address | 3c EIN | 3d phone
    4: filed under prior name? if yes, sponsor/EIN/plan number
    5: total participants at beginning of plan year + 6 at end of year (active/retired/terminated vested/total)
    7: was a Schedule X filed? (yes/no for each: A, C, D, G, H, I, R)
    8a–8d: pension plan features (defined benefit, defined contribution, frozen, etc.)
    9a–9b: welfare plan features (health, life, disability, etc.)
    10–12: insurance, fidelity bond, plan terminations
- SIGNATURES at bottom: plan administrator signature + date + name + plan sponsor signature + date + name (and DFE signature where applicable).

===== DATA TO FILL =====
{persona_bullets}

===== FORMATTING & STYLE =====
- Type values in a typewriter monospaced or simple sans-serif font, BLACK ink.
- For checkboxes: hand-drawn X marks (small, in pen). Check ONLY what the persona explicitly indicates.
- For signatures: realistic blue-ink CURSIVE handwriting, looks hand-signed.
- For dollar amounts and participant counts: digits only, no $ sign, right-align in narrow boxes.
- Dates: MM/DD/YYYY in the printed format.
- DO NOT add text outside printed form boxes. DO NOT alter any printed label.
- DO NOT check every checkbox — most boxes should remain unchecked unless the persona explicitly indicates that category.
"""

FORM_CONFIGS: dict[str, dict] = {
    "v1_vaers": {
        "label": "VAERS-1 Vaccine Adverse Event Reporting System form",
        "prompt": VAERS_PROMPT,
        "persona_hint": "Patient + reporter for a vaccine adverse event report. Include patient demographics, reporter info, adverse event narrative + onset date, vaccine table (1-3 vaccines with manufacturer/lot/route/site/dose number), medical history, lab data, facility. Use _checked suffix for outcome and sex booleans.",
        "max_pages": 1,  # VAERS is 1 page front
    },
    "v1_ub04": {
        "label": "UB-04 / CMS-1450 institutional hospital claim form",
        "prompt": UB04_PROMPT,
        "persona_hint": "Hospital/SNF/home-health claim. Include billing provider, patient demographics, admission/discharge dates, payer chain (A/B/C), insured info, principal + secondary diagnoses (ICD-10 with POA indicator), procedures (ICD-10-PCS), and 5-15 service line rows (FL42-49: revenue code + HCPCS + date + units + charges). Use service_line_N_<field> keys for the table. Include condition codes, occurrence codes, value codes.",
        "max_pages": 1,
    },
    "v1_form_5500": {
        "label": "IRS/DOL Form 5500 Annual Return/Report of Employee Benefit Plan",
        "prompt": FORM5500_PROMPT,
        "persona_hint": "ERISA-covered employee benefit plan return. Mix of pension and welfare plans, various employer sizes (small 50-employee 401k vs Fortune 500 health plan vs union multiemployer pension). Include plan sponsor, EIN, plan number, effective date, participant counts (active/retired/terminated/total at start vs end of year), schedule indicators (Schedule A/C/D/H/I/R yes/no), plan features. Use _checked for box selections.",
        "max_pages": 2,  # main 5500 page 1 + optional schedule
    },
}


# =====================================================================
# Persona generation (Claude)
# =====================================================================

_anthropic = None
def get_anthropic():
    global _anthropic
    if _anthropic is None:
        from anthropic import Anthropic
        _anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic


PERSONA_PROMPT_TEMPLATE = """Generate {n} diverse synthetic personas for filling out a "{doc_label}".

{extra_hint}

Each persona is a JSON object with realistic but completely fictional values for
a typical instance of this document. Personas should be DIVERSE across geography,
scale/context, industry/condition mix, and edge cases (joint/multi-party, deceased,
foreign nationals, trusts) where the doc type allows.

Return a single JSON array of {n} persona objects. Each persona must include:
  - `persona_label` — short string e.g. "Pediatric MMR reaction, Iowa"
  - `seed` — integer
  - `fields` — flat dict of SPECIFIC FIELD-HINT → VALUE entries

CRITICAL — FIELD KEY NAMING:
- Each key must be SPECIFIC and DISTINCT. NEVER use bare generic keys like
  "address", "phone", "email", "name", "date". Always qualify.
- Aim for 60-150 keys per persona for a typical multi-section form.

Boolean / checkbox keys MUST end in _checked, _is_yes, _is_owner, _is_male, etc.
For radio-style groups, pick ONE true and leave the others false (or omit).

Formatting:
- Phone numbers: (xxx) 555-0xxx (always 555 exchange so they're obviously demo)
- SSN: 123-45-6789 with last 4 = 6789
- EIN: 12-3456789
- Dates: MM/DD/YYYY
- Dollar amounts: plain numbers (no commas, no $)
- ICD-10 codes (real-form codes like Z23 / E11.9 / I10 are fine)

DO NOT use any real person's name, address, or business. Invent everything.
Return ONLY the JSON array, no preamble or fences.
"""


def _parse_personas_with_salvage(text: str) -> list[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Salvage: walk character-by-character collecting complete top-level objects
    objs = []
    depth = 0
    in_string = False
    escape = False
    start = None
    started_array = False
    for i, c in enumerate(text):
        if not started_array:
            if c == "[":
                started_array = True
            continue
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blob = text[start:i + 1]
                try:
                    objs.append(json.loads(blob))
                except json.JSONDecodeError:
                    pass
                start = None
    return objs


def load_or_generate_personas(doc_type: str, n: int, force: bool = False) -> list[dict]:
    cache = PERSONAS_DIR / f"{doc_type}.json"
    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        if len(cached) >= n:
            return cached[:n]
    PERSONAS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_anthropic()
    cfg = FORM_CONFIGS[doc_type]
    prompt = PERSONA_PROMPT_TEMPLATE.format(
        n=n,
        doc_label=cfg["label"],
        extra_hint=cfg.get("persona_hint", ""),
    )
    print(f"  [persona-gen] calling Claude for {n} personas of {doc_type}…")
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    personas = _parse_personas_with_salvage(text)
    if not personas:
        raise ValueError(f"could not parse personas from Claude: {text[:300]}…")
    cache.write_text(json.dumps(personas, indent=2))
    print(f"  [persona-gen] cached {len(personas)} personas → {cache}")
    return personas[:n]


# =====================================================================
# Persona → bullet list for prompt
# =====================================================================

def persona_to_bullets(persona: dict, max_lines: int = 200) -> str:
    f = persona.get("fields", persona)
    lines = []
    # Sort keys for stable output, but group service_line/condition_code/etc together
    def _sort_key(k: str):
        m = re.match(r"(.+?)_(\d+)(?:_|$)", k)
        if m:
            return (m.group(1), int(m.group(2)), k)
        return (k, 0, k)
    keys = sorted(f.keys(), key=_sort_key)
    for k in keys[:max_lines]:
        v = f[k]
        if v is None or v == "" or v is False:
            continue
        # Skip exclusively false booleans (kept _checked=True ones)
        if isinstance(v, bool):
            v = "✓" if v else ""
            if not v:
                continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, separators=(", ", ":"))
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) if lines else "(no persona data)"


# =====================================================================
# Image-edit + PDF write
# =====================================================================

_openai = None
def get_openai():
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai


def render_page_png(pdf: Path, page_idx: int = 0, dpi: int = 144) -> bytes:
    with fitz.open(pdf) as doc:
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")


def gpt_image_edit(page_png: bytes, prompt: str, model: str = "gpt-image-2") -> bytes | None:
    client = get_openai()
    try:
        resp = client.images.edit(
            model=model,
            image=("page.png", io.BytesIO(page_png), "image/png"),
            prompt=prompt,
            size="1024x1536",
            n=1,
        )
        return base64.b64decode(resp.data[0].b64_json)
    except Exception as e:
        print(f"   [image-gen err] {e}", flush=True)
        return None


def fill_one(src_pdf: Path, persona: dict, out_pdf: Path, cfg: dict, model: str) -> bool:
    prompt = cfg["prompt"].format(
        persona_label=persona.get("persona_label", "synthetic persona"),
        persona_bullets=persona_to_bullets(persona),
    )
    new_doc = fitz.open()
    with fitz.open(src_pdf) as src:
        n_pages = min(src.page_count, cfg.get("max_pages", 1))
        for pidx in range(n_pages):
            png = render_page_png(src_pdf, page_idx=pidx)
            filled = gpt_image_edit(png, prompt, model)
            if filled is None:
                return False
            rect = src[pidx].rect
            page = new_doc.new_page(width=rect.width, height=rect.height)
            page.insert_image(rect, stream=filled, keep_proportion=False)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    new_doc.save(out_pdf, garbage=4, deflate=True)
    new_doc.close()
    return True


# =====================================================================
# Source PDF heuristic — skip obvious guides/instructions
# =====================================================================

GUIDE_RE = re.compile(
    r"(instruction|guide|tips|how[_-]?to|factsheet|complete[d]?|policies|sample|filled)",
    re.IGNORECASE,
)


def pick_blank_sources(type_dir: Path) -> list[Path]:
    """Prefer files with 'blank' / 'form' in name; exclude guides + already-filled samples."""
    all_pdfs = sorted((type_dir / "sources").glob("*.pdf"))
    blanks = [p for p in all_pdfs if not GUIDE_RE.search(p.name) or "blank" in p.name.lower() or "_form" in p.name.lower()]
    # Last-ditch: if we filtered out everything, take all
    return blanks or all_pdfs


# =====================================================================
# main
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", required=True, choices=list(FORM_CONFIGS.keys()),
                    help="Form type to fill")
    ap.add_argument("--doc", help="Restrict to one source PDF (filename)")
    ap.add_argument("--persona", type=int, help="Restrict to one persona index")
    ap.add_argument("--variants", type=int, default=5, help="Personas per blank (default 5)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-suffix", default="_v2fill")
    ap.add_argument("--image-model", default=os.environ.get("IMAGE_GEN_MODEL", "gpt-image-2"))
    ap.add_argument("--regen-personas", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = FORM_CONFIGS[args.type]
    type_dir = DATASETS / args.type
    pdfs = pick_blank_sources(type_dir)
    if args.doc:
        pdfs = [p for p in pdfs if p.name == args.doc]
    if not pdfs:
        print(f"!! no source PDFs found in {type_dir}/sources/")
        return 1

    personas = load_or_generate_personas(args.type, args.variants,
                                         force=args.regen_personas)
    if args.persona is not None:
        personas = [personas[args.persona]]

    work = [(pdf, pi, persona) for pdf in pdfs for pi, persona in enumerate(personas)]
    print(f"Plan: {len(work)} fills ({len(pdfs)} blanks × {len(personas)} personas) for {args.type}")
    print(f"  Workers: {args.workers}, model: {args.image_model}, max_pages: {cfg.get('max_pages', 1)}")
    for p in pdfs:
        print(f"  blank: {p.name}")
    if args.dry_run:
        for pdf, pi, ps in work[:6]:
            print(f"  {pdf.name} × p{pi:02d} ({ps.get('persona_label','?')})")
        return 0

    t0 = time.time()
    n_done = n_fail = 0
    def _job(pdf, pi, persona):
        out = type_dir / "synthetic" / f"{pdf.stem}__p{pi:02d}{args.out_suffix}.pdf"
        ok = fill_one(pdf, persona, out, cfg, args.image_model)
        return (pdf, pi, persona, out, ok)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, p, pi, persona) for p, pi, persona in work]
        for i, fut in enumerate(as_completed(futures), 1):
            pdf, pi, persona, out, ok = fut.result()
            if ok:
                n_done += 1
                sz = out.stat().st_size // 1024
                print(f"  [{i}/{len(work)}] ✓ {out.name} ({sz}KB, {persona.get('persona_label','?')})", flush=True)
            else:
                n_fail += 1
                print(f"  [{i}/{len(work)}] ✗ {pdf.name} × p{pi:02d}", flush=True)
    print(f"\nDone — {n_done}/{len(work)} (fail {n_fail}) in {time.time()-t0:.0f}s")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
