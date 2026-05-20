#!/usr/bin/env python3
"""Fill blank v1_* PDFs with synthetic data and compile to a vision-only PDF.

Pipeline per blank PDF × persona:
  1. Generate 5 diverse personas (Claude text, cached per doc_type)
  2. For each page, choose a fill mode:
       a. ACROFORM    — page has AcroForm widgets and --no-image-gen is not set;
                        write field values directly to widgets (cheap, exact).
       b. IMAGE-GEN   — render page → gpt-image-1 image-edit with persona spec
                        → swap page contents with the returned PNG.
                        Forced when (no widgets) OR (handwriting wanted).
  3. If --force-vision (default), rasterize every page at the end and save as
     image-only PDF so parsers can't cheat by reading the PDF text layer.
  4. Save to <type>/synthetic/<basename>__p<N>.pdf and append to
     <type>/_synthetic_manifest.csv.

Env vars required:
  ANTHROPIC_API_KEY  — for persona generation
  OPENAI_API_KEY     — for gpt-image-1 page editing
  IMAGE_GEN_MODEL    — override image model (default: gpt-image-1)

Run modes:
  --smoke              one small PDF as a demo
  --type v1_cms_1500   restrict to a single type
  --doc <fname>        restrict to a specific blank
  --variants N         personas per template (default 5)
  --handwriting        force gpt-image-1 path even on AcroForm pages
  --no-image-gen       skip gpt-image-1 entirely (AcroForm only; missing pages stay blank)
  --no-force-vision    keep the PDF text layer (default is to rasterize)
  --max-pages N        cap pages per PDF (cost guard)
  --limit N            cap number of blanks processed
  --regen-personas     re-call Claude even if cached
  --dry-run            print plan without writing
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # pymupdf

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
PERSONAS_DIR = DATASETS / "_personas"
PERSONAS_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_MODEL = "claude-sonnet-4-5"
IMAGE_GEN_MODEL_DEFAULT = "gpt-image-2"


# ---------------------------------------------------------------------------
# .env loader (no external deps)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Lazy SDK clients
# ---------------------------------------------------------------------------

_anthropic_client = None
_openai_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (in env or .env)")
        _anthropic_client = Anthropic(api_key=key)
    return _anthropic_client


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set (in env or .env)")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


# ---------------------------------------------------------------------------
# Persona generation (Claude text)
# ---------------------------------------------------------------------------

TYPE_DESCRIPTIONS = {
    "v1_acord_125_126": "ACORD 125/126 Commercial Insurance Application — applicant business profile + general-liability section",
    "v1_acord_4":       "ACORD Homeowner Application or Property Loss Notice",
    "v1_acord_25":      "ACORD 25 Certificate of Liability Insurance",
    "v1_bol_inland":    "Inland Bill of Lading — shipper/consignee/carrier freight document",
    "v1_cd":            "TILA-RESPA Closing Disclosure — final mortgage loan terms + closing costs",
    "v1_clinical_crf":  "Clinical case report form / public-health surveillance form (CDC malaria, NHSN SSI)",
    "v1_clinical_facesheet": "Hospital/clinic patient face sheet / demographic intake form",
    "v1_clinical_consent": "Clinical trial informed consent form (IRB)",
    "v1_cms_1500":      "CMS 1500 Health Insurance Claim Form (medical billing claim)",
    "v1_commercial_invoice": "Commercial invoice for international shipment (customs/CBP)",
    "v1_fnma_710":      "Fannie Mae / Freddie Mac Form 710 Mortgage Assistance Application",
    "v1_med_protocol":  "Clinical trial protocol document",
    "v1_packing_list":  "Export/shipping packing list",
    "v1_pod":           "Proof of delivery / delivery receipt",
    "v1_real_estate_contract": "Residential real estate purchase contract",
    "v1_urar":          "Uniform Residential Appraisal Report (Fannie Mae Form 1004)",
    # ─── Finance split additions (May 2026) ───
    "v1_bol_ocean":     "Ocean Bill of Lading / master bill / sea waybill — container shipment",
    "v1_eob":           "Explanation of Benefits — medical insurance claim summary (service-line table)",
    "v1_sec_13f":       "SEC Form 13F-HR — institutional investor holdings table",
    "v1_sec_def14a":    "SEC Form DEF 14A — proxy statement (executive comp, director comp, audit fee tables)",
    "v1_sec_form_3_4_5": "SEC Form 3/4/5 — insider beneficial-ownership transactions (checkbox grids)",
    "v1_sec_form_adv":  "SEC Form ADV — Investment Adviser registration (massive checkbox + form-field doc)",
    "v1_fec_form5":     "FEC Form 5 — Independent Expenditures",
    "v1_fec_form6":     "FEC Form 6 — 48-Hour Notice (loans / late debts)",
    "v1_fec_form7":     "FEC Form 7 — Communications Cost",
    "v1_ffiec_call_report": "FFIEC UBPR — Bank Call Report (dense financial schedules RC-A..RC-S, RI)",
    "v1_ffiec_hmda":    "FFIEC HMDA aggregate disclosure (mortgage application table)",
    "v1_ferc_form1":    "FERC Form 1 — Electric Utility Annual Report (Account 100-940 schedules)",
    "v1_ferc_form2":    "FERC Form 2 — Natural Gas Pipeline Annual",
    "v1_naic_annual_statement": "NAIC Annual Statement — insurance statutory (Schedule D/F/P)",
    "v1_ferpa_release": "FERPA Authorization to Release Education Records",
    "v1_irs_specialty": "IRS specialty forms — Schedule K-1, Form 2848, Form 8949, Form 5471",
    "v1_intl_tax_ca":   "Canada CRA tax forms — T1, T2, T4, T5, T2125",
    "v1_intl_tax_uk":   "UK HMRC tax forms — SA100, SA101, SA105, P11D, CT600",
    "v1_intl_tax_au":   "Australia ATO tax forms — Individual return, BAS",
    "v1_intl_tax_in":   "India ITR-1 (Sahaj) through ITR-7",
    "v1_intl_tax_de":   "Germany ESt 1A Mantelbogen + Anlagen N/Kind/R/KAP",
    "v1_intl_tax_nz":   "NZ IR3/IR4/IR526",
    "v1_intl_tax_sg":   "Singapore IRAS Form B/B1/P",
    "v1_intl_tax_za":   "South Africa SARS ITR12/IT14",
    "v1_intl_tax_jp":   "Japan NTA Kakutei Shinkoku",
    "v1_intl_tax_br":   "Brazil DIRPF",
}


PERSONA_PROMPT = """Generate {n} diverse synthetic personas for filling out a "{doc_type_label}".

Each persona is a JSON object with realistic but completely fictional values for
a typical instance of this document. The personas should be DIVERSE across:
- Geography (different US states, or international where appropriate)
- Scale / context (small/personal vs mid-size vs large)
- Industry / condition mix
- Edge cases (joint owners, deceased parties, foreign nationals, trusts) where
  the doc type allows

Return a single JSON array of {n} persona objects. Each persona must include:
  - `persona_label` — short string e.g. "Small contractor, Texas"
  - `seed` — integer
  - `fields` — flat dict of SPECIFIC FIELD-HINT → VALUE entries

CRITICAL — FIELD KEY NAMING:
- Each key must be SPECIFIC and DISTINCT. NEVER use bare generic keys like
  "address", "phone", "email", "name", "date". Always qualify:
    GOOD:  applicant_email, co_applicant_email, agency_email,
           applicant_mailing_street, applicant_mailing_city, applicant_mailing_state,
           applicant_mailing_zip, property_street, property_city,
           policy_effective_date, policy_expiration_date, date_of_birth,
           applicant_primary_phone, applicant_secondary_phone, agency_phone,
           applicant_first_name, applicant_last_name, co_applicant_first_name
    BAD:   address, phone, email, date, name (these will match too many fields)
- Forms often repeat similar fields in different sections (applicant vs co-applicant,
  primary vs secondary, billing vs property). Generate DISTINCT keys for each section.
- Aim for 60-120 keys per persona for a typical multi-section form.

Boolean / checkbox keys MUST end in a verb-style suffix like _checked, _is_yes,
_is_owner, _is_male, _is_homeowner, _smoker. Set true/false explicitly.
For radio-style groups (Yes/No, M/F/O, Primary/Secondary), pick ONE true and
leave the others false (or omit).

Formatting rules:
- Phone numbers: (xxx) 555-0xxx (always 555 exchange so they're obviously demo)
- SSN: 123-45-6789 style with last 4 = 6789 (clearly fake)
- EIN: 12-3456789
- Dates: MM/DD/YYYY
- Dollar amounts: plain numbers (no commas, no $)

DO NOT use any real person's name, address, or business. Invent everything. Use
naming patterns that signal "fake": e.g. "Demo Construction LLC", "Sample Patient",
"Test Holdings Inc." — but vary it (some personas can use plausible non-real names
like "Hawthorne Partners Ltd" so the dataset isn't monotone).

Return ONLY the JSON array, no preamble or fences.
"""


def load_or_generate_personas(doc_type: str, n: int, force: bool = False) -> list[dict]:
    cache = PERSONAS_DIR / f"{doc_type}.json"
    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        if len(cached) >= n:
            return cached[:n]
    client = get_anthropic_client()
    label = TYPE_DESCRIPTIONS.get(doc_type, doc_type)
    prompt = PERSONA_PROMPT.format(n=n, doc_type_label=label)
    # 16k output for dense forms (acord_125_126 etc.) — was 8k, hit limit mid-string
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    personas = _parse_personas_with_salvage(text)
    if not personas:
        raise ValueError(f"could not parse personas from Claude response: {text[:200]}…")
    cache.write_text(json.dumps(personas, indent=2))
    return personas[:n]


def _parse_personas_with_salvage(text: str) -> list[dict]:
    """Best-effort JSON parse — falls back to salvaging complete objects from a truncated array."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find the [...] envelope and re-parse
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Salvage: walk character-by-character counting braces, collect every
    # complete top-level object inside the array.
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
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start:i+1]
                    try:
                        objs.append(json.loads(chunk))
                    except json.JSONDecodeError:
                        pass
                    start = None
    return objs


# ---------------------------------------------------------------------------
# AcroForm widget filling
# ---------------------------------------------------------------------------

def label_to_key(label: str) -> str:
    """Normalize label to snake_case token sequence (joined by _)."""
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return s[:80]


# Words to ignore when computing token overlap (too common to be discriminating)
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "or", "and", "is", "no",
    "if", "for", "by", "as", "at", "with", "from", "this", "that",
    "please", "see", "use", "applicable", "include", "complete",
    "address", "name", "date", "phone", "email", "number", "code",
    "type", "info", "information", "amount", "yes", "value", "describe",
    "specify", "select", "all", "any", "other", "etc",
}


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _token_overlap_score(a_tokens: list[str], b_tokens: list[str]) -> tuple[int, int]:
    """Return (specific_overlap, total_overlap) where specific_overlap excludes stopwords."""
    sa = set(a_tokens)
    sb = set(b_tokens)
    overlap = sa & sb
    specific = overlap - _STOPWORDS
    return (len(specific), len(overlap))


def lookup_value(label: str, kind: str, persona_fields: dict) -> Any:
    if kind == "signature":
        return None  # drawn during handwriting pass

    key = label_to_key(label)
    if kind == "checkbox":
        # Strict: only check if persona has explicit True/False match
        v = persona_fields.get(key)
        if isinstance(v, bool):
            return v
        # Try keys whose token-overlap is high
        best = None
        best_score = (0, 0)
        for k, val in persona_fields.items():
            if not isinstance(val, bool):
                continue
            sc = _token_overlap_score(_tokens(key), _tokens(k))
            if sc > best_score:
                best, best_score = val, sc
        if best is not None and best_score[0] >= 1:
            return best
        return False  # default uncheck (was random 30%)

    # Text path
    if not key:
        return ""
    label_tokens = _tokens(key)
    # 1. Exact key match
    if key in persona_fields and isinstance(persona_fields[key], (str, int, float)):
        return str(persona_fields[key])
    # 2. Token-overlap match — require at least 2 specific (non-stopword) tokens overlap,
    #    OR 1 specific token if the label is short (<=3 tokens)
    best_val = None
    best_score = (0, 0)
    min_specific = 1 if len(label_tokens) <= 3 else 2
    for k, val in persona_fields.items():
        if not isinstance(val, (str, int, float)):
            continue
        sc = _token_overlap_score(label_tokens, _tokens(k))
        if sc[0] < min_specific:
            continue
        if sc > best_score:
            best_val = val
            best_score = sc
    if best_val is not None:
        return str(best_val)

    # 3. Conservative generic fallback — require very specific label patterns
    return _generic_text_for_strict(label, label_tokens)


_DATE_SUFFIXES = ("date", "_date", "date_of_birth", "birthdate", "effective_date",
                  "expiration_date", "issued_date", "discharge_date", "admission_date")


def _generic_text_for_strict(label: str, tokens: list[str]) -> str:
    """Only fire when label is unambiguously a generic type, not a custom one
    that just contains the trigger word."""
    L = label.lower()
    last = tokens[-1] if tokens else ""
    # Date: must end in 'date' or be exactly "date of birth" / etc.
    if last == "date" and len(tokens) <= 4:
        return ""  # let it be empty; date probably has its own persona key
    # Phone: must be exactly "phone" or end in "_phone" / "phone_number"
    if last == "phone" or (len(tokens) >= 2 and tokens[-2:] == ["phone", "number"]):
        return ""  # we don't want to plaster the same demo phone everywhere
    # Email: same
    if last == "email" or (len(tokens) >= 2 and tokens[-2:] == ["email", "address"]):
        return ""
    # Zip
    if last == "zip" or "zip_code" in L:
        return ""
    # Amount fields — leave blank rather than write '1000' everywhere
    return ""


def _infer_label_from_proximity(page: fitz.Page, widget_rect: fitz.Rect,
                                words: list, max_left: float = 220,
                                max_above: float = 14) -> str:
    """Find printed text immediately to the LEFT (same row) or ABOVE the widget."""
    wx0, wy0, wx1, wy1 = widget_rect
    # Same-row left-side candidates
    left_cands = []
    above_cands = []
    for (x0, y0, x1, y1, text, *_) in words:
        if not text or not text.strip():
            continue
        same_row = not (y1 < wy0 - 3 or y0 > wy1 + 3)
        if same_row and x1 <= wx0 + 1 and (wx0 - x1) <= max_left:
            left_cands.append((wx0 - x1, x0, y0, text))
            continue
        # Above (close vertically, horizontally overlapping)
        vert_gap = wy0 - y1
        if 0 < vert_gap < max_above and not (x1 < wx0 - 4 or x0 > wx1 + 4):
            above_cands.append((vert_gap, x0, y0, text))
    if left_cands:
        # Take up to ~5 closest words to the left, in left-to-right order
        left_cands.sort()
        chosen = left_cands[:6]
        chosen.sort(key=lambda t: t[1])  # sort by x for natural reading
        return " ".join(t[3] for t in chosen)[:80]
    if above_cands:
        above_cands.sort()
        chosen = above_cands[:6]
        chosen.sort(key=lambda t: (t[2], t[1]))
        return " ".join(t[3] for t in chosen)[:80]
    return ""


def fill_acroform_widgets(page: fitz.Page, persona_fields: dict, log: list[str]) -> tuple[int, int]:
    """Returns (n_filled, n_with_values).
    n_with_values < n_filled if many fields got empty strings (label match failed)."""
    words = page.get_text("words")
    n_filled = 0
    n_with_values = 0
    for w in (page.widgets() or []):
        ft = w.field_type_string.lower()
        kind = ("checkbox" if "checkbox" in ft or "btn" in ft
                else "signature" if "signature" in ft else "text")
        # Try explicit label first; then proximity-inferred
        label = (w.field_label or "").strip()
        if not label:
            label = _infer_label_from_proximity(page, w.rect, words)
        if not label:
            label = w.field_name or ""
        v = lookup_value(label, kind, persona_fields)
        try:
            if kind == "checkbox":
                w.field_value = bool(v)
                w.update()
                n_filled += 1
                if v:
                    n_with_values += 1
            elif kind == "text":
                if v:
                    w.field_value = str(v)
                    w.update()
                    n_filled += 1
                    n_with_values += 1
                else:
                    n_filled += 1
        except Exception:
            pass
    return n_filled, n_with_values


# ---------------------------------------------------------------------------
# gpt-image-1 page editing
# ---------------------------------------------------------------------------

IMAGE_EDIT_PROMPT = """TASK: You are given an image of a BLANK PRINTED FORM. Your job is to produce the SAME EXACT FORM, with values written into the existing fillable areas. The output is meant to look like that same blank form after someone filled it in by hand or typewriter.

ABSOLUTE RULES — DO NOT VIOLATE:
1. Treat the input image as the canonical form layout. Do NOT redesign, re-style, re-order, re-flow, or simplify any portion of the form.
2. Reproduce EVERY printed label, section heading, table border, line, checkbox, footer text, page number, watermark, and logo at the EXACT SAME pixel position and with the EXACT SAME wording as the input image. Letter-perfect.
3. The output image dimensions, aspect ratio, margins, and column structure must match the input exactly. If the input has 4 columns, the output has the same 4 columns in the same place.
4. Only ADD ink where there is a blank fillable area (underline, empty box, empty cell, empty checkbox). Do not modify anything that was already printed.
5. Typed text: clean black ink, sized so it fits inside the blank area without overlapping printed labels.
6. Handwriting (signatures, initials, "sign here" lines): blue or black cursive scribble, realistic.
7. Checkboxes: draw a hand-drawn X or ✓ ONLY for boxes whose label matches a value in the persona below. Leave all other boxes EMPTY (do not check them).
8. If a fillable area has no obvious value to write (because none of the persona values match its label), leave it BLANK.
9. Do not add commentary, watermarks, "DEMO" stamps, or anything not in the original.

Persona context: {persona_label}

Values to fill in (KEY → VALUE, match each to the most relevant blank area on the form):
{values_md}

REMINDER: The output must be visually indistinguishable from the input form, except for the added field values. Preserve everything else exactly.
"""


def render_page_png(page: fitz.Page, dpi: int = 144) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def persona_values_markdown(persona_fields: dict, max_entries: int = 60) -> str:
    lines = []
    for k, v in list(persona_fields.items())[:max_entries]:
        if not isinstance(v, (str, int, float, bool)):
            continue
        nice = k.replace("_", " ").title()
        lines.append(f"- {nice}: {v}")
    return "\n".join(lines)


def gpt_image_edit_page(page_png: bytes, persona: dict, page_size_pt: tuple[float, float],
                        model: str, log: list[str]) -> bytes | None:
    client = get_openai_client()
    label = persona.get("persona_label", "synthetic persona")
    fields = persona.get("fields", persona)  # back-compat if persona is flat
    values_md = persona_values_markdown(fields)
    prompt = IMAGE_EDIT_PROMPT.format(persona_label=label, values_md=values_md)
    # Choose output size to roughly preserve aspect ratio.
    # gpt-image-1 valid sizes: 1024x1024, 1024x1536, 1536x1024
    w, h = page_size_pt
    if h >= w:  # portrait → 1024x1536
        size = "1024x1536"
    else:       # landscape → 1536x1024
        size = "1536x1024"
    try:
        resp = client.images.edit(
            model=model,
            image=("page.png", io.BytesIO(page_png), "image/png"),
            prompt=prompt,
            size=size,
            n=1,
        )
        b64 = resp.data[0].b64_json
        return base64.b64decode(b64)
    except Exception as e:
        log.append(f"   [image-gen err] {e}")
        return None


def replace_page_with_image(doc: fitz.Document, page_idx: int, img_png: bytes):
    """Replace page contents with the given PNG, preserving page dimensions."""
    old_page = doc[page_idx]
    rect = old_page.rect
    # Make a new blank page same size, insert image, then delete old page
    doc.insert_page(page_idx, width=rect.width, height=rect.height)
    new_page = doc[page_idx]
    new_page.insert_image(rect, stream=img_png, keep_proportion=False)
    # Now delete the original (which shifted to page_idx + 1)
    doc.delete_page(page_idx + 1)


# ---------------------------------------------------------------------------
# Force vision: rasterize whole PDF
# ---------------------------------------------------------------------------

def rasterize_to_image_only(in_pdf: Path, out_pdf: Path, dpi: int = 150):
    src = fitz.open(in_pdf)
    dst = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        rect = page.rect
        new_page = dst.new_page(width=rect.width, height=rect.height)
        new_page.insert_image(rect, pixmap=pix)
    dst.save(out_pdf, garbage=4, deflate=True)
    dst.close()
    src.close()


# ---------------------------------------------------------------------------
# Per-PDF orchestrator
# ---------------------------------------------------------------------------

def fill_one_pdf(src_pdf: Path, persona: dict, out_pdf: Path,
                 image_model: str,
                 use_image_gen: bool, force_handwriting: bool,
                 force_vision: bool, max_pages: int | None,
                 log: list[str]):
    """Returns dict with stats per page."""
    fields = persona.get("fields", persona)
    doc = fitz.open(src_pdf)
    n_pages = doc.page_count
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)
    stats = {"acroform_pages": 0, "imagegen_pages": 0, "skipped_pages": 0,
             "widgets_filled": 0}

    # First pass: classify each page and fill AcroForm widgets where applicable.
    # If AcroForm fills end up mostly empty (no usable labels), fall back to image-gen.
    page_modes: list[str] = []  # one of: acroform, imagegen, skipped
    ACRO_VALUE_RATIO_MIN = 0.30  # require at least 30% of widgets to receive a value
    for pidx in range(n_pages):
        page = doc[pidx]
        widgets = list(page.widgets() or [])
        if widgets and not force_handwriting:
            n_filled, n_with_values = fill_acroform_widgets(page, fields, log)
            ratio = (n_with_values / max(n_filled, 1)) if n_filled else 0
            if n_filled > 5 and ratio < ACRO_VALUE_RATIO_MIN and use_image_gen:
                # Too many empty fields — undo widget edits (re-open the page)
                # and route to image-gen instead. Simpler: just mark page as
                # imagegen; the few filled widgets remain but won't matter once
                # we replace the page with a fully generated image.
                page_modes.append("imagegen")
                log.append(f"   page {pidx+1}: ACROFORM had low match rate "
                           f"({n_with_values}/{n_filled}) → falling back to IMAGE-GEN")
                # NOTE: we don't reset the widget edits; the page is overwritten anyway.
            else:
                stats["acroform_pages"] += 1
                stats["widgets_filled"] += n_with_values
                page_modes.append("acroform")
                log.append(f"   page {pidx+1}: ACROFORM ({n_with_values}/{n_filled} widgets got values)")
        elif use_image_gen:
            page_modes.append("imagegen")
        else:
            page_modes.append("skipped")
            stats["skipped_pages"] += 1
            log.append(f"   page {pidx+1}: SKIPPED (no widgets and --no-image-gen)")

    # Bake widgets into page content so the values actually render when rasterized
    if stats["widgets_filled"] > 0:
        try:
            doc.bake(annots=True, widgets=True)
            log.append(f"   baked {stats['widgets_filled']} widgets into page content")
        except Exception as e:
            log.append(f"   [warn] bake() failed: {e}")

    # Second pass: gpt-image-1 for pages flagged imagegen
    for pidx, mode in enumerate(page_modes):
        if mode != "imagegen":
            continue
        if pidx >= doc.page_count:
            continue
        page = doc[pidx]
        png = render_page_png(page, dpi=144)
        filled_png = gpt_image_edit_page(png, persona,
                                         (page.rect.width, page.rect.height),
                                         image_model, log)
        if filled_png is None:
            stats["skipped_pages"] += 1
            log.append(f"   page {pidx+1}: IMAGE-GEN failed — leaving original")
            continue
        replace_page_with_image(doc, pidx, filled_png)
        stats["imagegen_pages"] += 1
        log.append(f"   page {pidx+1}: IMAGE-GEN ({image_model})")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if force_vision:
        # Save to a temp text-layer PDF, then rasterize
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        doc.save(tmp_path, garbage=4, deflate=True)
        doc.close()
        rasterize_to_image_only(tmp_path, out_pdf, dpi=150)
        tmp_path.unlink(missing_ok=True)
        log.append(f"   → rasterized to image-only PDF (vision-forcing)")
    else:
        doc.save(out_pdf, garbage=4, deflate=True)
        doc.close()
    return stats


# ---------------------------------------------------------------------------
# Manifest discovery + write-back
# ---------------------------------------------------------------------------

def list_blanks(doc_type: str | None = None, doc_name: str | None = None) -> list[tuple[str, str]]:
    out = []
    for folder in sorted(DATASETS.glob("v1_*")):
        if doc_type and folder.name != doc_type:
            continue
        m = folder / "_sources_manifest.csv"
        if not m.exists():
            continue
        with m.open() as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if "blank" not in (r.get("is_filled") or "").lower():
                continue
            if doc_name and r["local_filename"] != doc_name:
                continue
            out.append((folder.name, r["local_filename"]))
    return out


def append_synth_manifest(doc_type: str, parent_fname: str, out_fname: str,
                          persona_label: str, persona_idx: int,
                          mode_flags: dict, stats: dict):
    folder = DATASETS / doc_type
    mpath = folder / "_synthetic_manifest.csv"
    fields = ["filename", "parent_blank", "persona_idx", "persona_label",
              "image_gen", "handwriting", "force_vision",
              "acroform_pages", "imagegen_pages", "widgets_filled",
              "created_at"]
    new_row = {
        "filename": out_fname,
        "parent_blank": parent_fname,
        "persona_idx": persona_idx,
        "persona_label": persona_label,
        "image_gen": mode_flags.get("image_gen"),
        "handwriting": mode_flags.get("handwriting"),
        "force_vision": mode_flags.get("force_vision"),
        "acroform_pages": stats.get("acroform_pages", 0),
        "imagegen_pages": stats.get("imagegen_pages", 0),
        "widgets_filled": stats.get("widgets_filled", 0),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if mpath.exists():
        with mpath.open() as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    # Keep only the latest entry per filename
    rows = [r for r in rows if r.get("filename") != out_fname] + [new_row]
    with mpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", help="restrict to one v1_* doc type")
    ap.add_argument("--doc", help="restrict to one specific filename (use with --type)")
    ap.add_argument("--variants", type=int, default=5, help="personas per template")
    ap.add_argument("--handwriting", action="store_true",
                    help="force gpt-image-1 even on AcroForm pages")
    ap.add_argument("--no-image-gen", action="store_true",
                    help="skip gpt-image-1 entirely; leave non-widget pages blank")
    ap.add_argument("--no-force-vision", action="store_true",
                    help="keep PDF text layer (default rasterizes for vision-only output)")
    ap.add_argument("--image-model", default=os.environ.get("IMAGE_GEN_MODEL", IMAGE_GEN_MODEL_DEFAULT))
    ap.add_argument("--max-pages", type=int, default=None,
                    help="cap pages per PDF (cost guard for image-gen)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of blanks processed")
    ap.add_argument("--smoke", action="store_true",
                    help="just one small blank as a demo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--regen-personas", action="store_true")
    ap.add_argument("--out-suffix", default="",
                    help="extra suffix on output basename (e.g. '_hw' produces foo__p00_hw.pdf)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers for the (PDF × persona) loop (default 1)")
    args = ap.parse_args()

    use_image_gen = not args.no_image_gen
    force_vision = not args.no_force_vision
    mode_flags = {
        "image_gen": use_image_gen,
        "handwriting": args.handwriting,
        "force_vision": force_vision,
    }

    blanks = list_blanks(args.type, args.doc)
    if not blanks:
        print("no blanks match the filter")
        return 1

    if args.smoke:
        def page_count(p):
            try:
                with fitz.open(DATASETS / p[0] / "sources" / p[1]) as d:
                    return d.page_count
            except Exception:
                return 999
        blanks = sorted(blanks, key=page_count)[:1]
    elif args.limit:
        blanks = blanks[: args.limit]

    print(f"Plan: {len(blanks)} blank(s) × {args.variants} persona(s) = "
          f"{len(blanks) * args.variants} output PDF(s)")
    print(f"  image-gen={use_image_gen}  handwriting={args.handwriting}  "
          f"force-vision={force_vision}  image-model={args.image_model}")
    if args.max_pages is not None:
        print(f"  --max-pages={args.max_pages}")
    if args.dry_run:
        for dt, fn in blanks:
            print(f"  {dt}/{fn}")
        return 0

    # Build full work list (PDF × persona)
    work_items = []
    for dt, fn in blanks:
        src = DATASETS / dt / "sources" / fn
        if not src.exists():
            print(f"[skip] {dt}/{fn} not on disk")
            continue
        try:
            personas = load_or_generate_personas(dt, args.variants, force=args.regen_personas)
        except Exception as e:
            print(f"[err] persona load {dt}: {e}")
            continue
        for pi, persona in enumerate(personas):
            label = persona.get("persona_label", f"p{pi}")
            out = DATASETS / dt / "synthetic" / f"{src.stem}__p{pi:02d}{args.out_suffix}.pdf"
            work_items.append((dt, fn, src, persona, pi, label, out))

    print(f"\nWork queue: {len(work_items)} items, --workers={args.workers}")
    t0 = time.time()
    n_out = 0
    lock = __import__("threading").Lock()

    def _process(item):
        dt, fn, src, persona, pi, label, out = item
        log: list[str] = []
        try:
            stats = fill_one_pdf(
                src, persona, out,
                image_model=args.image_model,
                use_image_gen=use_image_gen,
                force_handwriting=args.handwriting,
                force_vision=force_vision,
                max_pages=args.max_pages,
                log=log,
            )
        except Exception as e:
            return (dt, fn, out, label, log, str(e))
        return (dt, fn, out, label, log, None)

    if args.workers <= 1:
        for item in work_items:
            dt, fn, _, _, pi, label, out = item
            print(f"\n→ {dt}/{fn} × [{pi+1}] {label}")
            res = _process(item)
            dt2, fn2, out2, label2, log, err = res
            if err:
                print(f"   [fatal err] {err}")
            else:
                for line in log: print(line)
                append_synth_manifest(dt2, fn2, out2.name, label2, pi, mode_flags, {})
                n_out += 1
                size_kb = out2.stat().st_size // 1024 if out2.exists() else 0
                print(f"   wrote {out2.relative_to(REPO)} ({size_kb} KB)")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Launching {args.workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_process, it) for it in work_items]
            for i, fut in enumerate(as_completed(futures), 1):
                dt2, fn2, out2, label2, log, err = fut.result()
                if err:
                    print(f"  [{i}/{len(work_items)}] FAIL {dt2}/{fn2}: {err}", flush=True)
                else:
                    n_out += 1
                    with lock:
                        # persona_idx isn't easy to recover here; reconstruct from filename
                        m = re.search(r"__p(\d+)", out2.stem)
                        pi = int(m.group(1)) if m else 0
                        append_synth_manifest(dt2, fn2, out2.name, label2, pi, mode_flags, {})
                    size_kb = out2.stat().st_size // 1024 if out2.exists() else 0
                    print(f"  [{i}/{len(work_items)}] ✓ {out2.relative_to(REPO)} ({size_kb}KB, {label2})", flush=True)
    print(f"\nDone — {n_out}/{len(work_items)} synthetic PDFs in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
