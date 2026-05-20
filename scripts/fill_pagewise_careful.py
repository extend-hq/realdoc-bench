#!/usr/bin/env python3
"""Careful pagewise template fill — vision-derived schema → coherent persona →
checkbox-disciplined gpt-image-2 fill.

Pipeline per template PDF:
  1. For each page, render image and ask Claude vision to extract a per-page
     SCHEMA: every form field (with line/box id), every checkbox group, every
     table.  Output = structured JSON.
  2. Show user the schema.
  3. Ask Claude to construct ONE coherent persona/scenario that fits the form
     type + all pages.  Output = persona JSON.
  4. Show user the persona.
  5. For each page, decide WHICH checkboxes to check based on persona; produce
     a "fill plan" (per-field values + explicit checkbox decisions).
  6. Render gpt-image-2 fill with explicit per-page prompt:
        - List exact field values to type
        - List EXACTLY which checkboxes to check; "LEAVE ALL OTHERS UNCHECKED"
        - Preserve form layout absolutely
  7. Save filled per-page PDF to v1.1_pagewise_filled/<type>/<stem>__p<NN>_filled.pdf

Usage:
  python -u scripts/fill_pagewise_careful.py --pdf <path>
  python -u scripts/fill_pagewise_careful.py --template-idx 21
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
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
OUT_ROOT = DATASETS / "v1.1_pagewise_filled"


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k):
            os.environ[k] = v
load_dotenv()


_anthropic = None
def get_anthropic():
    global _anthropic
    if _anthropic is None:
        from anthropic import Anthropic
        _anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic


_openai = None
def get_openai():
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: per-page schema extraction via Claude vision
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_EXTRACT_PROMPT = """You are extracting a STRUCTURED SCHEMA of a single page of a form. Return ONLY a JSON object — no preamble, no fences.

Look carefully at this page and identify EVERY form field, EVERY checkbox group, and EVERY table.

Schema format:
{{
  "page_label": "<short string identifying this page — e.g. 'Form 2848 Page 1: Power of Attorney parts I-V'>",
  "form_id": "<official form number if visible, e.g. 'Form 2848 (Rev. ...)' >",
  "sections": [
    {{
      "section_id": "<numbered line e.g. '1' or 'Part II'>",
      "section_label": "<what this section is for>",
      "form_fields": [
        {{"field_id": "<e.g. '1a'>", "label": "<what to fill in>", "field_type": "string|number|date|address|phone|ssn|ein|signature|other"}}
      ],
      "checkbox_groups": [
        {{
          "group_id": "<e.g. '3' or 'Filing status'>",
          "group_label": "<what this group is asking>",
          "selection_type": "exactly_one | at_most_one | any_combination",
          "options": [
            {{"option_id": "a", "label": "Tax Information Authorization"}},
            {{"option_id": "b", "label": "Other (specify): ____________"}}
          ]
        }}
      ],
      "tables": [
        {{"table_id": "<e.g. 'Part II Representatives'>", "columns": ["Name & Address", "CAF No.", "PTIN", "Telephone", "Fax"], "n_rows_blank": 4}}
      ]
    }}
  ]
}}

RULES:
- For checkbox_groups: identify the actual grouping intended by the form (e.g. all "filing status" options together). Set selection_type based on the form's instruction.
- For tables: include all visible column headers + how many empty rows are available.
- Capture EVERY visible checkbox — do not skip any.
- form_fields should NOT include checkbox-based fields (those go in checkbox_groups).
- Keep labels SHORT (≤ 15 words).
"""


def render_page_b64(pdf_path: Path, page_idx: int, dpi: int = 150) -> str:
    with fitz.open(pdf_path) as doc:
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return base64.b64encode(pix.tobytes("png")).decode()


def _parse_json_lenient(text: str) -> dict:
    """Parse JSON, falling back to json_repair if standard parsing fails."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception as e:
            raise json.JSONDecodeError(f"json_repair also failed: {e}", text, 0)


def extract_page_schema(pdf_path: Path, page_idx: int, model: str = "claude-sonnet-4-5") -> dict:
    b64 = render_page_b64(pdf_path, page_idx)
    client = get_anthropic()
    resp = client.messages.create(
        model=model, max_tokens=16000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": SCHEMA_EXTRACT_PROMPT},
        ]}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return _parse_json_lenient(text)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: coherent persona generation
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_PROMPT = """You will design ONE coherent scenario/persona for filling out this form. Look at the schemas of ALL pages below and design values that make sense TOGETHER (not random).

Form type: {form_type_label}

Page schemas:
{schemas_block}

Return ONLY a JSON object with:
{{
  "scenario_summary": "<≤ 30 words describing the realistic scenario>",
  "field_values": {{
    "<field_id from any page>": "<realistic value>",
    ...
  }},
  "checkbox_decisions": {{
    "<group_id from any page>": ["<option_ids to check>"]  (note: empty list = NONE checked; for exactly_one groups, exactly 1 option)
  }},
  "table_rows": {{
    "<table_id>": [
      {{"<column>": "<value>", ...}},  // 1 row of realistic data
      ...
    ]
  }}
}}

RULES:
- Use realistic but synthetic data (no real person's name/SSN; use 555 phone exchange, 123-45-6789 SSN style, demo addresses).
- Be CONSERVATIVE with checkboxes: for "at_most_one" groups, only check if the scenario REQUIRES it; for "any_combination", only check the ones that match the scenario.
- For tables: only fill 1-3 rows (don't fill every blank row); be plausible.
- Be coherent: if the form is a Power of Attorney for tax matters, ALL the data should be about that one tax case (one taxpayer, one representative, one set of tax years/forms).
"""


def build_persona(form_type_label: str, page_schemas: list[dict], model: str = "claude-sonnet-4-5") -> dict:
    schemas_text = json.dumps(page_schemas, indent=2)
    prompt = PERSONA_PROMPT.format(
        form_type_label=form_type_label,
        schemas_block=schemas_text[:30000],
    )
    client = get_anthropic()
    resp = client.messages.create(
        model=model, max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return _parse_json_lenient(text)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: render gpt-image-2 fill prompt per page
# ─────────────────────────────────────────────────────────────────────────────

FILL_PROMPT_TEMPLATE = """TASK: Fill out the form in the input image with the values below. The output must be VISUALLY IDENTICAL to the input form except that empty fields and checkboxes are filled in.

===== FORM PRESERVATION — ABSOLUTE RULES (READ BEFORE FILLING) =====
- This is the form: {form_id}
- DO NOT regenerate the form, redesign the layout, modernize the typography, change ANY printed text, alter borders or boxes, or replace it with a "clean" version.
- Preserve every printed label, instruction, line number, table border, header, footer, watermark, and ink color EXACTLY.
- ONLY add typed values where empty fillable areas exist, AND check the specific checkboxes listed below.

Scenario: {scenario_summary}

===== FIELD VALUES (type these into the corresponding fields) =====
{field_values_block}

===== CHECKBOX DECISIONS — HARD CONSTRAINTS =====
Check EXACTLY these checkboxes and NO OTHERS:
{checkboxes_to_check_block}

DO NOT check any other checkbox on this page. Many forms have dozens of checkboxes; leave all but the ones above EMPTY.

===== TABLE ROWS (fill these rows; leave other rows BLANK) =====
{table_rows_block}

===== FORMATTING =====
{formatting_block}
- Checkbox marking: see the WRITING INSTRUMENT block below — use the EXACT marker style specified for this page. Each page's checkbox marks should look like they were made by the same instrument as the text values on that page. Do NOT use a generic ✓ symbol; render an organic, hand-drawn mark.
- Dates: MM/DD/YYYY in the printed format (use slashes if the form has them; otherwise plain text)
- Dollar amounts: digits + decimal, no $ sign in box-format fields

{instrument_block}

===== SIGNATURE RULES — be realistic =====
- Signatures must look HAND-DRAWN with REAL CURSIVE handwriting, in BLUE INK (or black if the document context implies it).
- A real signature has these qualities:
    * Flowing connected letters — NOT block printing, NOT individual disconnected letters
    * First name is usually MORE legible than the last (last name often becomes a stylized flourish or scribble)
    * Slight variation in stroke pressure / thickness across the signature
    * Some letters may be loops or partly illegible — that's normal
    * Slight slant (usually rightward) and natural baseline wobble (not perfectly straight)
    * Signature should be roughly aligned to the signature LINE, sitting on or slightly above it
    * Size: about 60-80% of the signature line height
- DO NOT produce a block-letter "signature" or a perfectly clean script. Make it look like a real person signed their name on paper.
- If the SAME signer appears more than once on a page, the signature shape should be CONSISTENT but not pixel-identical (slight natural variation, like a real person signing twice).
- For initialed lines (e.g. "INIT. HERE __"): use 2-3 letters in cursive, also slightly imperfect.

===== DIFFERENT SIGNERS ON THE SAME PAGE — VERY IMPORTANT =====
{signers_block}
- Different signers MUST have visually DIFFERENT handwriting (different slant, different letter shapes, different baseline, different stroke weight). A patient signature and a physician/witness signature on the same page MUST look like they were signed by two different people.
- Physician/doctor signatures are typically faster, smaller, more illegible (a quick scrawl ending in a tail).
- Patient signatures tend to be more deliberate, slightly larger, more legible first name.
- Witness/staff/interpreter signatures vary — pick a distinct style per role.

- DO NOT add text outside the printed form areas.
"""


# Per-convention formatting blocks for the {formatting_block} placeholder
_FORMATTING_TYPED = (
    "- Typed text: simple sans-serif or typewriter font, black ink, fits within the printed box "
    "(small enough not to overflow). All field values are typed/keyboard-entered."
)

_FORMATTING_HANDWRITTEN = (
    "- ALL field values are HAND-WRITTEN by a real person on paper using the writing instrument "
    "described in the WRITING INSTRUMENT block below (use that ink color + tool consistently for ALL "
    "handwriting on the page).\n"
    "- THIS IS REAL ADULT HANDWRITING — NOT a 'handwriting font'. It must look authentically MESSY:\n"
    "    * Letters are clearly drawn by a HAND with a PEN — uneven, organic, NOT typographic.\n"
    "    * MIX cursive script and block printing within the same word when natural (e.g. 'Aitken' "
    "with a printed 'A' and cursive 'itken').\n"
    "    * Letter HEIGHTS vary — some 'h' or 'l' are taller, some shorter; baseline drifts up and down "
    "across a single word.\n"
    "    * SLANT varies: most adult writers consistently slant 5-15° right OR left (pick one and stick to "
    "it across the page), with random per-letter variation of ±3°.\n"
    "    * Stroke WEIGHT is uneven — the same letter is darker where pen pressed hard, lighter on the "
    "release stroke. Some down-strokes thick, up-strokes thin.\n"
    "    * Specific human idiosyncrasies: open or closed '4', barred or unbarred '7', looped or "
    "unlooped 'l', dotted or undotted 'i'. PICK a few quirks and apply them consistently.\n"
    "    * INK FLOW: occasional skip (white gap mid-stroke from a dry pen), occasional blot (small dark "
    "spot where pen stopped), occasional smudge (slight smear where hand dragged).\n"
    "    * CROSS-OUTS / OVERWRITES: 1-2 letters per page may be visibly corrected (a line struck through, "
    "or an overwriting with the corrected letter slightly darker).\n"
    "    * Letters that touch printed FORM LINES — the writing usually sits ON or slightly above the "
    "line, NOT centered perfectly; sometimes a descender (g, y, p) cuts BELOW the line.\n"
    "    * Margins between letters NOT uniform — sometimes letters touch, sometimes there are gaps.\n"
    "    * Long answers RUN OUT OF SPACE — last few words get smaller/crammed at the line end.\n"
    "- Numbers and dates are HANDWRITTEN too (not typed). Currency/SSN/phone all handwritten.\n"
    "- DO NOT produce: typed-looking text, neatly-uniform letters, perfectly-spaced characters, font-like "
    "handwriting (Comic Sans, Caveat, Bradley Hand, Marker Felt — all forbidden), or any handwriting "
    "that looks computer-generated. If a literate adult drew this, would the proportions look slightly "
    "off and unique to them?\n"
    "- Imagine the actual person writing: tired ER nurse / 70-year-old retiree with shaky hand / "
    "harried parent in waiting room / rushed lab tech / patient with carpal tunnel. Their REAL "
    "handwriting carries the marks of their physical state."
)

_FORMATTING_MIXED = (
    "- Field values use a MIX of typed text and handwriting depending on the field:\n"
    "    * Pre-populated label-like fields (printed dates, MRN stickers, account numbers) — looks typed/printed\n"
    "    * Patient-completed fields (name, address, history, dates, narrative answers) — HAND-WRITTEN in "
    "BLACK or BLUE ballpoint pen, slightly messy, normal adult handwriting (NOT a font, NOT decorative)\n"
    "    * When in doubt, prefer handwritten (most clinic intakes are filled by hand)\n"
    "- Handwriting style: mix of cursive and print, variable letter sizes/spacing, slight slant, baseline "
    "wobble, variable stroke pressure. Should look like ONE real person filled it out with a pen — "
    "not a stylized handwriting font."
)


def _formatting_block_for(analysis: dict | None) -> str:
    convention = (analysis or {}).get("handwriting_convention", "typed").lower()
    if convention == "handwritten":
        return _FORMATTING_HANDWRITTEN
    if convention == "mixed":
        return _FORMATTING_MIXED
    return _FORMATTING_TYPED


# Writing-instrument variety. Each is concrete enough that gpt-image-2 should
# produce visually distinct output. Weighted by realism in clinic/office context.
_INSTRUMENTS = [
    # (text-tool description, checkbox-mark description, weight)
    ("BIC-style medium-point blue ballpoint pen, slightly skipping ink, ~0.7mm line weight",
     "small slanted ✗ (two strokes that don't perfectly meet at the center), in blue ballpoint",
     3),
    ("BIC-style medium-point black ballpoint pen, consistent ink, ~0.7mm line weight",
     "small ✓ checkmark (short up-stroke then long down-stroke), in black ballpoint, fits inside box",
     3),
    ("cheap stick pen with dark navy ink, occasional ink blobs at letter starts, ~0.8mm",
     "filled-in box (vigorous scribble inside the square, almost solid), in navy ink",
     2),
    ("uni-ball gel pen in deep black ink, smooth flow, very dark consistent strokes, ~0.5mm",
     "neat ✓ checkmark in gel pen, dark and crisp, sitting in box",
     2),
    ("Pilot fine-tip ballpoint, blue-black ink, sharp clean lines, ~0.4mm",
     "small ✗ with crossing strokes, in blue-black, slightly slanted",
     2),
    ("fine-tip felt-tip pen, black ink, slightly fuzzy strokes (ink bleeding into paper fibers), ~0.6mm",
     "tick mark (just a long single down-stroke or a dot), in felt-tip black",
     2),
    ("Sharpie fine-point, BLACK ink, slightly bleed-through, thicker strokes ~1.0mm",
     "thick filled X in Sharpie black, covers most of the box",
     1),
    ("Sharpie ultra-fine, BLUE ink, slight bleed, ~0.6mm",
     "thick blue ✓ checkmark, leaves a slight bleed-through halo, fits in box",
     1),
    ("ballpoint pen running low on ink — letters fade in/out, occasionally needing retraced strokes",
     "barely-visible ✗ that had to be retraced (you can see the double stroke)",
     1),
    ("#2 pencil (rare — usually only for scratch fields), grey graphite, slightly smudged, ~0.5mm",
     "lightly drawn ✓ in pencil, can see paper texture through it",
     1),
    ("blue ballpoint with mild leftward slant (left-handed writer), occasionally smearing previous letters",
     "slanted ✗ that smudges slightly to the right (drag from left hand)",
     1),
]


def _pick_instrument(seed: str) -> tuple[str, str]:
    """Deterministically pick (text_tool, checkbox_mark) from _INSTRUMENTS by seed."""
    import hashlib
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    pool = []
    for desc_text, desc_box, w in _INSTRUMENTS:
        pool.extend([(desc_text, desc_box)] * w)
    return pool[h % len(pool)]


def _writing_instrument_block(seed: str, convention: str) -> str:
    """Emit a writing-instrument block when handwriting is involved."""
    if convention not in ("handwritten", "mixed"):
        return ""
    text_tool, checkbox_mark = _pick_instrument(seed)
    return (
        "===== WRITING INSTRUMENT (use this consistently for ALL handwriting on this page) =====\n"
        f"- Tool/ink: {text_tool}\n"
        f"- Checkbox marker style for THIS page: {checkbox_mark}\n"
        "- Use the SAME tool for all field values, signatures, and checkbox marks on this page. "
        "DIFFERENT pages get DIFFERENT tools (selected by the seed), so the dataset varies across forms.\n"
    )


_SIGNER_ROLE_HINTS = {
    "patient": ("PATIENT", "deliberate signature, slightly larger, first name legible, last name flowing"),
    "physician": ("PHYSICIAN/DOCTOR", "quick scrawl, smaller, mostly illegible, ends in a tail or flourish"),
    "doctor": ("PHYSICIAN/DOCTOR", "quick scrawl, smaller, mostly illegible, ends in a tail or flourish"),
    "provider": ("PROVIDER", "quick clinical signature, smaller, partly illegible"),
    "practitioner": ("PRACTITIONER", "clinical signature, moderate speed, partly legible"),
    "acupuncturist": ("ACUPUNCTURIST", "clinical signature, moderate speed"),
    "witness": ("WITNESS", "ordinary adult cursive, distinct from the patient and physician"),
    "interpreter": ("INTERPRETER", "ordinary adult cursive, distinct style"),
    "guardian": ("GUARDIAN/PARENT", "deliberate adult cursive"),
    "representative": ("REPRESENTATIVE", "deliberate adult cursive"),
    "responsible": ("RESPONSIBLE PARTY", "deliberate adult cursive"),
    "agent": ("AGENT/POA", "ordinary adult cursive"),
    "officer": ("OFFICER", "professional signature with title"),
    "agent_signature": ("AGENT", "professional signature"),
    "broker": ("BROKER", "professional signature"),
    "producer": ("PRODUCER", "professional signature"),
    "applicant": ("APPLICANT", "deliberate signature, legible first name"),
    "preparer": ("PREPARER/TAX PRO", "professional rapid signature"),
    "taxpayer": ("TAXPAYER", "deliberate signature, legible first name"),
    "spouse": ("SPOUSE", "deliberate signature, distinct from taxpayer"),
}


def _infer_signer_role(field_id: str, label: str) -> str | None:
    """Return a role keyword if this field is a signature/initial field."""
    s = (field_id + " " + label).lower()
    if not any(t in s for t in ("sign", "initial")):
        return None
    for key in _SIGNER_ROLE_HINTS.keys():
        if key in s:
            return key
    return "generic"


def _build_signers_block(page_schema: dict, persona: dict) -> str:
    """Emit per-signer style guidance for signature fields on this page."""
    signers_meta = persona.get("signers", {}) or {}
    fv = persona.get("field_values", {})
    seen = []  # (field_id, role_key, signer_name)
    for sec in page_schema.get("sections", []):
        for f in sec.get("form_fields", []):
            fid = f.get("field_id", "")
            label = f.get("label", "")
            role = _infer_signer_role(fid, label)
            if role is None:
                continue
            # Try to get signer name from explicit field_values, else from "signers" meta, else patient_name fallback
            name = (fv.get(fid) or "").strip()
            if not name and role in signers_meta:
                name = signers_meta[role].get("name", "") if isinstance(signers_meta[role], dict) else str(signers_meta[role])
            if not name:
                # Fallback: patient name from common field ids
                if role in ("patient", "applicant", "taxpayer", "responsible", "guardian", "representative", "generic"):
                    name = fv.get("patient_name") or fv.get("applicant_name") or fv.get("taxpayer_name") or ""
            seen.append((fid, role, label, name))
    if len(seen) <= 1:
        return "(only one signer on this page — no cross-signer concern)"
    lines = ["The following fields are signature/initial fields on THIS page — each is a DIFFERENT person with DIFFERENT handwriting:"]
    for fid, role, label, name in seen:
        role_label, style_hint = _SIGNER_ROLE_HINTS.get(role, ("SIGNER", "ordinary adult cursive"))
        name_part = f' (sign as: "{name}")' if name else " (use a plausible name distinct from the other signers)"
        lines.append(f"  • [{fid}] {role_label} — {label[:60]}{name_part} — style: {style_hint}")
    return "\n".join(lines)


def build_fill_prompt(page_schema: dict, persona: dict, analysis: dict | None = None) -> tuple[str, dict]:
    """Returns (prompt_text, debug_dict) for one page.

    analysis is the per-template analysis dict (template_analysis.json); only
    used to read handwriting_convention. If omitted, defaults to typed.
    """
    form_id = page_schema.get("form_id", "(unknown form)")
    scenario = persona.get("scenario_summary", "")

    # ── Field values for this page only ──
    field_lines = []
    for sec in page_schema.get("sections", []):
        sec_id = sec.get("section_id", "")
        for f in sec.get("form_fields", []):
            fid = f.get("field_id", "")
            val = persona.get("field_values", {}).get(fid, "")
            if val:
                field_lines.append(f"  [{sec_id} / {fid}] {f.get('label','')}: {val}")
    fv_block = "\n".join(field_lines) if field_lines else "  (no scalar fields to fill on this page)"

    # ── Checkbox decisions ──
    cb_lines = []
    for sec in page_schema.get("sections", []):
        for grp in sec.get("checkbox_groups", []):
            gid = grp.get("group_id", "")
            options_by_id = {o.get("option_id"): o.get("label", "") for o in grp.get("options", [])}
            decided = persona.get("checkbox_decisions", {}).get(gid, [])
            for opt_id in decided:
                if opt_id in options_by_id:
                    cb_lines.append(f"  ✓ Group [{gid}] '{grp.get('group_label','')}' → option {opt_id}: {options_by_id[opt_id]}")
    cb_block = "\n".join(cb_lines) if cb_lines else "  (no checkboxes to check on this page — LEAVE ALL UNCHECKED)"

    # ── Table rows ──
    tr_lines = []
    for sec in page_schema.get("sections", []):
        for t in sec.get("tables", []):
            tid = t.get("table_id", "")
            rows = persona.get("table_rows", {}).get(tid, [])
            if rows:
                cols = t.get("columns", [])
                tr_lines.append(f"  Table [{tid}] columns: {cols}")
                for ri, row in enumerate(rows, 1):
                    cells = " | ".join(f"{c}={row.get(c,'')}" for c in cols)
                    tr_lines.append(f"    Row {ri}: {cells}")
    tr_block = "\n".join(tr_lines) if tr_lines else "  (no tables to fill on this page)"

    signers_block = _build_signers_block(page_schema, persona)
    formatting_block = _formatting_block_for(analysis)
    # Deterministic instrument selection per (page_schema, persona)
    convention = (analysis or {}).get("handwriting_convention", "typed").lower()
    seed_str = f"{form_id}|{page_schema.get('page_label','')}|{persona.get('scenario_summary','')[:40]}"
    instrument_block = _writing_instrument_block(seed_str, convention)
    prompt = FILL_PROMPT_TEMPLATE.format(
        form_id=form_id,
        scenario_summary=scenario,
        field_values_block=fv_block,
        checkboxes_to_check_block=cb_block,
        table_rows_block=tr_block,
        signers_block=signers_block,
        formatting_block=formatting_block,
        instrument_block=instrument_block,
    )
    debug = {"field_count": len(field_lines),
             "checkbox_count": len(cb_lines),
             "table_row_count": sum(len(persona.get("table_rows", {}).get(t.get("table_id",""), []))
                                   for sec in page_schema.get("sections", [])
                                   for t in sec.get("tables", []))}
    return prompt, debug


def fill_page_via_gpt_image(pdf_path: Path, page_idx: int, prompt: str,
                            out_pdf: Path, model: str = "gpt-image-2") -> bool:
    """Render page → gpt-image-2 edit with the prompt → save as new single-page PDF."""
    with fitz.open(pdf_path) as src:
        page = src[page_idx]
        rect = page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(144/72, 144/72), alpha=False)
        png_bytes = pix.tobytes("png")
    client = get_openai()
    # Per-request timeout + bounded retries. gpt-image-2 is ~10-30s normally;
    # if it hangs past 180s the request is dead — abort and let caller retry.
    client_with_timeout = client.with_options(timeout=180.0, max_retries=2)
    try:
        resp = client_with_timeout.images.edit(
            model=model,
            image=("page.png", io.BytesIO(png_bytes), "image/png"),
            prompt=prompt,
            size="1024x1536",
            n=1,
        )
        filled_png = base64.b64decode(resp.data[0].b64_json)
    except Exception as e:
        print(f"  ! gpt-image-2 error: {type(e).__name__}: {str(e)[:200]}")
        return False
    # Save as single-page PDF preserving the original page dimensions
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    new_doc = fitz.open()
    new_page = new_doc.new_page(width=rect.width, height=rect.height)
    new_page.insert_image(rect, stream=filled_png, keep_proportion=False)
    new_doc.save(out_pdf, garbage=4, deflate=True)
    new_doc.close()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_LABELS = {
    "v1_acord_125_126": "ACORD 125/126 Commercial Insurance Application",
    "v1_acord_4":       "ACORD Homeowner App / Property Loss Notice",
    "v1_acord_25":      "ACORD 25 Certificate of Liability Insurance",
    "v1_fec_form3":     "FEC Form 3 / 3X campaign finance report",
    "v1_form_5500":     "DOL Form 5500 — ERISA pension plan annual return",
    "v1_irs_specialty": "IRS specialty form (see filename)",
    "v1_intl_tax_ca":   "Canada CRA tax form (see filename)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to blank template PDF")
    ap.add_argument("--type", required=True, help="v1_<type> folder name (e.g. v1_irs_specialty)")
    ap.add_argument("--persona-cache", default=None,
                    help="Reuse an existing persona JSON (skip steps 1+2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Produce schema + persona + prompts but skip gpt-image-2 calls")
    ap.add_argument("--anthropic-model", default="claude-sonnet-4-5")
    ap.add_argument("--image-model", default="gpt-image-2")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel page fills (default 4 — gpt-image-2 calls are ~130s each)")
    args = ap.parse_args()

    src_pdf = Path(args.pdf).resolve()
    if not src_pdf.exists():
        print(f"!! {src_pdf} not found"); return 1
    with fitz.open(src_pdf) as d:
        n_pages = d.page_count
    print(f"\n=== Template: {src_pdf.name}  ({n_pages} pages, type={args.type}) ===\n")

    # work_dir is needed for schema cache lookup
    work_dir = OUT_ROOT / args.type / src_pdf.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    schemas_cache = work_dir / "schemas.json"

    # ── STEP 1: per-page schema extraction (reuse cache if available + persona is cached too)
    if args.persona_cache and schemas_cache.exists():
        print(f"STEP 1 — reusing cached schemas from {schemas_cache.relative_to(REPO)}")
        schemas = json.loads(schemas_cache.read_text())
        if len(schemas) != n_pages:
            print(f"  ! cached schemas don't match page count ({len(schemas)} vs {n_pages}) — re-extracting")
            schemas = None
        else:
            for i, schema in enumerate(schemas):
                n_fields = sum(len(s.get("form_fields", [])) for s in schema.get("sections", []))
                n_cb_groups = sum(len(s.get("checkbox_groups", [])) for s in schema.get("sections", []))
                n_cb_options = sum(len(g.get("options", []))
                                    for s in schema.get("sections", [])
                                    for g in s.get("checkbox_groups", []))
                n_tables = sum(len(s.get("tables", [])) for s in schema.get("sections", []))
                print(f"  page {i+1}: {n_fields} fields, {n_cb_groups} checkbox groups "
                      f"({n_cb_options} options total), {n_tables} tables  (cached)")
    else:
        schemas = None

    if schemas is None:
        print(f"STEP 1 — vision schema extraction for {n_pages} pages…")
        schemas = []
        for i in range(n_pages):
            t0 = time.time()
            schema = extract_page_schema(src_pdf, i, args.anthropic_model)
            schemas.append(schema)
            n_fields = sum(len(s.get("form_fields", [])) for s in schema.get("sections", []))
            n_cb_groups = sum(len(s.get("checkbox_groups", [])) for s in schema.get("sections", []))
            n_cb_options = sum(len(g.get("options", []))
                                for s in schema.get("sections", [])
                                for g in s.get("checkbox_groups", []))
            n_tables = sum(len(s.get("tables", [])) for s in schema.get("sections", []))
            print(f"  page {i+1}: {n_fields} fields, {n_cb_groups} checkbox groups "
                  f"({n_cb_options} options total), {n_tables} tables  ({time.time()-t0:.1f}s)")
        schemas_cache.write_text(json.dumps(schemas, indent=2))
        print(f"  → schemas saved to {schemas_cache.relative_to(REPO)}")

    # ── STEP 2: coherent persona
    if args.persona_cache and Path(args.persona_cache).exists():
        persona = json.loads(Path(args.persona_cache).read_text())
        print(f"\nSTEP 2 — using cached persona from {args.persona_cache}")
    else:
        print(f"\nSTEP 2 — building coherent persona…")
        t0 = time.time()
        label = TEMPLATE_LABELS.get(args.type, args.type)
        persona = build_persona(label, schemas, args.anthropic_model)
        (work_dir / "persona.json").write_text(json.dumps(persona, indent=2))
        print(f"  scenario: {persona.get('scenario_summary', '')}")
        print(f"  field values: {len(persona.get('field_values', {}))}")
        cb = persona.get("checkbox_decisions", {})
        print(f"  checkbox groups with decisions: {len(cb)}")
        for gid, opts in list(cb.items())[:6]:
            print(f"    [{gid}] check options: {opts}")
        print(f"  → persona saved to {(work_dir / 'persona.json').relative_to(REPO)}  ({time.time()-t0:.1f}s)")

    # ── STEP 3: per-page fill (parallel via ThreadPoolExecutor)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"\nSTEP 3 — render fill prompts + gpt-image-2 ({args.workers} workers parallel)…")
    # Build prompts up-front (cheap, no network)
    prompts: dict[int, tuple[str, dict, Path]] = {}
    for i in range(n_pages):
        prompt, debug = build_fill_prompt(schemas[i], persona)
        (work_dir / f"prompt_p{i+1:02d}.txt").write_text(prompt)
        out_pdf = work_dir / f"{src_pdf.stem}__p{i+1:02d}_filled.pdf"
        prompts[i] = (prompt, debug, out_pdf)
        print(f"  page {i+1}: {debug['field_count']} fields, "
              f"{debug['checkbox_count']} checkboxes, "
              f"{debug['table_row_count']} table rows in fill plan")

    if args.dry_run:
        print(f"\n(dry-run — skipping gpt-image-2 calls)")
        print(f"All artifacts in: {work_dir.relative_to(REPO)}/")
        return 0

    t_fill_start = time.time()
    def _fill_job(i):
        prompt, debug, out_pdf = prompts[i]
        t0 = time.time()
        ok = fill_page_via_gpt_image(src_pdf, i, prompt, out_pdf, args.image_model)
        return (i, ok, out_pdf, time.time() - t0)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_fill_job, i) for i in range(n_pages)]
        for fut in as_completed(futures):
            i, ok, out_pdf, dt = fut.result()
            if ok:
                print(f"  ✓ page {i+1} filled → {out_pdf.relative_to(REPO)}  ({dt:.0f}s)", flush=True)
            else:
                print(f"  ✗ page {i+1} fill failed", flush=True)
    wall = time.time() - t_fill_start
    print(f"\n  fill wall time: {wall:.0f}s (was ~{n_pages * 130}s sequential)")

    print(f"\nAll artifacts in: {work_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
