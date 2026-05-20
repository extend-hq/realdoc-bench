#!/usr/bin/env python3
"""Generate per-type JSON Schemas for v1.1 doc types.

Strategy (per type):

1. **Reuse v0 schemas** when a clear mapping exists (v0 hand-crafted schemas
   are tailored to specific filings but capture the right field taxonomy).
2. **Generate from PDF page images + parsed markdown** otherwise.

   Vision matters: parsed markdown rarely surfaces checkboxes faithfully
   (a checked box and an unchecked box both serialize as similar text), so
   relying solely on the markdown would overfit our benchmark schemas to
   Extend's strengths. We send Claude:
     - 1-2 source-PDF page images (canonical truth — discovers checkboxes,
       visual layout, table cells)
     - the parsed markdown (text content, labels)
     - optionally a v0 schema for dialect inspiration

   Emphasis:
     - checkboxes  (`x-capabilities: ["form_region","checkbox"]`)
     - tables      (`x-capabilities: ["table_structured"]`)
     - key-values  (`x-capabilities: ["form_region"]`)

Output:
  datasets/extraction/v1.1/_schemas/<type_name>.json   — one per type
  datasets/extraction/v1.1/_schemas/_MANIFEST.md       — provenance table

Idempotent: skips types whose schema already exists. Use --force to regen.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
V1_1 = REPO / "datasets" / "extraction" / "v1.1"
SCHEMAS_OUT = V1_1 / "_schemas"
V0_SCHEMAS = REPO / "datasets" / "extraction" / "v0" / "schemas"
V0_TAX_TEMPLATE = REPO / "datasets" / "extraction" / "v0_tax" / "_schema_template.json"


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


# ─────────────────────────────────────────────────────────────────────────────
# Reuse map: v1.1 type → existing v0 schema (verbatim copy)
# These v0 schemas are bespoke-per-doc but capture the right field taxonomy
# for the type; copying gives v1.1 a head start.
# ─────────────────────────────────────────────────────────────────────────────
V0_REUSE_MAP: dict[str, Path] = {
    # v0 single-doc folders → their own schema
    "v0_fin_10k_001": V0_SCHEMAS / "fin_10k_001.json",
    "v0_fin_audit_001": V0_SCHEMAS / "fin_audit_001.json",
    "v0_fin_audit_002": V0_SCHEMAS / "fin_audit_002.json",
    "v0_med_autopsy_001": V0_SCHEMAS / "med_autopsy_001.json",
    "v0_med_protocol_001": V0_SCHEMAS / "med_protocol_001.json",
    "v0_tech_datasheet_001": V0_SCHEMAS / "tech_datasheet_001.json",
    # v0_tax has a type-level template (vs per-doc)
    "v0_tax": V0_TAX_TEMPLATE,
    # v1.1 types whose v0 sibling schema is reusable as a starting point
    "v1_sec_10k": V0_SCHEMAS / "fin_10k_001.json",
    "v1_audit_report": V0_SCHEMAS / "fin_audit_001.json",
    "v1_med_autopsy": V0_SCHEMAS / "med_autopsy_001.json",
    "v1_med_protocol": V0_SCHEMAS / "med_protocol_001.json",
    "v1_tech_datasheet": V0_SCHEMAS / "tech_datasheet_001.json",
}

# Human labels for prompt
TYPE_LABELS: dict[str, str] = {
    "v1_acord_125_126": "ACORD 125/126 Commercial Insurance Application",
    "v1_acord_25": "ACORD 25 Certificate of Liability Insurance",
    "v1_acord_4": "ACORD Homeowner Application / Property Loss Notice",
    "v1_air_waybill": "IATA Air Waybill (AWB) — air cargo shipment document",
    "v1_bankruptcy_schedule": "US Bankruptcy Court Schedule (Chapter 11/7) — creditor list / petition",
    "v1_bol_inland": "Inland Bill of Lading — shipper / consignee / carrier freight document",
    "v1_budget_990": "IRS Form 990 — nonprofit annual return",
    "v1_budget_amendment": "Municipal / agency budget amendment / transfer form",
    "v1_budget_cafr": "Comprehensive Annual Financial Report (CAFR / ACFR)",
    "v1_budget_sf424a": "SF-424A federal grant budget information form",
    "v1_cd": "TILA-RESPA Closing Disclosure — final mortgage loan terms",
    "v1_clinical_consent": "Clinical trial informed consent form (IRB)",
    "v1_clinical_crf": "Clinical case report form / public-health surveillance form",
    "v1_clinical_facesheet": "Hospital / clinic patient face sheet / demographic intake",
    "v1_cmr": "CMR international road transport consignment note",
    "v1_cms_1500": "CMS-1500 Health Insurance Claim Form (medical billing)",
    "v1_commercial_invoice": "Commercial Invoice (customs / CBP) for international shipment",
    "v1_esg": "ESG / sustainability / CDP corporate disclosure report",
    "v1_eu_sad": "EU Single Administrative Document (customs declaration)",
    "v1_fec_form3": "FEC Form 3 / 3X / 3P — campaign finance report (US Federal Elections)",
    "v1_fnma_710": "Fannie Mae / Freddie Mac Form 710 Mortgage Assistance Application",
    "v1_form_5500": "DOL / IRS Form 5500 — ERISA annual return for employee benefit plan",
    "v1_hud1": "HUD-1 Settlement Statement — real estate closing",
    "v1_lloyds_mrc": "Lloyd's Market Reform Contract (MRC) — London insurance market",
    "v1_loc": "Letter of Credit (SWIFT MT700) / documentary credit",
    "v1_packing_list": "Export / shipping packing list",
    "v1_phyto_cert": "Phytosanitary certificate (agricultural export)",
    "v1_pod": "Proof of delivery / delivery receipt",
    "v1_real_estate_contract": "Residential real estate purchase contract",
    "v1_sec_10q": "SEC Form 10-Q — quarterly filing",
    "v1_ub04": "UB-04 / CMS-1450 Institutional Hospital Claim Form",
    "v1_urar": "Uniform Residential Appraisal Report (Fannie Mae Form 1004)",
    "v1_urla_1003": "Uniform Residential Loan Application (URLA Form 1003)",
    "v1_vaers": "VAERS-1 Vaccine Adverse Event Reporting System form",
    "v1_w2_1099": "IRS Form W-2 / 1099 — wage and tax statement",
}

DOMAIN_TO_X_DOMAIN: dict[str, str] = {
    "Medical": "healthcare",
    "Financial_(incl._shipping-cargo)": "financial_services",
    "Mortgage_and_Legal": "mortgage_legal",
    "Tax": "tax",
    "Other_(technical)": "tech",
    "Budget_and_nonprofit_financials": "budget_nonprofit",
}


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic client
# ─────────────────────────────────────────────────────────────────────────────
_anthropic = None
def get_anthropic():
    global _anthropic
    if _anthropic is None:
        from anthropic import Anthropic
        _anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic


SCHEMA_PROMPT = """You are designing a JSON Schema (draft-07, Extend dialect) for extracting structured data from a "{doc_label}".

You will be given:
  - 1-2 page IMAGES of an example document of this type (TRUST THESE FOR VISUAL ELEMENTS — checkboxes, table cell structure, signature blocks)
  - The same document(s) as parsed markdown (USE FOR TEXT CONTENT — field labels, values, headings)

IMPORTANT: The parsed markdown often loses checkbox state and visual layout. ALWAYS prefer what you see in the IMAGE over what the markdown shows when they disagree about visual elements. Discover checkboxes by examining the image directly — every visible ☐ or ☑ on the form is a candidate boolean field, regardless of whether the markdown surfaces it.

Your job: produce a JSON Schema that captures ALL extractable fields visible in the example, ESPECIALLY emphasizing these three categories:

1. **CHECKBOXES** — boolean fields, or mutually-exclusive radio groups (use `enum` + nullable). Every checkbox field MUST include `x-capabilities: ["form_region", "checkbox"]`. **Discover these from the IMAGES, not the markdown.** Examples: yes/no questions, insurance-type selectors, sex/relationship/status flags, condition codes, "check all that apply" lists.
2. **TABLES** — recurring rows of structured data. Use `type: "array", items: {{"type": "object", "properties": {{...}}}}` with `x-capabilities: ["table_structured"]`. Include all columns visible in the image (count column headers from the IMAGE — markdown often loses column count). Examples: service-line tables (CMS-1500/UB-04), schedule line items (Form 5500), diagnosis-code grids, expense breakdowns, payment schedules.
3. **KEY-VALUE PAIRS** — scalar fields printed in fillable boxes (`x-capabilities: ["form_region"]`) or labelled in prose (`x-capabilities: ["key_value"]`). Examples: names, addresses, dates, ID numbers, dollar amounts, plan numbers.

DIALECT RULES (Extend extensions on top of draft-07):
- Every field MUST have a 1-line `description` saying what to extract and where.
- Every field MUST have `x-difficulty`: "easy" | "intermediate" | "hard"
- Every field MUST have `x-capabilities` (array). Standard tags:
    `form_region`, `checkbox`, `table_structured`, `cross_reference`,
    `signature`, `key_value`, `enum`, `currency`, `date_extraction`, `null_discipline`
- Nullable fields: `"type": ["string", "null"]` (NOT `nullable: true`)
- Dates: `"type": "string"` with `"format": "date"`
- Currency / dollar amounts: `"type": "number"` (do not use `"currency"`)
- Mutually-exclusive options: use `enum` + nullable

TOP-LEVEL STRUCTURE (REQUIRED):
{{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "<human-readable type label>",
  "description": "<1-2 sentence doc type description>",
  "x-doc-type": "{x_doc_type}",
  "x-domain": "{x_domain}",
  "type": "object",
  "properties": {{ <field definitions> }},
  "required": [<fields that are ALWAYS present, conservatively>]
}}

GUIDELINES:
- Aim for 25-80 fields total. Be exhaustive for SHORT forms (CMS-1500, ACORD 25) and selective for LONG documents (10-K, 990 — focus on cover-page / summary fields plus 2-3 representative tables).
- For documents with MANY discrete checkboxes (e.g. medical claim form Box 1 insurance type), define ONE enum field with all options rather than 7 separate booleans.
- For nested tables-within-tables, flatten to top-level array fields with descriptive names.
- DO NOT invent fields not visible in the example documents. Ground every field in the markdown.

{v0_inspiration_block}

=== EXAMPLE PARSED MARKDOWN(S) (text content reference — but TRUST IMAGES for visual elements) ===
{example_block}

Return ONLY the JSON schema, no preamble, no fences."""


def truncate_md(md: str, limit: int = 6000) -> str:
    if len(md) <= limit:
        return md
    half = limit // 2
    return md[:half] + "\n\n…[truncated]…\n\n" + md[-half:]


def render_page_b64(pdf: Path, page_idx: int, dpi: int = 120) -> str | None:
    """Render a single PDF page as base64 PNG. Returns None on failure."""
    try:
        with fitz.open(pdf) as doc:
            if page_idx >= doc.page_count:
                return None
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return base64.b64encode(pix.tobytes("png")).decode()
    except Exception:
        return None


def pick_pages_to_send(pdf: Path) -> list[int]:
    """Pick 1-2 page indices to send as images. Page 0 always; if doc is
    > 4 pages, also send a mid page (often where tables live)."""
    try:
        with fitz.open(pdf) as doc:
            pc = doc.page_count
    except Exception:
        return [0]
    if pc <= 1:
        return [0]
    if pc <= 4:
        return [0]
    # Multi-page: send page 0 (cover/header) + ~25% in (likely body / tables)
    mid = min(pc - 1, max(1, pc // 4))
    return [0, mid]


def generate_schema(type_name: str, x_domain: str, parsed_mds: list[Path],
                    source_pdfs: list[Path], v0_inspiration: Path | None,
                    model: str) -> dict:
    label = TYPE_LABELS.get(type_name, type_name.replace("v1_", "").replace("_", " "))
    x_doc_type = type_name.replace("v1_", "").replace("v0_", "")

    # Markdown excerpt (for text content reference)
    examples = []
    for i, md_path in enumerate(parsed_mds[:1], 1):  # 1 md is plenty when we also have images
        try:
            md = md_path.read_text()
        except Exception:
            continue
        examples.append(f"--- Markdown of {md_path.stem} ---\n{truncate_md(md)}")
    example_block = "\n\n".join(examples) if examples else "(no parsed markdown available)"

    # Page images (for visual elements — checkboxes, table cells, layout)
    image_blocks: list[dict] = []
    image_notes: list[str] = []
    if source_pdfs:
        # Take 1-2 pages from the FIRST source PDF (matches the markdown above)
        pdf = source_pdfs[0]
        for page_idx in pick_pages_to_send(pdf):
            b64 = render_page_b64(pdf, page_idx)
            if b64:
                image_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                })
                image_notes.append(f"page {page_idx + 1} of {pdf.name}")
    image_intro = (f"\n(Images attached: {', '.join(image_notes)})" if image_notes else
                   "\n(No source images available — inferring from markdown only)")

    v0_block = ""
    if v0_inspiration and v0_inspiration.exists():
        v0_schema_text = v0_inspiration.read_text()
        if len(v0_schema_text) > 8000:
            v0_schema_text = v0_schema_text[:8000] + "\n... (truncated)"
        v0_block = (f"\n=== EXISTING SCHEMA FOR A RELATED DOC TYPE (use as inspiration for "
                    f"dialect, capability tagging, and difficulty calibration; "
                    f"adapt fields to this type) ===\n{v0_schema_text}\n")

    prompt = SCHEMA_PROMPT.format(
        doc_label=label,
        x_doc_type=x_doc_type,
        x_domain=x_domain,
        v0_inspiration_block=v0_block,
        example_block=example_block,
    ) + image_intro

    # Build multimodal content: images first (so model anchors on visuals), then text
    content: list[dict] = list(image_blocks)
    content.append({"type": "text", "text": prompt})

    client = get_anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def discover_types(v1_1: Path) -> list[tuple[str, str, Path]]:
    """Yield (type_name, domain_dir, type_dir) for every v1.1 type that has sources."""
    out = []
    for domain_dir in sorted(d for d in v1_1.iterdir() if d.is_dir() and not d.name.startswith("_")):
        for type_dir in sorted(t for t in domain_dir.iterdir() if t.is_dir()):
            sources = type_dir / "sources"
            if sources.exists() and any(sources.glob("*.pdf")):
                out.append((type_dir.name, domain_dir.name, type_dir))
    return out


def find_parsed_mds(type_name: str, domain: str, parse_run: Path) -> list[Path]:
    """Find parsed markdowns for this type in the latest parse run."""
    md_dir = parse_run / domain / type_name
    if not md_dir.exists():
        return []
    return sorted(md_dir.glob("*.md"))


def find_source_pdfs(type_dir: Path) -> list[Path]:
    return sorted((type_dir / "sources").glob("*.pdf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-run", default=None,
                    help="runs/extend_v2_parse/<ts>/ to source markdowns from "
                         "(default: latest)")
    ap.add_argument("--type", action="append",
                    help="Restrict to one or more types (repeatable)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="Regenerate schemas even if they already exist")
    ap.add_argument("--model", default="claude-sonnet-4-5")
    args = ap.parse_args()

    SCHEMAS_OUT.mkdir(parents=True, exist_ok=True)

    parse_runs_root = REPO / "runs" / "extend_v2_parse"
    if args.parse_run:
        parse_run = Path(args.parse_run).resolve()
    else:
        runs = sorted(parse_runs_root.glob("*"))
        if not runs:
            print("!! no parse runs found")
            return 1
        parse_run = runs[-1]
    print(f"Using parse run: {parse_run.relative_to(REPO)}")

    types = discover_types(V1_1)
    if args.type:
        types = [t for t in types if t[0] in args.type]
    print(f"Found {len(types)} v1.1 types")

    # Plan
    plan_reuse: list[tuple[str, str, Path]] = []   # (type_name, domain, v0_schema_path)
    plan_gen: list[tuple[str, str, list[Path], list[Path], Path | None]] = []  # type, domain, mds, pdfs, v0_insp
    plan_skip: list[str] = []

    for type_name, domain, type_dir in types:
        out_path = SCHEMAS_OUT / f"{type_name}.json"
        if out_path.exists() and not args.force:
            plan_skip.append(type_name)
            continue
        v0_path = V0_REUSE_MAP.get(type_name)
        # For v0/* and v0_tax types: REUSE the v0 schema directly (no LLM call)
        if v0_path and v0_path.exists() and type_name.startswith("v0_"):
            plan_reuse.append((type_name, domain, v0_path))
            continue
        # For v1_* types with a v0 sibling: GENERATE with v0 as inspiration
        mds = find_parsed_mds(type_name, domain, parse_run)
        pdfs = find_source_pdfs(type_dir)
        plan_gen.append((type_name, domain, mds, pdfs, v0_path))

    print(f"  reuse v0 verbatim: {len(plan_reuse)}")
    print(f"  generate via Claude: {len(plan_gen)}")
    print(f"  skip (already exist): {len(plan_skip)}")

    # Execute reuse copies (instant)
    reuse_results: list[tuple[str, Path]] = []
    for type_name, domain, v0_path in plan_reuse:
        out = SCHEMAS_OUT / f"{type_name}.json"
        shutil.copy2(v0_path, out)
        reuse_results.append((type_name, v0_path))
        print(f"  ✓ reuse  {type_name}  ← {v0_path.relative_to(REPO)}")

    # Execute LLM-generated schemas in parallel
    t0 = time.time()
    gen_results: list[tuple[str, Path | None, str]] = []  # type, out_path, status_msg

    def _job(type_name: str, domain: str, mds: list[Path], pdfs: list[Path], v0_insp: Path | None):
        x_dom = DOMAIN_TO_X_DOMAIN.get(domain, domain.lower())
        try:
            schema = generate_schema(type_name, x_dom, mds, pdfs, v0_insp, args.model)
            out = SCHEMAS_OUT / f"{type_name}.json"
            out.write_text(json.dumps(schema, indent=2))
            n_fields = len(schema.get("properties", {}))
            n_checkboxes = sum(
                1 for v in schema.get("properties", {}).values()
                if isinstance(v, dict) and "checkbox" in (v.get("x-capabilities") or [])
            )
            n_tables = sum(
                1 for v in schema.get("properties", {}).values()
                if isinstance(v, dict) and "table_structured" in (v.get("x-capabilities") or [])
            )
            return (type_name, out, f"ok ({n_fields}f, {n_checkboxes}cb, {n_tables}tb, "
                    f"{len(mds)}md+{len(pdfs[:1])}img{', +v0' if v0_insp else ''})")
        except Exception as e:
            return (type_name, None, f"ERR {type(e).__name__}: {str(e)[:200]}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, *args_): args_[0] for args_ in plan_gen}
        for i, fut in enumerate(as_completed(futs), 1):
            type_name, out, msg = fut.result()
            gen_results.append((type_name, out, msg))
            print(f"  [{i}/{len(plan_gen)}] {type_name}: {msg}", flush=True)

    # Manifest
    manifest = SCHEMAS_OUT / "_MANIFEST.md"
    with manifest.open("w") as f:
        f.write(f"# v1.1 schemas — provenance\n\n")
        f.write(f"Generated by `scripts/generate_v1_1_schemas.py` from parse run "
                f"`{parse_run.name}`.\n\n")
        f.write("## Reused verbatim from v0\n\n")
        f.write("| Type | Source schema |\n|---|---|\n")
        for type_name, v0_path in sorted(reuse_results):
            f.write(f"| `{type_name}` | `{v0_path.relative_to(REPO)}` |\n")
        if not reuse_results:
            f.write("| _(none)_ | |\n")
        f.write("\n## Generated via Claude from parsed markdown\n\n")
        f.write("| Type | Status |\n|---|---|\n")
        for type_name, out, msg in sorted(gen_results):
            f.write(f"| `{type_name}` | {msg} |\n")
        f.write(f"\n## Skipped (already existed; use `--force` to regenerate)\n\n")
        for t in plan_skip:
            f.write(f"- `{t}`\n")

    wall = time.time() - t0
    n_ok = sum(1 for _, out, _ in gen_results if out is not None)
    n_fail = len(gen_results) - n_ok
    print(f"\nDone — reused {len(reuse_results)}, generated {n_ok}, failed {n_fail} "
          f"({wall:.0f}s)")
    print(f"  schemas: {SCHEMAS_OUT.relative_to(REPO)}/")
    print(f"  manifest: {manifest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
